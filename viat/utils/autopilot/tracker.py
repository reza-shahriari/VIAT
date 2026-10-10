"""Life-cycle multi-object tracker for the Autopilot pipeline.

Every object goes tentative -> confirmed -> lost -> retired, which is what replaces
"watching for objects that enter and leave". Association follows ByteTrack
(confident detections first, then low-score ones for already-tracked objects) with
BoT-SORT camera-motion compensation, plus a size-normalised centre-distance stage
for tiny objects whose boxes barely overlap between frames. Births follow SAM 3's
gate: a confident detection that no existing track explains.
"""
import numpy as np

from .geometry import (iou_matrix, ioa_matrix, centers, linear_assignment,
                       xyxy_to_xyah, xyah_to_xyxy)


class KalmanXYAH:
    """Constant-velocity Kalman filter on (cx, cy, aspect, height), as in ByteTrack."""
    std_pos = 1.0 / 20
    std_vel = 1.0 / 160

    def __init__(self):
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, 4 + i] = 1.0
        self.H = np.eye(4, 8)

    def initiate(self, meas):
        h = meas[3]
        mean = np.r_[meas, np.zeros(4)]
        std = [2 * self.std_pos * h, 2 * self.std_pos * h, 1e-2, 2 * self.std_pos * h,
               10 * self.std_vel * h, 10 * self.std_vel * h, 1e-5, 10 * self.std_vel * h]
        return mean, np.diag(np.square(std))

    def predict(self, mean, cov):
        h = mean[3]
        std = [self.std_pos * h, self.std_pos * h, 1e-2, self.std_pos * h,
               self.std_vel * h, self.std_vel * h, 1e-5, self.std_vel * h]
        return self.F @ mean, self.F @ cov @ self.F.T + np.diag(np.square(std))

    def update(self, mean, cov, meas):
        h = mean[3]
        r = np.diag(np.square([self.std_pos * h, self.std_pos * h, 1e-1, self.std_pos * h]))
        s = self.H @ cov @ self.H.T + r
        k = cov @ self.H.T @ np.linalg.inv(s)
        mean = mean + k @ (meas - self.H @ mean)
        cov = cov - k @ s @ k.T
        return mean, cov

    @staticmethod
    def warp(mean, cov, affine):
        """Apply the camera motion (2x3 affine, prev -> cur) to a predicted state."""
        rot = affine[:, :2]
        scale = float(np.sqrt(abs(np.linalg.det(rot)))) or 1.0
        mean = mean.copy()
        mean[:2] = rot @ mean[:2] + affine[:, 2]
        mean[4:6] = rot @ mean[4:6]
        mean[3] *= scale
        mean[7] *= scale
        t = np.eye(8)
        t[:2, :2] = rot
        t[4:6, 4:6] = rot
        return mean, t @ cov @ t.T


class Track:
    def __init__(self, track_id, frame, box, score, det_idx, cls_vec, kf):
        self.id = track_id
        self.mean, self.cov = kf.initiate(xyxy_to_xyah(box))
        self.state = 'tentative'
        self.hits = 1
        self.misses = 0  # consecutive detection frames without a match
        self.time_since_update = 0
        self.obs = {frame: (np.asarray(box, np.float32), float(score), int(det_idx), 'd')}
        self.cls_sum = np.asarray(cls_vec, np.float32) * 1.0

    @property
    def box(self):
        return xyah_to_xyxy(self.mean)

    def top_class(self):
        return int(np.argmax(self.cls_sum)) if self.cls_sum.size else 0


class LifecycleTracker:
    def __init__(self, cfg, fps, id_start=1):
        self.cfg = cfg
        self.fps = fps
        self.kf = KalmanXYAH()
        self.next_id = id_start
        self.active = []
        self.finished = []
        self.rejected = []
        self.max_lost = max(1, int(round(cfg.max_lost_sec * fps)))

    def flush(self):
        """End of a shot: confirmed tracks finish, unconfirmed ones are rejected."""
        for t in self.active:
            (self.finished if t.state != 'tentative' else self.rejected).append(t)
        self.active = []

    def step(self, frame, rec, affine):
        cfg = self.cfg
        for t in self.active:
            t.mean, t.cov = self.kf.predict(t.mean, t.cov)
            if affine is not None:
                t.mean, t.cov = self.kf.warp(t.mean, t.cov, affine)
            t.time_since_update += 1
            if t.time_since_update > 10:
                t.mean[4:] *= 0.5  # a long-lost object should not fly away on old velocity

        if rec is None:  # frame skipped by the detection stride: predict only
            self._retire()
            return

        boxes, scores, cls = rec['boxes'], rec['scores'], rec['cls'].astype(np.float32)
        det_cls = cls.argmax(1) if len(cls) else np.zeros(0, int)
        high = np.where(scores >= cfg.high_score)[0]
        low = np.where((scores >= cfg.low_score) & (scores < cfg.high_score))[0]

        tracks = list(self.active)
        unmatched_t = list(range(len(tracks)))
        matched = {}

        def match(t_idx, d_idx, min_iou):
            if not t_idx or not len(d_idx):
                return t_idx, list(d_idx)
            tb = np.stack([tracks[i].box for i in t_idx])
            iou = iou_matrix(tb, boxes[d_idx])
            same = np.array([[tracks[i].top_class() == det_cls[d] for d in d_idx] for i in t_idx])
            sim = iou * np.where(same, 1.0, cfg.cross_class_factor)
            pairs, ut, ud = linear_assignment(1.0 - sim, 1.0 - min_iou)
            for r, c in pairs:
                matched[t_idx[r]] = d_idx[c]
            return [t_idx[r] for r in ut], [d_idx[c] for c in ud]

        # 1) confident detections against every live track
        unmatched_t, rem_high = match(unmatched_t, list(high), cfg.match_iou)
        # 2) low-score detections only rescue tracks that were seen last frame
        recent = [i for i in unmatched_t if tracks[i].time_since_update <= 1]
        older = [i for i in unmatched_t if tracks[i].time_since_update > 1]
        recent, _ = match(recent, list(low), cfg.low_match_iou)
        unmatched_t = recent + older
        # 3) tiny / fast objects: size-normalised centre distance
        if unmatched_t and rem_high:
            tb = np.stack([tracks[i].box for i in unmatched_t])
            db = boxes[rem_high]
            tc, dc = centers(tb), centers(db)
            t_size = np.sqrt(np.clip((tb[:, 2] - tb[:, 0]) * (tb[:, 3] - tb[:, 1]), 1, None))
            d_size = np.sqrt(np.clip((db[:, 2] - db[:, 0]) * (db[:, 3] - db[:, 1]), 1, None))
            dist = np.linalg.norm(tc[:, None] - dc[None], axis=2) / t_size[:, None]
            ratio = d_size[None] / t_size[:, None]
            dist[(ratio < 0.5) | (ratio > 2.0)] = 1e6
            pairs, ut, ud = linear_assignment(dist, cfg.dist_match)
            for r, c in pairs:
                matched[unmatched_t[r]] = rem_high[c]
            unmatched_t = [unmatched_t[r] for r in ut]
            rem_high = [rem_high[c] for c in ud]

        for t_i, d_i in matched.items():
            t = tracks[t_i]
            t.mean, t.cov = self.kf.update(t.mean, t.cov, xyxy_to_xyah(boxes[d_i]))
            t.obs[frame] = (boxes[d_i].copy(), float(scores[d_i]), int(d_i), 'd')
            t.cls_sum += cls[d_i]
            t.hits += 1
            t.misses = 0
            t.time_since_update = 0
            if t.state == 'tentative' and t.hits >= cfg.min_hits:
                t.state = 'confirmed'
            elif t.state == 'lost':
                t.state = 'confirmed'

        for t_i in unmatched_t:
            t = tracks[t_i]
            t.misses += 1
            if t.state == 'confirmed':
                t.state = 'lost'

        # births: confident detections that no live track explains. With boxes (unlike
        # SAM 3's masks) a person on a vehicle lies inside the vehicle's box, so only a
        # same-class track, or a near-identical box of another class, explains a detection.
        live = [(t.box, t.top_class()) for t in self.active]
        for d_i in sorted(rem_high, key=lambda d: -scores[d]):
            if scores[d_i] < cfg.birth_score:
                continue
            if live:
                lb = np.stack([b for b, _ in live])
                same = np.array([c == det_cls[d_i] for _, c in live])
                iou = iou_matrix(boxes[d_i:d_i + 1], lb)[0]
                ioa = ioa_matrix(boxes[d_i:d_i + 1], lb)[0]
                if (((iou >= cfg.birth_explained_iou) & same).any() or (iou >= 0.6).any() or
                        ((ioa >= 0.7) & same).any()):
                    continue
            t = Track(self.next_id, frame, boxes[d_i], scores[d_i], d_i, cls[d_i], self.kf)
            self.next_id += 1
            self.active.append(t)
            live.append((t.box, t.top_class()))

        self._retire()

    def _retire(self):
        keep = []
        for t in self.active:
            if t.state == 'tentative' and t.misses >= self.cfg.tentative_max_miss:
                self.rejected.append(t)
            elif t.state != 'tentative' and t.time_since_update > self.max_lost:
                self.finished.append(t)
            else:
                keep.append(t)
        self.active = keep
