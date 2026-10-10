"""Offline (whole-video) track repair for the Autopilot pipeline.

Because labeling is offline, every track can be fixed with hindsight:
  1. extend each track backward/forward along weak detections (objects are usually
     detected late when they enter and early-lost when they leave),
  2. link fragments of the same object across short gaps (StrongSORT AFLink idea),
  3. suppress duplicate tracks of one object (SAM2BMP tracklet NMS),
  4. fill short gaps by camera-compensated interpolation,
  5. vote the class over the whole track.
"""
import numpy as np

from .geometry import iou_matrix, centers, box_area


class TrackRec:
    """A finished track: ``boxes[frame] = (box, score, det_idx, kind)``.

    kind: 'd' detection, 'e' extension along a weak detection, 'i' interpolation.
    """

    def __init__(self, track_id, shot, boxes, cls_sum):
        self.id = track_id
        self.shot = shot
        self.boxes = dict(boxes)
        self.cls_sum = np.asarray(cls_sum, np.float32).copy()
        self.links = 0
        self.ext = {'back': 0, 'fwd': 0}
        self.overlaps = {}  # other track id -> mean IoU over shared frames
        self.route = 'accept'
        self.risk = 0.0
        self.reasons = []
        self.promoted = False  # created from persistent low-confidence candidates

    @property
    def frames(self):
        return sorted(self.boxes)

    @property
    def start(self):
        return min(self.boxes)

    @property
    def end(self):
        return max(self.boxes)

    def det_frames(self):
        return [f for f in self.frames if self.boxes[f][3] == 'd']

    def top_class(self):
        return int(np.argmax(self.cls_sum)) if self.cls_sum.size else 0

    def mean_score(self):
        s = [self.boxes[f][1] for f in self.boxes if self.boxes[f][3] in 'de']
        return float(np.mean(s)) if s else 0.0


def _velocity_ref(trk, path, frames):
    """Per-frame centre velocity (shot reference coords) from a few observations."""
    obs = [f for f in frames if trk.boxes[f][3] != 'i']
    if len(obs) < 2:
        return np.zeros(2)
    f0, f1 = obs[0], obs[-1]
    c0 = centers([path.to_ref(f0, trk.boxes[f0][0])])[0]
    c1 = centers([path.to_ref(f1, trk.boxes[f1][0])])[0]
    return (c1 - c0) / max(f1 - f0, 1)


def _shift(box, d):
    return np.asarray(box, np.float32) + np.array([d[0], d[1], d[0], d[1]], np.float32)


class Repairer:
    def __init__(self, cfg, fps, records, path, shots):
        self.cfg = cfg
        self.fps = fps
        self.records = records
        self.path = path
        self.shots = shots

    # ------------------------------------------------------------------ helpers
    def _used(self, tracks):
        used = {}
        for t in tracks:
            for f, (_b, _s, d, kind) in t.boxes.items():
                if kind in 'de' and d >= 0:
                    used.setdefault(f, set()).add(d)
        return used

    def _frame_index(self, tracks):
        idx = {}
        for t in tracks:
            for f, v in t.boxes.items():
                idx.setdefault(f, []).append((t.id, v[0]))
        return idx

    # --------------------------------------------------------------- 1. extension
    def extend(self, tracks):
        cfg = self.cfg
        used = self._used(tracks)
        by_frame = self._frame_index(tracks)
        max_steps = max(1, int(round(cfg.ext_max_sec * self.fps)))
        for t in tracks:
            shot_start, shot_end = self.shots[t.shot]
            for direction, key in ((-1, 'back'), (1, 'fwd')):
                frames = t.frames
                anchor = frames[0] if direction < 0 else frames[-1]
                ref_frames = frames[:6] if direction < 0 else frames[-6:]
                vel = _velocity_ref(t, self.path, ref_frames)
                last_f, last_ref = anchor, self.path.to_ref(anchor, t.boxes[anchor][0])
                cur, misses, added = anchor, 0, 0
                cls = t.top_class()
                for _ in range(max_steps):
                    nf = cur + direction
                    if nf < shot_start or nf > shot_end or nf in t.boxes:
                        break
                    cur = nf
                    rec = self.records.get(nf)
                    if rec is None:
                        continue  # stride-skipped frame; the gap filler handles it
                    pred = self.path.from_ref(nf, _shift(last_ref, vel * (nf - last_f)))
                    best, best_iou = -1, cfg.ext_iou
                    taken = used.get(nf, set())
                    if len(rec['boxes']):
                        ious = iou_matrix([pred], rec['boxes'])[0]
                        for d in np.argsort(-ious):
                            if ious[d] < best_iou:
                                break
                            if d in taken or float(rec['cls'][d][cls]) <= 0:
                                continue
                            best = int(d)
                            break
                    if best < 0:
                        misses += 1
                        if misses > cfg.ext_bridge:
                            break
                        continue
                    box = rec['boxes'][best]
                    others = [b for tid, b in by_frame.get(nf, []) if tid != t.id]
                    if others and iou_matrix([box], others).max() >= 0.7:
                        break  # walking into another object
                    t.boxes[nf] = (box.copy(), float(rec['scores'][best]), best, 'e')
                    t.cls_sum += rec['cls'][best].astype(np.float32) * 0.5
                    used.setdefault(nf, set()).add(best)
                    by_frame.setdefault(nf, []).append((t.id, box))
                    last_f, last_ref = nf, self.path.to_ref(nf, box)
                    misses = 0
                    added += 1
                t.ext[key] += added
        return tracks

    # ---------------------------------------------------------------- 2. linking
    def _mean_desc(self, t, frames):
        ds = []
        for f in frames:
            box, _s, d, kind = t.boxes[f]
            rec = self.records.get(f)
            if kind in 'de' and rec is not None and d >= 0 and d < len(rec['desc']):
                v = rec['desc'][d].astype(np.float32)
                if v.sum() > 0:
                    ds.append(v)
        return np.mean(ds, 0) if ds else None

    @staticmethod
    def bhattacharyya(a, b):
        if a is None or b is None:
            return None
        a, b = a / max(a.sum(), 1e-6), b / max(b.sum(), 1e-6)
        return float(np.sqrt(max(0.0, 1.0 - np.sum(np.sqrt(a * b)))))

    def link(self, tracks):
        cfg = self.cfg
        max_gap = max(1, int(round(cfg.link_gap_sec * self.fps)))
        while True:
            by_id = {t.id: t for t in tracks}
            cands = []
            starts = sorted(tracks, key=lambda t: t.start)
            for a in tracks:
                a_frames = a.frames
                a_end = a_frames[-1]
                a_box = a.boxes[a_end][0]
                a_ref = self.path.to_ref(a_end, a_box)
                a_size = float(np.sqrt(max(box_area([a_ref])[0], 16.0)))
                vel = _velocity_ref(a, self.path, a_frames[-8:])
                a_desc = self._mean_desc(a, a_frames[-5:])
                for b in starts:
                    if b.id == a.id or b.shot != a.shot:
                        continue
                    gap = b.start - a_end
                    if gap <= 0:
                        continue
                    if gap > max_gap:
                        break
                    if b.top_class() != a.top_class():
                        continue
                    b_ref = self.path.to_ref(b.start, b.boxes[b.start][0])
                    pred = _shift(a_ref, vel * min(gap, self.fps))
                    dist = float(np.linalg.norm(centers([pred])[0] - centers([b_ref])[0])) / a_size
                    allowed = cfg.link_dist * (1.0 + gap / self.fps)
                    ratio = float(box_area([b_ref])[0]) / max(float(box_area([a_ref])[0]), 1.0)
                    if dist > allowed or not (0.4 <= ratio <= 2.5):
                        continue
                    app = self.bhattacharyya(a_desc, self._mean_desc(b, b.frames[:5]))
                    if app is not None and app > cfg.link_app:
                        continue
                    cands.append((dist / allowed + (app or 0.0), a.id, b.id))
            if not cands:
                return tracks
            cands.sort()
            heads, tails, merged = set(), set(), set()
            for _cost, a_id, b_id in cands:
                if a_id in heads or b_id in tails or a_id in merged or b_id in merged:
                    continue
                a, b = by_id[a_id], by_id[b_id]
                a.boxes.update(b.boxes)
                a.cls_sum += b.cls_sum
                a.links += 1 + b.links
                a.ext['fwd'] = b.ext['fwd']
                heads.add(a_id)
                tails.add(b_id)
                merged.add(b_id)
            tracks = [t for t in tracks if t.id not in merged]

    # ------------------------------------------------------------ 3. duplicates
    def suppress_duplicates(self, tracks):
        cfg = self.cfg
        pair_iou = {}
        by_frame = {}
        for t in tracks:
            for f, v in t.boxes.items():
                by_frame.setdefault(f, []).append((t.id, v[0]))
        for f, items in by_frame.items():
            if len(items) < 2:
                continue
            ids = [i for i, _ in items]
            ious = iou_matrix([b for _, b in items], [b for _, b in items])
            rows, cols = np.where(np.triu(ious, 1) > 0.1)
            for r, c in zip(rows, cols):
                key = (min(ids[r], ids[c]), max(ids[r], ids[c]))
                acc = pair_iou.setdefault(key, [0.0, 0])
                acc[0] += float(ious[r, c])
                acc[1] += 1
        by_id = {t.id: t for t in tracks}
        dropped = set()
        for (i, j), (iou_sum, n) in sorted(pair_iou.items(), key=lambda kv: -kv[1][1]):
            if i in dropped or j in dropped:
                continue
            mean_iou = iou_sum / n
            a, b = by_id[i], by_id[j]
            if n >= 3:
                a.overlaps[j] = max(a.overlaps.get(j, 0.0), mean_iou)
                b.overlaps[i] = max(b.overlaps.get(i, 0.0), mean_iou)
            if n < cfg.dup_min_frames or mean_iou < cfg.dup_iou:
                continue
            weak, strong = sorted((a, b), key=lambda t: len(t.det_frames()) * t.mean_score())
            shared = set(weak.boxes) & set(strong.boxes)
            if len(shared) >= 0.6 * len(weak.boxes):
                dropped.add(weak.id)
                strong.overlaps.pop(weak.id, None)
            else:
                for f in shared:
                    weak.boxes.pop(f, None)
        return [t for t in tracks if t.id not in dropped and t.boxes], len(dropped)

    # ------------------------------------------------------------- 4. gap filling
    def fill_gaps(self, tracks):
        max_gap = max(1, int(round(self.cfg.gap_fill_sec * self.fps)))
        for t in tracks:
            frames = t.frames
            for f0, f1 in zip(frames[:-1], frames[1:]):
                gap = f1 - f0
                if gap <= 1 or gap - 1 > max_gap:
                    continue
                b0 = self.path.to_ref(f0, t.boxes[f0][0])
                b1 = self.path.to_ref(f1, t.boxes[f1][0])
                score = min(t.boxes[f0][1], t.boxes[f1][1])
                for f in range(f0 + 1, f1):
                    a = (f - f0) / gap
                    t.boxes[f] = (self.path.from_ref(f, (1 - a) * b0 + a * b1), score, -1, 'i')
        return tracks

    # --------------------------------------------------------------------- run
    def run(self, tracks):
        tracks = self.extend(tracks)
        tracks = self.link(tracks)
        tracks, n_dup = self.suppress_duplicates(tracks)
        tracks = self.fill_gaps(tracks)
        return tracks, n_dup


def class_vote(trk, class_names):
    total = float(trk.cls_sum.sum())
    if total <= 0:
        return 0, 1.0, {}
    shares = {class_names[i]: float(v) / total for i, v in enumerate(trk.cls_sum) if v > 0}
    top = trk.top_class()
    return top, float(trk.cls_sum[top]) / total, shares

