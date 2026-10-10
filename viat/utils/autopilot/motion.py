"""Camera-motion estimation and shot-cut detection for the Autopilot pipeline.

Drone and handheld footage moves the camera, so every motion test in the tracker
runs in camera-compensated coordinates (the BoT-SORT "GMC" idea): sparse optical
flow between consecutive frames, fitted with a RANSAC similarity transform.
"""
import cv2
import numpy as np

from .geometry import IDENTITY, to3x3, warp_box


class MotionEstimator:
    def __init__(self, max_side=960, max_corners=600):
        self.max_side = max_side
        self.max_corners = max_corners
        self.prev_gray = None
        self.prev_hist = None

    def reset(self):
        self.prev_gray = None
        self.prev_hist = None

    def update(self, frame_bgr):
        """Return ``(affine_prev_to_cur, inlier_ratio, hist_corr)`` for this frame.

        The affine maps full-resolution coordinates of the previous frame into the
        current one. ``hist_corr`` is the colour-histogram correlation with the
        previous frame (low values indicate a cut).
        """
        h, w = frame_bgr.shape[:2]
        scale = min(1.0, self.max_side / float(max(h, w)))
        small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale))) if scale < 1.0 else frame_bgr
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist)

        affine, inlier_ratio, hist_corr = IDENTITY.copy(), 1.0, 1.0
        if self.prev_gray is not None:
            hist_corr = float(cv2.compareHist(self.prev_hist, hist, cv2.HISTCMP_CORREL))
            affine, inlier_ratio = self._estimate(self.prev_gray, gray, scale)
        self.prev_gray, self.prev_hist = gray, hist
        return affine, inlier_ratio, hist_corr

    def _estimate(self, prev, cur, scale):
        pts = cv2.goodFeaturesToTrack(prev, maxCorners=self.max_corners, qualityLevel=0.01,
                                      minDistance=8, blockSize=3)
        if pts is None or len(pts) < 12:
            return IDENTITY.copy(), 0.0
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None, winSize=(21, 21), maxLevel=3)
        if nxt is None:
            return IDENTITY.copy(), 0.0
        ok = status.reshape(-1) == 1
        p0, p1 = pts.reshape(-1, 2)[ok], nxt.reshape(-1, 2)[ok]
        if len(p0) < 12:
            return IDENTITY.copy(), 0.0
        affine, inliers = cv2.estimateAffinePartial2D(p0, p1, method=cv2.RANSAC,
                                                      ransacReprojThreshold=3.0)
        if affine is None or inliers is None:
            return IDENTITY.copy(), 0.0
        affine = affine.astype(np.float64)
        affine[:, 2] /= scale  # back to full-resolution pixels
        return affine, float(inliers.sum()) / float(len(pts))


def is_cut(hist_corr, inlier_ratio, hard_thr=0.35, soft_thr=0.6, flow_thr=0.25):
    """A hard cut: colours change abruptly, or change a lot and flow tracking fails."""
    return hist_corr < hard_thr or (hist_corr < soft_thr and inlier_ratio < flow_thr)


class CameraPath:
    """Maps boxes between each frame and its shot's first frame (the shot reference)."""

    def __init__(self, motion, shots):
        self.to_ref_m = {}
        self.from_ref_m = {}
        self.shot_of = {}
        for shot_idx, (start, end) in enumerate(shots):
            m = np.eye(3)
            for f in range(start, end + 1):
                if f > start:
                    affine = motion.get(f, (IDENTITY, 1.0, 1.0))[0]
                    # affine maps f-1 -> f; invert to go f -> f-1, then f-1 -> ref.
                    m = m @ np.linalg.inv(to3x3(affine))
                self.to_ref_m[f] = m
                self.from_ref_m[f] = np.linalg.inv(m)
                self.shot_of[f] = shot_idx

    def to_ref(self, frame, box):
        return warp_box(box, self.to_ref_m.get(frame, np.eye(3)))

    def from_ref(self, frame, box):
        return warp_box(box, self.from_ref_m.get(frame, np.eye(3)))


def split_shots(frames, motion, enabled=True):
    """Split a sorted frame list into shots at detected cuts (and at frame gaps)."""
    shots = []
    if not frames:
        return shots
    start = prev = frames[0]
    for f in frames[1:]:
        cut = f != prev + 1
        if enabled and not cut:
            _, inlier, hist = motion.get(f, (IDENTITY, 1.0, 1.0))
            cut = is_cut(hist, inlier)
        if cut:
            shots.append((start, prev))
            start = f
        prev = f
    shots.append((start, prev))
    return shots
