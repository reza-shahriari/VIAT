"""Per-frame candidate detection for the Autopilot pipeline.

Runs every selected detector (its own model instances, so the interactive Auto Label
tool is never disturbed) on the full frame, optional overlapping tiles and an
optional horizontal flip. Views of the same model are merged here; merging across
models happens later in ``fusion.py`` because it needs per-video score calibration.
Candidates are kept at a low score on purpose: recall is decided here, precision
later by the tracker, the offline repair, the judges and the review.
"""
import cv2
import numpy as np

from .geometry import cluster_fuse
from .models import make_backend, match_prompt

DESCRIPTOR_DIM = 32


def appearance_descriptor(frame, box):
    """Small HSV colour histogram of the central part of a box (used for re-linking)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    mx, my = (x2 - x1) * 0.1, (y2 - y1) * 0.1
    x1, y1 = int(max(0, x1 + mx)), int(max(0, y1 + my))
    x2, y2 = int(min(w, x2 - mx)), int(min(h, y2 - my))
    if x2 - x1 < 3 or y2 - y1 < 3:
        return np.zeros(DESCRIPTOR_DIM, np.float16)
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [8, 4], [0, 180, 0, 256]).reshape(-1)
    s = hist.sum()
    return (hist / s if s > 0 else hist).astype(np.float16)


def tile_grid(width, height, n, overlap):
    if n <= 1:
        return []
    tw = int(np.ceil(width / (n - (n - 1) * overlap)))
    th = int(np.ceil(height / (n - (n - 1) * overlap)))
    xs = np.linspace(0, max(0, width - tw), n).astype(int)
    ys = np.linspace(0, max(0, height - th), n).astype(int)
    return [(int(x), int(y), int(min(width, x + tw)), int(min(height, y + th))) for y in ys for x in xs]


class CandidateDetector:
    def __init__(self, cfg, log=print):
        self.cfg = cfg
        self.log = log
        self.prompts, self.prompt_class = cfg.prompt_table()
        self.n_classes = len(cfg.classes)
        self.backends = []
        self.exemplars = []

    def set_exemplars(self, exemplars):
        """Provide optional visual queries to backends that support them."""
        self.exemplars = list(exemplars or [])
        for backend in self.backends:
            backend.set_exemplars(self.exemplars)

    def has(self, key):
        return any(b.key == key for b in self.backends)

    def load_one(self, key):
        """Load one model; returns an error string, or None when it is ready."""
        backend = make_backend(key, self.log)
        try:
            ok, msg = backend.load()
        except Exception as exc:
            ok, msg = False, str(exc)
        if not ok:
            return f"{key}: {msg}"
        backend.set_prompts(self.prompts)
        backend.set_exemplars(self.exemplars)
        self.backends.append(backend)
        self.log(f"[Autopilot] loaded detector {key}")
        return None

    def _run(self, backend, image):
        try:
            return backend.predict(image)
        except Exception as exc:  # one bad frame must not kill an overnight run
            self.log(f"[Autopilot] {backend.key} error: {exc}")
            return []

    def detect(self, frame, keys=None):
        """Return ``{model_key: {'boxes','scores','cls','desc'}}`` with raw (uncalibrated) scores."""
        h, w = frame.shape[:2]
        views = [(frame, 0, 0, False, None)]
        for (x1, y1, x2, y2) in tile_grid(w, h, self.cfg.tiles, self.cfg.tile_overlap):
            views.append((np.ascontiguousarray(frame[y1:y2, x1:x2]), x1, y1, False, (x1, y1, x2, y2)))
        if self.cfg.flip_tta:
            views.append((np.ascontiguousarray(frame[:, ::-1]), 0, 0, True, None))
        out = {}
        for backend in self.backends:
            if keys is not None and backend.key not in keys:
                continue
            boxes, scores, cls_vecs = [], [], []
            for image, ox, oy, flipped, tile in views:
                for box, score, label in self._run(backend, image):
                    c = match_prompt(label, self.prompt_class)
                    if c is None:
                        continue
                    x1, y1, x2, y2 = [float(v) for v in box]
                    if flipped:
                        x1, x2 = w - x2, w - x1
                    if tile is not None and self._touches_inner_edge((x1, y1, x2, y2), tile, w, h):
                        continue  # cut by a tile border; another tile or the full frame sees it whole
                    x1, x2, y1, y2 = x1 + ox, x2 + ox, y1 + oy, y2 + oy
                    if x2 - x1 < 1 or y2 - y1 < 1:
                        continue
                    vec = np.zeros(self.n_classes, np.float32)
                    vec[c] = score
                    boxes.append([x1, y1, x2, y2])
                    scores.append(score)
                    cls_vecs.append(vec)
            cls_arr = np.asarray(cls_vecs, np.float32).reshape(-1, self.n_classes)
            fb, fs, fc, _ = cluster_fuse(boxes, scores, cls_arr, np.zeros(len(boxes), int),
                                         self.cfg.fuse_iou, 1)
            if not backend.constant:  # drop what stays below the candidate level after calibration
                keep = 0.5 * fs / backend.ref_score >= self.cfg.candidate_score
                fb, fs, fc = fb[keep], fs[keep], fc[keep]
            desc = np.zeros((len(fb), DESCRIPTOR_DIM), np.float16)
            for i in range(len(fb)):
                desc[i] = appearance_descriptor(frame, fb[i])
            out[backend.key] = {'boxes': fb, 'scores': fs, 'cls': fc.astype(np.float16), 'desc': desc}
        return out

    @staticmethod
    def _touches_inner_edge(box, tile, width, height, margin=3):
        x1, y1, x2, y2 = box
        tx1, ty1, tx2, ty2 = tile
        tw, th = tx2 - tx1, ty2 - ty1
        return ((tx1 > 0 and x1 <= margin) or (ty1 > 0 and y1 <= margin) or
                (tx2 < width and x2 >= tw - margin) or (ty2 < height and y2 >= th - margin))
