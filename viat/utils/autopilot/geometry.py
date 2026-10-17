"""Box and camera-motion geometry shared by the Autopilot pipeline.

All boxes are float ``[x1, y1, x2, y2]`` in pixel coordinates. Affine matrices are
2x3 (OpenCV convention) and map points from one frame into another.
"""
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment as _lsa
except ImportError:  # scipy is optional; fall back to greedy matching
    _lsa = None


def as_boxes(boxes):
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def box_area(boxes):
    b = as_boxes(boxes)
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def _intersection(a, b):
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0])
    iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2])
    iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    return np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)


def iou_matrix(a, b):
    a, b = as_boxes(a), as_boxes(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    inter = _intersection(a, b)
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def ioa_matrix(a, b):
    """Fraction of each box in ``a`` that is covered by each box in ``b``."""
    a, b = as_boxes(a), as_boxes(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    return _intersection(a, b) / np.maximum(box_area(a)[:, None], 1e-6)


def box_iou(a, b):
    return float(iou_matrix([a], [b])[0, 0])


def centers(boxes):
    b = as_boxes(boxes)
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], axis=1)


def xyxy_to_xyah(box):
    x1, y1, x2, y2 = [float(v) for v in box]
    w, h = max(x2 - x1, 1e-3), max(y2 - y1, 1e-3)
    return np.array([x1 + w / 2, y1 + h / 2, w / h, h], dtype=np.float64)


def xyah_to_xyxy(xyah):
    cx, cy, a, h = [float(v) for v in xyah[:4]]
    w = a * h
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


IDENTITY = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float64)


def to3x3(affine):
    m = np.eye(3, dtype=np.float64)
    m[:2] = affine
    return m


def warp_box(box, affine3):
    """Warp a box through a 3x3 transform and return the axis-aligned bounding box."""
    x1, y1, x2, y2 = [float(v) for v in box]
    pts = np.array([[x1, y1, 1], [x2, y1, 1], [x1, y2, 1], [x2, y2, 1]], dtype=np.float64)
    out = pts @ affine3.T
    xs, ys = out[:, 0], out[:, 1]
    # Keep the box size stable under rotation: use the centre + scaled size instead
    # of the (inflated) hull of rotated corners.
    cx, cy = xs.mean(), ys.mean()
    scale = np.sqrt(abs(np.linalg.det(affine3[:2, :2]))) or 1.0
    hw, hh = (x2 - x1) * scale / 2, (y2 - y1) * scale / 2
    return np.array([cx - hw, cy - hh, cx + hw, cy + hh], dtype=np.float32)


def linear_assignment(cost, max_cost):
    """Min-cost one-to-one matching; pairs with cost > max_cost are not matched."""
    if cost.size == 0:
        return [], list(range(cost.shape[0])), list(range(cost.shape[1]))
    if _lsa is not None:
        rows, cols = _lsa(np.where(cost > max_cost, max_cost + 1e3, cost))
        pairs = [(r, c) for r, c in zip(rows, cols) if cost[r, c] <= max_cost]
    else:
        pairs = []
        used_r, used_c = set(), set()
        for idx in np.argsort(cost, axis=None):
            r, c = divmod(int(idx), cost.shape[1])
            if cost[r, c] > max_cost:
                break
            if r in used_r or c in used_c:
                continue
            pairs.append((r, c))
            used_r.add(r)
            used_c.add(c)
    matched_r = {r for r, _ in pairs}
    matched_c = {c for _, c in pairs}
    return (pairs,
            [r for r in range(cost.shape[0]) if r not in matched_r],
            [c for c in range(cost.shape[1]) if c not in matched_c])


def cluster_fuse(boxes, scores, cls_vecs, model_ids, iou_thr, n_models, constant_models=()):
    """Fuse overlapping detections from several views/models (WBF-style).

    Detections are clustered greedily by score at ``iou_thr`` regardless of class, so
    the same object reported as "car" and "truck" becomes one candidate whose class
    vector keeps both scores for later voting. The fused box is the score-weighted
    mean; the fused score is the best member score, scaled by how many models agree
    when more than one model ran.
    """
    boxes = as_boxes(boxes)
    n = len(boxes)
    if n == 0:
        return (np.zeros((0, 4), np.float32), np.zeros(0, np.float32),
                np.zeros((0, cls_vecs.shape[1] if cls_vecs.ndim == 2 else 0), np.float32),
                np.zeros(0, np.int8))
    scores = np.asarray(scores, dtype=np.float32)
    ious = iou_matrix(boxes, boxes)
    order = np.argsort(-scores)
    assigned = np.zeros(n, dtype=bool)
    out_b, out_s, out_c, out_m = [], [], [], []
    for i in order:
        if assigned[i]:
            continue
        members = np.where((ious[i] >= iou_thr) & ~assigned)[0]
        assigned[members] = True
        w = scores[members][:, None]
        out_b.append((boxes[members] * w).sum(0) / max(float(w.sum()), 1e-6))
        out_c.append(cls_vecs[members].max(0))
        models = {int(model_ids[m]) for m in members}
        best = float(scores[members].max())
        if n_models > 1:
            agree = len(models) / n_models
            best *= (1.0 + agree) / 2.0
        # A cluster made only of constant-score models (Florence-2, LocateAnything)
        # can support a track but never create one on its own.
        if models and models.issubset(set(constant_models)):
            best = min(best, 0.3)
        out_s.append(best)
        out_m.append(len(models))
    return (np.asarray(out_b, np.float32), np.asarray(out_s, np.float32),
            np.asarray(out_c, np.float32), np.asarray(out_m, np.int8))


def box_nms(boxes, scores, iou_thresh=0.4):
    """Greedy Non-Maximum Suppression on boxes and scores. Returns list of kept indices."""
    b = as_boxes(boxes)
    s = np.asarray(scores, dtype=np.float32)
    if len(b) == 0:
        return []
    order = np.argsort(-s)
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        ious = iou_matrix(b[i:i + 1], b[order[1:]])[0]
        order = order[1:][ious < iou_thresh]
    return keep

