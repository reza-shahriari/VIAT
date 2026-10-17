"""Cross-model calibration and fusion for Autopilot.

1. Each model's raw scores are mapped onto one scale: the model's typical
   "confident" score (``ref_score``) becomes 0.5. Optionally the scale adapts per
   video when a model is systematically under-confident there (bounded, so a video
   full of junk cannot inflate itself).
2. Candidates of all models are clustered per frame (WBF-style). The fused score is
   the best calibrated score, raised when several models agree. Each fused
   candidate keeps every model's calibrated score (``mscores``), which the judges
   and the learned judge use ("YOLO-World saw it, SAM 3 did not").
"""
import numpy as np

from .geometry import iou_matrix
from .models import make_backend

CONSTANT_VALUE = 0.35  # calibrated score of a constant-score model's box (votes, never births)


def model_refs(keys):
    refs, constant = {}, {}
    for k in keys:
        b = make_backend(k)
        refs[k] = b.ref_score
        constant[k] = b.constant
    return refs, constant


def calibration(raw, keys, cfg):
    """Per-model multiplier mapping raw scores onto the common scale for this video."""
    refs, constant = model_refs(keys)
    scale = {}
    for k in keys:
        if constant[k]:
            scale[k] = None
            continue
        s = 0.5 / refs[k]
        if cfg.video_calibration:
            vals = np.concatenate([r[k]['scores'] for r in raw.values() if r is not None and k in r] or
                                  [np.zeros(0)])
            if len(vals) >= 200:
                p95 = float(np.percentile(vals, 95))
                if p95 * s < 0.6:  # under-confident here: lift, but at most 1.6x
                    s *= min(1.6, 0.6 / max(p95 * s, 1e-6))
        scale[k] = s
    return scale


def fuse_frame(rec, keys, scale, cfg, n_classes):
    """Fuse one frame's per-model candidates into the tracker's record format."""
    boxes, scores, cls, desc, mids = [], [], [], [], []
    for mi, k in enumerate(keys):
        r = rec.get(k) if rec else None
        if r is None or not len(r['scores']):
            continue
        if scale[k] is None:
            s = np.full(len(r['scores']), CONSTANT_VALUE, np.float32)
            c = (r['cls'].astype(np.float32) > 0) * CONSTANT_VALUE
        else:
            s = np.clip(r['scores'] * scale[k], 0, 1).astype(np.float32)
            c = np.clip(r['cls'].astype(np.float32) * scale[k], 0, 1)
        boxes.append(r['boxes'])
        scores.append(s)
        cls.append(c)
        desc.append(r['desc'])
        mids.append(np.full(len(s), mi))
    m = len(keys)
    if not boxes:
        return {'boxes': np.zeros((0, 4), np.float32), 'scores': np.zeros(0, np.float32),
                'cls': np.zeros((0, n_classes), np.float16), 'nmod': np.zeros(0, np.int8),
                'desc': np.zeros((0, 32), np.float16), 'mscores': np.zeros((0, m), np.float16)}
    boxes, scores = np.concatenate(boxes), np.concatenate(scores)
    cls, desc, mids = np.concatenate(cls), np.concatenate(desc), np.concatenate(mids)
    constant_ids = {i for i, k in enumerate(keys) if scale[k] is None}
    real_models = max(1, m - len(constant_ids))
    ious = iou_matrix(boxes, boxes)
    order = np.argsort(-scores)
    taken = np.zeros(len(boxes), bool)
    ob, os_, oc, on, od, om = [], [], [], [], [], []
    for i in order:
        if taken[i]:
            continue
        # one member per model: a model's own duplicates were merged already
        members = [j for j in np.where((ious[i] >= cfg.fuse_iou) & ~taken)[0]]
        best_per_model = {}
        for j in members:
            if mids[j] not in best_per_model or scores[j] > scores[best_per_model[mids[j]]]:
                best_per_model[mids[j]] = j
        members = list(best_per_model.values())
        taken[members] = True
        mode = getattr(cfg, 'fusion_box_mode', 'best')
        if mode == 'best':
            best_idx = max(members, key=lambda j: scores[j])
            ob.append(boxes[best_idx])
        elif mode == 'union':
            mb = boxes[members]
            ob.append(np.array([mb[:, 0].min(), mb[:, 1].min(), mb[:, 2].max(), mb[:, 3].max()], dtype=np.float32))
        else:
            w = scores[members][:, None]
            ob.append((boxes[members] * w).sum(0) / max(float(w.sum()), 1e-6))
        oc.append(cls[members].max(0))
        ms = np.zeros(m, np.float32)
        for j in members:
            ms[mids[j]] = scores[j]
        om.append(ms)
        voters = {int(mids[j]) for j in members}
        real = voters - constant_ids
        best = float(max(scores[j] for j in members if int(mids[j]) in real)) if real else CONSTANT_VALUE
        if m > 1:
            # constant-score models count as half a vote
            agree = (len(real) + 0.5 * len(voters & constant_ids)) / (real_models + 0.5 * len(constant_ids))
            best *= (1.0 + agree) / 2.0
            if not real:
                best = min(best, CONSTANT_VALUE)
        os_.append(best)
        on.append(len(real) if real else 0)
        od.append(desc[max(members, key=lambda j: scores[j])])
    return {'boxes': np.asarray(ob, np.float32), 'scores': np.asarray(os_, np.float32),
            'cls': np.asarray(oc, np.float16), 'nmod': np.asarray(on, np.int8),
            'desc': np.asarray(od, np.float16), 'mscores': np.asarray(om, np.float16),
            'n_real_models': real_models}
