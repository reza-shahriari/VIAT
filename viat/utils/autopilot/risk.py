"""Track risk scoring and possible-miss discovery for the Autopilot pipeline.

Each signal is a published failure indicator (SAM 3 detector support, DAM4SAM /
AWS geometry jumps, class-vote margin, fragment links, overlaps...). Signals are
combined as a noisy-OR into a 0..1 risk, and every signal leaves a human-readable
reason with the frame to look at, so the reviewer sees *why* a track is flagged.
"""
import numpy as np

from .geometry import iou_matrix, ioa_matrix, box_area, centers
from .repair import class_vote


def _noisy_or(ps):
    keep = 1.0
    for p in ps:
        keep *= 1.0 - min(max(p, 0.0), 0.99)
    return 1.0 - keep


def score_track(trk, ctx):
    """Return ``(risk, reasons, features)``.

    reasons are ``(code, text, frame, weight)``; features are the numbers the learned
    judge is trained on (calibrated scores: a model's typical confident score is 0.5).
    """
    fps, path, records, n_models, names = ctx['fps'], ctx['path'], ctx['records'], ctx['n_models'], ctx['names']
    frames = trk.frames
    det = trk.det_frames()
    span = frames[-1] - frames[0] + 1
    reasons = []

    def add(code, text, frame, p):
        if p > 0.01:
            reasons.append((code, text, int(frame), float(p)))

    mean = trk.mean_score()
    if mean < 0.8:
        add('low_score', f'low detector confidence (avg {mean:.2f})', frames[len(frames) // 2],
            0.8 * (1.0 - mean / 0.8))

    # Part-vs-whole confusion: a weak track that lives inside another object's box is
    # often a piece of that object (a hatch or antenna read as "person").
    inside, host = 0, {}
    for f in frames:
        b = trk.boxes[f][0]
        others = [(tid, ob) for tid, ob in ctx['by_frame'].get(f, []) if tid != trk.id]
        if not others:
            continue
        ioa = ioa_matrix([b], [ob for _, ob in others])[0]
        j = int(np.argmax(ioa))
        if ioa[j] >= 0.8:
            inside += 1
            host[others[j][0]] = host.get(others[j][0], 0) + 1
    if host and inside >= 0.5 * len(frames) and mean < 0.8:
        hid = max(host.items(), key=lambda kv: kv[1])[0]
        add('inside_other', f'lies inside track #{hid} - may be a part of it', frames[len(frames) // 2], 0.35)

    support = len(det) / float(span)
    longest, run_start, cur, cur_start = 0, frames[0], 0, frames[0]
    det_set = set(det)
    for f in range(frames[0], frames[-1] + 1):
        if f in det_set:
            cur = 0
        else:
            if cur == 0:
                cur_start = f
            cur += 1
            if cur > longest:
                longest, run_start = cur, cur_start
    if support < 0.5:
        add('weak_support', f'detector saw it on only {support:.0%} of its frames',
            run_start, 0.5 * (1.0 - support / 0.5))
    if longest > fps:
        add('unsupported_run', f'no detection for {longest} frames (from frame {run_start + 1})',
            run_start, min(0.5, 0.2 * longest / fps))

    top, margin, shares = class_vote(trk, names)
    if margin < 0.75 and len(shares) > 1:
        parts = ' / '.join(f'{k} {v:.0%}' for k, v in sorted(shares.items(), key=lambda kv: -kv[1])[:3])
        add('class_unsure', f'class unsure: {parts}', frames[len(frames) // 2], 0.7 * (0.75 - margin) / 0.75)

    # Box jumps, judged in camera-compensated coordinates against the track's own
    # typical motion, with a pixel floor so detector jitter on tiny boxes is not a jump.
    jumps, first_jump = 0, None
    obs = [f for f in frames if trk.boxes[f][3] in 'de']
    refs = {f: path.to_ref(f, trk.boxes[f][0]) for f in obs}
    steps = [(f0, f1) for f0, f1 in zip(obs[:-1], obs[1:]) if f1 - f0 <= 2]
    moves = [np.linalg.norm(centers([refs[f1]])[0] - centers([refs[f0]])[0]) / (f1 - f0) for f0, f1 in steps]
    typical = float(np.median(moves)) if moves else 0.0
    for (f0, f1), move in zip(steps, moves):
        b0, b1 = refs[f0], refs[f1]
        a0, a1 = float(box_area([b0])[0]), float(box_area([b1])[0])
        size = float(np.sqrt(max(min(a0, a1), 1.0)))
        ratio = max(a0, a1) / max(min(a0, a1), 1.0)
        moved = move - typical > max(0.75 * size, 4.0)
        resized = ratio > 1.8 and size >= 16 and iou_matrix([b0], [b1])[0, 0] < 0.5
        if moved or resized:
            jumps += 1
            if first_jump is None:
                first_jump = f1
    if jumps:
        add('box_jump', f'box jumps {jumps}x (first at frame {first_jump + 1})', first_jump, min(0.6, 0.25 * jumps))

    if getattr(trk, 'promoted', False):
        add('persistent_low', 'never confidently detected, but seen consistently - confirm it is real',
            frames[len(frames) // 2], 0.4)

    if trk.links:
        add('linked', f'joined from {trk.links + 1} fragments - check identity', frames[0], min(0.5, 0.2 * trk.links))

    ext = trk.ext['back'] + trk.ext['fwd']
    if ext > fps / 2:
        f = frames[0] if trk.ext['back'] >= trk.ext['fwd'] else frames[-1]
        add('extended', f'extended {ext} frames along weak detections', f, 0.15)

    if span < fps:
        add('short', f'short track ({span / fps:.1f} s)', frames[len(frames) // 2], 0.3 * (1.0 - span / fps))

    sides = [min(trk.boxes[f][0][2] - trk.boxes[f][0][0], trk.boxes[f][0][3] - trk.boxes[f][0][1]) for f in frames]
    med = float(np.median(sides))
    if med < 12:
        add('tiny', f'tiny object (~{med:.0f} px)', frames[len(frames) // 2], 0.2)

    if trk.overlaps:
        other, ov = max(trk.overlaps.items(), key=lambda kv: kv[1])
        if ov > 0.3:
            add('overlap', f'overlaps track #{other} (IoU {ov:.2f})', frames[len(frames) // 2], 0.4)

    if n_models > 1 and det:
        single = sum(1 for f in det if int(records[f]['nmod'][trk.boxes[f][2]]) <= 1) / float(len(det))
        if single > 0.8:
            add('one_model', f'only 1 of {n_models} detectors found it', frames[len(frames) // 2], 0.25)

    if len(det) >= 9:
        third = len(det) // 3
        d0 = ctx['desc'](trk, det[:third])
        d1 = ctx['desc'](trk, det[-third:])
        dist = ctx['bhatt'](d0, d1)
        if dist is not None and dist > 0.6:
            add('appearance', 'appearance changes along the track (possible ID switch)', det[-third], 0.35)

    reasons.sort(key=lambda r: -r[3])
    det_scores = [trk.boxes[f][1] for f in det] or [0.0]
    agree = [float(records[f]['nmod'][trk.boxes[f][2]]) for f in det
             if records.get(f) is not None and 0 <= trk.boxes[f][2] < len(records[f]['nmod'])]
    feats = {
        'mean_score': mean, 'max_score': float(max(det_scores)), 'p25_score': float(np.percentile(det_scores, 25)),
        'support': support, 'span_sec': span / fps, 'log_size': float(np.log1p(med)),
        'links': float(trk.links), 'ext_frac': ext / float(len(frames)), 'class_margin': margin,
        'agree': (float(np.mean(agree)) / max(n_models, 1)) if agree else 0.0,
        'inside_frac': inside / float(len(frames)), 'jumps': float(jumps),
        'overlap': float(max(trk.overlaps.values())) if trk.overlaps else 0.0,
        'promoted': float(bool(getattr(trk, 'promoted', False))),
        'longest_gap_sec': longest / fps,
    }
    feats = {k: round(float(v), 4) for k, v in feats.items()}
    return _noisy_or([r[3] for r in reasons]), reasons, feats


def find_possible_misses(records, tracks, frames, path, cfg, fps, names):
    """Groups of confident-ish candidates that persist but were never tracked."""
    covered = {}
    for t in tracks:
        for f, v in t.boxes.items():
            covered.setdefault(f, []).append(v[0])
    min_len = max(5, int(round(cfg.miss_min_sec * fps)))
    open_groups, done = [], []
    for f in frames:
        rec = records.get(f)
        if rec is None:
            continue
        idx = np.where(rec['scores'] >= cfg.miss_score)[0]
        if len(idx) and covered.get(f):
            iou = iou_matrix(rec['boxes'][idx], covered[f]).max(1)
            idx = idx[iou < 0.3]
        still_open = []
        taken = set()
        for g in open_groups:
            last_f, last_box = g['last']
            if f - last_f > 3 or path.shot_of.get(f) != path.shot_of.get(last_f):
                done.append(g)
                continue
            prev_ref = path.to_ref(last_f, last_box)
            cand = [i for i in idx if i not in taken]
            if cand:
                ious = iou_matrix([path.from_ref(f, prev_ref)], rec['boxes'][cand])[0]
                j = int(np.argmax(ious))
                if ious[j] >= 0.3:
                    i = cand[j]
                    taken.add(i)
                    g['boxes'][f] = (rec['boxes'][i].copy(), float(rec['scores'][i]))
                    g['cls'] += rec['cls'][i].astype(np.float32)
                    g['last'] = (f, rec['boxes'][i])
            still_open.append(g)
        for i in idx:
            if i not in taken:
                still_open.append({'boxes': {f: (rec['boxes'][i].copy(), float(rec['scores'][i]))},
                                   'cls': rec['cls'][i].astype(np.float32), 'last': (f, rec['boxes'][i])})
        open_groups = still_open
    done.extend(open_groups)
    misses = []
    for g in done:
        if len(g['boxes']) < min_len:
            continue
        fs = sorted(g['boxes'])
        scores = [g['boxes'][f][1] for f in fs]
        misses.append({
            'class': names[int(np.argmax(g['cls']))] if len(names) else '',
            'start': fs[0], 'end': fs[-1], 'mean_score': float(np.mean(scores)),
            'coverage': len(fs) / float(fs[-1] - fs[0] + 1),
            'frame': fs[int(np.argmax(scores))],
            'boxes': [[int(f)] + [round(float(v), 1) for v in g['boxes'][f][0]] + [round(g['boxes'][f][1], 3)] for f in fs],
        })
    misses.sort(key=lambda m: -(m['mean_score'] * len(m['boxes'])))
    return misses
