"""Measure Autopilot (detectors, union, judges) against hand-labeled videos.

For every VIDEO.mp4 with a VIAT/Raya label file VIDEO.txt next to it:
  * runs Autopilot with the class names from the label file (prompts from
    autopilot_prompts.json), judges in score-only mode,
  * marks each track real/false by matching it against the hand labels,
  * reports box precision/recall before and after judge auto-rejection, and how well
    each judge separates real from false tracks (AUC, best reject threshold),
  * optionally logs every track as a decision for the learned judge and trains it.

    python autopilot_benchmark.py --videos ~/Videos/Done --limit 10 --frames 2000 \
        --model yolov8x-worldv2.pt --model sam3.1_l.pt --judge clip --judge sam_zoom \
        --compare-models --export-decisions --train-judge

Nothing is downloaded and VIAT projects are never touched; results go to the
benchmark work dir and the Markdown report.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

from viat.utils.autopilot import AutopilotConfig, FrameSource, run_autopilot
from viat.utils.autopilot.judges import combine
from viat.utils.autopilot.learned import LearnedJudge, log_decision, train
from viat.utils.autopilot.weights import is_available, missing_message

QUALITY = {'max': (1, 2, True), 'high': (1, 2, False), 'balanced': (1, 0, False), 'fast': (3, 0, False)}
BENCH_DIR = os.path.join(os.path.expanduser('~'), '.config', 'VideoAnnotationTool', 'autopilot_bench')


# --------------------------------------------------------------------------- labels
def read_labels(txt_path):
    text = open(txt_path, encoding='utf-8', errors='replace').read()
    parts = text.split('###')
    header = parts[1] if len(parts) >= 3 else ''
    names = [n.strip() for n in re.findall(r'^- (.+)$', header, re.M)]
    gt = {}
    for f, line in enumerate(parts[-1].strip().splitlines()):
        objs = []
        for o in re.findall(r'\[([^\]]*)\]', line):
            v = o.split(',')
            if len(v) < 5 or not v[0].strip():
                continue
            c, x, y, w, h = int(float(v[0])), float(v[1]), float(v[2]), float(v[3]), float(v[4])
            objs.append((c, [x, y, x + w, y + h]))
        if objs:
            gt[f] = objs
    return names, gt


def prompts_for(name, table):
    key = re.sub(r'[^a-z]', '', name.lower())
    if key in table:
        return table[key]
    words = re.sub(r'(?<!^)(?=[A-Z])', ' ', name).replace('_', ' ').lower().split()
    return [' '.join(words)] if words else [name]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


# ----------------------------------------------------------------------- evaluation
def label_tracks(result, gt, names):
    """real = >=50% of its boxes overlap a hand label (IoU>=0.5); class_ok = the matched label's class."""
    for t in result['tracks']:
        hits, same = 0, 0
        for row in t['boxes']:
            best, best_c = 0.0, None
            for c, b in gt.get(row[0], []):
                v = iou(row[1:5], b)
                if v > best:
                    best, best_c = v, c
            if best >= 0.5:
                hits += 1
                if best_c is not None and best_c < len(names) and \
                        names[best_c].lower() == t['class'].lower():
                    same += 1
        t['_real'] = hits >= 0.5 * len(t['boxes'])
        t['_class_ok'] = t['_real'] and same >= 0.5 * hits


def box_metrics(result, gt, keep, start, end):
    tp = fp = fn = 0
    bad = 0
    pred = {}
    for t in result['tracks']:
        if keep(t):
            for row in t['boxes']:
                pred.setdefault(row[0], []).append(row[1:5])
    for f in range(start, end + 1):
        g = [b for _, b in gt.get(f, [])]
        p = pred.get(f, [])
        used = set()
        err = False
        for gb in g:
            best, bi = 0.0, -1
            for i, pb in enumerate(p):
                if i not in used:
                    v = iou(gb, pb)
                    if v > best:
                        best, bi = v, i
            if best >= 0.5:
                tp += 1
                used.add(bi)
            else:
                fn += 1
                err = True
        if len(p) > len(used):
            fp += len(p) - len(used)
            err = True
        bad += err
    n = end - start + 1
    return {'precision': tp / max(1, tp + fp), 'recall': tp / max(1, tp + fn), 'tp': tp, 'fp': fp, 'fn': fn,
            'bad_frames': bad / max(1, n)}


def auc(pos, neg):
    """Probability a real track scores higher than a false one (Mann-Whitney)."""
    if not pos or not neg:
        return None
    pos, neg = np.asarray(pos), np.asarray(neg)
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg)))


def judge_table(rows):
    """For each judge: AUC and the box-weighted effect of rejecting tracks with p <= t."""
    out = {}
    for key in ('clip', 'sam_zoom', 'vlm', 'combined', 'learned'):
        vals = [(r['p'].get(key), r['real'], r['n']) for r in rows if r['p'].get(key) is not None]
        if not vals:
            continue
        a = auc([p for p, real, _ in vals if real], [p for p, real, _ in vals if not real])
        thr = []
        for t in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
            rej = [(real, n) for p, real, n in vals if p <= t]
            rej_boxes = sum(n for _, n in rej)
            false_boxes = sum(n for real, n in rej if not real)
            all_false = sum(n for _, real, n in vals if not real)
            thr.append({'t': t, 'rejected_tracks': len(rej), 'precision': false_boxes / rej_boxes if rej_boxes else None,
                        'false_removed': false_boxes / all_false if all_false else None,
                        'real_boxes_lost': rej_boxes - false_boxes})
        ok = [x for x in thr if x['precision'] is not None and x['precision'] >= 0.95]
        out[key] = {'n': len(vals), 'auc': a, 'thresholds': thr, 'suggested': max(ok, key=lambda x: x['t'])['t'] if ok else None}
    return out


# ------------------------------------------------------------------------------ run
def make_cfg(args, names, table, models, judges, frames_total):
    stride, tiles, flip = QUALITY[args.quality]
    classes = []
    seen = set()
    for n in names:
        if n.lower() in seen:
            continue
        seen.add(n.lower())
        classes.append({'name': n, 'prompts': prompts_for(n, table)})
    end = min(frames_total, args.frames) - 1 if args.frames else -1
    return AutopilotConfig(classes=classes, models=models, start_frame=0, end_frame=end, stride=stride, tiles=tiles,
                           flip_tta=flip, judges=judges, auto_reject=False, vlm_on='all',
                           **({'clip_model': args.clip_model} if args.clip_model else {}))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--videos', help='folder with VIDEO.mp4 + VIDEO.txt pairs')
    ap.add_argument('--video', action='append', default=[], help='one video (its .txt must sit next to it)')
    ap.add_argument('--limit', type=int, default=0, help='max videos')
    ap.add_argument('--frames', type=int, default=1500, help='max frames per video (0 = all)')
    ap.add_argument('--model', dest='models', action='append', help='detector (repeatable)')
    ap.add_argument('--judge', dest='judges', action='append', default=None,
                    help='clip | sam_zoom | vlm | learned (repeatable; default clip+sam_zoom+learned)')
    ap.add_argument('--quality', choices=sorted(QUALITY), default='high')
    ap.add_argument('--prompts', default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                      'autopilot_prompts.json'))
    ap.add_argument('--clip-model', help='CLIP judge weights, e.g. openai/clip-vit-large-patch14')
    ap.add_argument('--compare-models', action='store_true', help='also run each detector alone')
    ap.add_argument('--export-decisions', action='store_true', help='log tracks as learned-judge training data')
    ap.add_argument('--train-judge', action='store_true', help='train the learned judge after the run')
    ap.add_argument('--workdir', default=BENCH_DIR)
    ap.add_argument('--report', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports',
                                                     'autopilot_benchmark.md'))
    args = ap.parse_args()
    models = args.models or ['yolov8x-worldv2.pt']
    judges = args.judges if args.judges is not None else ['clip', 'sam_zoom', 'learned']

    missing = [missing_message(m) for m in models if not is_available(m)]
    if missing:
        print('Missing weights (nothing is downloaded):\n  ' + '\n  '.join(missing))
        models = [m for m in models if is_available(m)]
        if not models:
            return 1
    table = {}
    if os.path.exists(args.prompts):
        table = {k.lower(): v for k, v in json.load(open(args.prompts)).items() if not k.startswith('_')}

    pairs = [(v, os.path.splitext(v)[0] + '.txt') for v in args.video]
    if args.videos:
        for v in sorted(glob.glob(os.path.join(os.path.expanduser(args.videos), '*.mp4'))):
            if os.path.exists(os.path.splitext(v)[0] + '.txt'):
                pairs.append((v, os.path.splitext(v)[0] + '.txt'))
    if args.limit:
        pairs = pairs[:args.limit]
    if not pairs:
        print('No video + label pairs found')
        return 1

    all_rows, per_video, compare = [], [], {}
    for vi, (video, txt) in enumerate(pairs):
        names, gt = read_labels(txt)
        if not names or not gt:
            print(f'skip {video}: no labels')
            continue
        src = FrameSource(video_path=video)
        print(f'\n[{vi + 1}/{len(pairs)}] {os.path.basename(video)}  classes={names}')
        cfg = make_cfg(args, names, table, models, judges, src.count)

        def progress(stage, done, total):
            print(f'\r  {stage}: {done}/{total}      ', end='', flush=True)

        t0 = time.time()
        try:
            res = run_autopilot(src, cfg, progress=progress, workdir=args.workdir, log=lambda *_: None)
        except Exception as exc:
            print(f'\n  failed: {exc}')
            continue
        print(f'\r  done in {time.time() - t0:.0f}s: {res["stats"]["tracks"]} tracks        ')
        s, e = res['start_frame'], res['end_frame']
        label_tracks(res, gt, names)
        rows = []
        for t in res['tracks']:
            p = dict(t.get('judges') or {})
            c = combine({k: v for k, v in p.items() if k != 'learned'})
            if c is not None:
                p['combined'] = c
            rows.append({'p': p, 'real': t['_real'], 'n': len(t['boxes']), 'track': t})
            if args.export_decisions:
                log_decision(t, int(t['_real']), source='benchmark', video=os.path.basename(video))
        all_rows += rows
        raw = box_metrics(res, gt, lambda t: True, s, e)
        flagged_ok = box_metrics(res, gt, lambda t: t['route'] != 'review' or t['_real'], s, e)
        v = {'video': os.path.basename(video), 'frames': e - s + 1, 'tracks': len(res['tracks']),
             'real_tracks': sum(r['real'] for r in rows), 'raw': raw, 'after_review': flagged_ok,
             'flagged': sum(1 for t in res['tracks'] if t['route'] == 'review'),
             'judges': res['stats'].get('judges', {}), 'res': res, 'gt': gt}
        per_video.append(v)
        print(f"  raw P {raw['precision']:.3f} R {raw['recall']:.3f} | after reviewing {v['flagged']} flagged: "
              f"P {flagged_ok['precision']:.3f} R {flagged_ok['recall']:.3f}")
        if args.compare_models and len(models) > 1:
            for m in models:
                cfg_m = make_cfg(args, names, table, [m], [], src.count)
                r_m = run_autopilot(src, cfg_m, progress=progress, workdir=args.workdir, log=lambda *_: None)
                met = box_metrics(r_m, gt, lambda t: True, s, e)
                compare.setdefault(m, []).append(met)
                print(f"\r  {m} alone: P {met['precision']:.3f} R {met['recall']:.3f}            ")
            compare.setdefault('union', []).append(raw)

    jt = judge_table(all_rows)
    learned = LearnedJudge.load()
    if learned is not None and all_rows:
        lrows = [{'p': {'learned': learned.predict(r['track']['features'], r['track'].get('judges') or {})},
                  'real': r['real'], 'n': r['n']} for r in all_rows]
        jt.update({k: v for k, v in judge_table(lrows).items() if k == 'learned'})

    # simulate auto-rejection exactly as the pipeline does it (combined score <= configured threshold);
    # a threshold tuned on these same videos would flatter the result
    thr = AutopilotConfig().reject_below if (jt.get('combined') or {}).get('n') else None
    for v in per_video:
        if thr is not None:
            def keep(t, thr=thr):
                p = combine({k: x for k, x in (t.get('judges') or {}).items() if k != 'learned'})
                return p is None or p > thr
            v['auto_reject'] = box_metrics(v['res'], v['gt'], keep, v['res']['start_frame'], v['res']['end_frame'])

    train_msg = ''
    if args.train_judge:
        try:
            summary = train()
            train_msg = (f"Learned judge trained on {summary['rows']} decisions ({summary['real']} real): "
                         f"cross-validated AUC {summary['cv_auc']:.3f}. Strongest features: " +
                         ', '.join(f"{k} ({w:+.2f})" for k, w in summary['top_features']))
        except ValueError as exc:
            train_msg = f'Learned judge not trained: {exc}'
        print(train_msg)

    write_report(args, models, judges, per_video, jt, compare, thr, train_msg)
    print(f'\nReport: {args.report}')
    return 0


def write_report(args, models, judges, per_video, jt, compare, thr, train_msg):
    L = [f"# Autopilot benchmark ({time.strftime('%Y-%m-%d %H:%M')})", '',
         f"Detectors: {', '.join(models)} - quality `{args.quality}` - judges: {', '.join(judges) or 'none'} - "
         f"up to {args.frames or 'all'} frames per video.", '',
         'Ground truth is your hand labels. Objects your labeling policy skips (background cars, the turret '
         'commander...) count as false positives here, so precision is a lower bound.', '',
         '## Per video', '',
         '| Video | Frames | Tracks (real) | Raw P / R | Auto-reject P / R | After reviewing flagged P / R | Flagged |',
         '|---|---|---|---|---|---|---|']
    tot = {'raw': [0, 0, 0], 'auto_reject': [0, 0, 0], 'after_review': [0, 0, 0]}
    for v in per_video:
        cells = []
        for key in ('raw', 'auto_reject', 'after_review'):
            m = v.get(key)
            if m:
                cells.append(f"{m['precision']:.3f} / {m['recall']:.3f}")
                tot[key][0] += m['tp']
                tot[key][1] += m['fp']
                tot[key][2] += m['fn']
            else:
                cells.append('-')
        L.append(f"| {v['video'][:50]} | {v['frames']} | {v['tracks']} ({v['real_tracks']}) | {cells[0]} | "
                 f"{cells[1]} | {cells[2]} | {v['flagged']} |")
    tcells = []
    for key in ('raw', 'auto_reject', 'after_review'):
        tp, fp, fn = tot[key]
        tcells.append(f"**{tp / max(1, tp + fp):.3f} / {tp / max(1, tp + fn):.3f}**" if tp + fp + fn else '-')
    L.append(f"| **All** | | | {tcells[0]} | {tcells[1]} | {tcells[2]} | |")
    L += ['', f"Auto-reject column: tracks whose combined judge score is <= {thr} removed (the default "
          '`reject_below`, as the pipeline does it).' if thr is not None else 'No judge scores, so no auto-reject column.']
    if compare:
        L += ['', '## Detectors alone vs union (all videos)', '', '| Detectors | Precision | Recall |', '|---|---|---|']
        for k, mets in compare.items():
            tp, fp, fn = sum(m['tp'] for m in mets), sum(m['fp'] for m in mets), sum(m['fn'] for m in mets)
            L.append(f"| {k} | {tp / max(1, tp + fp):.3f} | {tp / max(1, tp + fn):.3f} |")
    L += ['', '## Judges', '',
          'AUC: chance a real track scores above a false one (0.5 = useless, 1.0 = perfect). For each reject '
          'threshold t: share of rejected boxes that were really false (precision), share of all false boxes '
          'removed, and real boxes lost.', '']
    for k, j in jt.items():
        L.append(f"### {k}: AUC {j['auc']:.3f} over {j['n']} tracks" if j['auc'] is not None else f"### {k}")
        L += ['', '| t | rejected tracks | precision | false removed | real boxes lost |', '|---|---|---|---|---|']
        for x in j['thresholds']:
            prec = f"{x['precision']:.3f}" if x['precision'] is not None else '-'
            rem = f"{x['false_removed']:.3f}" if x['false_removed'] is not None else '-'
            L.append(f"| {x['t']} | {x['rejected_tracks']} | {prec} | {rem} | {x['real_boxes_lost']} |")
        L += ['', f"Suggested reject threshold (>= 95% precision): {j['suggested']}", '']
    if per_video:
        loaded = per_video[-1]['judges']
        if loaded.get('missing'):
            L += ['## Missing pieces', ''] + [f"- {m}" for m in loaded['missing']] + ['']
        secs = {}
        for v in per_video:
            for k, s in (v['judges'].get('seconds') or {}).items():
                secs.setdefault(k, []).append(s)
        if secs:
            L += ['## Judge time', ''] + [f"- {k}: {sum(s):.0f} s total" for k, s in secs.items()] + ['']
    if train_msg:
        L += ['## Learned judge', '', train_msg, '']
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(L) + '\n')


if __name__ == '__main__':
    sys.exit(main())
