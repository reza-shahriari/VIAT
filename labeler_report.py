#!/usr/bin/env python3
"""
labeler_report.py — standalone payment-report CLI for VIAT project JSON files.

Not wired into the VIAT UI at all -- run it directly from the command line
against a folder of saved VIAT project (.json) files:

    python labeler_report.py /path/to/folder
    python labeler_report.py /path/to/folder --recursive --iou-threshold 0.3
    python labeler_report.py /path/to/folder --csv report.csv

For each project file it reconstructs, purely from the geometry of the saved
bounding boxes (never from the 'source'/'original_source' metadata, and
regardless of which tool actually produced them), how many distinct SAM
"prompts" were placed and how many additional frames each prompt was tracked
across, plus a straight count of blur regions and deleted frames. These are
the four raw numbers the payment unit (tracked box + prompt + blur) is built
from; this script deliberately does not decide the price, just the counts.

Tracklet reconstruction:
    Frame-by-frame boxes of the same class are linked across consecutive
    frame numbers using IoU, greedily assigning the highest-IoU pairs first
    (same idea a simple SORT-style tracker uses). A gap between two
    annotated frames breaks the chain UNLESS every frame in the gap is
    marked deleted (deleting a frame is an export choice, not evidence
    tracking stopped there -- see --max-gap to also bridge short gaps that
    aren't from deleted frames, e.g. a brief missed detection). Each
    finished chain of length L counts as 1 prompt + (L - 1) tracked frames
    -- a box that never continues into another frame is still 1 prompt with
    0 tracked frames, so every box in the file is accounted for in exactly
    one of prompt_count / tracked_count. This is a heuristic estimate, not
    a substitute for logging real SAM prompt events -- it will over- or
    under-count when two separate real prompts happen to sit right next to
    each other with high IoU, or when a genuinely continuous track has an
    undeleted gap wider than --max-gap.
"""

import argparse
import csv
import json
import os
import sys


def iou(rect_a, rect_b):
    """IoU of two {'x','y','width','height'} rects."""
    ax1, ay1 = rect_a['x'], rect_a['y']
    ax2, ay2 = ax1 + rect_a['width'], ay1 + rect_a['height']
    bx1, by1 = rect_b['x'], rect_b['y']
    bx2, by2 = bx1 + rect_b['width'], by1 + rect_b['height']

    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_w, inter_h = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, rect_a['width']) * max(0.0, rect_a['height'])
    area_b = max(0.0, rect_b['width']) * max(0.0, rect_b['height'])
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def link_items_across_frames(items_by_frame, iou_threshold, group_key_fn, rect_fn,
                              deleted_frames=None, max_gap=0):
    """Generic frame-to-frame IoU linker, shared by box-tracklet
    reconstruction and blur-region tracked/manual inference.

    items_by_frame: {frame_num_or_str: [raw_item, ...]}. group_key_fn(item)
    partitions items that are allowed to link to each other (class_name for
    boxes; blur shape 'type' for blur regions -- a real continuation keeps
    the same shape). rect_fn(item) -> {'x','y','width','height'}.

    Returns a list of chains; each chain is the list of raw_item objects
    (same references as in items_by_frame) linked together in frame order.
    A lone, never-continued item is still a chain of length 1.

    A gap between two populated frames only breaks a chain if it can't be
    explained away: a gap where every skipped frame is in *deleted_frames*
    is bridged for free (deleting a frame is an export choice, not evidence
    tracking/blurring stopped), and a gap of up to *max_gap* frames of
    anything else (e.g. a brief missed detection) is bridged too if given."""
    deleted_frames = deleted_frames or set()
    frame_nums = sorted(int(k) for k in items_by_frame)

    # active[group_key] = list of {'rect', 'chain': [items...], 'last_frame'}
    active = {}
    finished_chains = []

    prev_frame_num = None
    for f in frame_nums:
        items = items_by_frame[str(f)] if str(f) in items_by_frame else items_by_frame[f]

        if prev_frame_num is not None and f != prev_frame_num + 1:
            gap_frames = set(range(prev_frame_num + 1, f))
            bridge = gap_frames.issubset(deleted_frames) or len(gap_frames) <= max_gap
            if not bridge:
                for group_chains in active.values():
                    for t in group_chains:
                        finished_chains.append(t['chain'])
                active = {}

        items_by_group = {}
        for item in items:
            rect = rect_fn(item)
            if not rect:
                continue
            key = group_key_fn(item)
            items_by_group.setdefault(key, []).append(item)

        new_active = {}
        all_groups = set(items_by_group.keys()) | set(active.keys())
        for key in all_groups:
            open_chains = active.get(key, [])
            cur_items = items_by_group.get(key, [])

            # Build every (iou, chain_idx, item_idx) candidate pair above
            # threshold, then greedily assign highest-IoU pairs first.
            candidates = []
            for ti, t in enumerate(open_chains):
                for bi, item in enumerate(cur_items):
                    score = iou(t['rect'], rect_fn(item))
                    if score >= iou_threshold:
                        candidates.append((score, ti, bi))
            candidates.sort(key=lambda c: c[0], reverse=True)

            matched_chains = set()
            matched_items = set()
            extended = []
            for score, ti, bi in candidates:
                if ti in matched_chains or bi in matched_items:
                    continue
                matched_chains.add(ti)
                matched_items.add(bi)
                t = open_chains[ti]
                extended.append({
                    'rect': rect_fn(cur_items[bi]),
                    'chain': t['chain'] + [cur_items[bi]],
                    'last_frame': f,
                })

            # Chains that didn't get extended this frame are finished.
            for ti, t in enumerate(open_chains):
                if ti not in matched_chains:
                    finished_chains.append(t['chain'])

            # Unmatched items in this frame start new chains.
            for bi, item in enumerate(cur_items):
                if bi not in matched_items:
                    extended.append({'rect': rect_fn(item), 'chain': [item], 'last_frame': f})

            if extended:
                new_active[key] = extended

        active = new_active
        prev_frame_num = f

    for group_chains in active.values():
        for t in group_chains:
            finished_chains.append(t['chain'])

    return finished_chains


def reconstruct_tracklets(frame_annotations, iou_threshold, deleted_frames=None, max_gap=0):
    """Link boxes across consecutive frames by class + IoU. Returns a list
    of tracklet lengths (number of frames each reconstructed chain spans)."""
    chains = link_items_across_frames(
        frame_annotations, iou_threshold,
        group_key_fn=lambda ann: ann.get('class_name', 'unknown'),
        rect_fn=lambda ann: ann.get('rect'),
        deleted_frames=deleted_frames, max_gap=max_gap,
    )
    return [len(c) for c in chains]


def _blur_rect(region):
    return {'x': region.get('x', 0), 'y': region.get('y', 0),
            'width': region.get('w', 0), 'height': region.get('h', 0)}


def infer_blur_origins(blur_regions, iou_threshold, deleted_frames=None, max_gap=0):
    """For blur regions that don't already carry a real 'origin' tag (saved
    by a VIAT build from before that field existed), guess whether each was
    produced by tracking or a one-off manual/converted action, using the
    same frame-to-frame IoU linking as box tracklets: a blur region that
    persists across multiple consecutive frames is what SAM-tracking auto-
    blur leaves behind; a blur region that never continues into another
    frame is a one-off action. Grouped by shape ('type') since a real
    continuation keeps the same shape.

    NOTE: this can't tell tracking apart from the "blur this annotation
    across a frame range" bulk action (main.py's duplicate-rect-per-frame
    blur), which also produces a multi-frame chain of identical rects --
    geometry alone can't distinguish the two. Treat 'inferred_tracking' as
    "multi-frame, tracking-shaped", not a certainty.

    Returns counts: {'tracking': n, 'manual': n} for the *inferred* portion
    only (items that already had a real origin are left untouched)."""
    deleted_frames = deleted_frames or set()
    to_infer = {}
    for frame_key, regions in blur_regions.items():
        needs_guess = [r for r in regions if r.get('origin', 'unknown') == 'unknown']
        if needs_guess:
            to_infer[frame_key] = needs_guess

    if not to_infer:
        return {'tracking': 0, 'manual': 0}

    chains = link_items_across_frames(
        to_infer, iou_threshold,
        group_key_fn=lambda r: r.get('type', 'unknown'),
        rect_fn=_blur_rect,
        deleted_frames=deleted_frames, max_gap=max_gap,
    )

    counts = {'tracking': 0, 'manual': 0}
    for chain in chains:
        bucket = 'tracking' if len(chain) > 1 else 'manual'
        counts[bucket] += len(chain)
    return counts


def analyze_file(path, iou_threshold, max_gap=0):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'frame_annotations' not in data and 'blur_regions' not in data and 'deleted_frames' not in data:
        return None  # doesn't look like a VIAT project file

    deleted_frames = set(data.get('deleted_frames') or [])
    frame_annotations = data.get('frame_annotations') or {}
    tracklet_lengths = reconstruct_tracklets(frame_annotations, iou_threshold, deleted_frames, max_gap)

    prompt_count = len(tracklet_lengths)
    tracked_count = sum(max(0, L - 1) for L in tracklet_lengths)
    total_boxes = sum(tracklet_lengths)

    blur_regions = data.get('blur_regions') or {}
    blur_count = 0
    blur_by_origin = {'tracking': 0, 'manual': 0, 'converted_from_box': 0, 'unknown': 0}
    for regions in blur_regions.values():
        for region in regions:
            blur_count += 1
            origin = region.get('origin', 'unknown')
            blur_by_origin[origin] = blur_by_origin.get(origin, 0) + 1

    # Files saved before blur regions recorded their own origin have no way
    # to know for certain -- infer it from geometry instead, same idea as
    # the box tracklets above.
    inferred = infer_blur_origins(blur_regions, iou_threshold, deleted_frames, max_gap)

    deleted_count = len(deleted_frames)

    avg_boxes_per_frame = (total_boxes / len(frame_annotations)) if frame_annotations else 0
    # A file where most boxes never link into any tracklet (prompt_count is
    # a large fraction of total_boxes) doesn't look like SAM-prompt-and-track
    # work at all -- more likely a dense, independently-labeled-per-frame
    # dataset (e.g. a VisDrone-style crowd scene) where IoU linking is
    # inherently ambiguous. Flag it instead of quietly polluting the totals.
    suspicious = bool(total_boxes) and (prompt_count / total_boxes > 0.5) and avg_boxes_per_frame > 15

    return {
        'file': os.path.basename(path),
        'path': path,
        'prompt_count': prompt_count,
        'tracked_count': tracked_count,
        'total_boxes': total_boxes,
        'blur_count': blur_count,
        'blur_tracking': blur_by_origin['tracking'],
        'blur_manual': blur_by_origin['manual'],
        'blur_converted': blur_by_origin['converted_from_box'],
        'blur_unknown': blur_by_origin['unknown'],
        'blur_inferred_tracking': inferred['tracking'],
        'blur_inferred_manual': inferred['manual'],
        'deleted_count': deleted_count,
        'payable_units': prompt_count + tracked_count + blur_count,
        'avg_boxes_per_frame': avg_boxes_per_frame,
        'suspicious': suspicious,
    }


def find_project_files(folder, recursive):
    if recursive:
        for root, _, files in os.walk(folder):
            for fname in files:
                if fname.lower().endswith('.json'):
                    yield os.path.join(root, fname)
    else:
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith('.json'):
                yield os.path.join(folder, fname)


def print_report(results):
    header = f"{'File':<45} {'Prompts':>8} {'Tracked':>8} {'Boxes':>8} {'Blur':>6} {'Deleted':>8} {'Payable':>8}"
    print(header)
    print('-' * len(header))
    totals = {'prompt_count': 0, 'tracked_count': 0, 'total_boxes': 0,
              'blur_count': 0, 'blur_tracking': 0, 'blur_manual': 0,
              'blur_converted': 0, 'blur_unknown': 0, 'blur_inferred_tracking': 0,
              'blur_inferred_manual': 0, 'deleted_count': 0, 'payable_units': 0}
    for r in results:
        name = r['file'] if len(r['file']) <= 45 else r['file'][:42] + '...'
        flag = '  [!] dense per-frame data, not tracked -- see below' if r['suspicious'] else ''
        print(f"{name:<45} {r['prompt_count']:>8} {r['tracked_count']:>8} "
              f"{r['total_boxes']:>8} {r['blur_count']:>6} {r['deleted_count']:>8} {r['payable_units']:>8}{flag}")
        for k in totals:
            totals[k] += r[k]

    print('-' * len(header))
    print(f"{'ALL (' + str(len(results)) + ' file(s))':<45} {totals['prompt_count']:>8} "
          f"{totals['tracked_count']:>8} {totals['total_boxes']:>8} {totals['blur_count']:>6} "
          f"{totals['deleted_count']:>8} {totals['payable_units']:>8}")

    print(f"\nBlur breakdown (ALL) -- recorded at save time: tracking={totals['blur_tracking']}  "
          f"manual={totals['blur_manual']}  converted_from_box={totals['blur_converted']}  "
          f"unknown={totals['blur_unknown']}")
    if totals['blur_unknown']:
        combined_tracking = totals['blur_tracking'] + totals['blur_inferred_tracking']
        combined_manual = totals['blur_manual'] + totals['blur_converted'] + totals['blur_inferred_manual']
        print(f"  {totals['blur_unknown']} of those are 'unknown' (saved by a VIAT build from before blur "
              "regions recorded their origin) -- inferred from IoU-linking across frames instead "
              f"(a chain spanning multiple frames ~= tracked, a one-off region ~= manual/converted):")
        print(f"  inferred: tracking-like={totals['blur_inferred_tracking']}  "
              f"manual/one-off-like={totals['blur_inferred_manual']}")
        print(f"  combined best-effort total: tracking={combined_tracking}  manual-or-converted={combined_manual}")
        print("  (can't be exact: a bulk 'blur this box across frames N-M' action also produces a multi-frame "
              "chain and looks identical to tracking geometrically.)")

        clean = [r for r in results if not r['suspicious']]
        if clean and len(clean) != len(results):
            c_track = sum(r['blur_tracking'] + r['blur_inferred_tracking'] for r in clean)
            c_manual = sum(r['blur_manual'] + r['blur_converted'] + r['blur_inferred_manual'] for r in clean)
            print(f"  excluding the {len(results) - len(clean)} flagged dense-data file(s) above "
                  f"(where blur linking is just as unreliable as prompt linking): "
                  f"tracking={c_track}  manual-or-converted={c_manual}")

    suspicious = [r for r in results if r['suspicious']]
    if suspicious:
        print(f"\n[!] {len(suspicious)} file(s) look like dense, independently-labeled-per-frame data "
              "(most boxes never link into any tracklet) rather than SAM-prompt-and-track work. "
              "Their prompt_count is likely meaningless -- review manually before paying against it:")
        for r in suspicious:
            print(f"    {r['file']}  (avg {r['avg_boxes_per_frame']:.0f} boxes/frame, "
                  f"{r['prompt_count']}/{r['total_boxes']} boxes never linked)")


def write_csv(results, csv_path):
    fieldnames = ['file', 'prompt_count', 'tracked_count', 'total_boxes',
                  'blur_count', 'blur_tracking', 'blur_manual', 'blur_converted',
                  'blur_unknown', 'blur_inferred_tracking', 'blur_inferred_manual',
                  'deleted_count', 'payable_units']
    totals = {k: 0 for k in fieldnames if k != 'file'}
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})
            for k in totals:
                totals[k] += r[k]
        writer.writerow({'file': 'ALL', **totals})
    print(f"\nWrote CSV report to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('folder', help='Folder containing VIAT project .json files')
    parser.add_argument('--recursive', action='store_true', help='Also scan subfolders')
    parser.add_argument('--iou-threshold', type=float, default=0.3,
                         help='Minimum IoU to link a box into the previous frame\'s tracklet (default: 0.3)')
    parser.add_argument('--max-gap', type=int, default=0,
                         help='Bridge tracklet gaps of up to this many non-deleted frames '
                              '(e.g. a brief missed detection) without counting a new prompt. '
                              'Gaps fully covered by deleted_frames are always bridged regardless. (default: 0)')
    parser.add_argument('--csv', metavar='PATH', help='Also write a CSV report to this path')
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"Not a folder: {args.folder}", file=sys.stderr)
        sys.exit(1)

    results = []
    skipped = []
    for path in find_project_files(args.folder, args.recursive):
        try:
            r = analyze_file(path, args.iou_threshold, args.max_gap)
        except Exception as e:
            skipped.append((path, str(e)))
            continue
        if r is None:
            skipped.append((path, 'not a VIAT project file'))
            continue
        results.append(r)

    if not results:
        print("No VIAT project JSON files found.", file=sys.stderr)
        sys.exit(1)

    results.sort(key=lambda r: r['file'])
    print_report(results)

    if args.csv:
        write_csv(results, args.csv)

    if skipped:
        print(f"\nSkipped {len(skipped)} file(s):")
        for path, reason in skipped:
            print(f"  {path}: {reason}")


if __name__ == '__main__':
    main()
