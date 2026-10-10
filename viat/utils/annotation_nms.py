"""
Overlap cleanup for annotations: duplicate removal (NMS) and ignore regions.

- Duplicate removal: removes boxes left behind when auto-label, SAM tracking/detection, or a
  propagate step re-annotates an object that was already labelled. For every group of overlapping
  boxes it keeps the one that was there first -- manually drawn / verified annotations win over
  AI-generated ones, and otherwise the older annotation (earlier in the frame's list) wins.
- Ignore regions: boxes of chosen classes (e.g. 'shadow', or hand-drawn 'ignore' areas) remove
  every other box that lies mostly inside them. This is measured as the fraction of the *other*
  box's area covered by the region, not IoU, so a small box inside a large region is caught too.

The same rules are used by Tools > Remove Duplicate Annotations and by auto-label's
"Ignore Region (Suppress)" class action.
"""


def rect_to_xyxy(rect):
    """QRect-like (x, y, width, height) -> [x1, y1, x2, y2]."""
    return [rect.x(), rect.y(), rect.x() + rect.width(), rect.y() + rect.height()]


def _area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def _intersection(box_a, box_b):
    return (max(0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]))
            * max(0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1])))


def box_iou(box_a, box_b):
    """IoU of two [x1, y1, x2, y2] boxes."""
    inter = _intersection(box_a, box_b)
    union = _area(box_a) + _area(box_b) - inter
    return inter / union if union > 0 else 0.0


def fraction_inside(box, region):
    """Fraction of `box`'s area covered by `region` (both [x1, y1, x2, y2])."""
    area = _area(box)
    return _intersection(box, region) / area if area > 0 else 0.0


def inside_any_region(box, regions, min_fraction):
    return any(fraction_inside(box, r) >= min_fraction for r in regions)


def _is_protected(ann):
    """Manually drawn or user-verified annotations are always preferred over AI output."""
    return getattr(ann, 'verified', False) or getattr(ann, 'original_source', getattr(ann, 'source', None)) == 'manual'


def fraction_of_smaller_inside(box_a, box_b):
    """Fraction of the smaller box's area that overlaps the other box (1.0 = fully nested)."""
    smaller = min(_area(box_a), _area(box_b))
    return _intersection(box_a, box_b) / smaller if smaller > 0 else 0.0


KEEP_LARGER, KEEP_OLDER = 'larger', 'older'


def clean_annotations(annotations, remove_duplicates=True, iou_threshold=0.5, class_agnostic=False,
                      nested_overlap=0.8, keep=KEEP_LARGER,
                      region_classes=(), region_overlap=0.6, remove_regions=False):
    """
    Returns (kept, removed) lists for one frame's annotations, preserving their original order.

    Ignore regions are applied first: any box whose class is not in `region_classes` and that is at
    least `region_overlap` inside a region box is removed (region boxes are removed too when
    `remove_regions`).

    Duplicate removal then runs on what's left. Two boxes are duplicates when their IoU is at least
    `iou_threshold`, or -- for boxes nested inside each other, whose IoU is low -- when at least
    `nested_overlap` of the smaller box lies inside the other (None disables the nested test).
    With `class_agnostic=False` only same-class boxes compete. The survivor is chosen with manual /
    verified boxes first, then the larger box (`keep='larger'`) or the older box (`keep='older'`).
    """
    boxes = [rect_to_xyxy(a.rect) for a in annotations]
    region_classes = set(region_classes)
    regions = [boxes[i] for i, a in enumerate(annotations) if a.class_name in region_classes]
    removed_idx = set()
    for i, ann in enumerate(annotations):
        if ann.class_name in region_classes:
            if remove_regions:
                removed_idx.add(i)
        elif regions and inside_any_region(boxes[i], regions, region_overlap):
            removed_idx.add(i)

    if remove_duplicates:
        candidates = [i for i in range(len(annotations)) if i not in removed_idx]
        if keep == KEEP_LARGER:
            rank = lambda i: (not _is_protected(annotations[i]), -_area(boxes[i]), i)
        else:
            rank = lambda i: (not _is_protected(annotations[i]), i)
        kept_idx = []
        for i in sorted(candidates, key=rank):
            if any((class_agnostic or annotations[k].class_name == annotations[i].class_name)
                   and (box_iou(boxes[k], boxes[i]) >= iou_threshold
                        or (nested_overlap is not None
                            and fraction_of_smaller_inside(boxes[k], boxes[i]) >= nested_overlap))
                   for k in kept_idx):
                removed_idx.add(i)
            else:
                kept_idx.append(i)

    kept = [a for i, a in enumerate(annotations) if i not in removed_idx]
    removed = [annotations[i] for i in sorted(removed_idx)]
    return kept, removed


def plan_cleanup(frame_annotations, frames, should_stop=None, on_progress=None, **options):
    """
    Computes, without modifying anything, {frame: kept_annotations} for every frame in `frames`
    where clean_annotations(**options) would remove at least one box. Safe to run off the UI
    thread while the annotations aren't being edited. Returns None if `should_stop()` became true.
    """
    plan = {}
    frames = list(frames)
    for n, f in enumerate(frames):
        if should_stop is not None and should_stop():
            return None
        anns = frame_annotations.get(f)
        if anns and len(anns) > 1 or (anns and options.get('remove_regions')):
            kept, removed = clean_annotations(list(anns), **options)
            if removed:
                plan[f] = kept
        if on_progress is not None and n % 200 == 0:
            on_progress(n, len(frames))
    return plan


def plan_removed_count(frame_annotations, plan):
    return sum(len(frame_annotations[f]) - len(kept) for f, kept in plan.items())


def apply_plan(frame_annotations, plan):
    """Applies a plan from plan_cleanup in place. Returns the number of annotations removed."""
    removed = 0
    for f, kept in plan.items():
        anns = frame_annotations[f]
        removed += len(anns) - len(kept)
        # Mutate the existing list: the canvas holds a reference to the current frame's list
        anns[:] = kept
    return removed
