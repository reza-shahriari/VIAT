"""
Video Dataset to YOLO Dataset Converter Engine.

Converts a folder of videos + per-frame annotation files (.txt in Raya format)
into a YOLO-formatted image dataset (images/ + labels/ + data.yaml).

Architecture
------------
Conversion is split into two phases:

  Phase A - build_plan(): parses every annotation file and (cheaply, via video
    metadata only - no pixel decode) checks each video's actual frame count
    against its annotation file, resolves local class names to global YOLO
    classes with a per-class "fate" (map / skip / soft-delete / purge),
    computes class-balancing targets, and decides - purely from that - which
    frame indices will be extracted from each video. This is fast enough to
    call repeatedly as a live preview/dry-run while the user edits settings.

  Phase B - execute_plan(): decodes only the frames the plan asked for,
    applies padding removal, near-duplicate detection, smart cropping and
    augmentation, and writes images/labels/data.yaml to disk.

convert_video_dataset_to_yolo() is a thin backward-compatible wrapper that
runs both phases back to back - this is what the CLI script and the simplest
callers use. scan_dataset_statistics() runs Phase A with no balancing/fate
config applied, for populating the "what classes exist" UI before the user
has made any choices.

Features
--------
- Per-class fates: map to a (possibly renamed) global class, skip (drop the
  boxes, keep the frame), soft-delete (drop the boxes, drop the frame if
  nothing else survives), or purge (drop the whole frame).
- Class balancing: manual per-class instance caps, or "auto: balance to the
  rarest class" - both realized as *deterministic, evenly-spaced* frame
  selection (not independent random sampling) so kept frames of an
  over-represented class stay spread across the whole video instead of
  clustering, and a configurable minimum frame gap keeps near-adjacent,
  near-duplicate frames from both being kept. A frame is only dropped for
  balancing if *every* class it contains has already hit its target -
  a scarce class sharing a frame with an over-represented one is protected.
- Per-video output cap, applied as the same evenly-spaced selection.
- Annotation/video frame-count mismatch reporting.
- Near-duplicate frame detection (perceptual dHash) during extraction.
- Letterbox / pillarbox padding auto-detection & removal.
- Background frame thinning (randomly dropping empty [] frames).
- Smart object-focused multi-cropping: crop size adapts to the object
  cluster it frames (instead of a fixed window), oversized clusters split
  instead of clipping objects at the edge, clusters are prioritized by
  object smallness/class rarity when capped, overlapping crops are
  deduplicated, and background frames get the same crop treatment instead
  of being saved full-resolution.
- Horizontal Flip Augmentation with inverted YOLO coordinates.
- Flexible Export Modes (Single folder, Preserve Split Hierarchy, Auto-Split)
- Standalone CLI & GUI Progress Generator
"""

import os
import re
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    import yaml
except ImportError:
    yaml = None

# Regex patterns
NC_LINE_RE = re.compile(r"^-?\s*nc\s*:\s*\d+", re.IGNORECASE)
BULLET_RE = re.compile(r"^[\-\*•]\s*")
BOX_GROUP_RE = re.compile(r"\[([^\[\]]*)\]")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v", ".wmv"}

# Class-fate sentinels (values a resolved local class name can carry besides
# a global class index)
SKIP = "SKIP"
SOFT_DELETE = "SOFT_DELETE"
PURGE = "PURGE"


# --------------------------------------------------------------------------- #
# Annotation Parsing
# --------------------------------------------------------------------------- #

def parse_annotation_file(path: Path) -> Tuple[List[str], List[str]]:
    """
    Parse a Raya format per-frame annotation file.

    Returns:
        local_class_names: List of class names defined in the header
        frame_lines: List of non-header lines corresponding sequentially to video frames
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    raw_lines = text.splitlines()

    names = []
    header_end = None
    in_names_block = False
    in_hash_header = False

    for i, raw in enumerate(raw_lines):
        line = raw.strip()
        if line == "###":
            in_hash_header = not in_hash_header
            if not in_hash_header:
                header_end = i + 1
                break
            continue

        if in_hash_header:
            if line.lower().startswith("names:"):
                in_names_block = True
                continue
            if NC_LINE_RE.match(line):
                continue
            if in_names_block and (line.startswith("- ") or line.startswith("* ")):
                name = BULLET_RE.sub("", line).strip()
                if name:
                    names.append(name)
            continue

        if not in_names_block:
            if line.lower().startswith("names:"):
                in_names_block = True
            continue
        if NC_LINE_RE.match(line):
            header_end = i + 1
            break
        if line == "":
            continue
        name = BULLET_RE.sub("", line).strip()
        if name:
            names.append(name)

    if header_end is None:
        header_end = 0

    while header_end < len(raw_lines):
        candidate = raw_lines[header_end].strip()
        if (
            candidate == ""
            or candidate.upper() in ("DELETED;", "DELETE;", "DELETED", "DELETE")
            or BOX_GROUP_RE.search(candidate)
            or candidate == "[]"
        ):
            break
        header_end += 1

    frame_lines = raw_lines[header_end:]
    return names, frame_lines


def parse_frame_line(line: str):
    """
    Parse a single frame's annotation line.

    Returns:
        'DELETED;'              -> frame must be dropped entirely
        []                      -> empty/background frame
        [(cls, x, y, w, h), ...] -> list of boxes (pixel, top-left x,y, width, height)
        None                    -> unparseable / blank
    """
    s = line.strip()
    if s == "":
        return None
    cleaned_upper = s.replace(";", "").strip().upper()
    if cleaned_upper in ("DELETED", "DELETE"):
        return "DELETED;"
    if s.replace(";", "").strip() == "[]":
        return []

    groups = BOX_GROUP_RE.findall(s)
    if not groups:
        return None

    boxes = []
    for g in groups:
        g = g.strip()
        if g == "":
            continue
        parts = [p.strip() for p in g.split(",")]
        if len(parts) < 5:
            continue
        try:
            cls = int(float(parts[0]))
            x, y, w, h = (float(v) for v in parts[1:5])
        except ValueError:
            continue
        boxes.append((cls, x, y, w, h))
    return boxes


# --------------------------------------------------------------------------- #
# Video discovery
# --------------------------------------------------------------------------- #

def discover_video_pairs(source_dir: Path) -> List[Tuple[Path, Path]]:
    """Walk source_dir for video files with a same-stem .txt annotation file."""
    video_pairs = []
    for root_p, _, files in os.walk(source_dir):
        for f in sorted(files):
            ext = os.path.splitext(f)[1].lower()
            if ext in VIDEO_EXTENSIONS and not f.lower().startswith("outvideo"):
                v_path = Path(root_p) / f
                txt_path = v_path.with_suffix(".txt")
                if txt_path.is_file():
                    video_pairs.append((v_path, txt_path))
    return video_pairs


# --------------------------------------------------------------------------- #
# Near-duplicate detection (perceptual hash)
# --------------------------------------------------------------------------- #

def compute_dhash(image: Any, hash_size: int = 8) -> Optional[int]:
    """Cheap difference-hash of an image for near-duplicate detection."""
    if cv2 is None or np is None or image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    diff = resized[:, 1:] > resized[:, :-1]
    h = 0
    for bit in diff.flatten():
        h = (h << 1) | int(bit)
    return h


def hamming_distance(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# --------------------------------------------------------------------------- #
# Padding Detection
# --------------------------------------------------------------------------- #

def detect_content_bbox(frame: Any, black_thresh: int = 16) -> Tuple[int, int, int, int]:
    """
    Detect bounding box of non-padding (non-black) content in a frame.

    Returns (left, top, right, bottom) in pixel coords (exclusive right/bottom).
    """
    if frame is None or np is None or cv2 is None:
        return 0, 0, frame.shape[1] if frame is not None else 0, frame.shape[0] if frame is not None else 0

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    h, w = gray.shape

    row_max = gray.max(axis=1)
    col_max = gray.max(axis=0)

    rows = np.where(row_max > black_thresh)[0]
    cols = np.where(col_max > black_thresh)[0]

    if rows.size == 0 or cols.size == 0:
        return 0, 0, w, h

    top, bottom = int(rows[0]), int(rows[-1]) + 1
    left, right = int(cols[0]), int(cols[-1]) + 1

    if right - left < 4 or bottom - top < 4:
        return 0, 0, w, h

    return left, top, right, bottom


# --------------------------------------------------------------------------- #
# Smart Object-Focused Cropping
# --------------------------------------------------------------------------- #

def cluster_bounding_boxes(boxes: List[Tuple[int, float, float, float, float]], distance_threshold: float = 100.0) -> List[List[Tuple[int, float, float, float, float]]]:
    """Group bounding boxes that are close to each other into clusters."""
    if not boxes:
        return []
    if len(boxes) == 1:
        return [boxes]

    clusters = []
    visited = [False] * len(boxes)

    for i in range(len(boxes)):
        if visited[i]:
            continue
        cluster = [boxes[i]]
        visited[i] = True

        to_check = [boxes[i]]
        while to_check:
            curr = to_check.pop(0)
            c_cls, c_x, c_y, c_w, c_h = curr
            c_cx, c_cy = c_x + c_w / 2.0, c_y + c_h / 2.0

            for j in range(len(boxes)):
                if not visited[j]:
                    o_cls, o_x, o_y, o_w, o_h = boxes[j]
                    o_cx, o_cy = o_x + o_w / 2.0, o_y + o_h / 2.0
                    dist = math.hypot(c_cx - o_cx, c_cy - o_cy)
                    if dist < distance_threshold + max(c_w, c_h) / 2.0 + max(o_w, o_h) / 2.0:
                        visited[j] = True
                        cluster.append(boxes[j])
                        to_check.append(boxes[j])

        clusters.append(cluster)
    return clusters


def _cluster_bbox(cluster: List[Tuple[int, float, float, float, float]]) -> Tuple[float, float, float, float]:
    x1 = min(b[1] for b in cluster)
    y1 = min(b[2] for b in cluster)
    x2 = max(b[1] + b[3] for b in cluster)
    y2 = max(b[2] + b[4] for b in cluster)
    return x1, y1, x2, y2


def _split_oversized_cluster(cluster, max_w: float, max_h: float, depth: int = 0):
    """
    Recursively bisect a cluster along its longer axis until every sub-cluster's
    bounding box fits inside (max_w, max_h) - so a crop window never has to
    silently clip objects at its edge because the cluster it was built around
    didn't actually fit.
    """
    if len(cluster) <= 1 or depth > 4:
        return [cluster]
    x1, y1, x2, y2 = _cluster_bbox(cluster)
    if (x2 - x1) <= max_w and (y2 - y1) <= max_h:
        return [cluster]

    axis_is_x = (x2 - x1) >= (y2 - y1)
    centers = [(b[1] + b[3] / 2.0) if axis_is_x else (b[2] + b[4] / 2.0) for b in cluster]
    order = sorted(range(len(cluster)), key=lambda i: centers[i])
    mid = len(order) // 2
    left = [cluster[i] for i in order[:mid]]
    right = [cluster[i] for i in order[mid:]]
    if not left or not right:
        return [cluster]

    result = []
    for sub in (left, right):
        result.extend(_split_oversized_cluster(sub, max_w, max_h, depth + 1))
    return result


def _score_cluster(cluster, class_priority: Optional[Dict[int, float]] = None) -> float:
    """
    Priority score used when there are more clusters than max_crops - higher
    wins. Favors clusters of small (distant) objects and of rarer classes,
    since those are the ones the smart-crop feature exists to help.
    """
    areas = [b[3] * b[4] for b in cluster]
    avg_area = sum(areas) / len(areas) if areas else 1.0
    size_score = 1.0 / (1.0 + avg_area)
    if class_priority:
        prio = sum(class_priority.get(b[0], 1.0) for b in cluster) / len(cluster)
    else:
        prio = 1.0
    return size_score * prio


def _iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


def generate_background_crops(
    frame: Any, crop_w: int, crop_h: int, max_crops: int, rng: random.Random, square_crops: bool = False
) -> List[Tuple[Any, List[str]]]:
    """
    Tile an oversized background (no-box) frame into up to max_crops
    non-overlapping crop-sized windows, instead of saving the whole
    full-resolution frame untouched.
    """
    img_h, img_w = frame.shape[:2]
    if square_crops:
        side = min(crop_w, crop_h, img_w, img_h)
        target_w, target_h = side, side
    else:
        target_w, target_h = min(crop_w, img_w), min(crop_h, img_h)
    max_x, max_y = max(0, img_w - target_w), max(0, img_h - target_h)

    n = max(1, max_crops)
    windows = []
    attempts = 0
    while len(windows) < n and attempts < n * 10:
        attempts += 1
        x1 = rng.randint(0, max_x) if max_x > 0 else 0
        y1 = rng.randint(0, max_y) if max_y > 0 else 0
        window = (x1, y1, x1 + target_w, y1 + target_h)
        if any(_iou(window, w) > 0.3 for w in windows):
            continue
        windows.append(window)

    return [(frame[y1:y2, x1:x2], []) for (x1, y1, x2, y2) in windows]


def _composition_score(
    window: Tuple[int, int, int, int],
    other_boxes: List[Tuple[int, float, float, float, float]],
    min_visibility: float,
) -> float:
    """
    How "clean" a candidate crop window's composition is with respect to
    boxes other than the one it's built around: +1 for each other box that
    ends up either meaningfully included (>= min_visibility) or cleanly
    excluded (~0 visible), 0 for one left dangling half-cut in between -
    visible in the frame but too truncated to be kept as a label, which
    just adds a confusing unlabeled distractor to the crop.
    """
    wx1, wy1, wx2, wy2 = window
    score = 0.0
    for (_g_idx, bx, by, bw, bh) in other_boxes:
        area = bw * bh
        if area <= 0:
            continue
        ix1, iy1 = max(wx1, bx), max(wy1, by)
        ix2, iy2 = min(wx2, bx + bw), min(wy2, by + bh)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        vis = (iw * ih) / area
        if vis <= 0.02 or vis >= min_visibility:
            score += 1.0
    return score


def _pick_crop_position(
    cl_x1: float, cl_y1: float, cl_x2: float, cl_y2: float,
    target_w: int, target_h: int, img_w: int, img_h: int,
    other_boxes: List[Tuple[int, float, float, float, float]],
    min_visibility: float,
    wide_position: bool,
    context_padding: float,
    rng: random.Random,
) -> Tuple[Optional[Tuple[int, int]], bool]:
    """
    Pick a crop's top-left corner. When `wide_position` is on, the object
    is allowed to land anywhere in the crop - not just near-center - as
    long as it stays fully inside the window; how far it can actually move
    is naturally limited by how much smaller the cluster is than the crop
    and by the frame's edges (the valid range shrinks to ~0 near an edge or
    when the crop is barely bigger than the object). Several candidate
    positions are sampled and scored on how *cleanly* nearby other objects
    come out - each one either meaningfully included or fully excluded,
    never dangling half-visible in between (a half-visible box compounds
    badly with any occlusion already baked into its label - see the caller).

    Returns (best_position, is_clean) - is_clean is True only if a position
    was found where every other box is fully resolved (no dangling box at
    all); the caller uses that to decide whether this crop is safe to keep.
    """
    if wide_position:
        x_lo, x_hi = max(0, cl_x2 - target_w), min(img_w - target_w, cl_x1)
        y_lo, y_hi = max(0, cl_y2 - target_h), min(img_h - target_h, cl_y1)
        if x_hi < x_lo:
            x_lo = x_hi = max(0, min(cl_x1, img_w - target_w))
        if y_hi < y_lo:
            y_lo = y_hi = max(0, min(cl_y1, img_h - target_h))
        attempts = 8 if other_boxes else 1
    else:
        # Legacy behavior: only a small nudge around dead-center.
        cl_cx, cl_cy = (cl_x1 + cl_x2) / 2.0, (cl_y1 + cl_y2) / 2.0
        jitter_x = context_padding * target_w * 0.5
        jitter_y = context_padding * target_h * 0.5
        x_lo = max(0, min(cl_cx - target_w / 2.0 - jitter_x, img_w - target_w))
        x_hi = max(0, min(cl_cx - target_w / 2.0 + jitter_x, img_w - target_w))
        y_lo = max(0, min(cl_cy - target_h / 2.0 - jitter_y, img_h - target_h))
        y_hi = max(0, min(cl_cy - target_h / 2.0 + jitter_y, img_h - target_h))
        attempts = 4 if other_boxes else 1

    best_pos, best_score, best_clean = None, -1.0, False
    for _ in range(attempts):
        x1 = x_lo if x_hi <= x_lo else rng.uniform(x_lo, x_hi)
        y1 = y_lo if y_hi <= y_lo else rng.uniform(y_lo, y_hi)
        x1, y1 = int(x1), int(y1)
        if other_boxes:
            score = _composition_score((x1, y1, x1 + target_w, y1 + target_h), other_boxes, min_visibility)
            clean = score == len(other_boxes)
        else:
            score, clean = 0.0, True
        if score > best_score:
            best_score, best_pos, best_clean = score, (x1, y1), clean
        if best_clean:
            break  # found a fully clean composition - no need to keep rolling

    return best_pos, best_clean


def _build_cluster_crop(
    frame: Any,
    boxes: List[Tuple[int, float, float, float, float]],
    cluster,
    img_w: int,
    img_h: int,
    crop_w: int,
    crop_h: int,
    min_crop_w: int,
    min_crop_h: int,
    context_padding: float,
    min_visibility: float,
    min_box_size: float,
    square_crops: bool,
    use_default_size: bool,
    wide_position: bool,
    rng: random.Random,
):
    """
    Build one candidate crop window (+ translated labels) around a cluster.
    `use_default_size` takes the fixed crop_w x crop_h window instead of
    sizing down to hug the cluster - mixing both gives the dataset both
    tightly-zoomed and default-context views of the same object(s) instead
    of always one fixed relationship between object size and crop size.

    A box that's already partly occluded by something unlabeled only has
    its *visible* fraction annotated - we have no way to know that from the
    box alone. If a crop then *also* truncates that box down to just
    min_visibility, the two effects compound (e.g. a box already only 30%
    of the real object, cropped down to 40% of itself, leaves ~12% of the
    real object standing in for the whole class - actively misleading).
    Since we can't see the hidden occlusion, the safe move is to never let
    *our own* cropping introduce a second truncation: this only returns a
    crop where the cluster's own box(es) fit fully inside the window, and
    where every *other* nearby box is either meaningfully included or
    fully excluded - never left dangling half-visible. When no such clean
    window exists (busy/overlapping scene), it fails (None) rather than
    returning a truncated crop - the object stays labeled at full fidelity
    in the whole-frame "_df" output instead of being force-cropped.

    Returns (window, cropped_img, yolo_labels) or (None, None, None) if no
    clean crop is achievable.
    """
    cl_x1, cl_y1, cl_x2, cl_y2 = _cluster_bbox(cluster)
    cl_w, cl_h = cl_x2 - cl_x1, cl_y2 - cl_y1

    if use_default_size:
        target_w, target_h = crop_w, crop_h
    else:
        # Adaptive crop size: hug the cluster (+ padding) instead of always
        # using the fixed max window, so a small/isolated object actually
        # ends up occupying more of the crop rather than just being
        # relocated onto a same-size canvas.
        target_w = min(crop_w, max(min_crop_w, int(cl_w * (1 + context_padding * 2)) + 1))
        target_h = min(crop_h, max(min_crop_h, int(cl_h * (1 + context_padding * 2)) + 1))

    if square_crops:
        side = max(target_w, target_h)
        side = min(side, min(crop_w, crop_h))
        side = max(side, min(min_crop_w, min_crop_h))
        target_w = target_h = side

    target_w = min(target_w, img_w)
    target_h = min(target_h, img_h)
    if target_w <= 0 or target_h <= 0:
        return None, None, None
    if cl_w > target_w or cl_h > target_h:
        # The cluster itself doesn't fit in this window - any crop here
        # would truncate the object we're building the crop around.
        return None, None, None

    other_boxes = [b for b in boxes if b not in cluster]
    pos, is_clean = _pick_crop_position(
        cl_x1, cl_y1, cl_x2, cl_y2, target_w, target_h, img_w, img_h,
        other_boxes, min_visibility, wide_position, context_padding, rng,
    )
    if pos is None or not is_clean:
        return None, None, None
    crop_x1, crop_y1 = pos
    crop_x2, crop_y2 = crop_x1 + target_w, crop_y1 + target_h
    window = (crop_x1, crop_y1, crop_x2, crop_y2)

    cropped_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    actual_cw, actual_ch = cropped_img.shape[1], cropped_img.shape[0]
    if actual_cw <= 0 or actual_ch <= 0:
        return None, None, None

    yolo_labels = []
    for (g_idx, bx, by, bw, bh) in boxes:
        orig_area = bw * bh
        if orig_area <= 0:
            continue
        rx1, ry1 = max(0.0, bx - crop_x1), max(0.0, by - crop_y1)
        rx2 = min(float(actual_cw), bx + bw - crop_x1)
        ry2 = min(float(actual_ch), by + bh - crop_y1)
        if rx2 <= rx1 or ry2 <= ry1:
            continue

        clipped_w, clipped_h = rx2 - rx1, ry2 - ry1
        if clipped_w < min_box_size or clipped_h < min_box_size:
            continue
        if (clipped_w * clipped_h / orig_area) < min_visibility:
            continue

        cx = (rx1 + clipped_w / 2.0) / actual_cw
        cy = (ry1 + clipped_h / 2.0) / actual_ch
        yolo_labels.append(
            f"{g_idx} {cx:.6f} {cy:.6f} {clipped_w / actual_cw:.6f} {clipped_h / actual_ch:.6f}"
        )

    return window, cropped_img, yolo_labels


def generate_smart_crops(
    frame: Any,
    boxes: List[Tuple[int, float, float, float, float]],
    is_background: bool,
    crop_w: int = 640,
    crop_h: int = 640,
    min_crop_w: int = 320,
    min_crop_h: int = 320,
    min_visibility: float = 0.7,
    context_padding: float = 0.2,
    max_crops_fg: int = 3,
    max_crops_bg: int = 1,
    min_box_size: float = 2.0,
    overlap_iou_threshold: float = 0.5,
    class_priority: Optional[Dict[int, float]] = None,
    square_crops: bool = False,
    default_size_crop_chance: float = 0.3,
    wide_position: bool = True,
    rng: Optional[random.Random] = None,
) -> List[Tuple[Any, List[str]]]:
    """
    Generate object-focused cropped sub-images and translated YOLO labels.

    `boxes` are already resolved to global class indices: (g_idx, x, y, w, h).
    If the image already fits inside crop dimensions, returns it as-is.
    Background (no-box) frames get tiled via generate_background_crops
    instead of being passed through full-resolution, capped at `max_crops_bg`
    rather than `max_crops_fg` - an empty background tile carries no object
    diversity to gain from taking as many of them as a busy foreground frame.

    Crops are filled up to `max_crops_fg` by cycling through the (priority-
    ordered) object clusters rather than taking exactly one crop per
    cluster - so a frame with a single isolated object can still yield
    several differently-jittered/sized crops of it instead of just one, and
    a frame with fewer clusters than max_crops_fg doesn't leave budget unused.
    Each attempt randomly takes either the adaptive cluster-hugging size or
    the fixed crop_w x crop_h size (`default_size_crop_chance`), so the
    dataset gets a mix of tightly-zoomed and default-context views.
    `square_crops` forces every crop window to be square, for models/
    pipelines that expect square input instead of an arbitrary rectangle.
    """
    if rng is None:
        rng = random.Random()

    img_h, img_w = frame.shape[:2]
    if img_w <= crop_w and img_h <= crop_h:
        yolo_labels = []
        for (g_idx, x, y, w, h) in boxes:
            if w >= min_box_size and h >= min_box_size:
                cx, cy = (x + w / 2.0) / img_w, (y + h / 2.0) / img_h
                yolo_labels.append(f"{g_idx} {cx:.6f} {cy:.6f} {w / img_w:.6f} {h / img_h:.6f}")
        return [(frame, yolo_labels)]

    if is_background or not boxes:
        return generate_background_crops(frame, crop_w, crop_h, max_crops_bg, rng, square_crops)

    clusters = cluster_bounding_boxes(boxes, distance_threshold=max(crop_w, crop_h) * 0.4)
    if not clusters:
        return generate_background_crops(frame, crop_w, crop_h, max_crops_bg, rng, square_crops)

    # A cluster whose own extent exceeds the crop window would otherwise get
    # silently clipped at its edges - split it instead.
    split_clusters = []
    for cl in clusters:
        split_clusters.extend(_split_oversized_cluster(cl, crop_w, crop_h))

    # Prioritize small/distant objects and rare classes when we have more
    # candidate clusters than max_crops_fg allows, instead of picking randomly.
    scored = sorted(split_clusters, key=lambda cl: _score_cluster(cl, class_priority), reverse=True)

    accepted_windows: List[Tuple[float, float, float, float]] = []
    results: List[Tuple[Any, List[str]]] = []

    # Finding a *clean* (non-truncating) window can take a few tries in a
    # busy scene, so give this more headroom than a plain "one shot per
    # cluster" budget would - a cluster that never finds a clean crop
    # simply won't get one, and stays fully represented in the "_df" image.
    max_attempts = max(max_crops_fg * 6, len(scored) * 6)
    attempts = 0
    ci = 0
    while len(results) < max_crops_fg and attempts < max_attempts:
        cluster = scored[ci % len(scored)]
        ci += 1
        attempts += 1

        use_default_size = rng.random() < default_size_crop_chance
        window, cropped_img, yolo_labels = _build_cluster_crop(
            frame, boxes, cluster, img_w, img_h, crop_w, crop_h, min_crop_w, min_crop_h,
            context_padding, min_visibility, min_box_size, square_crops, use_default_size, wide_position, rng,
        )
        if window is None:
            continue
        if any(_iou(window, w) > overlap_iou_threshold for w in accepted_windows):
            continue  # near-duplicate of an already-chosen crop in this frame

        accepted_windows.append(window)
        results.append((cropped_img, yolo_labels))

    # NOTE: no "fall back to a random background tile" here on purpose. Boxes
    # exist in this frame (checked at the top of this function) - if none of
    # them could get a clean crop, a random tile would have zero awareness of
    # where those boxes actually are and could land right on top of one,
    # silently saving it as an unlabeled "background" image. Better to return
    # nothing here and let the whole-frame "_df" image (which always carries
    # every box, uncropped) be that object's sole representation instead.
    return results


# --------------------------------------------------------------------------- #
# Class Resolution & YAML Helpers
# --------------------------------------------------------------------------- #

def load_yaml_classes(yaml_path: Path) -> Tuple[List[str], Dict[str, Any]]:
    """Read existing class list and metadata from a YOLO data.yaml file."""
    if not yaml_path.is_file():
        return [], {}
    text = yaml_path.read_text(encoding="utf-8")
    data = yaml.safe_load(text) if yaml else {}
    if not data:
        return [], {}
    names = data.get("names", [])
    if isinstance(names, dict):
        names_list = [names[k] for k in sorted(names.keys(), key=lambda k: int(k))]
    elif isinstance(names, list):
        names_list = list(names)
    else:
        names_list = []
    return names_list, data


def build_class_resolution(
    all_local_names: List[str],
    global_names: List[str],
    class_config: Optional[Dict[str, Dict[str, Any]]] = None,
    manual_class_map: Optional[Dict[str, str]] = None,
    auto_create_new: bool = True,
) -> Tuple[Dict[str, Any], List[str], Dict[str, str]]:
    """
    Resolve local class names to a fate: a global YOLO index (map), or one
    of SKIP / SOFT_DELETE / PURGE.

    class_config: {local_name_lower: {"fate": "map"|"skip"|"soft"|"purge",
                                       "target": "<global class name>"}}
        "target" is only used for fate=="map" and defaults to the local name
        itself. Classes absent from class_config default to fate "map".
    manual_class_map: legacy {local_name: target_name} map, folded in as the
        "target" for any class not given an explicit class_config entry.

    Returns (resolution, global_names, local_to_global_name) where
    local_to_global_name only contains entries for "map"-fated classes,
    local_lower -> global_name_lower (used for class-balancing counts).
    """
    global_lookup = {n.strip().lower(): i for i, n in enumerate(global_names)}
    class_config = {k.strip().lower(): v for k, v in (class_config or {}).items()}
    manual_map = {k.strip().lower(): v for k, v in (manual_class_map or {}).items()}

    unique_local = []
    seen = set()
    for n in all_local_names:
        key = n.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique_local.append(n)

    resolution: Dict[str, Any] = {}
    local_to_global: Dict[str, str] = {}

    for n in unique_local:
        key = n.strip().lower()
        cfg = class_config.get(key, {})
        fate = cfg.get("fate", "map")

        if fate == "skip":
            resolution[key] = SKIP
            continue
        if fate == "soft":
            resolution[key] = SOFT_DELETE
            continue
        if fate == "purge":
            resolution[key] = PURGE
            continue

        # fate == "map"
        display_target = cfg.get("target") or manual_map.get(key) or n
        target = display_target.strip().lower()
        if target in global_lookup:
            resolution[key] = global_lookup[target]
            local_to_global[key] = target
            continue
        if auto_create_new:
            global_names.append(display_target)
            new_idx = len(global_names) - 1
            global_lookup[target] = new_idx
            resolution[key] = new_idx
            local_to_global[key] = target
            continue

        # Can't resolve and can't create a new class: fall back to skipping
        # its boxes rather than silently dropping frames or crashing.
        resolution[key] = SKIP

    return resolution, global_names, local_to_global


# --------------------------------------------------------------------------- #
# Class Balancing Helpers
# --------------------------------------------------------------------------- #

def select_with_gap(indices_sorted: List[int], quota: Optional[int], min_gap: int) -> set:
    """
    Deterministically pick `quota` values out of `indices_sorted` (already
    sorted ascending), spread as evenly as possible across the whole range,
    honoring a minimum frame-index gap between picks where possible. This
    replaces independent-per-frame random sampling so that thinning an
    over-represented class doesn't just keep two adjacent near-duplicate
    frames by chance while dropping frames from elsewhere in the video.
    """
    n = len(indices_sorted)
    if quota is None or quota >= n:
        return set(indices_sorted)
    if quota <= 0:
        return set()

    step = n / float(quota)
    chosen = []
    last_frame = -10 ** 9
    for k in range(quota):
        pos = min(int(round(k * step)), n - 1)
        p = pos
        while p < n and indices_sorted[p] - last_frame < min_gap:
            p += 1
        if p >= n:
            p = pos
            while p >= 0 and indices_sorted[p] - last_frame < min_gap:
                p -= 1
        if 0 <= p < n:
            chosen.append(indices_sorted[p])
            last_frame = indices_sorted[p]
    return set(chosen)


def allocate_quota_across_videos(per_video_counts: Dict[Any, int], target_total: Optional[int]) -> Dict[Any, int]:
    """Split a global per-class target across the videos that contain it, proportional to each video's share."""
    total = sum(per_video_counts.values())
    if target_total is None or total <= target_total:
        return dict(per_video_counts)

    raw = {v: (c * target_total / total) for v, c in per_video_counts.items()}
    floors = {v: int(math.floor(r)) for v, r in raw.items()}
    remainder = target_total - sum(floors.values())
    order = sorted(raw.keys(), key=lambda v: raw[v] - floors[v], reverse=True)
    alloc = dict(floors)
    for v in order[:max(0, remainder)]:
        alloc[v] += 1
    return alloc


def compute_class_targets(
    resolved_instance_counts: Dict[str, int],
    balance_mode: str,
    manual_caps: Optional[Dict[str, int]] = None,
    classes_in_balance: Optional[set] = None,
) -> Dict[str, Optional[int]]:
    """
    balance_mode: "none" (no capping) | "manual" (use manual_caps) |
                  "auto_min" (cap every included class to the rarest one's count)
    Returns {global_class_name_lower: target_count_or_None} - None means uncapped.
    """
    targets: Dict[str, Optional[int]] = {name: None for name in resolved_instance_counts}
    manual_caps = {k.strip().lower(): v for k, v in (manual_caps or {}).items()}

    if balance_mode == "manual":
        for name, cap in manual_caps.items():
            if name in resolved_instance_counts and cap > 0:
                targets[name] = min(cap, resolved_instance_counts[name])
    elif balance_mode == "auto_min":
        pool = classes_in_balance or set(resolved_instance_counts.keys())
        counts_in_pool = [c for n, c in resolved_instance_counts.items() if n in pool and c > 0]
        if counts_in_pool:
            min_count = min(counts_in_pool)
            for name in pool:
                if name in resolved_instance_counts:
                    targets[name] = min_count
        for name, cap in manual_caps.items():
            if name in resolved_instance_counts and cap > 0:
                current = targets.get(name)
                targets[name] = min(cap, current) if current is not None else cap

    return targets


# --------------------------------------------------------------------------- #
# Phase A: Planning
# --------------------------------------------------------------------------- #

def build_plan(
    source_dir: Path,
    yaml_path: Optional[Path] = None,
    class_config: Optional[Dict[str, Dict[str, Any]]] = None,
    manual_class_map: Optional[Dict[str, str]] = None,
    auto_create_classes: bool = True,
    dist: int = 1,
    bg_remove_percent: float = 0.0,
    balance_mode: str = "none",
    manual_caps: Optional[Dict[str, int]] = None,
    classes_in_balance: Optional[List[str]] = None,
    min_frame_gap: int = 0,
    max_frames_per_video: Optional[int] = None,
    random_seed: Optional[int] = 42,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    Fast, decode-free planning pass: figure out exactly which frame indices
    will be extracted from each video before touching any pixels. Cheap
    enough to call as a live preview/dry-run.
    """
    rng = random.Random(random_seed)
    source_dir = Path(source_dir).expanduser().resolve()
    video_pairs = discover_video_pairs(source_dir)
    if not video_pairs:
        raise ValueError(f"No valid video + .txt annotation pairs found in {source_dir}")

    global_names: List[str] = []
    if yaml_path and Path(yaml_path).is_file():
        global_names, _ = load_yaml_classes(Path(yaml_path))

    parsed: Dict[Path, Tuple[List[str], List[str]]] = {}
    all_local_names: List[str] = []
    for video_path, txt_path in video_pairs:
        local_names, frame_lines = parse_annotation_file(txt_path)
        parsed[video_path] = (local_names, frame_lines)
        all_local_names.extend(local_names)

    resolution, global_names, local_to_global = build_class_resolution(
        all_local_names, list(global_names), class_config, manual_class_map, auto_create_classes
    )

    mismatches: List[Dict[str, Any]] = []
    per_video_class_frames: Dict[Path, Dict[str, List[int]]] = {}
    frame_plan: Dict[Path, Dict[int, Dict[str, Any]]] = {}
    deleted_total = 0

    total_videos = len(video_pairs)
    for v_i, (video_path, txt_path) in enumerate(video_pairs):
        if cancel_callback and cancel_callback():
            return {"cancelled": True}
        if progress_callback:
            progress_callback(int(v_i / total_videos * 30), f"Planning {txt_path.name}...")

        local_names, frame_lines = parsed[video_path]

        actual_frames = None
        if cv2 is not None:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                actual_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
        if actual_frames is not None and actual_frames != len(frame_lines):
            mismatches.append({
                "video": str(video_path),
                "annotated_lines": len(frame_lines),
                "actual_frames": actual_frames,
                "diff": actual_frames - len(frame_lines),
            })

        v_class_frames: Dict[str, List[int]] = {}
        v_frame_plan: Dict[int, Dict[str, Any]] = {}

        for frame_idx, line in enumerate(frame_lines):
            if frame_idx % dist != 0:
                continue
            parsed_line = parse_frame_line(line)
            if parsed_line == "DELETED;":
                deleted_total += 1
                continue
            if parsed_line is None:
                continue

            if len(parsed_line) == 0:
                v_frame_plan[frame_idx] = {"boxes": [], "is_background": True}
                continue

            kept_boxes = []
            has_purge = False
            for (cls_idx, x, y, w, h) in parsed_line:
                if cls_idx < 0 or cls_idx >= len(local_names):
                    continue
                lname = local_names[cls_idx].strip().lower()
                fate = resolution.get(lname, SKIP)
                if fate == PURGE:
                    has_purge = True
                    break
                if fate == SKIP or fate == SOFT_DELETE:
                    continue
                gname = local_to_global.get(lname, lname)
                kept_boxes.append((fate, gname, x, y, w, h))

            if has_purge:
                continue  # whole frame dropped

            if not kept_boxes:
                # everything present was skip/soft-delete -> becomes background
                v_frame_plan[frame_idx] = {"boxes": [], "is_background": True}
                continue

            v_frame_plan[frame_idx] = {"boxes": kept_boxes, "is_background": False}
            for (_gidx, gname, *_rest) in kept_boxes:
                v_class_frames.setdefault(gname, []).append(frame_idx)

        per_video_class_frames[video_path] = {k: sorted(set(v)) for k, v in v_class_frames.items()}
        frame_plan[video_path] = v_frame_plan

    resolved_instance_counts: Dict[str, int] = {}
    for v_class_frames in per_video_class_frames.values():
        for gname, frames in v_class_frames.items():
            resolved_instance_counts[gname] = resolved_instance_counts.get(gname, 0) + len(frames)

    balance_pool = {c.strip().lower() for c in classes_in_balance} if classes_in_balance else None
    targets = compute_class_targets(resolved_instance_counts, balance_mode, manual_caps, balance_pool)

    keep_sets: Dict[Path, Dict[str, set]] = {}
    for gname, target in targets.items():
        if target is None:
            continue
        per_video_counts = {
            v: len(fr[gname]) for v, fr in per_video_class_frames.items() if gname in fr
        }
        if not per_video_counts:
            continue
        alloc = allocate_quota_across_videos(per_video_counts, target)
        for video_path, quota in alloc.items():
            frames_sorted = per_video_class_frames[video_path][gname]
            keep_sets.setdefault(video_path, {})[gname] = select_with_gap(frames_sorted, quota, min_frame_gap)

    total_balance_dropped = 0
    total_bg_dropped = 0
    final_plan: Dict[Path, List[int]] = {}

    for video_path, v_frame_plan in frame_plan.items():
        survivors = []
        for frame_idx in sorted(v_frame_plan.keys()):
            entry = v_frame_plan[frame_idx]
            if entry["is_background"]:
                if bg_remove_percent > 0 and rng.random() * 100 < bg_remove_percent:
                    total_bg_dropped += 1
                    continue
                survivors.append(frame_idx)
                continue

            classes_here = {gname for (_gidx, gname, *_r) in entry["boxes"]}
            v_keep_sets = keep_sets.get(video_path, {})
            frame_ok = False
            for gname in classes_here:
                if targets.get(gname) is None:
                    frame_ok = True
                    break
                if frame_idx in v_keep_sets.get(gname, set()):
                    frame_ok = True
                    break
            if not frame_ok:
                total_balance_dropped += 1
                continue
            survivors.append(frame_idx)

        if max_frames_per_video and len(survivors) > max_frames_per_video:
            survivors = sorted(select_with_gap(survivors, max_frames_per_video, 0))

        final_plan[video_path] = survivors

    if progress_callback:
        progress_callback(35, "Planning complete.")

    return {
        "cancelled": False,
        "video_pairs": video_pairs,
        "parsed": parsed,
        "resolution": resolution,
        "global_names": global_names,
        "local_to_global": local_to_global,
        "frame_plan": frame_plan,
        "final_plan": final_plan,
        "targets": targets,
        "resolved_instance_counts": resolved_instance_counts,
        "mismatches": mismatches,
        "deleted_total": deleted_total,
        "balance_dropped_total": total_balance_dropped,
        "bg_dropped_total": total_bg_dropped,
    }


def scan_dataset_statistics(
    source_dir: Path,
    yaml_path: Optional[Path] = None,
    class_config: Optional[Dict[str, Dict[str, Any]]] = None,
    manual_class_map: Optional[Dict[str, str]] = None,
    auto_create_classes: bool = True,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """
    Discovery pass for the UI: what classes exist in this source folder, how
    many instances/frames each has, and what they currently resolve to.
    Thin wrapper over build_plan() with no balancing applied, so it's cheap
    and safe to call as soon as a source folder is picked - before the user
    has configured any class fates.
    """
    plan = build_plan(
        source_dir=source_dir, yaml_path=yaml_path, class_config=class_config,
        manual_class_map=manual_class_map, auto_create_classes=auto_create_classes,
        dist=1, bg_remove_percent=0.0, balance_mode="none",
        progress_callback=progress_callback, cancel_callback=cancel_callback,
    )
    if plan.get("cancelled"):
        return plan

    resolution = plan["resolution"]
    local_to_global = plan["local_to_global"]

    raw_instance_counts: Dict[str, int] = {}
    raw_frame_counts: Dict[str, int] = {}
    for _video_path, (local_names, frame_lines) in plan["parsed"].items():
        for line in frame_lines:
            boxes = parse_frame_line(line)
            if not boxes or not isinstance(boxes, list):
                continue
            seen_in_frame = set()
            for (cls_idx, *_rest) in boxes:
                if 0 <= cls_idx < len(local_names):
                    cname = local_names[cls_idx].strip().lower()
                    raw_instance_counts[cname] = raw_instance_counts.get(cname, 0) + 1
                    if cname not in seen_in_frame:
                        raw_frame_counts[cname] = raw_frame_counts.get(cname, 0) + 1
                        seen_in_frame.add(cname)

    classes_report = {}
    for cname in sorted(raw_instance_counts.keys()):
        fate = resolution.get(cname, SKIP)
        classes_report[cname] = {
            "instance_count": raw_instance_counts[cname],
            "frame_count": raw_frame_counts.get(cname, 0),
            "fate": (
                "purge" if fate == PURGE else
                "soft" if fate == SOFT_DELETE else
                "skip" if fate == SKIP else
                "map"
            ),
            "resolved_to": local_to_global.get(cname, cname),
        }

    return {
        "cancelled": False,
        "video_pairs": plan["video_pairs"],
        "classes": classes_report,
        "resolved_instance_counts": plan["resolved_instance_counts"],
        "global_names": plan["global_names"],
        "mismatches": plan["mismatches"],
        "total_videos": len(plan["video_pairs"]),
    }


# --------------------------------------------------------------------------- #
# Phase B: Execution
# --------------------------------------------------------------------------- #

def execute_plan(
    plan: Dict[str, Any],
    output_dir: Path,
    source_dir: Path,
    img_ext: str = ".jpg",
    remove_padding: bool = False,
    black_thresh: int = 16,
    enable_dedup: bool = False,
    dedup_hamming_threshold: int = 4,
    dedup_window: int = 40,
    enable_smart_crop: bool = False,
    crop_size: Tuple[int, int] = (640, 640),
    min_crop_size: Tuple[int, int] = (320, 320),
    max_crops_per_frame: int = 3,
    max_bg_crops_per_frame: int = 1,
    min_visibility: float = 0.7,
    context_padding: float = 0.2,
    overlap_iou_threshold: float = 0.5,
    square_crops: bool = False,
    default_size_crop_chance: float = 0.3,
    wide_position: bool = True,
    include_default_frame: bool = True,
    default_frame_chance: float = 1.0,
    min_box_size_px: float = 2.0,
    flip_augment_percent: float = 0.0,
    split_mode: str = "single",
    split_ratios: Tuple[float, float, float] = (0.8, 0.2, 0.0),
    random_seed: Optional[int] = 42,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
):
    """Execute a plan produced by build_plan(): decode only the planned frames and write the dataset."""
    rng = random.Random(random_seed)
    output_dir = Path(output_dir).expanduser().resolve()
    source_dir = Path(source_dir).expanduser().resolve()

    video_pairs = plan["video_pairs"]
    final_plan = plan["final_plan"]
    frame_plan_all = plan["frame_plan"]
    global_names = plan["global_names"]
    resolved_counts = plan.get("resolved_instance_counts", {})

    global_lookup_all = {n.strip().lower(): i for i, n in enumerate(global_names)}
    class_priority: Dict[int, float] = {}
    if resolved_counts:
        max_count = max(resolved_counts.values())
        for gname, cnt in resolved_counts.items():
            idx = global_lookup_all.get(gname)
            if idx is not None and cnt > 0:
                class_priority[idx] = max_count / float(cnt)

    total_videos = len(video_pairs)
    total_saved = 0
    total_dedup_skipped = 0

    clip_split_map = {}
    if split_mode == "auto":
        shuffled = list(video_pairs)
        rng.shuffle(shuffled)
        n_total = len(shuffled)
        n_train = int(n_total * split_ratios[0])
        n_val = int(n_total * split_ratios[1])
        for idx, (vp, _) in enumerate(shuffled):
            clip_split_map[vp] = "train" if idx < n_train else ("val" if idx < n_train + n_val else "test")

    for v_idx, (video_path, txt_path) in enumerate(video_pairs):
        if cancel_callback and cancel_callback():
            yield 0, "Conversion cancelled by user."
            return

        pct = int(35 + (v_idx / total_videos) * 60)
        v_stem = video_path.stem
        msg = f"Converting video {v_idx + 1}/{total_videos}: {v_stem}"
        if progress_callback:
            progress_callback(pct, msg)
        yield pct, msg

        wanted_frames = set(final_plan.get(video_path, []))
        if not wanted_frames:
            continue

        if split_mode == "auto":
            split_tag = clip_split_map.get(video_path, "train")
            dest_img_dir = output_dir / split_tag / "images"
            dest_lbl_dir = output_dir / split_tag / "labels"
        elif split_mode == "preserve":
            rel_parent = video_path.parent.relative_to(source_dir)
            if str(rel_parent) != ".":
                dest_img_dir = output_dir / rel_parent / "images"
                dest_lbl_dir = output_dir / rel_parent / "labels"
            else:
                dest_img_dir = output_dir / "images"
                dest_lbl_dir = output_dir / "labels"
        else:
            dest_img_dir = output_dir / "images"
            dest_lbl_dir = output_dir / "labels"
        dest_img_dir.mkdir(parents=True, exist_ok=True)
        dest_lbl_dir.mkdir(parents=True, exist_ok=True)

        v_frame_plan = frame_plan_all[video_path]

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue

        recent_hashes: List[Tuple[int, int]] = []
        max_wanted = max(wanted_frames)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx > max_wanted:
                break
            if frame_idx not in wanted_frames:
                frame_idx += 1
                continue

            entry = v_frame_plan.get(frame_idx, {"boxes": [], "is_background": True})
            img = frame
            left_pad, top_pad = 0, 0
            if remove_padding:
                l, t, r, b = detect_content_bbox(img, black_thresh)
                if (l, t, r, b) != (0, 0, img.shape[1], img.shape[0]):
                    img = img[t:b, l:r]
                    left_pad, top_pad = l, t

            if enable_dedup and cv2 is not None and np is not None:
                h = compute_dhash(img)
                if h is not None:
                    if any(hamming_distance(h, ph) <= dedup_hamming_threshold for _, ph in recent_hashes):
                        total_dedup_skipped += 1
                        frame_idx += 1
                        continue
                    recent_hashes.append((frame_idx, h))
                    if len(recent_hashes) > dedup_window:
                        recent_hashes.pop(0)

            adjusted_boxes = []
            for (g_idx, _gname, bx, by, bw, bh) in entry["boxes"]:
                nbx, nby = bx - left_pad, by - top_pad
                if nbx + bw > 0 and nby + bh > 0 and nbx < img.shape[1] and nby < img.shape[0]:
                    adjusted_boxes.append((g_idx, max(0.0, nbx), max(0.0, nby), bw, bh))

            is_background = entry["is_background"]

            def _full_frame_labels():
                h_img, w_img = img.shape[0], img.shape[1]
                lines = []
                for (g_idx, bx, by, bw, bh) in adjusted_boxes:
                    if bw >= min_box_size_px and bh >= min_box_size_px:
                        cx = (bx + bw / 2.0) / w_img
                        cy = (by + bh / 2.0) / h_img
                        lines.append(f"{g_idx} {cx:.6f} {cy:.6f} {bw / w_img:.6f} {bh / h_img:.6f}")
                return lines

            # outputs: list of (image, yolo_labels, filename_suffix)
            if enable_smart_crop and (img.shape[1] > crop_size[0] or img.shape[0] > crop_size[1]):
                crops_to_save = generate_smart_crops(
                    img, adjusted_boxes, is_background,
                    crop_w=crop_size[0], crop_h=crop_size[1],
                    min_crop_w=min_crop_size[0], min_crop_h=min_crop_size[1],
                    min_visibility=min_visibility, context_padding=context_padding,
                    max_crops_fg=max_crops_per_frame, max_crops_bg=max_bg_crops_per_frame,
                    min_box_size=min_box_size_px,
                    overlap_iou_threshold=overlap_iou_threshold, class_priority=class_priority,
                    square_crops=square_crops, default_size_crop_chance=default_size_crop_chance,
                    wide_position=wide_position, rng=rng,
                )
                outputs = [(c_img, c_labels, f"_c{i}") for i, (c_img, c_labels) in enumerate(crops_to_save)]
                # Alongside the object-focused crop(s), (almost) always also keep the
                # whole (post-padding-removal) frame with every one of its labels, so
                # the model also sees full-scene context and not only zoomed sub-crops.
                # If no crop at all could be produced (busy/occluded scene, no clean
                # window found for anything), this becomes the ONLY representation of
                # those objects - always include it in that case, regardless of the
                # configured chance, so the frame isn't silently dropped altogether.
                if include_default_frame and (not outputs or rng.random() < default_frame_chance):
                    outputs.append((img, _full_frame_labels(), "_df"))
            else:
                # Frame already fits within the crop size - it IS the default view,
                # nothing else to add.
                outputs = [(img, _full_frame_labels(), "")]

            for (crop_img, yolo_labels, suffix) in outputs:
                base = f"{v_stem}_{frame_idx:06d}{suffix}"
                cv2.imwrite(str(dest_img_dir / f"{base}{img_ext}"), crop_img)
                (dest_lbl_dir / f"{base}.txt").write_text(
                    "\n".join(yolo_labels) + ("\n" if yolo_labels else ""), encoding="utf-8"
                )
                total_saved += 1

                if flip_augment_percent > 0 and rng.random() * 100 < flip_augment_percent:
                    flipped_img = cv2.flip(crop_img, 1)
                    flipped_labels = []
                    for lbl_line in yolo_labels:
                        parts = lbl_line.split()
                        if len(parts) >= 5:
                            gid, fcx, fcy, fnw, fnh = parts[0], parts[1], parts[2], parts[3], parts[4]
                            flipped_labels.append(f"{gid} {1.0 - float(fcx):.6f} {fcy} {fnw} {fnh}")
                    flip_base = f"{base}_flip"
                    cv2.imwrite(str(dest_img_dir / f"{flip_base}{img_ext}"), flipped_img)
                    (dest_lbl_dir / f"{flip_base}.txt").write_text(
                        "\n".join(flipped_labels) + ("\n" if flipped_labels else ""), encoding="utf-8"
                    )
                    total_saved += 1

            frame_idx += 1

        cap.release()

    yaml_out_path = output_dir / "data.yaml"
    out_yaml_dict = {
        "path": str(output_dir),
        "names": {i: name for i, name in enumerate(global_names)} if global_names else {},
        "nc": len(global_names),
    }
    if split_mode == "auto":
        out_yaml_dict["train"] = "train/images"
        out_yaml_dict["val"] = "val/images"
        if split_ratios[2] > 0:
            out_yaml_dict["test"] = "test/images"
    else:
        out_yaml_dict["train"] = "images"
        out_yaml_dict["val"] = "images"

    if yaml:
        yaml_out_path.write_text(yaml.safe_dump(out_yaml_dict, sort_keys=False), encoding="utf-8")

    summary_msg = (
        f"Completed! {total_saved} images written across {total_videos} videos. "
        f"{plan.get('deleted_total', 0)} DELETED frames dropped, "
        f"{plan.get('bg_dropped_total', 0)} background frames thinned, "
        f"{plan.get('balance_dropped_total', 0)} frames dropped for class balance, "
        f"{total_dedup_skipped} near-duplicate frames skipped."
    )
    if progress_callback:
        progress_callback(100, summary_msg)
    yield 100, summary_msg


# --------------------------------------------------------------------------- #
# Backward-compatible combined entry point
# --------------------------------------------------------------------------- #

def convert_video_dataset_to_yolo(
    source_dir: Path,
    output_dir: Path,
    yaml_path: Optional[Path] = None,
    dist: int = 1,
    img_ext: str = ".jpg",
    remove_padding: bool = False,
    black_thresh: int = 16,
    bg_remove_percent: float = 0.0,
    enable_smart_crop: bool = False,
    crop_size: Tuple[int, int] = (640, 640),
    min_crop_size: Tuple[int, int] = (320, 320),
    max_crops_per_frame: int = 3,
    max_bg_crops_per_frame: int = 1,
    min_visibility: float = 0.7,
    context_padding: float = 0.2,
    overlap_iou_threshold: float = 0.5,
    square_crops: bool = False,
    default_size_crop_chance: float = 0.3,
    wide_position: bool = True,
    include_default_frame: bool = True,
    default_frame_chance: float = 1.0,
    max_instances_per_class: Optional[Dict[str, int]] = None,
    balance_mode: str = "manual",
    classes_in_balance: Optional[List[str]] = None,
    min_frame_gap: int = 0,
    max_frames_per_video: Optional[int] = None,
    enable_dedup: bool = False,
    dedup_hamming_threshold: int = 4,
    dedup_window: int = 40,
    flip_augment_percent: float = 0.0,
    min_box_size_px: float = 2.0,
    manual_class_map: Optional[Dict[str, str]] = None,
    class_config: Optional[Dict[str, Dict[str, Any]]] = None,
    auto_create_classes: bool = True,
    split_mode: str = "single",  # "single" | "preserve" | "auto"
    split_ratios: Tuple[float, float, float] = (0.8, 0.2, 0.0),  # train, val, test
    random_seed: Optional[int] = 42,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
):
    """
    Backward-compatible combined entry point: runs build_plan() then
    execute_plan() back to back. Yields (progress_percent, status_message).
    """
    effective_balance_mode = balance_mode
    if balance_mode == "manual" and not max_instances_per_class:
        effective_balance_mode = "none"

    plan = build_plan(
        source_dir=source_dir, yaml_path=yaml_path, class_config=class_config,
        manual_class_map=manual_class_map, auto_create_classes=auto_create_classes,
        dist=dist, bg_remove_percent=bg_remove_percent,
        balance_mode=effective_balance_mode, manual_caps=max_instances_per_class,
        classes_in_balance=classes_in_balance, min_frame_gap=min_frame_gap,
        max_frames_per_video=max_frames_per_video, random_seed=random_seed,
        progress_callback=progress_callback, cancel_callback=cancel_callback,
    )
    if plan.get("cancelled"):
        yield 0, "Conversion cancelled by user."
        return

    if plan["mismatches"]:
        names = ", ".join(Path(m["video"]).name for m in plan["mismatches"][:5])
        more = "" if len(plan["mismatches"]) <= 5 else f" (+{len(plan['mismatches']) - 5} more)"
        msg = f"Note: {len(plan['mismatches'])} video(s) have annotation/frame-count mismatches: {names}{more}"
        if progress_callback:
            progress_callback(35, msg)
        yield 35, msg

    for pct, msg in execute_plan(
        plan, output_dir=output_dir, source_dir=source_dir, img_ext=img_ext,
        remove_padding=remove_padding, black_thresh=black_thresh,
        enable_dedup=enable_dedup, dedup_hamming_threshold=dedup_hamming_threshold,
        dedup_window=dedup_window, enable_smart_crop=enable_smart_crop,
        crop_size=crop_size, min_crop_size=min_crop_size, max_crops_per_frame=max_crops_per_frame,
        max_bg_crops_per_frame=max_bg_crops_per_frame,
        min_visibility=min_visibility, context_padding=context_padding,
        overlap_iou_threshold=overlap_iou_threshold, square_crops=square_crops,
        default_size_crop_chance=default_size_crop_chance, wide_position=wide_position,
        include_default_frame=include_default_frame,
        default_frame_chance=default_frame_chance, min_box_size_px=min_box_size_px,
        flip_augment_percent=flip_augment_percent, split_mode=split_mode, split_ratios=split_ratios,
        random_seed=random_seed, progress_callback=progress_callback, cancel_callback=cancel_callback,
    ):
        yield pct, msg
