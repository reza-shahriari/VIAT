"""
Video Dataset to YOLO Dataset Converter Engine.

Converts a folder of videos + per-frame annotation files (.txt in Raya format)
into a YOLO-formatted image dataset (images/ + labels/ + data.yaml).

Features:
- O(N) Two-Pass Uniform Class Balancing & Instance Capping
- Smart Object-Focused Multi-Cropping (generating high-res sub-images from distant objects)
- Letterbox / Pillarbox Padding Auto-Detection & Removal
- Background Frame Thinning (randomly dropping empty [] frames)
- Horizontal Flip Augmentation with Inverted YOLO Coordinates
- Minimum Bounding Box Size Filtering
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
BULLET_RE = re.compile(r"^[\-\*\u2022]\s*")
BOX_GROUP_RE = re.compile(r"\[([^\[\]]*)\]")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v", ".wmv"}


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
# Fast O(N) Text Scanner & Class Sampling Rates
# --------------------------------------------------------------------------- #

def scan_dataset_class_statistics(video_pairs: List[Tuple[Path, Path]]) -> Tuple[Dict[str, int], Dict[Path, Tuple[List[str], List[str]]]]:
    """
    Fast O(N) text-only scan across all annotation files in the dataset.
    
    Returns:
        global_class_counts: Total instance count per class name (lowercase)
        parsed_files: Cached {video_path: (local_names, frame_lines)}
    """
    global_class_counts = {}
    parsed_files = {}

    for video_path, txt_path in video_pairs:
        local_names, frame_lines = parse_annotation_file(txt_path)
        parsed_files[video_path] = (local_names, frame_lines)

        for line in frame_lines:
            boxes = parse_frame_line(line)
            if boxes and isinstance(boxes, list):
                for (cls_idx, x, y, w, h) in boxes:
                    if 0 <= cls_idx < len(local_names):
                        cname = local_names[cls_idx].strip().lower()
                        global_class_counts[cname] = global_class_counts.get(cname, 0) + 1

    return global_class_counts, parsed_files


def compute_class_sampling_rates(
    global_class_counts: Dict[str, int],
    max_instances_per_class: Optional[Dict[str, int]] = None
) -> Dict[str, float]:
    """
    Compute uniform sampling rate r_c in [0.0, 1.0] for each class across all videos.
    
    r_c = min(1.0, max_instances / total_count)
    """
    rates = {}
    if not max_instances_per_class:
        for cname in global_class_counts:
            rates[cname] = 1.0
        return rates

    max_map = {k.strip().lower(): v for k, v in max_instances_per_class.items()}

    for cname, count in global_class_counts.items():
        if cname in max_map and max_map[cname] > 0 and count > 0:
            rates[cname] = min(1.0, float(max_map[cname]) / float(count))
        else:
            rates[cname] = 1.0
    return rates


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
# Smart Object-Focused Cropping & Pre-Augmentations
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


def generate_smart_crops(
    frame: Any,
    boxes: List[Tuple[int, float, float, float, float]],
    local_names: List[str],
    resolution: Dict[str, Optional[int]],
    crop_w: int = 640,
    crop_h: int = 640,
    min_visibility: float = 0.4,
    context_padding: float = 0.2,
    max_crops: int = 3,
    min_box_size: float = 2.0,
    rng: Optional[random.Random] = None,
) -> List[Tuple[Any, List[str]]]:
    """
    Generate object-focused cropped sub-images and translated YOLO labels.
    
    If the image is already smaller or equal to crop dimensions, returns the frame as-is.
    Otherwise, generates multiple diverse crops centered around object clusters.
    
    Returns:
        List of (cropped_image, list_of_yolo_label_strings)
    """
    if rng is None:
        rng = random.Random()

    img_h, img_w = frame.shape[:2]
    if img_w <= crop_w and img_h <= crop_h:
        # Frame fits inside crop dimensions directly
        yolo_labels = []
        for (cls_idx, x, y, w, h) in boxes:
            if 0 <= cls_idx < len(local_names):
                name = local_names[cls_idx].strip().lower()
                g_idx = resolution.get(name)
                if g_idx is not None and w >= min_box_size and h >= min_box_size:
                    cx = (x + w / 2.0) / img_w
                    cy = (y + h / 2.0) / img_h
                    nw = w / img_w
                    nh = h / img_h
                    yolo_labels.append(f"{g_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        return [(frame, yolo_labels)]

    clusters = cluster_bounding_boxes(boxes, distance_threshold=max(crop_w, crop_h) * 0.4)
    if not clusters:
        # Background frame: take a random or center crop
        max_x = max(0, img_w - crop_w)
        max_y = max(0, img_h - crop_h)
        cx1 = rng.randint(0, max_x) if max_x > 0 else 0
        cy1 = rng.randint(0, max_y) if max_y > 0 else 0
        cx2 = min(cx1 + crop_w, img_w)
        cy2 = min(cy1 + crop_h, img_h)
        crop_img = frame[cy1:cy2, cx1:cx2]
        return [(crop_img, [])]

    # Limit to max_crops
    if len(clusters) > max_crops:
        clusters = rng.sample(clusters, max_crops)

    results = []
    target_w = min(crop_w, img_w)
    target_h = min(crop_h, img_h)

    for cluster in clusters:
        # Bounding box of the entire cluster
        cl_x1 = min(b[1] for b in cluster)
        cl_y1 = min(b[2] for b in cluster)
        cl_x2 = max(b[1] + b[3] for b in cluster)
        cl_y2 = max(b[2] + b[4] for b in cluster)

        cl_w = cl_x2 - cl_x1
        cl_h = cl_y2 - cl_y1
        cl_center_x = (cl_x1 + cl_x2) / 2.0
        cl_center_y = (cl_y1 + cl_y2) / 2.0

        # Calculate crop bounds with context jitter
        jitter_x = (rng.random() * 2 - 1) * (context_padding * target_w)
        jitter_y = (rng.random() * 2 - 1) * (context_padding * target_h)

        crop_center_x = cl_center_x + jitter_x
        crop_center_y = cl_center_y + jitter_y

        crop_x1 = int(crop_center_x - target_w / 2.0)
        crop_y1 = int(crop_center_y - target_h / 2.0)

        # Clamp within frame boundaries
        crop_x1 = max(0, min(crop_x1, img_w - target_w))
        crop_y1 = max(0, min(crop_y1, img_h - target_h))
        crop_x2 = crop_x1 + target_w
        crop_y2 = crop_y1 + target_h

        cropped_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        actual_cw = cropped_img.shape[1]
        actual_ch = cropped_img.shape[0]

        # Translate and clip all frame boxes into this crop window
        yolo_labels = []
        for (cls_idx, bx, by, bw, bh) in boxes:
            if cls_idx < 0 or cls_idx >= len(local_names):
                continue
            name = local_names[cls_idx].strip().lower()
            g_idx = resolution.get(name)
            if g_idx is None:
                continue

            orig_area = bw * bh
            if orig_area <= 0:
                continue

            # Translate to crop coordinates
            rx1 = max(0.0, bx - crop_x1)
            ry1 = max(0.0, by - crop_y1)
            rx2 = min(float(actual_cw), bx + bw - crop_x1)
            ry2 = min(float(actual_ch), by + bh - crop_y1)

            if rx2 <= rx1 or ry2 <= ry1:
                continue  # box is completely outside crop

            clipped_w = rx2 - rx1
            clipped_h = ry2 - ry1
            clipped_area = clipped_w * clipped_h

            if clipped_w < min_box_size or clipped_h < min_box_size:
                continue

            # Visibility check
            if (clipped_area / orig_area) < min_visibility:
                continue

            # YOLO normalized coordinates
            cx = (rx1 + clipped_w / 2.0) / actual_cw
            cy = (ry1 + clipped_h / 2.0) / actual_ch
            nw = clipped_w / actual_cw
            nh = clipped_h / actual_ch
            yolo_labels.append(f"{g_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

        results.append((cropped_img, yolo_labels))

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
    manual_map: Optional[Dict[str, str]] = None,
    auto_create_new: bool = True
) -> Tuple[Dict[str, Optional[int]], List[str]]:
    """
    Resolve local class names to global YOLO indices.
    """
    global_lookup = {n.strip().lower(): i for i, n in enumerate(global_names)}
    manual = {k.strip().lower(): v for k, v in (manual_map or {}).items()}
    resolution: Dict[str, Optional[int]] = {}

    unique_local = []
    for n in all_local_names:
        key = n.strip().lower()
        if key and key not in [u.strip().lower() for u in unique_local]:
            unique_local.append(n)

    for n in unique_local:
        key = n.strip().lower()
        if key in global_lookup:
            resolution[key] = global_lookup[key]
            continue
        if key in manual:
            target = manual[key].strip().lower()
            if target in global_lookup:
                resolution[key] = global_lookup[target]
                continue
            elif auto_create_new:
                global_names.append(manual[key])
                new_idx = len(global_names) - 1
                global_lookup[target] = new_idx
                resolution[key] = new_idx
                continue
        if auto_create_new:
            global_names.append(n)
            new_idx = len(global_names) - 1
            global_lookup[key] = new_idx
            resolution[key] = new_idx
            continue

    return resolution, global_names


# --------------------------------------------------------------------------- #
# Main Conversion Engine (Generator with Progress)
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
    max_crops_per_frame: int = 3,
    min_visibility: float = 0.4,
    context_padding: float = 0.2,
    max_instances_per_class: Optional[Dict[str, int]] = None,
    flip_augment_percent: float = 0.0,
    min_box_size_px: float = 2.0,
    manual_class_map: Optional[Dict[str, str]] = None,
    auto_create_classes: bool = True,
    split_mode: str = "single",  # "single" | "preserve" | "auto"
    split_ratios: Tuple[float, float, float] = (0.8, 0.2, 0.0),  # train, val, test
    random_seed: Optional[int] = 42,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
):
    """
    Main generator for converting a video dataset to YOLO format.
    
    Yields (progress_percent: int, status_message: str).
    """
    rng = random.Random(random_seed)
    source_dir = Path(source_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    if not source_dir.is_dir():
        raise ValueError(f"Source folder not found: {source_dir}")

    # Discover videos and annotations (flat or split subfolders)
    video_pairs = []
    for root_p, _, files in os.walk(source_dir):
        for f in sorted(files):
            ext = os.path.splitext(f)[1].lower()
            if ext in VIDEO_EXTENSIONS and not f.lower().startswith("outvideo"):
                v_path = Path(root_p) / f
                txt_path = v_path.with_suffix(".txt")
                if txt_path.is_file():
                    video_pairs.append((v_path, txt_path))

    if not video_pairs:
        raise ValueError(f"No valid video + .txt annotation pairs found in {source_dir}")

    # Pass 1: Fast O(N) Text Scan
    if progress_callback:
        progress_callback(5, f"Scanning {len(video_pairs)} annotation files...")
    global_counts, parsed_files = scan_dataset_class_statistics(video_pairs)
    sampling_rates = compute_class_sampling_rates(global_counts, max_instances_per_class)

    # Class resolution & YAML setup
    global_names = []
    yaml_data = {}
    if yaml_path and Path(yaml_path).is_file():
        global_names, yaml_data = load_yaml_classes(Path(yaml_path))

    all_local_names = []
    for _, (local_names, _) in parsed_files.items():
        all_local_names.extend(local_names)

    resolution, global_names = build_class_resolution(
        all_local_names, global_names, manual_class_map, auto_create_new=auto_create_classes
    )

    # Prepare directories
    total_videos = len(video_pairs)
    total_saved_images = 0
    total_deleted_frames = 0
    total_skipped_bg = 0
    total_downsampled = 0

    # Split assignment for video clips if "auto"
    clip_split_map = {}
    if split_mode == "auto":
        shuffled_pairs = list(video_pairs)
        rng.shuffle(shuffled_pairs)
        n_total = len(shuffled_pairs)
        n_train = int(n_total * split_ratios[0])
        n_val = int(n_total * split_ratios[1])
        for idx, (vp, _) in enumerate(shuffled_pairs):
            if idx < n_train:
                clip_split_map[vp] = "train"
            elif idx < n_train + n_val:
                clip_split_map[vp] = "val"
            else:
                clip_split_map[vp] = "test"

    # Pass 2: Frame Extraction & Processing
    for v_idx, (video_path, txt_path) in enumerate(video_pairs):
        if cancel_callback and cancel_callback():
            yield 0, "Conversion cancelled by user."
            return

        pct = int(10 + (v_idx / total_videos) * 85)
        v_stem = video_path.stem
        msg = f"Processing video {v_idx + 1}/{total_videos}: {v_stem}"
        if progress_callback:
            progress_callback(pct, msg)
        yield pct, msg

        # Determine split subfolder
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

        local_names, frame_lines = parsed_files[video_path]
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % dist == 0 and frame_idx < len(frame_lines):
                parsed = parse_frame_line(frame_lines[frame_idx])

                if parsed == "DELETED;":
                    total_deleted_frames += 1
                elif parsed is not None:
                    is_background = (len(parsed) == 0)

                    # Background frame thinning
                    if is_background and bg_remove_percent > 0 and rng.random() * 100 < bg_remove_percent:
                        total_skipped_bg += 1
                        frame_idx += 1
                        continue

                    # Uniform Class Balancing Check
                    if not is_background and sampling_rates:
                        # Find minimum rate (highest priority class) in this frame
                        frame_rates = []
                        for (c_idx, _, _, _, _) in parsed:
                            if 0 <= c_idx < len(local_names):
                                c_name = local_names[c_idx].strip().lower()
                                frame_rates.append(sampling_rates.get(c_name, 1.0))
                        min_rate = min(frame_rates) if frame_rates else 1.0

                        # Sample with probability min_rate across whole dataset
                        if min_rate < 1.0 and rng.random() > min_rate:
                            total_downsampled += 1
                            frame_idx += 1
                            continue

                    # Letterbox/Pillarbox padding removal
                    img = frame
                    left_pad, top_pad = 0, 0
                    if remove_padding:
                        l, t, r, b = detect_content_bbox(img, black_thresh)
                        if (l, t, r, b) != (0, 0, img.shape[1], img.shape[0]):
                            img = img[t:b, l:r]
                            left_pad, top_pad = l, t

                    # Shift box coordinates if padding was removed
                    adjusted_boxes = []
                    if not is_background:
                        for (cls_id, bx, by, bw, bh) in parsed:
                            nbx = bx - left_pad
                            nby = by - top_pad
                            if nbx + bw > 0 and nby + bh > 0 and nbx < img.shape[1] and nby < img.shape[0]:
                                adjusted_boxes.append((cls_id, max(0.0, nbx), max(0.0, nby), bw, bh))

                    # Smart Cropping or Full Frame
                    crops_to_save = []
                    if enable_smart_crop and not is_background and (img.shape[1] > crop_size[0] or img.shape[0] > crop_size[1]):
                        crops_to_save = generate_smart_crops(
                            img, adjusted_boxes, local_names, resolution,
                            crop_w=crop_size[0], crop_h=crop_size[1],
                            min_visibility=min_visibility, context_padding=context_padding,
                            max_crops=max_crops_per_frame, min_box_size=min_box_size_px, rng=rng
                        )
                    else:
                        # Full frame mode
                        h_img, w_img = img.shape[0], img.shape[1]
                        label_lines = []
                        for (cls_id, bx, by, bw, bh) in adjusted_boxes:
                            if 0 <= cls_id < len(local_names):
                                name = local_names[cls_id].strip().lower()
                                g_idx = resolution.get(name)
                                if g_idx is not None and bw >= min_box_size_px and bh >= min_box_size_px:
                                    cx = (bx + bw / 2.0) / w_img
                                    cy = (by + bh / 2.0) / h_img
                                    nw = bw / w_img
                                    nh = bh / h_img
                                    label_lines.append(f"{g_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
                        crops_to_save = [(img, label_lines)]

                    # Save each crop and labels
                    for c_i, (crop_img, yolo_labels) in enumerate(crops_to_save):
                        crop_suffix = f"_c{c_i}" if len(crops_to_save) > 1 else ""
                        base_filename = f"{v_stem}_{frame_idx:06d}{crop_suffix}"
                        img_filename = f"{base_filename}{img_ext}"
                        txt_filename = f"{base_filename}.txt"

                        cv2.imwrite(str(dest_img_dir / img_filename), crop_img)
                        (dest_lbl_dir / txt_filename).write_text(
                            "\n".join(yolo_labels) + ("\n" if yolo_labels else ""), encoding="utf-8"
                        )
                        total_saved_images += 1

                        # Optional Horizontal Flip Augmentation
                        if flip_augment_percent > 0 and rng.random() * 100 < flip_augment_percent:
                            flipped_img = cv2.flip(crop_img, 1)
                            flipped_labels = []
                            for lbl_line in yolo_labels:
                                parts = lbl_line.split()
                                if len(parts) >= 5:
                                    gid, fcx, fcy, fnw, fnh = parts[0], float(parts[1]), parts[2], parts[3], parts[4]
                                    flipped_cx = 1.0 - fcx
                                    flipped_labels.append(f"{gid} {flipped_cx:.6f} {fcy} {fnw} {fnh}")

                            flip_base = f"{base_filename}_flip"
                            cv2.imwrite(str(dest_img_dir / f"{flip_base}{img_ext}"), flipped_img)
                            (dest_lbl_dir / f"{flip_base}.txt").write_text(
                                "\n".join(flipped_labels) + ("\n" if flipped_labels else ""), encoding="utf-8"
                            )
                            total_saved_images += 1

            frame_idx += 1

        cap.release()

    # Write final data.yaml
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
        f"Completed! {total_saved_images} images written, {total_deleted_frames} DELETED frames dropped, "
        f"{total_skipped_bg} empty frames skipped, {total_downsampled} frames balanced uniformly across {total_videos} videos."
    )
    if progress_callback:
        progress_callback(100, summary_msg)
    yield 100, summary_msg
