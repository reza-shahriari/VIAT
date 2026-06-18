"""Dataset-level operations for VIAT.

  * move_frames_to          -- generalized: move image+label to any subfolder.
  * remove_bad_frames       -- alias for move_frames_to(dest="removed").
  * move_to_review_label    -- move to "review_label/" (CHANGE LABEL queue).
  * remove_grayscale_images -- detect & move grayscale images.
  * remove_duplicate_groups -- Roboflow .rf. dedup: keep 1 per group, move rest.
  * remove_class_and_images -- move all frames whose labels contain a given class.
  * remap_class             -- rename a class everywhere (memory + disk).
  * merge_classes           -- combine N classes into one.

All ops are format-aware (use the label_format plugins) and non-destructive
(files are MOVED, not deleted, so they're recoverable). Each op appends an
entry to DATASET_LOG.md when the logging module is available.
"""

import os
import shutil
import math
import random
from collections import defaultdict
from typing import Dict, List, Optional

from .label_formats import get_format

# Optional: dataset logging
try:
    from .dataset_log import append_dataset_log
except ImportError:
    def append_dataset_log(app, *a, **k):
        pass


# --------------------------------------------------------------------------- #
# Core: generalized move
# --------------------------------------------------------------------------- #


def move_frames_to(
    app,
    frame_indices: List[int],
    *,
    dest_subfolder: str = "removed",
    also_remove_from_disk: bool = True,
    log_operation: str = None,
    log_details: str = "",
) -> Dict:
    """Move the given frames' image + label files into ``<root>/<dest_subfolder>/``.

    Non-destructive: files are MOVED (not deleted), preserving split structure.
    In-memory state (image_files, frame_annotations, frame_to_split) is updated
    and reindexed.

    Args:
        app: the VideoAnnotationTool main window.
        frame_indices: list of frame indices to move.
        dest_subfolder: name of the subfolder under the dataset root.
        also_remove_from_disk: if False, only removes from in-memory state.
        log_operation: if set, append a row to DATASET_LOG.md with this name.
        log_details: extra detail string for the log entry.

    Returns:
        dict: moved_images, moved_labels, skipped, dest_dir
    """
    info = getattr(app, "_viat_dataset_info", None)
    frame_to_split = getattr(app, "_viat_frame_to_split", None)
    image_files = list(getattr(app, "image_files", []) or [])

    if not image_files:
        return {"moved_images": 0, "moved_labels": 0, "skipped": len(frame_indices), "dest_dir": None}

    root = info.root if info else os.path.dirname(os.path.commonpath(image_files))
    dest_root = os.path.join(root, dest_subfolder)
    os.makedirs(dest_root, exist_ok=True)

    moved_images = 0
    moved_labels = 0
    skipped = 0
    moved_set = set(frame_indices)

    for idx in sorted(set(frame_indices), reverse=True):
        if not (0 <= idx < len(image_files)):
            skipped += 1
            continue
        img_path = image_files[idx]
        split_name = frame_to_split[idx] if frame_to_split and idx < len(frame_to_split) else "root"

        dest_dir = os.path.join(dest_root, split_name)
        os.makedirs(dest_dir, exist_ok=True)

        # 1. Move image
        img_name = os.path.basename(img_path)
        dest_img = os.path.join(dest_dir, img_name)
        if also_remove_from_disk and os.path.isfile(img_path):
            try:
                if os.path.exists(dest_img):
                    base, ext = os.path.splitext(img_name)
                    dest_img = os.path.join(dest_dir, f"{base}_{idx}{ext}")
                shutil.move(img_path, dest_img)
                moved_images += 1
            except OSError:
                pass

        # 2. Move label
        label_path = _find_label_for_frame(app, info, idx)
        if label_path and also_remove_from_disk and os.path.isfile(label_path):
            lbl_name = os.path.basename(label_path)
            dest_lbl = os.path.join(dest_dir, lbl_name)
            try:
                if os.path.exists(dest_lbl):
                    base, ext = os.path.splitext(lbl_name)
                    dest_lbl = os.path.join(dest_dir, f"{base}_{idx}{ext}")
                shutil.move(label_path, dest_lbl)
                moved_labels += 1
            except OSError:
                pass

        # 3. Remove from in-memory state
        del image_files[idx]
        if hasattr(app, "frame_annotations") and idx in app.frame_annotations:
            del app.frame_annotations[idx]
        if frame_to_split and idx < len(frame_to_split):
            frame_to_split.pop(idx)

    # Reindex frame_annotations
    if hasattr(app, "frame_annotations"):
        old = dict(app.frame_annotations)
        app.frame_annotations = {}
        old_to_new = {}
        new_idx = 0
        for old_idx in range(len(image_files) + len(moved_set)):
            if old_idx in moved_set:
                continue
            old_to_new[old_idx] = new_idx
            new_idx += 1
        for old_idx, anns in old.items():
            if old_idx in old_to_new:
                app.frame_annotations[old_to_new[old_idx]] = anns

    app.image_files = image_files
    app.total_frames = len(image_files)

    result = {
        "moved_images": moved_images,
        "moved_labels": moved_labels,
        "skipped": skipped,
        "dest_dir": dest_root,
    }

    # Log
    if log_operation:
        detail = log_details or f"moved to {dest_subfolder}/"
        append_dataset_log(app, log_operation, affected=moved_images, details=detail)

    return result


# --------------------------------------------------------------------------- #
# Named wrappers (user-facing)
# --------------------------------------------------------------------------- #


def remove_bad_frames(app, frame_indices: List[int], **kwargs) -> Dict:
    """Alias: move frames to ``removed/``."""
    return move_frames_to(
        app, frame_indices, dest_subfolder="removed",
        log_operation="Removed bad frames",
        **kwargs,
    )


def move_to_review_label(app, frame_indices: List[int], **kwargs) -> Dict:
    """Move frames to ``review_label/`` (the "CHANGE LABEL" queue)."""
    return move_frames_to(
        app, frame_indices, dest_subfolder="review_label",
        log_operation="Moved to review_label",
        log_details="marked for label review (CHANGE LABEL)",
        **kwargs,
    )


def move_to_removed(app, frame_indices: List[int], **kwargs) -> Dict:
    """Move frames to ``removed/``."""
    return move_frames_to(
        app, frame_indices, dest_subfolder="removed",
        log_operation="Removed frames",
        **kwargs,
    )


def _find_label_for_frame(app, info, frame_idx: int) -> Optional[str]:
    """Locate the on-disk label file for a frame using the format plugin."""
    if not info or frame_idx >= len(app.image_files):
        return None
    split_name = (
        app._viat_frame_to_split[frame_idx]
        if hasattr(app, "_viat_frame_to_split") and frame_idx < len(app._viat_frame_to_split)
        else None
    )
    split = None
    for s in info.splits:
        if s.name == split_name:
            split = s
            break
    if split is None and info.splits:
        split = info.splits[0]
    if split is None:
        return None
    fmt_name = split.label_format or info.label_format or "yolo"
    fmt = get_format(fmt_name)
    if fmt is None:
        return None
    try:
        return fmt.find_label_file(app.image_files[frame_idx], split.label_dirs)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Remap / merge classes
# --------------------------------------------------------------------------- #


def remap_class(
    app,
    old_name: str,
    new_name: str,
    *,
    rewrite_disk: bool = True,
    merge_color: bool = True,
) -> Dict:
    """Rename ``old_name`` -> ``new_name`` across the whole dataset.

    Updates:
      * in-memory annotations (all frames)
      * canvas.class_colors / class_attributes
      * class-list files (data.yaml names + classes.txt) if rewrite_disk
      * every label file on disk (via the format plugin) if rewrite_disk

    If ``new_name`` already exists, this effectively merges the two classes.
    """
    info = getattr(app, "_viat_dataset_info", None)
    changed_frames = 0
    changed_boxes = 0

    # 1. In-memory annotations
    for fidx, anns in getattr(app, "frame_annotations", {}).items():
        touched = False
        for ann in anns:
            if getattr(ann, "class_name", None) == old_name:
                ann.class_name = new_name
                if merge_color and new_name in getattr(app.canvas, "class_colors", {}):
                    ann.color = app.canvas.class_colors[new_name]
                changed_boxes += 1
                touched = True
        if touched:
            changed_frames += 1

    # 2. canvas.class_colors / class_attributes
    colors = getattr(app.canvas, "class_colors", {}) or {}
    attrs = getattr(app.canvas, "class_attributes", {}) or {}
    if old_name in colors:
        if new_name not in colors:
            colors[new_name] = colors.pop(old_name)
        else:
            colors.pop(old_name, None)
    if old_name in attrs:
        if new_name not in attrs:
            attrs[new_name] = attrs.pop(old_name)
        else:
            attrs.pop(old_name, None)
    app.canvas.class_colors = colors
    app.canvas.class_attributes = attrs
    app.class_attributes = attrs  # legacy alias

    # 3. Capture ORIGINAL classes (before rename) for on-disk rewrite,
    #    then update DatasetInfo.classes to the new list.
    original_classes = list(info.classes) if info is not None else []
    if info is not None:
        info.classes = [new_name if c == old_name else c for c in info.classes]
        # dedupe while preserving order
        seen = set()
        deduped = []
        for c in info.classes:
            if c not in seen:
                seen.add(c)
                deduped.append(c)
        info.classes = deduped

    # 4. Refresh current canvas display if we're on a touched frame
    if hasattr(app, "current_frame") and app.current_frame in getattr(app, "frame_annotations", {}):
        app.canvas.annotations = app.frame_annotations[app.current_frame]
        app.canvas.update()
    if hasattr(app, "refresh_class_ui"):
        app.refresh_class_ui()

    # 5. Rewrite on-disk files. We pass the ORIGINAL classes so the loader
    #    decodes class indices into OLD names, applies the mapping, then
    #    re-serializes with the NEW classes list.
    disk_rewritten = 0
    if rewrite_disk and info is not None:
        disk_rewritten = _rewrite_class_on_disk(
            app, info, {old_name: new_name}, original_classes=original_classes
        )

    result = {
        "changed_frames": changed_frames,
        "changed_boxes": changed_boxes,
        "disk_files_rewritten": disk_rewritten,
    }

    append_dataset_log(
        app, "Remapped class",
        affected=changed_boxes,
        details=f"'{old_name}' -> '{new_name}' ({disk_files_rewritten} files rewritten)",
    )
    return result


def merge_classes(app, old_names: List[str], new_name: str, *, rewrite_disk: bool = True) -> Dict:
    """Merge several classes into one (e.g. ['car','truck'] -> 'vehicle')."""
    total = {"changed_frames": 0, "changed_boxes": 0, "disk_files_rewritten": 0}
    for old in old_names:
        if old == new_name:
            continue
        r = remap_class(app, old, new_name, rewrite_disk=rewrite_disk)
        total["changed_frames"] = max(total["changed_frames"], r["changed_frames"])
        total["changed_boxes"] += r["changed_boxes"]
        total["disk_files_rewritten"] += r["disk_files_rewritten"]
    append_dataset_log(
        app, "Merged classes",
        affected=total["changed_boxes"],
        details=f"{old_names} -> '{new_name}'",
    )
    return total


# --------------------------------------------------------------------------- #
# Grayscale image removal
# --------------------------------------------------------------------------- #


def is_grayscale(path: str) -> bool:
    """Return True if the image at *path* is grayscale (R==G==B everywhere).

    Uses a 32x32 downsampled RGB check for speed.
    """
    try:
        import cv2
        import numpy as np
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return False
        # Downsample for speed
        h, w = img.shape[:2]
        if h > 32 or w > 32:
            img = cv2.resize(img, (32, 32))
        # cv2 is BGR; check all channels equal
        b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        return bool(np.all(b == g) and np.all(g == r))
    except Exception:
        return False


def remove_grayscale_images(app, *, dest_subfolder: str = "removed/grayscale") -> Dict:
    """Detect and move all grayscale images to ``<root>/removed/grayscale/``.

    Scans every image in app.image_files, moves grayscale ones (image + label)
    using the generalized move_frames_to. Non-destructive.

    Returns:
        dict: moved_images, grayscale_indices (list), dest_dir
    """
    image_files = list(getattr(app, "image_files", []) or [])
    gray_indices = []
    for i, img_path in enumerate(image_files):
        if is_grayscale(img_path):
            gray_indices.append(i)

    if not gray_indices:
        append_dataset_log(app, "Removed grayscale", affected=0, details="none found")
        return {"moved_images": 0, "grayscale_indices": [], "dest_dir": None}

    result = move_frames_to(
        app, gray_indices,
        dest_subfolder=dest_subfolder,
        log_operation="Removed grayscale images",
        log_details=f"{len(gray_indices)} grayscale -> {dest_subfolder}/",
    )
    result["grayscale_indices"] = gray_indices
    return result


# --------------------------------------------------------------------------- #
# Roboflow duplicate removal (.rf. dedup)
# --------------------------------------------------------------------------- #


def _roboflow_base_name(filename: str) -> str:
    """Extract the base name for grouping Roboflow duplicates.

    Roboflow augmentations share a stem before ``.rf.``:
        image_001.rf.abc123.jpg  ->  image_001
        image_001.jpg            ->  image_001
    """
    name = os.path.basename(filename)
    stem = os.path.splitext(name)[0]
    if ".rf." in stem:
        stem = stem.split(".rf.")[0]
    return stem


def remove_duplicate_groups(
    app,
    *,
    dest_subfolder: str = "removed/duplicates",
    keep: str = "random",
) -> Dict:
    """Remove Roboflow duplicate groups, keeping one per group.

    Groups images by their Roboflow base name (everything before ``.rf.``).
    For each group with >1 image, keeps one and moves the rest to
    ``<root>/removed/duplicates/``.

    Args:
        app: main window.
        dest_subfolder: where to move duplicates.
        keep: "random" (keep a random one) or "first" (keep the first by name).

    Returns:
        dict: moved_images, groups_processed, kept_per_group, dest_dir
    """
    image_files = list(getattr(app, "image_files", []) or [])

    groups = defaultdict(list)
    for i, img_path in enumerate(image_files):
        base = _roboflow_base_name(img_path)
        groups[base].append(i)

    to_move = []
    groups_processed = 0
    kept = {}
    for base, indices in groups.items():
        if len(indices) <= 1:
            continue
        groups_processed += 1
        if keep == "random":
            keep_idx = random.choice(indices)
        else:
            keep_idx = indices[0]
        kept[base] = keep_idx
        for idx in indices:
            if idx != keep_idx:
                to_move.append(idx)

    if not to_move:
        append_dataset_log(app, "Removed duplicates", affected=0, details="no groups found")
        return {"moved_images": 0, "groups_processed": 0, "kept_per_group": {}, "dest_dir": None}

    result = move_frames_to(
        app, to_move,
        dest_subfolder=dest_subfolder,
        log_operation="Removed Roboflow duplicates",
        log_details=f"{len(to_move)} duplicates from {groups_processed} groups -> {dest_subfolder}/",
    )
    result["groups_processed"] = groups_processed
    result["kept_per_group"] = kept
    return result


# --------------------------------------------------------------------------- #
# Remove class + all images containing that class
# --------------------------------------------------------------------------- #


def remove_class_and_images(
    app,
    class_names: List[str],
    *,
    dest_subfolder: str = "removed/class_filtered",
) -> Dict:
    """Move all frames whose labels contain any of *class_names* to a subfolder.

    Scans in-memory frame_annotations for any annotation whose class_name is in
    the given list. Moves those frames (image + label) to
    ``<root>/removed/class_filtered/``.

    Args:
        app: main window.
        class_names: list of class names to remove (e.g. ["car", "truck"]).
        dest_subfolder: where to move the frames.

    Returns:
        dict: moved_images, matched_frames (list), dest_dir
    """
    target = set(class_names)
    matched = []
    for fidx, anns in getattr(app, "frame_annotations", {}).items():
        for ann in anns:
            if getattr(ann, "class_name", None) in target:
                matched.append(fidx)
                break

    if not matched:
        append_dataset_log(
            app, "Removed class+images", affected=0,
            details=f"no frames matched classes {class_names}",
        )
        return {"moved_images": 0, "matched_frames": [], "dest_dir": None}

    result = move_frames_to(
        app, matched,
        dest_subfolder=dest_subfolder,
        log_operation="Removed class + images",
        log_details=f"classes={list(target)}, {len(matched)} frames -> {dest_subfolder}/",
    )
    result["matched_frames"] = matched
    return result


# --------------------------------------------------------------------------- #
# Auto-import YOLO detections
# --------------------------------------------------------------------------- #


def auto_import_detections(
    app,
    json_path: str,
    *,
    move_to_review: bool = True,
    add_as_annotations: bool = False,
    review_subfolder: str = "review_label",
    bbox_cls=None,
) -> Dict:
    """Import a YOLO-detections JSON and move flagged images to review_label/.

    Reads the JSON produced by yolo_detect.py. For each image that has
    detections:
      1. Moves the image + its existing label to review_label/.
      2. (optionally) Copies the detections JSON to review_label/ so the user
         can import them when reviewing.

    Note: detections are NOT added to the main frame_annotations, because the
    images are moved OUT of the main dataset. The detections JSON is preserved
    alongside the moved images so the user can import it when they open the
    review_label/ folder.

    Args:
        app: the VideoAnnotationTool main window.
        json_path: path to the _viat_detections.json file.
        move_to_review: if True, move flagged images to review_label/.
        add_as_annotations: if True, add YOLO detections to frame_annotations
            for images that AREN'T moved (rarely useful). Default False.
        review_subfolder: where to move flagged images.
        bbox_cls: the BoundingBox class (for constructing annotations).

    Returns:
        dict: {flagged_images, moved_images, annotations_added, total_detections,
               json_copied_to}
    """
    import json
    import shutil as _shutil

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # The JSON has { "_metadata": {...}, "frames": { "0000": {"actors": {...}} } }
    metadata = data.get("_metadata", {})
    frames = data.get("frames", data)  # fallback: top-level is frames

    # Map: image file path -> detections
    image_files_list = metadata.get("image_files", [])
    flagged_paths = []
    detections_by_path = {}
    total_detections = 0

    for frame_key, frame_data in frames.items():
        try:
            frame_idx = int(frame_key)
        except (ValueError, TypeError):
            continue

        actors = frame_data.get("actors", {}) if isinstance(frame_data, dict) else {}
        if not actors:
            continue

        # Find the image path for this frame index
        if frame_idx < len(image_files_list):
            img_path = image_files_list[frame_idx]
        else:
            continue

        flagged_paths.append(img_path)
        detections_by_path[img_path] = list(actors.values())
        total_detections += len(actors)

    if not flagged_paths:
        append_dataset_log(
            app, "Auto-import detections", affected=0,
            details="no detections found in JSON",
        )
        return {
            "flagged_images": 0,
            "moved_images": 0,
            "annotations_added": 0,
            "total_detections": 0,
            "json_copied_to": None,
        }

    # Find frame indices in app.image_files that match flagged paths
    image_files = list(getattr(app, "image_files", []) or [])
    flagged_abspaths = {os.path.abspath(p) for p in flagged_paths}
    flagged_indices = []
    for idx, img_path in enumerate(image_files):
        if os.path.abspath(img_path) in flagged_abspaths:
            flagged_indices.append(idx)

    # Move flagged images to review_label/
    moved_images = 0
    review_dir = None
    if move_to_review and flagged_indices:
        result = move_to_review_label(app, flagged_indices)
        moved_images = result["moved_images"]
        review_dir = result.get("dest_dir")

    # Copy the detections JSON into review_label/ so it travels with the images
    json_copied_to = None
    if review_dir and os.path.isdir(review_dir):
        json_dest = os.path.join(review_dir, "_viat_detections.json")
        try:
            _shutil.copy2(json_path, json_dest)
            json_copied_to = json_dest
        except OSError:
            pass

    # Optionally add annotations (only for images that weren't moved, which
    # is rare since all flagged images get moved). This is a no-op when
    # move_to_review=True (default).
    annotations_added = 0
    if add_as_annotations and bbox_cls is not None:
        from PyQt5.QtCore import QRect
        from PyQt5.QtGui import QColor
        import random

        existing_colors = dict(getattr(app.canvas, "class_colors", {}) or {})
        if not hasattr(app.canvas, "class_attributes") or app.canvas.class_attributes is None:
            app.canvas.class_attributes = {}

        new_image_files = list(getattr(app, "image_files", []) or [])
        path_to_new_idx = {}
        for new_idx, img_path in enumerate(new_image_files):
            path_to_new_idx[os.path.abspath(img_path)] = new_idx

        for img_path, dets in detections_by_path.items():
            abs_path = os.path.abspath(img_path)
            if abs_path not in path_to_new_idx:
                continue  # image was moved out
            new_idx = path_to_new_idx[abs_path]

            for det_idx, det in enumerate(dets):
                class_name = det.get("class", "unknown")
                bbox = det.get("bbox", [0, 0, 0, 0])
                if len(bbox) != 4:
                    continue
                x, y, w, h = bbox

                if class_name not in existing_colors:
                    existing_colors[class_name] = QColor(
                        random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                    )
                if class_name not in app.canvas.class_attributes:
                    app.canvas.class_attributes[class_name] = {
                        "Size": {"type": "int", "default": -1, "min": 0, "max": 100},
                        "Quality": {"type": "int", "default": -1, "min": 0, "max": 100},
                    }
                color = existing_colors[class_name]

                rect = QRect(int(x), int(y), max(1, int(w)), max(1, int(h)))
                ann = bbox_cls(
                    rect=rect,
                    class_name=class_name,
                    attributes={"actor_id": f"yolo_{det_idx}"},
                    color=color,
                    source="detected",
                    score=float(det.get("score", 1.0)),
                )
                ann.verified = False

                if new_idx not in app.frame_annotations:
                    app.frame_annotations[new_idx] = []
                app.frame_annotations[new_idx].append(ann)
                annotations_added += 1

        app.canvas.class_colors = existing_colors
        app.class_attributes = app.canvas.class_attributes

    append_dataset_log(
        app, "Auto-import YOLO detections", affected=moved_images,
        details=f"{total_detections} detections in {len(flagged_paths)} images -> {review_subfolder}/",
    )

    return {
        "flagged_images": len(flagged_paths),
        "moved_images": moved_images,
        "annotations_added": annotations_added,
        "total_detections": total_detections,
    }


def _rewrite_class_on_disk(app, info, mapping: Dict[str, str], original_classes: List[str] = None) -> int:
    """Rewrite label files on disk applying a class-name mapping.

    For per-image formats (YOLO, Pascal VOC): rewrite each file.
    For dataset-wide formats (COCO): rewrite the categories + annotations.

    original_classes: the class list BEFORE the remap. Used to decode the
    existing label files (their class indices map to old names). If None,
    falls back to info.classes (which has already been remapped -- this only
    works if no actual rename happened).
    """
    from .label_formats.coco import CocoLabelFormat

    rewritten = 0
    image_files = getattr(app, "image_files", []) or []
    frame_to_split = getattr(app, "_viat_frame_to_split", []) or []

    # original_classes = how the on-disk files are currently indexed.
    # new_classes = what we want the output files to use.
    old_classes = list(original_classes) if original_classes else list(info.classes)
    new_classes = list(info.classes)

    for fidx, img_path in enumerate(image_files):
        split_name = frame_to_split[fidx] if fidx < len(frame_to_split) else None
        split = next((s for s in info.splits if s.name == split_name), None)
        if split is None:
            continue
        fmt_name = split.label_format or info.label_format or "yolo"
        fmt = get_format(fmt_name)
        if fmt is None:
            continue

        # COCO dataset-wide: handled once per split, not per image.
        if isinstance(fmt, CocoLabelFormat):
            continue

        label_path = _find_label_for_frame(app, info, fidx)
        if not label_path or not os.path.isfile(label_path):
            continue

        # Load existing boxes using the OLD class names so we can detect
        # which ones need remapping.
        try:
            img_size = _img_size(img_path)
            if img_size is None:
                continue
            boxes = fmt.load(label_path, img_size, old_classes)
        except Exception:
            continue

        changed = False
        for b in boxes:
            if b["class_name"] in mapping:
                b["class_name"] = mapping[b["class_name"]]
                changed = True
        if not changed:
            continue

        try:
            content = fmt.dump(boxes, img_size, new_classes)
            with open(label_path, "w", encoding="utf-8") as f:
                f.write(content)
            rewritten += 1
        except Exception:
            continue

    # COCO dataset-wide rewrite
    for split in info.splits:
        fmt_name = split.label_format or info.label_format or "yolo"
        if fmt_name != "coco":
            continue
        coco = get_format("coco")
        coco.discover(split.label_dirs)
        if coco._file and os.path.isfile(coco._file):
            try:
                _rewrite_coco_categories(coco._file, mapping)
                rewritten += 1
            except Exception:
                pass

    # Update classes.txt / data.yaml
    _rewrite_class_files(info, mapping)

    return rewritten


def _rewrite_coco_categories(coco_path: str, mapping: Dict[str, str]) -> None:
    import json

    with open(coco_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for cat in data.get("categories", []):
        if cat.get("name") in mapping:
            cat["name"] = mapping[cat["name"]]
    # Note: annotations reference category_id, not name, so no annotation
    # changes are needed for a pure rename. For a merge where two categories
    # collapse, you'd also need to remap annotation.category_id -- left as a
    # follow-up since COCO merges are rare in Roboflow exports.
    with open(coco_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _rewrite_class_files(info, mapping: Dict[str, str]) -> None:
    """Rewrite classes.txt / obj.names / data.yaml names after a remap."""
    # classes.txt / obj.names
    for name in ("classes.txt", "obj.names", "labels.txt"):
        p = os.path.join(info.root, name)
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    lines = [l.rstrip("\n") for l in f]
                new_lines = [mapping.get(l.strip(), l.strip()) for l in lines if l.strip()]
                # dedupe
                seen = set()
                deduped = []
                for l in new_lines:
                    if l not in seen:
                        seen.add(l)
                        deduped.append(l)
                with open(p, "w", encoding="utf-8") as f:
                    f.write("\n".join(deduped) + "\n")
            except OSError:
                pass

    # data.yaml (best-effort: rewrite the names: block)
    for name in ("data.yaml", "data.yml"):
        p = os.path.join(info.root, name)
        if not os.path.isfile(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            continue
        out = []
        in_names = False
        for line in lines:
            s = line.strip()
            if s.startswith("names:"):
                in_names = True
                # rebuild as inline list from info.classes
                out.append("names: [" + ", ".join(info.classes) + "]\n")
                continue
            if in_names:
                if s.startswith("- ") or re.match_num_colon(s):
                    continue  # drop old list entries
                in_names = False
            out.append(line)
        try:
            with open(p, "w", encoding="utf-8") as f:
                f.writelines(out)
        except OSError:
            pass


def re_match_num_colon(s):
    import re

    return re.match(r"^\d+\s*:", s)


def _img_size(path):
    try:
        import cv2

        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        h, w = img.shape[:2]
        return (w, h)
    except Exception:
        return None
