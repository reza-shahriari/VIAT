"""
Dataset merger for VIAT.

Merges multiple image datasets into a single target dataset:

  * Reads the target dataset's data.yaml to get the class list.
  * Auto-maps classes by name; asks the user for unmatched classes.
  * Options for split assignment:
    - Keep original splits (train/valid/test)
    - Random split (user specifies %)
    - All in one split (e.g. everything in train)
  * Renames files: <dataset_name>_<frame_number>.<ext>
    - dataset_name defaults to the source folder name, user can override
    - frame_number is zero-padded (001, 002, ...)
  * Copies images + rewrites labels (using the label_format plugins) with
    the new class indices (from the target data.yaml).

Usage from main.py:
    from utils.dataset_merger import merge_dataset_into_target
    result = merge_dataset_into_target(
        app, source_folder, target_folder,
        dataset_name="my_dataset",
        split_mode="keep",  # "keep" | "random" | "all_train"
        class_mapping=None,  # auto + user-provided
    )
"""

import os
import shutil
import random
from typing import Dict, List, Optional, Tuple

from .dataset_manager import scan_dataset, DatasetInfo
from .label_formats import get_format


def merge_dataset_into_target(
    app,
    source_folder: str,
    target_folder: str,
    *,
    dataset_name: str = None,
    split_mode: str = "keep",  # "keep" | "random" | "all_train" | "all_valid" | "all_test"
    random_valid_pct: int = 10,
    class_mapping: Dict[str, str] = None,
    progress_callback=None,
) -> Dict:
    """Merge a source dataset into a target dataset folder.

    Args:
        app: the VideoAnnotationTool main window (for bbox_cls).
        source_folder: path to the source dataset to merge IN.
        target_folder: path to the target (main) dataset.
        dataset_name: name prefix for renamed files. If None, defaults to
            the source folder name.
        split_mode: how to assign splits:
            "keep"       -- keep original train/valid/test
            "random"     -- random split (random_valid_pct % to valid, rest train)
            "all_train"  -- everything in train
            "all_valid"  -- everything in valid
            "all_test"   -- everything in test
        random_valid_pct: percentage for valid when split_mode="random".
        class_mapping: {source_class: target_class}. If None, auto-maps by
            name; unmatched classes are returned for user resolution.
        progress_callback: optional callable(current, total, message).

    Returns:
        dict: {images_copied, labels_copied, classes_mapped,
               unmatched_classes, skipped}
    """
    if dataset_name is None:
        dataset_name = os.path.basename(source_folder)

    # Scan source
    source_info = scan_dataset(source_folder)
    if source_info.image_count == 0:
        return {"images_copied": 0, "labels_copied": 0, "error": "No images in source"}

    # Scan target (to get the class list + structure)
    target_info = scan_dataset(target_folder)

    # Get target classes (from data.yaml or classes.txt)
    target_classes = target_info.classes
    if not target_classes:
        # If target has no classes, use the source's classes
        target_classes = source_info.classes

    # Auto-map classes by name
    source_classes = source_info.classes
    if class_mapping is None:
        class_mapping = {}
        unmatched = []
        for src_cls in source_classes:
            if src_cls in target_classes:
                class_mapping[src_cls] = src_cls
            else:
                unmatched.append(src_cls)
    else:
        unmatched = [c for c in source_classes if c not in class_mapping]

    # If there are unmatched classes, we can't proceed -- the caller must
    # resolve them first (show a dialog asking the user).
    if unmatched:
        return {
            "images_copied": 0,
            "labels_copied": 0,
            "error": "unmatched_classes",
            "unmatched_classes": unmatched,
            "target_classes": target_classes,
        }

    # Build target class index (name -> index) for label rewriting
    target_class_index = {name: i for i, name in enumerate(target_classes)}

    # Determine target structure
    # If split_mode is anything other than "keep", ALWAYS create split folders
    # (train/, valid/, test/). If "keep" and target has splits, use them.
    # If "keep" and target has no splits, use flat structure.
    target_has_splits = (
        split_mode != "keep"
        or any(s.name in ("train", "valid", "test") for s in target_info.splits)
    )

    images_copied = 0
    labels_copied = 0
    skipped = 0
    total = source_info.image_count

    for src_split in source_info.splits:
        fmt_name = src_split.label_format or source_info.label_format or "yolo"
        fmt = get_format(fmt_name)
        if fmt is None:
            skipped += len(src_split.images)
            continue

        for i, img_path in enumerate(src_split.images):
            if progress_callback:
                progress_callback(images_copied + skipped, total, f"Merging {os.path.basename(img_path)}")

            # Determine target split
            if split_mode == "keep":
                tgt_split = src_split.name if src_split.name in ("train", "valid", "test") else "train"
            elif split_mode == "random":
                tgt_split = "valid" if random.random() * 100 < random_valid_pct else "train"
            elif split_mode == "all_train":
                tgt_split = "train"
            elif split_mode == "all_valid":
                tgt_split = "valid"
            elif split_mode == "all_test":
                tgt_split = "test"
            else:
                tgt_split = "train"

            # Target directories
            if target_has_splits:
                tgt_img_dir = os.path.join(target_folder, tgt_split, "images")
                tgt_lbl_dir = os.path.join(target_folder, tgt_split, "labels")
            else:
                tgt_img_dir = os.path.join(target_folder, "images")
                tgt_lbl_dir = os.path.join(target_folder, "labels")
            os.makedirs(tgt_img_dir, exist_ok=True)
            os.makedirs(tgt_lbl_dir, exist_ok=True)

            # New filename: datasetname_frame001.jpg
            ext = os.path.splitext(img_path)[1]
            new_name = f"{dataset_name}_{images_copied + 1:04d}{ext}"
            new_img_path = os.path.join(tgt_img_dir, new_name)
            new_lbl_stem = f"{dataset_name}_{images_copied + 1:04d}"

            # Copy image
            try:
                shutil.copy2(img_path, new_img_path)
                images_copied += 1
            except OSError:
                skipped += 1
                continue

            # Find + rewrite label
            label_path = fmt.find_label_file(img_path, src_split.label_dirs)
            if label_path and os.path.isfile(label_path):
                try:
                    # Get image size for YOLO de/normalization
                    img_size = _image_size(img_path)
                    if img_size is None:
                        skipped += 1
                        continue

                    # Load boxes using SOURCE classes
                    boxes = fmt.load(label_path, img_size, source_classes)

                    # Apply class mapping + reindex
                    for b in boxes:
                        src_cls = b["class_name"]
                        tgt_cls = class_mapping.get(src_cls, src_cls)
                        b["class_name"] = tgt_cls
                        b["class_index"] = target_class_index.get(tgt_cls, 0)

                    # Write with TARGET classes
                    content = fmt.dump(boxes, img_size, target_classes)
                    new_lbl_path = os.path.join(tgt_lbl_dir, new_lbl_stem + fmt.extensions[0])
                    with open(new_lbl_path, "w", encoding="utf-8") as f:
                        f.write(content)
                    labels_copied += 1
                except Exception:
                    pass

    # Update target data.yaml if it exists
    _update_target_yaml(target_folder, target_classes)

    if progress_callback:
        progress_callback(total, total, "Done")

    return {
        "images_copied": images_copied,
        "labels_copied": labels_copied,
        "classes_mapped": len(class_mapping),
        "skipped": skipped,
        "dataset_name": dataset_name,
    }


def _image_size(path):
    try:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        h, w = img.shape[:2]
        return (w, h)
    except Exception:
        return None


def _update_target_yaml(target_folder, classes):
    """Update or create data.yaml in the target folder with the full class list."""
    yaml_path = os.path.join(target_folder, "data.yaml")
    # Read existing
    existing = {}
    if os.path.isfile(yaml_path):
        try:
            import yaml
            with open(yaml_path, "r", encoding="utf-8") as f:
                existing = yaml.safe_load(f) or {}
        except ImportError:
            pass
        except Exception:
            pass

    # Merge classes (add any new ones)
    existing_classes = existing.get("names", [])
    if isinstance(existing_classes, list):
        all_classes = list(existing_classes)
        for c in classes:
            if c not in all_classes:
                all_classes.append(c)
    else:
        all_classes = list(classes)

    # Write
    try:
        import yaml
        existing["names"] = all_classes
        existing["nc"] = len(all_classes)
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(existing, f, default_flow_style=False)
    except ImportError:
        # Fallback: write a simple YAML
        with open(yaml_path, "w", encoding="utf-8") as f:
            f.write(f"path: .\nnc: {len(all_classes)}\n")
            f.write("names: [" + ", ".join(all_classes) + "]\n")


def find_unmatched_classes(source_folder: str, target_folder: str) -> Dict:
    """Pre-check: find which source classes don't match target classes.

    Returns {matched: {src: tgt}, unmatched: [src_classes], target_classes: [...]}
    """
    source_info = scan_dataset(source_folder)
    target_info = scan_dataset(target_folder)
    target_classes = target_info.classes or source_info.classes

    matched = {}
    unmatched = []
    for src_cls in source_info.classes:
        if src_cls in target_classes:
            matched[src_cls] = src_cls
        else:
            unmatched.append(src_cls)

    return {
        "matched": matched,
        "unmatched": unmatched,
        "target_classes": target_classes,
        "source_classes": source_info.classes,
    }
