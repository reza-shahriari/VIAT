import os
import shutil
from typing import Dict, List, Tuple, Any

from utils.dataset_manager import DatasetInfo, _resolve_classes, _detect_layout_and_splits, IMAGE_EXTENSIONS
from utils.label_formats import get_format, all_formats

def _image_size(path):
    try:
        import cv2
        import numpy as np
        img_array = np.fromfile(path, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        h, w = img.shape[:2]
        return (w, h)
    except Exception:
        return None

def scan_main_folder(main_folder: str) -> List[str]:
    """Finds subdirectories within a main folder and sorts them alphabetically."""
    dataset_paths = []
    if not os.path.isdir(main_folder):
        return dataset_paths

    for entry in os.scandir(main_folder):
        if entry.is_dir():
            # We don't do deep scan_dataset here to save time
            dataset_paths.append(entry.path)
            
    # Sort alphabetically
    dataset_paths.sort()
    return dataset_paths

def _get_lazy_images(image_dir):
    if not os.path.isdir(image_dir):
        return
    try:
        for entry in os.scandir(image_dir):
            if entry.is_file() and entry.name.lower().endswith(IMAGE_EXTENSIONS):
                yield entry.path
    except OSError:
        pass

def _lazy_get_splits(d_path):
    info = DatasetInfo(root=d_path, layout="simple", splits=[])
    _resolve_classes(info)
    _detect_layout_and_splits(info)
    
    for split in info.splits:
        first_img = next(_get_lazy_images(split.image_dir), None)
        if not first_img:
            continue
            
        fmt_name = None
        img_size = _image_size(first_img)
        for name, fmt in all_formats():
            try:
                lp = fmt.find_label_file(first_img, split.label_dirs)
                if not lp: continue
                boxes = fmt.load(lp, img_size, info.classes)
                if boxes:
                    fmt_name = name
                    break
            except Exception:
                continue
        if not fmt_name:
            for name, fmt in all_formats():
                try:
                    lp = fmt.find_label_file(first_img, split.label_dirs)
                    if lp:
                        fmt_name = name
                        break
                except Exception:
                    continue
        split.label_format = fmt_name
        
    return info

def extract_class_samples(d_path: str, progress_callback=None) -> Dict[str, Dict[str, Any]]:
    """
    Extracts a sample image and bounding box for each unique class found in a single dataset.
    Returns:
        Dict mapping `class_name` to a dict containing:
            - 'dataset': the dataset path it was found in
            - 'img_path': path to the image
            - 'box': the bounding box dict
    """
    class_samples = {}
    
    if progress_callback:
        progress_callback(0, 0, f"Scanning {os.path.basename(d_path)}...")
        
    info = _lazy_get_splits(d_path)
    
    source_classes = info.classes
    needed_classes = set(source_classes)
    
    found_all = False
    for split in info.splits:
        if found_all:
            break
            
        fmt_name = split.label_format or info.label_format or "yolo"
        fmt = get_format(fmt_name)
        if not fmt:
            continue
            
        for img_path in _get_lazy_images(split.image_dir):
            label_path = fmt.find_label_file(img_path, split.label_dirs)
            if label_path and os.path.isfile(label_path):
                # --- FAST PRE-CHECK ---
                if fmt_name == "yolo" and source_classes:
                    target_indices_str = {str(i) for i, c in enumerate(source_classes) if c in needed_classes}
                    has_target = False
                    try:
                        with open(label_path, "r", encoding="utf-8", errors="replace") as f:
                            for line in f:
                                parts = line.strip().split()
                                if parts and parts[0] in target_indices_str:
                                    has_target = True
                                    break
                    except Exception:
                        pass
                    if not has_target:
                        continue
                        
                img_size = _image_size(img_path)
                if not img_size:
                    continue
                    
                try:
                    boxes = fmt.load(label_path, img_size, source_classes)
                    for b in boxes:
                        cls_name = b["class_name"]
                        if cls_name not in class_samples:
                            class_samples[cls_name] = {
                                "dataset": d_path,
                                "img_path": img_path,
                                "box": b
                            }
                            if cls_name in needed_classes:
                                needed_classes.remove(cls_name)
                                
                    if source_classes and not needed_classes:
                        found_all = True
                        break
                except Exception:
                    pass
                    
    if progress_callback:
        progress_callback(1, 1, "Scanning complete.")
        
    return class_samples

def execute_extraction(
    d_path: str,
    target_folder: str,
    target_class_name: str,
    selected_classes: List[str],
    progress_callback=None
) -> Dict[str, Any]:
    """
    Extracts selected classes from a single dataset and appends them to a new single-class dataset.
    """
    os.makedirs(target_folder, exist_ok=True)
    
    tgt_img_dir = os.path.join(target_folder, "images")
    tgt_lbl_dir = os.path.join(target_folder, "labels")
    os.makedirs(tgt_img_dir, exist_ok=True)
    os.makedirs(tgt_lbl_dir, exist_ok=True)
    
    # Calculate starting index based on existing files in target dir
    existing_images = [f for f in os.listdir(tgt_img_dir) if os.path.isfile(os.path.join(tgt_img_dir, f))]
    start_idx = len(existing_images)
    
    images_copied = 0
    labels_copied = 0
    skipped = 0
    
    d_name = os.path.basename(d_path)
    
    if progress_callback:
        progress_callback(0, 0, f"Initializing {d_name}...")
        
    info = _lazy_get_splits(d_path)
        
    current_image_idx = 0
    
    target_classes = [target_class_name]
    source_classes = info.classes
    
    for split in info.splits:
        fmt_name = split.label_format or info.label_format or "yolo"
        fmt = get_format(fmt_name)
        if not fmt:
            continue
            
        for img_path in _get_lazy_images(split.image_dir):
            current_image_idx += 1
            if progress_callback and current_image_idx % 10 == 0:
                progress_callback(current_image_idx, 0, f"Processing {os.path.basename(img_path)}")
                
            label_path = fmt.find_label_file(img_path, split.label_dirs)
            if not label_path or not os.path.isfile(label_path):
                skipped += 1
                continue
                
            # --- FAST PRE-CHECK ---
            if fmt_name == "yolo" and source_classes:
                target_indices_str = {str(i) for i, c in enumerate(source_classes) if c in selected_classes}
                has_target = False
                try:
                    with open(label_path, "r", encoding="utf-8", errors="replace") as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts and parts[0] in target_indices_str:
                                has_target = True
                                break
                except Exception:
                    pass
                if not has_target:
                    skipped += 1
                    continue
                    
            img_size = _image_size(img_path)
            if not img_size:
                skipped += 1
                continue
                
            try:
                boxes = fmt.load(label_path, img_size, source_classes)
                # Filter and remap boxes
                new_boxes = []
                for b in boxes:
                    if b["class_name"] in selected_classes:
                        new_b = b.copy()
                        new_b["class_name"] = target_class_name
                        new_b["class_index"] = 0
                        new_boxes.append(new_b)
                        
                if not new_boxes:
                    skipped += 1
                    continue
                    
                ext = os.path.splitext(img_path)[1]
                new_idx = start_idx + images_copied + 1
                # Format: original_folder_name_00001.jpg
                new_name = f"{d_name}_{new_idx:05d}{ext}"
                new_img_path = os.path.join(tgt_img_dir, new_name)
                new_lbl_stem = f"{d_name}_{new_idx:05d}"
                
                try:
                    shutil.copy2(img_path, new_img_path)
                except OSError:
                    skipped += 1
                    continue
                    
                content = fmt.dump(new_boxes, img_size, target_classes)
                new_lbl_path = os.path.join(tgt_lbl_dir, new_lbl_stem + fmt.extensions[0])
                with open(new_lbl_path, "w", encoding="utf-8") as f:
                    f.write(content)
                    
                images_copied += 1
                labels_copied += 1
            except Exception:
                skipped += 1
                
    # Update target data.yaml
    yaml_path = os.path.join(target_folder, "data.yaml")
    try:
        import yaml
        yaml_content = {
            "names": target_classes,
            "nc": 1,
            "path": target_folder,
            "train": "images",
            "val": "images",
            "test": ""
        }
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, sort_keys=False)
    except Exception:
        pass
        
    if progress_callback:
        progress_callback(1, 1, "Done")
        
    return {
        "images_copied": images_copied,
        "labels_copied": labels_copied,
        "skipped": skipped
    }
