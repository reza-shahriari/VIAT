import os
import shutil
import random
from typing import Dict, Any

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


def execute_background_removal(
    d_path: str,
    percentage: float,
    action_type: str = "remove",  # "remove" or "move"
    progress_callback=None
) -> Dict[str, Any]:
    """
    Identifies background images (no labels or empty labels) in a dataset,
    and removes or moves a specified percentage of them.
    
    action_type:
      - "remove": os.remove
      - "move": move to removed_backgrounds/ folder
    """
    
    if progress_callback:
        progress_callback(0, 0, f"Scanning dataset {os.path.basename(d_path)} for background images...")
        
    info = _lazy_get_splits(d_path)
    
    backgrounds = []
    total_images = 0
    
    source_classes = info.classes
    
    # Identify background images
    for split in info.splits:
        fmt_name = split.label_format or info.label_format or "yolo"
        fmt = get_format(fmt_name)
        if not fmt:
            continue
            
        for img_path in _get_lazy_images(split.image_dir):
            total_images += 1
            if progress_callback and total_images % 50 == 0:
                progress_callback(total_images, 0, f"Scanning: {total_images} images checked...")
                
            label_path = fmt.find_label_file(img_path, split.label_dirs)
            is_background = False
            
            if not label_path or not os.path.isfile(label_path):
                is_background = True
            else:
                img_size = _image_size(img_path)
                if img_size:
                    try:
                        boxes = fmt.load(label_path, img_size, source_classes)
                        if not boxes:
                            is_background = True
                    except Exception:
                        pass
                else:
                    # If we can't read the image, we just skip it
                    pass
                    
            if is_background:
                backgrounds.append({
                    'img': img_path,
                    'lbl': label_path if label_path and os.path.isfile(label_path) else None,
                    'split': split
                })

    num_backgrounds = len(backgrounds)
    num_to_process = int(num_backgrounds * (percentage / 100.0))
    
    if num_to_process == 0:
        if progress_callback:
            progress_callback(total_images, total_images, "No background images to process.")
        return {
            'total_images': total_images,
            'total_backgrounds': num_backgrounds,
            'processed': 0,
            'action': action_type
        }
        
    # Randomly select subset
    random.shuffle(backgrounds)
    selected_backgrounds = backgrounds[:num_to_process]
    
    processed_count = 0
    
    # Create removed_backgrounds folder if moving
    moved_folder = os.path.join(d_path, "removed_backgrounds")
    if action_type == "move":
        os.makedirs(moved_folder, exist_ok=True)
        os.makedirs(os.path.join(moved_folder, "images"), exist_ok=True)
        os.makedirs(os.path.join(moved_folder, "labels"), exist_ok=True)
    
    # Process them
    for idx, bg in enumerate(selected_backgrounds):
        img_p = bg['img']
        lbl_p = bg['lbl']
        
        if progress_callback and idx % 10 == 0:
            progress_callback(idx, num_to_process, f"Processing {idx}/{num_to_process} backgrounds...")
            
        try:
            if action_type == "move":
                tgt_img = os.path.join(moved_folder, "images", os.path.basename(img_p))
                shutil.move(img_p, tgt_img)
                if lbl_p:
                    tgt_lbl = os.path.join(moved_folder, "labels", os.path.basename(lbl_p))
                    shutil.move(lbl_p, tgt_lbl)
            else:
                os.remove(img_p)
                if lbl_p:
                    os.remove(lbl_p)
            processed_count += 1
        except Exception:
            pass
            
    if progress_callback:
        progress_callback(num_to_process, num_to_process, f"Completed: {processed_count} backgrounds {action_type}d.")
        
    return {
        'total_images': total_images,
        'total_backgrounds': num_backgrounds,
        'processed': processed_count,
        'action': action_type
    }
