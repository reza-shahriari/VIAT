import os
from typing import Dict, List, Tuple, Any, Optional

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

def hex_to_bgr(hex_color: str) -> Tuple[int, int, int]:
    """Convert hex color string to BGR tuple."""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])

def bgr_to_hex(bgr_color: Tuple[int, int, int]) -> str:
    """Convert BGR tuple to hex color string."""
    return f"#{bgr_color[2]:02X}{bgr_color[1]:02X}{bgr_color[0]:02X}"

def scan_mask_colors(
    rgb_files: List[str],
    masks_dir: str,
    ignore_hex: Optional[str] = "#000000",
    progress_callback=None
) -> List[Tuple[int, int, int]]:
    """
    Scans segmentation masks to find all unique colors.
    Returns a list of unique BGR color tuples.
    """
    if cv2 is None or np is None:
        raise ImportError("OpenCV and numpy are required to import masks.")

    ignore_bgr = hex_to_bgr(ignore_hex) if ignore_hex else None
    unique_colors_set = set()

    # Find all mask files in the masks_dir
    mask_files = []
    if os.path.exists(masks_dir):
        for f in os.listdir(masks_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                mask_files.append(os.path.join(masks_dir, f))

    for i, mask_path in enumerate(mask_files):
        if progress_callback:
            # Note: progress might slightly mismatch if called from UI expecting rgb_files length, but it's fine
            progress_callback(i, max(len(mask_files), 1))

        mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask_img is None:
            continue
            
        # Fast check: instantly skip images that are completely black
        if not np.any(mask_img):
            continue

        # Determine unique colors
        if len(mask_img.shape) == 2:
            unique_colors = np.unique(mask_img)
            unique_colors_bgr = [(int(val), int(val), int(val)) for val in unique_colors]
        else:
            mask_img_bgr = mask_img[:, :, :3]
            pixels = mask_img_bgr.reshape(-1, 3)
            unique_colors = np.unique(pixels, axis=0)
            unique_colors_bgr = [(int(c[0]), int(c[1]), int(c[2])) for c in unique_colors]

        for color_bgr in unique_colors_bgr:
            if ignore_bgr is not None and tuple(color_bgr) == ignore_bgr:
                continue
            unique_colors_set.add(tuple(color_bgr))
            
        # Break early once we actually find non-ignored colors, in case the first frames are purely background
        if len(unique_colors_set) > 0:
            break

    if progress_callback:
        progress_callback(len(mask_files), max(len(mask_files), 1))

    return list(unique_colors_set)

def _mask_to_boxes(
    mask: "np.ndarray", 
    min_area: int, 
    simplify: int = 10, 
    merge_all: bool = True
) -> List[Dict[str, Any]]:
    """Extract contours from a binary mask and compute bbox + polygon.

    Returns an empty list if no contour meets the min_area threshold.
    If merge_all is True, returns a single bounding box containing all valid contours.
    If merge_all is False, returns a list of bounding boxes, one for each valid contour.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []

    # Keep only contours above min_area
    valid = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not valid:
        return []

    if merge_all:
        # Merge all valid contours' bounding box
        all_points = np.vstack(valid)
        x, y, w, h = cv2.boundingRect(all_points)

        # Build a simplified polygon from the largest contour
        largest = max(valid, key=cv2.contourArea)
        epsilon = simplify
        approx = cv2.approxPolyDP(largest, epsilon, True)
        polygon = [(int(p[0][0]), int(p[0][1])) for p in approx]

        return [{
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "segmentation": polygon if len(polygon) >= 3 else None,
            "area": int(cv2.contourArea(largest)),
        }]
    else:
        boxes = []
        for c in valid:
            x, y, w, h = cv2.boundingRect(c)
            epsilon = simplify
            approx = cv2.approxPolyDP(c, epsilon, True)
            polygon = [(int(p[0][0]), int(p[0][1])) for p in approx]
            boxes.append({
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "segmentation": polygon if len(polygon) >= 3 else None,
                "area": int(cv2.contourArea(c)),
            })
        return boxes

def import_segmentation_masks(
    rgb_files: List[str],
    masks_dir: str,
    bbox_cls: Any,
    merge_all: bool = True,
    min_area: int = 10,
    simplify: int = 10,
    ignore_hex: Optional[str] = "#000000",
    color_to_class: Dict[Tuple[int, int, int], str] = None,
    progress_callback=None,
    tolerance: int = 5
) -> Tuple[Dict[int, List[Any]], Dict[str, Any]]:
    """
    Reads masks corresponding to RGB files, extracts bounding boxes based on color,
    and returns a dictionary mapping frame_index to a list of BoundingBox objects.

    Returns: (frame_annotations, stats)
    """
    if cv2 is None or np is None:
        raise ImportError("OpenCV and numpy are required to import masks.")

    frame_annotations = {}
    stats = {
        "images_processed": 0,
        "masks_found": 0,
        "boxes_extracted": 0,
        "classes_found": set(),
    }

    ignore_bgr = hex_to_bgr(ignore_hex) if ignore_hex else None

    # Find all mask files in the masks_dir
    mask_files = {}
    if os.path.exists(masks_dir):
        for f in os.listdir(masks_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                base_name = os.path.splitext(f)[0]
                mask_files[base_name] = os.path.join(masks_dir, f)

    for i, rgb_path in enumerate(rgb_files):
        if progress_callback:
            progress_callback(i, len(rgb_files))

        base_rgb = os.path.splitext(os.path.basename(rgb_path))[0]
        mask_path = mask_files.get(base_rgb)
        
        if not mask_path:
            continue

        stats["masks_found"] += 1

        mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
        if mask_img is None:
            continue

        stats["images_processed"] += 1
        
        # Determine unique colors to process
        if color_to_class is not None:
            unique_colors_bgr = list(color_to_class.keys())
        else:
            if len(mask_img.shape) == 2:
                unique_colors = np.unique(mask_img)
                unique_colors_bgr = [(int(val), int(val), int(val)) for val in unique_colors]
            else:
                mask_img_bgr = mask_img[:, :, :3]
                pixels = mask_img_bgr.reshape(-1, 3)
                unique_colors = np.unique(pixels, axis=0)
                unique_colors_bgr = [(int(c[0]), int(c[1]), int(c[2])) for c in unique_colors]
                    
        # Prepare BGR mask for thresholding
        if len(mask_img.shape) == 2:
            mask_img_bgr = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
        else:
            mask_img_bgr = mask_img[:, :, :3]

        boxes_for_frame = []

        for color_bgr in unique_colors_bgr:
            if ignore_bgr is not None and color_bgr == ignore_bgr:
                continue

            # Create binary mask for this color with tolerance
            lower = np.array([max(0, c - tolerance) for c in color_bgr], dtype=np.uint8)
            upper = np.array([min(255, c + tolerance) for c in color_bgr], dtype=np.uint8)
            bin_mask = cv2.inRange(mask_img_bgr, lower, upper)

            # Clean up with morphology
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)
            bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)

            extracted = _mask_to_boxes(bin_mask, min_area, simplify, merge_all)
            if not extracted:
                continue

            # Assign class name based on color
            if color_to_class and tuple(color_bgr) in color_to_class:
                class_name = color_to_class[tuple(color_bgr)]
            else:
                if color_bgr[0] == color_bgr[1] == color_bgr[2]:
                    class_name = f"mask_val_{color_bgr[0]}"
                else:
                    class_name = f"mask_color_#{color_bgr[2]:02X}{color_bgr[1]:02X}{color_bgr[0]:02X}"
            
            stats["classes_found"].add(class_name)

            # Create BoundingBox objects
            from PyQt5.QtCore import QRect
            for box_dict in extracted:
                rect = QRect(box_dict["x"], box_dict["y"], max(1, box_dict["w"]), max(1, box_dict["h"]))
                ann = bbox_cls(
                    rect=rect,
                    class_name=class_name,
                    source="manual",
                    score=1.0,
                    segmentation=box_dict.get("segmentation"),
                )
                ann.verified = True
                # If instance segmentation, give a unique actor id based on the box properties or just a sequential id
                if not merge_all:
                    ann.attributes["actor_id"] = f"{class_name}_{len(boxes_for_frame)}"

                boxes_for_frame.append(ann)
                stats["boxes_extracted"] += 1

        if boxes_for_frame:
            frame_annotations[i] = boxes_for_frame

    if progress_callback:
        progress_callback(len(rgb_files), len(rgb_files))

    stats["classes_found"] = list(stats["classes_found"])
    return frame_annotations, stats
