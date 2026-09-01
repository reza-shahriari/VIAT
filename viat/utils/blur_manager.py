"""
BlurManager — manages per-frame blur regions for privacy masking.

Blur regions are stored as metadata and applied non-destructively:
  - displayed as a real-time blurred overlay on the canvas
  - baked into pixel data only at export time

Data format per frame:
    blur_regions[frame_idx] = [
        {"type": "pen"|"bbox", "x": int, "y": int,
         "w": int, "h": int, "kernel": int},
        ...
    ]
"""

import cv2
import numpy as np


class BlurManager:
    """Manages per-frame blur regions and applies Gaussian blur to numpy frames."""

    def __init__(self):
        # { frame_idx(int): [ {type, x, y, w, h, kernel}, ... ] }
        self.blur_regions = {}

    # ------------------------------------------------------------------
    # Adding regions
    # ------------------------------------------------------------------

    def add_pen_stroke(self, frame_idx: int, cx: int, cy: int,
                       radius: int, kernel: int):
        """Add a square pen patch centred at (cx, cy) with given radius."""
        x = max(0, cx - radius)
        y = max(0, cy - radius)
        w = radius * 2
        h = radius * 2
        self._add_region(frame_idx, "pen", x, y, w, h, kernel)

    def add_bbox_region(self, frame_idx: int, rect, kernel: int, margin: int = 0):
        """Add a blur region that covers a QRect bounding box, optionally expanded by margin."""
        if margin > 0:
            x = max(0, int(rect.x() - margin))
            y = max(0, int(rect.y() - margin))
            w = int(rect.width() + 2 * margin)
            h = int(rect.height() + 2 * margin)
        else:
            x = max(0, int(rect.x()))
            y = max(0, int(rect.y()))
            w = int(rect.width())
            h = int(rect.height())
        self._add_region(
            frame_idx, "bbox",
            x, y, w, h,
            kernel
        )

    def add_polygon_region(self, frame_idx: int, polygon, kernel: int, margin: int = 0):
        """Add a blur region defined by a polygon, optionally dilated by margin."""
        if not polygon:
            return
            
        pts = np.array(polygon, dtype=np.int32)
        if len(pts) < 3:
            return

        if margin > 0:
            x_min, y_min, w_box, h_box = cv2.boundingRect(pts)
            pad = int(margin + 5)
            mask_w = w_box + 2 * pad
            mask_h = h_box + 2 * pad
            mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
            local_pts = pts - [x_min - pad, y_min - pad]
            cv2.fillPoly(mask, [local_pts], 255)
            ksize = 2 * int(margin) + 1
            kernel_elem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
            dilated = cv2.dilate(mask, kernel_elem)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_c = max(contours, key=cv2.contourArea)
                shifted_pts = largest_c.reshape(-1, 2) + [x_min - pad, y_min - pad]
                pts = np.maximum(shifted_pts, 0)
                
        x, y, w, h = cv2.boundingRect(pts)
        
        # Simplify the polygon to drastically reduce the number of points for JSON saving
        # An epsilon of 2.0 pixels is a good balance for blur mask boundaries without losing shape
        epsilon = 2.0
        approx_pts = cv2.approxPolyDP(pts, epsilon, True)
        
        if frame_idx not in self.blur_regions:
            self.blur_regions[frame_idx] = []
        kernel = max(3, kernel | 1)
        
        # Store as lists for JSON serialization (approx_pts has shape (M, 1, 2))
        poly_list = [[float(pt[0][0]), float(pt[0][1])] for pt in approx_pts]
        
        self.blur_regions[frame_idx].append({
            "type": "polygon", "x": int(x), "y": int(y),
            "w": int(w), "h": int(h), "kernel": kernel,
            "points": poly_list
        })

    def _add_region(self, frame_idx, rtype, x, y, w, h, kernel):
        if frame_idx not in self.blur_regions:
            self.blur_regions[frame_idx] = []
        # Ensure kernel is at least 3 and odd
        kernel = max(3, kernel | 1)
        self.blur_regions[frame_idx].append({
            "type": rtype, "x": x, "y": y,
            "w": w, "h": h, "kernel": kernel
        })

    # ------------------------------------------------------------------
    # Removing regions
    # ------------------------------------------------------------------

    def clear_frame(self, frame_idx: int):
        """Remove all blur regions from a single frame."""
        self.blur_regions.pop(frame_idx, None)

    def clear_range(self, start_frame: int, end_frame: int):
        """Remove all blur regions from frames in range [start_frame, end_frame]."""
        for f in range(start_frame, end_frame + 1):
            self.blur_regions.pop(f, None)

    def clear_all(self):
        """Remove all blur regions from every frame."""
        self.blur_regions.clear()

    def remove_pen_strokes(self, frame_idx: int):
        """Remove only pen-type blur regions from a frame."""
        if frame_idx in self.blur_regions:
            self.blur_regions[frame_idx] = [
                r for r in self.blur_regions[frame_idx]
                if r["type"] != "pen"
            ]
            if not self.blur_regions[frame_idx]:
                del self.blur_regions[frame_idx]

    def remove_bbox_regions(self, frame_idx: int):
        """Remove only bbox-type blur regions from a frame."""
        if frame_idx in self.blur_regions:
            self.blur_regions[frame_idx] = [
                r for r in self.blur_regions[frame_idx]
                if r["type"] != "bbox"
            ]
            if not self.blur_regions[frame_idx]:
                del self.blur_regions[frame_idx]

    # ------------------------------------------------------------------
    # Applying blur
    # ------------------------------------------------------------------

    def apply_blur_to_frame(self, frame: np.ndarray, frame_idx: int) -> np.ndarray:
        """
        Return a copy of *frame* with all stored blur regions applied.
        The original array is NOT modified.
        If no regions exist for frame_idx, returns the original array.
        """
        regions = self.blur_regions.get(frame_idx)
        if not regions:
            return frame

        result = frame.copy()
        h, w = result.shape[:2]

        for region in regions:
            x1 = max(0, int(round(region.get("x", 0))))
            y1 = max(0, int(round(region.get("y", 0))))
            x2 = min(w, int(round(x1 + region.get("w", 0))))
            y2 = min(h, int(round(y1 + region.get("h", 0))))
            if x2 <= x1 or y2 <= y1:
                continue
            raw_kernel = int(round(region.get("kernel", 151)))
            kernel = max(3, raw_kernel | 1)  # must be odd and >= 3
            
            patch = result[y1:y2, x1:x2].copy()
            blurred = cv2.GaussianBlur(patch, (kernel, kernel), 0)
            
            if region.get("type") == "polygon" and "points" in region:
                points = np.array(region["points"], dtype=np.int32)
                mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                shifted_points = points - [x1, y1]
                cv2.fillPoly(mask, [shifted_points], 255)
                
                mask_bool = mask == 255
                result[y1:y2, x1:x2][mask_bool] = blurred[mask_bool]
            else:
                result[y1:y2, x1:x2] = blurred

        return result

    def has_blur(self, frame_idx: int) -> bool:
        """Return True if the given frame has any blur regions."""
        return bool(self.blur_regions.get(frame_idx))

    def get_regions(self, frame_idx: int):
        """Return list of region dicts for the given frame (read-only)."""
        return self.blur_regions.get(frame_idx, [])

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise blur_regions to a JSON-compatible dict."""
        return {
            str(k): list(v)
            for k, v in self.blur_regions.items()
        }

    def from_dict(self, data: dict):
        """Restore blur_regions from a JSON-loaded dict."""
        self.blur_regions = {}
        if not data:
            return
        for k, v in data.items():
            try:
                frame_idx = int(k)
                if isinstance(v, list):
                    clean_list = []
                    for item in v:
                        if isinstance(item, dict):
                            clean_item = dict(item)
                            clean_item["x"] = int(round(clean_item.get("x", 0)))
                            clean_item["y"] = int(round(clean_item.get("y", 0)))
                            clean_item["w"] = int(round(clean_item.get("w", 0)))
                            clean_item["h"] = int(round(clean_item.get("h", 0)))
                            clean_item["kernel"] = max(3, int(round(clean_item.get("kernel", 151))) | 1)
                            clean_list.append(clean_item)
                    if clean_list:
                        self.blur_regions[frame_idx] = clean_list
            except (ValueError, TypeError):
                pass
