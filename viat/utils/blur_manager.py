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

    def add_bbox_region(self, frame_idx: int, rect, kernel: int):
        """Add a blur region that exactly covers a QRect bounding box."""
        self._add_region(
            frame_idx, "bbox",
            rect.x(), rect.y(),
            rect.width(), rect.height(),
            kernel
        )

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
            x1 = max(0, region["x"])
            y1 = max(0, region["y"])
            x2 = min(w, x1 + region["w"])
            y2 = min(h, y1 + region["h"])
            if x2 <= x1 or y2 <= y1:
                continue
            kernel = max(3, region["kernel"] | 1)  # must be odd
            patch = result[y1:y2, x1:x2]
            blurred = cv2.GaussianBlur(patch, (kernel, kernel), 0)
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
                self.blur_regions[int(k)] = list(v)
            except (ValueError, TypeError):
                pass
