"""
Segmentation video labeling for VIAT.

Workflow:
  1. Load a "segmentation video" -- a video where each object is rendered as
     a distinct solid color (e.g. a Blender/Cycles mask pass, or a semantic
     segmentation visualization).
  2. User clicks on a colored region in the canvas to pick an object's color.
  3. The manager tracks that color across a frame range using OpenCV
     color thresholding (inRange) + contour detection, computing the
     bounding box (and optional polygon) of the matching region for each
     frame.
  4. The resulting annotations are stored in frame_annotations with a unique
     actor_id, ready to be exported to VIAT JSON.

The user can:
  * Pick multiple objects (each click adds a new tracked object).
  * Set a frame range (default: entire video, or a selected sub-range).
  * Adjust the color tolerance (how close a pixel must be to the picked color).
  * Set a minimum area threshold (ignore tiny noise regions).
  * Preview the mask before committing.
  * Assign a class name + actor_id to each tracked object.
"""

import os
from typing import Dict, List, Optional, Tuple

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


class SegmentationVideoLabeler:
    """Tracks colored regions across video frames and generates annotations."""

    def __init__(self, app):
        self.app = app

        # Picked objects: list of dicts:
        #   {color_hsv: (h,s,v), tolerance: int, min_area: int,
        #    class_name: str, actor_id: str, boxes: {frame: box_dict}}
        self.tracked_objects: List[dict] = []

        # Frame range to process
        self.start_frame: int = 0
        self.end_frame: int = 0

        # Default settings
        self.default_tolerance: int = 15  # HSV hue tolerance
        self.default_min_area: int = 100  # minimum contour area in pixels
        self.default_simplify: int = 10   # polygon simplification epsilon

    # ------------------------------------------------------------------ #
    # Color picking
    # ------------------------------------------------------------------ #

    def pick_color_from_frame(self, frame, x: int, y: int) -> Tuple[int, int, int]:
        """Pick the HSV color at (x, y) in the given BGR frame.

        Returns (H, S, V) in OpenCV's HSV ranges (H: 0-179, S: 0-255, V: 0-255).
        """
        if frame is None:
            return (0, 0, 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[y, x]
        return (int(h), int(s), int(v))

    def add_tracked_object(
        self,
        color_hsv: Tuple[int, int, int],
        class_name: str,
        actor_id: str = None,
        tolerance: int = None,
        min_area: int = None,
    ) -> dict:
        """Register a new object to track by color.

        Args:
            color_hsv: (H, S, V) picked from the frame.
            class_name: class label for this object.
            actor_id: unique ID (auto-generated if None).
            tolerance: HSV tolerance (default self.default_tolerance).
            min_area: minimum contour area (default self.default_min_area).

        Returns the tracked-object dict.
        """
        if actor_id is None:
            actor_id = f"seg_obj_{len(self.tracked_objects) + 1}"

        obj = {
            "color_hsv": color_hsv,
            "tolerance": tolerance or self.default_tolerance,
            "min_area": min_area or self.default_min_area,
            "class_name": class_name,
            "actor_id": actor_id,
            "boxes": {},  # {frame_num: box_dict}
        }
        self.tracked_objects.append(obj)
        return obj

    # ------------------------------------------------------------------ #
    # Color tracking (core)
    # ------------------------------------------------------------------ #

    def _build_mask(self, frame_hsv, color_hsv, tolerance) -> "np.ndarray":
        """Build a binary mask for pixels matching the target HSV color."""
        h, s, v = color_hsv
        # Hue wraps around 0/180 in OpenCV, so handle the wrap
        lower_h = (h - tolerance) % 180
        upper_h = (h + tolerance) % 180

        if lower_h <= upper_h:
            lower = np.array([lower_h, max(0, s - tolerance * 2), max(0, v - tolerance * 2)])
            upper = np.array([upper_h, 255, 255])
            mask = cv2.inRange(frame_hsv, lower, upper)
        else:
            # Hue wraps: combine two ranges
            lower1 = np.array([0, max(0, s - tolerance * 2), max(0, v - tolerance * 2)])
            upper1 = np.array([upper_h, 255, 255])
            lower2 = np.array([lower_h, max(0, s - tolerance * 2), max(0, v - tolerance * 2)])
            upper2 = np.array([179, 255, 255])
            mask1 = cv2.inRange(frame_hsv, lower1, upper1)
            mask2 = cv2.inRange(frame_hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)

        # Clean up: morphological opening then closing
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    def _mask_to_box(
        self, mask, min_area: int, simplify: int = 10
    ) -> Optional[dict]:
        """Extract the largest contour from a mask and compute bbox + polygon.

        Returns None if no contour meets the min_area threshold.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Keep only contours above min_area
        valid = [c for c in contours if cv2.contourArea(c) >= min_area]
        if not valid:
            return None

        # Merge all valid contours' bounding box
        all_points = np.vstack(valid)
        x, y, w, h = cv2.boundingRect(all_points)

        # Build a simplified polygon from the largest contour
        largest = max(valid, key=cv2.contourArea)
        epsilon = simplify
        approx = cv2.approxPolyDP(largest, epsilon, True)
        polygon = [(int(p[0][0]), int(p[0][1])) for p in approx]

        return {
            "x": int(x),
            "y": int(y),
            "w": int(w),
            "h": int(h),
            "segmentation": polygon if len(polygon) >= 3 else None,
            "area": int(cv2.contourArea(largest)),
        }

    def track_object_on_frame(self, obj: dict, frame) -> Optional[dict]:
        """Track a single object on a single frame.

        Returns the box dict, or None if the object isn't visible.
        """
        if frame is None:
            return None
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = self._build_mask(hsv, obj["color_hsv"], obj["tolerance"])
        return self._mask_to_box(mask, obj["min_area"], self.default_simplify)

    def track_all_objects(
        self,
        video_path: str,
        *,
        start_frame: int = None,
        end_frame: int = None,
        progress_callback=None,
    ) -> Dict:
        """Track all registered objects across the video frame range.

        Args:
            video_path: path to the segmentation video.
            start_frame: first frame (default 0).
            end_frame: last frame (default: video end).
            progress_callback: optional callable(current, total) for UI updates.

        Returns:
            {frames_processed, total_boxes, per_object: {actor_id: count}}
        """
        if not self.tracked_objects:
            return {"frames_processed": 0, "total_boxes": 0, "per_object": {}}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"frames_processed": 0, "total_boxes": 0, "per_object": {}}

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if start_frame is None:
            start_frame = 0
        if end_frame is None or end_frame >= total:
            end_frame = total - 1

        frames_processed = 0
        total_boxes = 0
        per_object: Dict[str, int] = {obj["actor_id"]: 0 for obj in self.tracked_objects}

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_num in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            for obj in self.tracked_objects:
                mask = self._build_mask(hsv, obj["color_hsv"], obj["tolerance"])
                box = self._mask_to_box(mask, obj["min_area"], self.default_simplify)
                if box:
                    obj["boxes"][frame_num] = box
                    per_object[obj["actor_id"]] += 1
                    total_boxes += 1

            frames_processed += 1
            if progress_callback:
                progress_callback(frame_num - start_frame + 1, end_frame - start_frame + 1)

        cap.release()
        return {
            "frames_processed": frames_processed,
            "total_boxes": total_boxes,
            "per_object": per_object,
        }

    # ------------------------------------------------------------------ #
    # Commit to app's frame_annotations
    # ------------------------------------------------------------------ #

    def commit_to_app(self, app, bbox_cls) -> int:
        """Write all tracked boxes into app.frame_annotations as BoundingBox objects.

        Each box gets:
          - class_name from the tracked object
          - actor_id in attributes
          - verified = True (user confirmed by picking)
          - segmentation polygon if available

        Returns the number of annotations added.
        """
        from PyQt5.QtCore import QRect
        from PyQt5.QtGui import QColor
        import random

        added = 0
        for obj in self.tracked_objects:
            class_name = obj["class_name"]
            actor_id = obj["actor_id"]

            # Ensure class exists on canvas
            if class_name not in app.canvas.class_colors:
                app.canvas.class_colors[class_name] = QColor(
                    random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                )

            color = app.canvas.class_colors[class_name]
            for frame_num, box in obj["boxes"].items():
                rect = QRect(box["x"], box["y"], max(1, box["w"]), max(1, box["h"]))
                ann = bbox_cls(
                    rect=rect,
                    class_name=class_name,
                    attributes={"actor_id": actor_id},
                    color=color,
                    source="manual",
                    score=1.0,
                    segmentation=box.get("segmentation"),
                )
                ann.verified = True  # user picked this, so it's accepted

                if frame_num not in app.frame_annotations:
                    app.frame_annotations[frame_num] = []
                app.frame_annotations[frame_num].append(ann)
                added += 1

        return added

    # ------------------------------------------------------------------ #
    # Export to VIAT JSON
    # ------------------------------------------------------------------ #

    def to_viat_json(self) -> str:
        """Export all tracked objects to VIAT JSON string."""
        from .label_formats.viat_json import ViatJsonLabelFormat

        boxes_by_frame = {}
        for obj in self.tracked_objects:
            for frame_num, box in obj["boxes"].items():
                if frame_num not in boxes_by_frame:
                    boxes_by_frame[frame_num] = []
                boxes_by_frame[frame_num].append({
                    "class_name": obj["class_name"],
                    "actor_id": obj["actor_id"],
                    "x": box["x"],
                    "y": box["y"],
                    "w": box["w"],
                    "h": box["h"],
                    "verified": True,
                    "segmentation": box.get("segmentation"),
                })

        fmt = ViatJsonLabelFormat()
        return fmt.dump(boxes_by_frame, (0, 0), [])

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #

    def reset(self):
        """Clear all tracked objects."""
        self.tracked_objects = []
