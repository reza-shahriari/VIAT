"""
Video border detection & label adjustment for VIAT.

Problem: some videos have black/gray border columns on the left and/or right
edges (letterboxing, pillarboxing, or scene-rendering artifacts). Labels that
fall fully or mostly (≥80%) inside these borders should be removed; labels
that partially overlap the border should be clipped to the non-border area.

This module:
  1. Samples frames from the video to detect border columns.
  2. For each annotation, computes what fraction of its area is in the border.
  3. Removes annotations that are ≥80% in the border.
  4. Clips annotations that partially overlap the border (adjusts bbox + seg).
  5. Leaves annotations outside the border unchanged.

The video file itself is NEVER modified -- only the in-memory annotations.
"""

import os
from typing import Dict, List, Optional, Tuple

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None


# --------------------------------------------------------------------------- #
# Border detection
# --------------------------------------------------------------------------- #


def detect_video_borders(
    video_path: str,
    *,
    num_samples: int = 20,
    dark_threshold: int = 30,
    variance_threshold: float = 10.0,
    min_border_width: int = 2,
) -> Dict:
    """Detect left/right black/gray border columns in a video.

    Samples ``num_samples`` frames evenly across the video. For each column,
    computes the mean brightness and variance across all sampled pixels in
    that column. A column is "border" if:
      * mean brightness < dark_threshold (dark)
      * variance < variance_threshold (uniform -- catches gray as well as black)

    Args:
        video_path: path to the video file.
        num_samples: how many frames to sample.
        dark_threshold: max mean brightness for a border column.
        variance_threshold: max variance for a border column (uniform color).
        min_border_width: minimum consecutive border columns to count as a border.

    Returns:
        dict: {left_border: int, right_border: int, frame_width: int,
               sampled: int, detected: bool}
        left_border = number of border columns from the left edge.
        right_border = number of border columns from the right edge.
    """
    if cv2 is None:
        return {"left_border": 0, "right_border": 0, "frame_width": 0, "sampled": 0, "detected": False}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"left_border": 0, "right_border": 0, "frame_width": 0, "sampled": 0, "detected": False}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    if total_frames <= 0 or width <= 0:
        cap.release()
        return {"left_border": 0, "right_border": 0, "frame_width": width, "sampled": 0, "detected": False}

    # Accumulate per-column mean and variance across sampled frames
    col_means = np.zeros(width, dtype=np.float64)
    col_vars = np.zeros(width, dtype=np.float64)
    sampled = 0

    step = max(1, total_frames // num_samples)
    for f in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Per-column mean and std
        col_means += gray.mean(axis=0)
        col_vars += gray.std(axis=0)
        sampled += 1
        if sampled >= num_samples:
            break

    cap.release()

    if sampled == 0:
        return {"left_border": 0, "right_border": 0, "frame_width": width, "sampled": 0, "detected": False}

    col_means /= sampled
    col_vars /= sampled

    # Left border: consecutive columns from the left that are dark + uniform
    left_border = 0
    for x in range(width):
        if col_means[x] < dark_threshold and col_vars[x] < variance_threshold:
            left_border += 1
        else:
            break
        if left_border >= width // 2:  # sanity: don't call the whole frame a border
            break

    # Right border: consecutive columns from the right
    right_border = 0
    for x in range(width - 1, -1, -1):
        if col_means[x] < dark_threshold and col_vars[x] < variance_threshold:
            right_border += 1
        else:
            break
        if right_border >= width // 2:
            break

    if left_border < min_border_width:
        left_border = 0
    if right_border < min_border_width:
        right_border = 0

    return {
        "left_border": left_border,
        "right_border": right_border,
        "frame_width": width,
        "sampled": sampled,
        "detected": (left_border > 0 or right_border > 0),
    }


# --------------------------------------------------------------------------- #
# Annotation adjustment
# --------------------------------------------------------------------------- #


def adjust_annotations_for_borders(
    app,
    *,
    left_border: int = 0,
    right_border: int = 0,
    frame_width: int = None,
    removal_threshold: float = 0.8,
    dry_run: bool = False,
) -> Dict:
    """Adjust all annotations based on detected video borders.

    For each annotation in every frame:
      * Compute what fraction of the annotation's bbox area is inside the
        border region (left_border columns + right_border columns).
      * If ≥ ``removal_threshold`` (80% by default) of the annotation is in
        the border → remove it.
      * If partially in the border → clip the bbox to the non-border area.
        Also clip the segmentation polygon if present.
      * If not in the border → leave unchanged.

    Args:
        app: the VideoAnnotationTool main window.
        left_border: width of left border in pixels.
        right_border: width of right border in pixels.
        frame_width: total frame width (if None, inferred from video/image).
        removal_threshold: fraction of area in border to trigger removal (0-1).
        dry_run: if True, don't modify anything, just report what would happen.

    Returns:
        dict: {removed, clipped, unchanged, total, borders}
    """
    if frame_width is None:
        frame_width = getattr(app, "video_width", None)
        if frame_width is None and getattr(app, "cap", None) and app.cap.isOpened():
            frame_width = int(app.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if frame_width is None:
            # image dataset
            if getattr(app, "image_files", None) and len(app.image_files) > 0:
                img = cv2.imread(app.image_files[0])
                if img is not None:
                    frame_width = img.shape[1]
        if frame_width is None:
            frame_width = 1920  # fallback

    non_border_left = left_border
    non_border_right = frame_width - right_border  # exclusive boundary

    removed = 0
    clipped = 0
    unchanged = 0
    total = 0

    frame_annotations = getattr(app, "frame_annotations", {})

    for frame_num in list(frame_annotations.keys()):
        anns = frame_annotations[frame_num]
        new_anns = []
        for ann in anns:
            total += 1
            rect = ann.rect
            x1 = rect.x()
            y1 = rect.y()
            x2 = rect.x() + rect.width()
            y2 = rect.y() + rect.height()

            # Compute the portion of the bbox that overlaps the border
            border_area = 0
            # Left border overlap
            if x1 < non_border_left:
                overlap_x1 = x1
                overlap_x2 = min(x2, non_border_left)
                if overlap_x2 > overlap_x1:
                    border_area += (overlap_x2 - overlap_x1) * rect.height()
            # Right border overlap
            if x2 > non_border_right:
                overlap_x1 = max(x1, non_border_right)
                overlap_x2 = x2
                if overlap_x2 > overlap_x1:
                    border_area += (overlap_x2 - overlap_x1) * rect.height()

            bbox_area = rect.width() * rect.height()
            if bbox_area <= 0:
                new_anns.append(ann)
                unchanged += 1
                continue

            in_border_fraction = border_area / bbox_area

            if in_border_fraction >= removal_threshold:
                # Remove this annotation
                removed += 1
                continue

            if border_area > 0:
                # Clip the bbox to the non-border area
                new_x1 = max(x1, non_border_left)
                new_x2 = min(x2, non_border_right)
                if new_x2 <= new_x1:
                    # Fully in border after clipping (shouldn't happen, but guard)
                    removed += 1
                    continue

                if not dry_run:
                    from PyQt5.QtCore import QRect
                    ann.rect = QRect(new_x1, y1, new_x2 - new_x1, rect.height())

                    # Clip segmentation polygon if present
                    if getattr(ann, "segmentation", None):
                        ann.segmentation = _clip_polygon_x(
                            ann.segmentation, non_border_left, non_border_right
                        )
                clipped += 1
                new_anns.append(ann)
            else:
                new_anns.append(ann)
                unchanged += 1

        if not dry_run:
            frame_annotations[frame_num] = new_anns

    return {
        "removed": removed,
        "clipped": clipped,
        "unchanged": unchanged,
        "total": total,
        "borders": {"left": left_border, "right": right_border, "width": frame_width},
    }


def _clip_polygon_x(polygon, x_min, x_max):
    """Clip a polygon's x-coordinates to [x_min, x_max].

    This is a simple clamp -- for a proper polygon clip you'd use Sutherland-
    Hodgman, but clamping x is sufficient for border removal (the border is
    a vertical strip, so we just need to cut x-extent).
    """
    return [(max(x_min, min(x_max, x)), y) for (x, y) in polygon]


# --------------------------------------------------------------------------- #
# Convenience: detect + adjust in one call
# --------------------------------------------------------------------------- #


def detect_and_adjust_borders(app, video_path=None, *, removal_threshold=0.8, dry_run=False) -> Dict:
    """Detect video borders and adjust annotations in one call.

    Args:
        app: main window.
        video_path: path to video (if None, uses app.video_filename).
        removal_threshold: fraction in border to trigger removal.
        dry_run: if True, only report.

    Returns:
        dict with detection results + adjustment results.
    """
    if video_path is None:
        video_path = getattr(app, "video_filename", None)
    if not video_path or not os.path.isfile(video_path):
        return {"error": "No video path", "detected": False}

    detection = detect_video_borders(video_path)
    if not detection["detected"]:
        return {"detection": detection, "adjustment": {"removed": 0, "clipped": 0, "unchanged": 0, "total": 0}, "detected": False}

    adjustment = adjust_annotations_for_borders(
        app,
        left_border=detection["left_border"],
        right_border=detection["right_border"],
        frame_width=detection["frame_width"],
        removal_threshold=removal_threshold,
        dry_run=dry_run,
    )

    return {"detection": detection, "adjustment": adjustment, "detected": True}
