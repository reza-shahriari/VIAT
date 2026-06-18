"""
Interpolation module for VIAT.

This module provides functionality to interpolate annotations between keyframes,
reducing the manual annotation workload by automatically generating annotations
for intermediate frames.
"""

from PyQt5.QtCore import QRect
from .annotation import BoundingBox
import numpy as np


class InterpolationManager:
    """
    Manages the interpolation of annotations between keyframes.
    """

    def __init__(self, main_window):
        """
        Initialize the interpolation manager.

        Args:
            main_window: Reference to the main application window
        """
        self.main_window = main_window
        self.is_active = False
        self.interval = 10  # frames between keyframes (user-configurable, min 2)

        # --- New, simple cycle state machine ---
        # cycle is None when no cycle is in progress. Otherwise:
        #   {"anchor": int, "target": int, "phase": "awaiting_second" | "reviewing"}
        # See get_next_frame() for the full workflow.
        self.cycle = None

        # Legacy attributes kept for backward compatibility with older code/UI.
        self.last_annotated_frame = None
        self.pending_interpolation = False
        self.workflow_state = {
            "state": "idle",
            "keyframes": [],
            "interpolated_frames": [],
            "review_queue": [],
            "current_frame": 0,
        }

    def set_active(self, active):
        """Enable or disable interpolation mode."""
        self.is_active = active
        self.reset_cycle()
        if active:
            self.main_window.statusBar.showMessage(
                f"Interpolation mode active. Interval: {self.interval} frames. "
                f"Label a frame, then press Next to jump ahead by {self.interval}.",
                5000,
            )
        else:
            self.main_window.statusBar.showMessage("Interpolation mode disabled.")
            self.last_annotated_frame = None
            self.pending_interpolation = False

    def set_interval(self, interval):
        """Set the interval between keyframes."""
        self.interval = max(2, interval)  # Minimum interval is 2
        if self.is_active:
            self.main_window.statusBar.showMessage(
                f"Interpolation interval set to {self.interval} frames."
            )

    # ------------------------------------------------------------------
    # New workflow (driven by Next/Prev in main_window).
    # ------------------------------------------------------------------

    def reset_cycle(self):
        """Clear the current interpolation cycle. Safe to call anytime."""
        self.cycle = None

    def get_next_frame(self, current_frame):
        """
        Decide which frame to go to when the user presses Next while
        interpolation mode is active. Implements the workflow:

          * No active cycle, current frame `c` is labeled -> start cycle:
                anchor = c, target = c + interval, phase = awaiting_second.
                Jump to target.   ('label n, Next -> n+x')
          * phase == awaiting_second, at target, target is labeled
                -> interpolate(anchor, target), phase = reviewing,
                   go to anchor+1.
          * phase == awaiting_second, at target, target NOT labeled (user
            just pressed Next without annotating) -> abort cycle, normal +1.
                ('did not label n+x, just Next -> work as is')
          * phase == reviewing, anchor < c < target
                -> if c+1 < target: go to c+1 (next review frame)
                   else (c == target-1): review done, go to target+1,
                   clear cycle.   ('after the next I will be in n+x+1')
          * phase == reviewing, c == anchor (user stepped back to keyframe)
                -> resume review: go to anchor+1.
          * Otherwise (out of range / not labeled with no cycle) -> normal +1.

        Returns the frame number to seek to (clamped to [0, total-1]).
        """
        total = getattr(self.main_window, "total_frames", 10 ** 9)
        x = max(2, self.interval)
        c = current_frame
        labeled = self.has_annotation(c)

        def clamp(v):
            return max(0, min(v, total - 1))

        # No active cycle ------------------------------------------------
        if self.cycle is None:
            if labeled and c + x <= total - 1:
                self.cycle = {
                    "anchor": c,
                    "target": clamp(c + x),
                    "phase": "awaiting_second",
                }
                self.main_window.statusBar.showMessage(
                    f"Interpolation: labeled keyframe {c}. Jumping to "
                    f"{self.cycle['target']} (interval={x}). Label it to "
                    f"interpolate, or just press Next to cancel.",
                    5000,
                )
                return self.cycle["target"]
            # Not labeled, or not enough room for a full interval: normal step.
            return clamp(c + 1)

        anchor = self.cycle["anchor"]
        target = self.cycle["target"]
        phase = self.cycle["phase"]

        # Waiting for the second keyframe -------------------------------
        if phase == "awaiting_second":
            if c == target and labeled:
                # Second keyframe labeled -> interpolate, begin review.
                self.interpolate_annotations(anchor, target)
                self.cycle["phase"] = "reviewing"
                self.main_window.statusBar.showMessage(
                    f"Interpolation: interpolated frames {anchor + 1}..{target - 1}. "
                    f"Reviewing.",
                    4000,
                )
                return clamp(anchor + 1)
            # User didn't label the target (or navigated away): abort.
            self.reset_cycle()
            self.main_window.statusBar.showMessage(
                "Interpolation: cycle cancelled (target frame was not labeled).",
                3000,
            )
            return clamp(c + 1)

        # Reviewing interpolated frames ---------------------------------
        if phase == "reviewing":
            if c == anchor:
                # User stepped back to the anchor keyframe; resume review.
                return clamp(anchor + 1)
            if anchor < c < target:
                if c + 1 < target:
                    return clamp(c + 1)
                # c == target-1: review finished. Skip the keyframe at
                # `target`, start fresh from target+1 (becomes new 'n').
                nxt = clamp(target + 1)
                self.reset_cycle()
                self.main_window.statusBar.showMessage(
                    f"Interpolation: review complete. Continuing from {nxt}.",
                    4000,
                )
                return nxt
            # Outside the review range: abort, normal step.
            self.reset_cycle()
            return clamp(c + 1)

        # Defensive fallback
        self.reset_cycle()
        return clamp(c + 1)

    def get_prev_frame(self, current_frame):
        """
        Prev ALWAYS steps back exactly one frame, regardless of interpolation
        state (per user requirement). The cycle is left untouched so the user
        can step back to inspect a frame and then continue with Next.
        """
        return max(0, current_frame - 1)

    # ------------------------------------------------------------------
    # Backward-compatibility shims for the old workflow hooks.
    # The new workflow is driven entirely by get_next_frame()/
    # get_prev_frame() via the Next/Prev buttons. These shims keep
    # existing callers harmless.
    # ------------------------------------------------------------------

    def on_frame_annotated(self, frame_num):
        """Deprecated no-op. Workflow is now driven by Next/Prev."""
        return

    def check_pending_interpolation(self, new_frame):
        """Deprecated no-op."""
        return

    def perform_pending_interpolation(self):
        """
        Manual interpolation used by the toolbar 'Interpolate' button.
        Interpolates between the two nearest annotated keyframes that bracket
        the current frame; if none bracket it, falls back to the nearest pair
        before it.
        """
        c = getattr(self.main_window, "current_frame", 0)
        prev_kf = self.find_prev_annotated_frame(c)
        next_kf = self.find_next_annotated_frame(c)
        if prev_kf is not None and next_kf is not None and next_kf - prev_kf > 1:
            return self.interpolate_annotations(prev_kf, next_kf)
        if prev_kf is not None:
            pp = self.find_prev_annotated_frame(prev_kf)
            if pp is not None and prev_kf - pp > 1:
                return self.interpolate_annotations(pp, prev_kf)
        self.main_window.statusBar.showMessage(
            "Interpolation: need at least two annotated keyframes to interpolate.",
            4000,
        )
        return False

    def interpolate_annotations(self, start_frame, end_frame, method="linear"):
        """
        Interpolate annotations between two keyframes.

        Args:
            start_frame: The starting keyframe
            end_frame: The ending keyframe
            method: Interpolation method ("linear" or "smooth")

        Returns:
            bool: True if interpolation was successful, False otherwise
        """
        if start_frame not in self.main_window.frame_annotations:
            self.main_window.statusBar.showMessage(
                f"Error: Start frame {start_frame} has no annotations."
            )
            return False

        if end_frame not in self.main_window.frame_annotations:
            self.main_window.statusBar.showMessage(
                f"Error: End frame {end_frame} has no annotations."
            )
            return False

        start_annotations = self.main_window.frame_annotations[start_frame]
        end_annotations = self.main_window.frame_annotations[end_frame]

        if not start_annotations or not end_annotations:
            self.main_window.statusBar.showMessage(
                "Error: Both start and end frames must have annotations."
            )
            return False

        # Match annotations between start and end frames
        matched_annotations = self._match_annotations(start_annotations, end_annotations)

        if not matched_annotations:
            self.main_window.statusBar.showMessage(
                "Error: Could not match any annotations between frames."
            )
            return False

        # Choose interpolation method
        if method == "smooth" and end_frame - start_frame > 2:
            return self._smooth_interpolate(start_frame, end_frame, matched_annotations)
        else:
            return self._linear_interpolate(start_frame, end_frame, matched_annotations)

    def _linear_interpolate(self, start_frame, end_frame, matched_annotations):
        """
        Perform linear interpolation between frames
        
        Args:
            start_frame: Starting frame number
            end_frame: Ending frame number
            matched_annotations: List of (start_ann, end_ann) tuples
            
        Returns:
            bool: Success status
        """
        # Interpolate between matched annotations
        for frame_idx in range(start_frame + 1, end_frame):
            # Skip if frame already has annotations
            if frame_idx in self.main_window.frame_annotations and self.main_window.frame_annotations[frame_idx]:
                continue

            # Calculate interpolation factor (0 to 1)
            alpha = (frame_idx - start_frame) / (end_frame - start_frame)

            # Create interpolated annotations
            frame_annotations = []
            for start_ann, end_ann in matched_annotations:
                interpolated_ann = self._interpolate_annotation(start_ann, end_ann, alpha)
                frame_annotations.append(interpolated_ann)

            # Save interpolated annotations
            self.main_window.frame_annotations[frame_idx] = frame_annotations
            
            # Update UI if this is the current frame
            if self.main_window.current_frame == frame_idx:
                self.main_window.canvas.annotations = frame_annotations
                self.main_window.canvas.update()
                if hasattr(self.main_window, 'annotation_dock'):
                    self.main_window.annotation_dock.update_annotation_list()

        return True

    def _smooth_interpolate(self, start_frame, end_frame, matched_annotations):
        """
        Perform smooth interpolation between frames using cubic or quadratic interpolation.
        
        Args:
            start_frame: Starting frame number
            end_frame: Ending frame number
            matched_annotations: List of (start_ann, end_ann) tuples
            
        Returns:
            bool: Success status
        """
        # Find any intermediate keyframes
        intermediate_frames = []
        for frame in range(start_frame + 1, end_frame):
            if frame in self.main_window.frame_annotations and self.main_window.frame_annotations[frame]:
                intermediate_frames.append(frame)
        
        # If no intermediate frames, fall back to linear interpolation
        if not intermediate_frames:
            return self._linear_interpolate(start_frame, end_frame, matched_annotations)
            
        # Group annotations across all keyframes
        keyframes = [start_frame] + intermediate_frames + [end_frame]
        annotation_groups = self._group_annotations_across_frames(keyframes)
        
        if not annotation_groups:
            self.main_window.statusBar.showMessage(
                "Error: Could not match annotations across keyframes for smooth interpolation.",
                3000
            )
            return False
            
        # For each frame to interpolate
        for frame_idx in range(start_frame + 1, end_frame):
            # Skip keyframes
            if frame_idx in keyframes:
                continue
                
            # Calculate interpolated annotations for this frame
            frame_annotations = []
            
            for group in annotation_groups:
                # Get surrounding keyframes for interpolation
                surrounding_frames = [f for f in keyframes if f in group]
                if len(surrounding_frames) < 2:
                    continue
                    
                # Extract coordinates for these keyframes
                x_coords = [group[f].rect.x() for f in surrounding_frames]
                y_coords = [group[f].rect.y() for f in surrounding_frames]
                w_values = [group[f].rect.width() for f in surrounding_frames]
                h_values = [group[f].rect.height() for f in surrounding_frames]
                
                # Determine interpolation method based on available points
                if len(surrounding_frames) >= 4:
                    # Use cubic interpolation
                    x = self._cubic_interpolate(surrounding_frames, x_coords, frame_idx)
                    y = self._cubic_interpolate(surrounding_frames, y_coords, frame_idx)
                    w = self._cubic_interpolate(surrounding_frames, w_values, frame_idx)
                    h = self._cubic_interpolate(surrounding_frames, h_values, frame_idx)
                elif len(surrounding_frames) == 3:
                    # Use quadratic interpolation
                    x = self._quadratic_interpolate(surrounding_frames, x_coords, frame_idx)
                    y = self._quadratic_interpolate(surrounding_frames, y_coords, frame_idx)
                    w = self._quadratic_interpolate(surrounding_frames, w_values, frame_idx)
                    h = self._quadratic_interpolate(surrounding_frames, h_values, frame_idx)
                else:
                    # Use linear interpolation
                    closest_start = max([f for f in surrounding_frames if f <= frame_idx])
                    closest_end = min([f for f in surrounding_frames if f >= frame_idx])
                    alpha = (frame_idx - closest_start) / (closest_end - closest_start)
                    
                    start_x, end_x = group[closest_start].rect.x(), group[closest_end].rect.x()
                    start_y, end_y = group[closest_start].rect.y(), group[closest_end].rect.y()
                    start_w, end_w = group[closest_start].rect.width(), group[closest_end].rect.width()
                    start_h, end_h = group[closest_start].rect.height(), group[closest_end].rect.height()
                    
                    x = int(start_x * (1 - alpha) + end_x * alpha)
                    y = int(start_y * (1 - alpha) + end_y * alpha)
                    w = int(start_w * (1 - alpha) + end_w * alpha)
                    h = int(start_h * (1 - alpha) + end_h * alpha)
                
                # Create interpolated rectangle
                rect = QRect(int(max(0, x)), int(max(0, y)), int(max(1, w)), int(max(1, h)))
                
                # Get attributes and class from nearest keyframe
                distances = [abs(frame_idx - f) for f in surrounding_frames]
                nearest_idx = distances.index(min(distances))
                nearest_frame = surrounding_frames[nearest_idx]
                nearest_ann = group[nearest_frame]
                
                # Calculate confidence score (decreases with distance from keyframes)
                min_dist = min(distances)
                max_dist = (end_frame - start_frame) / 2
                confidence = 0.9 - 0.5 * (min_dist / max_dist)
                confidence = max(0.3, min(0.9, confidence))
                
                # Create annotation
                interpolated_ann = BoundingBox(
                    rect,
                    nearest_ann.class_name,
                    nearest_ann.attributes.copy() if hasattr(nearest_ann, 'attributes') else {},
                    nearest_ann.color,
                    source='smooth_interpolated',
                    score=confidence
                )
                
                frame_annotations.append(interpolated_ann)
            
            # Save interpolated annotations for this frame
            if frame_annotations:
                self.main_window.frame_annotations[frame_idx] = frame_annotations
                
                # Update UI if this is the current frame
                if self.main_window.current_frame == frame_idx:
                    self.main_window.canvas.annotations = frame_annotations
                    self.main_window.canvas.update()
                    if hasattr(self.main_window, 'annotation_dock'):
                        self.main_window.annotation_dock.update_annotation_list()
        
        return True

    def _cubic_interpolate(self, x, y, x_interp):
        """Cubic interpolation for smooth curves with 4+ points"""
        # Convert to numpy arrays for easier calculation
        x_array = np.array(x, dtype=float)
        y_array = np.array(y, dtype=float)
        
        # Fit a cubic polynomial
        z = np.polyfit(x_array, y_array, 3)
                # Fit a cubic polynomial
        z = np.polyfit(x_array, y_array, 3)
        p = np.poly1d(z)
        
        # Evaluate at interpolation point
        return int(p(x_interp))

    def _quadratic_interpolate(self, x, y, x_interp):
        """Quadratic interpolation for smoother curves with 3 points"""
        # Convert to numpy arrays
        x_array = np.array(x, dtype=float)
        y_array = np.array(y, dtype=float)
        
        # Fit a quadratic polynomial
        z = np.polyfit(x_array, y_array, 2)
        p = np.poly1d(z)
        
        # Evaluate at interpolation point
        return int(p(x_interp))

    def _match_annotations(self, start_annotations, end_annotations):
        """
        Match annotations between start and end frames based on class and IOU.

        Args:
            start_annotations: List of annotations from the start frame
            end_annotations: List of annotations from the end frame

        Returns:
            list: List of tuples (start_annotation, end_annotation) of matched annotations
        """
        matched_pairs = []

        # Group annotations by class
        start_by_class = {}
        end_by_class = {}

        for ann in start_annotations:
            if ann.class_name not in start_by_class:
                start_by_class[ann.class_name] = []
            start_by_class[ann.class_name].append(ann)

        for ann in end_annotations:
            if ann.class_name not in end_by_class:
                end_by_class[ann.class_name] = []
            end_by_class[ann.class_name].append(ann)

        # For each class, match annotations by IOU or position
        for class_name in set(start_by_class.keys()) & set(end_by_class.keys()):
            start_anns = start_by_class[class_name]
            end_anns = end_by_class[class_name]

            # If only one annotation of this class in both frames, match them
            if len(start_anns) == 1 and len(end_anns) == 1:
                matched_pairs.append((start_anns[0], end_anns[0]))
                continue

            # Otherwise, match by IOU or center distance
            used_end_anns = set()
            for start_ann in start_anns:
                best_match = None
                best_score = -1  # Higher is better

                for i, end_ann in enumerate(end_anns):
                    if i in used_end_anns:
                        continue

                    # Calculate IOU
                    iou = self._calculate_iou(start_ann.rect, end_ann.rect)
                    if iou > best_score:
                        best_score = iou
                        best_match = i

                # If we found a match with reasonable IOU
                if best_match is not None and best_score > 0.1:
                    matched_pairs.append((start_ann, end_anns[best_match]))
                    used_end_anns.add(best_match)

        return matched_pairs

    def _group_annotations_across_frames(self, frame_list):
        """
        Group matching annotations across multiple frames.
        
        Args:
            frame_list: List of frame numbers to match
            
        Returns:
            list: List of dictionaries mapping frame numbers to matching annotations
        """
        # Get all annotations for each frame
        frame_annotations = {
            frame: self.main_window.frame_annotations[frame]
            for frame in frame_list if frame in self.main_window.frame_annotations
        }
        
        # Group annotations by class first
        class_groups = {}
        for frame, annotations in frame_annotations.items():
            for ann in annotations:
                if ann.class_name not in class_groups:
                    class_groups[ann.class_name] = {}
                if frame not in class_groups[ann.class_name]:
                    class_groups[ann.class_name][frame] = []
                class_groups[ann.class_name][frame].append(ann)
        
        # For each class, match annotations across frames
        annotation_groups = []
        
        for class_name, frames_dict in class_groups.items():
            # Sort frames
            sorted_frames = sorted(frames_dict.keys())
            
            # If only one annotation per frame for this class, matching is simple
            if all(len(frames_dict[f]) == 1 for f in sorted_frames):
                group = {f: frames_dict[f][0] for f in sorted_frames}
                annotation_groups.append(group)
                continue
            
            # For multiple annotations of same class, use IOU tracking through frames
            # Start with each annotation in the first frame
            first_frame = sorted_frames[0]
            for ann in frames_dict[first_frame]:
                group = {first_frame: ann}
                
                # For each subsequent frame
                for i in range(1, len(sorted_frames)):
                    prev_frame = sorted_frames[i-1]
                    curr_frame = sorted_frames[i]
                    
                    # Find best matching annotation in current frame
                    prev_ann = group[prev_frame]
                    best_match = None
                    best_iou = 0.1  # Minimum IOU threshold
                    
                    for curr_ann in frames_dict[curr_frame]:
                        iou = self._calculate_iou(prev_ann.rect, curr_ann.rect)
                        if iou > best_iou:
                            best_iou = iou
                            best_match = curr_ann
                    
                    if best_match:
                        group[curr_frame] = best_match
                    else:
                        # Break the chain if no good match found
                        break
                
                # Only add groups with annotations for at least 2 frames
                if len(group) >= 2:
                    annotation_groups.append(group)
        
        return annotation_groups

    def _interpolate_annotation(self, start_ann, end_ann, alpha):
        """
        Interpolate between two annotations.

        Args:
            start_ann: Starting annotation
            end_ann: Ending annotation
            alpha: Interpolation factor (0 to 1)

        Returns:
            BoundingBox: Interpolated annotation
        """
        # Interpolate rectangle
        start_rect = start_ann.rect
        end_rect = end_ann.rect

        # Linear interpolation of rectangle coordinates
        x = int(start_rect.x() * (1 - alpha) + end_rect.x() * alpha)
        y = int(start_rect.y() * (1 - alpha) + end_rect.y() * alpha)
        width = int(start_rect.width() * (1 - alpha) + end_rect.width() * alpha)
        height = int(start_rect.height() * (1 - alpha) + end_rect.height() * alpha)

        # Create interpolated rectangle
        interpolated_rect = QRect(x, y, width, height)

        # Calculate confidence score that decreases as we move away from keyframes
        # The score will be highest at keyframes (alpha=0 or alpha=1) and lowest in the middle (alpha=0.5)
        confidence_score = 0.8 - 3.2 * alpha * (1.0 - alpha)
        
        # Interpolate attributes
        interpolated_attributes = self._interpolate_attributes(start_ann.attributes, end_ann.attributes, alpha)
        
        # Create new annotation with interpolated rectangle and confidence score
        interpolated_ann = BoundingBox(
            interpolated_rect,
            start_ann.class_name,
            interpolated_attributes,
            start_ann.color,
            source='interpolated',
            score=confidence_score
        )

        return interpolated_ann

    def _interpolate_attributes(self, start_attrs, end_attrs, alpha):
        """
        Interpolate between two sets of attributes.

        Args:
            start_attrs: Starting attributes dictionary
            end_attrs: Ending attributes dictionary
            alpha: Interpolation factor (0 to 1)

        Returns:
            dict: Interpolated attributes
        """
        # Handle None values
        if start_attrs is None:
            start_attrs = {}
        if end_attrs is None:
            end_attrs = {}
            
        interpolated_attrs = {}

        # Get all attribute keys from both dictionaries
        all_keys = set(start_attrs.keys()) | set(end_attrs.keys())

        for key in all_keys:
            # If key exists in both dictionaries
            if key in start_attrs and key in end_attrs:
                start_val = start_attrs[key]
                end_val = end_attrs[key]

                # Interpolate based on attribute type
                if isinstance(start_val, (int, float)) and isinstance(end_val, (int, float)):
                    # Linear interpolation for numeric values
                    interpolated_val = start_val * (1 - alpha) + end_val * alpha
                    # Convert to int if both original values were ints
                    if isinstance(start_val, int) and isinstance(end_val, int):
                        interpolated_val = int(round(interpolated_val))
                    interpolated_attrs[key] = interpolated_val
                elif isinstance(start_val, bool) and isinstance(end_val, bool):
                    # For boolean, use the value that has higher weight based on alpha
                    interpolated_attrs[key] = end_val if alpha > 0.5 else start_val
                else:
                    # For strings or mixed types, use the value that has higher weight
                    interpolated_attrs[key] = end_val if alpha > 0.5 else start_val
            # If key exists only in one dictionary, use that value
            elif key in start_attrs:
                interpolated_attrs[key] = start_attrs[key]
            else:
                interpolated_attrs[key] = end_attrs[key]

        return interpolated_attrs

    def _calculate_iou(self, rect1, rect2):
        """
        Calculate Intersection over Union between two rectangles.
        
        Args:
            rect1: First QRect
            rect2: Second QRect
            
        Returns:
            float: IoU value between 0 and 1
        """
        # Get coordinates
        x1, y1, w1, h1 = rect1.x(), rect1.y(), rect1.width(), rect1.height()
        x2, y2, w2, h2 = rect2.x(), rect2.y(), rect2.width(), rect2.height()
        
        # Calculate intersection coordinates
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        # No intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        # Calculate areas
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        rect1_area = w1 * h1
        rect2_area = w2 * h2
        union_area = rect1_area + rect2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / float(union_area) if union_area > 0 else 0.0
        return iou

    def is_keyframe(self, frame_num=None):
        """
        Check if the given frame is a keyframe in the current interpolation
        cycle (anchor or target). Falls back to 'has annotations' when no
        cycle is active, so the UI indicator keeps working.
        """
        if frame_num is None:
            frame_num = self.main_window.current_frame

        if self.cycle is not None:
            if frame_num == self.cycle["anchor"] or frame_num == self.cycle["target"]:
                return True

        return (
            frame_num in self.main_window.frame_annotations
            and len(self.main_window.frame_annotations[frame_num]) > 0
        )

    def _resolve_annotation_conflicts_with_weighted_average(self, existing_annotations, interpolated_annotations, frame_num):
        """
        Resolve conflicts between existing and interpolated annotations using confidence scores
        and weighted averaging when scores are similar.
        
        Args:
            existing_annotations: List of existing annotations in the frame
            interpolated_annotations: List of newly interpolated annotations
            frame_num: Frame number
            
        Returns:
            list: Final list of annotations after conflict resolution
        """
        # If no existing annotations, just use interpolated ones
        if not existing_annotations:
            return interpolated_annotations
            
        # If no interpolated annotations, keep existing ones
        if not interpolated_annotations:
            return existing_annotations
            
        final_annotations = []
        used_existing = set()
        used_interpolated = set()
        
        # Compare each pair of existing and interpolated annotations
        for i, existing in enumerate(existing_annotations):
            for j, interpolated in enumerate(interpolated_annotations):
                # Skip if either annotation has already been used
                if i in used_existing or j in used_interpolated:
                    continue
                    
                # Calculate IoU between annotations
                iou = self._calculate_iou(existing.rect, interpolated.rect)
                
                if iou > 0.5:  # High overlap - potential conflict
                    # Get confidence scores (default to 0.5 if not present)
                    existing_conf = getattr(existing, 'score', 0.5)
                    interpolated_conf = getattr(interpolated, 'score', 0.5)
                    existing_source = getattr(existing, 'source', 'detected')
                    interpolated_source = getattr(interpolated, 'source', 'interpolated')
                    
                    # If existing is manual, always keep it
                    if existing_source == 'manual':
                        final_annotations.append(existing)
                    # If interpolated is manual (unlikely), keep it
                    elif interpolated_source == 'manual':
                        final_annotations.append(interpolated)
                    # NEW: Handle detected vs interpolated conflict
                    elif existing_source == 'detected' and interpolated_source == 'interpolated':
                        # Use interpolated attributes but create a hybrid annotation
                        hybrid_ann = self._create_hybrid_annotation(existing, interpolated)
                        # Change the source type to 'interpolated' as requested
                        hybrid_ann.source = 'interpolated'
                        final_annotations.append(hybrid_ann)
                    # If scores are similar (within 0.2 of each other), use weighted average
                    elif abs(existing_conf - interpolated_conf) < 0.2:
                        # Create a weighted average annotation
                        weighted_ann = self._create_weighted_average_annotation(
                            existing, interpolated, existing_conf, interpolated_conf
                        )
                        final_annotations.append(weighted_ann)
                    # Otherwise, use the one with higher confidence and preserve its source
                    elif existing_conf >= interpolated_conf:
                        # Keep the existing annotation with its original source
                        final_annotations.append(existing)
                    else:
                        # Use the interpolated annotation with 'interpolated' source
                        final_annotations.append(interpolated)
                        
                    used_existing.add(i)
                    used_interpolated.add(j)
        
        # Add remaining existing annotations that didn0t have conflicts
        for i, annotation in enumerate(existing_annotations):
            if i not in used_existing:
                final_annotations.append(annotation)
                
        # Add remaining interpolated annotations that didn't have conflicts
        for j, annotation in enumerate(interpolated_annotations):
            if j not in used_interpolated:
                final_annotations.append(annotation)
                
        return final_annotations

    def _create_hybrid_annotation(self, detected_ann, interpolated_ann):
        """
        Create a hybrid annotation that uses the detected bounding box but interpolated attributes.
        
        Args:
            detected_ann: The detected annotation (for bounding box)
            interpolated_ann: The interpolated annotation (for attributes)
        
        Returns:
            BoundingBox: New hybrid annotation
        """
        # Use the detected bounding box (as it's likely more accurate)
        hybrid_rect = detected_ann.rect
        
        # Use interpolated attributes (as requested)
        hybrid_attributes = {}
        if hasattr(interpolated_ann, 'attributes') and interpolated_ann.attributes:
            hybrid_attributes = interpolated_ann.attributes.copy()
        
        # If detected annotation has attributes that interpolated doesn't have, add them
        if hasattr(detected_ann, 'attributes') and detected_ann.attributes:
            for key, value in detected_ann.attributes.items():
                if key not in hybrid_attributes:
                    hybrid_attributes[key] = value
        
        # Use class name from detected (more reliable)
        class_name = detected_ann.class_name
        
        # Use color from detected annotation
        color = detected_ann.color
        
        # Calculate hybrid confidence score (average of both, slightly boosted for agreement)
        detected_conf = getattr(detected_ann, 'score', 0.5)
        interpolated_conf = getattr(interpolated_ann, 'score', 0.5)
        hybrid_score = min(1.0, (detected_conf + interpolated_conf) / 2 * 1.1)
        
        # Create hybrid annotation
        from .annotation import BoundingBox
        hybrid_ann = BoundingBox(
            hybrid_rect,
            class_name,
            hybrid_attributes,
            color,
            source='interpolated',  # Set to interpolated as requested
            score=hybrid_score
        )
        
        return hybrid_ann

    def _create_weighted_average_annotation(self, ann1, ann2, score1, score2):
        """
        Create a new annotation that is a weighted average of two annotations.
        
        Args:
            ann1: First annotation
            ann2: Second annotation
            score1: Confidence score of first annotation
            score2: Confidence score of second annotation
            
        Returns:
            BoundingBox: New annotation with weighted average properties
        """
        # Calculate weights based on scores
        total_score = score1 + score2
        weight1 = score1 / total_score
        weight2 = score2 / total_score
        
        # Get rectangle coordinates
        x1, y1, w1, h1 = ann1.rect.x(), ann1.rect.y(), ann1.rect.width(), ann1.rect.height()
        x2, y2, w2, h2 = ann2.rect.x(), ann2.rect.y(), ann2.rect.width(), ann2.rect.height()
        
        # Calculate weighted average coordinates
        x = int(x1 * weight1 + x2 * weight2)
        y = int(y1 * weight1 + y2 * weight2)
        w = int(w1 * weight1 + w2 * weight2)
        h = int(h1 * weight1 + h2 * weight2)
        
        # Create new rectangle
        from PyQt5.QtCore import QRect
        weighted_rect = QRect(x, y, w, h)
        
        # Use class name from annotation with higher score
        class_name = ann1.class_name if score1 >= score2 else ann2.class_name
        
        # Merge attributes with preference to the higher-scored annotation
        attributes = {}
        if hasattr(ann1, 'attributes') and ann1.attributes:
            attributes.update(ann1.attributes)
        if hasattr(ann2, 'attributes') and ann2.attributes:
            # For attributes in both annotations, use weighted average for numeric values
            for key, value2 in ann2.attributes.items():
                if key in attributes:
                    value1 = attributes[key]
                    if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
                        attributes[key] = value1 * weight1 + value2 * weight2
                        if isinstance(value1, int) and isinstance(value2, int):
                            attributes[key] = int(round(attributes[key]))
                    elif score2 > score1:  # For non-numeric, prefer the higher-scored annotation
                        attributes[key] = value2
                else:
                    attributes[key] = value2
        
        # Use color from the class
        color = ann1.color
        
        # Calculate combined score (slightly higher than average to reward agreement)
        combined_score = min(1.0, (score1 + score2) / 2 * 1.1)
        
        # Create new annotation
        from .annotation import BoundingBox
        weighted_ann = BoundingBox(
            weighted_rect,
            class_name,
            attributes,
            color,
            source='interpolated',
            score=combined_score
        )
        
        return weighted_ann

    def start_workflow(self, start_frame):
        """Deprecated. The workflow now starts automatically when you label
        a frame and press Next while interpolation mode is active."""
        self.reset_cycle()
        return

    def advance_workflow(self, user_action=None):
        """Deprecated. Use Next/Prev; the workflow advances automatically."""
        return

    def has_annotation(self, frame):
        anns = self.main_window.frame_annotations
        return frame in anns and len(anns[frame]) > 0

    def find_next_annotated_frame(self, start):
        anns = self.main_window.frame_annotations
        total = getattr(self.main_window, "total_frames", 999999)
        for f in range(start + 1, total):
            if self.has_annotation(f):
                return f
        return None

    def find_prev_annotated_frame(self, start):
        anns = self.main_window.frame_annotations
        for f in range(start - 1, -1, -1):
            if self.has_annotation(f):
                return f
        return None

    def get_next_frame_for_workflow(self, current_frame):
        """Deprecated alias for get_next_frame(). Kept for older callers."""
        return self.get_next_frame(current_frame)

    def _interpolate_and_queue_review(self, state):
        """Deprecated no-op (old workflow helper)."""
        return
