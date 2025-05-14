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
        self.interval = 5  # Default interval between keyframes
        self.last_annotated_frame = None
        self.pending_interpolation = False

    def set_active(self, active):
        """Enable or disable interpolation mode."""
        self.is_active = active
        if active:
            self.main_window.statusBar.showMessage(
                f"Interpolation mode active. Interval: {self.interval} frames."
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

    def on_frame_annotated(self, frame_num):
        """
        Called when a frame is annotated.
        
        Args:
            frame_num: The frame number that was annotated
        """
        if not self.is_active:
            return
            
        if self.last_annotated_frame is None:
            self.last_annotated_frame = frame_num
            next_keyframe = frame_num + self.interval
            
            # Make sure it's within video bounds
            if hasattr(self.main_window, 'total_frames'):
                next_keyframe = min(next_keyframe, self.main_window.total_frames - 1)
                
            self.main_window.statusBar.showMessage(
                f"Frame {frame_num} annotated. Please annotate frame {next_keyframe} next."
            )
        else:
            # This is the second or later annotated frame
            # Check if we should interpolate
            start_frame = self.last_annotated_frame
            end_frame = frame_num
            
            # Only interpolate if frames are interval distance apart
            if abs(end_frame - start_frame) >= 2:
                # Set flag for pending interpolation
                self.pending_interpolation = True
                self.main_window.statusBar.showMessage(
                    f"Ready to interpolate between frames {start_frame} and {end_frame}. "
                    f"Move to another frame or click 'Interpolate' to apply."
                )
            
            # Update last annotated frame
            self.last_annotated_frame = frame_num

    def check_pending_interpolation(self, new_frame):
        """
        Check if there's a pending interpolation when changing frames.
        
        Args:
            new_frame: The new frame being navigated to
        """
        if not self.is_active or not self.pending_interpolation:
            return
            
        # If we're moving away from the last annotated frame, perform interpolation
        if new_frame != self.last_annotated_frame:
            self.perform_pending_interpolation()

    def perform_pending_interpolation(self):
        """Perform any pending interpolation."""
        if not self.pending_interpolation or self.last_annotated_frame is None:
            return False
            
        # Find the previous annotated frame
        prev_frame = None
        for frame in sorted(self.main_window.frame_annotations.keys()):
            if frame < self.last_annotated_frame and len(self.main_window.frame_annotations[frame]) > 0:
                prev_frame = frame
                
        if prev_frame is None:
            self.pending_interpolation = False
            return False
            
        # Perform interpolation
        success = self.interpolate_annotations(prev_frame, self.last_annotated_frame)
        self.pending_interpolation = False
        
        if success:
            self.main_window.statusBar.showMessage(
                f"Interpolated annotations between frames {prev_frame} and {self.last_annotated_frame}."
            )
            
            # Suggest next frame to annotate
            next_keyframe = self.last_annotated_frame + self.interval
            
            # Make sure it's within video bounds
            if hasattr(self.main_window, 'total_frames'):
                next_keyframe = min(next_keyframe, self.main_window.total_frames - 1)
                
        return success

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
        Check if the specified frame is a keyframe (has annotations).
        
        Args:
            frame_num: Frame number to check, or None for current frame
            
        Returns:
            bool: True if frame is a keyframe
        """
        if frame_num is None:
            frame_num = self.main_window.current_frame
            
        # A frame is a keyframe if it has annotations
        has_annotations = (
            frame_num in self.main_window.frame_annotations and 
            len(self.main_window.frame_annotations[frame_num]) > 0
        )
        
        return has_annotations

    
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
                    existing_source = getattr(existing, 'source', 'detected')  # Default to 'detected' if not specified
                    
                    # If existing is manual, always keep it
                    if existing_source == 'manual':
                        final_annotations.append(existing)
                    # If interpolated is manual (unlikely), keep it
                    elif getattr(interpolated, 'source', '') == 'manual':
                        final_annotations.append(interpolated)
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
        
        # Add remaining existing annotations that didn't have conflicts
        for i, annotation in enumerate(existing_annotations):
            if i not in used_existing:
                final_annotations.append(annotation)
                
        # Add remaining interpolated annotations that didn't have conflicts
        for j, annotation in enumerate(interpolated_annotations):
            if j not in used_interpolated:
                final_annotations.append(annotation)
                
        return final_annotations

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

    def get_next_frame(self, current_frame, annotation=None):
        """
        Determine the next frame to navigate to based on the current frame.
        
        Workflow:
        1. Start at current frame, get annotation
        2. Jump to frame+2, get annotation
        3. Interpolate frame+1, show it to user
        4. Jump to frame+4, get annotation
        5. Interpolate frame+3, show it to user
        6. Jump to frame+9 (based on interval), get annotation
        7. Interpolate frames 5-8, show them to user in sequence
        8. Jump to frame+14 (based on 2*interval), get annotation
        9. And so on...
        
        Args:
            current_frame: The current frame number
            annotation: Optional annotation for the current frame
            
        Returns:
            int: The next frame number to navigate to
        """
        if not self.is_active:
            return current_frame + 1
        
        # Get all keyframes (frames with annotations)
        keyframes = sorted([
            f for f in self.main_window.frame_annotations.keys() 
            if len(self.main_window.frame_annotations[f]) > 0
        ])
        
        # Track which frames we've visited for interpolation review
        if not hasattr(self.main_window, 'interpolation_visited'):
            self.main_window.interpolation_visited = set()
        
        # Track the last keyframe we annotated
        if not hasattr(self.main_window, 'last_annotated_keyframe'):
            self.main_window.last_annotated_keyframe = None
        
        # Check if current frame is a keyframe
        is_keyframe = current_frame in keyframes
        
        # If current frame has no annotations but should be a keyframe, stay on it
        if current_frame not in keyframes and self.main_window.last_annotated_keyframe is not None:
            # Check if this is a frame we should annotate as a keyframe
            if current_frame == self.main_window.last_annotated_keyframe + 2 or \
            current_frame == self.main_window.last_annotated_keyframe + 4 or \
            current_frame == self.main_window.last_annotated_keyframe + self.interval + 4 or \
            current_frame == self.main_window.last_annotated_keyframe + 2 * self.interval:
                # This should be a keyframe, stay on it for annotation
                # Even if there's nothing to annotate, the user should confirm this
                return current_frame
        
        # If current frame is a keyframe or just got annotated
        if is_keyframe or annotation is not None:
            # Update the last annotated keyframe
            self.main_window.last_annotated_keyframe = current_frame
            
            # Determine the next keyframe to navigate to
            if len(keyframes) <= 1:
                # This is the first keyframe, go to frame+2
                next_keyframe = current_frame + 2
            elif current_frame == keyframes[0] + 2:
                # This is the second keyframe, go to frame+1 for interpolation
                # Interpolate between first and second keyframe
                self.interpolate_annotations(keyframes[0], current_frame)
                return keyframes[0] + 1
            elif current_frame == keyframes[0] + 4:
                # This is the third keyframe, go to frame+3 for interpolation
                # Interpolate between second and third keyframe
                self.interpolate_annotations(keyframes[0] + 2, current_frame)
                return keyframes[0] + 3
            elif current_frame >= keyframes[0] + 9:
                # This is a later keyframe, go to the first interpolated frame after the previous keyframe
                prev_keyframes = [kf for kf in keyframes if kf < current_frame]
                if prev_keyframes:
                    prev_keyframe = max(prev_keyframes)
                    # Interpolate between previous keyframe and current keyframe
                    self.interpolate_annotations(prev_keyframe, current_frame)
                    # Go to the first frame after previous keyframe
                    return prev_keyframe + 1
            
            # Default: go to next keyframe based on pattern
            if current_frame == keyframes[0]:
                # First keyframe, go to frame+2
                next_keyframe = current_frame + 2
            elif current_frame == keyframes[0] + 2:
                # Second keyframe, go to frame+4
                next_keyframe = keyframes[0] + 4
            elif current_frame == keyframes[0] + 4:
                # Third keyframe, go to frame+9 (or based on interval)
                next_keyframe = keyframes[0] + 4 + self.interval
            else:
                # Later keyframes, go to frame+2*interval
                next_keyframe = current_frame + 2 * self.interval
            
            # Make sure it's within video bounds
            if hasattr(self.main_window, 'total_frames'):
                next_keyframe = min(next_keyframe, self.main_window.total_frames - 1)
            
            return next_keyframe
        
        # If current frame is an interpolated frame
        else:
            # Mark this frame as visited
            self.main_window.interpolation_visited.add(current_frame)
            
            # Find surrounding keyframes
            prev_keyframes = [kf for kf in keyframes if kf < current_frame]
            next_keyframes = [kf for kf in keyframes if kf > current_frame]
            
            prev_keyframe = max(prev_keyframes) if prev_keyframes else None
            next_keyframe = min(next_keyframes) if next_keyframes else None
            
            # If we have surrounding keyframes
            if prev_keyframe is not None and next_keyframe is not None:
                # Check if there are more interpolated frames to visit
                for frame in range(current_frame + 1, next_keyframe):
                    if frame not in self.main_window.interpolation_visited:
                        # Go to next unvisited interpolated frame
                        return frame
                
                # All interpolated frames visited, go to next keyframe in pattern
                if next_keyframe == keyframes[0] + 4:
                    # After reviewing interpolated frames between 2nd and 3rd keyframe,
                    # go to frame+9 (or based on interval)
                    next_pattern_keyframe = keyframes[0] + 4 + self.interval
                else:
                    # After reviewing other interpolated frames, go to frame+2*interval
                    next_pattern_keyframe = next_keyframe + 2 * self.interval
                
                # Make sure it's within video bounds
                if hasattr(self.main_window, 'total_frames'):
                    next_pattern_keyframe = min(next_pattern_keyframe, self.main_window.total_frames - 1)
                
                return next_pattern_keyframe
            
            # If we don't have surrounding keyframes
            elif next_keyframe is not None:
                # Go to the next keyframe
                return next_keyframe
            elif prev_keyframe is not None:
                # Go to the next keyframe in pattern
                if prev_keyframe == keyframes[0]:
                    # After first keyframe, go to frame+2
                    return prev_keyframe + 2
                elif prev_keyframe == keyframes[0] + 2:
                    # After second keyframe, go to frame+4
                    return keyframes[0] + 4
                elif prev_keyframe == keyframes[0] + 4:
                    # After third keyframe, go to frame+9 (or based on interval)
                    next_pattern_keyframe = keyframes[0] + 4 + self.interval
                else:
                    # After later keyframes, go to frame+2*interval
                    next_pattern_keyframe = prev_keyframe + 2 * self.interval
                
                # Make sure it's within video bounds
                if hasattr(self.main_window, 'total_frames'):
                    next_pattern_keyframe = min(next_pattern_keyframe, self.main_window.total_frames - 1)
                
                return next_pattern_keyframe
            else:
                # No keyframes at all, move to next frame
                return current_frame + 1
