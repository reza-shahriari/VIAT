"""
Interpolation module for VIAT.

This module provides functionality to interpolate annotations between keyframes,
reducing the manual annotation workload by automatically generating annotations
for intermediate frames.
"""

from PyQt5.QtCore import QRect
from .annotation import BoundingBox


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
        self.last_annotated_frame = None  # Last frame that was annotated
        self.pending_interpolation = False  # Flag to indicate pending interpolation

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

    def interpolate_annotations(self, start_frame, end_frame):
        """
        Interpolate annotations between two keyframes.

        Args:
            start_frame: The starting keyframe
            end_frame: The ending keyframe

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

        return True

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

                    # Calculate IOU or center distance
                    iou = self._calculate_iou(start_ann.rect, end_ann.rect)
                    if iou > best_score:
                        best_score = iou
                        best_match = i

                # If we found a match with reasonable IOU
                if best_match is not None and best_score > 0.1:
                    matched_pairs.append((start_ann, end_anns[best_match]))
                    used_end_anns.add(best_match)

        return matched_pairs

    def perform_interpolation(self, start_frame, end_frame, start_annotations, end_annotations):
        """
        Perform interpolation between two keyframes with conflict resolution.
        
        Args:
            start_frame: Starting keyframe number
            end_frame: Ending keyframe number
            start_annotations: List of annotations in the start frame
            end_annotations: List of annotations in the end frame
        """
        # Skip if frames are adjacent
        if end_frame - start_frame <= 1:
            return
            
        # For each frame between start and end
        for frame_num in range(start_frame + 1, end_frame):
            # Skip if frame already has manual annotations
            if self._has_manual_annotations(frame_num):
                continue
                
            # Get existing annotations (could be detected or previously interpolated)
            existing_annotations = self._get_frame_annotations(frame_num)
            
            # Create interpolated annotations for this frame
            interpolated_annotations = self._interpolate_annotations(
                start_frame, end_frame, frame_num, 
                start_annotations, end_annotations
            )
            
            # Resolve conflicts between existing and interpolated annotations using weighted average
            final_annotations = self._resolve_annotation_conflicts_with_weighted_average(
                existing_annotations, interpolated_annotations, frame_num
            )
            
            # Update frame with final annotations
            self.main_window.frame_annotations[frame_num] = final_annotations
            
            # If current frame is being displayed, update canvas
            if self.main_window.current_frame == frame_num:
                self.main_window.canvas.annotations = final_annotations
                self.main_window.canvas.update()
                self.main_window.update_annotation_list()

    def _has_manual_annotations(self, frame_num):
        """
        Check if a frame has manual annotations.
        
        Args:
            frame_num: Frame number to check
            
        Returns:
            bool: True if frame has manual annotations
        """
        # This would require tracking which annotations are manual
        # You could add a 'source' attribute to annotations (manual, detected, interpolated)
        if frame_num not in self.main_window.frame_annotations:
            return False
            
        for annotation in self.main_window.frame_annotations[frame_num]:
            if hasattr(annotation, 'source') and annotation.source == 'manual':
                return True
        return False

    def _get_frame_annotations(self, frame_num):
        """Get existing annotations for a frame."""
        if frame_num in self.main_window.frame_annotations:
            return self.main_window.frame_annotations[frame_num].copy()
        return []

    def _resolve_annotation_conflicts(self, existing_annotations, interpolated_annotations, frame_num):
        """
        Resolve conflicts between existing and interpolated annotations using confidence scores.
        
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
                iou = self._calculate_iou(existing, interpolated)
                
                if iou > 0.5:  # High overlap - potential conflict
                    # Get confidence scores (default to 0.5 if not present)
                    existing_conf = getattr(existing, 'confidence', 0.5)
                    interpolated_conf = getattr(interpolated, 'confidence', 0.5)
                    
                    # If existing is manual, always keep it
                    if getattr(existing, 'source', '') == 'manual':
                        final_annotations.append(existing)
                    # If interpolated is manual (unlikely), keep it
                    elif getattr(interpolated, 'source', '') == 'manual':
                        final_annotations.append(interpolated)
                    # Otherwise, use the one with higher confidence
                    elif existing_conf >= interpolated_conf:
                        final_annotations.append(existing)
                    else:
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

    def _calculate_iou(self, box1, box2):
        """
        Calculate Intersection over Union between two bounding boxes.
        
        Args:
            box1: First annotation
            box2: Second annotation
            
        Returns:
            float: IoU value between 0 and 1
        """
        # Get coordinates of both boxes
        x1, y1, w1, h1 = box1.rect.x(), box1.rect.y(), box1.rect.width(), box1.rect.height()
        x2, y2, w2, h2 = box2.rect.x(), box2.rect.y(), box2.rect.width(), box2.rect.height()
        
        # Calculate coordinates of intersection
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        # No intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        # Calculate area of intersection
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate area of both boxes
        box1_area = w1 * h1
        box2_area = w2 * h2
        
        # Calculate IoU
        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou

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
        # Using a quadratic function: score = 1 - 4 * alpha * (1 - alpha)
        # This gives score=0.8 at alpha=0 or alpha=1, and score=0.0 at alpha=0.5
        confidence_score = 0.8 - 3.2 * alpha * (1.0 - alpha)
        
        # Get attributes from start annotation
        interpolated_attributes = self._interpolate_attributes(start_ann.attributes, end_ann.attributes, alpha)
        
        # Add confidence score to attributes
        if interpolated_attributes is None:
            interpolated_attributes = {}

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

    def is_keyframe(self, frame_num=None):
        """
        Check if the specified frame (or current frame) is a keyframe.
        In our simplified approach, any frame with annotations is considered a keyframe.
        
        Args:
            frame_num: The frame number to check, or None for current frame
            
        Returns:
            bool: True if the frame is a keyframe (has annotations), False otherwise
        """
        if frame_num is None:
            frame_num = self.main_window.current_frame
            
        # In our simplified approach, a frame is a keyframe if:
        # 1. It has annotations
        # 2. It's not an interpolated frame (it was manually annotated)
        
        # Check if frame has annotations
        has_annotations = (
            frame_num in self.main_window.frame_annotations and 
            len(self.main_window.frame_annotations[frame_num]) > 0
        )
        
        # For simplicity, we'll consider the last annotated frame as a keyframe
        is_last_annotated = frame_num == self.last_annotated_frame
        
        return has_annotations and is_last_annotated

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
                    
                    # If existing is manual, always keep it
                    if getattr(existing, 'source', '') == 'manual':
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
                    # Otherwise, use the one with higher confidence
                    elif existing_conf >= interpolated_conf:
                        final_annotations.append(existing)
                    else:
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
            source='weighted_average',
            score=combined_score
        )
        
        return weighted_ann
