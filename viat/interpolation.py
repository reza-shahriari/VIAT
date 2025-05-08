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
            
        # If this is the first annotated frame, store it and jump to the next keyframe
        if self.last_annotated_frame is None:
            self.last_annotated_frame = frame_num
            next_keyframe = frame_num + self.interval
            
            # Make sure it's within video bounds
            if hasattr(self.main_window, 'total_frames'):
                next_keyframe = min(next_keyframe, self.main_window.total_frames - 1)
                
            # Jump to the next keyframe
            self.main_window.set_current_frame(next_keyframe)
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
                
            # Jump to the next keyframe
            self.main_window.set_current_frame(next_keyframe)
            
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

    def _calculate_iou(self, rect1, rect2):
        """
        Calculate Intersection over Union (IOU) between two QRect objects.

        Args:
            rect1: First rectangle
            rect2: Second rectangle

        Returns:
            float: IOU value between 0 and 1
        """
        intersection = rect1.intersected(rect2)
        if intersection.isEmpty():
            return 0

        intersection_area = intersection.width() * intersection.height()
        rect1_area = rect1.width() * rect1.height()
        rect2_area = rect2.width() * rect2.height()
        union_area = rect1_area + rect2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0

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

        # Create new annotation with interpolated rectangle
        interpolated_ann = BoundingBox(
            interpolated_rect,
            start_ann.class_name,
            self._interpolate_attributes(start_ann.attributes, end_ann.attributes, alpha),
            start_ann.color
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
