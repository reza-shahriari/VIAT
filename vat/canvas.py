"""
VideoCanvas class for video annotation tool.

This module provides a PyQt5 widget for displaying video frames and creating/editing annotations.
The canvas supports different annotation methods (drag, two-click), zooming, and various
interaction modes for creating and modifying bounding box annotations.

Key features:
- Display video frames with proper aspect ratio
- Create bounding box annotations with different methods
- Select, resize, and modify existing annotations
- Support for multiple annotation classes with color coding
- Zoom and pan functionality
- Coordinate transformation between display and image space

The VideoCanvas integrates with the main application window to manage annotations
across video frames and update the UI accordingly.
"""


import cv2
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import Qt, QRect, QPoint
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QPixmap, QImage
from annotation import BoundingBox
import random

class VideoCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent  
        self.selected_bbox = None  

        self.pixmap = None
        self.annotations = []  # List of BoundingBox objects
        self.current_annotation = None
        self.drawing = False
        self.setMouseTracking(True)
        self.annotation_type = "box"  # Default annotation type
        self.aspect_ratio = 16/9  # Initial default, will be updated from video
        self.start_point = None
        self.current_point = None
        self.selected_annotation = None
        self.current_class = "Drone"  # Default class
        self.class_colors = {
            "Drone":QColor(0, 255, 255),  
        }
        self.annotation_method = "Drag"  # Default method
        self.is_drawing = False
        self.start_point = None
        self.end_point = None
        self.resize_handle = None
        self.resize_start = None
        
        # Two-click method state
        self.first_click_pos = None
        
        # Edge modification state
        self.edge_resize = None  # Can be "top", "bottom", "left", "right", or None
        self.handle_size = 8  # Size of resize handles in pixels
        
        # Set focus policy to receive keyboard events
        self.setFocusPolicy(Qt.StrongFocus)
        
        self.scale_factor = 1.0  # Default scale (no zoom)
        self.maintain_aspect_ratio = True
    def set_annotation_method(self, method):
        """Set the current annotation method."""
        self.annotation_method = method
        
        # Reset state variables
        self.is_drawing = False
        self.start_point = None
        self.end_point = None
        self.first_click_pos = None
        
        # Update cursor
        if method == "Drag":
            self.setCursor(Qt.ArrowCursor)
        elif method == "Two-Click":
            self.setCursor(Qt.CrossCursor)
        
        # Update the display
        self.update()

    def set_frame(self, frame):
        """Set the current frame to display."""
        if frame is None:
            self.pixmap = None
            return
        
        # Convert OpenCV BGR image to RGB for Qt
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        
        # Create QImage from the RGB data
        q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Create pixmap from QImage
        self.pixmap = QPixmap.fromImage(q_img)
        
        # Reset zoom when loading a new frame
        self.scale_factor = 1.0
        
        # Update widget
        self.update()

    
    def set_current_class(self, class_name):
        """Set the current class for new annotations."""
        self.current_class = class_name
        
        # If the class doesn't exist in the color map, add it with a random color
        if class_name not in self.class_colors:
            self.class_colors[class_name] = QColor(
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )
    def paintEvent(self, event):
        """Paint the frame and annotations."""
        painter = QPainter(self)
        
        # Fill the entire background with a neutral color
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        
        if self.pixmap:
            # Calculate scaled dimensions while maintaining aspect ratio
            pixmap_width = self.pixmap.width()
            pixmap_height = self.pixmap.height()
            
            # Calculate the widget's aspect ratio
            widget_ratio = self.width() / self.height()
            pixmap_ratio = pixmap_width / pixmap_height
            
            # Calculate the display rectangle to fill the widget while maintaining aspect ratio
            if widget_ratio > pixmap_ratio:
                # Widget is wider than the pixmap - fill height
                display_height = self.height()
                display_width = int(display_height * pixmap_ratio)
            else:
                # Widget is taller than the pixmap - fill width
                display_width = self.width()
                display_height = int(display_width / pixmap_ratio)
            
            # Calculate centered position
            x = (self.width() - display_width) // 2
            y = (self.height() - display_height) // 2
            
            # Draw the image with proper scaling
            target_rect = QRect(x, y, display_width, display_height)
            painter.drawPixmap(target_rect, self.pixmap, self.pixmap.rect())
            
            # Store the display rectangle for coordinate transformations
            self.display_rect = target_rect
            # Draw annotations
            for i, annotation in enumerate(self.annotations):
                # Set pen based on selection
                if annotation == self.selected_annotation:
                    pen = QPen(QColor(0, 255, 255), 2, Qt.SolidLine)
                else:
                    pen = QPen(annotation.color, 2, Qt.SolidLine)
                
                painter.setPen(pen)
                
                # Transform annotation coordinates to display coordinates
                display_rect = self.transform_rect_to_display(annotation.rect)
                
                # Draw the bounding box
                painter.drawRect(display_rect)
                
                # Draw class name
                text_rect = QRect(
                    display_rect.x(),
                    display_rect.y() - 20,
                    display_rect.width(),
                    20
                )
                painter.fillRect(text_rect, QColor(0, 0, 0, 128))
                painter.setPen(QColor(255, 255, 255))
                painter.drawText(text_rect, Qt.AlignCenter, annotation.class_name)
                
                # Draw resize handles if selected
                if annotation == self.selected_annotation:
                    self.draw_resize_handles(painter, display_rect)
            
            # Draw current annotation being created
            if self.is_drawing and self.start_point and self.end_point:
                # Set pen for new annotation
                pen = QPen(self.class_colors.get(self.current_class, QColor(255, 0, 0)), 2, Qt.DashLine)
                painter.setPen(pen)
                
                # Transform drawing coordinates to display coordinates
                start_display = self.transform_point_to_display(self.start_point)
                end_display = self.transform_point_to_display(self.end_point)
                
                # Calculate rectangle
                rect = QRect(
                    min(start_display.x(), end_display.x()),
                    min(start_display.y(), end_display.y()),
                    abs(end_display.x() - start_display.x()),
                    abs(end_display.y() - start_display.y())
                )
                
                # Draw the rectangle
                painter.drawRect(rect)
            
            # For two-click method, draw first point marker
            if self.annotation_method == "Two-Click" and self.first_click_pos:
                pen = QPen(self.class_colors.get(self.current_class, QColor(255, 0, 0)), 2, Qt.SolidLine)
                painter.setPen(pen)
                
                # Transform first click position to display coordinates
                first_click_display = self.transform_point_to_display(self.first_click_pos)
                
                # Draw a cross at the first click position
                size = 10
                painter.drawLine(
                    first_click_display.x() - size,
                    first_click_display.y(),
                    first_click_display.x() + size,
                    first_click_display.y()
                )
                painter.drawLine(
                    first_click_display.x(),
                    first_click_display.y() - size,
                    first_click_display.x(),
                    first_click_display.y() + size
                )

    def resizeEvent(self, event):
        """Handle resize events to adjust the video display."""
        super().resizeEvent(event)
        # Call fit_to_view to adjust the scale factor when the widget is resized
        self.fit_to_view()
    
    def fit_to_view(self):
        """Scale the image to fit the widget size."""
        if not self.pixmap:
            return
        
        # Calculate the scale factor to fit the widget
        widget_width = self.width()
        widget_height = self.height()
        
        pixmap_width = self.pixmap.width()
        pixmap_height = self.pixmap.height()
        
        # Calculate scale factors for width and height
        width_scale = widget_width / pixmap_width
        height_scale = widget_height / pixmap_height
        
        # Use the smaller scale factor to ensure the entire image fits
        self.scale_factor = min(width_scale, height_scale)
        
        self.update()
    def draw_resize_handles(self, painter, rect):
        """Draw resize handles for the selected annotation."""
        # Draw handles at corners and edges
        handle_size = self.handle_size
        
        # Set pen and brush for handles
        painter.setPen(QPen(QColor(0, 255, 255), 1, Qt.SolidLine))
        painter.setBrush(QBrush(QColor(0, 255, 255, 128)))
        
        # Top edge
        painter.drawRect(QRect(rect.x() + rect.width() // 2 - handle_size // 2, rect.y() - handle_size // 2, handle_size, handle_size))
        
        # Bottom edge
        painter.drawRect(QRect(rect.x() + rect.width() // 2 - handle_size // 2, rect.y() + rect.height() - handle_size // 2, handle_size, handle_size))
        
        # Left edge
        painter.drawRect(QRect(rect.x() - handle_size // 2, rect.y() + rect.height() // 2 - handle_size // 2, handle_size, handle_size))
        
        # Right edge
        painter.drawRect(QRect(rect.x() + rect.width() - handle_size // 2, rect.y() + rect.height() // 2 - handle_size // 2, handle_size, handle_size))
        
        # Top-left corner
        painter.drawRect(QRect(rect.x() - handle_size // 2, rect.y() - handle_size // 2, handle_size, handle_size))
        
        # Top-right corner
        painter.drawRect(QRect(rect.x() + rect.width() - handle_size // 2, rect.y() - handle_size // 2, handle_size, handle_size))
        
        # Bottom-left corner
        painter.drawRect(QRect(rect.x() - handle_size // 2, rect.y() + rect.height() - handle_size // 2, handle_size, handle_size))
        
        # Bottom-right corner
        painter.drawRect(QRect(rect.x() + rect.width() - handle_size // 2, rect.y() + rect.height() - handle_size // 2, handle_size, handle_size))
        

    def get_resize_handle(self, pos):
        """Determine if position is on a resize handle of the selected annotation."""
        if not self.selected_annotation or not self.pixmap or not hasattr(self, 'display_rect'):
            return None
        
        # Get the rectangle in display coordinates
        rect = self.transform_rect_to_display(self.selected_annotation.rect)
        
        handle_size = self.handle_size
        
        # Check if position is on any of the handles
        # Top edge
        top_handle = QRect(
            rect.x() + rect.width() // 2 - handle_size // 2,
            rect.y() - handle_size // 2,
            handle_size,
            handle_size
        )
        if top_handle.contains(pos):
            return "top"
        
        # Bottom edge
        bottom_handle = QRect(
            rect.x() + rect.width() // 2 - handle_size // 2,
            rect.y() + rect.height() - handle_size // 2,
            handle_size,
            handle_size
        )
        if bottom_handle.contains(pos):
            return "bottom"
        
        # Left edge
        left_handle = QRect(
            rect.x() - handle_size // 2,
            rect.y() + rect.height() // 2 - handle_size // 2,
            handle_size,
            handle_size
        )
        if left_handle.contains(pos):
            return "left"
        
        # Right edge
        right_handle = QRect(
            rect.x() + rect.width() - handle_size // 2,
            rect.y() + rect.height() // 2 - handle_size // 2,
            handle_size,
            handle_size
        )
        if right_handle.contains(pos):
            return "right"
        
        # Top-left corner
        top_left_handle = QRect(
            rect.x() - handle_size // 2,
            rect.y() - handle_size // 2,
            handle_size,
            handle_size
        )
        if top_left_handle.contains(pos):
            return "top-left"
        
        # Top-right corner
        top_right_handle = QRect(
            rect.x() + rect.width() - handle_size // 2,
            rect.y() - handle_size // 2,
            handle_size,
            handle_size
        )
        if top_right_handle.contains(pos):
            return "top-right"
        
        # Bottom-left corner
        bottom_left_handle = QRect(
            rect.x() - handle_size // 2,
            rect.y() + rect.height() - handle_size // 2,
            handle_size,
            handle_size
        )
        if bottom_left_handle.contains(pos):
            return "bottom-left"
        
        # Bottom-right corner
        bottom_right_handle = QRect(
            rect.x() + rect.width() - handle_size // 2,
            rect.y() + rect.height() - handle_size // 2,
            handle_size,
            handle_size
        )
        if bottom_right_handle.contains(pos):
            return "bottom-right"
        
        return None

    def get_annotation_at_position(self, pos):
        """Find annotation at the given position."""
        if not self.pixmap or not hasattr(self, 'display_rect'):
            return None
        
        # Transform the position to image coordinates
        image_pos = self.transform_point_to_image(pos)
        if image_pos is None:
            return None
        
        # Check annotations in reverse order (top-most first)
        for annotation in reversed(self.annotations):
            if annotation.rect.contains(image_pos):
                return annotation
        
        return None

    def mousePressEvent(self, event):
        """Handle mouse press events."""
        if self.main_window.is_playing:
            self.main_window.toggle_play()
        if not self.pixmap or not hasattr(self, 'display_rect'):
            return
        
        # Check if the click is within the display rectangle
        if not self.display_rect.contains(event.pos()):
            return
        
        # Transform the event position to image coordinates
        image_pos = self.transform_point_to_image(event.pos())
        if image_pos is None:
            return
        
        # Check if we're resizing an existing annotation
        if self.selected_annotation:
            resize_handle = self.get_resize_handle(event.pos())
            if resize_handle:
                self.resize_handle = resize_handle
                self.resize_start = event.pos()
                return
        
        # Handle based on annotation method
        if self.annotation_method == "Drag":
            # Check if clicking on an existing annotation
            annotation = self.get_annotation_at_position(event.pos())
            if annotation:
                self.selected_annotation = annotation
                if hasattr(self.main_window, 'annotation_dock'):
                    self.main_window.annotation_dock.update_annotation_list()
                self.update()
                return
            
            # Start drawing a new annotation
            self.is_drawing = True
            self.start_point = image_pos
            self.end_point = image_pos
        
        elif self.annotation_method == "Two-Click":
            # Check if clicking on an existing annotation
            annotation = self.get_annotation_at_position(event.pos())
            if annotation:
                self.selected_annotation = annotation
                if hasattr(self.main_window, 'annotation_dock'):
                    self.main_window.annotation_dock.update_annotation_list()
                self.update()
                return
            
            # First click sets the first corner
            if not self.first_click_pos:
                self.first_click_pos = image_pos
            else:
                # Second click completes the annotation
                # Create a rectangle from the two points
                x1, y1 = self.first_click_pos.x(), self.first_click_pos.y()
                x2, y2 = image_pos.x(), image_pos.y()
                
                rect = QRect(
                    min(x1, x2),
                    min(y1, y2),
                    abs(x2 - x1),
                    abs(y2 - y1)
                )
                
                # Create a new annotation
                from annotation import BoundingBox
                annotation = BoundingBox(
                    rect,
                    self.current_class,
                    {},
                    self.class_colors.get(self.current_class, QColor(255, 0, 0))
                )
                
                # Add the annotation
                self.annotations.append(annotation)
                self.selected_annotation = annotation
                
                # Reset state
                self.first_click_pos = None
                
                # Update parent
                if hasattr(self.main_window, 'add_annotation'):
                    self.main_window.add_annotation(annotation)
                
                # Update UI
                if hasattr(self.main_window, 'update_annotation_list'):
                    self.main_window.update_annotation_list()
        self.update()   
   
    def mouseMoveEvent(self, event):
        """Handle mouse move events."""
        if not self.pixmap or not hasattr(self, 'display_rect'):
            return
        
        # Transform the event position to image coordinates
        image_pos = self.transform_point_to_image(event.pos())
        
        # Check if we're resizing an annotation
        if self.resize_handle and self.selected_annotation:
            # Calculate the difference from the resize start position in display coordinates
            current_rect = self.transform_rect_to_display(self.selected_annotation.rect)
            
            # Modify the rectangle based on which handle is being dragged
            if self.resize_handle == "top":
                current_rect.setTop(event.pos().y())
            elif self.resize_handle == "bottom":
                current_rect.setBottom(event.pos().y())
            elif self.resize_handle == "left":
                current_rect.setLeft(event.pos().x())
            elif self.resize_handle == "right":
                current_rect.setRight(event.pos().x())
            elif self.resize_handle == "top-left":
                current_rect.setTopLeft(event.pos())
            elif self.resize_handle == "top-right":
                current_rect.setTopRight(event.pos())
            elif self.resize_handle == "bottom-left":
                current_rect.setBottomLeft(event.pos())
            elif self.resize_handle == "bottom-right":
                current_rect.setBottomRight(event.pos())
            
            # Transform back to image coordinates
            new_rect = self.transform_rect_to_image(current_rect)
            
            # Ensure the rectangle has positive width and height
            if new_rect and new_rect.width() > 5 and new_rect.height() > 5:
                self.selected_annotation.rect = new_rect
            
            # Update the resize start position
            self.resize_start = event.pos()
            self.update()
            return
        
        # Handle based on annotation method
        if self.annotation_method == "Drag" and self.is_drawing and image_pos:
            # Update end point for drag annotation
            self.end_point = image_pos
            self.update()
        
        # Update cursor based on what's under it
        if self.selected_annotation:
            resize_handle = self.get_resize_handle(event.pos())
            if resize_handle:
                if resize_handle in ["top", "bottom"]:
                    self.setCursor(Qt.SizeVerCursor)
                elif resize_handle in ["left", "right"]:
                    self.setCursor(Qt.SizeHorCursor)
                elif resize_handle in ["top-left", "bottom-right"]:
                    self.setCursor(Qt.SizeFDiagCursor)
                elif resize_handle in ["top-right", "bottom-left"]:
                    self.setCursor(Qt.SizeBDiagCursor)
                return
        
        # Default cursor based on annotation method
        if self.annotation_method == "Drag":
            self.setCursor(Qt.ArrowCursor)
        elif self.annotation_method == "Two-Click":
            self.setCursor(Qt.CrossCursor)

    def mouseReleaseEvent(self, event):
        """Handle mouse release events."""
        if not self.pixmap or not hasattr(self, 'display_rect'):
            return
        
        # If we were resizing, stop resizing
        if self.resize_handle:
            self.resize_handle = None
            self.resize_start = None
            self.update()
            return
        
        # Transform the event position to image coordinates
        image_pos = self.transform_point_to_image(event.pos())
        if image_pos is None:
            return
        
        # Handle based on annotation method
        if self.annotation_method == "Drag" and self.is_drawing:
            # Update end point
            self.end_point = image_pos
            
            # Create a rectangle from the start and end points
            x1, y1 = self.start_point.x(), self.start_point.y()
            x2, y2 = self.end_point.x(), self.end_point.y()
            
            # Check if the rectangle is too small
            if abs(x2 - x1) < 5 or abs(y2 - y1) < 5:
                # Reset state
                self.is_drawing = False
                self.start_point = None
                self.end_point = None
                self.update()
                return
            
            rect = QRect(
                min(x1, x2),
                min(y1, y2),
                abs(x2 - x1),
                abs(y2 - y1)
            )
            
            # Create a new annotation
            from annotation import BoundingBox
            annotation = BoundingBox(
                rect,
                self.current_class,
                {},
                self.class_colors.get(self.current_class, QColor(255, 0, 0))
            )
            
            # Add the annotation
            self.annotations.append(annotation)
            self.selected_annotation = annotation
            
            # Reset state
            self.is_drawing = False
            self.start_point = None
            self.end_point = None
            
            # Update parent
            if hasattr(self.main_window, 'add_annotation'):
                self.main_window.add_annotation(annotation)
            
            # Update UI
            if hasattr(self.main_window, 'update_annotation_list'):
                self.main_window.update_annotation_list()
        
        self.update()

    def keyPressEvent(self, event):
        """Handle key press events."""
        # Cancel drawing with Escape
        if event.key() == Qt.Key_Escape:
            if self.annotation_method == "Two-Click" and self.first_click_pos:
                self.first_click_pos = None
                self.update()
            elif self.annotation_method == "Drag" and self.is_drawing:
                self.is_drawing = False
                self.start_point = None
                self.end_point = None
                self.update()
            elif self.selected_annotation:
                self.selected_annotation = None
                self.update()
        
        # Delete selected annotation with Delete key
        elif event.key() == Qt.Key_Delete and self.selected_annotation:
            if self.selected_annotation in self.annotations:
                self.annotations.remove(self.selected_annotation)
                self.selected_annotation = None
                self.parent.update_annotation_list()
                self.update()
   
    def screen_to_video_coords(self, screen_pos):
        """Convert screen coordinates to video coordinates"""
        if not self.pixmap:
            return None
        
        # Calculate the scaled size maintaining aspect ratio
        canvas_width = self.width()
        canvas_height = self.height()
        
        if canvas_width / canvas_height > self.aspect_ratio:
            # Canvas is wider than the video
            scaled_height = canvas_height
            scaled_width = int(scaled_height * self.aspect_ratio)
        else:
            # Canvas is taller than the video
            scaled_width = canvas_width
            scaled_height = int(scaled_width / self.aspect_ratio)
        
        # Calculate position to center the video
        x_offset = (canvas_width - scaled_width) // 2
        y_offset = (canvas_height - scaled_height) // 2
        
        # Check if click is within the video frame
        if (x_offset <= screen_pos.x() <= x_offset + scaled_width and
            y_offset <= screen_pos.y() <= y_offset + scaled_height):
            
            # Convert to video coordinates
            video_x = (screen_pos.x() - x_offset) * self.pixmap.width() / scaled_width
            video_y = (screen_pos.y() - y_offset) * self.pixmap.height() / scaled_height
            
            # Constrain to video boundaries
            video_x = max(0, min(video_x, self.pixmap.width() - 1))
            video_y = max(0, min(video_y, self.pixmap.height() - 1))
            
            return QPoint(int(video_x), int(video_y))
        
        return None
    
    def show_context_menu(self, position, annotation):
        """Show context menu for annotation"""
        menu = QMenu()
        
        # Add actions
        edit_action = menu.addAction("Edit")
        delete_action = menu.addAction("Delete")
        menu.addSeparator()
        change_class_menu = menu.addMenu("Change Class")
        
        # Add class options to submenu
        for class_name in self.class_colors.keys():
            class_action = change_class_menu.addAction(class_name)
            class_action.setData(class_name)
        
        # Show menu and get selected action
        action = menu.exec_(self.mapToGlobal(position))
        
        if action == edit_action:
            # Open edit dialog
            if hasattr(self.parent(), 'edit_annotation'):
                self.parent().edit_annotation(annotation)
        
        elif action == delete_action:
            # Use main window's delete_annotation method
            self.main_window.delete_annotation(annotation)
        
        elif action and action.parent() == change_class_menu:
            # Change class
            new_class = action.data()
            annotation.class_name = new_class
            annotation.color = self.class_colors.get(new_class, QColor(255, 0, 0))
            if hasattr(self.parent(), 'update_annotation_list'):
                self.parent().update_annotation_list()
            self.update()
    
    def finalize_annotation(self):
        """Finalize the current annotation being drawn"""
        if self.drawing and self.start_point and self.end_point:
            # Create rectangle from points
            rect = QRect(self.start_point, self.end_point).normalized()
            
            # Skip if too small
            if rect.width() < 5 or rect.height() < 5:
                self.drawing = False
                self.start_point = None
                self.end_point = None
                return
            
            # Get current class
            if not self.current_class:
                self.main_window.logger.warning("No class selected for annotation")
                self.drawing = False
                self.start_point = None
                self.end_point = None
                return
            
            # Get default attributes for the class
            attributes = {"size": -1, "quality": -1}
            
            # Check if we should inherit values from previous frame
            current_frame = self.main_window.current_frame
            if current_frame > 0:
                prev_frame = current_frame - 1
                if prev_frame in self.main_window.frame_annotations:
                    for prev_annotation in self.main_window.frame_annotations[prev_frame]:
                        if prev_annotation.class_name == self.current_class:
                            # Inherit non-default attribute values
                            for attr_name, attr_value in prev_annotation.attributes.items():
                                if attr_value != -1:
                                    attributes[attr_name] = attr_value
            
            # Create annotation
            annotation = BoundingBox(rect, self.current_class, attributes)
            
            # Add to current frame
            if current_frame not in self.main_window.frame_annotations:
                self.main_window.frame_annotations[current_frame] = []
            
            self.main_window.frame_annotations[current_frame].append(annotation)
            
            # Update UI
            self.main_window.annotation_dock.update_annotation_list()
            self.main_window.logger.info(f"Added annotation: {self.current_class}")
            
            # Reset drawing state
            self.drawing = False
            self.start_point = None
            self.end_point = None
            self.update()

    def select_annotation(self, annotation):
        """Select an annotation and highlight it on the canvas"""
        # Deselect all annotations first
        for ann in self.annotations:
            ann.selected = False
        
        # Select the specified annotation
        if annotation in self.annotations:
            annotation.selected = True
            self.selected_annotation = annotation
        else:
            # If the annotation is in frame_annotations but not in canvas.annotations
            current_frame = self.main_window.current_frame
            if current_frame in self.main_window.frame_annotations:
                if annotation in self.main_window.frame_annotations[current_frame]:
                    # Add it to canvas annotations if it's not already there
                    if annotation not in self.annotations:
                        self.annotations.append(annotation)
                    annotation.selected = True
                    self.selected_annotation = annotation
        
        # Also select the annotation in the annotation dock's list
        if hasattr(self.main_window, 'annotation_dock'):
            annotation_dock = self.main_window.annotation_dock
            for i in range(annotation_dock.annotations_list.count()):
                item = annotation_dock.annotations_list.item(i)
                if item.data(Qt.UserRole) == annotation:
                    annotation_dock.annotations_list.setCurrentItem(item)
                    break
        
        # Update the canvas to show the selection
        self.update()
    
    def zoom_in(self):
        """Increase the zoom level."""
        self.scale_factor = min(3.0, self.scale_factor * 1.2)  # Limit max zoom to 3x
        self.update()

    def zoom_out(self):
        """Decrease the zoom level."""
        self.scale_factor = max(0.2, self.scale_factor / 1.2)  # Limit min zoom to 0.2x
        self.update()

    def reset_zoom(self):
        """Reset zoom to original size."""
        self.scale_factor = 1.0
        self.update()

    def wheelEvent(self, event):
        """Handle mouse wheel events for zooming."""
        if not self.pixmap:
            return
        
        # Get the delta value
        delta = event.angleDelta().y()
        
        # Zoom in or out based on wheel direction
        if delta > 0:
            self.zoom_in()
        else:
            self.zoom_out()

    def transform_point_to_display(self, point):
        """Transform a point from image coordinates to display coordinates."""
        if not hasattr(self, 'display_rect') or not self.pixmap:
            return point
        
        # Calculate the scaling factors
        x_scale = self.display_rect.width() / self.pixmap.width()
        y_scale = self.display_rect.height() / self.pixmap.height()
        
        # Transform the point
        display_x = self.display_rect.x() + int(point.x() * x_scale)
        display_y = self.display_rect.y() + int(point.y() * y_scale)
        
        return QPoint(display_x, display_y)

    def transform_point_to_image(self, point):
        """Transform a point from display coordinates to image coordinates."""
        if not hasattr(self, 'display_rect') or not self.pixmap:
            return point
        
        # Check if point is within display rectangle
        if not self.display_rect.contains(point):
            return None
        
        # Calculate the scaling factors
        x_scale = self.pixmap.width() / self.display_rect.width()
        y_scale = self.pixmap.height() / self.display_rect.height()
        
        # Transform the point
        image_x = int((point.x() - self.display_rect.x()) * x_scale)
        image_y = int((point.y() - self.display_rect.y()) * y_scale)
        
        # Ensure the point is within image bounds
        image_x = max(0, min(image_x, self.pixmap.width() - 1))
        image_y = max(0, min(image_y, self.pixmap.height() - 1))
        
        return QPoint(image_x, image_y)

    def transform_rect_to_display(self, rect):
        """Transform a rectangle from image coordinates to display coordinates."""
        if not hasattr(self, 'display_rect') or not self.pixmap:
            return rect
        
        # Transform top-left and bottom-right points
        top_left = self.transform_point_to_display(rect.topLeft())
        bottom_right = self.transform_point_to_display(rect.bottomRight())
        
        # Create a new rectangle
        return QRect(top_left, bottom_right)

    def transform_rect_to_image(self, rect):
        """Transform a rectangle from display coordinates to image coordinates."""
        if not hasattr(self, 'display_rect') or not self.pixmap:
            return rect
        
        # Transform top-left and bottom-right points
        top_left = self.transform_point_to_image(rect.topLeft())
        bottom_right = self.transform_point_to_image(rect.bottomRight())
        
        if top_left is None or bottom_right is None:
            return None
        
        # Create a new rectangle
        return QRect(top_left, bottom_right)
    def fit_to_view(self):
        """Scale the image to fit the widget size."""
        if not self.pixmap:
            return
        
        # Calculate the scale factor to fit the widget
        widget_width = self.width()
        widget_height = self.height()
        
        pixmap_width = self.pixmap.width()
        pixmap_height = self.pixmap.height()
        
        # Calculate scale factors for width and height
        width_scale = widget_width / pixmap_width
        height_scale = widget_height / pixmap_height
        
        # Use the smaller scale factor to ensure the entire image fits
        self.scale_factor = min(width_scale, height_scale) * 0.95  # 95% to add a small margin
        
        self.update()