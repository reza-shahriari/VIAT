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
- Edge movement for precise bounding box adjustments
- Right-click context menu for editing and deleting annotations
"""

import cv2
from PyQt5.QtWidgets import QWidget, QMenu
from PyQt5.QtCore import Qt, QRect, QPoint
from PyQt5.QtGui import QPainter, QPen, QColor, QBrush, QPixmap, QImage
from annotation import BoundingBox
import random

# Edge detection constants
EDGE_NONE = 0
EDGE_TOP = 1
EDGE_RIGHT = 2
EDGE_BOTTOM = 3
EDGE_LEFT = 4

class VideoCanvas(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent  
        self.selected_bbox = None  
        self.resize_mode = None
        self.drag_start_pos = None
        self.original_rect = None
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
            "Drone": QColor(0, 255, 255),  
        }
        self.annotation_method = "Drag"  # Default method
        self.is_drawing = False
        self.start_point = None
        self.end_point = None
        self.resize_handle = None
        self.resize_start = None
        
        # Edge movement properties
        self.edge_moving = False
        self.active_edge = EDGE_NONE
        self.edge_start_pos = None
        
        # Set focus policy to receive keyboard events
        self.setFocusPolicy(Qt.StrongFocus)
        
        # Enable context menu
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
    
    def set_frame(self, frame):
        """Set the current frame to display"""
        if frame is None:
            return
            
        # Convert OpenCV BGR format to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        
        # Update aspect ratio based on the frame
        self.aspect_ratio = w / h
        
        # Create QImage from numpy array
        bytes_per_line = ch * w
        q_img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Convert to QPixmap for display
        self.pixmap = QPixmap.fromImage(q_img)
        self.update()
    
    def set_current_class(self, class_name):
        """Set the current annotation class"""
        if class_name in self.class_colors:
            self.current_class = class_name
    
    def paintEvent(self, event):
        """Paint the canvas with the current frame and annotations"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Fill background with dark color
        painter.fillRect(self.rect(), QColor(40, 40, 40))
        
        if self.pixmap:
            # Calculate display rectangle maintaining aspect ratio
            display_rect = self.get_display_rect()
            
            # Draw the image
            painter.drawPixmap(display_rect, self.pixmap)
            
            # Draw annotations
            for annotation in self.annotations:
                # Convert annotation rectangle to display coordinates
                display_rect = self.image_to_display_rect(annotation.rect)
                
                # Set pen based on selection status
                if annotation == self.selected_annotation:
                    pen = QPen(QColor(255, 255, 0), 2)  # Yellow for selected
                else:
                    pen = QPen(annotation.color, 2)
                
                painter.setPen(pen)
                painter.drawRect(display_rect)
                
                # Draw class label
                text_rect = QRect(display_rect.left(), display_rect.top() - 20, 
                                 display_rect.width(), 20)
                painter.fillRect(text_rect, annotation.color)
                painter.setPen(QPen(QColor(0, 0, 0)))
                painter.drawText(text_rect, Qt.AlignCenter, annotation.class_name)
            
            # Draw current annotation if drawing
            if self.is_drawing and self.start_point and self.current_point:
                rect = QRect(self.start_point, self.current_point).normalized()
                display_rect = self.image_to_display_rect(rect)
                painter.setPen(QPen(self.class_colors.get(self.current_class, QColor(255, 0, 0)), 2, Qt.DashLine))
                painter.drawRect(display_rect)
            
            # Draw edge indicators for selected annotation
            if self.selected_annotation and not self.is_drawing:
                display_rect = self.image_to_display_rect(self.selected_annotation.rect)
                
                # Draw handles on the edges
                handle_size = 8
                painter.setPen(QPen(QColor(255, 255, 0), 1))
                painter.setBrush(QBrush(QColor(255, 255, 0)))
                
                # Top edge
                painter.drawRect(QRect(display_rect.center().x() - handle_size//2, 
                                      display_rect.top() - handle_size//2,
                                      handle_size, handle_size))
                
                # Right edge
                painter.drawRect(QRect(display_rect.right() - handle_size//2, 
                                      display_rect.center().y() - handle_size//2,
                                      handle_size, handle_size))
                
                # Bottom edge
                painter.drawRect(QRect(display_rect.center().x() - handle_size//2, 
                                      display_rect.bottom() - handle_size//2,
                                      handle_size, handle_size))
                
                # Left edge
                painter.drawRect(QRect(display_rect.left() - handle_size//2, 
                                      display_rect.center().y() - handle_size//2,
                                      handle_size, handle_size))
    
    def get_display_rect(self):
        """Calculate the display rectangle maintaining aspect ratio"""
        if not self.pixmap:
            return QRect()
            
        # Get widget dimensions
        w = self.width()
        h = self.height()
        
        # Calculate display dimensions maintaining aspect ratio
        if w / h > self.aspect_ratio:
            # Widget is wider than the image aspect ratio
            display_height = h
            display_width = int(h * self.aspect_ratio)
            x = (w - display_width) // 2
            y = 0
        else:
            # Widget is taller than the image aspect ratio
            display_width = w
            display_height = int(w / self.aspect_ratio)
            x = 0
            y = (h - display_height) // 2
            
        return QRect(x, y, display_width, display_height)
    
    def display_to_image_pos(self, pos):
        """Convert display coordinates to image coordinates"""
        if not self.pixmap:
            return pos
            
        display_rect = self.get_display_rect()
        
        # Check if point is within display area
        if not display_rect.contains(pos):
            return None
            
        # Calculate relative position within display rect
        rel_x = (pos.x() - display_rect.left()) / display_rect.width()
        rel_y = (pos.y() - display_rect.top()) / display_rect.height()
        
        # Convert to image coordinates
        img_x = int(rel_x * self.pixmap.width())
        img_y = int(rel_y * self.pixmap.height())
        
        return QPoint(img_x, img_y)
    
    def image_to_display_pos(self, pos):
        """Convert image coordinates to display coordinates"""
        if not self.pixmap or not pos:
            return pos
            
        display_rect = self.get_display_rect()
        
        # Calculate relative position within image
        rel_x = pos.x() / self.pixmap.width()
        rel_y = pos.y() / self.pixmap.height()
        
        # Convert to display coordinates
        disp_x = int(display_rect.left() + rel_x * display_rect.width())
        disp_y = int(display_rect.top() + rel_y * display_rect.height())
        
        return QPoint(disp_x, disp_y)
    
    def image_to_display_rect(self, rect):
        """Convert image rectangle to display rectangle"""
        if not self.pixmap:
            return rect
            
        # Convert top-left and bottom-right points
        top_left = self.image_to_display_pos(rect.topLeft())
        bottom_right = self.image_to_display_pos(rect.bottomRight())
        
        if top_left and bottom_right:
            return QRect(top_left, bottom_right)
        return QRect()
    
    def display_to_image_rect(self, rect):
        """Convert display rectangle to image rectangle"""
        if not self.pixmap:
            return rect
            
        # Convert top-left and bottom-right points
        top_left = self.display_to_image_pos(rect.topLeft())
        bottom_right = self.display_to_image_pos(rect.bottomRight())
        
        if top_left and bottom_right:
            return QRect(top_left, bottom_right)
        return QRect()
    
    def detect_edge(self, rect, pos, threshold=8):
        """
        Detect if the cursor is near an edge of the rectangle.
        
        Args:
            rect: QRect object in display coordinates
            pos: QPoint cursor position
            threshold: Distance threshold to consider cursor near an edge
            
        Returns:
            Edge constant (EDGE_NONE, EDGE_TOP, EDGE_RIGHT, EDGE_BOTTOM, EDGE_LEFT)
        """
        if not rect.adjusted(-threshold, -threshold, threshold, threshold).contains(pos):
            return EDGE_NONE
            
        # Check if cursor is near any edge
        if abs(pos.y() - rect.top()) <= threshold:
            return EDGE_TOP
        elif abs(pos.x() - rect.right()) <= threshold:
            return EDGE_RIGHT
        elif abs(pos.y() - rect.bottom()) <= threshold:
            return EDGE_BOTTOM
        elif abs(pos.x() - rect.left()) <= threshold:
            return EDGE_LEFT
        
        return EDGE_NONE
    
    def get_edge_cursor(self, edge):
        """Return the appropriate cursor for the given edge"""
        if edge in (EDGE_TOP, EDGE_BOTTOM):
            return Qt.SizeVerCursor
        elif edge in (EDGE_LEFT, EDGE_RIGHT):
            return Qt.SizeHorCursor
        return Qt.ArrowCursor
    
    def move_edge(self, rect, edge, pos, start_pos):
        """
        Move a specific edge of the rectangle based on cursor movement.
        
        Args:
            rect: QRect to modify
            edge: Which edge to move (EDGE_TOP, EDGE_RIGHT, etc.)
            pos: Current cursor position
            start_pos: Starting cursor position
            
        Returns:
            Modified QRect
        """
        delta_x = pos.x() - start_pos.x()
        delta_y = pos.y() - start_pos.y()
        
        new_rect = QRect(rect)
        
        if edge == EDGE_TOP:
            new_top = rect.top() + delta_y
            if new_top < rect.bottom():
                new_rect.setTop(new_top)
        elif edge == EDGE_RIGHT:
            new_right = rect.right() + delta_x
            if new_right > rect.left():
                new_rect.setRight(new_right)
        elif edge == EDGE_BOTTOM:
            new_bottom = rect.bottom() + delta_y
            if new_bottom > rect.top():
                new_rect.setBottom(new_bottom)
        elif edge == EDGE_LEFT:
            new_left = rect.left() + delta_x
            if new_left < rect.right():
                new_rect.setLeft(new_left)
        
        return new_rect
    
    def find_annotation_at_pos(self, pos):
        """Find annotation at the given position"""
        if not self.pixmap:
            return None
            
        # Convert to image coordinates
        img_pos = self.display_to_image_pos(pos)
        if not img_pos:
            return None
            
        # Check each annotation in reverse order (top-most first)
        for annotation in reversed(self.annotations):
            if annotation.rect.contains(img_pos):
                return annotation
                
        return None
    
    def mousePressEvent(self, event):
        """Handle mouse press events"""
        if not self.pixmap:
            return
            
        if event.button() == Qt.LeftButton:
            # Convert to image coordinates
            img_pos = self.display_to_image_pos(event.pos())
            if not img_pos:
                return
                
            # Check if we're on an edge of the selected annotation
            if self.selected_annotation:
                display_rect = self.image_to_display_rect(self.selected_annotation.rect)
                edge = self.detect_edge(display_rect, event.pos())
                
                if edge != EDGE_NONE:
                    self.edge_moving = True
                    self.active_edge = edge
                    self.edge_start_pos = event.pos()
                    self.original_rect = QRect(self.selected_annotation.rect)
                    return
            
            # Check if clicking on an existing annotation
            annotation = self.find_annotation_at_pos(event.pos())
            if annotation:
                self.selected_annotation = annotation
                self.drag_start_pos = img_pos
                self.original_rect = QRect(annotation.rect)
                self.update()
                return
                
            # Start drawing a new annotation
            self.is_drawing = True
            self.start_point = img_pos
            self.current_point = img_pos
            self.selected_annotation = None
            self.update()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move events"""
        if not self.pixmap:
            return
            
        # If we're moving an edge
        if self.edge_moving and self.selected_annotation:
            # Get the current display rect
                        # Get the current display rect
            display_rect = self.image_to_display_rect(self.original_rect)
            
            # Move the edge in display coordinates
            new_display_rect = self.move_edge(
                display_rect, 
                self.active_edge, 
                event.pos(), 
                self.edge_start_pos
            )
            
            # Convert back to image coordinates
            new_img_rect = self.display_to_image_rect(new_display_rect)
            
            # Update the annotation with the new rectangle
            if new_img_rect:
                self.selected_annotation.rect = new_img_rect
                self.update()
            return
        
        # If we're dragging an annotation
        if self.drag_start_pos and self.selected_annotation and not self.is_drawing:
            img_pos = self.display_to_image_pos(event.pos())
            if not img_pos:
                return
                
            # Calculate the delta movement in image coordinates
            delta_x = img_pos.x() - self.drag_start_pos.x()
            delta_y = img_pos.y() - self.drag_start_pos.y()
            
            # Create a new rectangle with the offset
            new_rect = QRect(
                self.original_rect.x() + delta_x,
                self.original_rect.y() + delta_y,
                self.original_rect.width(),
                self.original_rect.height()
            )
            
            # Update the annotation
            self.selected_annotation.rect = new_rect
            self.update()
            return
        
        # If we're drawing a new annotation
        if self.is_drawing:
            img_pos = self.display_to_image_pos(event.pos())
            if img_pos:
                self.current_point = img_pos
                self.update()
            return
        
        # Update cursor based on what's under it
        if self.selected_annotation:
            display_rect = self.image_to_display_rect(self.selected_annotation.rect)
            edge = self.detect_edge(display_rect, event.pos())
            
            if edge != EDGE_NONE:
                self.setCursor(self.get_edge_cursor(edge))
            else:
                # Check if cursor is over the selected annotation
                annotation = self.find_annotation_at_pos(event.pos())
                if annotation == self.selected_annotation:
                    self.setCursor(Qt.SizeAllCursor)  # Move cursor
                else:
                    self.setCursor(Qt.ArrowCursor)
        else:
            # Check if cursor is over any annotation
            annotation = self.find_annotation_at_pos(event.pos())
            if annotation:
                self.setCursor(Qt.PointingHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release events"""
        if not self.pixmap:
            return
            
        if event.button() == Qt.LeftButton:
            # If we were moving an edge
            if self.edge_moving:
                self.edge_moving = False
                self.active_edge = EDGE_NONE
                self.edge_start_pos = None
                
                # Update the annotation list in the main window
                if self.main_window:
                    self.main_window.update_annotation_list()
                return
            
            # If we were dragging an annotation
            if self.drag_start_pos and self.selected_annotation and not self.is_drawing:
                self.drag_start_pos = None
                self.original_rect = None
                
                # Update the annotation list in the main window
                if self.main_window:
                    self.main_window.update_annotation_list()
                return
            
            # If we were drawing a new annotation
            if self.is_drawing and self.start_point and self.current_point:
                # Create a rectangle from the start and current points
                rect = QRect(self.start_point, self.current_point).normalized()
                
                # Only create annotation if it has a minimum size
                if rect.width() > 5 and rect.height() > 5:
                    # Create a new bounding box annotation
                    color = self.class_colors.get(self.current_class, QColor(255, 0, 0))
                    bbox = BoundingBox(rect, self.current_class, {}, color)
                    
                    # Add to annotations list
                    self.annotations.append(bbox)
                    self.selected_annotation = bbox
                    
                    # Update the annotation list in the main window
                    if self.main_window:
                        self.main_window.update_annotation_list()
                
                # Reset drawing state
                self.is_drawing = False
                self.start_point = None
                self.current_point = None
                self.update()
    
    def show_context_menu(self, position):
        """Show context menu for right-click actions"""
        if not self.pixmap:
            return
            
        # Find annotation at the clicked position
        annotation = self.find_annotation_at_pos(position)
        
        # If we have an annotation, show context menu
        if annotation:
            self.selected_annotation = annotation
            self.update()
            
            # Create context menu
            context_menu = QMenu(self)
            
            # Add actions
            edit_action = context_menu.addAction("Edit Annotation")
            delete_action = context_menu.addAction("Delete Annotation")
            change_class_menu = context_menu.addMenu("Change Class")
            
            # Add class options to submenu
            for class_name in self.class_colors.keys():
                action = change_class_menu.addAction(class_name)
                action.setData(class_name)
            
            # Show the menu and get the selected action
            action = context_menu.exec_(self.mapToGlobal(position))
            
            # Handle the selected action
            if action:
                if action == edit_action:
                    # Call the edit annotation method in the main window
                    if self.main_window:
                        self.main_window.edit_annotation(self.selected_annotation)
                elif action == delete_action:
                    # Call the delete annotation method in the main window
                    if self.main_window:
                        self.main_window.delete_selected_annotation()
                elif action.parent() == change_class_menu:
                    # Change the class of the annotation
                    new_class = action.data()
                    if new_class in self.class_colors:
                        self.selected_annotation.class_name = new_class
                        self.selected_annotation.color = self.class_colors[new_class]
                        self.update()
                        
                        # Update the annotation list in the main window
                        if self.main_window:
                            self.main_window.update_annotation_list()
    
    def keyPressEvent(self, event):
        """Handle keyboard events"""
        # Delete key to remove selected annotation
        if event.key() == Qt.Key_Delete and self.selected_annotation:
            if self.main_window:
                self.main_window.delete_selected_annotation()
        # Escape key to cancel drawing or selection
        elif event.key() == Qt.Key_Escape:
            if self.is_drawing:
                self.is_drawing = False
                self.start_point = None
                self.current_point = None
            else:
                self.selected_annotation = None
            self.update()
        else:
            super().keyPressEvent(event)

