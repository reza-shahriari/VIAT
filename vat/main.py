import os
import sys
import cv2
import numpy as np
import random
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QSlider, QLabel, QAction, QFileDialog, QDockWidget,
    QToolBar, QStatusBar, QComboBox, QMessageBox, QListWidget, 
    QListWidgetItem, QMenu, QDialog, QFormLayout, QSpinBox, QTextEdit,
    QDialogButtonBox, QLineEdit, QColorDialog, QActionGroup
)
from PyQt5.QtCore import Qt, QTimer, QRect, QPoint,QEvent
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QIcon, QPalette

from canvas import VideoCanvas
from annotation import BoundingBox
from widgets.styles import StyleManager
from widgets import AnnotationDock, ClassDock, AnnotationToolbar
from utils import save_project, load_project, export_annotations
from config import DEFAULT_SETTINGS, STYLE_CONFIGS, EXPORT_FORMATS
from utils.logger import Logger
import datetime
class VideoAnnotationTool(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Annotation Tool")
        self.logger = Logger()
        self.logger.info("Starting Video Annotation Tool")
        self.setGeometry(*DEFAULT_SETTINGS["window_geometry"])
        self.current_style = DEFAULT_SETTINGS["default_style"]
        self.playback_speed = DEFAULT_SETTINGS["default_playback_speed"]
          
        self.annotation_methods = {
            "Drag": "drag_annotation",
            "Two-Click": "two_click_annotation"
        }
        self.current_annotation_method = "Drag" 
        self.styles = {}
        self.class_attributes = {}
        for style_name, style_config in STYLE_CONFIGS.items():
            style_function = getattr(StyleManager, style_config["function"])
            self.styles[style_name] = style_function
        
        
        self.cap = None  
        self.annotations = []
        self.frame_annotations = {}  
        self.current_frame = 0
        self.init_icon()
        self.init_ui()

        self.auto_save_timer = QTimer()
        self.auto_save_timer.timeout.connect(self.auto_save)
        self.auto_save_timer.start(DEFAULT_SETTINGS["auto_save_interval"] * 1000)
    
    def init_icon(self):
        self.setWindowIcon(QIcon("vat/Icon/Icon.png"))

    def init_canvas(self):
        """Initialize the canvas with the current annotation method."""
        if hasattr(self, 'canvas'):
            self.canvas.set_annotation_method(self.current_annotation_method) 
    

    def init_ui(self):
        # Create central widget with layout
        QApplication.instance().installEventFilter(self)

        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        
        # Create video canvas
        self.canvas = VideoCanvas(self)
        main_layout.addWidget(self.canvas)
        
        # Create playback controls panel
        playback_panel = QWidget()
        playback_layout = QHBoxLayout(playback_panel)
        
        # Previous frame button
        self.prev_frame_button = QPushButton("Previous Frame")
        self.prev_frame_button.clicked.connect(self.previous_frame)
        playback_layout.addWidget(self.prev_frame_button)
        
        # Play/Pause button
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        playback_layout.addWidget(self.play_button)
        
        # Next frame button
        self.next_frame_button = QPushButton("Next Frame")
        self.next_frame_button.clicked.connect(self.next_frame)
        playback_layout.addWidget(self.next_frame_button)
        
        # Frame slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)  # Will be updated when video is loaded
        self.frame_slider.valueChanged.connect(self.go_to_frame)
        playback_layout.addWidget(self.frame_slider)
        
        # Add playback panel to main layout
        main_layout.addWidget(playback_panel)
        
        # Set central widget
        self.setCentralWidget(central_widget)
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create toolbar
        self.toolbar = AnnotationToolbar(self)
        self.addToolBar(self.toolbar)
        self.class_selector = self.toolbar.class_selector
        
        # Create dock widgets
        self.annotation_dock = AnnotationDock(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.annotation_dock)
        
        self.class_dock = ClassDock(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.class_dock)
        
        # Create status bar
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
        
        # Set up timer for video playback
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.next_frame)
        self.is_playing = False
        
        # Initialize annotation list
        self.update_annotation_list()
        
        # Start performance monitoring
        QTimer.singleShot(5000, self.monitor_performance)  # Start after 5 seconds
        self.init_canvas()

    def eventFilter(self, obj, event):
        """Global event filter to catch events regardless of focus."""
        from PyQt5.QtCore import QEvent
        
        if event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Right:
                self.next_frame()  # Left arrow goes to next frame as requested
                return True
            elif event.key() == Qt.Key_Left:
                self.previous_frame()  # Right arrow goes to previous frame as requested
                return True
            elif event.key() == Qt.Key_Space:
                self.toggle_play()
                return True
        
        # Pass the event to the default event filter
        return super().eventFilter(obj, event)
   
    def auto_save(self):
        """Auto-save the current project."""
        if not self.canvas.annotations:
            return
            
        try:
            # Create auto-save directory if it doesn't exist
            os.makedirs('autosave', exist_ok=True)
            
            # Save to timestamped file
            filename = f'autosave/vat_autosave_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
            save_project(filename, self.canvas.annotations, self.canvas.class_colors)
            self.logger.info(f"Auto-saved project to {filename}")
        except Exception as e:
            self.logger.error(f"Auto-save failed: {str(e)}")

    def create_menu_bar(self):
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        open_action = QAction("Open Video", self)
        open_action.triggered.connect(self.open_video)
        file_menu.addAction(open_action)
        
        save_action = QAction("Save Project", self)
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)
        
        load_action = QAction("Load Project", self)
        load_action.triggered.connect(self.load_project)
        file_menu.addAction(load_action)
        
        export_action = QAction("Export Annotations", self)
        export_action.triggered.connect(self.export_annotations)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        
        clear_action = QAction("Clear Annotations", self)
        clear_action.triggered.connect(self.clear_annotations)
        edit_menu.addAction(clear_action)
        
        # Tools menu
        tools_menu = menubar.addMenu("Tools")
        
        auto_label_action = QAction("Auto Label", self)
        auto_label_action.triggered.connect(self.auto_label)
        tools_menu.addAction(auto_label_action)
        
        track_action = QAction("Track Objects", self)
        track_action.triggered.connect(self.track_objects)
        tools_menu.addAction(track_action)
        
        # Annotation Methods menu
        annotation_menu = menubar.addMenu("Annotation Methods")
        
        # Create action group for annotation methods to make them exclusive
        annotation_group = QActionGroup(self)
        annotation_group.setExclusive(True)
        
        # Add annotation method options
        for method_name in self.annotation_methods.keys():
            method_action = QAction(method_name, self, checkable=True)
            if method_name == self.current_annotation_method:
                method_action.setChecked(True)
            method_action.triggered.connect(lambda checked, m=method_name: self.change_annotation_method(m))
            annotation_group.addAction(method_action)
            annotation_menu.addAction(method_action)
        
        # Style menu
        style_menu = menubar.addMenu("Style")
        
        # Create action group for styles to make them exclusive
        style_group = QActionGroup(self)
        style_group.setExclusive(True)
        
        # Add style options
        for style_name in self.styles.keys():
            style_action = QAction(style_name, self, checkable=True)
            if style_name == self.current_style:
                style_action.setChecked(True)
            style_action.triggered.connect(lambda checked, s=style_name: self.change_style(s))
            style_group.addAction(style_action)
            style_menu.addAction(style_action)
        
        # Help menu
        help_menu = menubar.addMenu("Help")
        
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

        view_menu = menubar.addMenu("View")

        zoom_in_action = QAction("Zoom In", self)
        zoom_in_action.setShortcut("Ctrl++")
        zoom_in_action.triggered.connect(self.zoom_in)
        view_menu.addAction(zoom_in_action)

        zoom_out_action = QAction("Zoom Out", self)
        zoom_out_action.setShortcut("Ctrl+-")
        zoom_out_action.triggered.connect(self.zoom_out)
        view_menu.addAction(zoom_out_action)

        reset_zoom_action = QAction("Reset Zoom", self)
        reset_zoom_action.setShortcut("Ctrl+0")
        reset_zoom_action.triggered.connect(self.reset_zoom)
        view_menu.addAction(reset_zoom_action)
        fit_to_view_action = QAction("Fit to View", self)
        fit_to_view_action.setShortcut("Ctrl+F")
        fit_to_view_action.triggered.connect(self.fit_to_view)
        view_menu.addAction(fit_to_view_action)

    def change_annotation_method(self, method_name):
        """Change the annotation method."""
        if method_name not in self.annotation_methods:
            self.logger.warning(f"Annotation method '{method_name}' not found, using default")
            method_name = "Drag"
        
        try:
            self.current_annotation_method = method_name
            self.canvas.set_annotation_method(method_name)
            self.statusBar.showMessage(f"Annotation method changed to {method_name}")
            self.logger.info(f"Annotation method changed to {method_name}")
        except Exception as e:
            error_msg = f"Error changing annotation method: {str(e)}"
            self.statusBar.showMessage(error_msg)
            self.logger.error(error_msg)
            # Fallback to default method
            self.current_annotation_method = "Drag"
            self.canvas.set_annotation_method("Drag")

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for navigation and playback control."""
        if event.key() == Qt.Key_Left:
            # Left arrow key - go to previous frame
            self.previous_frame()
            event.accept()  # Mark event as handled
        elif event.key() == Qt.Key_Right:
            # Right arrow key - go to next frame
            self.next_frame()
            event.accept()  # Mark event as handled
        elif event.key() == Qt.Key_Space:
            # Space key - toggle play/pause
            self.toggle_play()
            event.accept()  # Mark event as handled
        else:
            
            super().keyPressEvent(event)

    def update_annotation_list(self):
        """Update the annotations list widget"""
        self.annotation_dock.update_annotation_list()
    
    def open_video(self):
        """Open a video file with improved error handling."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if not filename:
            return
            
        self.logger.info(f"Opening video: {filename}")
            
        # Close any existing video
        if self.cap:
            self.cap.release()
        
        try:
            # Open the video file
            self.cap = cv2.VideoCapture(filename)
            
            if not self.cap.isOpened():
                error_msg = f"Could not open video file: {filename}"
                self.logger.error(error_msg)
                QMessageBox.critical(self, "Error", error_msg)
                self.cap = None
                return
            
            # Get video properties
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Read the first frame
            ret, frame = self.cap.read()
            if ret:
                self.canvas.set_frame(frame)
                
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
                if hasattr(self, 'frame_slider'):
                    self.frame_slider.setMinimum(0)
                    self.frame_slider.setMaximum(total_frames - 1)
                    self.frame_slider.setValue(0)
                self.statusBar.showMessage(f"Loaded video: {os.path.basename(filename)} ({width}x{height}, {fps:.2f} fps, {frame_count} frames)")
                self.logger.info(f"Successfully loaded video: {width}x{height}, {fps:.2f} fps, {frame_count} frames")
            else:
                error_msg = "Could not read video frame!"
                self.logger.error(error_msg)
                QMessageBox.critical(self, "Error", error_msg)
                self.cap.release()
                self.cap = None
        except Exception as e:
            error_msg = f"Error opening video: {str(e)}"
            self.logger.error(error_msg)
            QMessageBox.critical(self, "Error", error_msg)
            if self.cap:
                self.cap.release()
            self.cap = None
    
    def next_frame(self):
        """Go to the next frame in the video with optimized image handling."""
        if self.cap and self.cap.isOpened():
            try:
                # Save current frame annotations
                if hasattr(self.canvas, 'annotations'):
                    self.frame_annotations[self.current_frame] = self.canvas.annotations.copy()
                
                ret, frame = self.cap.read()
                if ret:
                    # Update current frame
                    self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                    
                    # Optimize image for better performance
                    from utils.performance import PerformanceMonitor
                    optimized_frame = PerformanceMonitor.optimize_image(frame)
                    
                    # Load annotations for this frame
                    if self.current_frame in self.frame_annotations:
                        self.canvas.annotations = self.frame_annotations[self.current_frame].copy()
                    else:
                        self.canvas.annotations = []
                    
                    # Update canvas with new frame
                    self.canvas.set_frame(optimized_frame)
                    
                    # Update slider position
                    if hasattr(self, 'frame_slider'):
                        self.frame_slider.setValue(self.current_frame)
                    
                    # Update annotation list
                    self.update_annotation_list()
                    
                    # Update status bar
                    total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.statusBar.showMessage(f"Frame: {self.current_frame+1}/{total_frames}")
                else:
                    # End of video
                    self.play_timer.stop()
                    self.is_playing = False
                    self.statusBar.showMessage("End of video")
                    self.logger.info("Reached end of video")
                    # Rewind to beginning
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            except Exception as e:
                error_msg = f"Error reading frame: {str(e)}"
                self.statusBar.showMessage(error_msg)
                self.logger.error(error_msg)
                self.play_timer.stop()
                self.is_playing = False

    def go_to_frame(self, frame_number):
        """Go to a specific frame in the video."""
        if not self.cap:
            return
        
        # Save current frame annotations
        if hasattr(self.canvas, 'annotations'):
            self.frame_annotations[self.current_frame] = self.canvas.annotations.copy()
        
        # Get total frames
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Validate frame number
        if 0 <= frame_number < total_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            if ret:
                # Update current frame
                self.current_frame = frame_number
                
                # Load annotations for this frame
                if self.current_frame in self.frame_annotations:
                    self.canvas.annotations = self.frame_annotations[self.current_frame].copy()
                else:
                    self.canvas.annotations = []
                
                # Update canvas with new frame
                self.canvas.set_frame(frame)
                
                # Update annotation list
                self.update_annotation_list()
                
                self.statusBar.showMessage(f"Frame: {frame_number+1}/{total_frames}")
            else:
                self.statusBar.showMessage(f"Failed to read frame {frame_number}")
        else:
            self.statusBar.showMessage(f"Invalid frame number. Valid range: 0-{total_frames-1}")

    def previous_frame(self):
        """Go to the previous frame in the video."""
        if not self.cap:
            return
        
        # Save current frame annotations
        if hasattr(self.canvas, 'annotations'):
            self.frame_annotations[self.current_frame] = self.canvas.annotations.copy()
        
        # Get current position
        current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # Move to previous frame
        if current_pos > 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 2)  # -2 because next_frame will advance by 1
            ret, frame = self.cap.read()
            if ret:
                # Update current frame
                self.current_frame = current_pos - 2
                
                # Load annotations for this frame
                if self.current_frame in self.frame_annotations:
                    self.canvas.annotations = self.frame_annotations[self.current_frame].copy()
                else:
                    self.canvas.annotations = []
                
                # Update canvas with new frame
                self.canvas.set_frame(frame)
                
                # Update slider position
                if hasattr(self, 'frame_slider'):
                    self.frame_slider.setValue(self.current_frame)
                
                # Update annotation list
                self.update_annotation_list()
                
                # Update status bar
                total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.statusBar.showMessage(f"Frame: {self.current_frame+1}/{total_frames}")
            else:
                self.statusBar.showMessage("Failed to read previous frame")
        else:
            # Already at first frame
            self.statusBar.showMessage("Already at first frame")
 
    def save_project(self):
        """Save the current project with per-frame annotations."""
        # Check if we have any annotations
        has_annotations = False
        for frame_annotations in self.frame_annotations.values():
            if frame_annotations:
                has_annotations = True
                break
        
        if not has_annotations and not self.canvas.annotations:
            QMessageBox.warning(self, "Save Project", "No annotations to save!")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if filename:
            try:
                # Save current frame annotations before saving project
                if hasattr(self.canvas, 'annotations'):
                    self.frame_annotations[self.current_frame] = self.canvas.annotations.copy()
                
                # Create project data
                project_data = {
                    "frame_annotations": {},
                    "class_colors": {},
                    "video_path": self.cap.get(cv2.CAP_PROP_POS_AVI_RATIO) if self.cap else None,
                    "current_frame": self.current_frame
                }
                
                # Convert annotations to serializable format
                for frame_num, annotations in self.frame_annotations.items():
                    project_data["frame_annotations"][str(frame_num)] = []
                    for annotation in annotations:
                        # Convert QRect to dictionary
                        rect_dict = {
                            "x": annotation.rect.x(),
                            "y": annotation.rect.y(),
                            "width": annotation.rect.width(),
                            "height": annotation.rect.height()
                        }
                        
                        # Convert QColor to dictionary
                        color_dict = {
                            "r": annotation.color.red(),
                            "g": annotation.color.green(),
                            "b": annotation.color.blue(),
                            "a": annotation.color.alpha()
                        }
                        
                        # Create annotation dictionary
                        annotation_dict = {
                            "rect": rect_dict,
                            "class_name": annotation.class_name,
                            "attributes": annotation.attributes,
                            "color": color_dict
                        }
                        
                        project_data["frame_annotations"][str(frame_num)].append(annotation_dict)
                
                # Convert class colors to serializable format
                for class_name, color in self.canvas.class_colors.items():
                    project_data["class_colors"][class_name] = {
                        "r": color.red(),
                        "g": color.green(),
                        "b": color.blue(),
                        "a": color.alpha()
                    }
                
                # Save to file
                with open(filename, 'w') as f:
                    import json
                    json.dump(project_data, f, indent=2)
                
                self.statusBar.showMessage(f"Project saved to {os.path.basename(filename)}")
                self.logger.info(f"Project saved to {filename}")
            except Exception as e:
                error_msg = f"Failed to save project: {str(e)}"
                self.logger.error(error_msg)
                QMessageBox.critical(self, "Error", error_msg)
    
    def load_project(self):
        """Load a saved project with per-frame annotations."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Project", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if filename:
            try:
                # Load project data
                with open(filename, 'r') as f:
                    import json
                    project_data = json.load(f)
                
                # Clear existing annotations
                self.frame_annotations = {}
                self.canvas.annotations = []
                
                # Load class colors
                class_colors = {}
                for class_name, color_dict in project_data.get("class_colors", {}).items():
                    class_colors[class_name] = QColor(
                        color_dict.get("r", 255),
                        color_dict.get("g", 0),
                        color_dict.get("b", 0),
                        color_dict.get("a", 255)
                    )
                
                self.canvas.class_colors = class_colors
                
                # Load frame annotations
                from annotation import BoundingBox
                
                for frame_str, annotations_data in project_data.get("frame_annotations", {}).items():
                    frame_num = int(frame_str)
                    frame_annotations = []
                    
                    for annotation_dict in annotations_data:
                        # Create QRect from dictionary
                        rect_dict = annotation_dict.get("rect", {})
                        rect = QRect(
                            rect_dict.get("x", 0),
                            rect_dict.get("y", 0),
                            rect_dict.get("width", 0),
                            rect_dict.get("height", 0)
                        )
                        
                        # Get class name and attributes
                        class_name = annotation_dict.get("class_name", "default")
                        attributes = annotation_dict.get("attributes", {})
                        
                        # Create color from dictionary
                        color_dict = annotation_dict.get("color", {})
                        color = QColor(
                            color_dict.get("r", 255),
                            color_dict.get("g", 0),
                            color_dict.get("b", 0),
                            color_dict.get("a", 255)
                        )
                        
                        # Create bounding box
                        bbox = BoundingBox(rect, class_name, attributes, color)
                        frame_annotations.append(bbox)
                    
                    self.frame_annotations[frame_num] = frame_annotations
                
                # Load current frame if specified
                current_frame = project_data.get("current_frame", 0)
                
                # If we have a video open, go to the specified frame
                if self.cap and self.cap.isOpened():
                    self.go_to_frame(current_frame)
                else:
                    # Just set the current frame and annotations
                    self.current_frame = current_frame
                    if current_frame in self.frame_annotations:
                        self.canvas.annotations = self.frame_annotations[current_frame].copy()
                
                # Update UI
                self.toolbar.update_class_selector()
                self.class_dock.update_class_list()
                self.update_annotation_list()
                self.canvas.update()
                
                self.statusBar.showMessage(f"Project loaded from {os.path.basename(filename)}")
                self.logger.info(f"Project loaded from {filename}")
            except Exception as e:
                error_msg = f"Failed to load project: {str(e)}"
                self.logger.error(error_msg)
                QMessageBox.critical(self, "Error", error_msg)

    def export_annotations(self):
        """Export annotations to various formats using configuration."""
        if not self.canvas.annotations:
            QMessageBox.warning(self, "Export Annotations", "No annotations to export!")
            return
        
        # Create dialog for export options
        dialog = QDialog(self)
        dialog.setWindowTitle("Export Annotations")
        dialog.setMinimumWidth(300)
        
        layout = QVBoxLayout(dialog)
        
        # Format selection
        format_label = QLabel("Export Format:")
        format_combo = QComboBox()
        format_combo.addItems(list(EXPORT_FORMATS.keys()))
        
        # Add description label
        description_label = QLabel()
        description_label.setWordWrap(True)
        
        # Update description when format changes
        def update_description():
            format_name = format_combo.currentText()
            if format_name in EXPORT_FORMATS:
                description_label.setText(EXPORT_FORMATS[format_name]["description"])
                
        format_combo.currentTextChanged.connect(update_description)
        update_description()  # Set initial description
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        
        # Add widgets to layout
        layout.addWidget(format_label)
        layout.addWidget(format_combo)
        layout.addWidget(description_label)
        layout.addWidget(buttons)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            format_name = format_combo.currentText()
            format_config = EXPORT_FORMATS[format_name]
            
            # Get export filename based on format
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Annotations", "", 
                f"{format_name} Files (*.{format_config['extension']});;All Files (*)"
            )
            
            if filename:
                try:
                    # Get image dimensions from canvas
                    image_width = self.canvas.pixmap.width() if self.canvas.pixmap else 640
                    image_height = self.canvas.pixmap.height() if self.canvas.pixmap else 480
                    
                    export_annotations(filename, self.canvas.annotations, image_width, image_height, format_config['format_id'])
                    self.statusBar.showMessage(f"Annotations exported to {os.path.basename(filename)}")
                    self.logger.info(f"Exported annotations to {filename} in {format_name} format")
                except Exception as e:
                    error_msg = f"Failed to export annotations: {str(e)}"
                    self.logger.error(error_msg)
                    QMessageBox.critical(self, "Error", error_msg)
    
    def clear_annotations(self):
        """Clear all annotations"""
        if not self.canvas.annotations:
            return
        
        reply = QMessageBox.question(
            self, "Clear Annotations", 
            "Are you sure you want to clear all annotations?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.canvas.annotations = []
            self.canvas.selected_annotation = None
            self.update_annotation_list()
            self.canvas.update()
            self.statusBar.showMessage("All annotations cleared")
    
    def auto_label(self):
        """Auto-label objects in the current frame"""
        if not self.canvas.pixmap:
            QMessageBox.warning(self, "Auto Label", "Please open a video first!")
            return
        
        # This is a placeholder for actual auto-labeling functionality
        # In a real implementation, you would use a pre-trained model to detect objects
        
        QMessageBox.information(
            self, "Auto Label", 
            "Auto-labeling functionality is not implemented in this demo.\n\n"
            "In a real implementation, this would use a pre-trained model (like YOLO, SSD, or Faster R-CNN) "
            "to automatically detect and label objects in the current frame."
        )
    
    def track_objects(self):
        """Track objects across frames"""
        if not self.canvas.pixmap or not self.canvas.annotations:
            QMessageBox.warning(self, "Track Objects", "Please open a video and create annotations first!")
            return
        
        # This is a placeholder for actual object tracking functionality
        
        QMessageBox.information(
            self, "Track Objects", 
            "Object tracking functionality is not implemented in this demo.\n\n"
            "In a real implementation, this would use tracking algorithms (like KCF, CSRT, or DeepSORT) "
            "to track the annotated objects across video frames."
        )
    
    def change_style(self, style_name):
        """Change the application style with improved error handling."""
        if style_name not in self.styles:
            self.logger.warning(f"Style '{style_name}' not found, using default")
            style_name = DEFAULT_SETTINGS["default_style"]
        
        try:
            success = self.styles[style_name]()
            if success:
                self.current_style = style_name
                self.statusBar.showMessage(f"Style changed to {style_name}")
                self.logger.info(f"Style changed to {style_name}")
            else:
                self.statusBar.showMessage(f"Failed to apply {style_name} style, using fallback")
                self.logger.warning(f"Failed to apply {style_name} style, using fallback")
        except Exception as e:
            error_msg = f"Error changing style: {str(e)}"
            self.statusBar.showMessage(error_msg)
            self.logger.error(error_msg)
            # Fallback to default style
            self.styles[DEFAULT_SETTINGS["default_style"]]()
            self.current_style = DEFAULT_SETTINGS["default_style"]    
    
    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self, "About Video Annotation Tool", 
            "Video Annotation Tool (VAT)\n\n"
            "A tool for annotating objects in videos for computer vision tasks.\n\n"
            "Features:\n"
            "- Bounding box annotations\n"
            "- Multiple object classes\n"
            "- Export to common formats (COCO, YOLO, Pascal VOC)\n"
            "- Project saving and loading\n\n"
            "Created as a demonstration of PyQt5 capabilities."
        )

    def add_annotation(self, annotation):
        """Add a new annotation to the current frame"""
        if self.current_frame not in self.frame_annotations:
            self.frame_annotations[self.current_frame] = []
        
        self.frame_annotations[self.current_frame].append(annotation)
        
        # Update the annotation dock list and select the new annotation
        if hasattr(self, 'annotation_dock'):
            self.annotation_dock.update_annotation_list()
            
            # Select the newly added annotation in the list
            for i in range(self.annotation_dock.annotations_list.count()):
                item = self.annotation_dock.annotations_list.item(i)
                if item.data(Qt.UserRole) == annotation:
                    self.annotation_dock.annotations_list.setCurrentItem(item)
                    break
        
        self.logger.info(f"Added annotation: {annotation.class_name}")
  
    def select_annotation(self, annotation):
        """Select an annotation on the canvas"""
        self.selected_annotation = annotation
        self.update()  # Redraw the canvas to show the selection

    
    def edit_annotation(self, annotation):
        """Edit an existing annotation"""
        if not annotation:
            return
        
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Annotation")
        dialog.setMinimumWidth(300)
        
        layout = QVBoxLayout(dialog)
        
        # Class selection
        class_label = QLabel("Class:")
        class_combo = QComboBox()
        class_combo.addItems(list(self.canvas.class_colors.keys()))
        class_combo.setCurrentText(annotation.class_name)
        
        # Coordinates
        coords_layout = QFormLayout()
        x_spin = QSpinBox()
        x_spin.setRange(0, self.canvas.pixmap.width() if self.canvas.pixmap else 1000)
        x_spin.setValue(annotation.rect.x())
        y_spin = QSpinBox()
        y_spin.setRange(0, self.canvas.pixmap.height() if self.canvas.pixmap else 1000)
        y_spin.setValue(annotation.rect.y())
        width_spin = QSpinBox()
        width_spin.setRange(5, self.canvas.pixmap.width() if self.canvas.pixmap else 1000)
        width_spin.setValue(annotation.rect.width())
        height_spin = QSpinBox()
        height_spin.setRange(5, self.canvas.pixmap.height() if self.canvas.pixmap else 1000)
        height_spin.setValue(annotation.rect.height())
        
        coords_layout.addRow("X:", x_spin)
        coords_layout.addRow("Y:", y_spin)
        coords_layout.addRow("Width:", width_spin)
        coords_layout.addRow("Height:", height_spin)
        
        # Attributes
        attributes_label = QLabel("Attributes (key=value, one per line):")
        attributes_text = QTextEdit()
        attributes_text.setMaximumHeight(100)
        attributes_text.setText("\n".join(f"{k}={v}" for k, v in annotation.attributes.items()))
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        
        # Add widgets to layout
        layout.addWidget(class_label)
        layout.addWidget(class_combo)
        layout.addLayout(coords_layout)
        layout.addWidget(attributes_label)
        layout.addWidget(attributes_text)
        layout.addWidget(buttons)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            # Parse attributes
            attributes = {}
            for line in attributes_text.toPlainText().strip().split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    attributes[key.strip()] = value.strip()
        
            # Update annotation
            annotation.rect = QRect(x_spin.value(), y_spin.value(), width_spin.value(), height_spin.value())
            annotation.class_name = class_combo.currentText()
            annotation.color = self.canvas.class_colors.get(annotation.class_name, QColor(255, 0, 0))
            annotation.attributes = attributes
        
            self.update_annotation_list()
            self.canvas.update()
    
    def delete_annotation(self, annotation):
        """Delete an annotation from the current frame"""
        current_frame = self.current_frame
        
        # Remove from frame_annotations
        if current_frame in self.frame_annotations:
            if annotation in self.frame_annotations[current_frame]:
                self.frame_annotations[current_frame].remove(annotation)
        
        # Remove from canvas annotations
        if annotation in self.canvas.annotations:
            self.canvas.annotations.remove(annotation)
        
        # Update the annotation dock list
        if hasattr(self, 'annotation_dock'):
            self.annotation_dock.update_annotation_list()
        
        # Update canvas
        self.canvas.update()
        
        self.logger.info(f"Deleted annotation: {annotation.class_name}")
   
    def delete_selected_annotation(self):
        """Delete the selected annotation"""
        if not self.canvas.selected_annotation:
            return
        
        reply = QMessageBox.question(
            self, "Delete Annotation", 
            "Are you sure you want to delete this annotation?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            self.canvas.annotations.remove(self.canvas.selected_annotation)
            self.canvas.selected_annotation = None
            self.update_annotation_list()
            self.canvas.update()
    
    def add_class(self):
        """Add a new class"""
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Class")
        
        layout = QVBoxLayout(dialog)
        
        # Class name
        name_layout = QHBoxLayout()
        name_label = QLabel("Class Name:")
        name_edit = QLineEdit()
        name_layout.addWidget(name_label)
        name_layout.addWidget(name_edit)
        
        # Color selection
        color_layout = QHBoxLayout()
        color_label = QLabel("Color:")
        color_button = QPushButton()
        color_button.setAutoFillBackground(True)
        
        # Set initial color
        color = QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        color_button.setStyleSheet(f"background-color: {color.name()}")
        
        def choose_color():
            nonlocal color
            new_color = QColorDialog.getColor(color, dialog, "Select Color")
            if new_color.isValid():
                color = new_color
                color_button.setStyleSheet(f"background-color: {color.name()}")

        color_button.clicked.connect(choose_color)
        color_layout.addWidget(color_label)
        color_layout.addWidget(color_button)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        # Add widgets to layout
        layout.addLayout(name_layout)
        layout.addLayout(color_layout)
        layout.addWidget(buttons)

        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            class_name = name_edit.text().strip()
            
            if not class_name:
                QMessageBox.warning(self, "Add Class", "Class name cannot be empty!")
                return
            
            if class_name in self.canvas.class_colors:
                QMessageBox.warning(self, "Add Class", f"Class '{class_name}' already exists!")
                return
            
            # Add class to canvas
            self.canvas.class_colors[class_name] = color
            
            # Update UI
            self.toolbar.update_class_selector()
            self.class_dock.update_class_list()
            
            # Set as current class
            self.canvas.set_current_class(class_name)
            self.class_selector.setCurrentText(class_name)
    
    def edit_selected_class(self):
        """Edit the selected class"""
        item = self.class_dock.classes_list.currentItem()
        if not item:
            return
        
        class_name = item.text()
        current_color = self.canvas.class_colors.get(class_name, QColor(255, 0, 0))
        
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Class")
        
        layout = QVBoxLayout(dialog)
        
        # Class name
        name_layout = QHBoxLayout()
        name_label = QLabel("Class Name:")
        name_edit = QLineEdit(class_name)
        name_layout.addWidget(name_label)
        name_layout.addWidget(name_edit)
        
        # Color selection
        color_layout = QHBoxLayout()
        color_label = QLabel("Color:")
        color_button = QPushButton()
        color_button.setAutoFillBackground(True)
        
        # Set initial color
        color = current_color
        color_button.setStyleSheet(f"background-color: {color.name()}")
        
        def choose_color():
            nonlocal color
            new_color = QColorDialog.getColor(color, dialog, "Select Color")
            if new_color.isValid():
                color = new_color
                color_button.setStyleSheet(f"background-color: {color.name()}")
        
        color_button.clicked.connect(choose_color)
        color_layout.addWidget(color_label)
        color_layout.addWidget(color_button)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        
        # Add widgets to layout
        layout.addLayout(name_layout)
        layout.addLayout(color_layout)
        layout.addWidget(buttons)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            new_class_name = name_edit.text().strip()
            
            if not new_class_name:
                QMessageBox.warning(self, "Edit Class", "Class name cannot be empty!")
                return
            
            if new_class_name != class_name and new_class_name in self.canvas.class_colors:
                QMessageBox.warning(self, "Edit Class", f"Class '{new_class_name}' already exists!")
                return
            
            # Update class
            if new_class_name != class_name:
                # Update class name in annotations
                for annotation in self.canvas.annotations:
                    if annotation.class_name == class_name:
                        annotation.class_name = new_class_name
            
                # Update class colors dictionary
                self.canvas.class_colors[new_class_name] = color
                del self.canvas.class_colors[class_name]
            else:
                # Just update the color
                self.canvas.class_colors[class_name] = color
            
            # Update annotations with new color
            for annotation in self.canvas.annotations:
                if annotation.class_name == new_class_name:
                    annotation.color = color
            
            # Update UI
            self.toolbar.update_class_selector()
            self.class_dock.update_class_list()
            self.update_annotation_list()
            
            # Update canvas
            self.canvas.update()
    
    def delete_selected_class(self):
        """Delete the selected class"""
        item = self.class_dock.classes_list.currentItem()
        if not item:
            return
        
        class_name = item.text()
        
        # Check if class is in use
        in_use = False
        for annotation in self.canvas.annotations:
            if annotation.class_name == class_name:
                in_use = True
                break
        
        message = f"Are you sure you want to delete the class '{class_name}'?"
        if in_use:
            message += "\n\nThis class is currently in use by annotations. Deleting it will remove all annotations of this class."
        
        reply = QMessageBox.question(
            self, "Delete Class", 
            message,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # Remove annotations of this class
            self.canvas.annotations = [a for a in self.canvas.annotations if a.class_name != class_name]
            
            # Remove class from colors dictionary
            if class_name in self.canvas.class_colors:
                del self.canvas.class_colors[class_name]
            
            # Update UI
            self.toolbar.update_class_selector()
            self.class_dock.update_class_list()
            self.update_annotation_list()
            
            # Update canvas
            if self.class_selector.count() > 0:
                self.canvas.set_current_class(self.class_selector.currentText())
            self.canvas.update()
    
    def monitor_performance(self):
        """Monitor application performance."""
        from utils.performance import PerformanceMonitor
        
        memory_usage = PerformanceMonitor.get_memory_usage()
        self.logger.info(f"Memory usage: {memory_usage:.2f} MB")
        
        # Update status bar with memory usage
        self.statusBar.showMessage(f"Memory usage: {memory_usage:.2f} MB", 3000)  # Show for 3 seconds
        
        # Schedule next monitoring
        QTimer.singleShot(60000, self.monitor_performance)  # Check every minute

    def toggle_play(self):
        """Toggle video playback."""
        if not self.cap:
            QMessageBox.warning(self, "Play Video", "Please open a video first!")
            return
        
        if self.is_playing:
            # Stop playback
            self.play_timer.stop()
            self.is_playing = False
            self.statusBar.showMessage("Paused")
            # Update play/pause button text if you have one
            if hasattr(self, 'play_button'):
                self.play_button.setText("Play")
        else:
            # Start playback
            # Get the frame rate from the video
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30  # Default to 30 fps if unable to determine
            
            # Set timer interval based on fps and playback speed
            interval = int(1000 / (fps * self.playback_speed))
            self.play_timer.setInterval(interval)
            
            # Start the timer
            self.play_timer.start()
            self.is_playing = True
            self.statusBar.showMessage("Playing")
            # Update play/pause button text if you have one
            if hasattr(self, 'play_button'):
                self.play_button.setText("Pause")

    def zoom_in(self):
        """Zoom in on the canvas."""
        if hasattr(self, 'canvas'):
            self.canvas.zoom_in()

    def zoom_out(self):
        """Zoom out on the canvas."""
        if hasattr(self, 'canvas'):
            self.canvas.zoom_out()

    def reset_zoom(self):
        """Reset zoom to original size."""
        if hasattr(self, 'canvas'):
            self.canvas.reset_zoom()

    def fit_to_view(self):
        """Fit the video to the view."""
        if hasattr(self, 'canvas'):
            self.canvas.fit_to_view()