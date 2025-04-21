"""
Video Annotation Tool (VAT) - Main Application

This module contains the main application window and program entry point for the
Video Annotation Tool. It provides the UI framework and coordinates between the
different components of the application.
"""

import os
import sys
import random
import cv2
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QPushButton, QSlider, QLabel, QAction, QFileDialog, QDockWidget,
    QToolBar, QStatusBar, QComboBox, QMessageBox, QListWidget, 
    QListWidgetItem, QMenu, QDialog, QFormLayout, QSpinBox, QTextEdit,
    QDialogButtonBox, QLineEdit, QColorDialog, QActionGroup
)
from PyQt5.QtCore import Qt, QTimer, QRect, QPoint
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QIcon, QPalette

from canvas import VideoCanvas
from annotation import BoundingBox
from widgets.styles import StyleManager
from widgets import AnnotationDock, ClassDock, AnnotationToolbar
from utils import save_project, load_project, export_annotations


class VideoAnnotationTool(QMainWindow):
    """
    Main application window for the Video Annotation Tool.
    
    This class manages the UI components, video playback, and annotation functionality.
    It serves as the central coordinator between different parts of the application.
    """
    
    def __init__(self):
        """Initialize the main application window and its components."""
        super().__init__()
        
        # Basic window setup
        self.setWindowTitle("Video Annotation Tool")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize properties
        self.init_properties()
        
        # Set up the user interface
        self.init_ui()
        
    def init_properties(self):
        """Initialize the application properties and state variables."""
        # Available styles
        self.styles = {
            "Default": StyleManager.set_default_style,
            "Fusion": StyleManager.set_fusion_style,
            "Windows": StyleManager.set_windows_style,
            "Dark": StyleManager.set_dark_style,
            "Light": StyleManager.set_light_style,
            "Blue": StyleManager.set_blue_style,
            "Green": StyleManager.set_green_style,
        }
        
        # Annotation methods
        self.annotation_methods = {
            "Rectangle": "Draw rectangular bounding boxes",
            "Polygon": "Draw polygon shapes",
            "Point": "Mark specific points"
        }
        self.current_annotation_method = "Rectangle"
        
        # Application state
        self.current_style = "Default"
        self.playback_speed = 1.0
        self.cap = None  # Video capture object
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0
        self.zoom_level = 1.0
        
        # Annotation data
        self.frame_annotations = {}  # Dictionary to store annotations by frame number
        
        # Project state
        self.project_file = None
        self.project_modified = False
        self.autosave_timer = None
        self.autosave_interval = 300000  # 5 minutes in milliseconds

    def init_ui(self):
        """Initialize the user interface components."""
        # Create central widget with video canvas
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)
        self.canvas = VideoCanvas(self)
        main_layout.addWidget(self.canvas)
        playback_controls = self.create_playback_controls()
        main_layout.addWidget(playback_controls)
    
        self.setCentralWidget(central_widget)
        # Create UI components
        self.create_menu_bar()
        self.create_toolbar()
        self.create_dock_widgets()
        self.create_status_bar()
        
        # Set up timer for video playback
        self.setup_playback_timer()
        
        # Initialize annotation list
        self.update_annotation_list()
        
    def create_menu_bar(self):
        """Create the application menu bar and its actions."""
        menubar = self.menuBar()
        
        
        # File menu
        self.create_file_menu(menubar)
        
        # Edit menu
        self.create_edit_menu(menubar)
        
        # Tools menu
        self.create_tools_menu(menubar)
        
        # Style menu
        self.create_style_menu(menubar)
        
        # Help menu
        self.create_help_menu(menubar)
    
    def create_file_menu(self, menubar):
        """Create the File menu and its actions."""
        file_menu = menubar.addMenu("File")
        
        # Open Video action
        open_action = QAction("Open Video", self)
        open_action.triggered.connect(self.open_video)
        file_menu.addAction(open_action)
        
        # Save Project action
        save_action = QAction("Save Project", self)
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)
        
        # Load Project action
        load_action = QAction("Load Project", self)
        load_action.triggered.connect(self.load_project)
        file_menu.addAction(load_action)
        
        # Export Annotations action
        export_action = QAction("Export Annotations", self)
        export_action.triggered.connect(self.export_annotations)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
    
    def create_edit_menu(self, menubar):
        """Create the Edit menu and its actions."""
        edit_menu = menubar.addMenu("Edit")
        
        # Clear Annotations action
        clear_action = QAction("Clear Annotations", self)
        clear_action.triggered.connect(self.clear_annotations)
        edit_menu.addAction(clear_action)
        
        # Add Annotation action
        add_action = QAction("Add Annotation", self)
        add_action.triggered.connect(self.add_annotation)
        edit_menu.addAction(add_action)
        
        # Add Class action
        add_class_action = QAction("Add Class", self)
        add_class_action.triggered.connect(self.add_class)
        edit_menu.addAction(add_class_action)
    
    def create_tools_menu(self, menubar):
        """Create the Tools menu and its actions."""
        tools_menu = menubar.addMenu("Tools")
        
        # Auto Label action
        auto_label_action = QAction("Auto Label", self)
        auto_label_action.triggered.connect(self.auto_label)
        tools_menu.addAction(auto_label_action)
        
        # Track Objects action
        track_action = QAction("Track Objects", self)
        track_action.triggered.connect(self.track_objects)
        tools_menu.addAction(track_action)
    
    def create_style_menu(self, menubar):
        """Create the Style menu and its actions."""
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
    
    def create_help_menu(self, menubar):
        """Create the Help menu and its actions."""
        help_menu = menubar.addMenu("Help")
        
        # About action
        about_action = QAction("About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def create_toolbar(self):
        """Create the annotation toolbar."""
        self.toolbar = AnnotationToolbar(self)
        self.addToolBar(self.toolbar)
        self.class_selector = self.toolbar.class_selector
    
    def create_dock_widgets(self):
        """Create and set up the dock widgets."""
        # Annotation dock
        self.annotation_dock = AnnotationDock(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.annotation_dock)
        
        # Class dock
        self.class_dock = ClassDock(self)
        self.addDockWidget(Qt.RightDockWidgetArea, self.class_dock)
    
    def create_status_bar(self):
        """Create the status bar."""
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
    def create_playback_controls(self):
        """Create video playback controls."""
        # Create a widget to hold the controls
        playback_widget = QWidget()
        playback_layout = QHBoxLayout(playback_widget)
        
        # Play/Pause button
        self.play_button = QPushButton()
        self.play_button.setIcon(QIcon.fromTheme("media-playback-start"))
        self.play_button.setToolTip("Play/Pause")
        self.play_button.clicked.connect(self.play_pause_video)
        
        # Previous frame button
        prev_button = QPushButton()
        prev_button.setIcon(QIcon.fromTheme("media-skip-backward"))
        prev_button.setToolTip("Previous Frame")
        prev_button.clicked.connect(self.prev_frame)
        
        # Next frame button
        next_button = QPushButton()
        next_button.setIcon(QIcon.fromTheme("media-skip-forward"))
        next_button.setToolTip("Next Frame")
        next_button.clicked.connect(self.next_frame)
        
        # Frame slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(100)  # Will be updated when video is loaded
        self.frame_slider.valueChanged.connect(self.slider_changed)
        
        # Frame counter label
        self.frame_label = QLabel("0/0")
        
        # Add widgets to layout
        playback_layout.addWidget(prev_button)
        playback_layout.addWidget(self.play_button)
        playback_layout.addWidget(next_button)
        playback_layout.addWidget(self.frame_slider)
        playback_layout.addWidget(self.frame_label)
        
        return playback_widget

    def setup_playback_timer(self):
        """Set up the timer for video playback."""
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.next_frame)
    def update_annotation_list(self):
        """Update the annotation list in the UI."""
        if hasattr(self, 'annotation_dock'):
            self.annotation_dock.update_annotation_list()

    #
    # Video handling methods
    #
    def change_annotation_method(self, method_name):
        """Change the current annotation method."""
        if method_name in self.annotation_methods:
            self.current_annotation_method = method_name
            
            # Update canvas annotation mode if needed
            if hasattr(self.canvas, 'set_annotation_mode'):
                self.canvas.set_annotation_mode(method_name)

    def zoom_in(self):
        """Zoom in on the canvas."""
        self.zoom_level *= 1.2
        if hasattr(self.canvas, 'set_zoom'):
            self.canvas.set_zoom(self.zoom_level)
        else:
            self.canvas.update()

    def zoom_out(self):
        """Zoom out on the canvas."""
        self.zoom_level /= 1.2
        if hasattr(self.canvas, 'set_zoom'):
            self.canvas.set_zoom(self.zoom_level)
        else:
            self.canvas.update()

    def reset_zoom(self):
        """Reset zoom to default level."""
        self.zoom_level = 1.0
        if hasattr(self.canvas, 'set_zoom'):
            self.canvas.set_zoom(self.zoom_level)
        else:
            self.canvas.update()

    
    def open_video(self):
        """Open a video file."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Open Video", "", "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if filename:
            self.load_video_file(filename)
    
    def load_video_file(self, filename):
        """Load a video file and display the first frame."""
        # Close any existing video
        if self.cap:
            self.cap.release()
        
        # Open the video file
        self.cap = cv2.VideoCapture(filename)
        
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open video file!")
            self.cap = None
            return False
        
        # Get video properties
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        
        # Update slider range
        self.frame_slider.setMaximum(self.total_frames - 1)
        
        # Read the first frame
        ret, frame = self.cap.read()
        if ret:
            self.canvas.set_frame(frame)
            self.update_frame_info()
            self.statusBar.showMessage(f"Loaded video: {os.path.basename(filename)}")
            return True
        else:
            QMessageBox.critical(self, "Error", "Could not read video frame!")
            self.cap.release()
            self.cap = None
            return False

        
    def slider_changed(self, value):
        """Handle slider value changes."""
        if self.cap and self.cap.isOpened():
            # Calculate the frame number from slider value
            frame_number = int(value)
            
            # Set video to that frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame_number
                self.canvas.set_frame(frame)
                self.update_frame_info()
    def update_frame_info(self):
        """Update frame information in the UI."""
        if self.cap and self.cap.isOpened():
            # Update frame label
            self.frame_label.setText(f"{self.current_frame}/{self.total_frames}")
            
            # Update slider position (without triggering valueChanged)
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(self.current_frame)
            self.frame_slider.blockSignals(False)
    def prev_frame(self):
        """Go to the previous frame in the video."""
        if self.cap and self.cap.isOpened():
            # Get current position
            current_pos = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            
            # Go back one frame
            if current_pos > 1:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 2)
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame = current_pos - 1
                    self.canvas.set_frame(frame)
                    self.update_frame_info()
    def next_frame(self):
        """Go to the next frame in the video."""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame += 1
                self.canvas.set_frame(frame)
                self.update_frame_info()
            else:
                # End of video
                self.play_timer.stop()
                self.is_playing = False
                self.statusBar.showMessage("End of video")
                # Rewind to beginning
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame = 0
                ret, frame = self.cap.read()
                if ret:
                    self.canvas.set_frame(frame)
                    self.update_frame_info()

    def play_pause_video(self):
        """Toggle between playing and pausing the video."""
        if not self.cap:
            return
            
        if self.is_playing:
            self.play_timer.stop()
            self.is_playing = False
            self.statusBar.showMessage("Paused")
        else:
            # Set timer interval based on playback speed
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            interval = int(1000 / (fps * self.playback_speed))
            self.play_timer.start(interval)
            self.is_playing = True
            self.statusBar.showMessage("Playing")
    
    #
    # Project handling methods
    #
    
    def save_project(self):
        """Save the current project."""
        if not self.canvas.annotations:
            QMessageBox.warning(self, "Save Project", "No annotations to save!")
            return
        
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save Project", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if filename:
            save_project(filename, self.canvas.annotations, self.canvas.class_colors)
            self.statusBar.showMessage(f"Project saved to {os.path.basename(filename)}")
    
    def load_project(self):
        """Load a saved project."""
        filename, _ = QFileDialog.getOpenFileName(
            self, "Load Project", "", "JSON Files (*.json);;All Files (*)"
        )
        
        if filename:
            try:
                annotations, class_colors = load_project(filename, BoundingBox)
                
                # Update canvas
                self.canvas.annotations = annotations
                self.canvas.class_colors = class_colors
                
                # Update UI
                self.update_annotation_list()
                self.toolbar.update_class_selector()
                self.class_dock.update_class_list()
                self.canvas.update()
                
                self.statusBar.showMessage(f"Project loaded from {os.path.basename(filename)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load project: {str(e)}")
    
    def export_annotations(self):
        """Export annotations to various formats."""
        if not self.canvas.annotations:
            QMessageBox.warning(self, "Export Annotations", "No annotations to export!")
            return
        
        # Create dialog for export options
        dialog = self.create_export_dialog()
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            format_combo = dialog.findChild(QComboBox)
            format_type = format_combo.currentText()
            
            self.export_annotations_with_format(format_type)
    
    def create_export_dialog(self):
        """Create a dialog for export options."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Export Annotations")
        dialog.setMinimumWidth(300)
        
        layout = QVBoxLayout(dialog)
        
        # Format selection
        format_label = QLabel("Export Format:")
        format_combo = QComboBox()
        format_combo.addItems(["COCO JSON", "YOLO TXT", "Pascal VOC XML"])
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        
        # Add widgets to layout
        layout.addWidget(format_label)
        layout.addWidget(format_combo)
        layout.addWidget(buttons)
        
        return dialog
    
    def export_annotations_with_format(self, format_type):
        """Export annotations with the specified format."""
        # Get export filename based on format
        if format_type == "COCO JSON":
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Annotations", "", "JSON Files (*.json);;All Files (*)"
            )
            export_format = "coco"
        elif format_type == "YOLO TXT":
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Annotations", "", "Text Files (*.txt);;All Files (*)"
            )
            export_format = "yolo"
        elif format_type == "Pascal VOC XML":
            filename, _ = QFileDialog.getSaveFileName(
                self, "Export Annotations", "", "XML Files (*.xml);;All Files (*)"
            )
            export_format = "pascal_voc"
        else:
            return
        
        if filename:
            try:
                # Get image dimensions from canvas
                image_width = self.canvas.pixmap.width() if self.canvas.pixmap else 640
                image_height = self.canvas.pixmap.height() if self.canvas.pixmap else 480
                
                export_annotations(filename, self.canvas.annotations, image_width, image_height, export_format)
                self.statusBar.showMessage(f"Annotations exported to {os.path.basename(filename)}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export annotations: {str(e)}")
    
    #
    # Annotation handling methods
    #
    
    def update_annotation_list(self):
        """Update the annotations list widget."""
        self.annotation_dock.update_annotation_list()
    
    def clear_annotations(self):
        """Clear all annotations."""
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
    
    def add_annotation(self):
        """Add annotation manually."""
        if not self.cap:
            QMessageBox.warning(self, "Add Annotation", "Please open a video first!")
            return
        
        # Create dialog
        dialog = self.create_annotation_dialog()
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            # Get form widgets
            class_combo = dialog.findChild(QComboBox)
            x_spin = dialog.findChildren(QSpinBox)[0]
            y_spin = dialog.findChildren(QSpinBox)[1]
            width_spin = dialog.findChildren(QSpinBox)[2]
            height_spin = dialog.findChildren(QSpinBox)[3]
            size_spin = dialog.findChildren(QSpinBox)[4]
            quality_spin = dialog.findChildren(QSpinBox)[5]
            
            # Create attributes dictionary
            attributes = {
                "Size": size_spin.value(),
                "Quality": quality_spin.value()
            }
        
            # Create rectangle
            rect = QRect(x_spin.value(), y_spin.value(), width_spin.value(), height_spin.value())
        
            # Get class and color
            class_name = class_combo.currentText()
            color = self.canvas.class_colors.get(class_name, QColor(255, 0, 0))
        
            # Create bounding box
            bbox = BoundingBox(rect, class_name, attributes, color)
        
            # Add to annotations
            self.canvas.annotations.append(bbox)
            
            # Save to frame annotations
            self.frame_annotations[self.current_frame] = self.canvas.annotations
            
            self.update_annotation_list()
            self.canvas.update()

    
    def create_annotation_dialog(self):
        """Create a dialog for adding or editing annotations."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Annotation")
        dialog.setMinimumWidth(300)
        
        layout = QVBoxLayout(dialog)
        
        # Class selection
        class_label = QLabel("Class:")
        class_combo = QComboBox()
        class_combo.addItems(list(self.canvas.class_colors.keys()))
        
        # Coordinates
        coords_layout = QFormLayout()
        x_spin = QSpinBox()
        x_spin.setRange(0, self.canvas.pixmap.width() if self.canvas.pixmap else 1000)
        y_spin = QSpinBox()
        y_spin.setRange(0, self.canvas.pixmap.height() if self.canvas.pixmap else 1000)
        width_spin = QSpinBox()
        width_spin.setRange(5, self.canvas.pixmap.width() if self.canvas.pixmap else 1000)
        height_spin = QSpinBox()
        height_spin.setRange(5, self.canvas.pixmap.height() if self.canvas.pixmap else 1000)
        
        coords_layout.addRow("X:", x_spin)
        coords_layout.addRow("Y:", y_spin)
        coords_layout.addRow("Width:", width_spin)
        coords_layout.addRow("Height:", height_spin)
        
        # Attributes
        attributes_layout = QFormLayout()
        size_spin = QSpinBox()
        size_spin.setRange(0, 100)
        size_spin.setValue(1)  # Default value
        
        quality_spin = QSpinBox()
        quality_spin.setRange(0, 100)
        quality_spin.setValue(1)  # Default value
        
        attributes_layout.addRow("Size (0-100):", size_spin)
        attributes_layout.addRow("Quality (0-100):", quality_spin)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        
        # Add widgets to layout
        layout.addWidget(class_label)
        layout.addWidget(class_combo)
        layout.addLayout(coords_layout)
        layout.addLayout(attributes_layout)
        layout.addWidget(buttons)
        
        return dialog

    
    def parse_attributes(self, text):
        """Parse attributes from text input."""
        attributes = {}
        for line in text.strip().split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                attributes[key.strip()] = value.strip()
        return attributes
    
    def edit_annotation(self, annotation):
        """Edit the properties of an annotation."""
        
        if not annotation:
            return
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Annotation")
        
        layout = QVBoxLayout()
        form_layout = QFormLayout()
        
        # Class selector
        class_combo = QComboBox()
        class_combo.addItems(self.canvas.class_colors.keys())
        class_combo.setCurrentText(annotation.class_name)
        form_layout.addRow("Class:", class_combo)
        
        # Size attribute (numeric 0-100)
        size_spinner = QSpinBox()
        size_spinner.setRange(0, 100)
        size_spinner.setValue(int(annotation.attributes.get("Size", -1)))
        form_layout.addRow("Size (0-100):", size_spinner)
        
        # Quality attribute (numeric 0-100)
        quality_spinner = QSpinBox()
        quality_spinner.setRange(0, 100)
        quality_spinner.setValue(int(annotation.attributes.get("Quality", -1)))
        form_layout.addRow("Quality (0-100):", quality_spinner)
        
        # Add custom attributes if needed
        # For example, an ID field
        id_field = QLineEdit(getattr(annotation, 'id', ''))
        form_layout.addRow("ID:", id_field)
        
        layout.addLayout(form_layout)
        
        # Add OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        dialog.setLayout(layout)
        
        # If dialog is accepted, update the annotation
        if dialog.exec_() == QDialog.Accepted:
            annotation.class_name = class_combo.currentText()
            annotation.color = self.canvas.class_colors[annotation.class_name]
            annotation.id = id_field.text()
            annotation.attributes["Size"] = size_spinner.value()
            annotation.attributes["Quality"] = quality_spinner.value()
            
            # Update canvas and mark project as modified
            self.canvas.update()
            self.project_modified = True
            
            # Update annotation list
            self.update_annotation_list()
            
            # Save to frame annotations
            self.frame_annotations[self.current_frame] = self.canvas.annotations

    
    def delete_selected_annotation(self):
        """Delete the currently selected annotation."""
        if hasattr(self.canvas, 'selected_annotation') and self.canvas.selected_annotation:
            # Remove from annotations list
            self.canvas.annotations.remove(self.canvas.selected_annotation)
            
            # Clear selection
            self.canvas.selected_annotation = None
            
            # Update canvas and mark project as modified
            self.canvas.update()
            self.project_modified = True
            
            # Update annotation list in UI if it exists
            if hasattr(self, 'update_annotation_list'):
                self.update_annotation_list()
    
    #
    # Class handling methods
    #
    
    def add_class(self):
        """Add a new class."""
        # Create dialog
        dialog = self.create_class_dialog()
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            # Get form widgets
            name_edit = dialog.findChild(QLineEdit)
            color_button = dialog.findChild(QPushButton)
            
            class_name = name_edit.text().strip()
            
            if not class_name:
                QMessageBox.warning(self, "Add Class", "Class name cannot be empty!")
                return
            
            if class_name in self.canvas.class_colors:
                QMessageBox.warning(self, "Add Class", f"Class '{class_name}' already exists!")
                return
            
            # Get color from button's stylesheet
            color_str = color_button.styleSheet().split(":")[-1].strip()
            color = QColor(color_str)
            
            # Add class to canvas
            self.canvas.class_colors[class_name] = color
            
            # Update UI
            self.toolbar.update_class_selector()
            self.class_dock.update_class_list()
            
            # Set as current class
            self.canvas.set_current_class(class_name)
            self.class_selector.setCurrentText(class_name)
    
    def create_class_dialog(self, class_name=None, color=None):
        """Create a dialog for adding or editing classes."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Class" if class_name is None else "Edit Class")
        
        layout = QVBoxLayout(dialog)
        
        # Class name
        name_layout = QHBoxLayout()
        name_label = QLabel("Class Name:")
        name_edit = QLineEdit(class_name if class_name else "")
        name_layout.addWidget(name_label)
        name_layout.addWidget(name_edit)
        
        # Color selection
        color_layout = QHBoxLayout()
        color_label = QLabel("Color:")
        color_button = QPushButton()
        color_button.setAutoFillBackground(True)
        
        # Set initial color
        if color is None:
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
        
        return dialog
    
    def edit_selected_class(self):
        """Edit the selected class."""
        item = self.class_dock.classes_list.currentItem()
        if not item:
            return
        
        class_name = item.text()
        current_color = self.canvas.class_colors.get(class_name, QColor(255, 0, 0))
        
        # Create dialog
        dialog = self.create_class_dialog(class_name, current_color)
        
        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            # Get form widgets
            name_edit = dialog.findChild(QLineEdit)
            color_button = dialog.findChild(QPushButton)
            
            new_class_name = name_edit.text().strip()
            
            if not new_class_name:
                QMessageBox.warning(self, "Edit Class", "Class name cannot be empty!")
                return
            
            if new_class_name != class_name and new_class_name in self.canvas.class_colors:
                QMessageBox.warning(self, "Edit Class", f"Class '{new_class_name}' already exists!")
                return
            
            # Get color from button's stylesheet
            color_str = color_button.styleSheet().split(":")[-1].strip()
            color = QColor(color_str)
            
            # Update class
            self.update_class(class_name, new_class_name, color)
    
    def update_class(self, old_name, new_name, color):
        """Update a class with new name and color."""
        # Update class name in annotations
        if old_name != new_name:
            for annotation in self.canvas.annotations:
                if annotation.class_name == old_name:
                    annotation.class_name = new_name
        
            # Update class colors dictionary
            self.canvas.class_colors[new_name] = color
            del self.canvas.class_colors[old_name]
        else:
            # Just update the color
            self.canvas.class_colors[old_name] = color
        
        # Update annotations with new color
        for annotation in self.canvas.annotations:
            if annotation.class_name == new_name:
                annotation.color = color
        
        # Update UI
        self.toolbar.update_class_selector()
        self.class_dock.update_class_list()
        self.update_annotation_list()
        
        # Update canvas
        self.canvas.update()
    
    def delete_selected_class(self):
        """Delete the selected class."""
        item = self.class_dock.classes_list.currentItem()
        if not item:
            return
        
        class_name = item.text()
        
        # Check if class is in use
        in_use = any(annotation.class_name == class_name for annotation in self.canvas.annotations)
        
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
    
    #
    # Tool methods
    #
    
    def auto_label(self):
        """Auto-label objects in the current frame."""
        if not self.canvas.pixmap:
            QMessageBox.warning(self, "Auto Label", "Please open a video first!")
            return
        
        # This is a placeholder for actual auto-labeling functionality
        QMessageBox.information(
            self, "Auto Label", 
            "Auto-labeling functionality is not implemented in this demo.\n\n"
            "In a real implementation, this would use a pre-trained model (like YOLO, SSD, or Faster R-CNN) "
            "to automatically detect and label objects in the current frame."
        )
    
    def track_objects(self):
        """Track objects across frames."""
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
    
    #
    # UI utility methods
    #
    
    def change_style(self, style_name):
        """Change the application style."""
        if style_name in self.styles:
            self.styles[style_name]()
            self.current_style = style_name
            self.statusBar.showMessage(f"Style changed to {style_name}")
    
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self, "About Video Annotation Tool", 
            "Video Annotation Tool (VAT)\n\n"
            "A tool for annotating objects in videos for computer vision tasks.\n\n"
            "Features:\n"
            "- Bounding box annotations with edge movement for precise adjustments\n"
            "- Multiple object classes with customizable colors\n"
            "- Export to common formats (COCO, YOLO, Pascal VOC)\n"
            "- Project saving and loading\n"
            "- Right-click context menu for quick editing\n\n"
            "Created as a demonstration of PyQt5 capabilities."
        )


