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
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_video)
        file_menu.addAction(open_action)
        
        # Save Project action
        save_action = QAction("Save Project", self)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.save_project)
        file_menu.addAction(save_action)
        
        # Load Project action
        load_action = QAction("Load Project", self)
        load_action.setShortcut("Ctrl+L")
        load_action.triggered.connect(self.load_project)
        file_menu.addAction(load_action)
        
        # Import Annotations action
        import_action = QAction("Import Annotations", self)
        import_action.setShortcut("Ctrl+I")
        import_action.triggered.connect(self.import_annotations)
        file_menu.addAction(import_action)
        
        # Export Annotations action
        export_action = QAction("Export Annotations", self)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.export_annotations)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        # Exit action
        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
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
            
            # Check for annotation files with the same name
            self.check_for_annotation_files(filename)
            
            return True
        else:
            QMessageBox.critical(self, "Error", "Could not read video frame!")
            self.cap.release()
            self.cap = None
            return False

    def check_for_annotation_files(self, video_filename):
        """
        Check if annotation files with the same base name as the video exist.
        If found, ask the user if they want to import them.
        
        Args:
            video_filename (str): Path to the video file
        """
        # Get the directory and base name without extension
        directory = os.path.dirname(video_filename)
        base_name = os.path.splitext(os.path.basename(video_filename))[0]
        
        # List of possible annotation file extensions to check
        extensions = ['.txt', '.json', '.xml']
        
        # Find matching annotation files
        annotation_files = []
        for ext in extensions:
            potential_file = os.path.join(directory, base_name + ext)
            if os.path.exists(potential_file):
                annotation_files.append(potential_file)
        
        if annotation_files:
            # Create a message with the found files
            message = "Found the following annotation file(s):\n\n"
            for file in annotation_files:
                message += f"- {os.path.basename(file)}\n"
            message += "\nWould you like to import annotations from one of these files?"
            
            # Show dialog asking if user wants to import
            reply = QMessageBox.question(
                self, "Annotation Files Found", 
                message,
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            
            if reply == QMessageBox.Yes:
                # If multiple files found, let user choose which one to import
                if len(annotation_files) > 1:
                    self.show_annotation_file_selection_dialog(annotation_files)
                else:
                    # Only one file found, import it directly
                    self.import_annotations(annotation_files[0])

    def show_annotation_file_selection_dialog(self, annotation_files):
        """
        Show a dialog for the user to select which annotation file to import.
        
        Args:
            annotation_files (list): List of annotation file paths
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Annotation File")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout(dialog)
        
        # Add explanation label
        label = QLabel("Multiple annotation files found. Please select which one to import:")
        layout.addWidget(label)
        
        # Create list widget with annotation files
        list_widget = QListWidget()
        for file in annotation_files:
            item = QListWidgetItem(os.path.basename(file))
            item.setData(Qt.UserRole, file)  # Store full path as data
            list_widget.addItem(item)
        
        layout.addWidget(list_widget)
        
        # Add buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        # Show dialog and process result
        if dialog.exec_() == QDialog.Accepted:
            selected_items = list_widget.selectedItems()
            if selected_items:
                selected_file = selected_items[0].data(Qt.UserRole)
                self.import_annotations(selected_file)

    def update_frame_info(self):
        """Update frame information in the UI."""
        if self.cap and self.cap.isOpened():
            # Update frame label
            self.frame_label.setText(f"{self.current_frame}/{self.total_frames}")
            
            # Update slider position (without triggering valueChanged)
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(self.current_frame)
            self.frame_slider.blockSignals(False)
 
    def update_frame_annotations(self):
        """Update annotations for the current frame."""
        # Save current annotations to frame_annotations dictionary
        if hasattr(self.canvas, 'annotations') and self.canvas.annotations:
            self.frame_annotations[self.current_frame] = self.canvas.annotations.copy()
        
        # Load annotations for the new current frame
        if self.current_frame in self.frame_annotations:
            self.canvas.annotations = self.frame_annotations[self.current_frame]
        else:
            self.canvas.annotations = []
        
        # Update the annotation list in the UI
        self.update_annotation_list()
        
        # Update the canvas
        self.canvas.update()

    def load_current_frame_annotations(self):
        """Load annotations for the current frame into the canvas."""
        # Clear canvas selection
        self.canvas.selected_annotation = None
        
        # Check if we have annotations for this frame
        if self.current_frame in self.frame_annotations:
            # Set the canvas annotations to the current frame's annotations
            self.canvas.annotations = self.frame_annotations[self.current_frame]
        else:
            # No annotations for this frame yet
            self.canvas.annotations = []
        
        # Update the annotation dock
        if hasattr(self, 'annotation_dock'):
            self.annotation_dock.update_annotation_list()
        
        # Update the canvas
        self.canvas.update()
  
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
                
                # Load annotations for the new frame
                self.load_current_frame_annotations()

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
                    
                    # Load annotations for the new frame
                    self.load_current_frame_annotations()

    def next_frame(self):
        """Go to the next frame in the video."""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame += 1
                self.canvas.set_frame(frame)
                self.update_frame_info()
                
                # Load annotations for the new frame
                self.load_current_frame_annotations()
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
                    
                    # Load annotations for the new frame
                    self.load_current_frame_annotations()

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
    def add_empty_annotation(self):
        """Add a new empty annotation to the current frame."""
        if not self.cap:
            QMessageBox.warning(self, "Add Annotation", "Please open a video first!")
            return
        
        # Create a default rectangle in the center of the frame
        if self.canvas.pixmap:
            center_x = self.canvas.pixmap.width() // 2
            center_y = self.canvas.pixmap.height() // 2
            rect = QRect(center_x - 50, center_y - 50, 100, 100)
            
            # Create a new bounding box with the current class
            class_name = self.canvas.current_class
            color = self.canvas.class_colors.get(class_name, QColor(255, 0, 0))
            bbox = BoundingBox(rect, class_name, {"Size": -1, "Quality": -1}, color)
            
            # Add to annotations
            self.canvas.annotations.append(bbox)
            self.canvas.selected_annotation = bbox
            
            # Update frame_annotations dictionary
            self.frame_annotations[self.current_frame] = self.canvas.annotations
            
            # Update UI
            self.update_annotation_list()
            self.canvas.update()

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

    def delete_annotation(self, annotation):
        """Delete the specified annotation."""
        if annotation in self.canvas.annotations:
            # Remove from annotations list
            self.canvas.annotations.remove(annotation)
            
            # Clear selection if this was the selected annotation
            if self.canvas.selected_annotation == annotation:
                self.canvas.selected_annotation = None
            
            # Update canvas and mark project as modified
            self.canvas.update()
            self.project_modified = True
            
            # Update frame_annotations dictionary
            self.frame_annotations[self.current_frame] = self.canvas.annotations.copy()
            
            # Update annotation list
            self.update_annotation_list()
    
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
            
            # Update frame_annotations dictionary
            self.frame_annotations[self.current_frame] = self.canvas.annotations.copy()
            
            # Update annotation list in UI if it exists
            if hasattr(self, 'update_annotation_list'):
                self.update_annotation_list()
    #
    # Import handling methods
    #
    def update_class_ui_after_import(self):
        """Update the class-related UI components after importing annotations."""
        # Update class selector in toolbar
        if hasattr(self, 'toolbar') and hasattr(self.toolbar, 'update_class_selector'):
            self.toolbar.update_class_selector()
        
        # Update class dock if it exists
        if hasattr(self, 'class_dock') and hasattr(self.class_dock, 'update_class_list'):
            self.class_dock.update_class_list()
        
        # Set current class to first class if available
        if self.canvas.class_colors and hasattr(self.canvas, 'set_current_class'):
            first_class = next(iter(self.canvas.class_colors))
            self.canvas.set_current_class(first_class)
            
            # Update class selector if it exists
            if hasattr(self, 'class_selector') and self.class_selector.count() > 0:
                self.class_selector.setCurrentText(first_class)
    def import_annotations(self, filename=None):
        """
        Import annotations from various formats (YOLO, Pascal VOC, COCO, Raya, Raya Text).
        The format is automatically detected based on file extension and content.
        
        Args:
            filename (str, optional): Path to the annotation file. If None, a file dialog will be shown.
        """
        if not self.cap:
            QMessageBox.warning(self, "Import Annotations", "Please open a video first!")
            return
        
        # If no filename provided, show file dialog to select one
        if not filename:
            filename, _ = QFileDialog.getOpenFileName(
                self, "Import Annotations", "", 
                "All Supported Files (*.json *.xml *.txt);;JSON Files (*.json);;XML Files (*.xml);;Text Files (*.txt);;All Files (*)"
            )
            
            if not filename:
                return
        
        try:
            # Detect format based on file extension and content
            format_type = self.detect_annotation_format(filename)
            
            if not format_type:
                QMessageBox.warning(self, "Import Annotations", 
                                "Could not automatically detect the annotation format. Please ensure the file is in YOLO, Pascal VOC, COCO, or Raya format.")
                return
            
            # Import annotations based on detected format
            self.statusBar.showMessage(f"Importing {format_type} annotations...")
            
            # Get image dimensions from canvas
            image_width = self.canvas.pixmap.width() if self.canvas.pixmap else 640
            image_height = self.canvas.pixmap.height() if self.canvas.pixmap else 480
            
            # Import annotations based on format
            if format_type == "COCO":
                self.import_coco_annotations(filename, image_width, image_height)
            elif format_type == "YOLO":
                self.import_yolo_annotations(filename, image_width, image_height)
            elif format_type == "Pascal VOC":
                self.import_pascal_voc_annotations(filename, image_width, image_height)
            elif format_type == "Raya":
                self.import_raya_annotations(filename, image_width, image_height)

            
            # Update UI
            self.update_annotation_list()
            self.update_class_ui_after_import()
            self.canvas.update()
            
            self.statusBar.showMessage(f"Successfully imported {format_type} annotations from {os.path.basename(filename)}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to import annotations: {str(e)}")


    def detect_annotation_format(self, filename):
        """
        Detect the annotation format based on file extension and content.
        
        Args:
            filename (str): Path to the annotation file
            
        Returns:
            str: Detected format ("COCO", "YOLO", "Pascal VOC", "Raya") or None if not detected
        """
        # Check file extension
        ext = os.path.splitext(filename)[1].lower()
        
        # Read file content
        try:
            with open(filename, 'r') as f:
                content = f.read(1000)  # Read first 1000 chars to detect format
        except:
            return None
        
        # Detect format based on extension and content
        if ext == '.json':
            if '"images"' in content and '"annotations"' in content and '"categories"' in content:
                return "COCO"
        elif ext == '.xml':
            if '<annotation>' in content and '<object>' in content:
                return "Pascal VOC"
        elif ext == '.txt':
            # Check for Raya text format first (lines with [] or [class,x,y,w,h,size,quality])
            lines = content.strip().split('\n')
            if lines and all(line.strip() == "[]" or 
                            (line.strip().startswith('[') and 
                            line.strip().endswith(';') and 
                            ',' in line) 
                            for line in lines if line.strip()):
                return "Raya"
            
            # YOLO format typically has space-separated numbers (class x y w h)
            if lines and all(len(line.split()) == 5 and line.split()[0].isdigit() for line in lines if line.strip()):
                return "YOLO"
        
        # If no format detected, try more detailed analysis
        if ext == '.json':
            try:
                import json
                data = json.loads(content)
                if isinstance(data, dict):
                    if 'annotations' in data and 'images' in data:
                        return "COCO"
            except:
                pass
        
        return "Raya"

    def import_coco_annotations(self, filename, image_width, image_height):
        """
        Import annotations from COCO JSON format.
        
        Args:
            filename (str): Path to the COCO JSON file
            image_width (int): Width of the image/video frame
            image_height (int): Height of the image/video frame
        """
        import json
        
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Create a mapping from category ID to category name
        categories = {cat['id']: cat['name'] for cat in data.get('categories', [])}
        
        # Create a mapping from image ID to frame number
        images = {img['id']: img.get('frame_id', 0) for img in data.get('images', [])}
        
        # Process annotations
        for ann in data.get('annotations', []):
            image_id = ann.get('image_id')
            category_id = ann.get('category_id')
            bbox = ann.get('bbox', [0, 0, 0, 0])  # [x, y, width, height]
            
            # Skip if missing essential data
            if not all([image_id is not None, category_id is not None, bbox]):
                continue
            
            # Get frame number and category name
            frame_num = images.get(image_id, 0)
            class_name = categories.get(category_id, f"class_{category_id}")
            
            # Create QRect from COCO bbox [x, y, width, height]
            rect = QRect(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
            
            # Get or create color for this class
            if class_name not in self.canvas.class_colors:
                self.canvas.class_colors[class_name] = QColor(
                    random.randint(0, 255), 
                    random.randint(0, 255), 
                    random.randint(0, 255)
                )
            color = self.canvas.class_colors[class_name]
            
            # Create attributes dictionary
            attributes = {
                "Size": ann.get('size', -1),
                "Quality": ann.get('quality', -1),
                "id": str(ann.get('id', ''))
            }
            
            # Create bounding box
            bbox_obj = BoundingBox(rect, class_name, attributes, color)
            
            # Add to frame annotations
            if frame_num not in self.frame_annotations:
                self.frame_annotations[frame_num] = []
            self.frame_annotations[frame_num].append(bbox_obj)
            
            # If this is the current frame, add to canvas annotations
            if frame_num == self.current_frame:
                self.canvas.annotations.append(bbox_obj)

    def import_yolo_annotations(self, filename, image_width, image_height):
        """
        Import annotations from YOLO format.
        
        Args:
            filename (str): Path to the YOLO txt file
            image_width (int): Width of the image/video frame
            image_height (int): Height of the image/video frame
        """
        # YOLO format: class_id x_center y_center width height
        # All values are normalized [0-1]
        
        # First, try to find a classes.txt file in the same directory
        classes_file = os.path.join(os.path.dirname(filename), 'classes.txt')
        class_names = []
        
        if os.path.exists(classes_file):
            with open(classes_file, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
        
        # Read annotations
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Process each line
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            
            try:
                class_id = int(parts[0])
                x_center = float(parts[1]) * image_width
                y_center = float(parts[2]) * image_height
                width = float(parts[3]) * image_width
                height = float(parts[4]) * image_height
                
                # Calculate top-left corner from center
                x = x_center - (width / 2)
                y = y_center - (height / 2)
                
                # Create QRect
                rect = QRect(int(x), int(y), int(width), int(height))
                
                # Get class name
                if class_id < len(class_names):
                    class_name = class_names[class_id]
                else:
                    class_name = f"class_{class_id}"
                
                # Get or create color for this class
                if class_name not in self.canvas.class_colors:
                    self.canvas.class_colors[class_name] = QColor(
                        random.randint(0, 255), 
                        random.randint(0, 255), 
                        random.randint(0, 255)
                    )
                color = self.canvas.class_colors[class_name]
                
                # Create attributes dictionary
                attributes = {
                    "Size": -1,
                    "Quality": -1
                }
                
                # Create bounding box
                bbox_obj = BoundingBox(rect, class_name, attributes, color)
                
                # Add to current frame annotations
                self.canvas.annotations.append(bbox_obj)
                self.frame_annotations[self.current_frame] = self.canvas.annotations
                
            except (ValueError, IndexError) as e:
                print(f"Error parsing YOLO line: {line}. Error: {e}")

    def import_pascal_voc_annotations(self, filename, image_width, image_height):
        """
        Import annotations from Pascal VOC XML format.
        
        Args:
            filename (str): Path to the Pascal VOC XML file
            image_width (int): Width of the image/video frame
            image_height (int): Height of the image/video frame
        """
        import xml.etree.ElementTree as ET
        
        try:
            tree = ET.parse(filename)
            root = tree.getroot()
            
            # Process each object
            for obj in root.findall('./object'):
                class_name = obj.find('name').text
                
                # Get bounding box
                bndbox = obj.find('bndbox')
                xmin = int(float(bndbox.find('xmin').text))
                ymin = int(float(bndbox.find('ymin').text))
                xmax = int(float(bndbox.find('xmax').text))
                ymax = int(float(bndbox.find('ymax').text))
                
                # Create QRect
                rect = QRect(xmin, ymin, xmax - xmin, ymax - ymin)
                
                # Get or create color for this class
                if class_name not in self.canvas.class_colors:
                    self.canvas.class_colors[class_name] = QColor(
                        random.randint(0, 255), 
                        random.randint(0, 255), 
                        random.randint(0, 255)
                    )
                color = self.canvas.class_colors[class_name]
                
                # Create attributes dictionary
                attributes = {
                    "Size": -1,
                    "Quality": -1
                }
                
                # Check for additional attributes
                for attr in obj.findall('./attribute'):
                    name = attr.find('name')
                    value = attr.find('value')
                    if name is not None and value is not None:
                        attributes[name.text] = value.text
                
                # Create bounding box
                bbox_obj = BoundingBox(rect, class_name, attributes, color)
                
                # Add to current frame annotations
                self.canvas.annotations.append(bbox_obj)
                self.frame_annotations[self.current_frame] = self.canvas.annotations
                
        except Exception as e:
            raise Exception(f"Error parsing Pascal VOC XML: {str(e)}")

    def import_raya_annotations(self, filename, image_width, image_height):
        """
        Import annotations from Raya text format.
        
        Format: [class,x,y,width,height,size,quality,shadow(optional)];
        If no detection: []
        
        Args:
            filename (str): Path to the Raya text file
            image_width (int): Width of the image/video frame
            image_height (int): Height of the image/video frame
        """
        try:
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            # Process each line (each line represents a frame)
            for frame_num, line in enumerate(lines):
                line = line.strip()
                
                # Skip empty frames or frames with no detections
                if not line or line == "[]":
                    continue
                
                # Check if the line contains annotations
                if not ('[' in line and ']' in line):
                    continue
                    
                # Extract content between the outermost brackets
                start_idx = line.find('[')
                end_idx = line.rfind(']')
                if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
                    continue
                    
                content = line[start_idx+1:end_idx]
                
                # Split by semicolon for multiple annotations
                annotations = content.split(';')
                frame_annotations = []
                
                for annotation in annotations:
                    if not annotation.strip():
                        continue
                    
                    # Parse the annotation values
                    parts = annotation.split(',')
                    
                    # Ensure we have at least the minimum required fields
                    if len(parts) < 6:
                        continue
                    
                    try:
                        # Remove any remaining brackets
                        parts = [p.strip('[]') for p in parts]
                        
                        class_id = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        size = float(parts[5])
                        quality = float(parts[6]) if len(parts) > 6 else 100.0
                        shadow = float(parts[7]) if len(parts) > 7 else 0.0
                        
                        # Create class name based on class ID
                        class_name = "Drone"
                        
                        # Create QRect
                        rect = QRect(int(x), int(y), int(width), int(height))
                        
                        # Get or create color for this class
                        if class_name not in self.canvas.class_colors:
                            self.canvas.class_colors[class_name] = QColor(
                                random.randint(0, 255), 
                                random.randint(0, 255), 
                                random.randint(0, 255)
                            )
                        color = self.canvas.class_colors[class_name]
                        
                        # Create attributes dictionary
                        attributes = {
                            "Size": int(size),
                            "Quality": int(quality),
                            "Shadow": shadow if len(parts) > 7 else 0
                        }
                        
                        # Create bounding box
                        bbox_obj = BoundingBox(rect, class_name, attributes, color)
                        frame_annotations.append(bbox_obj)
                        
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing Raya annotation: {annotation}. Error: {e}")
                
                # Add to frame annotations
                if frame_annotations:
                    self.frame_annotations[frame_num] = frame_annotations
                    
                    # If this is the current frame, update canvas annotations
                    if frame_num == self.current_frame:
                        self.canvas.annotations = frame_annotations.copy()
                        
        except Exception as e:
            raise Exception(f"Error parsing Raya text file: {str(e)}")


    
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


