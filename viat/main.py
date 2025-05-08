"""
Video Annotation Tool (VAT) - Main Application

This module contains the main application window and program entry point for the
Video Annotation Tool. It provides the UI framework and coordinates between the
different components of the application.
"""

import os
import random
import cv2
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QLabel,
    QAction,
    QFileDialog,
    QStatusBar,
    QComboBox,
    QMessageBox,
    QListWidget,
    QListWidgetItem,
    QDialog,
    QFormLayout,
    QSpinBox,
    QDialogButtonBox,
    QLineEdit,
    QColorDialog,
    QActionGroup,
    QGroupBox,
    QDoubleSpinBox,
    QApplication,
    QProgressBar,
    QCheckBox,
    QTextEdit,
    QPlainTextEdit,
)
from PyQt5.QtCore import Qt, QTimer, QRect, QDateTime, QEvent
from PyQt5.QtGui import QColor, QIcon, QImage, QPixmap
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .canvas import VideoCanvas
from .annotation import BoundingBox, AnnotationManager, ClassManager
from .widgets import AnnotationDock, StyleManager, ClassDock, AnnotationToolbar
from .interpolation import InterpolationManager
from utils.dataset_manager import import_dataset_dialog, load_dataset
from utils.dataset_manager import export_dataset_dialog, export_dataset


import numpy as np
import json
from utils import (
    save_project,
    load_project,
    export_annotations,
    get_config_directory,
    get_recent_projects,
    get_last_project,
    save_last_state,
    load_last_state,
    export_image_dataset_pascal_voc,
    export_image_dataset_yolo,
    export_image_dataset_coco,
    export_standard_annotations,
    mse_similarity,
    calculate_frame_hash,
    create_thumbnail,
    import_annotations,
    UICreator,
)
from utils.icon_provider import IconProvider
from natsort import natsorted
from copy import deepcopy


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
        self.video_filename = ""
        # Set up the user interface
        self.setup_ui()
        self.canvas.smart_edge_enabled = False
        self.setup_autosave()

        self.init_managers()
        self.ui_creator.create_interpolation_ui()
        # Install event filter to handle global shortcuts
        QApplication.instance().installEventFilter(self)

        # Load last project if available
        QTimer.singleShot(100, self.load_last_project)

    def init_managers(self):
        self.annotation_manager = AnnotationManager(self, self.canvas)
        self.class_manager = ClassManager(self)
        self.interpolation_manager = InterpolationManager(self)

    def load_last_project(self):
        """Load the last project that was open."""
        # Try to load from application state first
        if self.load_application_state():
            return

        # If that fails, try to get the most recent project
        last_project = get_last_project()
        if last_project and os.path.exists(last_project):
            self.load_project(last_project)

    def init_properties(self):
        """Initialize the application properties and state variables."""
        # Available styles
        self.duplicate_frames_enabled = True
        self.duplicate_frames_cache = {}  # Maps frame hash to list of frame numbers
        self.frame_hashes = {}
        self.undo_stack = []
        self.redo_stack = []
        self.max_undo_steps = 20
        self.max_redo_steps = 20
        self.styles = {}
        self.icon_provider = IconProvider()
        self._class_refresh_scheduled = False
        self.setFocusPolicy(Qt.StrongFocus)
        for style_name in StyleManager.get_available_styles():
            method_name = f"set_{style_name.lower().replace(' ', '_')}_style"
            if hasattr(StyleManager, method_name):
                self.styles[style_name] = getattr(StyleManager, method_name)
            elif style_name.lower() == "default":
                self.styles[style_name] = StyleManager.set_darkmodern_style

        # Annotation methods
        self.annotation_methods = {
            "Rectangle": "Draw rectangular bounding boxes",
            "Polygon": "Draw polygon shapes",
            "Point": "Mark specific points",
        }
        self.current_annotation_method = "Rectangle"

        # Application state
        self.current_style = "DarkModern"
        self.playback_speed = 1.0
        self.cap = None  # Video capture object
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0
        self.zoom_level = 1.0

        # Add annotation attribute settings
        self.auto_show_attribute_dialog = (
            True  # Show attribute dialog when creating annotation
        )
        self.use_previous_attributes = True  # Use attributes from previous annotations

        # Annotation data
        self.frame_annotations = {}  # Dictionary to store annotations by frame number

        # Class attribute configurations
        self.canvas_class_attributes = {
            "Quad": {
                "Size": {"type": "int", "default": -1, "min": 0, "max": 100},
                "Quality": {"type": "int", "default": -1, "min": 0, "max": 100},
            }
        }

        # Project state
        self.project_file = None
        self.project_modified = False
        self.autosave_timer = None
        self.autosave_enabled = True
        self.autosave_interval = 5000
        self.autosave_file = None
        self.last_autosave_time = None

        # Add image dataset flag
        self.is_image_dataset = False
        self.image_files = []

    def setup_ui(self):
        """Set up the user interface."""
        # Create UI creator
        self.ui_creator = UICreator(self)

        # Create menu bar
        self.ui_creator.create_menu_bar()

        # Create toolbar
        self.ui_creator.create_toolbar()

        # Create dock widgets
        self.ui_creator.create_dock_widgets()

        # Create status bar
        self.ui_creator.create_status_bar()

        # Set up playback timer
        self.ui_creator.setup_playback_timer()

        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Create layout
        layout = QVBoxLayout(central_widget)

        # Create canvas
        self.canvas = VideoCanvas(self)
        layout.addWidget(self.canvas)

        if hasattr(self.canvas, "annotationChanged"):
            self.canvas.annotationChanged.connect(self.save_undo_state)
        if hasattr(self.canvas, "annotationMoved"):
            self.canvas.annotationMoved.connect(self.save_undo_state)
        if hasattr(self.canvas, "annotationResized"):
            self.canvas.annotationResized.connect(self.save_undo_state)
        # Add playback controls
        playback_controls = self.ui_creator.create_playback_controls()
        layout.addWidget(playback_controls)

        # Set layout margins
        layout.setContentsMargins(5, 5, 5, 5)

        # Set initial window size
        self.resize(1200, 800)

        # Set window title
        self.setWindowTitle("VIAT - Video Image Annotation Tool")

        # Set window icon
        self.setWindowIcon(self.icon_provider.get_icon("app-icon"))

    def setup_playback_timer(self):
        """Set up the timer for video playback."""
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.next_frame)

    def toggle_attribute_dialog(self):
        """Toggle automatic attribute dialog display."""
        self.auto_show_attribute_dialog = not self.auto_show_attribute_dialog
        self.statusBar.showMessage(
            f"Attribute dialog for new annotations {'enabled' if self.auto_show_attribute_dialog else 'disabled'}",
            3000,
        )

    def toggle_previous_attributes(self):
        """Toggle using previous annotation attributes as default."""
        self.use_previous_attributes = not self.use_previous_attributes
        self.statusBar.showMessage(
            f"Using previous annotation attributes as default {'enabled' if self.use_previous_attributes else 'disabled'}",
            3000,
        )

    def toggle_autosave(self):
        """Toggle auto-save functionality."""
        self.autosave_enabled = not self.autosave_enabled

        if self.autosave_enabled:
            self.autosave_timer.start(self.autosave_interval)
            self.statusBar.showMessage("Auto-save enabled", 3000)
        else:
            self.autosave_timer.stop()
            self.statusBar.showMessage("Auto-save disabled", 3000)

    def toggle_smart_edge(self):
        """Toggle smart edge movement functionality."""
        is_active = self.smart_edge_action.isChecked()
        self.canvas.smart_edge_enabled = is_active

        if is_active:
            self.statusBar.showMessage(
                "Smart Edge Movement enabled - edges will snap to image features"
            )
        else:
            self.statusBar.showMessage("Smart Edge Movement disabled")

    def change_annotation_method(self, method_name):
        """Change the current annotation method."""
        if method_name in ["Drag", "TwoClick"]:
            # Update canvas annotation method
            self.canvas.set_annotation_method(method_name)

            if method_name == "TwoClick":
                self.statusBar.showMessage(
                    "Two-click mode: Click first corner, then click second corner to create box. Press ESC to cancel."
                )
            else:
                self.statusBar.showMessage(
                    f"Annotation method changed to {method_name}"
                )

    def clear_recent_projects(self):
        """Clear the list of recent projects."""
        config_dir = get_config_directory()
        recent_projects_file = os.path.join(config_dir, "recent_projects.json")

        with open(recent_projects_file, "w") as f:
            json.dump([], f)

        self.update_recent_projects_menu()
        self.statusBar.showMessage("Recent projects cleared", 3000)

    def update_recent_projects_menu(self):
        """Update the recent projects menu with the latest projects."""
        self.recent_projects_menu.clear()

        recent_projects = get_recent_projects()
        if not recent_projects:
            no_recent = QAction("No Recent Projects", self)
            no_recent.setEnabled(False)
            self.recent_projects_menu.addAction(no_recent)
            return

        for project_path in recent_projects:
            project_name = os.path.basename(project_path)
            action = QAction(project_name, self)
            action.setData(project_path)
            action.triggered.connect(
                lambda checked, path=project_path: self.load_project(path)
            )
            self.recent_projects_menu.addAction(action)

        self.recent_projects_menu.addSeparator()
        clear_action = QAction("Clear Recent Projects", self)
        clear_action.triggered.connect(self.clear_recent_projects)
        self.recent_projects_menu.addAction(clear_action)

    def set_autosave_interval(self, interval_ms):
        """Set the auto-save interval."""
        self.autosave_interval = interval_ms

        # Restart timer if it's active
        if self.autosave_enabled and self.autosave_timer.isActive():
            self.autosave_timer.stop()
            self.autosave_timer.start(self.autosave_interval)

        # Convert to minutes for display
        minutes = interval_ms / 60000
        self.statusBar.showMessage(
            f"Auto-save interval set to {minutes} minute{'s' if minutes != 1 else ''}",
            3000,
        )

    #
    # Video handling methods
    #

    def zoom_in(self):
        """Zoom in on the canvas."""
        self.zoom_level *= 1.2
        self.canvas.set_zoom(self.zoom_level)

    def zoom_out(self):
        """Zoom out on the canvas."""
        self.zoom_level /= 1.2
        self.canvas.set_zoom(self.zoom_level)

    def reset_zoom(self):
        """Reset zoom to default level."""
        self.zoom_level = 1.0
        self.canvas.set_zoom(self.zoom_level)

    def open_video(self):
        """Open a video file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )

        if filename:
            # Reset image dataset related state
            self.image_dataset_info = None
            self.is_image_dataset = False

            # Clear existing annotations
            self.canvas.annotations = []
            self.frame_annotations = {}

            # Reset frame-related variables
            self.current_frame = 0
            self.frame_hashes = {}
            self.duplicate_frames_cache = {}
            self.load_video_file(filename)

    def load_video_file(self, filename):
        """Load a video file and display the first frame."""
        # Close any existing video
        if self.cap:
            self.cap.release()

        # Open the video file
        self.cap = cv2.VideoCapture(filename, cv2.CAP_ANY)

        if not self.cap.isOpened():
            QMessageBox.critical(self, "Error", "Could not open video file!")
            self.cap = None
            return False

        self.video_filename = filename
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

            # Set up auto-save for this video
            self.video_filename = filename
            video_base = os.path.splitext(filename)[0]
            self.autosave_file = f"{video_base}_autosave.json"

            # Start auto-save timer
            if self.autosave_enabled and not self.autosave_timer.isActive():
                self.autosave_timer.start(self.autosave_interval)

            # Check if we need to scan for duplicate frames
            if self.duplicate_frames_enabled and not self.frame_hashes:
                # Ask if user wants to scan for duplicates
                reply = QMessageBox.question(
                    self,
                    "Duplicate Frame Detection",
                    "Would you like to scan this video for duplicate frames?\n"
                    "(This will help automatically propagate annotations)",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )

                if reply == QMessageBox.Yes:
                    QTimer.singleShot(500, self.scan_video_for_duplicates)
            elif self.duplicate_frames_enabled and self.frame_hashes:
                # We already have frame hashes for this video
                duplicate_count = sum(
                    len(frames) - 1
                    for frames in self.duplicate_frames_cache.values()
                    if len(frames) > 1
                )
                self.statusBar.showMessage(
                    f"Loaded {duplicate_count} duplicate frames from project file", 5000
                )

            # Only check for annotation files if this is a direct video open, not from a project load
            if (
                not hasattr(self, "_loading_from_project")
                or not self._loading_from_project
            ):
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

        # Check for auto-save file first
        autosave_file = os.path.join(directory, f"{base_name}_autosave.json")

        # Only show auto-save prompt if we're not loading from a project
        if os.path.exists(autosave_file) and not hasattr(self, "_loading_from_project"):
            # Store that we've already shown the auto-save prompt for this video
            if not hasattr(self, "_autosave_prompted"):
                self._autosave_prompted = set()

            # Only show the prompt if we haven't already shown it for this video
            if video_filename not in self._autosave_prompted:
                self._autosave_prompted.add(video_filename)

                reply = QMessageBox.question(
                    self,
                    "Auto-Save Found",
                    "An auto-save file was found for this video.\nWould you like to load it?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )

                if reply == QMessageBox.Yes:
                    try:
                        # Set flag to prevent recursive auto-save prompts
                        self._loading_from_project = True
                        self.load_project(autosave_file)
                        self._loading_from_project = False
                        return
                    except Exception as e:
                        self._loading_from_project = False
                        QMessageBox.warning(
                            self,
                            "Auto-Save Error",
                            f"Error loading auto-save file: {str(e)}",
                        )

        # List of possible annotation file extensions to check
        extensions = [".txt", ".json", ".xml"]

        # Find matching annotation files
        annotation_files = []
        for ext in extensions:
            potential_file = os.path.join(directory, base_name + ext)
            if os.path.exists(potential_file) and potential_file != autosave_file:
                annotation_files.append(potential_file)
        # check if the json file in annotaton_files is project save not a coco
        for an in annotation_files:
            if an.endswith(".json"):
                with open(an, "r") as f:
                    data = json.load(f)
                    if "viat_project_identifier" in data:
                        annotation_files.remove(an)
        if annotation_files:
            # Create a message with the found files
            message = "Found the following annotation file(s):\n\n"
            for file in annotation_files:
                message += f"- {os.path.basename(file)}\n"
            message += "\nWould you like to import annotations from one of these files?"

            # Show dialog asking if user wants to import
            reply = QMessageBox.question(
                self,
                "Annotation Files Found",
                message,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
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
        Show a dialog for the user to select which annotation file(s) to import.

        Args:
            annotation_files (list): List of annotation file paths
        """
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Annotation Files")
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout(dialog)

        # Add explanation label
        label = QLabel(
            "Multiple annotation files found. Please select which ones to import:"
        )
        layout.addWidget(label)

        # Create list widget with annotation files
        list_widget = QListWidget()
        for file in annotation_files:
            item = QListWidgetItem(os.path.basename(file))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            list_widget.addItem(item)

        layout.addWidget(list_widget)

        # Add buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() == QDialog.Accepted:
            # Get selected files
            selected_files = []
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                if item.checkState() == Qt.Checked:
                    selected_files.append(annotation_files[i])

            # Import selected files
            if selected_files:
                self.import_multiple_annotations(selected_files)

    def import_multiple_annotations(self, annotation_files):
        """
        Import annotations from multiple files.

        Args:
            annotation_files (list): List of annotation file paths
        """
        # Create progress dialog
        progress = QDialog(self)
        progress.setWindowTitle("Importing Annotations")
        progress.setFixedSize(400, 100)
        progress_layout = QVBoxLayout(progress)

        status_label = QLabel("Importing annotations...")
        progress_layout.addWidget(status_label)

        progress_bar = QProgressBar()
        progress_bar.setRange(0, len(annotation_files))
        progress_layout.addWidget(progress_bar)

        # Non-blocking progress dialog
        progress.setModal(False)
        progress.show()
        QApplication.processEvents()

        # Import each file
        for i, file_path in enumerate(annotation_files):
            status_label.setText(f"Importing {os.path.basename(file_path)}...")
            progress_bar.setValue(i)
            QApplication.processEvents()

            try:
                self.import_annotations(file_path)
            except Exception as e:
                print(f"Error importing {file_path}: {str(e)}")

        # Close progress dialog
        progress.close()

        # Show success message
        QMessageBox.information(
            self,
            "Import Complete",
            f"Successfully imported annotations from {len(annotation_files)} files.",
        )

    def update_frame_info(self):
        """Update frame information in the UI."""
        if hasattr(self, "is_image_dataset") and self.is_image_dataset:
            # Update frame label for image datasets
            total = len(self.image_files) if self.image_files else 0
            self.frame_label.setText(f"{self.current_frame + 1}/{total}")

            # Update slider position (without triggering valueChanged)
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(self.current_frame)
            self.frame_slider.blockSignals(False)

            # Show current image filename in status bar
            if 0 <= self.current_frame < len(self.image_files):
                self.statusBar.showMessage(
                    f"Image: {os.path.basename(self.image_files[self.current_frame])}"
                )
        elif self.cap and self.cap.isOpened():
            # Update frame label for videos
            self.frame_label.setText(f"{self.current_frame}/{self.total_frames}")

            # Update slider position (without triggering valueChanged)
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(self.current_frame)
            self.frame_slider.blockSignals(False)

    def update_frame_annotations(self):
        """Update annotations for the current frame."""
        # Save current annotations to frame_annotations dictionary
        if hasattr(self.canvas, "annotations") and self.canvas.annotations:
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
        if hasattr(self, "annotation_dock"):
            self.annotation_dock.update_annotation_list()

        # Update the canvas
        self.canvas.update()

    def slider_changed(self, value):
        """Handle slider value changes."""
        # self.handle_unverified_annotations()
        if hasattr(self, "is_image_dataset") and self.is_image_dataset:
            if 0 <= value < len(self.image_files):
                # Only update if the value actually changed
                if value != self.current_frame:
                    self.current_frame = value
                    self.load_current_image()
                    self.update_frame_info()
                    # Load annotations for the new frame
                    self.load_current_frame_annotations()
        elif self.cap and self.cap.isOpened():
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
        """Go to the previous frame in the video or previous image in the dataset."""
        self.handle_unverified_annotations()
        if hasattr(self, "is_image_dataset") and self.is_image_dataset:
            if self.current_frame > 0:
                self.current_frame -= 1
                self.frame_slider.setValue(self.current_frame)
                self.load_current_image()
                self.update_frame_info()
        else:
            if (
                hasattr(self, "interpolation_manager")
                and self.interpolation_manager.is_active
            ):
                current_frame = self.current_frame

                # Check if current frame has annotations
                current_has_annotations = (
                    current_frame in self.frame_annotations
                    and len(self.frame_annotations[current_frame]) > 0
                )

                if current_has_annotations:
                    # If current frame has annotations, go to previous keyframe (current - interval)
                    prev_frame = max(
                        0, current_frame - self.interpolation_manager.interval
                    )
                else:
                    # If current frame has no annotations, find the previous frame with annotations
                    prev_frame = None
                    for frame in sorted(self.frame_annotations.keys(), reverse=True):
                        if (
                            frame < current_frame
                            and len(self.frame_annotations[frame]) > 0
                        ):
                            prev_frame = frame
                            break

                    # If no previous annotated frame found, just go to previous frame
                    if prev_frame is None:
                        prev_frame = max(0, current_frame - 1)
            else:
                prev_frame = max(0,self.current_frame-1)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, prev_frame)
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = prev_frame
                self.frame_slider.setValue(self.current_frame)
                self.canvas.set_frame(frame)
                self.update_frame_info()
                # Load annotations for the new frame
                self.load_current_frame_annotations()

    def next_frame(self):
        """Go to the next frame in the video or next image in the dataset."""
        self.handle_unverified_annotations()
        if hasattr(self, "is_image_dataset") and self.is_image_dataset:
            if self.current_frame < len(self.image_files) - 1:
                self.current_frame += 1
                # Update slider position first
                self.frame_slider.setValue(self.current_frame)
                self.load_current_image()
                self.update_frame_info()
                self.load_current_frame_annotations()
            else:
                if self.is_playing:
                    self.current_frame = 0
                    # Update slider position first
                    self.frame_slider.setValue(self.current_frame)
                    self.load_current_image()
                    self.update_frame_info()
                    self.load_current_frame_annotations()
                    self.statusBar.showMessage("Looping back to start of image dataset")
                else:
                    # Just show message if not playing
                    self.statusBar.showMessage("End of image dataset")
        else:
            if hasattr(self, 'interpolation_manager') and self.interpolation_manager.is_active:
                current_frame = self.current_frame
                
                # Check if current frame has annotations
                current_has_annotations = (
                    current_frame in self.frame_annotations and 
                    len(self.frame_annotations[current_frame]) > 0
                )
                
                if current_has_annotations:
                    self.perform_interpolation()
                    # If current frame has annotations, go to next keyframe (current + interval)
                    next_frame_number = current_frame + self.interpolation_manager.interval
                else:
                    # If current frame has no annotations, find the next frame with annotations
                    next_frame_number = None
                    for frame in sorted(self.frame_annotations.keys()):
                        if frame > current_frame and len(self.frame_annotations[frame]) > 0:
                            next_frame_number = frame
                            break
                    
                    # If no next annotated frame found, suggest one based on interval
                    if next_frame_number is None:
                        next_frame_number = current_frame + 1
            
            elif self.cap and self.cap.isOpened():
                # Calculate the next frame number explicitly
                next_frame_number = self.current_frame + 1
                
                # Check if we've reached the end of the video
                if next_frame_number >= self.total_frames:
                    # End of video
                    self.play_timer.stop()
                    self.is_playing = False
                    self.statusBar.showMessage("End of video")
                    # Rewind to beginning
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame = 0
                    # Update slider position first
                    self.frame_slider.setValue(self.current_frame)
                    ret, frame = self.cap.read()
                    if ret:
                        self.canvas.set_frame(frame)
                        self.update_frame_info()
                        self.load_current_frame_annotations()
                    return
        
            # Explicitly set the frame position instead of just reading the next frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame_number)
            ret, frame = self.cap.read()
            
            if ret:
                self.current_frame = next_frame_number
                # Update slider position first
                self.frame_slider.setValue(self.current_frame)
                self.canvas.set_frame(frame)
                self.update_frame_info()

                # Check for duplicate frames and propagate annotations if enabled
                if (
                    self.duplicate_frames_enabled
                    and self.current_frame in self.frame_hashes
                ):
                    current_hash = self.frame_hashes[self.current_frame]
                    # If this is a duplicate frame, check if any other frame with this hash has annotations
                    if (
                        current_hash in self.duplicate_frames_cache
                        and len(self.duplicate_frames_cache[current_hash]) > 1
                    ):
                        self.propagate_annotations_to_duplicate(current_hash)


    def play_pause_video(self):
        """Toggle between playing and pausing the video or image slideshow."""
        if hasattr(self, "is_image_dataset") and self.is_image_dataset:
            if self.is_playing:
                # Stop the slideshow
                self.play_timer.stop()
                self.is_playing = False
                self.play_button.setIcon(
                    self.icon_provider.get_icon("media-playback-start")
                )
                self.statusBar.showMessage("Slideshow paused")
            else:
                # Start the slideshow with a fixed interval (1 second per image)
                self.play_timer.start(1000)  # 1000ms = 1 second
                self.is_playing = True
                self.play_button.setIcon(
                    self.icon_provider.get_icon("media-playback-pause")
                )
                self.statusBar.showMessage("Slideshow playing")
            return

        if not self.cap:
            return

        if self.is_playing:
            self.play_timer.stop()
            self.is_playing = False
            self.play_button.setIcon(
                self.icon_provider.get_icon("media-playback-start")
            )
            self.statusBar.showMessage("Paused")
        else:
            # Set timer interval based on playback speed
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:  # Protect against invalid FPS
                fps = 30  # Use a default value
            interval = max(1, int(1000 / (fps * self.playback_speed)))
            self.play_timer.start(interval)
            self.is_playing = True
            self.play_button.setIcon(
                self.icon_provider.get_icon("media-playback-pause")
            )
            self.statusBar.showMessage(f"Playing at {fps:.1f} FPS")

    def eventFilter(self, obj, event):
        """Global event filter to handle shortcuts regardless of focus."""
        if event.type() == QEvent.KeyPress:
            # Handle arrow keys for frame navigation
            # Check if control is not pressed
            if not (QApplication.keyboardModifiers() & Qt.ControlModifier):
                if event.key() == Qt.Key_Right:
                    # Right or Down arrow - go to next frame
                    self.next_frame()
                    return True
                elif event.key() == Qt.Key_Left:
                    # Left or Up arrow - go to previous frame
                    self.prev_frame()
                    return True
            # Handle space key for play/pause
            elif event.key() == Qt.Key_Space:
                # Space - toggle play/pause
                self.play_pause_video()
                return True
                # Cycle through annotations with Tab
            elif event.key() == Qt.Key_Tab:
                # Check if the focused widget is an input field
                focused_widget = QApplication.focusWidget()
                # Only use Tab for cycling if we're not in an input field
                if isinstance(focused_widget, VideoCanvas):
                    self.cycle_annotation_selection()
                    event.accept()
                    return True
        # Let other events pass through
        return super().eventFilter(obj, event)

    #
    # Project handling methods
    #
    def save_project(self, save_as=False):
        """Save the current project."""
        if not save_as and self.project_file:
            filename = self.project_file
        else:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Project", "", "JSON Files (*.json);;All Files (*)"
            )

        if filename:
            # Get video path if available or image dataset info
            video_path = None
            image_dataset_info = None

            if hasattr(self, "is_image_dataset") and self.is_image_dataset:
                # For image datasets, store the folder and relative paths
                if self.image_files:
                    base_folder = os.path.dirname(self.image_files[0])
                    image_dataset_info = {
                        "is_image_dataset": True,
                        "base_folder": base_folder,
                        "image_files": [
                            os.path.relpath(f, base_folder) for f in self.image_files
                        ],
                    }
            else:
                # For videos, store the video path
                video_path = getattr(self, "video_filename", None)

            # Get class attributes if available
            class_attributes = getattr(self.canvas, "class_attributes", {})

            # Save project with additional data
            save_project(
                filename,
                self.canvas.annotations,
                self.canvas.class_colors,
                video_path=video_path,
                current_frame=self.current_frame,
                frame_annotations=self.frame_annotations,
                class_attributes=class_attributes,
                current_style=self.current_style,
                auto_show_attribute_dialog=self.auto_show_attribute_dialog,
                use_previous_attributes=self.use_previous_attributes,
                duplicate_frames_enabled=self.duplicate_frames_enabled,
                frame_hashes=self.frame_hashes,
                duplicate_frames_cache=self.duplicate_frames_cache,
                image_dataset_info=image_dataset_info,
            )

            self.project_file = filename
            self.project_modified = False
            self.statusBar.showMessage(f"Project saved to {os.path.basename(filename)}")

            # Update recent projects menu
            self.update_recent_projects_menu()

            # Save application state
            self.save_application_state()

            # Get video path if available
            video_path = getattr(self, "video_filename", None)

    def load_project(self, filename=None):
        """Load a saved project."""
        if not filename:
            filename, _ = QFileDialog.getOpenFileName(
                self, "Load Project", "", "JSON Files (*.json);;All Files (*)"
            )

        if filename and os.path.exists(filename):
            try:
                self._loading_from_project = True
                # Load project with additional data
                (
                    annotations,
                    class_colors,
                    video_path,
                    current_frame,
                    frame_annotations,
                    class_attributes,
                    current_style,
                    auto_show_attribute_dialog,
                    use_previous_attributes,
                    duplicate_frames_enabled,
                    frame_hashes,
                    duplicate_frames_cache,
                    image_dataset_info,
                ) = load_project(filename, BoundingBox)

                # Update canvas
                self.reset_media_state()
                self.canvas.annotations = annotations
                self.canvas.class_colors = class_colors

                # Update class attributes if available
                if class_attributes:
                    self.canvas.class_attributes = class_attributes

                # Apply the saved style if available
                if current_style and current_style in self.styles:
                    self.current_style = current_style
                    self.styles[current_style]()

                    # Update style menu to show the correct checked item
                    for action in self.style_menu.actions():
                        if action.text() == current_style:
                            action.setChecked(True)
                        else:
                            action.setChecked(False)

                # Store frame annotations
                self.frame_annotations = frame_annotations

                # Update annotation settings
                self.auto_show_attribute_dialog = auto_show_attribute_dialog
                self.use_previous_attributes = use_previous_attributes

                # Update duplicate frame detection settings
                self.duplicate_frames_enabled = duplicate_frames_enabled
                if hasattr(self, "duplicate_frames_action"):
                    self.duplicate_frames_action.setChecked(duplicate_frames_enabled)

                # Load frame hashes and duplicate frames cache
                self.frame_hashes = frame_hashes if frame_hashes else {}
                self.duplicate_frames_cache = (
                    duplicate_frames_cache if duplicate_frames_cache else {}
                )

                # Handle image dataset if present
                if image_dataset_info and image_dataset_info.get(
                    "is_image_dataset", False
                ):
                    self.load_image_dataset_from_project(
                        image_dataset_info, current_frame
                    )
                # Otherwise load video if path is available
                elif video_path and os.path.exists(video_path):
                    self.load_video_from_project(video_path, current_frame)

                # Update UI
                self.update_settings_menu_actions()

                # Update class dock
                if hasattr(self, "class_dock"):
                    self.class_dock.update_class_list()

                # Update annotation dock
                if hasattr(self, "annotation_dock"):
                    self.annotation_dock.update_class_selector()

                # Update toolbar class selector
                if hasattr(self, "toolbar") and hasattr(
                    self.toolbar, "update_class_selector"
                ):
                    self.toolbar.update_class_selector()

                # Update annotation list
                self.update_annotation_list()

                self.project_file = filename
                self.project_modified = False
                self.canvas.update()

                # Update recent projects menu
                self.update_recent_projects_menu()

                # Save application state
                self.save_application_state()

                self.statusBar.showMessage(
                    f"Project loaded from {os.path.basename(filename)}"
                )

                self._loading_from_project = False

            except Exception as e:
                self._loading_from_project = False
                QMessageBox.critical(self, "Error", f"Failed to load project: {str(e)}")

    def save_application_state(self):
        """Save the current application state."""
        if not hasattr(self, "project_file") or not self.project_file:
            return

        state = {
            "last_project": self.project_file,
            "current_style": self.current_style,
            "autosave_enabled": self.autosave_enabled,
            "autosave_interval": self.autosave_interval,
        }

        save_last_state(state)

    def load_application_state(self):
        """Load the last application state."""
        state = load_last_state()
        if not state:
            return False

        # Load last project if it exists
        last_project = state.get("last_project")
        if last_project and os.path.exists(last_project):
            self.load_project(last_project)
            return True

        return False

    def closeEvent(self, event):
        """Handle application close event."""
        if self.project_modified:
            reply = QMessageBox.question(
                self,
                "Save Project",
                "The project has been modified. Do you want to save changes?",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save,
            )

            if reply == QMessageBox.Save:
                self.save_project()
                event.accept()
            elif reply == QMessageBox.Cancel:
                event.ignore()
            else:
                event.accept()
        else:
            event.accept()

        # Save application state
        self.save_application_state()

        # Perform final auto-save if enabled
        if self.autosave_enabled:
            if (
                (hasattr(self, "project_file") and self.project_file)
                or (hasattr(self, "is_image_dataset") and self.is_image_dataset)
                or (hasattr(self, "video_filename") and self.video_filename)
            ):
                self.perform_autosave()

            self.perform_autosave()

    #
    # Annotation handling methods
    #
    def sync_annotation_selection(self, annotation):
        """
        Synchronize annotation selection between canvas and dock.

        Args:
            annotation: The annotation to select
        """
        # Update canvas selection
        if hasattr(self, "canvas"):
            old_block_state = self.canvas.blockSignals(True)
            self.canvas.selected_annotation = annotation
            self.canvas.update()
            self.canvas.blockSignals(old_block_state)

        # Update dock selection
        if hasattr(self, "annotation_dock"):
            self.annotation_dock.select_annotation_in_list(annotation)

    def export_annotations(self):
        """Export annotations to various formats."""
        # Check if we have any annotations either in the current frame or across all frames
        has_annotations = bool(self.canvas.annotations) or any(
            self.frame_annotations.values()
        )

        if not has_annotations:
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

        # Add appropriate formats based on dataset type
        if hasattr(self, "is_image_dataset") and self.is_image_dataset:
            format_combo.addItems(
                [
                    "COCO JSON",
                    "YOLO TXT",
                    "Pascal VOC XML",
                    "Raya TXT",
                ]
            )
        else:
            format_combo.addItems(
                [
                    "Raya TXT",
                    "COCO JSON",
                    "YOLO TXT",
                    "Pascal VOC XML",
                ]
            )

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
        # Determine default directory and filename
        default_dir = ""
        default_filename = ""

        # If we have a video file loaded, use its directory and name
        if (
            hasattr(self, "is_image_dataset")
            and self.is_image_dataset
            and self.image_files
        ):
            # For image datasets, use the folder name
            image_folder = os.path.dirname(self.image_files[0])
            folder_name = os.path.basename(image_folder)
            default_dir = image_folder
            default_filename = folder_name + "_annotations"
        elif hasattr(self, "video_filename") and self.video_filename:
            # For videos, use the video filename
            default_dir = os.path.dirname(self.video_filename)
            default_filename = os.path.splitext(os.path.basename(self.video_filename))[
                0
            ]

        # Get export filename based on format
        if format_type == "COCO JSON":
            default_path = (
                os.path.join(default_dir, default_filename + ".json")
                if default_filename
                else ""
            )
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Export Annotations",
                default_path,
                "JSON Files (*.json);;All Files (*)",
            )
            export_format = "coco"
        elif format_type == "YOLO TXT":
            # For YOLO, we need a directory, not a file
            if hasattr(self, "is_image_dataset") and self.is_image_dataset:
                default_path = os.path.join(default_dir, default_filename + "_yolo")
                export_dir = QFileDialog.getExistingDirectory(
                    self,
                    "Select Directory for YOLO Export",
                    default_path,
                    QFileDialog.ShowDirsOnly,
                )
                if export_dir:
                    export_image_dataset_yolo(
                        export_dir,
                        self.image_files,
                        self.frame_annotations,
                        self.canvas.class_colors,
                    )
                    self.statusBar.showMessage(
                        f"Annotations exported to YOLO format in {os.path.basename(export_dir)}"
                    )
                return
            else:
                default_path = (
                    os.path.join(default_dir, default_filename + ".txt")
                    if default_filename
                    else ""
                )
                filename, _ = QFileDialog.getSaveFileName(
                    self,
                    "Export Annotations",
                    default_path,
                    "Text Files (*.txt);;All Files (*)",
                )
                export_format = "yolo"
        elif format_type == "Pascal VOC XML":
            if hasattr(self, "is_image_dataset") and self.is_image_dataset:
                default_path = os.path.join(default_dir, default_filename + "_voc")
                export_dir = QFileDialog.getExistingDirectory(
                    self,
                    "Select Directory for Pascal VOC Export",
                    default_path,
                    QFileDialog.ShowDirsOnly,
                )
                if export_dir:
                    export_image_dataset_pascal_voc(
                        export_dir,
                        self.image_files,
                        self.frame_annotations,
                        self.canvas.pixmap,
                    )
                    self.statusBar.showMessage(
                        f"Annotations exported to Pascal VOC format in {os.path.basename(export_dir)}"
                    )
                return
            else:
                default_path = (
                    os.path.join(default_dir, default_filename + ".xml")
                    if default_filename
                    else ""
                )
                filename, _ = QFileDialog.getSaveFileName(
                    self,
                    "Export Annotations",
                    default_path,
                    "XML Files (*.xml);;All Files (*)",
                )
                export_format = "pascal_voc"
        elif format_type == "Raya TXT":
            default_path = (
                os.path.join(default_dir, default_filename + ".txt")
                if default_filename
                else ""
            )
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Export Annotations",
                default_path,
                "Text Files (*.txt);;All Files (*)",
            )
            export_format = "raya"
        else:
            return

        if filename:
            try:
                # Get image dimensions from canvas
                image_width = self.canvas.pixmap.width() if self.canvas.pixmap else 640
                image_height = (
                    self.canvas.pixmap.height() if self.canvas.pixmap else 480
                )

                # For image datasets, we need to handle the export differently for some formats
                if (
                    hasattr(self, "is_image_dataset")
                    and self.is_image_dataset
                    and export_format == "coco"
                ):
                    export_image_dataset_coco(
                        filename,
                        self.image_files,
                        self.frame_annotations,
                        self.canvas.class_colors,
                        image_width,
                        image_height,
                    )
                else:

                    export_standard_annotations(
                        filename,
                        self.frame_annotations,
                        self.canvas.annotations,
                        export_format,
                        image_width,
                        image_height,
                    )

                self.statusBar.showMessage(
                    f"Annotations exported to {os.path.basename(filename)}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Error", f"Failed to export annotations: {str(e)}"
                )
                import traceback

                traceback.print_exc()

    def edit_annotation(self, annotation, focus_first_field=False):
        """
        Edit the properties of an annotation.

        Args:
            annotation: The annotation to edit
            focus_first_field: Whether to focus on the first attribute field
        """
        self.save_undo_state()
        self.annotation_manager.edit_annotation(annotation, focus_first_field)

    def delete_annotation(self, annotation):
        """Delete the specified annotation."""
        if self.annotation_manager.delete_annotation(annotation):
            self.save_undo_state()
            # Mark project as modified
            self.project_modified = True

            # Update annotation list
            self.update_annotation_list()

    def delete_selected_annotations(self):
        """Delete all currently selected annotations."""
        if (
            not hasattr(self.canvas, "selected_annotations")
            or not self.canvas.selected_annotations
        ):
            # If no multi-selection, fall back to single selection delete
            self.delete_selected_annotation()
            return

        self.save_undo_state()

        # Get a copy of the selected annotations to avoid modification during iteration
        annotations_to_delete = self.canvas.selected_annotations.copy()

        # Remove all selected annotations
        for annotation in annotations_to_delete:
            if annotation in self.canvas.annotations:
                self.canvas.annotations.remove(annotation)

        # Clear selection
        self.canvas.selected_annotation = None
        self.canvas.selected_annotations = []

        # Update canvas and mark project as modified
        self.canvas.update()
        self.project_modified = True

        # Update frame_annotations dictionary
        self.frame_annotations[self.current_frame] = self.canvas.annotations.copy()

        # Update annotation list in UI
        self.update_annotation_list()

        # Show status message
        count = len(annotations_to_delete)
        self.statusBar.showMessage(f"Deleted {count} annotations", 3000)

    def delete_selected_annotation(self):
        """Delete the currently selected annotation."""
        if (
            hasattr(self.canvas, "selected_annotation")
            and self.canvas.selected_annotation
        ):
            self.save_undo_state()
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
            if hasattr(self, "update_annotation_list"):
                self.update_annotation_list()

    def add_empty_annotation(self):
        """Add a new empty annotation with default values."""
        self.save_undo_state()
        self.annotation_manager.add_empty_annotation()

    def update_annotation_list(self):
        """Update the annotation list in the UI and handle interpolation."""
        # Update the annotation dock
        if hasattr(self, "annotation_dock"):
            self.annotation_dock.update_annotation_list()

        # Save current annotations to frame_annotations
        self.frame_annotations[self.current_frame] = self.canvas.annotations.copy()

        # Check if we're in interpolation mode and should trigger the workflow
        if (
            hasattr(self, "interpolation_manager")
            and self.interpolation_manager.is_active
            and len(self.canvas.annotations) > 0
        ):

            # Notify the interpolation manager that this frame has been annotated
            self.interpolation_manager.on_frame_annotated(self.current_frame)

        # Handle duplicate frames if enabled
        if self.duplicate_frames_enabled and self.current_frame in self.frame_hashes:
            current_hash = self.frame_hashes[self.current_frame]
            if (
                current_hash in self.duplicate_frames_cache
                and len(self.duplicate_frames_cache[current_hash]) > 1
            ):
                # Propagate current frame annotations to all duplicates
                self.propagate_to_duplicate_frames(current_hash)

        # Perform autosave if enabled
        self.perform_autosave()

    def update_annotation_attributes(self, annotation, class_attributes):
        """
        Update annotation attributes based on class configuration.

        Args:
            annotation: The annotation to update
            class_attributes: The class attribute configuration
        """
        self.annotation_manager.update_annotation_attributes(
            annotation, class_attributes
        )

    def clear_annotations(self):
        """Clear all annotations."""
        self.save_undo_state()
        self.annotation_manager.clear_annotations()

    def add_annotation(self):
        """Add annotation manually."""
        self.save_undo_state()
        self.annotation_manager.add_annotation()

    def create_annotation_dialog(self):
        """Create a dialog for adding or editing annotations."""
        return self.annotation_manager.create_annotation_dialog()

    def parse_attributes(self, text):
        """Parse attributes from text input."""
        return self.annotation_manager.parse_attributes(text)

    #
    # Import handling methods
    #
    def update_class_ui_after_import(self):
        """Update the class-related UI components after importing annotations."""
        # Update class selector in toolbar
        if hasattr(self, "toolbar") and hasattr(self.toolbar, "update_class_selector"):
            self.toolbar.update_class_selector()

        # Update class dock if it exists
        if hasattr(self, "class_dock") and hasattr(
            self.class_dock, "update_class_list"
        ):
            self.class_dock.update_class_list()

        # Set current class to first class if available
        if self.canvas.class_colors and hasattr(self.canvas, "set_current_class"):
            first_class = next(iter(self.canvas.class_colors))
            self.canvas.set_current_class(first_class)

            # Update class selector if it exists
            if hasattr(self, "class_selector") and self.class_selector.count() > 0:
                self.class_selector.setCurrentText(first_class)

    def import_annotations(self, filename=None):
        """
        Import annotations from a file.

        Args:
            filename (str, optional): Path to the annotation file. If None, a file dialog will be shown.
        """
        # If no filename is provided, show a file dialog
        if filename is None:
            filename, _ = QFileDialog.getOpenFileName(
                self,
                "Import Annotations",
                "",
                "All Files (*);;JSON Files (*.json);;Text Files (*.txt);;XML Files (*.xml)",
            )

            # If user cancels the dialog, return
            if not filename:
                return
        self.save_undo_state()

        # Check if it's a VIAT project file
        try:
            with open(filename, "r") as f:
                data = json.load(f)
                if "viat_project_identifier" in data:
                    # This is a VIAT project file, not an annotation file
                    QMessageBox.information(
                        self,
                        "Project File Detected",
                        f"{os.path.basename(filename)} is a VIAT project file, not an annotation file. "
                        "Please use 'Open Project' to load this file.",
                    )
                    return
        except:
            pass

        # Get current frame dimensions
        if self.canvas.pixmap:
            image_width = self.canvas.pixmap.width()
            image_height = self.canvas.pixmap.height()
        else:
            image_width = 640
            image_height = 480

        try:
            # Import annotations
            from utils.file_operations import (
                import_annotations as import_annotations_func,
            )

            format_type, annotations, imported_frame_annotations = (
                import_annotations_func(
                    filename,
                    BoundingBox,
                    image_width,
                    image_height,
                    self.canvas.class_colors,
                )
            )

            # Update frame annotations
            for frame_num, anns in imported_frame_annotations.items():
                if frame_num not in self.frame_annotations:
                    self.frame_annotations[frame_num] = []
                self.frame_annotations[frame_num].extend(anns)

            # Update canvas annotations if we're on a frame that has imported annotations
            if self.current_frame in imported_frame_annotations:
                self.canvas.annotations.extend(
                    imported_frame_annotations[self.current_frame]
                )
                self.canvas.update()

            # Update annotation dock
            self.annotation_dock.update_annotation_list()

            # Show success message
            QMessageBox.information(
                self,
                "Import Successful",
                f"Successfully imported annotations from {os.path.basename(filename)} "
                f"({format_type} format).",
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Import Error", f"Error importing annotations: {str(e)}"
            )

    #
    # Input handling methods
    #

    def keyPressEvent(self, event):
        """Handle keyboard events."""
        # Handle arrow keys for frame navigation
        if event.key() == Qt.Key_Right:
            # Right or Down arrow - go to next frame
            self.next_frame()
            return
        elif event.key() == Qt.Key_Left:
            # Left or Up arrow - go to previous frame
            self.prev_frame()
            return
        elif event.key() == Qt.Key_Space:
            # Space - toggle play/pause
            self.play_pause_video()
            return True
        elif event.key() == Qt.Key_Delete:
            if (
                hasattr(self.canvas, "selected_annotations")
                and self.canvas.selected_annotations
            ):
                self.delete_selected_annotations()
            elif (
                hasattr(self.canvas, "selected_annotation")
                and self.canvas.selected_annotation
            ):
                self.delete_selected_annotation()
        # Toggle annotation method with 'M' key
        elif event.key() == Qt.Key_M:
            current_index = self.method_selector.currentIndex()
            new_index = (current_index + 1) % self.method_selector.count()
            self.method_selector.setCurrentIndex(new_index)
        # Batch edit annotations with 'B' key
        elif event.key() == Qt.Key_B:
            if hasattr(self, "annotation_dock"):
                self.annotation_dock.batch_edit_annotations()

        # Propagate annotations with 'P' key
        elif event.key() == Qt.Key_P and (event.modifiers() & Qt.ControlModifier):
            self.propagate_annotations()
        # Copy selected annotation with Ctrl+C
        elif event.key() == Qt.Key_C and (event.modifiers() & Qt.ControlModifier):
            self.copy_selected_annotation()
        # Paste annotation with Ctrl+V
        elif event.key() == Qt.Key_V and (event.modifiers() & Qt.ControlModifier):
            self.paste_annotation()
        # Cut selected annotation with Ctrl+X
        elif event.key() == Qt.Key_X and (event.modifiers() & Qt.ControlModifier):
            self.cut_selected_annotation()
        # Select all annotations with Ctrl+A
        elif event.key() == Qt.Key_A and (event.modifiers() & Qt.ControlModifier):
            self.select_all_annotations()
        # Undo with Ctrl+Z
        elif (
            event.key() == Qt.Key_Z
            and (event.modifiers() & Qt.ControlModifier)
            and not (event.modifiers() & Qt.ShiftModifier)
        ):
            self.undo()
        # Redo with Ctrl+Y or Ctrl+Shift+Z
        elif (event.key() == Qt.Key_Y and (event.modifiers() & Qt.ControlModifier)) or (
            event.key() == Qt.Key_Z
            and (event.modifiers() & Qt.ControlModifier)
            and (event.modifiers() & Qt.ShiftModifier)
        ):
            self.redo()
        else:
            super().keyPressEvent(event)

    #
    # Class handling methods
    #

    def add_class(self):
        self.save_undo_state()
        self.class_manager.add_class()

    def refresh_class_lists(self):
        """Refresh class lists in all docks with debouncing"""
        # If a refresh is already scheduled, don't schedule another one
        if self._class_refresh_scheduled:
            return

        # Set the flag to indicate a refresh is scheduled
        self._class_refresh_scheduled = True

        # Define the actual refresh function
        def do_refresh():
            if hasattr(self, "class_dock"):
                self.class_dock.update_class_list()
            if hasattr(self, "annotation_dock"):
                self.annotation_dock.update_class_selector()
            if hasattr(self, "toolbar") and hasattr(
                self.toolbar, "update_class_selector"
            ):
                self.toolbar.update_class_selector()
            # Reset the flag
            self._class_refresh_scheduled = False

        # Schedule the refresh after a short delay
        QTimer.singleShot(100, do_refresh)

    def edit_selected_class(self):
        """Edit the selected class with option to convert to another class."""
        self.save_undo_state()
        self.class_manager.edit_selected_class()

    def convert_class_with_attributes(self, old_class, new_class, keep_original=False):
        """
        Convert all annotations from one class to another with attribute handling.

        Args:
            old_class (str): The original class name
            new_class (str): The target class name
            keep_original (bool): Whether to keep original attributes or use target class defaults
        """
        self.save_undo_state()
        self.class_manager.convert_class_with_attributes(
            old_class, new_class, keep_original
        )

    def convert_class_with_attribute_mapping(self, old_class, new_class):
        """
        Convert class with custom attribute mapping.

        Args:
            old_class (str): The original class name
            new_class (str): The target class name
        """
        self.save_undo_state()
        self.class_manager.convert_class_with_attribute_mapping(old_class, new_class)

    def refresh_class_ui(self):
        """Refresh all UI components that display class information"""
        # Update class dock
        if hasattr(self, "class_dock"):
            self.class_dock.update_class_list()

        # Update annotation dock
        if hasattr(self, "annotation_dock"):
            self.annotation_dock.update_class_selector()

        # Update toolbar class selector
        if hasattr(self, "class_selector"):
            self.class_selector.blockSignals(True)
            current_text = self.class_selector.currentText()
            self.class_selector.clear()
            self.class_selector.addItems(sorted(self.canvas.class_colors.keys()))
            if current_text in self.canvas.class_colors:
                self.class_selector.setCurrentText(current_text)
            elif self.canvas.current_class in self.canvas.class_colors:
                self.class_selector.setCurrentText(self.canvas.current_class)
            self.class_selector.blockSignals(False)

        # Update canvas
        self.canvas.update()

    def convert_class(self, old_class, new_class):
        """Convert all annotations from one class to another."""
        self.save_undo_state()
        self.class_manager.convert_class(old_class, new_class)

    def update_class(self, old_name, new_name, color):
        """Update a class with new name and color."""
        self.save_undo_state()
        self.class_manager.update_class(old_name, new_name, color)

    def delete_selected_class(self):
        """Delete the selected class."""
        self.save_undo_state()
        self.class_manager.delete_selected_class()

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
            self,
            "Auto Label",
            "Auto-labeling functionality is not implemented in this demo.\n\n"
            "In a real implementation, this would use a pre-trained model (like YOLO, SSD, or Faster R-CNN) "
            "to automatically detect and label objects in the current frame.",
        )

    def track_objects(self):
        """Track objects across frames."""
        if not self.canvas.pixmap or not self.canvas.annotations:
            QMessageBox.warning(
                self,
                "Track Objects",
                "Please open a video and create annotations first!",
            )
            return

            # This is a placeholder for actual object tracking functionality
        QMessageBox.information(
            self,
            "Track Objects",
            "Object tracking functionality is not implemented in this demo.\n\n"
            "In a real implementation, this would use tracking algorithms (like KCF, CSRT, or DeepSORT) "
            "to track the annotated objects across video frames.",
        )

    #
    # UI utility methods
    #

    def change_style(self, style_name):
        """Change the application style."""
        if style_name in self.styles:
            self.styles[style_name]()
            self.current_style = style_name

            # Update canvas background based on style
            if style_name == "Dark":
                self.icon_provider.set_theme("dark")
                self.refresh_icons()

                self.canvas.setStyleSheet(
                    "background-color: #151515;"
                )  # Darker background
            elif style_name == "Light":
                self.icon_provider.set_theme("light")
                self.refresh_icons()
                self.canvas.setStyleSheet(
                    "background-color: #FFFFFF;"
                )  # White background
            elif style_name == "Blue":
                self.icon_provider.set_theme("light")
                self.refresh_icons()
                self.canvas.setStyleSheet(
                    "background-color: #E5F0FF;"
                )  # Light blue background
            elif style_name == "Green":
                self.icon_provider.set_theme("light")
                self.refresh_icons()
                self.canvas.setStyleSheet(
                    "background-color: #E5FFE5;"
                )  # Light green background
            elif "dark" in style_name:
                self.icon_provider.set_theme("dark")
                self.refresh_icons()
                self.canvas.setStyleSheet("")  # Default background
            else:
                self.icon_provider.set_theme("light")
                self.refresh_icons()
                self.canvas.setStyleSheet("")  # Default background
            # Clear any existing stylesheet for annotation dock
            if hasattr(self, "annotation_dock"):
                if style_name == "Dark":
                    self.annotation_dock.setStyleSheet(
                        """
                        QListWidget {
                            background-color: #252525;
                            color: #FFFFFF;
                            border: 1px solid #555555;
                        }
                    """
                    )
                else:
                    self.annotation_dock.setStyleSheet("")

            # Update class dock if it exists
            if hasattr(self, "class_dock"):
                if style_name == "Dark":
                    self.class_dock.setStyleSheet(
                        """
                        QListWidget {
                            background-color: #252525;
                            color: #FFFFFF;
                            border: 1px solid #555555;
                        }
                    """
                    )
                else:
                    self.class_dock.setStyleSheet("")

            self.statusBar.showMessage(f"Style changed to {style_name}")

    def refresh_icons(self):
        """Refresh all icons in the UI to match the current theme."""

        # Update play button icon
        if hasattr(self, "play_button"):
            icon_name = (
                "media-playback-pause" if self.is_playing else "media-playback-start"
            )
            self.play_button.setIcon(self.icon_provider.get_icon(icon_name))

        # Update prev/next buttons
        if hasattr(self, "prev_button"):
            self.prev_button.setIcon(self.icon_provider.get_icon("media-skip-backward"))

        if hasattr(self, "next_button"):
            self.next_button.setIcon(self.icon_provider.get_icon("media-skip-forward"))

        # Update toolbar if it exists
        if hasattr(self, "toolbar") and hasattr(self.toolbar, "refresh_icons"):
            self.toolbar.refresh_icons()

    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About Video Annotation Tool",
            "Video Annotation Tool (VAT)\n\n"
            "A tool for annotating objects in videos for computer vision tasks.\n\n"
            "Features:\n"
            "- Bounding box annotations with edge movement for precise adjustments\n"
            "- Multiple object classes with customizable colors\n"
            "- Export to common formats (COCO, YOLO, Pascal VOC)\n"
            "- Project saving and loading\n"
            "- Right-click context menu for quick editing\n\n"
            "Created as a demonstration of PyQt5 capabilities.",
        )

    def setup_autosave(self):
        """Set up auto-save functionality."""
        # Create auto-save timer
        self.autosave_timer = QTimer()
        self.autosave_timer.timeout.connect(self.perform_autosave)

        # Start timer if enabled
        if self.autosave_enabled:
            self.autosave_timer.start(self.autosave_interval)
            if hasattr(self, "statusBar") and self.statusBar:
                self.statusBar.showMessage("Auto-save enabled", 3000)

    def perform_autosave(self):
        """Perform auto-save of the current project."""
        if not self.autosave_enabled:
            return

        # Only auto-save if we have a project file, video file, or image dataset
        if not hasattr(self, "project_file") or not self.project_file:
            # Create auto-save filename based on video filename or image dataset folder
            if not self.autosave_file:
                if (
                    hasattr(self, "is_image_dataset")
                    and self.is_image_dataset
                    and self.image_files
                ):
                    # For image datasets, use the folder name
                    image_folder = os.path.dirname(self.image_files[0])
                    folder_name = os.path.basename(image_folder)
                    self.autosave_file = os.path.join(
                        image_folder, f"{folder_name}_autosave.json"
                    )
                elif hasattr(self, "video_filename") and self.video_filename:
                    # For videos, use the video filename
                    video_base = os.path.splitext(self.video_filename)[0]
                    self.autosave_file = f"{video_base}_autosave.json"
                else:
                    # No valid source to auto-save
                    return
        else:
            # Use the project file for auto-save
            self.autosave_file = self.project_file

        try:
            # Get video path or image dataset info
            video_path = None
            image_dataset_info = None

            if hasattr(self, "is_image_dataset") and self.is_image_dataset:
                # For image datasets, store the folder and relative paths
                if self.image_files:
                    base_folder = os.path.dirname(self.image_files[0])
                    image_dataset_info = {
                        "is_image_dataset": True,
                        "base_folder": base_folder,
                        "image_files": [
                            os.path.relpath(f, base_folder) for f in self.image_files
                        ],
                    }
            else:
                # For videos, store the video path
                video_path = getattr(self, "video_filename", None)

            # Get class attributes
            class_attributes = getattr(self.canvas, "class_attributes", {})

            # Save project with additional data
            save_project(
                self.autosave_file,
                self.canvas.annotations,
                self.canvas.class_colors,
                video_path=video_path,
                current_frame=self.current_frame,
                frame_annotations=self.frame_annotations,
                class_attributes=class_attributes,
                current_style=self.current_style,
                auto_show_attribute_dialog=self.auto_show_attribute_dialog,
                use_previous_attributes=self.use_previous_attributes,
                duplicate_frames_enabled=self.duplicate_frames_enabled,
                frame_hashes=self.frame_hashes,
                duplicate_frames_cache=self.duplicate_frames_cache,
                image_dataset_info=image_dataset_info,
            )

            self.last_autosave_time = QDateTime.currentDateTime()
            self.statusBar.showMessage(
                f"Auto-saved to {os.path.basename(self.autosave_file)}", 3000
            )
        except Exception as e:
            print(f"Auto-save failed: {str(e)}")

    def update_recent_projects_menu(self):
        """Update the recent projects menu with the latest projects."""
        self.recent_projects_menu.clear()

        recent_projects = get_recent_projects()
        if not recent_projects:
            no_recent = QAction("No Recent Projects", self)
            no_recent.setEnabled(False)
            self.recent_projects_menu.addAction(no_recent)
            return

        for project_path in recent_projects:
            project_name = os.path.basename(project_path)
            action = QAction(project_name, self)
            action.setData(project_path)
            action.triggered.connect(
                lambda checked, path=project_path: self.load_project(path)
            )
            self.recent_projects_menu.addAction(action)

        self.recent_projects_menu.addSeparator()
        clear_action = QAction("Clear Recent Projects", self)
        clear_action.triggered.connect(self.clear_recent_projects)
        self.recent_projects_menu.addAction(clear_action)

    def clear_recent_projects(self):
        """Clear the list of recent projects."""

        config_dir = get_config_directory()
        recent_projects_file = os.path.join(config_dir, "recent_projects.json")

        with open(recent_projects_file, "w") as f:
            json.dump([], f)

        self.update_recent_projects_menu()
        self.statusBar.showMessage("Recent projects cleared", 3000)

    def get_previous_annotation_attributes(self, class_name):
        """
        Find the most recent annotation of the same class and return its attributes.

        Args:
            class_name (str): The class name to match

        Returns:
            dict: Attributes dictionary or None if no previous annotation found
        """
        if not self.use_previous_attributes:
            return None

        # First check current frame
        for annotation in reversed(self.canvas.annotations):
            if annotation.class_name == class_name:
                return annotation.attributes.copy()

        # Then check previous frames in reverse order
        for frame_num in sorted(self.frame_annotations.keys(), reverse=True):
            if frame_num >= self.current_frame:
                continue

            for annotation in reversed(self.frame_annotations[frame_num]):
                if annotation.class_name == class_name:
                    return annotation.attributes.copy()

        # If no previous annotation found, return None
        return None

    def update_settings_menu_actions(self):
        """Update the settings menu actions to reflect current settings."""
        if not hasattr(self, "settings_menu"):
            return

        for action in self.settings_menu.actions():
            if action.text() == "Enable Auto-save":
                action.setChecked(self.autosave_enabled)
            elif action.text() == "Show Attribute Dialog for New Annotations":
                action.setChecked(self.auto_show_attribute_dialog)
            elif action.text() == "Use Previous Annotation Attributes as Default":
                action.setChecked(self.use_previous_attributes)

    def open_image_folder(self):
        """Open a folder of images."""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Open Image Folder", "", QFileDialog.ShowDirsOnly
        )

        if not folder_path:
            return

        # Import the dataset manager
        from utils.dataset_manager import detect_folder_type

        # Detect if this is a simple folder or a dataset
        folder_type = detect_folder_type(folder_path)

        if folder_type == "dataset":
            # Handle as a dataset
            self.open_image_dataset(folder_path)
        else:
            # Handle as a simple image folder
            self.open_simple_image_folder(folder_path)

    def open_simple_image_folder(self, folder_path):
        """Open a simple folder of images."""
        # Reset existing state
        self.reset_media_state()

        # Get all image files in the folder
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
        image_files = []

        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(folder_path, file))

        if not image_files:
            QMessageBox.warning(
                self,
                "Open Image Folder",
                "No image files found in the selected folder!",
            )
            return

        # Sort image files for consistent ordering
        image_files.sort()

        # Set up the image dataset interface
        self.image_files = image_files
        self.total_frames = len(image_files)
        self.current_frame = 0
        self.is_image_dataset = True

        # Load the first image
        self.load_current_image()

        # Update UI
        self.frame_slider.setMaximum(self.total_frames - 1)
        self.update_frame_info()

        # Update window title
        folder_name = os.path.basename(folder_path)
        self.setWindowTitle(f"Video Annotation Tool - Image Folder: {folder_name}")

        # Enable play button for image datasets
        if hasattr(self, "play_button"):
            self.play_button.setEnabled(True)
            self.play_button.setIcon(
                self.icon_provider.get_icon("media-playback-start")
            )

        # Set up auto-save for this dataset
        self.autosave_file = os.path.join(folder_path, f"{folder_name}_autosave.json")

        # Start auto-save timer
        if self.autosave_enabled and not self.autosave_timer.isActive():
            self.autosave_timer.start(self.autosave_interval)

        # Check for annotation files
        self.check_for_image_annotation_files(folder_path, folder_name)

    def open_image_dataset(self, folder_path=None):
        """Open an image dataset with advanced options."""
        if folder_path is None:
            folder_path = QFileDialog.getExistingDirectory(
                self, "Open Image Dataset", "", QFileDialog.ShowDirsOnly
            )

        if not folder_path:
            return

        # Show import dialog
        config = import_dataset_dialog(self, folder_path)

        if not config:
            return  # User cancelled

        # Reset existing state
        self.reset_media_state()

        # Load the dataset
        image_files, success_message = load_dataset(
            self, config, self.frame_annotations, self.canvas.class_colors, BoundingBox
        )

        if not image_files:
            QMessageBox.warning(
                self,
                "Open Image Dataset",
                "No image files found in the selected dataset!",
            )
            return

        # Set up the image dataset interface
        self.image_files = image_files
        self.total_frames = len(image_files)
        self.current_frame = 0
        self.is_image_dataset = True

        # Load the first image
        self.load_current_image()

        # Update UI
        self.frame_slider.setMaximum(self.total_frames - 1)
        self.update_frame_info()

        # Update window title
        folder_name = os.path.basename(folder_path)
        self.setWindowTitle(f"Video Annotation Tool - Image Dataset: {folder_name}")

        # Enable play button for image datasets
        if hasattr(self, "play_button"):
            self.play_button.setEnabled(True)
            self.play_button.setIcon(
                self.icon_provider.get_icon("media-playback-start")
            )

        # Set up auto-save for this dataset
        self.autosave_file = os.path.join(folder_path, f"{folder_name}_autosave.json")

        # Start auto-save timer
        if self.autosave_enabled and not self.autosave_timer.isActive():
            self.autosave_timer.start(self.autosave_interval)

        # Update UI
        self.update_annotation_list()
        self.toolbar.update_class_selector()
        self.class_dock.update_class_list()

        # Show success message
        self.statusBar.showMessage(success_message)

    def load_current_image(self):
        """Load the current image from the image dataset."""
        if not hasattr(self, "image_files") or not self.image_files:
            return

        if 0 <= self.current_frame < len(self.image_files):
            image_path = self.image_files[self.current_frame]

            # Load the image using OpenCV
            frame = cv2.imread(image_path)

            if frame is not None:

                # Set the frame to the canvas
                self.canvas.set_frame(frame)

                # Load annotations for this frame if they exist
                self.load_current_frame_annotations()

                # Update frame info and slider
                self.update_frame_info()

                return True
            else:
                self.statusBar.showMessage(
                    f"Error loading image: {os.path.basename(image_path)}"
                )
                return False

    def toggle_duplicate_frames_detection(self):
        """Toggle automatic duplicate frame detection and annotation propagation."""
        self.duplicate_frames_enabled = self.duplicate_frames_action.isChecked()

        if self.duplicate_frames_enabled:
            # Only prompt to scan if we don't already have frame hashes
            if not self.frame_hashes:
                reply = QMessageBox.question(
                    self,
                    "Duplicate Frame Detection",
                    "This will automatically propagate annotations to duplicate frames.\n\n"
                    "Do you want to scan the entire video now for duplicate frames?\n"
                    "(This may take some time for long videos)",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )

                if reply == QMessageBox.Yes:
                    self.scan_video_for_duplicates()

            self.statusBar.showMessage(
                "Duplicate frame detection enabled - annotations will be propagated automatically"
            )
        else:
            self.statusBar.showMessage("Duplicate frame detection disabled")

        # Mark project as modified
        self.project_modified = True

    def scan_video_for_duplicates(self):
        """Scan the entire video to identify duplicate frames."""
        if not self.cap or not self.cap.isOpened():
            QMessageBox.warning(self, "Scan Video", "Please open a video first!")
            return

        # Create progress dialog
        progress = QDialog(self)
        progress.setWindowTitle("Scanning Video")
        progress.setFixedSize(300, 100)
        layout = QVBoxLayout(progress)

        label = QLabel("Scanning for duplicate frames...")
        layout.addWidget(label)

        progress_bar = QProgressBar()
        progress_bar.setRange(0, self.total_frames)
        layout.addWidget(progress_bar)

        # Non-blocking progress dialog
        progress.setModal(False)
        progress.show()
        QApplication.processEvents()

        # Remember current position
        current_pos = self.current_frame

        # Reset cache
        self.duplicate_frames_cache = {}
        self.frame_hashes = {}

        # Scan video
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for frame_num in range(self.total_frames):
            ret, frame = self.cap.read()
            if not ret:
                break

            # Update progress
            progress_bar.setValue(frame_num)
            if frame_num % 10 == 0:  # Update UI every 10 frames
                QApplication.processEvents()

            # Calculate frame hash
            frame_hash = calculate_frame_hash(frame)
            self.frame_hashes[frame_num] = frame_hash

            # Add to duplicate cache
            if frame_hash in self.duplicate_frames_cache:
                self.duplicate_frames_cache[frame_hash].append(frame_num)
            else:
                self.duplicate_frames_cache[frame_hash] = [frame_num]

        # Restore position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        ret, frame = self.cap.read()
        if ret:
            self.canvas.set_frame(frame)

        # Close progress dialog
        progress.close()

        # Report results
        duplicate_count = sum(
            len(frames) - 1
            for frames in self.duplicate_frames_cache.values()
            if len(frames) > 1
        )
        QMessageBox.information(
            self,
            "Scan Complete",
            f"Found {duplicate_count} duplicate frames in {self.total_frames} total frames.",
        )

    def propagate_annotations_to_duplicate(self, frame_hash):
        """
        Propagate annotations from other frames with the same hash to the current frame.

        Args:
            frame_hash (str): The hash of the current frame
        """
        duplicate_frames = self.duplicate_frames_cache[frame_hash]

        # Skip if this is the first occurrence of this frame
        if duplicate_frames[0] == self.current_frame:
            return

        # Look for annotations in other frames with the same hash
        for frame_num in duplicate_frames:
            if frame_num != self.current_frame and frame_num in self.frame_annotations:
                # Found annotations in another frame with the same hash
                if not self.frame_annotations.get(self.current_frame):
                    # Copy annotations to current frame
                    self.frame_annotations[self.current_frame] = [
                        self.clone_annotation(ann)
                        for ann in self.frame_annotations[frame_num]
                    ]
                    self.statusBar.showMessage(
                        f"Automatically copied annotations from duplicate frame {frame_num}",
                        3000,
                    )
                    return

    def clone_annotation(self, annotation):
        """
        Create a deep copy of an annotation.

        Args:
            annotation: The annotation to clone

        Returns:
            A new annotation object with the same properties
        """

        return deepcopy(annotation)

    def propagate_annotations(self):
        """Propagate current frame annotations to a range of frames."""
        if not self.canvas.annotations:
            QMessageBox.warning(
                self,
                "Propagate Annotations",
                "No annotations in current frame to propagate!",
            )
            return

        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Propagate Annotations")
        dialog.setMinimumWidth(300)

        layout = QVBoxLayout(dialog)

        # Frame range selection
        range_group = QGroupBox("Frame Range")
        range_layout = QFormLayout(range_group)

        start_spin = QSpinBox()
        start_spin.setRange(0, self.total_frames - 1)
        start_spin.setValue(self.current_frame)

        end_spin = QSpinBox()
        end_spin.setRange(0, self.total_frames - 1)
        end_spin.setValue(min(self.current_frame + 10, self.total_frames - 1))

        range_layout.addRow("Start Frame:", start_spin)
        range_layout.addRow("End Frame:", end_spin)

        # Options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)

        overwrite_check = QCheckBox("Overwrite existing annotations")
        overwrite_check.setChecked(False)

        smart_check = QCheckBox("Smart propagation (skip duplicate frames)")
        smart_check.setChecked(True)
        smart_check.setEnabled(self.duplicate_frames_enabled)

        options_layout.addWidget(overwrite_check)
        options_layout.addWidget(smart_check)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        # Add widgets to layout
        layout.addWidget(range_group)
        layout.addWidget(options_group)
        layout.addWidget(buttons)

        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            self.save_undo_state()
            start_frame = start_spin.value()
            end_frame = end_spin.value()
            overwrite = overwrite_check.isChecked()
            smart = smart_check.isChecked() and self.duplicate_frames_enabled

            # Validate range
            if start_frame > end_frame:
                start_frame, end_frame = end_frame, start_frame

            # Get current annotations
            current_annotations = [
                self.clone_annotation(ann) for ann in self.canvas.annotations
            ]

            # Create progress dialog
            progress = QDialog(self)
            progress.setWindowTitle("Propagating Annotations")
            progress.setFixedSize(300, 100)
            progress_layout = QVBoxLayout(progress)

            label = QLabel(
                f"Propagating annotations to frames {start_frame}-{end_frame}..."
            )
            progress_layout.addWidget(label)

            progress_bar = QProgressBar()
            progress_bar.setRange(start_frame, end_frame)
            progress_layout.addWidget(progress_bar)

            # Non-blocking progress dialog
            progress.setModal(False)
            progress.show()
            QApplication.processEvents()

            # Track processed frames for smart propagation
            processed_hashes = set()

            # Propagate annotations
            for frame_num in range(start_frame, end_frame + 1):
                # Skip current frame
                if frame_num == self.current_frame:
                    continue

                # Update progress
                progress_bar.setValue(frame_num)
                if frame_num % 5 == 0:  # Update UI every 5 frames
                    QApplication.processEvents()

                # Skip frames with existing annotations if not overwriting
                if (
                    not overwrite
                    and frame_num in self.frame_annotations
                    and self.frame_annotations[frame_num]
                ):
                    continue

                # Smart propagation - skip duplicate frames that have already been processed
                if smart and frame_num in self.frame_hashes:
                    frame_hash = self.frame_hashes[frame_num]
                    if frame_hash in processed_hashes:
                        continue
                    processed_hashes.add(frame_hash)

                # Copy annotations to this frame
                self.frame_annotations[frame_num] = [
                    self.clone_annotation(ann) for ann in current_annotations
                ]

            # Close progress dialog
            progress.close()

            # Update UI if we're on one of the affected frames
            if start_frame <= self.current_frame <= end_frame:
                self.load_current_frame_annotations()

            self.statusBar.showMessage(
                f"Annotations propagated to frames {start_frame}-{end_frame}", 5000
            )

    def propagate_to_duplicate_frames(self, frame_hash):
        """
        Propagate current frame annotations to all duplicate frames with the same hash.

        Args:
            frame_hash (str): The hash of the current frame
        """
        if not frame_hash or frame_hash not in self.duplicate_frames_cache:
            return

        duplicate_frames = self.duplicate_frames_cache[frame_hash]
        if len(duplicate_frames) <= 1:
            return
        self.save_undo_state()

        # Get current frame annotations
        current_annotations = [
            self.clone_annotation(ann) for ann in self.canvas.annotations
        ]

        # Count how many frames will be updated
        update_count = 0

        # Copy to all duplicate frames
        for frame_num in duplicate_frames:
            if frame_num != self.current_frame:
                self.frame_annotations[frame_num] = [
                    self.clone_annotation(ann) for ann in current_annotations
                ]
                update_count += 1

        if update_count > 0:
            self.statusBar.showMessage(
                f"Automatically propagated annotations to {update_count} duplicate frames",
                3000,
            )

    def detect_similar_frames(self, reference_frame, similarity_threshold=0.9):
        """
        Detect frames that are similar to the reference frame.

        Args:
            reference_frame (int): The reference frame number
            similarity_threshold (float): Threshold for considering frames similar (0-1)

        Returns:
            list: List of frame numbers that are similar to the reference frame
        """
        if not self.cap or not self.cap.isOpened():
            return []

        # Get reference frame image
        current_pos = self.current_frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, reference_frame)
        ret, ref_frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
            return []

        # Create progress dialog
        progress = QDialog(self)
        progress.setWindowTitle("Finding Similar Frames")
        progress.setFixedSize(300, 100)
        layout = QVBoxLayout(progress)

        label = QLabel("Scanning for similar frames...")
        layout.addWidget(label)

        progress_bar = QProgressBar()
        progress_bar.setRange(0, self.total_frames)
        layout.addWidget(progress_bar)

        # Non-blocking progress dialog
        progress.setModal(False)
        progress.show()
        QApplication.processEvents()

        # Scan video
        similar_frames = [reference_frame]  # Include reference frame in results
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for frame_num in range(self.total_frames):
            # Skip reference frame
            if frame_num == reference_frame:
                continue

            # Update progress
            progress_bar.setValue(frame_num)
            if frame_num % 10 == 0:  # Update UI every 10 frames
                QApplication.processEvents()

            # Read frame
            ret, frame = self.cap.read()
            if not ret:
                break

            # Use mse_similarity from utils.im_tools
            similarity = mse_similarity(ref_frame, frame)

            # Add to similar frames if above threshold
            if similarity >= similarity_threshold:
                similar_frames.append(frame_num)

        # Restore position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)

        # Close progress dialog
        progress.close()

        return similar_frames

    def propagate_to_similar_frames(self):
        """Propagate current frame annotations to similar frames."""
        if not self.canvas.annotations:
            QMessageBox.warning(
                self,
                "Propagate Annotations",
                "No annotations in current frame to propagate!",
            )
            return

        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Propagate to Similar Frames")
        dialog.setMinimumWidth(350)

        layout = QVBoxLayout(dialog)

        # Similarity threshold
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Similarity Threshold:")
        threshold_slider = QSlider(Qt.Horizontal)
        threshold_slider.setRange(50, 99)
        threshold_slider.setValue(90)  # Default 0.9
        threshold_value = QLabel("0.90")

        def update_threshold_label(value):
            threshold_value.setText(f"{value/100:.2f}")

        threshold_slider.valueChanged.connect(update_threshold_label)

        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(threshold_slider)
        threshold_layout.addWidget(threshold_value)

        # Options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)

        overwrite_check = QCheckBox("Overwrite existing annotations")
        overwrite_check.setChecked(False)

        preview_check = QCheckBox("Preview similar frames before propagating")
        preview_check.setChecked(True)

        options_layout.addWidget(overwrite_check)
        options_layout.addWidget(preview_check)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        # Add widgets to layout
        layout.addLayout(threshold_layout)
        layout.addWidget(options_group)
        layout.addWidget(buttons)

        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            self.save_undo_state()
            similarity_threshold = threshold_slider.value() / 100.0
            overwrite = overwrite_check.isChecked()
            preview = preview_check.isChecked()

            # Find similar frames
            self.statusBar.showMessage(
                "Finding similar frames... This may take a moment."
            )
            QApplication.processEvents()

            similar_frames = self.detect_similar_frames(
                self.current_frame, similarity_threshold
            )

            if len(similar_frames) <= 1:
                QMessageBox.information(
                    self,
                    "No Similar Frames",
                    "No similar frames were found with the current threshold.",
                )
                return

            # Show preview if requested
            if preview:
                preview_result = self.preview_similar_frames(similar_frames)
                if not preview_result:
                    return  # User cancelled

            # Get current annotations
            current_annotations = [
                self.clone_annotation(ann) for ann in self.canvas.annotations
            ]

            # Propagate to similar frames
            propagated_count = 0
            for frame_num in similar_frames:
                # Skip current frame
                if frame_num == self.current_frame:
                    continue

                # Skip frames with existing annotations if not overwriting
                if (
                    not overwrite
                    and frame_num in self.frame_annotations
                    and self.frame_annotations[frame_num]
                ):
                    continue

                # Copy annotations to this frame
                self.frame_annotations[frame_num] = [
                    self.clone_annotation(ann) for ann in current_annotations
                ]
                propagated_count += 1

            self.statusBar.showMessage(
                f"Annotations propagated to {propagated_count} similar frames", 5000
            )

    def preview_similar_frames(self, frame_numbers):
        """
        Show a preview of similar frames and let the user select which ones to include.

        Args:
            frame_numbers (list): List of frame numbers to preview

        Returns:
            bool: True if user confirmed, False if cancelled
        """
        if not frame_numbers or len(frame_numbers) <= 1:
            return False

        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Preview Similar Frames")
        dialog.setMinimumWidth(600)
        dialog.setMinimumHeight(400)

        layout = QVBoxLayout(dialog)

        # Instructions
        instructions = QLabel("Select frames to propagate annotations to:")
        layout.addWidget(instructions)

        # Frame list
        frame_list = QListWidget()
        frame_list.setSelectionMode(QListWidget.MultiSelection)
        layout.addWidget(frame_list)

        # Remember current position
        current_pos = self.current_frame

        # Add frames to list
        for frame_num in frame_numbers:
            # Skip current frame
            if frame_num == self.current_frame:
                continue

            # Create item with frame number
            item = QListWidgetItem(f"Frame {frame_num}")
            item.setData(Qt.UserRole, frame_num)

            # Get frame thumbnail
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()
            if ret:
                # Create thumbnail
                thumbnail = create_thumbnail(frame, (160, 90))

                # Convert to QImage and QPixmap
                h, w, c = thumbnail.shape
                qimg = QImage(thumbnail.data, w, h, w * c, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)

                # Set icon
                item.setIcon(QIcon(pixmap))

            # Add to list and select by default
            frame_list.addItem(item)
            item.setSelected(True)

        # Restore position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)

        # Select/Deselect All buttons
        button_layout = QHBoxLayout()

        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(
            lambda: [
                frame_list.item(i).setSelected(True) for i in range(frame_list.count())
            ]
        )

        deselect_all_btn = QPushButton("Deselect All")
        deselect_all_btn.clicked.connect(
            lambda: [
                frame_list.item(i).setSelected(False) for i in range(frame_list.count())
            ]
        )

        button_layout.addWidget(select_all_btn)
        button_layout.addWidget(deselect_all_btn)
        layout.addLayout(button_layout)

        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            # Update frame_numbers to only include selected frames
            selected_frames = [self.current_frame]  # Always include current frame
            for i in range(frame_list.count()):
                item = frame_list.item(i)
                if item.isSelected():
                    selected_frames.append(item.data(Qt.UserRole))

            # Update the original list
            frame_numbers.clear()
            frame_numbers.extend(selected_frames)
            return True
        else:
            return False

    def clear_application_history():
        """Clear all application history files."""
        config_dir = get_config_directory()

        # List of files to delete
        files_to_delete = ["recent_projects.json", "last_state.json", "settings.json"]

        # Delete each file
        deleted_files = []
        for filename in files_to_delete:
            file_path = os.path.join(config_dir, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted_files.append(filename)

        return deleted_files

    def reset_application_state(self):
        """Reset the application to its initial state."""
        # Reset project-related variables
        self.project_file = None
        self.project_modified = False

        # Reset video-related variables
        if self.cap:
            self.cap.release()
            self.cap = None

        self.video_filename = ""
        self.current_frame = 0
        self.total_frames = 0
        self.is_playing = False

        # Reset frame slider
        self.frame_slider.setValue(0)
        self.frame_slider.setMaximum(100)
        self.frame_label.setText("0/0")

        # Reset annotations
        self.canvas.annotations = []
        self.frame_annotations = {}

        # Reset class information
        self.canvas.class_colors = {"Quad": QColor(0, 255, 255)}
        if hasattr(self.canvas, "class_attributes"):
            self.canvas.class_attributes = {}
        self.canvas.current_class = "Quad"

        # Reset duplicate frame detection
        self.duplicate_frames_enabled = False
        self.frame_hashes = {}
        self.duplicate_frames_cache = {}

        if hasattr(self, "duplicate_frames_action"):
            self.duplicate_frames_action.setChecked(False)

        # Reset canvas
        self.canvas.pixmap = None
        self.canvas.update()

        # Reset UI
        self.update_annotation_list()

        # Update class-related UI
        self.refresh_class_ui()

        # Reset to default style
        self.change_style("DarkModern")

        # Reset settings to defaults
        self.auto_show_attribute_dialog = True
        self.use_previous_attributes = True
        self.autosave_enabled = True
        self.autosave_interval = 5000  # 5 seconds

        # Update settings menu
        self.update_settings_menu_actions()

        # Update status bar
        self.statusBar.showMessage("Application reset to initial state")

    def delete_history(self):
        """Delete all application history and reset to initial state."""
        reply = QMessageBox.question(
            self,
            "Delete History",
            "This will delete all recent projects, saved settings, and application history.\n\n"
            "The application will return to its initial state after installation.\n\n"
            "Are you sure you want to continue?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply != QMessageBox.Yes:
            return

        try:
            # Get config directory
            config_dir = get_config_directory()

            # List of files to delete
            files_to_delete = [
                "recent_projects.json",
                "last_state.json",
                "settings.json",
            ]

            # Delete each file
            for filename in files_to_delete:
                file_path = os.path.join(config_dir, filename)
                if os.path.exists(file_path):
                    os.remove(file_path)

            # Clear recent projects menu
            self.update_recent_projects_menu()

            # Reset application state
            self.reset_application_state()

            # Show success message
            QMessageBox.information(
                self,
                "History Deleted",
                "Application history has been deleted successfully.\n\n"
                "The application has been reset to its initial state.",
            )

        except Exception as e:
            QMessageBox.critical(
                self, "Error", f"An error occurred while deleting history: {str(e)}"
            )

    def load_image_dataset_from_project(self, image_dataset_info, current_frame):
        """Load an image dataset from project information."""
        base_folder = image_dataset_info.get("base_folder", "")
        relative_paths = image_dataset_info.get("image_files", [])

        # Check if base folder exists
        if not os.path.exists(base_folder):
            # Ask user to locate the base folder
            msg = f"The original image folder '{base_folder}' was not found.\nWould you like to locate it?"
            reply = QMessageBox.question(
                self,
                "Folder Not Found",
                msg,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )

            if reply == QMessageBox.Yes:
                new_base_folder = QFileDialog.getExistingDirectory(
                    self, "Locate Image Folder", "", QFileDialog.ShowDirsOnly
                )
                if new_base_folder:
                    base_folder = new_base_folder
                else:
                    return False
            else:
                return False

        # Construct absolute paths
        image_files = []
        missing_files = []

        for rel_path in relative_paths:
            abs_path = os.path.join(base_folder, rel_path)
            if os.path.exists(abs_path):
                image_files.append(abs_path)
            else:
                missing_files.append(rel_path)

        # Warn about missing files
        if missing_files:
            if len(missing_files) > 10:
                missing_msg = (
                    "\n".join(missing_files[:10])
                    + f"\n... and {len(missing_files) - 10} more"
                )
            else:
                missing_msg = "\n".join(missing_files)

            QMessageBox.warning(
                self,
                "Missing Files",
                f"The following {len(missing_files)} image files were not found:\n\n{missing_msg}",
            )

        if not image_files:
            QMessageBox.critical(
                self, "Error", "No image files could be loaded from the project."
            )
            return False

        # Set up the image dataset
        self.image_files = image_files
        self.total_frames = len(image_files)
        self.current_frame = min(current_frame, self.total_frames - 1)
        self.is_image_dataset = True

        # Update UI
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(self.total_frames - 1)
        self.frame_slider.setValue(self.current_frame)

        self.load_current_image()

        self.update_frame_info()

        # Update window title
        self.setWindowTitle(
            f"Video Annotation Tool - Image Dataset: {os.path.basename(base_folder)}"
        )

        if hasattr(self, "play_button"):
            self.play_button.setEnabled(True)
            self.play_button.setIcon(
                self.icon_provider.get_icon("media-playback-start")
            )

        return True

    def load_video_from_project(self, video_path, current_frame):
        """Load a video from project information."""
        # Temporarily store the original method
        original_check_method = self.check_for_annotation_files

        # Replace with a dummy method that does nothing
        self.check_for_annotation_files = lambda x: None

        # Load the video file
        success = self.load_video_file(video_path)

        # Restore the original method
        self.check_for_annotation_files = original_check_method

        if success:
            # Set to the saved frame
            if current_frame > 0 and current_frame < self.total_frames:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame = current_frame
                    self.canvas.set_frame(frame)
                    self.update_frame_info()
                    self.load_current_frame_annotations()

        return success

    def scan_images_for_duplicates(self):
        """Scan all images in the dataset to identify duplicates."""
        if not hasattr(self, "image_files") or not self.image_files:
            return

        # Create progress dialog
        progress = QDialog(self)
        progress.setWindowTitle("Scanning Images")
        progress.setFixedSize(300, 100)
        layout = QVBoxLayout(progress)

        label = QLabel("Scanning for duplicate images...")
        layout.addWidget(label)

        progress_bar = QProgressBar()
        progress_bar.setRange(0, len(self.image_files))
        layout.addWidget(progress_bar)

        # Non-blocking progress dialog
        progress.setModal(False)
        progress.show()
        QApplication.processEvents()

        # Reset cache
        self.duplicate_frames_cache = {}
        self.frame_hashes = {}

        # Scan images
        for frame_num, image_path in enumerate(self.image_files):
            # Update progress
            progress_bar.setValue(frame_num)
            if frame_num % 5 == 0:  # Update UI every 5 images
                QApplication.processEvents()

            # Load image
            frame = cv2.imread(image_path)
            if frame is None:
                continue

            # Calculate frame hash
            frame_hash = calculate_frame_hash(frame)
            self.frame_hashes[frame_num] = frame_hash

            # Add to duplicate cache
            if frame_hash in self.duplicate_frames_cache:
                self.duplicate_frames_cache[frame_hash].append(frame_num)
            else:
                self.duplicate_frames_cache[frame_hash] = [frame_num]

        # Close progress dialog
        progress.close()

        # Report results
        duplicate_count = sum(
            len(frames) - 1
            for frames in self.duplicate_frames_cache.values()
            if len(frames) > 1
        )
        QMessageBox.information(
            self,
            "Scan Complete",
            f"Found {duplicate_count} duplicate images in {len(self.image_files)} total images.",
        )

    def set_slideshow_speed(self, speed_factor):
        """Set the speed of the image slideshow.

        Args:
            speed_factor (float): Speed multiplier (1.0 = 1 second per image)
        """
        if hasattr(self, "is_image_dataset") and self.is_image_dataset:
            # Calculate interval in milliseconds (1000ms / speed_factor)
            interval = int(1000 / speed_factor)

            # Update interval if currently playing
            if self.is_playing:
                self.play_timer.stop()
                self.play_timer.start(interval)

            self.statusBar.showMessage(f"Slideshow speed: {speed_factor}x")

    def setup_playback_timer(self):
        """Set up the timer for video playback or image slideshow."""
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.next_frame)

        # Initialize slideshow speed to 1 second per image
        self.slideshow_speed = 1.0

    def check_for_image_annotation_files(self, folder_path, folder_name):
        """
        Check if annotation files exist for this image dataset.

        Args:
            folder_path (str): Path to the image folder
            folder_name (str): Name of the image folder
        """
        # Check for auto-save file first
        autosave_file = os.path.join(folder_path, f"{folder_name}_autosave.json")

        if os.path.exists(autosave_file):
            reply = QMessageBox.question(
                self,
                "Auto-Save Found",
                "An auto-save file was found for this image dataset.\nWould you like to load it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )

            if reply == QMessageBox.Yes:
                try:
                    # Set flag to prevent recursive auto-save prompts
                    self._loading_from_project = True
                    self.load_project(autosave_file)
                    self._loading_from_project = False
                    return
                except Exception as e:
                    self._loading_from_project = False
                    QMessageBox.warning(
                        self,
                        "Auto-Save Error",
                        f"Error loading auto-save file: {str(e)}",
                    )

        # List of possible annotation file patterns
        annotation_patterns = [
            f"{folder_name}_annotations.json",  # COCO format
            f"{folder_name}_annotations.txt",  # Raya format
            "annotations.json",  # Common COCO filename
        ]

        # Find matching annotation files
        annotation_files = []
        for pattern in annotation_patterns:
            potential_file = os.path.join(folder_path, pattern)
            if os.path.exists(potential_file) and potential_file != autosave_file:
                annotation_files.append(potential_file)

        # Also check for YOLO format (classes.txt and image-specific .txt files)
        classes_file = os.path.join(folder_path, "classes.txt")
        if os.path.exists(classes_file):
            # Check if there are matching .txt files for images
            has_txt_annotations = False
            for image_path in self.image_files[:10]:  # Check first 10 images
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                txt_file = os.path.join(folder_path, f"{base_name}.txt")
                if os.path.exists(txt_file):
                    has_txt_annotations = True
                    break

            if has_txt_annotations:
                annotation_files.append(
                    classes_file
                )  # Use classes.txt as the identifier

        # Check if any of the files is a VIAT project file
        for i, file_path in enumerate(annotation_files[:]):
            if file_path.endswith(".json"):
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)
                        if "viat_project_identifier" in data:
                            # This is a VIAT project file, not an annotation export
                            annotation_files.remove(file_path)
                except:
                    pass

        if annotation_files:
            # Create a message with the found files
            message = "Found the following annotation file(s):\n\n"
            for file in annotation_files:
                message += f"- {os.path.basename(file)}\n"
            message += "\nWould you like to import annotations from one of these files?"

            # Show dialog asking if user wants to import
            reply = QMessageBox.question(
                self,
                "Annotation Files Found",
                message,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )

            if reply == QMessageBox.Yes:
                # If multiple files found, let user choose which one to import
                if len(annotation_files) > 1:
                    self.show_annotation_file_selection_dialog(annotation_files)
                else:
                    # Only one file found, import it directly
                    self.import_annotations(annotation_files[0])

    def reset_media_state(self):
        """Reset all state related to the current media (video or image dataset)"""
        # Reset canvas
        if hasattr(self, "canvas"):
            self.canvas.annotations = []
            self.canvas.selected_annotation = None
            self.canvas.update()

        # Reset annotation storage
        self.frame_annotations = {}
        self.current_frame = 0

        # Reset media-specific state
        self.video_path = None
        self.image_dataset_info = None
        self.is_image_dataset = False

        # Reset frame analysis data
        self.frame_hashes = {}
        self.duplicate_frames_cache = {}

        # Update UI
        if hasattr(self, "annotation_dock"):
            self.annotation_dock.update_annotation_list()

        # Reset status
        self.statusBar.showMessage("Ready")

    def export_image_dataset(self):
        """Export the current image dataset with advanced options."""
        if not hasattr(self, "is_image_dataset") or not self.is_image_dataset:
            QMessageBox.warning(
                self, "Export Image Dataset", "Please open an image dataset first!"
            )
            return

        # Import the dataset manager

        # Show export dialog
        config = export_dataset_dialog(self, self.image_files, self.frame_annotations)

        if not config:
            return  # User cancelled

        # Export the dataset
        result = export_dataset(
            self,
            config,
            self.image_files,
            self.frame_annotations,
            self.canvas.class_colors,
        )

        if result:
            self.statusBar.showMessage(result, 5000)

    def create_dataset(self):
        """Create a new dataset from the current annotations."""
        if (
            not hasattr(self, "is_image_dataset")
            or not self.is_image_dataset
            or not self.image_files
        ):
            QMessageBox.warning(
                self,
                "Create Dataset",
                "This feature is only available for image datasets.",
            )
            return

        # Check if we have any annotations
        has_annotations = any(self.frame_annotations.values())
        if not has_annotations:
            QMessageBox.warning(self, "Create Dataset", "No annotations to export!")
            return

        # Import the dataset manager
        from utils.dataset_manager import create_dataset_dialog, create_dataset

        # Show dialog to configure dataset
        config = create_dataset_dialog(
            self, self.image_files, self.frame_annotations, self.canvas.class_colors
        )

        if config:
            # Create the dataset
            success = create_dataset(
                self,
                config,
                self.image_files,
                self.frame_annotations,
                self.canvas.class_colors,
            )

            if success:
                QMessageBox.information(
                    self,
                    "Dataset Created",
                    f"Dataset created successfully in {config['output_dir']}",
                )

    def copy_selected_annotation(self):
        """Copy the currently selected annotation."""
        if hasattr(self, "canvas") and self.canvas.selected_annotation:
            self.clipboard_annotation = self.canvas.selected_annotation.copy()
            self.statusBar.showMessage("Annotation copied", 2000)

    def paste_annotation(self):
        """Paste the copied annotation to the current frame."""
        if hasattr(self, "clipboard_annotation") and self.clipboard_annotation:
            self.save_undo_state()
            new_annotation = self.clipboard_annotation.copy()

            # Add to current frame
            current_frame = self.current_frame
            if current_frame not in self.frame_annotations:
                self.frame_annotations[current_frame] = []

            self.frame_annotations[current_frame].append(new_annotation)

            # Update canvas
            self.canvas.annotations = self.frame_annotations.get(current_frame, [])
            self.canvas.selected_annotation = new_annotation
            self.canvas.update()

            # Update annotation list if it exists
            if hasattr(self, "annotation_dock"):
                self.annotation_dock.update_annotation_list()
            self.statusBar.showMessage("Annotation pasted", 2000)

    def cut_selected_annotation(self):
        """Cut (copy and delete) the selected annotation."""
        if hasattr(self, "canvas") and self.canvas.selected_annotation:
            self.save_undo_state()
            self.clipboard_annotation = self.canvas.selected_annotation.copy()

            # Then delete
            current_frame = self.current_frame
            if current_frame in self.frame_annotations:
                if (
                    self.canvas.selected_annotation
                    in self.frame_annotations[current_frame]
                ):
                    self.frame_annotations[current_frame].remove(
                        self.canvas.selected_annotation
                    )

            # Update canvas
            self.canvas.annotations = self.frame_annotations.get(current_frame, [])
            self.canvas.selected_annotation = None
            self.canvas.update()

            # Update annotation list if it exists
            if hasattr(self, "annotation_dock"):
                self.annotation_dock.update_annotation_list()
            self.statusBar.showMessage("Annotation cut", 2000)

    def select_all_annotations(self):
        """Select all annotations in the current frame."""
        current_frame = self.current_frame
        if (
            current_frame in self.frame_annotations
            and self.frame_annotations[current_frame]
        ):
            # Store all annotations in the current frame as selected
            self.canvas.selected_annotations = self.frame_annotations[
                current_frame
            ].copy()

            # Also set the primary selected annotation if it doesn't exist
            if not self.canvas.selected_annotation and self.canvas.selected_annotations:
                self.canvas.selected_annotation = self.canvas.selected_annotations[0]

            # Update the canvas
            self.canvas.update()

            # Update the annotation list in the dock if it exists
            if hasattr(self, "annotation_dock"):
                self.annotation_dock.select_annotation_in_list(None)  # Deselect current
                self.annotation_dock.select_all_in_list()
            count = len(self.frame_annotations[current_frame])
            self.statusBar.showMessage(
                f"Selected all {count} annotations in this frame", 3000
            )
        else:
            self.statusBar.showMessage("No annotations in this frame", 2000)

    def cycle_annotation_selection(self):
        """Cycle through annotations in the current frame or deselect if only one exists."""
        current_frame = self.current_frame
        if (
            current_frame not in self.frame_annotations
            or not self.frame_annotations[current_frame]
        ):
            self.statusBar.showMessage("No annotations in this frame", 2000)
            return

        annotations = self.frame_annotations[current_frame]

        # If nothing is selected, select the first annotation
        if not self.canvas.selected_annotation:
            self.canvas.selected_annotation = annotations[0]
            self.canvas.update()
            self.statusBar.showMessage(
                f"Selected annotation: {self.canvas.selected_annotation.class_name}",
                2000,
            )
            return

        # Find the index of the currently selected annotation
        try:
            current_index = annotations.index(self.canvas.selected_annotation)
            # If we're at the last annotation, deselect (adding a complete cycle behavior)
            if current_index == len(annotations) - 1:
                self.canvas.selected_annotation = None
                self.canvas.update()
                self.statusBar.showMessage("Annotation deselected", 2000)
            else:
                # Otherwise, select the next annotation
                next_index = current_index + 1
                self.canvas.selected_annotation = annotations[next_index]
                self.canvas.update()
                self.statusBar.showMessage(
                    f"Selected annotation: {self.canvas.selected_annotation.class_name}",
                    2000,
                )
        except ValueError:
            # If the selected annotation is not in the list, select the first one
            self.canvas.selected_annotation = annotations[0]
            self.canvas.update()
            self.statusBar.showMessage(
                f"Selected annotation: {self.canvas.selected_annotation.class_name}",
                2000,
            )

    #
    # Undo Handeling
    #

    def save_undo_state(self):
        """Save the current state for undo functionality."""
        # Create a deep copy of all frame annotations
        all_frame_annotations = {}
        for frame_num, annotations in self.frame_annotations.items():
            all_frame_annotations[frame_num] = [
                self.clone_annotation(ann) for ann in annotations
            ]

        # Create a deep copy of class colors
        class_colors = {}
        for class_name, color in self.canvas.class_colors.items():
            class_colors[class_name] = QColor(color)

        # Create a deep copy of class attributes if they exist
        class_attributes = None
        if hasattr(self.canvas, "class_attributes"):
            class_attributes = deepcopy(self.canvas.class_attributes)

        # Save the state
        undo_state = {
            "frame": self.current_frame,
            "all_annotations": all_frame_annotations,
            "current_annotations": (
                [self.clone_annotation(ann) for ann in self.canvas.annotations]
                if self.canvas.annotations
                else []
            ),
            "class_colors": class_colors,
            "class_attributes": class_attributes,
            "current_class": (
                self.canvas.current_class
                if hasattr(self.canvas, "current_class")
                else None
            ),
        }

        # Add to undo stack
        self.undo_stack.append(undo_state)

        # Clear the redo stack when a new action is performed
        self.redo_stack.clear()

        # Limit the size of the undo stack
        if len(self.undo_stack) > self.max_undo_steps:
            self.undo_stack.pop(0)

    def undo(self):
        """Undo the last annotation or class change."""
        if not self.undo_stack:
            self.statusBar.showMessage("Nothing to undo", 3000)
            return

        # Get the current state before undoing
        current_state = {
            "frame": self.current_frame,
            "all_annotations": {
                frame_num: [self.clone_annotation(ann) for ann in anns]
                for frame_num, anns in self.frame_annotations.items()
            },
            "current_annotations": (
                [self.clone_annotation(ann) for ann in self.canvas.annotations]
                if self.canvas.annotations
                else []
            ),
            "class_colors": {
                class_name: QColor(color)
                for class_name, color in self.canvas.class_colors.items()
            },
            "class_attributes": (
                deepcopy(self.canvas.class_attributes)
                if hasattr(self.canvas, "class_attributes")
                else None
            ),
            "current_class": (
                self.canvas.current_class
                if hasattr(self.canvas, "current_class")
                else None
            ),
        }

        # Add current state to redo stack
        self.redo_stack.append(current_state)

        # Limit the size of the redo stack
        if len(self.redo_stack) > self.max_redo_steps:
            self.redo_stack.pop(0)

        # Get the last state
        last_state = self.undo_stack.pop()
        frame = last_state["frame"]

        # Restore class information if it exists
        if "class_colors" in last_state and last_state["class_colors"]:
            self.canvas.class_colors = last_state["class_colors"]

        if "class_attributes" in last_state and last_state["class_attributes"]:
            self.canvas.class_attributes = last_state["class_attributes"]

        if "current_class" in last_state and last_state["current_class"]:
            self.canvas.current_class = last_state["current_class"]

        # Restore all frame annotations if they exist
        if "all_annotations" in last_state and last_state["all_annotations"]:
            self.frame_annotations = last_state["all_annotations"]

        # If we're undoing a change on the current frame
        if frame == self.current_frame:
            # Restore the annotations for the current frame
            if "current_annotations" in last_state:
                self.canvas.annotations = last_state["current_annotations"]
            else:
                self.canvas.annotations = self.frame_annotations.get(frame, [])

            self.canvas.selected_annotation = None
            self.canvas.update()

            # Update annotation list
            self.update_annotation_list()

            # Update class UI
            self.refresh_class_ui()

            self.statusBar.showMessage("Undo successful", 3000)
        else:
            # If the undo is for a different frame, we need to navigate to that frame first
            self.statusBar.showMessage(
                f"Undo refers to frame {frame}, navigating there first", 3000
            )

            # Navigate to the frame with the undo state
            if hasattr(self, "is_image_dataset") and self.is_image_dataset:
                self.current_frame = frame
                self.frame_slider.setValue(frame)
                self.load_current_image()
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ret, frame_img = self.cap.read()
                if ret:
                    self.current_frame = frame
                    self.frame_slider.setValue(frame)
                    self.canvas.set_frame(frame_img)

            # Restore the annotations for this frame
            if "current_annotations" in last_state:
                self.canvas.annotations = last_state["current_annotations"]
            else:
                self.canvas.annotations = self.frame_annotations.get(frame, [])

            self.canvas.selected_annotation = None
            self.canvas.update()

            # Update annotation list
            self.update_annotation_list()

            # Update class UI
            self.refresh_class_ui()

            self.statusBar.showMessage("Undo successful", 3000)

    def save_undo_state_without_clearing_redo(self):
        """Save the current state for undo functionality without clearing the redo stack."""
        # Create a deep copy of all frame annotations
        all_frame_annotations = {}
        for frame_num, annotations in self.frame_annotations.items():
            all_frame_annotations[frame_num] = [
                self.clone_annotation(ann) for ann in annotations
            ]

        # Create a deep copy of class colors
        class_colors = {}
        for class_name, color in self.canvas.class_colors.items():
            class_colors[class_name] = QColor(color)

        # Create a deep copy of class attributes if they exist
        class_attributes = None
        if hasattr(self.canvas, "class_attributes"):
            class_attributes = deepcopy(self.canvas.class_attributes)

        # Save the state
        undo_state = {
            "frame": self.current_frame,
            "all_annotations": all_frame_annotations,
            "current_annotations": (
                [self.clone_annotation(ann) for ann in self.canvas.annotations]
                if self.canvas.annotations
                else []
            ),
            "class_colors": class_colors,
            "class_attributes": class_attributes,
            "current_class": (
                self.canvas.current_class
                if hasattr(self.canvas, "current_class")
                else None
            ),
        }

        # Add to undo stack
        self.undo_stack.append(undo_state)

        # Limit the size of the undo stack
        if len(self.undo_stack) > self.max_undo_steps:
            self.undo_stack.pop(0)

    def redo(self):
        """Redo the last undone action."""
        if not self.redo_stack:
            self.statusBar.showMessage("Nothing to redo", 3000)
            return

        # Save current state to undo stack before redoing
        self.save_undo_state_without_clearing_redo()

        # Get the last state from redo stack
        redo_state = self.redo_stack.pop()
        frame = redo_state["frame"]

        # Restore class information if it exists
        if "class_colors" in redo_state and redo_state["class_colors"]:
            self.canvas.class_colors = redo_state["class_colors"]

        if "class_attributes" in redo_state and redo_state["class_attributes"]:
            self.canvas.class_attributes = redo_state["class_attributes"]

        if "current_class" in redo_state and redo_state["current_class"]:
            self.canvas.current_class = redo_state["current_class"]

        # Restore all frame annotations if they exist
        if "all_annotations" in redo_state and redo_state["all_annotations"]:
            self.frame_annotations = redo_state["all_annotations"]

        # If we're redoing a change on the current frame
        if frame == self.current_frame:
            # Restore the annotations for the current frame
            if "current_annotations" in redo_state:
                self.canvas.annotations = redo_state["current_annotations"]
            else:
                self.canvas.annotations = self.frame_annotations.get(frame, [])

            self.canvas.selected_annotation = None
            self.canvas.update()

            # Update annotation list
            self.update_annotation_list()

            # Update class UI
            self.refresh_class_ui()

            self.statusBar.showMessage("Redo successful", 3000)
        else:
            # If the redo is for a different frame, we need to navigate to that frame first
            self.statusBar.showMessage(
                f"Redo refers to frame {frame}, navigating there first", 3000
            )

            # Navigate to the frame with the redo state
            if hasattr(self, "is_image_dataset") and self.is_image_dataset:
                self.current_frame = frame
                self.frame_slider.setValue(frame)
                self.load_current_image()
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ret, frame_img = self.cap.read()
                if ret:
                    self.current_frame = frame
                    self.frame_slider.setValue(frame)
                    self.canvas.set_frame(frame_img)

            # Restore the annotations for this frame
            if "current_annotations" in redo_state:
                self.canvas.annotations = redo_state["current_annotations"]
            else:
                self.canvas.annotations = self.frame_annotations.get(frame, [])

            self.canvas.selected_annotation = None
            self.canvas.update()

            # Update annotation list
            self.update_annotation_list()

            # Update class UI
            self.refresh_class_ui()

            self.statusBar.showMessage("Redo successful", 3000)

    #
    # Interpolation methods
    #
    # Add these methods to the VideoAnnotationTool class

    # Add these methods to the VideoAnnotationTool class
    def toggle_interpolation_mode(self):
        """Toggle interpolation mode on/off."""
        is_active = self.toggle_interpolation_action.isChecked()
        self.interpolation_manager.set_active(is_active)

        # Show/hide the interpolation toolbar
        if hasattr(self, "interpolation_toolbar"):
            self.interpolation_toolbar.setVisible(is_active)

        # Update UI
        self.update_frame_display()

    def set_interpolation_interval(self):
        """Open dialog to set interpolation interval."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Set Keyframe Interval")

        layout = QVBoxLayout(dialog)

        form_layout = QFormLayout()
        interval_spinner = QSpinBox()
        interval_spinner.setRange(2, 100)
        interval_spinner.setValue(self.interpolation_manager.interval)
        form_layout.addRow("Frames between keyframes:", interval_spinner)

        layout.addLayout(form_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() == QDialog.Accepted:
            new_interval = interval_spinner.value()
            self.interpolation_manager.set_interval(new_interval)

            # Update spinner in toolbar if it exists
            if hasattr(self, "interval_spinner"):
                self.interval_spinner.setValue(new_interval)

    def perform_interpolation(self):
        """Manually trigger interpolation between annotated frames."""
        if not self.interpolation_manager.is_active:
            QMessageBox.warning(
                self, "Interpolation", "Interpolation mode is not active."
            )
            return
        self.interpolation_manager.perform_pending_interpolation()

    def update_frame_display(self):
        """Update the frame display with interpolation indicators."""
        # Update indicator if interpolation is active
        if (
            hasattr(self, "interpolation_manager")
            and self.interpolation_manager.is_active
        ):
            # Check if this frame has annotations
            has_annotations = (
                self.current_frame in self.frame_annotations
                and len(self.frame_annotations[self.current_frame]) > 0
            )

            # Update indicator in toolbar
            if hasattr(self, "keyframe_indicator"):
                if has_annotations:
                    # This is an annotated frame (either manually or interpolated)
                    if (
                        self.current_frame
                        == self.interpolation_manager.last_annotated_frame
                    ):
                        # This is the last manually annotated frame
                        self.keyframe_indicator.setStyleSheet(
                            "background-color: #FF5555; min-width: 16px;"
                        )
                        self.keyframe_indicator.setToolTip(
                            "Current frame is manually annotated"
                        )
                    else:
                        # This is an interpolated frame or another manually annotated frame
                        self.keyframe_indicator.setStyleSheet(
                            "background-color: #55AAFF; min-width: 16px;"
                        )
                        self.keyframe_indicator.setToolTip(
                            "Current frame has annotations"
                        )
                else:
                    self.keyframe_indicator.setStyleSheet(
                        "background-color: transparent; min-width: 16px;"
                    )
                    self.keyframe_indicator.setToolTip(
                        "Current frame has no annotations"
                    )

            # Add visual indicator to frame display
            if hasattr(self, "canvas"):
                if has_annotations:
                    if (
                        self.current_frame
                        == self.interpolation_manager.last_annotated_frame
                    ):
                        self.canvas.setStyleSheet(
                            "border: 2px solid #FF5555;"
                        )  # Red for manually annotated
                    else:
                        self.canvas.setStyleSheet(
                            "border: 2px solid #55AAFF;"
                        )  # Blue for other annotations
                else:
                    self.canvas.setStyleSheet("")

    def set_current_frame(self, frame_num):
        """Set the current frame and update display."""
        # Check for pending interpolation before changing frames
        if (
            hasattr(self, "interpolation_manager")
            and self.interpolation_manager.is_active
        ):
            self.interpolation_manager.check_pending_interpolation(frame_num)

        # Original implementation for setting the current frame
        if not self.cap:
            return

        # Ensure frame_num is within valid range
        frame_num = max(0, min(frame_num, self.total_frames - 1))

        # Set the frame position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()

        if not ret:
            return

        # Update current frame
        self.current_frame = frame_num

        # Display the frame
        self.canvas.set_frame(frame)

        self.update_frame_info()

        self.load_current_frame_annotations()
        # Update timeline slider
        if hasattr(self, "timeline_slider"):
            self.timeline_slider.setValue(frame_num)

        # Update frame counter
        if hasattr(self, "frame_counter"):
            self.frame_counter.setText(f"Frame: {frame_num}/{self.total_frames-1}")

        # Load annotations for this frame if they exist
        if frame_num in self.frame_annotations:
            self.canvas.annotations = self.frame_annotations[frame_num].copy()
        else:
            self.canvas.annotations = []

        # Update annotation list
        self.update_annotation_list()

        # Update frame display with indicators
        self.update_frame_display()

    def update_frame_display(self):
        """Update the frame display with keyframe and interpolation indicators."""
        # Update keyframe indicator if interpolation is active
        if (
            hasattr(self, "interpolation_manager")
            and self.interpolation_manager.is_active
        ):
            is_keyframe = self.interpolation_manager.is_keyframe()

            # Update indicator in toolbar
            if hasattr(self, "keyframe_indicator"):
                if is_keyframe:
                    self.keyframe_indicator.setStyleSheet(
                        "background-color: #FF5555; min-width: 16px;"
                    )
                    self.keyframe_indicator.setToolTip("Current frame is a keyframe")
                else:
                    # Check if this is an interpolated frame
                    is_interpolated = (
                        self.current_frame in self.frame_annotations
                        and len(self.frame_annotations[self.current_frame]) > 0
                    )

                    if is_interpolated:
                        self.keyframe_indicator.setStyleSheet(
                            "background-color: #55AAFF; min-width: 16px;"
                        )
                        self.keyframe_indicator.setToolTip(
                            "Current frame has interpolated annotations"
                        )
                    else:
                        self.keyframe_indicator.setStyleSheet(
                            "background-color: transparent; min-width: 16px;"
                        )
                        self.keyframe_indicator.setToolTip(
                            "Current frame is not a keyframe"
                        )

            # Add visual indicator to frame display
            if hasattr(self, "canvas"):
                if is_keyframe:
                    self.canvas.setStyleSheet("border: 2px solid #FF5555;")
                elif (
                    self.current_frame in self.frame_annotations
                    and len(self.frame_annotations[self.current_frame]) > 0
                ):
                    self.canvas.setStyleSheet(
                        "border: 2px solid #55AAFF;"
                    )  # Blue for interpolated frames
                else:
                    self.canvas.setStyleSheet("")

    #
    # Verifying
    #
    def toggle_verification_mode(self):
        """Toggle verification mode for annotations."""
        if not hasattr(self, "verification_mode_enabled"):
            self.verification_mode_enabled = False
        
        self.verification_mode_enabled = not self.verification_mode_enabled
        
        if self.verification_mode_enabled:
            self.statusBar.showMessage("Verification mode enabled - unverified annotations will be deleted when changing frames", 5000)
        else:
            self.statusBar.showMessage("Verification mode disabled", 3000)
        
        # Update UI to show current mode
        if hasattr(self, "verify_mode_action"):
            self.verify_mode_action.setChecked(self.verification_mode_enabled)

    def verify_selected_annotation(self):
        """Mark the selected annotation as verified."""
        if hasattr(self.canvas, "selected_annotation") and self.canvas.selected_annotation:
            self.save_undo_state()
            self.canvas.selected_annotation.verified = True
            self.canvas.update()
            self.update_annotation_list()
            self.statusBar.showMessage("Annotation verified", 2000)

    def verify_all_annotations(self):
        """Mark all annotations in the current frame as verified."""
        if self.canvas.annotations:
            self.save_undo_state()
            for annotation in self.canvas.annotations:
                annotation.verified = True
            self.canvas.update()
            self.update_annotation_list()
            self.statusBar.showMessage(f"All {len(self.canvas.annotations)} annotations verified", 2000)

    def handle_unverified_annotations(self):
        """Handle unverified annotations when changing frames."""
        if not hasattr(self, "verification_mode_enabled") or not self.verification_mode_enabled:
            return
        
        # Check if there are any unverified annotations in the current frame
        if self.current_frame in self.frame_annotations:
            unverified = [ann for ann in self.frame_annotations[self.current_frame] if not getattr(ann, 'verified', False)]
            
            if unverified:
                # Remove unverified annotations
                self.frame_annotations[self.current_frame] = [
                    ann for ann in self.frame_annotations[self.current_frame] 
                    if getattr(ann, 'verified', False)
                ]
                
                # Update canvas annotations
                self.canvas.annotations = self.frame_annotations[self.current_frame].copy()
                self.canvas.update()
                
                # Show message
                self.statusBar.showMessage(f"Removed {len(unverified)} unverified annotations", 3000)

    def toggle_pan_mode(self):
        """Toggle pan mode for the canvas"""
        enabled = self.pan_tool_action.isChecked()
        self.canvas.set_pan_mode(enabled)
        
        # Update cursor based on mode
        if enabled:
            self.canvas.setCursor(Qt.OpenHandCursor)
            self.statusBar.showMessage("Pan mode enabled. Left-click and drag to pan the canvas.", 3000)
        else:
            self.canvas.setCursor(Qt.ArrowCursor)
            self.statusBar.showMessage("Pan mode disabled.", 3000)
