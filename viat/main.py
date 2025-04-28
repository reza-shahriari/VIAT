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
)
from PyQt5.QtCore import Qt, QTimer, QRect, QDateTime, QEvent
from PyQt5.QtGui import QColor, QIcon, QImage, QPixmap
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .canvas import VideoCanvas
from .annotation import BoundingBox
from .widgets import AnnotationDock, StyleManager, ClassDock, AnnotationToolbar
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
    UICreator
)
from utils.icon_provider import IconProvider
from natsort import natsorted


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

        # Install event filter to handle global shortcuts
        QApplication.instance().installEventFilter(self)

        # Load last project if available
        QTimer.singleShot(100, self.load_last_project)

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
        self.frame_hashes = {}  # Maps frame number to its hash
        self.styles = {}
        self.icon_provider = IconProvider()

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
        self.cap = cv2.VideoCapture(filename)

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
        if hasattr(self, "is_image_dataset") and self.is_image_dataset:
            if self.current_frame > 0:
                self.current_frame -= 1
                self.load_current_image()
                self.update_frame_info()
        elif self.cap and self.cap.isOpened():
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
        """Go to the next frame in the video or next image in the dataset."""

        if hasattr(self, "is_image_dataset") and self.is_image_dataset:
            if self.current_frame < len(self.image_files) - 1:
                self.current_frame += 1
                self.load_current_image()
                self.update_frame_info()
                self.load_current_frame_annotations()
            else:
                if self.is_playing:
                    self.current_frame = 0
                    self.load_current_image()
                    self.update_frame_info()
                    self.load_current_frame_annotations()
                    self.statusBar.showMessage("Looping back to start of image dataset")
                else:
                    # Just show message if not playing
                    self.statusBar.showMessage("End of image dataset")
        elif self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame += 1
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
            interval = int(1000 / (fps * self.playback_speed))
            self.play_timer.start(interval)
            self.is_playing = True
            self.play_button.setIcon(
                self.icon_provider.get_icon("media-playback-pause")
            )
            self.statusBar.showMessage("Playing")

    def eventFilter(self, obj, event):
        """Global event filter to handle shortcuts regardless of focus."""
        if event.type() == QEvent.KeyPress:
            # Handle arrow keys for frame navigation
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
            )

            self.project_file = filename
            self.project_modified = False
            self.statusBar.showMessage(f"Project saved to {os.path.basename(filename)}")

            # Update recent projects menu
            self.update_recent_projects_menu()

            # Save application state
            self.save_application_state()

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
                self.update_annotation_list()
                self.toolbar.update_class_selector()
                self.class_dock.update_class_list()

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

    # Add these methods to save and load application state
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

    #
    # Annotation handling methods
    #

    def update_annotation_list(self):
        """Update the annotations list widget."""
        self.annotation_dock.update_annotation_list()

        # If duplicate frame detection is enabled, propagate annotations to duplicates
        if self.duplicate_frames_enabled and self.current_frame in self.frame_hashes:
            current_hash = self.frame_hashes[self.current_frame]
            if (
                current_hash in self.duplicate_frames_cache
                and len(self.duplicate_frames_cache[current_hash]) > 1
            ):
                # Propagate current frame annotations to all duplicates
                self.propagate_to_duplicate_frames(current_hash)

        self.perform_autosave()

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

            # Get default attributes
            default_attributes = {"Size": -1, "Quality": -1}

            # Check if we should use attributes from previous annotations
            if (
                hasattr(self, "use_previous_attributes")
                and self.use_previous_attributes
            ):
                prev_attributes = self.get_previous_annotation_attributes(class_name)
                if prev_attributes:
                    default_attributes = prev_attributes

            bbox = BoundingBox(rect, class_name, default_attributes, color)

            # Add to annotations
            self.canvas.annotations.append(bbox)
            self.canvas.selected_annotation = bbox

            # Update frame_annotations dictionary
            self.frame_annotations[self.current_frame] = self.canvas.annotations

            # Show attribute dialog if enabled
            if (
                hasattr(self, "auto_show_attribute_dialog")
                and self.auto_show_attribute_dialog
            ):
                self.edit_annotation(bbox, focus_first_field=True)

            # Update UI
            self.update_annotation_list()
            self.canvas.update()

    def clear_annotations(self):
        """Clear all annotations."""
        if not self.canvas.annotations:
            return

        reply = QMessageBox.question(
            self,
            "Clear Annotations",
            "Are you sure you want to clear all annotations?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
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
            attributes = {"Size": size_spin.value(), "Quality": quality_spin.value()}

            # Create rectangle
            rect = QRect(
                x_spin.value(), y_spin.value(), width_spin.value(), height_spin.value()
            )

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
        width_spin.setRange(
            5, self.canvas.pixmap.width() if self.canvas.pixmap else 1000
        )
        height_spin = QSpinBox()
        height_spin.setRange(
            5, self.canvas.pixmap.height() if self.canvas.pixmap else 1000
        )

        coords_layout.addRow("X:", x_spin)
        coords_layout.addRow("Y:", y_spin)
        coords_layout.addRow("Width:", width_spin)
        coords_layout.addRow("Height:", height_spin)

        # Attributes
        attributes_layout = QFormLayout()

        # Get default values from previous annotations if enabled
        default_size = -1
        default_quality = -1

        if hasattr(self, "use_previous_attributes") and self.use_previous_attributes:
            # Get the current selected class
            current_class = class_combo.currentText()
            prev_attributes = self.get_previous_annotation_attributes(current_class)

            if prev_attributes:
                default_size = prev_attributes.get("Size", -1)
                default_quality = prev_attributes.get("Quality", -1)

        # Create attribute spinboxes with default values
        size_spin = QSpinBox()
        size_spin.setRange(0, 100)
        size_spin.setValue(default_size)  # Use default or previous value

        quality_spin = QSpinBox()
        quality_spin.setRange(0, 100)
        quality_spin.setValue(default_quality)  # Use default or previous value

        attributes_layout.addRow("Size (0-100):", size_spin)
        attributes_layout.addRow("Quality (0-100):", quality_spin)

        # Update attributes when class changes
        def update_attributes_for_class(class_name):
            if (
                hasattr(self, "use_previous_attributes")
                and self.use_previous_attributes
            ):
                prev_attributes = self.get_previous_annotation_attributes(class_name)
                if prev_attributes:
                    size_spin.setValue(prev_attributes.get("Size", -1))
                    quality_spin.setValue(prev_attributes.get("Quality", -1))

        class_combo.currentTextChanged.connect(update_attributes_for_class)

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

        # Set tab order for easy navigation
        dialog.setTabOrder(class_combo, x_spin)
        dialog.setTabOrder(x_spin, y_spin)
        dialog.setTabOrder(y_spin, width_spin)
        dialog.setTabOrder(width_spin, height_spin)
        dialog.setTabOrder(height_spin, size_spin)
        dialog.setTabOrder(size_spin, quality_spin)

        # Focus on the first field
        class_combo.setFocus()

        return dialog

    def parse_attributes(self, text):
        """Parse attributes from text input."""
        attributes = {}
        for line in text.strip().split("\n"):
            if "=" in line:
                key, value = line.split("=", 1)
                attributes[key.strip()] = value.strip()
        return attributes

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
        # Toggle annotation method with 'M' key
        elif event.key() == Qt.Key_M:
            current_index = self.method_selector.currentIndex()
            new_index = (current_index + 1) % self.method_selector.count()
            self.method_selector.setCurrentIndex(new_index)
        # Batch edit annotations with 'B' key
        elif event.key() == Qt.Key_B:
            if hasattr(self, "annotation_dock"):
                self.annotation_dock.batch_edit_annotations()
        # Toggle smart edge with 'E' key
        elif event.key() == Qt.Key_E:
            if hasattr(self, "smart_edge_action"):
                self.smart_edge_action.setChecked(
                    not self.smart_edge_action.isChecked()
                )
                self.toggle_smart_edge()
        # Toggle attribute dialog with 'A' key
        elif event.key() == Qt.Key_A and (event.modifiers() & Qt.ControlModifier):
            self.auto_show_attribute_dialog = not self.auto_show_attribute_dialog
            self.statusBar.showMessage(
                f"Attribute dialog for new annotations {'enabled' if self.auto_show_attribute_dialog else 'disabled'}",
                3000,
            )
            # Update menu if it exists
            for action in self.settings_menu.actions():
                if action.text() == "Show Attribute Dialog for New Annotations":
                    action.setChecked(self.auto_show_attribute_dialog)
                    break
        # Propagate annotations with 'P' key
        elif event.key() == Qt.Key_P and (event.modifiers() & Qt.ControlModifier):
            self.propagate_annotations()
        else:
            super().keyPressEvent(event)

    def edit_annotation(self, annotation, focus_first_field=False):
        """
        Edit the properties of an annotation.

        Args:
            annotation: The annotation to edit
            focus_first_field: Whether to focus on the first attribute field
        """
        if not annotation:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Annotation")
        dialog.setMinimumWidth(350)

        layout = QVBoxLayout()
        form_layout = QFormLayout()

        # Class selector
        class_combo = QComboBox()
        class_combo.addItems(self.canvas.class_colors.keys())
        class_combo.setCurrentText(annotation.class_name)
        form_layout.addRow("Class:", class_combo)

        # Get class attribute configuration if available
        class_attributes = {}
        if hasattr(self.canvas, "class_attributes"):
            class_attributes = self.canvas.class_attributes.get(
                annotation.class_name, {}
            )

        # Create input widgets for all attributes
        attribute_widgets = {}
        first_widget = None

        for attr_name, attr_value in sorted(annotation.attributes.items()):
            # Get attribute type from class configuration or infer from value
            attr_type = "string"
            attr_min = None
            attr_max = None

            if attr_name in class_attributes:
                attr_config = class_attributes[attr_name]
                attr_type = attr_config.get("type", "string")
                attr_min = attr_config.get("min", None)
                attr_max = attr_config.get("max", None)
            elif isinstance(attr_value, int):
                attr_type = "int"
            elif isinstance(attr_value, float):
                attr_type = "float"
            elif isinstance(attr_value, bool):
                attr_type = "boolean"

            # Create appropriate input widget based on type
            if attr_type == "boolean":
                input_widget = QComboBox()
                input_widget.addItems(["False", "True"])
                input_widget.setCurrentText(str(bool(attr_value)))
            elif attr_type == "int":
                input_widget = QSpinBox()
                if attr_min is not None:
                    input_widget.setMinimum(attr_min)
                else:
                    input_widget.setMinimum(-999999)
                if attr_max is not None:
                    input_widget.setMaximum(attr_max)
                else:
                    input_widget.setMaximum(999999)
                input_widget.setValue(int(attr_value))
            elif attr_type == "float":
                input_widget = QDoubleSpinBox()
                if attr_min is not None:
                    input_widget.setMinimum(attr_min)
                else:
                    input_widget.setMinimum(-999999.0)
                if attr_max is not None:
                    input_widget.setMaximum(attr_max)
                else:
                    input_widget.setMaximum(999999.0)
                input_widget.setValue(float(attr_value))
                input_widget.setDecimals(2)
            else:  # string or default
                input_widget = QLineEdit()
                input_widget.setText(str(attr_value))

            # Store the first widget for focus
            if first_widget is None and attr_name in ["Size", "Quality"]:
                first_widget = input_widget

            form_layout.addRow(f"{attr_name}:", input_widget)
            attribute_widgets[attr_name] = input_widget

        layout.addLayout(form_layout)

        # Add OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        # Get direct references to the OK and Cancel buttons
        ok_button = button_box.button(QDialogButtonBox.Ok)
        cancel_button = button_box.button(QDialogButtonBox.Cancel)

        dialog.setLayout(layout)

        # Set focus on the first attribute field if requested
        if focus_first_field and first_widget:
            # Use singleShot timer to ensure focus happens after dialog is shown
            QTimer.singleShot(0, lambda: first_widget.setFocus())
            # For QSpinBox and QDoubleSpinBox, select all text
            if isinstance(first_widget, (QSpinBox, QDoubleSpinBox)):
                QTimer.singleShot(0, lambda: first_widget.selectAll())
            # For QLineEdit, select all text
            elif isinstance(first_widget, QLineEdit):
                QTimer.singleShot(0, lambda: first_widget.selectAll())

        # Set proper tab order
        previous_widget = class_combo
        for attr_name, widget in attribute_widgets.items():
            dialog.setTabOrder(previous_widget, widget)
            previous_widget = widget

        # Make sure the last tab goes to the OK button first, then Cancel
        dialog.setTabOrder(previous_widget, ok_button)
        dialog.setTabOrder(ok_button, cancel_button)

        # If dialog is accepted, update the annotation
        if dialog.exec_() == QDialog.Accepted:
            old_class = annotation.class_name
            new_class = class_combo.currentText()

            # Update class and color
            annotation.class_name = new_class
            annotation.color = self.canvas.class_colors[new_class]

            # If class changed, update attributes based on new class configuration
            if old_class != new_class and hasattr(self.canvas, "class_attributes"):
                new_class_attributes = self.canvas.class_attributes.get(new_class, {})
                self.update_annotation_attributes(annotation, new_class_attributes)

            # Update attribute values from the dialog
            for attr_name, widget in attribute_widgets.items():
                if attr_name in annotation.attributes:
                    if isinstance(widget, QComboBox):  # Boolean
                        annotation.attributes[attr_name] = (
                            widget.currentText() == "True"
                        )
                    elif isinstance(widget, QSpinBox):  # Int
                        annotation.attributes[attr_name] = widget.value()
                    elif isinstance(widget, QDoubleSpinBox):  # Float
                        annotation.attributes[attr_name] = widget.value()
                    else:  # String or other
                        annotation.attributes[attr_name] = widget.text()

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
        if (
            hasattr(self.canvas, "selected_annotation")
            and self.canvas.selected_annotation
        ):
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

    def import_annotations(self, filename):
        """
        Import annotations from a file.

        Args:
            filename (str): Path to the annotation file
        """
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
            from utils.file_operations import import_annotations

            format_type, annotations, imported_frame_annotations = import_annotations(
                filename, BoundingBox, image_width, image_height, self.class_colors
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
    # Class handling methods
    #

    def add_class(self):
        """Add a new class with custom attributes."""
        # Create dialog
        dialog = self.create_class_dialog()

        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            class_name = dialog.name_edit.text().strip()

            if not class_name:
                QMessageBox.warning(self, "Add Class", "Class name cannot be empty!")
                return

            if class_name in self.canvas.class_colors:
                QMessageBox.warning(
                    self, "Add Class", f"Class '{class_name}' already exists!"
                )
                return

            # Get color from dialog
            color = dialog.color

            # Process attributes
            attributes_config = {}
            for (
                _,
                name_edit,
                type_combo,
                default_edit,
                min_edit,
                max_edit,
            ) in dialog.attribute_widgets:
                attr_name = name_edit.text().strip()
                if attr_name:
                    attr_type = type_combo.currentText()

                    # Parse default value based on type
                    default_value = default_edit.text()
                    if attr_type == "int":
                        try:
                            default_value = int(default_value)
                        except ValueError:
                            default_value = 0
                    elif attr_type == "float":
                        try:
                            default_value = float(default_value)
                        except ValueError:
                            default_value = 0.0
                    elif attr_type == "boolean":
                        default_value = default_value.lower() in ["true", "1", "yes"]

                    # Parse min/max for numeric types
                    attr_config = {"type": attr_type, "default": default_value}

                    if attr_type in ["int", "float"]:
                        try:
                            attr_config["min"] = (
                                int(min_edit.text())
                                if attr_type == "int"
                                else float(min_edit.text())
                            )
                        except ValueError:
                            attr_config["min"] = 0

                        try:
                            attr_config["max"] = (
                                int(max_edit.text())
                                if attr_type == "int"
                                else float(max_edit.text())
                            )
                        except ValueError:
                            attr_config["max"] = 100

                    attributes_config[attr_name] = attr_config

            # Add class to canvas
            self.canvas.class_colors[class_name] = color

            # Store attributes configuration
            if not hasattr(self.canvas, "class_attributes"):
                self.canvas.class_attributes = {}
            self.canvas.class_attributes[class_name] = attributes_config

            # Update UI
            self.toolbar.update_class_selector()
            self.class_dock.update_class_list()

            # Set as current class
            self.canvas.set_current_class(class_name)
            self.class_selector.setCurrentText(class_name)

    def update_annotation_attributes(self, annotation, attributes_config):
        """Update annotation attributes based on class attribute configuration."""
        # Create a new attributes dictionary with defaults from the configuration
        new_attributes = {}

        # First, set defaults from configuration
        for attr_name, attr_config in attributes_config.items():
            new_attributes[attr_name] = attr_config["default"]

        # Then, preserve existing values where possible
        for attr_name, attr_value in annotation.attributes.items():
            if attr_name in new_attributes:
                # Keep existing value if it's within constraints
                if attr_name in attributes_config:
                    attr_type = attributes_config[attr_name]["type"]

                    if attr_type == "int":
                        try:
                            value = int(attr_value)
                            min_val = attributes_config[attr_name].get(
                                "min", float("-inf")
                            )
                            max_val = attributes_config[attr_name].get(
                                "max", float("inf")
                            )
                            if min_val <= value <= max_val:
                                new_attributes[attr_name] = value
                        except (ValueError, TypeError):
                            pass
                    elif attr_type == "float":
                        try:
                            value = float(attr_value)
                            min_val = attributes_config[attr_name].get(
                                "min", float("-inf")
                            )
                            max_val = attributes_config[attr_name].get(
                                "max", float("inf")
                            )
                            if min_val <= value <= max_val:
                                new_attributes[attr_name] = value
                        except (ValueError, TypeError):
                            pass
                    elif attr_type == "string":
                        new_attributes[attr_name] = str(attr_value)
                    elif attr_type == "boolean":
                        if isinstance(attr_value, bool):
                            new_attributes[attr_name] = attr_value
                        else:
                            try:
                                new_attributes[attr_name] = str(attr_value).lower() in [
                                    "true",
                                    "1",
                                    "yes",
                                ]
                            except (ValueError, TypeError):
                                pass

        # Update the annotation's attributes
        annotation.attributes = new_attributes

    def edit_selected_class(self):
        """Edit the selected class with custom attributes."""
        item = self.class_dock.classes_list.currentItem()
        if not item:
            return

        class_name = item.text()
        old_name = class_name  # Store the original class name
        current_color = self.canvas.class_colors.get(class_name, QColor(255, 0, 0))

        # Get current attributes configuration
        attributes = getattr(self.canvas, "class_attributes", {}).get(class_name, {})

        # Create dialog
        dialog = self.create_class_dialog(class_name, current_color, attributes)

        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            new_class_name = dialog.name_edit.text().strip()

            if not new_class_name:
                QMessageBox.warning(self, "Edit Class", "Class name cannot be empty!")
                return

            if (
                new_class_name != old_name
                and new_class_name in self.canvas.class_colors
            ):
                # Ask if user wants to merge classes
                reply = QMessageBox.question(
                    self,
                    "Merge Classes",
                    f"Class '{new_class_name}' already exists. Do you want to convert all '{old_name}' annotations to '{new_class_name}'?",
                    QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                    QMessageBox.No,
                )

                if reply == QMessageBox.Cancel:
                    return
                elif reply == QMessageBox.Yes:
                    # Get color from existing class
                    color = self.canvas.class_colors[new_class_name]
                    # Convert all annotations of old class to new class
                    self.convert_class(old_name, new_class_name)
                    # Remove old class
                    del self.canvas.class_colors[old_name]
                    if (
                        hasattr(self.canvas, "class_attributes")
                        and old_name in self.canvas.class_attributes
                    ):
                        del self.canvas.class_attributes[old_name]
                    # Update UI
                    self.toolbar.update_class_selector()
                    self.class_dock.update_class_list()
                    self.update_annotation_list()
                    self.canvas.update()
                    return
                else:
                    # User chose No, so don't proceed with the rename
                    return

            # Get color from dialog
            color = dialog.color

            # Process attributes
            attributes_config = {}
            for (
                _,
                name_edit,
                type_combo,
                default_edit,
                min_edit,
                max_edit,
            ) in dialog.attribute_widgets:
                attr_name = name_edit.text().strip()
                if attr_name:
                    attr_type = type_combo.currentText()

                    # Parse default value based on type
                    default_value = default_edit.text()
                    if attr_type == "int":
                        try:
                            default_value = int(default_value)
                        except ValueError:
                            default_value = 0
                    elif attr_type == "float":
                        try:
                            default_value = float(default_value)
                        except ValueError:
                            default_value = 0.0
                    elif attr_type == "boolean":
                        default_value = default_value.lower() in ["true", "1", "yes"]

                    # Parse min/max for numeric types
                    attr_config = {"type": attr_type, "default": default_value}

                    if attr_type in ["int", "float"]:
                        try:
                            attr_config["min"] = (
                                int(min_edit.text())
                                if attr_type == "int"
                                else float(min_edit.text())
                            )
                        except ValueError:
                            attr_config["min"] = 0

                        try:
                            attr_config["max"] = (
                                int(max_edit.text())
                                if attr_type == "int"
                                else float(max_edit.text())
                            )
                        except ValueError:
                            attr_config["max"] = 100

                    attributes_config[attr_name] = attr_config

            # Update class
            if not hasattr(self.canvas, "class_attributes"):
                self.canvas.class_attributes = {}

            # Update class name in annotations and class attributes
            if old_name != new_class_name:
                # Update class colors dictionary
                self.canvas.class_colors[new_class_name] = color
                del self.canvas.class_colors[old_name]

                # Update class attributes dictionary
                self.canvas.class_attributes[new_class_name] = attributes_config
                if old_name in self.canvas.class_attributes:
                    del self.canvas.class_attributes[old_name]

                # Update annotations
                for annotation in self.canvas.annotations:
                    if annotation.class_name == old_name:
                        annotation.class_name = new_class_name
                        annotation.color = color

                        # Update attributes based on new configuration
                        self.update_annotation_attributes(annotation, attributes_config)
            else:
                # Just update the color and attributes
                self.canvas.class_colors[class_name] = color
                self.canvas.class_attributes[class_name] = attributes_config

                # Update annotations with new color and attributes
                for annotation in self.canvas.annotations:
                    if annotation.class_name == class_name:
                        annotation.color = color
                        self.update_annotation_attributes(annotation, attributes_config)

            # Update UI
            self.toolbar.update_class_selector()
            self.class_dock.update_class_list()
            self.update_annotation_list()
            self.canvas.update()

    def create_class_dialog(self, class_name=None, color=None, attributes=None):
        """Create a dialog for adding or editing classes with custom attributes."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Add Class" if class_name is None else "Edit Class")
        dialog.setMinimumWidth(400)

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
            color = QColor(
                random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            )
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

        # Attributes section
        attributes_group = QGroupBox("Attributes")
        attributes_layout = QVBoxLayout(attributes_group)

        # List to store attribute widgets
        attribute_widgets = []

        # Function to add a new attribute row
        def add_attribute_row(
            name="", attr_type="int", default_value="0", min_value="0", max_value="100"
        ):
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)

            name_edit = QLineEdit(name)
            name_edit.setPlaceholderText("Attribute Name")

            type_combo = QComboBox()
            type_combo.addItems(["int", "float", "string", "boolean"])
            type_combo.setCurrentText(attr_type)

            default_edit = QLineEdit(default_value)
            default_edit.setPlaceholderText("Default")

            min_edit = QLineEdit(min_value)
            min_edit.setPlaceholderText("Min")

            max_edit = QLineEdit(max_value)
            max_edit.setPlaceholderText("Max")

            delete_btn = QPushButton("X")
            delete_btn.setMaximumWidth(30)
            delete_btn.clicked.connect(lambda: remove_attribute_row(row_widget))

            row_layout.addWidget(name_edit)
            row_layout.addWidget(type_combo)
            row_layout.addWidget(default_edit)
            row_layout.addWidget(min_edit)
            row_layout.addWidget(max_edit)
            row_layout.addWidget(delete_btn)

            attributes_layout.addWidget(row_widget)
            attribute_widgets.append(
                (row_widget, name_edit, type_combo, default_edit, min_edit, max_edit)
            )

            # Update type-dependent visibility
            def update_type_visibility():
                is_numeric = type_combo.currentText() in ["int", "float"]
                min_edit.setVisible(is_numeric)
                max_edit.setVisible(is_numeric)

            type_combo.currentTextChanged.connect(update_type_visibility)
            update_type_visibility()

            return row_widget

        def remove_attribute_row(row_widget):
            for widget, *_ in attribute_widgets[:]:
                if widget == row_widget:
                    attributes_layout.removeWidget(widget)
                    widget.deleteLater()
                    attribute_widgets.remove((widget, *_))
                    break

        # Add button for attributes
        add_attr_btn = QPushButton("Add Attribute")
        add_attr_btn.clicked.connect(lambda: add_attribute_row())
        attributes_layout.addWidget(add_attr_btn)

        # Add default attributes if editing an existing class
        if attributes:
            for attr_name, attr_info in attributes.items():
                add_attribute_row(
                    name=attr_name,
                    attr_type=attr_info.get("type", "int"),
                    default_value=str(attr_info.get("default", "0")),
                    min_value=str(attr_info.get("min", "0")),
                    max_value=str(attr_info.get("max", "100")),
                )
        else:
            # Add default Size and Quality attributes for new classes
            add_attribute_row("Size", "int", "-1", "0", "100")
            add_attribute_row("Quality", "int", "-1", "0", "100")

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        # Add widgets to layout
        layout.addLayout(name_layout)
        layout.addLayout(color_layout)
        layout.addWidget(attributes_group)
        layout.addWidget(buttons)

        # Store attribute widgets for access when dialog is accepted
        dialog.attribute_widgets = attribute_widgets
        dialog.name_edit = name_edit
        dialog.color = color

        return dialog

    def convert_class(self, old_class, new_class):
        """Convert all annotations from one class to another."""
        # Update annotations in current frame
        for annotation in self.canvas.annotations:
            if annotation.class_name == old_class:
                annotation.class_name = new_class
                annotation.color = self.canvas.class_colors[new_class]

        # Update annotations in all frames
        for frame_num, annotations in self.frame_annotations.items():
            for annotation in annotations:
                if annotation.class_name == old_class:
                    annotation.class_name = new_class
                    annotation.color = self.canvas.class_colors[new_class]

        self.statusBar.showMessage(
            f"Converted all '{old_class}' annotations to '{new_class}'"
        )

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
        in_use = any(
            annotation.class_name == class_name
            for annotation in self.canvas.annotations
        )

        message = f"Are you sure you want to delete the class '{class_name}'?"
        if in_use:
            message += "\n\nThis class is currently in use by annotations. Deleting it will remove all annotations of this class."

        reply = QMessageBox.question(
            self,
            "Delete Class",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            # Remove annotations of this class
            self.canvas.annotations = [
                a for a in self.canvas.annotations if a.class_name != class_name
            ]

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

    # Add this method to clear recent projects
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
        from copy import deepcopy

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
