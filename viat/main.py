"""
Video Annotation Tool (VAT) - Main Application

This module contains the main application window and program entry point for the
Video Annotation Tool. It provides the UI framework and coordinates between the
different components of the application.
"""

import os
import random
import math
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
from .logger import VIATLogger, log_exceptions
from .tracking.nossort import NOCSORT

logger = VIATLogger()


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
    export_dataset_dialog,
    export_dataset,
    import_dataset_dialog,
    load_dataset,
    PerfomanceManger,
    load_project_with_backup,
    backup_before_save,
)
# --- New dataset & label-format modules (patch2) ---
from utils.dataset_manager import (
    detect_folder_type as _viat_detect_folder_type,
    scan_dataset,
    load_dataset_into_app,
    DatasetInfo,
    SplitInfo,
)
from utils.dataset_ops import (
    remove_bad_frames as _viat_remove_bad_frames,
    remap_class as _viat_remap_class,
    merge_classes as _viat_merge_classes,
)
# --- New dataset ops + logging (patch3) ---
from utils.dataset_ops import (
    move_frames_to as _viat_move_frames_to,
    remove_bad_frames as _viat_remove_bad_frames_v2,
    move_to_removed as _viat_move_to_removed,
    move_to_review_label as _viat_move_to_review_label,
    remove_grayscale_images as _viat_remove_grayscale,
    remove_duplicate_groups as _viat_remove_dup_groups,
    remove_class_and_images as _viat_remove_class_and_images,
    remap_class as _viat_remap_class_v2,
    merge_classes as _viat_merge_classes_v2,
    auto_import_detections as _viat_auto_import_detections,
)
from utils.dataset_log import (
    init_dataset_log as _viat_init_dataset_log,
    append_dataset_log as _viat_append_dataset_log,
)
# --- Video annotation features (patch4) ---
from utils.dataset_manager import load_viat_json_for_video as _viat_load_json_video
from utils.video_border import detect_and_adjust_borders as _viat_detect_adjust_borders
from utils.video_border import detect_video_borders as _viat_detect_borders
from utils.object_visibility import ObjectVisibilityManager as _ViatObjectVisibilityManager
# --- Performance + segmentation video (patch6) ---
from utils.performance import PerformanceManager as _ViatPerformanceManager
from utils.seg_video_labeler import SegmentationVideoLabeler as _ViatSegLabeler
# --- Dataset merger + toolbar visibility + click-pick (patch10) ---
from utils.dataset_merger import (
    merge_dataset_into_target as _viat_merge_dataset,
    find_unmatched_classes as _viat_find_unmatched_classes,
)
from utils.icon_provider import IconProvider
from natsort import natsorted
from copy import deepcopy
from pathlib import Path

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

        self.dark_mode_enabled = False

        # Load last project if available
        QTimer.singleShot(100, self.load_last_project)

    def toggle_dark_mode(self, enabled: bool):
        self.dark_mode_enabled = enabled
        if enabled:
            try:
                import os
                style_path = os.path.join(os.path.dirname(__file__), "styles.qss")
                with open(style_path, "r") as f:
                    self.setStyleSheet(f.read())
            except Exception as e:
                print(f"Failed to load stylesheet: {e}")
        else:
            self.setStyleSheet("")


    # -------------------------------------------------------------------------
    # Initialization and Setup Methods
    # -------------------------------------------------------------------------
    @log_exceptions
    def init_managers(self):
        self.annotation_manager = AnnotationManager(self, self.canvas)
        self.class_manager = ClassManager(self)
        self.interpolation_manager = InterpolationManager(self)
        self.performance_manager = PerfomanceManger()
        # Frame cache + fast seek (patch6)
        self.viat_perf = _ViatPerformanceManager(self, cache_capacity=60)

    @log_exceptions
    def load_last_project(self):
        """Load the last project that was open."""
        # Try to load from application state first
        if self.load_application_state():
            return

        # If that fails, try to get the most recent project
        last_project = get_last_project()
        if last_project and os.path.exists(last_project):
            self.load_project(last_project)

    @log_exceptions
    def init_properties(self):
        """Initialize the application properties and state variables."""
        # Available styles
        self.duplicate_frames_enabled = True
        self.duplicate_frames_cache = {}  # Maps frame hash to list of frame numbers
        self.frame_hashes = {}
        self.integration_mode = False
        self.integration_main_dataset = ""
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
        self._annotations_imported = set()
        self.last_autosave_time = None
        self.tracking_mode_enabled = False
        self.verification_mode = False
        # When True, unverified annotations are removed when navigating away from
        # a frame. Default OFF (keep unverified labels) per user request.
        self.remove_unverified = False
        # Per-object visibility editing mode (patch4)
        self.object_visibility_manager = None
        # Performance manager (frame cache + fast seek)
        self.performance_manager = None
        # Segmentation video labeler
        self.seg_labeler = None
        # Add image dataset flag
        self.is_image_dataset = False
        self.image_files = []

    @log_exceptions
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

        # Integration Mode Banner (hidden by default)
        self.integration_banner = QWidget()
        banner_layout = QHBoxLayout(self.integration_banner)
        banner_layout.setContentsMargins(5, 5, 5, 5)
        self.integration_label = QLabel("<b>Integration Mode Active:</b> Please review the dataset and apply labels.")
        self.integration_label.setStyleSheet("color: #E6A23C;")
        self.btn_finish_integration = QPushButton("Finish & Merge")
        self.btn_finish_integration.setStyleSheet("background-color: #67C23A; color: white; font-weight: bold; padding: 5px;")
        self.btn_finish_integration.clicked.connect(self.viat_finish_integration)
        banner_layout.addWidget(self.integration_label)
        banner_layout.addStretch()
        banner_layout.addWidget(self.btn_finish_integration)
        self.integration_banner.setVisible(False)
        layout.addWidget(self.integration_banner)

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

        # Inject extra menu items for new features (patches 2-5)
        self.viat_setup_extra_menus()

    @log_exceptions
    def viat_setup_extra_menus(self):
        """Inject menu items for all new VIAT features into the menu bar.

        This is called from setup_ui() AFTER UICreator has built the
        existing menus, so we just add a new 'Dataset' menu and items to
        the 'Edit' and 'View' menus without touching UICreator.
        """
        menubar = self.menuBar()

        # --- Dataset menu (new top-level menu) ---
        dataset_menu = menubar.addMenu("&Dataset")

        # Image dataset operations
        img_menu = dataset_menu.addMenu("Image Dataset Operations")

        act = img_menu.addAction("Remove Current Image")
        act.setShortcut("Shift+X")
        act.triggered.connect(lambda: self.viat_move_current_to_removed())

        act = img_menu.addAction("Move to Review Label (CHANGE LABEL)")
        act.setShortcut("Shift+R")
        act.triggered.connect(lambda: self.viat_move_current_to_review_label())

        img_menu.addSeparator()

        act = img_menu.addAction("Remove Grayscale Images...")
        act.triggered.connect(lambda: self.viat_remove_grayscale())

        act = img_menu.addAction("Remove Roboflow Duplicates...")
        act.triggered.connect(lambda: self.viat_remove_duplicates())

        act = img_menu.addAction("Remove Class + Images...")
        act.triggered.connect(lambda: self.viat_remove_class_and_images_dialog())

        img_menu.addSeparator()

        act = img_menu.addAction("Remove Bad Frames (batch)...")
        act.triggered.connect(lambda: self.remove_bad_frames_dialog())

        act = img_menu.addAction("Remap Class...")
        act.triggered.connect(lambda: self.remap_class_dialog())

        act = img_menu.addAction("Merge Classes...")
        act.triggered.connect(lambda: self.merge_classes_dialog())

        img_menu.addSeparator()

        act = img_menu.addAction("Dataset Statistics...")
        act.triggered.connect(lambda: self.viat_dataset_stats())

        act = img_menu.addAction("View Dataset Log (DATASET_LOG.md)")
        act.triggered.connect(lambda: self.viat_view_dataset_log())

        img_menu.addSeparator()

        # Auto-import YOLO detections (produced by yolo_detect.py)
        act = img_menu.addAction("Auto-Import Detections (move to review_label)...")
        act.triggered.connect(lambda: self.viat_auto_import_detections())

        img_menu.addSeparator()

        act = img_menu.addAction("Merge Dataset into Current...")
        act.triggered.connect(lambda: self.viat_merge_dataset())

        img_menu.addSeparator()

        act = img_menu.addAction("Extract Single Class from Datasets...")
        act.triggered.connect(lambda: self.viat_extract_single_class_dataset())

        act = img_menu.addAction("Batch Prediction Queue Builder...")
        act.triggered.connect(lambda: self.viat_launch_batch_prediction_queue())
        
        act = img_menu.addAction("Remove Background Images (Percentage)...")
        act.triggered.connect(lambda: self.viat_remove_background_images())

        act = img_menu.addAction("Dataset Integration Wizard (Roadmap)...")
        act.triggered.connect(lambda: self.viat_launch_integration_wizard())

        # Video annotation operations
        vid_menu = dataset_menu.addMenu("Video Annotation Operations")

        act = vid_menu.addAction("Import VIAT JSON Annotations...")
        act.triggered.connect(lambda: self.viat_import_video_json())

        act = vid_menu.addAction("Fix Video Borders (remove/clip labels)...")
        act.triggered.connect(lambda: self.viat_detect_and_fix_borders())

        act = vid_menu.addAction("Object Visibility Mode...")
        act.triggered.connect(lambda: self.viat_start_object_visibility_mode())

        vid_menu.addSeparator()

        # Segmentation video labeling
        seg_menu = vid_menu.addMenu("Segmentation Video")
        act = seg_menu.addAction("Pick Object Color...")
        act.triggered.connect(lambda: self.viat_seg_video_pick_color())
        act = seg_menu.addAction("Track All Objects")
        act.triggered.connect(lambda: self.viat_seg_video_track_all())
        act = seg_menu.addAction("Export Seg Video JSON...")
        act.triggered.connect(lambda: self.viat_seg_video_export_json())

        vid_menu.addSeparator()

        act = vid_menu.addAction("Export VIAT JSON...")
        act.triggered.connect(lambda: self.viat_export_json())
        act = vid_menu.addAction("Toggle Remove Unverified Labels")
        act.setShortcut("Shift+U")
        act.triggered.connect(lambda: self.viat_toggle_remove_unverified())
        act = vid_menu.addAction("Frame Cache Stats...")
        act.triggered.connect(lambda: self.viat_perf_stats())
        act = vid_menu.addAction("Clear Frame Cache")
        act.triggered.connect(lambda: self.viat_clear_cache())

        # --- View menu additions ---
        # Find the View menu (it's usually the 3rd or 4th menu)
        view_menu = None
        for action in menubar.actions():
            if action.text() in ("&View", "View"):
                view_menu = action.menu()
                break
        if view_menu is None:
            view_menu = menubar.addMenu("&View")

        view_menu.addSeparator()
        act = view_menu.addAction("Toggle Segmentation Display")
        act.setShortcut("Shift+S")
        act.triggered.connect(lambda: self.viat_toggle_segmentation())

        act = view_menu.addAction("Dataset Statistics...")
        act.triggered.connect(lambda: self.viat_dataset_stats())

        act = view_menu.addAction("View Dataset Log...")
        act.triggered.connect(lambda: self.viat_view_dataset_log())

        # Store reference so we can re-inject if needed
        self._viat_extra_menus = dataset_menu

    @log_exceptions
    def viat_launch_integration_wizard(self):
        """Launches the Dataset Integration Roadmap Setup."""
        from widgets.dataset_wizard_dialog import DatasetWizardDialog
        from PyQt5.QtWidgets import QDialog, QMessageBox
        
        dialog = DatasetWizardDialog(self, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            settings = dialog.settings
            if not settings: return
            
            # 0. Offline Auto Tasks (run directly on folder before loading)
            try:
                from managers.dataset_integration import DatasetIntegrationManager
                manager = DatasetIntegrationManager(self)
                if settings.get("preflight"):
                    manager.run_preflight_check(settings["new_dataset"])
                if settings.get("remove_hash_duplicates"):
                    manager.remove_duplicates(settings["new_dataset"])
                if settings.get("standardize_format"):
                    manager.standardize_format(settings["new_dataset"])
                if settings.get("normalize_res"):
                    manager.normalize_resolution(settings["new_dataset"])
            except Exception as e:
                QMessageBox.warning(self, "Offline Task Error", f"Error during preflight/hash checks:\n{e}")
            
            # 1. Open the new dataset in the UI
            self.load_image_dataset_path(settings["new_dataset"])
            
            # 2. Enter Integration Mode
            self.integration_mode = True
            self.integration_main_dataset = settings["main_dataset"]
            
            # 3. Apply the Online Auto Tasks (if any)
            if settings["remove_grayscale"]:
                self.viat_remove_grayscale()
            if settings["remove_duplicates"]:
                self.viat_remove_duplicates()
            if settings["auto_import"] and settings.get("json_paths"):
                try:
                    from utils.dataset_ops import auto_import_detections
                    auto_import_detections(self, settings["json_paths"], target_classes=settings.get("target_classes"))
                    # Refresh UI
                    if self.current_frame >= self.total_frames:
                        self.current_frame = max(0, self.total_frames - 1)
                    self.frame_slider.setMaximum(max(0, self.total_frames - 1))
                    self.load_current_image()
                    self.update_frame_info()
                    self.update_annotation_list()
                except Exception as e:
                    QMessageBox.warning(self, "Auto-Import Error", f"Error applying auto-import:\n{e}")
                
            # 4. Show the Finish Button in Toolbar
            if hasattr(self, 'finish_integration_action'):
                self.finish_integration_action.setVisible(True)
            if hasattr(self, 'finish_integration_move_action'):
                self.finish_integration_move_action.setVisible(True)
            if hasattr(self, 'finish_integration_sep'):
                self.finish_integration_sep.setVisible(True)
            self.statusBar.showMessage("Integration Mode active. Please review the dataset.", 10000)

    @log_exceptions
    def viat_finish_integration(self):
        """Finalizes the integration mode and merges the dataset."""
        from PyQt5.QtWidgets import QMessageBox, QInputDialog
        import cv2
        if not self.integration_mode or not self.integration_main_dataset:
            return
            
        reply = QMessageBox.question(
            self, "Finish Integration",
            "Are you done reviewing? This will merge the current dataset into the Main Dataset.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            source_folder = ""
            if hasattr(self, "_viat_dataset_info") and self._viat_dataset_info:
                source_folder = self._viat_dataset_info.root
            elif self.image_files:
                source_folder = os.path.dirname(self.image_files[0])
                
            if not source_folder:
                QMessageBox.warning(self, "Error", "Could not determine current dataset path.")
                return
                
            # Prompt for resizing
            text, ok = QInputDialog.getText(
                self, "Resize Dataset",
                "Resize all images before merging? Enter W,H (or clear to skip):",
                text="640,640"
            )
            if ok and text.strip():
                try:
                    parts = text.split(',')
                    w, h = int(parts[0].strip()), int(parts[1].strip())
                    resized_count = 0
                    for root, dirs, files in os.walk(source_folder):
                        if "removed" in root or "review" in root:
                            continue
                        for file in files:
                            ext = os.path.splitext(file)[1].lower()
                            if ext in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}:
                                img_path = os.path.join(root, file)
                                img = cv2.imread(img_path)
                                if img is not None:
                                    img_resized = cv2.resize(img, (w, h))
                                    cv2.imwrite(img_path, img_resized)
                                    resized_count += 1
                    self.statusBar.showMessage(f"Resized {resized_count} images to {w}x{h}", 5000)
                except Exception as e:
                    QMessageBox.warning(self, "Resize Error", f"Failed to resize images:\n{e}")

            # Launch the merge dialog directly, skipping the source folder selection
            self.viat_merge_dataset(source_folder=source_folder, target_folder=self.integration_main_dataset)
            
            # Clean up integration mode
            self.integration_mode = False
            self.integration_main_dataset = ""
            if hasattr(self, 'finish_integration_action'):
                self.finish_integration_action.setVisible(False)
            if hasattr(self, 'finish_integration_move_action'):
                self.finish_integration_move_action.setVisible(False)
            if hasattr(self, 'finish_integration_sep'):
                self.finish_integration_sep.setVisible(False)

    @log_exceptions
    def viat_finish_integration_move(self):
        """Moves the dataset to a review folder instead of merging."""
        from PyQt5.QtWidgets import QMessageBox, QInputDialog, QFileDialog
        import shutil
        import os
        from datetime import datetime
        
        if not self.integration_mode:
            return
            
        reply = QMessageBox.question(
            self, "Move Dataset",
            "Are you sure you want to move this dataset to a review folder instead of merging?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            source_folder = ""
            if hasattr(self, "_viat_dataset_info") and self._viat_dataset_info:
                source_folder = self._viat_dataset_info.root
            elif self.image_files:
                source_folder = os.path.dirname(self.image_files[0])
                
            if not source_folder:
                QMessageBox.warning(self, "Error", "Could not determine current dataset path.")
                return
                
            # Prompt for target folder
            target_folder = QFileDialog.getExistingDirectory(
                self, "Select Review/Destination Folder", "", QFileDialog.ShowDirsOnly
            )
            if not target_folder:
                return
                
            # Prompt for details
            details, ok = QInputDialog.getMultiLineText(
                self, "Dataset Details",
                "Enter details for this dataset (will be saved to details.md):"
            )
            
            if ok:
                try:
                    # Save details.md in the target_folder (appending if exists)
                    dataset_name = os.path.basename(source_folder)
                    details_file = os.path.join(target_folder, "details.md")
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    
                    with open(details_file, "a", encoding="utf-8") as f:
                        f.write(f"\n## Dataset: {dataset_name} ({timestamp})\n")
                        if details.strip():
                            f.write(f"{details.strip()}\n")
                        else:
                            f.write("No details provided.\n")
                    
                    # Clear state to release file locks before moving
                    self.reset_media_state()
                    
                    dest_path = os.path.join(target_folder, dataset_name)
                    
                    # Handle name collision if dest_path already exists
                    if os.path.exists(dest_path):
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        dataset_name = f"{dataset_name}_{timestamp}"
                        dest_path = os.path.join(target_folder, dataset_name)
                        
                    shutil.move(source_folder, dest_path)
                    
                    self.statusBar.showMessage(f"Moved dataset to {dest_path}", 5000)
                    
                    # Clean up integration mode
                    self.integration_mode = False
                    self.integration_main_dataset = ""
                    if hasattr(self, 'finish_integration_action'):
                        self.finish_integration_action.setVisible(False)
                    if hasattr(self, 'finish_integration_move_action'):
                        self.finish_integration_move_action.setVisible(False)
                    if hasattr(self, 'finish_integration_sep'):
                        self.finish_integration_sep.setVisible(False)
                        
                except Exception as e:
                    QMessageBox.warning(self, "Move Error", f"Failed to move dataset:\n{e}")
    @log_exceptions
    def setup_playback_timer(self):
        """Set up the timer for video playback."""
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.next_frame)

    @log_exceptions
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

    # -------------------------------------------------------------------------
    # Video and Image Handling Methods
    # -------------------------------------------------------------------------

    @log_exceptions
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

    @log_exceptions
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
            video_base = os.path.dirname(filename)
            video_name = os.path.splitext(os.path.basename(filename))[0]
            auto_save_folder = os.path.join(video_base,'autosaves')
            os.makedirs(auto_save_folder,exist_ok=True)
            self.autosave_file = os.path.join(auto_save_folder,video_name+'_autosave.json')

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

    @log_exceptions
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

    @log_exceptions
    def open_image_folder(self):
        """Open a folder of images OR a labeled dataset via file dialog."""
        folder_path = QFileDialog.getExistingDirectory(
            self, "Open Image Folder / Dataset", "", QFileDialog.ShowDirsOnly
        )
        if not folder_path:
            return
        self.load_image_dataset_path(folder_path)

    @log_exceptions
    def load_image_dataset_path(self, folder_path):
        """Load a folder of images OR a labeled dataset directly from a path."""
        # Reset existing state
        self.reset_media_state()

        # Scan the folder (recurses into splits + label subfolders)
        info = scan_dataset(folder_path)
        self._viat_dataset_info = info

        if info.image_count == 0:
            QMessageBox.warning(
                self, "Open Image Folder",
                "No image files found in the selected folder!",
            )
            return

        # Load everything (all splits by default) into frame_annotations.
        result = load_dataset_into_app(self, info, BoundingBox)
        self._viat_frame_to_split = result["frame_to_split"]

        self.image_files = result["image_files"]
        self.total_frames = len(self.image_files)
        self.current_frame = 0
        self.is_image_dataset = True

        self.load_current_image()
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.update_frame_info()

        # Window title shows layout + per-split counts
        folder_name = os.path.basename(folder_path)
        counts = ", ".join(
            f"{k}={v}" for k, v in result["per_split_counts"].items()
        )
        self.setWindowTitle(
            f"VIAT - {folder_name} [{info.layout}] ({counts})"
        )

        if hasattr(self, "play_button"):
            self.play_button.setEnabled(True)
            self.play_button.setIcon(
                self.icon_provider.get_icon("media-playback-start")
            )

        self.autosave_file = os.path.join(folder_path, f"{folder_name}_autosave.json")
        if self.autosave_enabled and not self.autosave_timer.isActive():
            self.autosave_timer.start(self.autosave_interval)

        # Status message: classes source + conflict warning
        msg = (
            f"Loaded {self.total_frames} images across "
            f"{len(info.splits)} split(s); {len(info.classes)} classes "
            f"from {info.classes_source or 'labels (inferred)'}; "
            f"format={info.label_format or 'none'}"
        )
        if info.classes_conflict:
            msg += f"  ⚠ {info.classes_conflict}"
            QMessageBox.warning(self, "Class Name Conflict", info.classes_conflict)
        if result["warnings"]:
            msg += f"  ({len(result['warnings'])} parse warnings)"
        self.statusBar.showMessage(msg, 8000)

        self.update_annotation_list()
        self.refresh_class_ui()

        # Initialize the dataset workflow log (DATASET_LOG.md)
        try:
            _viat_init_dataset_log(self, info)
        except Exception:
            pass

    @log_exceptions
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

    @log_exceptions
    def open_image_dataset(self, folder_path=None):
        """Open a labeled dataset (Roboflow/splits/etc). Now delegates
        to the unified open_image_folder() flow."""
        self.open_image_folder()

    @log_exceptions
    def load_image_dataset_from_project(self, image_dataset_info, current_frame=None):
        """Load an image dataset from a saved project / autosave.

        current_frame is optional (defaults to self.current_frame, which
        load_project has just restored). Re-scans the dataset folder so
        that _viat_dataset_info, _viat_frame_to_split, and labels are
        populated, then refreshes the class UI.
        """
        if current_frame is None:
            current_frame = getattr(self, "current_frame", 0)

        base_folder = image_dataset_info.get("base_folder", "")
        if not os.path.exists(base_folder):
            reply = QMessageBox.question(
                self, "Folder Not Found",
                f"The original image folder '{base_folder}' was not found.\n"
                f"Would you like to locate it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            if reply == QMessageBox.Yes:
                new_base = QFileDialog.getExistingDirectory(
                    self, "Locate Image Folder", "", QFileDialog.ShowDirsOnly
                )
                if new_base:
                    base_folder = new_base
                else:
                    return False
            else:
                return False

        # Re-scan the dataset to get fresh info (layout, splits, labels)
        from utils.dataset_manager import scan_dataset, load_dataset_into_app
        info = scan_dataset(base_folder)
        self._viat_dataset_info = info

        # If the saved project has image_files as relative paths, use them;
        # otherwise use the scanned images.
        relative_paths = image_dataset_info.get("image_files", [])
        if relative_paths:
            image_files = []
            for rel in relative_paths:
                ap = os.path.join(base_folder, rel)
                if os.path.exists(ap):
                    image_files.append(ap)
        else:
            image_files = info.all_images

        if not image_files:
            QMessageBox.critical(self, "Error", "No image files could be loaded.")
            return False

        # Save the annotations and classes that were just loaded from the .viat file
        saved_annotations = getattr(self, "frame_annotations", {})
        saved_class_colors = getattr(self.canvas, "class_colors", {})
        saved_class_attrs = getattr(self.canvas, "class_attributes", {})

        # Load labels from disk to catch any new images/labels added externally
        result = load_dataset_into_app(self, info, BoundingBox)
        
        # CRITICAL FIX: The .viat project file is the source of truth for edits made in VIAT.
        # We must overlay the project-saved annotations and classes on top of the disk scan
        # so that unsaved VIAT edits and custom class names are not lost!
        for frame_idx, anns in saved_annotations.items():
            self.frame_annotations[frame_idx] = anns
            
        self.canvas.class_colors.update(saved_class_colors)
        self.canvas.class_attributes.update(saved_class_attrs)
        self.class_colors = self.canvas.class_colors
        self.class_attributes = self.canvas.class_attributes
        self._viat_frame_to_split = result["frame_to_split"]
        self.image_files = result["image_files"]
        self.total_frames = len(self.image_files)
        self.current_frame = min(current_frame, self.total_frames - 1)
        self.is_image_dataset = True

        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.frame_slider.setValue(self.current_frame)

        self.load_current_image()
        self.update_frame_info()

        folder_name = os.path.basename(base_folder)
        self.setWindowTitle(f"VIAT - Image Dataset: {folder_name}")

        if hasattr(self, "play_button"):
            self.play_button.setEnabled(True)
            self.play_button.setIcon(
                self.icon_provider.get_icon("media-playback-start")
            )

        # Initialize the dataset workflow log
        try:
            _viat_init_dataset_log(self, info)
        except Exception:
            pass

        # CRITICAL: refresh the class UI so the class selector / dock
        # shows the loaded classes (was missing -- classes were in
        # canvas.class_colors but the selector stayed empty).
        self.refresh_class_ui()
        self.update_annotation_list()

        return True

    @log_exceptions
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

    @log_exceptions
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


    @log_exceptions
    def setup_playback_timer(self):
        """Set up the timer for video playback or image slideshow."""
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.next_frame)

        # Initialize slideshow speed to 1 second per image
        self.slideshow_speed = 1.0


    @log_exceptions
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


    # -------------------------------------------------------------------------
    # Playback Control Methods
    # -------------------------------------------------------------------------

    @log_exceptions
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

    @log_exceptions
    def slider_changed(self, value):
        """Handle slider value changes (user drag only -- programmatic
        setValue blocks signals, so this only fires on genuine user
        interaction)."""
        # A manual slider drag means the user took control of navigation;
        # cancel any in-progress interpolation cycle so Next/Prev behave
        # predictably and slider/keyboard/mouse share the same state.
        if (
            hasattr(self, "interpolation_manager")
            and self.interpolation_manager.is_active
            and value != self.current_frame
        ):
            self.interpolation_manager.reset_cycle()

        if hasattr(self, "object_visibility_manager") and self.object_visibility_manager and self.object_visibility_manager.active:
            visible_frames = self.object_visibility_manager.get_visible_frame_numbers()
            if visible_frames and value not in visible_frames:
                # Value is outside the active range, force slider back to valid frame and ignore this event
                self.frame_slider.blockSignals(True)
                self.frame_slider.setValue(self.current_frame)
                self.frame_slider.blockSignals(False)
                return


        if hasattr(self, "is_image_dataset") and self.is_image_dataset:
            if 0 <= value < len(self.image_files):
                if value != self.current_frame:
                    self.current_frame = value
                    self.load_current_image()
                    self.update_frame_info()
                    self.load_current_frame_annotations()
                    self.update_frame_display()
        elif self.cap and self.cap.isOpened():
            frame_number = int(value)
            if frame_number != self.current_frame:
                self.seek_to_frame(frame_number)
                self.update_frame_display()

    @log_exceptions
    def seek_to_frame(self, frame_number):
        """Seek the video to an exact frame and refresh the display."""
        if not self.cap or not self.cap.isOpened():
            return False

        if frame_number < 0:
            frame_number = 0
        elif frame_number >= self.total_frames:
            frame_number = self.total_frames - 1

        if frame_number == self.current_frame + 1:
            ret, frame = self.cap.read()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()

        if not ret:
            return False

        self.current_frame = frame_number
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_frame)
        self.frame_slider.blockSignals(False)
        self.canvas.set_frame(frame)
        self.update_frame_info()
        self.load_current_frame_annotations()
        self.update_frame_display()
        return True

    @log_exceptions
    def prev_frame(self):
        """Go to the previous frame. ALWAYS steps back exactly one frame,
        regardless of interpolation mode (per user requirement)."""
        self.handle_unverified_annotations()
        if hasattr(self, "is_image_dataset") and self.is_image_dataset:
            if self.current_frame > 0:
                self.current_frame -= 1
                self.frame_slider.blockSignals(True)
                self.frame_slider.setValue(self.current_frame)
                self.frame_slider.blockSignals(False)
                self.load_current_image()
                self.update_frame_info()
                self.load_current_frame_annotations()
                self.update_frame_display()
        else:
            prev = max(0, self.current_frame - 1)
            if self.seek_to_frame(prev):
                self.update_frame_display()

    @log_exceptions
    def next_frame(self):
        """Go to the next frame, or drive the interpolation workflow when
        interpolation mode is active (and not during playback)."""
        self.handle_unverified_annotations()
        if hasattr(self, "is_image_dataset") and self.is_image_dataset:
            if self.current_frame < len(self.image_files) - 1:
                self.current_frame += 1
                self.frame_slider.blockSignals(True)
                self.frame_slider.setValue(self.current_frame)
                self.frame_slider.blockSignals(False)
                self.load_current_image()
                self.update_frame_info()
                self.load_current_frame_annotations()
                self.update_frame_display()
            else:
                if self.is_playing:
                    self.current_frame = 0
                    self.frame_slider.blockSignals(True)
                    self.frame_slider.setValue(self.current_frame)
                    self.frame_slider.blockSignals(False)
                    self.load_current_image()
                    self.update_frame_info()
                    self.load_current_frame_annotations()
                    self.statusBar.showMessage("Looping back to start of image dataset")
                    self.update_frame_display()
                else:
                    self.statusBar.showMessage("End of image dataset")
            return

        # --- Video ---
        next_frame_number = None

        # Interpolation workflow drives Next (single owner of jumps).
        # Note: object visibility mode no longer restricts navigation -- the
        # user can navigate freely; the canvas filter shows only the current
        # object's annotations.
        if (
            hasattr(self, "interpolation_manager")
            and self.interpolation_manager.is_active
            and not self.is_playing
        ):
            # Interpolation workflow drives Next (single owner of jumps).
            next_frame_number = self.interpolation_manager.get_next_frame(
                self.current_frame
            )
        elif self.cap and self.cap.isOpened():
            next_frame_number = self.current_frame + 1
            if next_frame_number >= self.total_frames:
                self.play_timer.stop()
                self.is_playing = False
                self.statusBar.showMessage("End of video")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.seek_to_frame(0)
                return

        if next_frame_number is not None and self.seek_to_frame(next_frame_number):
            self.update_frame_display()
            if (
                self.duplicate_frames_enabled
                and self.current_frame in self.frame_hashes
            ):
                current_hash = self.frame_hashes[self.current_frame]
                if (
                    current_hash in self.duplicate_frames_cache
                    and len(self.duplicate_frames_cache[current_hash]) > 1
                ):
                    self.propagate_annotations_to_duplicate(current_hash)

    @log_exceptions
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

    @log_exceptions
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

    # -------------------------------------------------------------------------
    # Annotation Handling Methods
    # -------------------------------------------------------------------------
    
    @log_exceptions
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

    @log_exceptions
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

    @log_exceptions
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

    @log_exceptions
    def edit_annotation(self, annotation, focus_first_field=False):
        """
        Edit the properties of an annotation.

        Args:
            annotation: The annotation to edit
            focus_first_field: Whether to focus on the first attribute field
        """
        self.save_undo_state()
        self.annotation_manager.edit_annotation(annotation, focus_first_field)

    @log_exceptions
    def delete_annotation(self, annotation):
        """Delete the specified annotation."""
        if self.annotation_manager.delete_annotation(annotation):
            self.save_undo_state()
            # Mark project as modified
            self.project_modified = True

            # Update annotation list
            self.update_annotation_list()

    @log_exceptions
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

    @log_exceptions
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

    @log_exceptions
    def add_empty_annotation(self):
        """Add a new empty annotation with default values."""
        self.save_undo_state()
        self.annotation_manager.add_empty_annotation()

    @log_exceptions
    def update_annotation_list(self):
        """Update the annotation list in the UI and handle interpolation."""
        # Update the annotation dock
        if hasattr(self, "annotation_dock"):
            self.annotation_dock.update_annotation_list()

        # Save current annotations to frame_annotations
        self.frame_annotations[self.current_frame] = self.canvas.annotations.copy()

        # The interpolation workflow is now driven entirely by Next/Prev
        # (see InterpolationManager.get_next_frame). No action needed here.

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

    @log_exceptions
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

    @log_exceptions
    def clear_annotations(self):
        """Clear all annotations."""
        self.save_undo_state()
        self.annotation_manager.clear_annotations()

    @log_exceptions
    def add_annotation(self):
        """Add annotation manually."""
        self.save_undo_state()
        self.annotation_manager.add_annotation()

    @log_exceptions
    def create_annotation_dialog(self):
        """Create a dialog for adding or editing annotations."""
        return self.annotation_manager.create_annotation_dialog()

    @log_exceptions
    def parse_attributes(self, text):
        """Parse attributes from text input."""
        return self.annotation_manager.parse_attributes(text)

    @log_exceptions
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

    @log_exceptions
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

    @log_exceptions
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

    # -------------------------------------------------------------------------
    # Class Management Methods
    # -------------------------------------------------------------------------

    @log_exceptions
    def add_class(self):
        self.save_undo_state()
        self.class_manager.add_class()

    @log_exceptions
    def import_classes_from_yolo_yaml(self):
        """Import annotation classes from a YOLO dataset YAML file."""
        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import Classes from YOLO YAML",
            "",
            "YAML Files (*.yaml *.yml);;All Files (*)",
        )
        if filename:
            self.save_undo_state()
            self.class_manager.import_classes_from_yolo_yaml(filename)

    @log_exceptions
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

    @log_exceptions
    def edit_selected_class(self):
        """Edit the selected class with option to convert to another class."""
        self.save_undo_state()
        self.class_manager.edit_selected_class()

    @log_exceptions
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

    @log_exceptions
    def convert_class_with_attribute_mapping(self, old_class, new_class):
        """
        Convert class with custom attribute mapping.

        Args:
            old_class (str): The original class name
            new_class (str): The target class name
        """
        self.save_undo_state()
        self.class_manager.convert_class_with_attribute_mapping(old_class, new_class)

    @log_exceptions
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

    @log_exceptions
    def convert_class(self, old_class, new_class):
        """Convert all annotations from one class to another."""
        self.save_undo_state()
        self.class_manager.convert_class(old_class, new_class)

    @log_exceptions
    def update_class(self, old_name, new_name, color):
        """Update a class with new name and color."""
        self.save_undo_state()
        self.class_manager.update_class(old_name, new_name, color)

    @log_exceptions
    def delete_selected_class(self):
        """Delete the selected class."""
        self.save_undo_state()
        self.class_manager.delete_selected_class()

    # -------------------------------------------------------------------------
    # Project Management Methods
    # -------------------------------------------------------------------------

    @log_exceptions
    def save_project(self, filename=False):
        """Save the current project."""
        if not filename and self.project_file:
            filename = self.project_file
        elif filename:
            pass
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
            backup_before_save(filename)
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
                tracking_mode_enabled=self.tracking_mode_enabled,
                interpolation_mode_active=self.interpolation_manager.is_active,
                verification_mode_enabled=self.verification_mode,
                annotations_imported_list=list(self._annotations_imported) if hasattr(self, "_annotations_imported") else [],
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

    @log_exceptions
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
            
            # Clear imported annotations tracking
            if hasattr(self, "_annotations_imported"):
                self._annotations_imported = set()

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



    @log_exceptions
    def load_project(self, filename=None):
        """Load a project from a file."""

        if not filename:
            # Ask for a file name
            filename, _ = QFileDialog.getOpenFileName(
                self, "Load Project", "", "VIAT Project Files (*.viat)"
            )
            if not filename:
                return False  # User cancelled

        try:
            # Use load_project_with_backup for safer loading
            project_data = load_project_with_backup(filename)
            if not project_data:
                QMessageBox.critical(
                    self,
                    "Error Loading Project",
                    "Failed to load project file and no valid backups found.",
                )
                return False
                
            # Extract data from project_data
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
                tracking_mode_enabled,
                interpolation_mode_active,
                verification_mode_enabled,
                annotations_imported_list,
            ) = load_project(filename, BoundingBox)
            # Set up the application with the loaded data
            self.canvas.annotations = annotations
            self.class_colors = class_colors
            self.canvas.class_colors = class_colors  # CRITICAL FIX
            self.current_frame = current_frame
            self.frame_annotations = frame_annotations
            self.class_attributes = class_attributes
            self.canvas.class_attributes = class_attributes  # CRITICAL FIX
            self.current_style = current_style
            self.auto_show_attribute_dialog = auto_show_attribute_dialog
            self.use_previous_attributes = use_previous_attributes
            self.duplicate_frames_enabled = duplicate_frames_enabled
            
            # Restore annotations imported tracking
            if annotations_imported_list:
                self._annotations_imported = set(annotations_imported_list)
            else:
                self._annotations_imported = set()
            
            if frame_hashes:
                self.frame_hashes = frame_hashes
            if duplicate_frames_cache:
                self.duplicate_frames_cache = duplicate_frames_cache
            if image_dataset_info:
                self.image_dataset_info = image_dataset_info
                
            # Set mode states
            self.toggle_tracking_mode(tracking_mode_enabled)
            if hasattr(self.interpolation_manager, "set_active"):
                self.interpolation_manager.set_active(interpolation_mode_active)
            self.verification_mode = verification_mode_enabled

            # Load the video if specified
            if video_path and os.path.exists(video_path):
                self.load_video_file(video_path)
            elif image_dataset_info:
                self.load_image_dataset_from_project(image_dataset_info, current_frame)
                
            # Update UI
            self.update_frame_display()
            self.update_annotation_list()
            
            # Set project path
            self.project_path = filename
            self.setWindowTitle(f"Video Annotation Tool - {os.path.basename(filename)}")
            self.project_modified = False
            self.statusBar.showMessage(f"Project loaded from {filename}", 5000)
            return True
            
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Project",
                f"Could not load project:\n{str(e)}",
            )
            return False

    @log_exceptions
    def save_application_state(self):
        """Save the current application state."""
        if not hasattr(self, "project_file") or not self.project_file:
            return

        state = {
                    'project_path': self.project_path,
                    'video_filename': self.video_filename,
                    'current_frame': self.current_frame,
                    'zoom_level': self.canvas.zoom_level if hasattr(self.canvas, 'zoom_level') else 1.0,
                    # Save mode states
                    'tracking_mode_enabled': self.tracking_mode_enabled,
                    'interpolation_mode_active': self.interpolation_manager.is_active if hasattr(self.interpolation_manager, 'is_active') else False,
                    'verification_mode_enabled': self.verification_mode if hasattr(self, 'verification_mode') else False
        }

        save_last_state(state)

    @log_exceptions
    def load_application_state(self):
        """Load the last application state."""
        state = load_last_state()
        if not state:
            return False

        # Load last project if it exists
        last_project = state.get("last_project")
        if last_project and os.path.exists(last_project):
            self.has_autosave = True
            self.load_project(last_project)
            return True

        return False

    @log_exceptions
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

    @log_exceptions
    def clear_recent_projects(self):
        """Clear the list of recent projects."""
        config_dir = get_config_directory()
        recent_projects_file = os.path.join(config_dir, "recent_projects.json")

        with open(recent_projects_file, "w") as f:
            json.dump([], f)

        self.update_recent_projects_menu()
        self.statusBar.showMessage("Recent projects cleared", 3000)

    @log_exceptions
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

    @log_exceptions
    def clear_recent_projects(self):
        """Clear the list of recent projects."""

        config_dir = get_config_directory()
        recent_projects_file = os.path.join(config_dir, "recent_projects.json")

        with open(recent_projects_file, "w") as f:
            json.dump([], f)

        self.update_recent_projects_menu()
        self.statusBar.showMessage("Recent projects cleared", 3000)

    # -------------------------------------------------------------------------
    # Import/Export Methods
    # -------------------------------------------------------------------------
  
    @log_exceptions
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

    @log_exceptions
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

    @log_exceptions
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

    @log_exceptions
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

    @log_exceptions
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

    @log_exceptions
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

    @log_exceptions
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
        self._annotations_imported.add(filename)
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

    @log_exceptions
    def check_for_annotation_files(self, video_filename):
        """
        Check if annotation files with the same base name as the video exist.
        If found, ask the user if they want to import them.

        Args:
            video_filename (str): Path to the video file
        """
            
        # If we've already imported annotations for this video, skip the check
        
        extensions = [".txt", ".json", ".xml"]
        file_basename, _ = os.path.splitext(video_filename)
        for vid in self._annotations_imported:
            for ext in extensions:
                if Path(file_basename + ext)== Path(vid):
                    return
        
        # Get the directory and base name without extension
        directory = os.path.join(os.path.dirname(video_filename),)
        save_path = os.path.join(directory,'auto_save')
        base_name = os.path.splitext(os.path.basename(video_filename))[0]

        # Check for auto-save file first
        autosave_file = os.path.join(save_path, f"{base_name}_autosave.json")

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

    @log_exceptions
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

    @log_exceptions
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

    # -------------------------------------------------------------------------
    # Clipboard Operations
    # -------------------------------------------------------------------------
    @log_exceptions
    def copy_selected_annotation(self):
        """Copy the currently selected annotation."""
        if hasattr(self, "canvas") and self.canvas.selected_annotation:
            self.clipboard_annotation = self.canvas.selected_annotation.copy()
            self.statusBar.showMessage("Annotation copied", 2000)

    @log_exceptions
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

    @log_exceptions
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

    # -------------------------------------------------------------------------
    # Duplicate Frame Detection and Handling
    # -------------------------------------------------------------------------

    @log_exceptions
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

    @log_exceptions
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

        self.frame_hashes, self.duplicate_frames_cache = (
            self.performance_manager.optimize_frame_hashes(
                self.frame_hashes, self.duplicate_frames_cache
            )
        )
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

    @log_exceptions
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

    @log_exceptions
    def clone_annotation(self, annotation):
        """
        Create a deep copy of an annotation.

        Args:
            annotation: The annotation to clone

        Returns:
            A new annotation object with the same properties
        """

        return deepcopy(annotation)

    @log_exceptions
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

    @log_exceptions
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

    # -------------------------------------------------------------------------
    # Annotation Propagation Methods
    # -------------------------------------------------------------------------
   
    @log_exceptions
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

    @log_exceptions
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

    @log_exceptions
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

    @log_exceptions
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

    # -------------------------------------------------------------------------
    # Interpolation Methods
    # -------------------------------------------------------------------------

    @log_exceptions
    def toggle_interpolation_mode(self):
        """Toggle interpolation mode on/off."""
        is_active = self.toggle_interpolation_action.isChecked()
        self.interpolation_manager.set_active(is_active)

        # Show/hide the interpolation toolbar
        if hasattr(self, "interpolation_toolbar"):
            self.interpolation_toolbar.setVisible(is_active)

        # Update UI
        self.update_frame_display()

    @log_exceptions
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

    @log_exceptions
    def perform_interpolation(self):
        """Manually trigger interpolation between annotated frames."""
        if not self.interpolation_manager.is_active:
            QMessageBox.warning(
                self, "Interpolation", "Interpolation mode is not active."
            )
            return
        self.interpolation_manager.perform_pending_interpolation()

    @log_exceptions
    def update_frame_display(self):
        """Update the keyframe/interpolation indicator for the current frame.

        Uses the interpolation cycle (anchor/target) when active, so the
        indicator reflects the actual workflow state instead of a stale
        'last_annotated_frame' value.
        """
        if not (
            hasattr(self, "interpolation_manager")
            and self.interpolation_manager.is_active
        ):
            if hasattr(self, "canvas"):
                self.canvas.setStyleSheet("")
            return

        has_annotations = (
            self.current_frame in self.frame_annotations
            and len(self.frame_annotations[self.current_frame]) > 0
        )
        is_keyframe = self.interpolation_manager.is_keyframe()

        if hasattr(self, "keyframe_indicator"):
            if is_keyframe:
                self.keyframe_indicator.setStyleSheet(
                    "background-color: #FF5555; min-width: 16px;"
                )
                self.keyframe_indicator.setToolTip("Current frame is a keyframe")
            elif has_annotations:
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
                self.keyframe_indicator.setToolTip("Current frame has no annotations")

        if hasattr(self, "canvas"):
            if is_keyframe:
                self.canvas.setStyleSheet("border: 2px solid #FF5555;")
            elif has_annotations:
                self.canvas.setStyleSheet("border: 2px solid #55AAFF;")
            else:
                self.canvas.setStyleSheet("")

    # -------------------------------------------------------------------------
    # Verification Methods
    # -------------------------------------------------------------------------

    @log_exceptions
    def toggle_verification_mode(self):
        """Toggle verification mode for annotations."""
        if not hasattr(self, "verification_mode_enabled"):
            self.verification_mode = False

        self.verification_mode = not self.verification_mode

        if self.verification_mode:
            self.statusBar.showMessage(
                "Verification mode enabled - unverified annotations will be deleted when changing frames",
                5000,
            )
        else:
            self.statusBar.showMessage("Verification mode disabled", 3000)

        # Update UI to show current mode
        if hasattr(self, "verify_mode_action"):
            self.verify_mode_action.setChecked(self.verification_mode)

    @log_exceptions
    def verify_selected_annotation(self):
        """Mark the selected annotation as verified."""
        if (
            hasattr(self.canvas, "selected_annotation")
            and self.canvas.selected_annotation
        ):
            self.save_undo_state()
            self.canvas.selected_annotation.verified = True
            self.canvas.update()
            self.update_annotation_list()
            self.statusBar.showMessage("Annotation verified", 2000)

    @log_exceptions
    def verify_all_annotations(self):
        """Mark all annotations in the current frame as verified."""
        if self.canvas.annotations:
            self.save_undo_state()
            for annotation in self.canvas.annotations:
                annotation.verified = True
            self.canvas.update()
            self.update_annotation_list()
            self.statusBar.showMessage(
                f"All {len(self.canvas.annotations)} annotations verified", 2000
            )

    @log_exceptions
    def handle_unverified_annotations(self):
        """Handle unverified annotations when changing frames.

        Only removes unverified annotations if self.remove_unverified is True
        (toggle, default OFF). When OFF, unverified labels are kept.
        """
        if not getattr(self, "remove_unverified", False):
            return

        # Check if there are any unverified annotations in the current frame
        if self.current_frame in self.frame_annotations:
            unverified = [
                ann
                for ann in self.frame_annotations[self.current_frame]
                if not getattr(ann, "verified", False)
            ]
            for ann in unverified:
                if ann.source=='interpolated':
                    ann.source = 'manual'
                    ann.verified = True
            if unverified:
                # Remove unverified annotations
                self.frame_annotations[self.current_frame] = [
                    ann
                    for ann in self.frame_annotations[self.current_frame]
                    if getattr(ann, "verified", False)
                ]

                # Update canvas annotations
                self.canvas.annotations = self.frame_annotations[
                    self.current_frame
                ].copy()
                self.canvas.update()

                # Show message
                self.statusBar.showMessage(
                    f"Removed {len(unverified)} unverified annotations", 3000
                )

    # -------------------------------------------------------------------------
    # UI Customization Methods
    # -------------------------------------------------------------------------

    @log_exceptions
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

    @log_exceptions
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

    @log_exceptions
    def toggle_attribute_dialog(self):
        """Toggle automatic attribute dialog display."""
        self.auto_show_attribute_dialog = not self.auto_show_attribute_dialog
        self.statusBar.showMessage(
            f"Attribute dialog for new annotations {'enabled' if self.auto_show_attribute_dialog else 'disabled'}",
            3000,
        )

    @log_exceptions
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

    @log_exceptions
    def toggle_previous_attributes(self):
        """Toggle using previous annotation attributes as default."""
        self.use_previous_attributes = not self.use_previous_attributes
        self.statusBar.showMessage(
            f"Using previous annotation attributes as default {'enabled' if self.use_previous_attributes else 'disabled'}",
            3000,
        )

    @log_exceptions
    def toggle_autosave(self):
        """Toggle auto-save functionality."""
        self.autosave_enabled = not self.autosave_enabled

        if self.autosave_enabled:
            self.autosave_timer.start(self.autosave_interval)
            self.statusBar.showMessage("Auto-save enabled", 3000)
        else:
            self.autosave_timer.stop()
            self.statusBar.showMessage("Auto-save disabled", 3000)

    @log_exceptions
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

    @log_exceptions
    def toggle_pan_mode(self):
        """Toggle pan mode for the canvas"""
        enabled = self.pan_tool_action.isChecked()
        self.canvas.set_pan_mode(enabled)

        # Update cursor based on mode
        if enabled:
            self.canvas.setCursor(Qt.OpenHandCursor)
            self.statusBar.showMessage(
                "Pan mode enabled. Left-click and drag to pan the canvas.", 3000
            )
        else:
            self.canvas.setCursor(Qt.ArrowCursor)
            self.statusBar.showMessage("Pan mode disabled.", 3000)

    @log_exceptions
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

    @log_exceptions
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
                    video_base = os.path.dirname(self.video_filename)
                    video_name = os.path.splitext(os.path.basename(self.video_filename))[0]
                    auto_save_folder = os.path.join(video_base, 'autosaves')
                    os.makedirs(auto_save_folder, exist_ok=True)
                    self.autosave_file = os.path.join(auto_save_folder, video_name + '_autosave.json')
                else:
                    # No valid source to auto-save
                    return
        else:
            # Use the project file for auto-save
            self.autosave_file = self.project_file

        try:
            # Store the original project_path
            original_project_path = self.project_path if hasattr(self, "project_path") else None
            
            # Temporarily set project_path to autosave_file
            self.project_path = self.autosave_file
            success = self.save_project(self.autosave_file)
            
            # Restore the original project_path
            self.project_path = original_project_path
            
            if success:
                self.last_autosave_time = QDateTime.currentDateTime()
                self.statusBar.showMessage(
                    f"Auto-saved to {os.path.basename(self.autosave_file)}", 3000
                )
        except Exception as e:
            print(f"Auto-save failed: {str(e)}")

    # -------------------------------------------------------------------------
    # Zoom and View Control Methods
    # -------------------------------------------------------------------------

    @log_exceptions
    def zoom_in(self):
        """Zoom in on the canvas."""
        self.zoom_level *= 1.2
        self.canvas.set_zoom(self.zoom_level)

    @log_exceptions
    def zoom_out(self):
        """Zoom out on the canvas."""
        self.zoom_level /= 1.2
        self.canvas.set_zoom(self.zoom_level)

    @log_exceptions
    def reset_zoom(self):
        """Reset zoom to default level."""
        self.zoom_level = 1.0
        self.canvas.set_zoom(self.zoom_level)
        self.canvas.reset_pan()
    # -------------------------------------------------------------------------
    # Tool Methods
    # -------------------------------------------------------------------------

    @log_exceptions
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

    @log_exceptions
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

    @log_exceptions
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

    # -------------------------------------------------------------------------
    # Undo/Redo Methods
    # -------------------------------------------------------------------------
    
    @log_exceptions
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

    @log_exceptions
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

    @log_exceptions
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

    @log_exceptions
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

    # -------------------------------------------------------------------------
    # Event Handling Methods
    # -------------------------------------------------------------------------

    @log_exceptions
    def eventFilter(self, obj, event):
        """Global event filter for frame-navigation shortcuts.

        Arrow keys (without Ctrl) step frames; Space toggles play/pause;
        Tab cycles annotation selection when the canvas has focus. These
        are handled HERE and only here, so there is a single owner of
        frame navigation and no double-stepping between the event filter
        and keyPressEvent.
        """
        if event.type() == QEvent.KeyPress:
            from PyQt5.QtWidgets import (
                QLineEdit, QPlainTextEdit, QTextEdit,
                QSpinBox, QDoubleSpinBox, QComboBox,
            )
            focused = QApplication.focusWidget()
            typing = isinstance(
                focused,
                (QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox),
            )
            mods = QApplication.keyboardModifiers()

            if (event.key() == Qt.Key_Right and not typing
                    and not (mods & Qt.ControlModifier)):
                self.next_frame()
                return True
            if (event.key() == Qt.Key_Left and not typing
                    and not (mods & Qt.ControlModifier)):
                self.prev_frame()
                return True
            if event.key() == Qt.Key_Space and not typing:
                self.play_pause_video()
                return True
            if event.key() == Qt.Key_Tab:
                if isinstance(focused, VideoCanvas):
                    self.cycle_annotation_selection()
                    event.accept()
                    return True
        return super().eventFilter(obj, event)

    @log_exceptions
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts that are NOT frame navigation.
        Arrow keys are handled globally in eventFilter so there is a
        single owner of frame stepping (no double jumps)."""
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
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
            return
        if event.key() == Qt.Key_M:
            current_index = self.method_selector.currentIndex()
            new_index = (current_index + 1) % self.method_selector.count()
            self.method_selector.setCurrentIndex(new_index)
            return
        if event.key() == Qt.Key_B:
            if hasattr(self, "annotation_dock"):
                self.annotation_dock.batch_edit_annotations()
            return
        if event.key() == Qt.Key_P and (event.modifiers() & Qt.ControlModifier):
            self.propagate_annotations()
            return
        if event.key() == Qt.Key_C and (event.modifiers() & Qt.ControlModifier):
            self.copy_selected_annotation()
            return
        if event.key() == Qt.Key_V and (event.modifiers() & Qt.ControlModifier):
            self.paste_annotation()
            return
        if event.key() == Qt.Key_X and (event.modifiers() & Qt.ControlModifier):
            self.cut_selected_annotation()
            return
        if event.key() == Qt.Key_A and (event.modifiers() & Qt.ControlModifier):
            self.select_all_annotations()
            return
        if (
            event.key() == Qt.Key_Z
            and (event.modifiers() & Qt.ControlModifier)
            and not (event.modifiers() & Qt.ShiftModifier)
        ):
            self.undo()
            return
        if (event.key() == Qt.Key_Y and (event.modifiers() & Qt.ControlModifier)) or (
            event.key() == Qt.Key_Z
            and (event.modifiers() & Qt.ControlModifier)
            and (event.modifiers() & Qt.ShiftModifier)
        ):
            self.redo()
            return
        super().keyPressEvent(event)

    @log_exceptions
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

    
    # -------------------------------------------------------------------------
    # Miscellaneous Methods
    # -------------------------------------------------------------------------

    # -----------------------------------------------------------------
    # Dataset operations (patch2): remove bad frames, remap/merge classes
    # -----------------------------------------------------------------

    def _viat_ensure_dataset(self):
        """Return the loaded DatasetInfo or show a warning."""
        info = getattr(self, "_viat_dataset_info", None)
        if info is None or not getattr(self, "is_image_dataset", False):
            QMessageBox.warning(
                self, "Dataset Operation",
                "Open an image dataset first (File > Open Image Folder).",
            )
            return None
        return info

    def viat_current_split(self):
        """Return the split name of the current frame, or 'root'."""
        f2s = getattr(self, "_viat_frame_to_split", None)
        if f2s and 0 <= self.current_frame < len(f2s):
            return f2s[self.current_frame]
        return "root"

    @log_exceptions
    def remove_bad_frame(self):
        """Discard the current frame: move image + label to discarded/."""
        if not self._viat_ensure_dataset():
            return
        if not (0 <= self.current_frame < self.total_frames):
            return
        fname = os.path.basename(self.image_files[self.current_frame])
        reply = QMessageBox.question(
            self, "Remove Bad Frame",
            f"Move this frame to the 'discarded' folder?\n\n"
            f"  {fname} (split: {self.viat_current_split()})\n\n"
            f"Image + label will be moved on disk (reversible).",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        result = _viat_remove_bad_frames(self, [self.current_frame])
        # Stay on the same index (now the next image) or clamp.
        if self.current_frame >= self.total_frames:
            self.current_frame = max(0, self.total_frames - 1)
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.load_current_image()
        self.update_frame_info()
        self.update_annotation_list()
        self.statusBar.showMessage(
            f"Discarded 1 frame -> {result['discarded_dir']} "
            f"({result['moved_images']} img, {result['moved_labels']} lbl)",
            5000,
        )

    @log_exceptions
    def remove_bad_frames_dialog(self):
        """Discard a range of frames (batch)."""
        if not self._viat_ensure_dataset():
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Remove Bad Frames (batch)")
        dialog.setMinimumWidth(360)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        start_spin = QSpinBox(); start_spin.setRange(0, max(0, self.total_frames - 1))
        start_spin.setValue(self.current_frame)
        end_spin = QSpinBox(); end_spin.setRange(0, max(0, self.total_frames - 1))
        end_spin.setValue(self.current_frame)
        form.addRow("Start frame:", start_spin)
        form.addRow("End frame:", end_spin)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return
        lo, hi = sorted([start_spin.value(), end_spin.value()])
        frames = list(range(lo, hi + 1))
        reply = QMessageBox.question(
            self, "Remove Bad Frames",
            f"Move {len(frames)} frames ({lo}..{hi}) to discarded/?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        result = _viat_remove_bad_frames(self, frames)
        if self.current_frame >= self.total_frames:
            self.current_frame = max(0, self.total_frames - 1)
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.load_current_image()
        self.update_frame_info()
        self.update_annotation_list()
        self.statusBar.showMessage(
            f"Discarded {result['moved_images']} images / "
            f"{result['moved_labels']} labels -> {result['discarded_dir']}",
            5000,
        )

    @log_exceptions
    def remap_class_dialog(self):
        """Rename a class everywhere (in-memory + on-disk labels)."""
        if not self._viat_ensure_dataset():
            return
        classes = sorted(getattr(self.canvas, "class_colors", {}).keys())
        if not classes:
            QMessageBox.information(self, "Remap Class", "No classes loaded.")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Remap Class")
        dialog.setMinimumWidth(380)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        old_combo = QComboBox(); old_combo.addItems(classes)
        new_edit = QLineEdit(); new_edit.setPlaceholderText("new class name")
        rewrite_check = QCheckBox("Also rewrite label files on disk"); rewrite_check.setChecked(True)
        form.addRow("Old class:", old_combo)
        form.addRow("New name:", new_edit)
        form.addRow("", rewrite_check)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return
        old = old_combo.currentText()
        new = new_edit.text().strip()
        if not new or old == new:
            return
        result = _viat_remap_class(self, old, new, rewrite_disk=rewrite_check.isChecked())
        self.refresh_class_ui()
        self.update_annotation_list()
        self.statusBar.showMessage(
            f"Remapped '{old}' -> '{new}': "
            f"{result['changed_boxes']} boxes in {result['changed_frames']} frames, "
            f"{result['disk_files_rewritten']} files rewritten.",
            6000,
        )

    @log_exceptions
    def merge_classes_dialog(self):
        """Merge several classes into one (e.g. car+truck -> vehicle)."""
        if not self._viat_ensure_dataset():
            return
        classes = sorted(getattr(self.canvas, "class_colors", {}).keys())
        if len(classes) < 2:
            QMessageBox.information(self, "Merge Classes", "Need at least 2 classes.")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Merge Classes")
        dialog.setMinimumWidth(420)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Select classes to merge INTO a single class:"))
        checks = []
        for c in classes:
            cb = QCheckBox(c); checks.append((c, cb)); layout.addWidget(cb)
        form = QFormLayout()
        new_edit = QLineEdit(); new_edit.setPlaceholderText("target class name")
        form.addRow("Merge into:", new_edit)
        rewrite_check = QCheckBox("Also rewrite label files on disk"); rewrite_check.setChecked(True)
        form.addRow("", rewrite_check)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return
        old_names = [c for c, cb in checks if cb.isChecked()]
        new = new_edit.text().strip()
        if not new or not old_names:
            return
        if new in old_names:
            old_names = [c for c in old_names if c != new]
        result = _viat_merge_classes(self, old_names, new, rewrite_disk=rewrite_check.isChecked())
        self.refresh_class_ui()
        self.update_annotation_list()
        self.statusBar.showMessage(
            f"Merged {len(old_names)} classes into '{new}': "
            f"{result['changed_boxes']} boxes, {result['disk_files_rewritten']} files rewritten.",
            6000,
        )

    @log_exceptions
    def viat_dataset_stats(self):
        """Show a quick dataset statistics dialog."""
        info = self._viat_ensure_dataset()
        if not info:
            return
        # per-split counts + per-class counts
        per_split = {}
        per_class = {}
        f2s = getattr(self, "_viat_frame_to_split", []) or []
        for fidx, anns in self.frame_annotations.items():
            sp = f2s[fidx] if fidx < len(f2s) else "root"
            per_split[sp] = per_split.get(sp, 0) + 1
            for a in anns:
                per_class[a.class_name] = per_class.get(a.class_name, 0) + 1
        lines = [f"Dataset: {os.path.basename(info.root)}", f"Layout: {info.layout}", ""]
        lines.append("Splits (annotated frames):")
        for s in info.splits:
            lines.append(f"  {s.name}: {per_split.get(s.name, 0)}/{len(s.images)} frames, format={s.label_format}")
        lines.append("")
        lines.append("Classes (box counts):")
        for c, n in sorted(per_class.items(), key=lambda x: -x[1]):
            lines.append(f"  {c}: {n}")
        lines.append("")
        lines.append(f"Classes source: {info.classes_source}")
        if info.classes_conflict:
            lines.append(f"⚠ {info.classes_conflict}")
            
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Dataset Statistics")
        msg_box.setText("\n".join(lines))
        msg_box.setIcon(QMessageBox.Information)
        
        plot_btn = msg_box.addButton("Plot Details", QMessageBox.ActionRole)
        msg_box.addButton(QMessageBox.Ok)
        
        msg_box.exec_()
        
        if msg_box.clickedButton() == plot_btn:
            try:
                import matplotlib.pyplot as plt
                
                plt.figure(figsize=(10, 5))
                
                # Subplot 1: Classes
                plt.subplot(1, 2, 1)
                classes = list(per_class.keys())
                counts = list(per_class.values())
                plt.bar(classes, counts, color='skyblue')
                plt.xlabel('Classes')
                plt.ylabel('Box Count')
                plt.title('Annotations per Class')
                plt.xticks(rotation=45, ha='right')
                
                # Subplot 2: Splits
                plt.subplot(1, 2, 2)
                splits = list(per_split.keys())
                split_counts = list(per_split.values())
                plt.bar(splits, split_counts, color='lightgreen')
                plt.xlabel('Splits')
                plt.ylabel('Annotated Frames')
                plt.title('Frames per Split')
                plt.xticks(rotation=45, ha='right')
                
                plt.tight_layout()
                plt.show()
            except ImportError:
                QMessageBox.warning(self, "Plotting Error", "Matplotlib is not installed. Cannot plot details.")
            except Exception as e:
                QMessageBox.warning(self, "Plotting Error", f"An error occurred while plotting: {e}")

    # -----------------------------------------------------------------
    # New dataset ops (patch3): move, grayscale, duplicates, class filter,
    #                           segmentation toggle, dataset log viewer
    # -----------------------------------------------------------------

    @log_exceptions
    def viat_move_current_to_removed(self):
        """Move the current frame to removed/ (image + label on disk).

        No confirmation dialog -- the user pressed Shift+X / Ctrl+X intentionally.
        """
        if not self._viat_ensure_dataset():
            return
        if not (0 <= self.current_frame < self.total_frames):
            return
        result = _viat_move_to_removed(self, [self.current_frame])
        if self.current_frame >= self.total_frames:
            self.current_frame = max(0, self.total_frames - 1)
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.load_current_image()
        self.update_frame_info()
        self.update_annotation_list()
        self.statusBar.showMessage(
            f"Moved to {result['dest_dir']} ({result['moved_images']} img, "
            f"{result['moved_labels']} lbl)", 3000,
        )

    @log_exceptions
    def viat_move_current_to_review_label(self):
        """Move the current frame to review_label/ (the CHANGE LABEL queue)."""
        if not self._viat_ensure_dataset():
            return
        if not (0 <= self.current_frame < self.total_frames):
            return
        fname = os.path.basename(self.image_files[self.current_frame])
        result = _viat_move_to_review_label(self, [self.current_frame])
        if self.current_frame >= self.total_frames:
            self.current_frame = max(0, self.total_frames - 1)
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.load_current_image()
        self.update_frame_info()
        self.update_annotation_list()
        self.statusBar.showMessage(
            f"Moved to review_label/ for label review: {fname}", 5000,
        )

    @log_exceptions
    def viat_remove_grayscale(self):
        """Detect and move all grayscale images to removed/grayscale/."""
        if not self._viat_ensure_dataset():
            return
        reply = QMessageBox.question(
            self, "Remove Grayscale Images",
            "Scan all images and move grayscale ones to removed/grayscale/?\n\n"
            "This may take a moment for large datasets.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        self.statusBar.showMessage("Scanning for grayscale images...")
        QApplication.processEvents()
        result = _viat_remove_grayscale(self)
        if self.current_frame >= self.total_frames:
            self.current_frame = max(0, self.total_frames - 1)
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.load_current_image()
        self.update_frame_info()
        self.update_annotation_list()
        self.statusBar.showMessage(
            f"Removed {result['moved_images']} grayscale images -> {result['dest_dir']}",
            5000,
        )

    @log_exceptions
    def viat_remove_duplicates(self):
        """Remove Roboflow duplicate groups (keep 1 per .rf. group)."""
        if not self._viat_ensure_dataset():
            return
        reply = QMessageBox.question(
            self, "Remove Roboflow Duplicates",
            "Group images by their Roboflow base name (before .rf.) and\n"
            "move duplicates to removed/duplicates/ (keeping 1 random per group)?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        result = _viat_remove_dup_groups(self)
        if self.current_frame >= self.total_frames:
            self.current_frame = max(0, self.total_frames - 1)
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.load_current_image()
        self.update_frame_info()
        self.update_annotation_list()
        self.statusBar.showMessage(
            f"Removed {result['moved_images']} duplicates from "
            f"{result['groups_processed']} groups -> {result['dest_dir']}",
            5000,
        )

    @log_exceptions
    def viat_remove_class_and_images_dialog(self):
        """Move all frames whose labels contain a selected class, or just remove the labels."""
        if not self._viat_ensure_dataset():
            return
        classes = sorted(getattr(self.canvas, "class_colors", {}).keys())
        if not classes:
            QMessageBox.information(self, "Remove Class", "No classes loaded.")
            return
        dialog = QDialog(self)
        dialog.setWindowTitle("Remove Class")
        dialog.setMinimumWidth(380)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel("Select classes to remove:"))
        checks = []
        for c in classes:
            cb = QCheckBox(c)
            checks.append((c, cb))
            layout.addWidget(cb)
            
        remove_images_cb = QCheckBox("Remove images as well (move frames to removed/class_filtered/)")
        remove_images_cb.setChecked(True)
        remove_images_cb.setStyleSheet("""
            QCheckBox {
                font-weight: bold;
                color: #e74c3c;
                margin-top: 10px;
                margin-bottom: 5px;
            }
        """)
        layout.addWidget(remove_images_cb)
        
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return
        selected = [c for c, cb in checks if cb.isChecked()]
        if not selected:
            return
            
        remove_images = remove_images_cb.isChecked()
        
        if remove_images:
            reply = QMessageBox.question(
                self, "Remove Class + Images",
                f"Move all frames containing classes {selected} to removed/class_filtered/?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
        else:
            reply = QMessageBox.question(
                self, "Remove Class Labels",
                f"Remove labels for classes {selected} from all frames (keep images)?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            
        if reply != QMessageBox.Yes:
            return
            
        result = _viat_remove_class_and_images(self, selected, remove_images=remove_images)
        if self.current_frame >= self.total_frames:
            self.current_frame = max(0, self.total_frames - 1)
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.load_current_image()
        self.update_frame_info()
        self.update_annotation_list()
        
        if remove_images:
            self.statusBar.showMessage(
                f"Moved {result.get('moved_images', 0)} frames (classes={selected}) -> {result.get('dest_dir', '')}",
                5000,
            )
        else:
            self.statusBar.showMessage(
                f"Removed {result.get('removed_boxes', 0)} labels for {selected} in {result.get('affected_frames', 0)} frames.",
                5000,
            )

    @log_exceptions
    def viat_toggle_segmentation(self):
        """Toggle showing segmentation polygon outlines on the canvas."""
        self.canvas.show_segmentation = not getattr(self.canvas, "show_segmentation", False)
        self.canvas.update()
        state = "ON" if self.canvas.show_segmentation else "OFF"
        self.statusBar.showMessage(f"Segmentation display: {state}", 3000)

    @log_exceptions
    def viat_view_dataset_log(self):
        """Open the DATASET_LOG.md file in the system text editor."""
        info = getattr(self, "_viat_dataset_info", None)
        if not info:
            QMessageBox.warning(self, "Dataset Log", "No dataset loaded.")
            return
        log_path = os.path.join(info.root, "DATASET_LOG.md")
        if not os.path.isfile(log_path):
            # Create it now
            try:
                _viat_init_dataset_log(self, info)
            except Exception:
                pass
        if os.path.isfile(log_path):
            # Read and show in a dialog
            content = open(log_path, "r", encoding="utf-8").read()
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Dataset Log — {os.path.basename(info.root)}")
            dialog.setMinimumWidth(600)
            dialog.setMinimumHeight(500)
            layout = QVBoxLayout(dialog)
            text_widget = QPlainTextEdit()
            text_widget.setPlainText(content)
            text_widget.setReadOnly(True)
            layout.addWidget(text_widget)
            buttons = QDialogButtonBox(QDialogButtonBox.Close)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            dialog.exec_()
        else:
            QMessageBox.warning(self, "Dataset Log", "Could not create DATASET_LOG.md")

    # -----------------------------------------------------------------
    # Video annotation features (patch4)
    # -----------------------------------------------------------------

    @log_exceptions
    def viat_import_video_json(self):
        """Import a VIAT custom JSON annotation file.

        Works for both video and image-dataset projects. If no media is
        open, infers total_frames from the JSON's max frame key.
        """
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import VIAT JSON Annotations", "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not filename:
            return

        # If no media is open, infer total_frames from the JSON
        if not self.cap and not self.is_image_dataset:
            try:
                import json
                with open(filename, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and data:
                    max_frame = max(int(k) for k in data.keys() if str(k).isdigit())
                    self.total_frames = max_frame + 1
                    self.frame_slider.setMaximum(max(0, self.total_frames - 1))
            except Exception as e:
                QMessageBox.warning(self, "Import JSON",
                    f"Could not read JSON file:\n{e}\n\n"
                    f"Make sure it's a VIAT JSON (frame-number keys with 'actors').")
                return

        self.save_undo_state()
        try:
            result = _viat_load_json_video(self, filename, BoundingBox)
        except Exception as e:
            QMessageBox.critical(self, "Import JSON",
                f"Failed to parse JSON:\n{e}\n\n"
                f"The file must be in VIAT JSON format:"
                f"  \"0000\": {{\"actors\": {{...}}}}")
            return
        self.refresh_class_ui()
        self.update_annotation_list()
        if self.current_frame in self.frame_annotations:
            self.canvas.annotations = self.frame_annotations[self.current_frame]
            self.canvas.update()
        if result['frames_loaded'] == 0:
            QMessageBox.warning(self, "Import JSON",
                "No frames were loaded. The file may not be in VIAT JSON format\n"
                "(expected: {\"0000\": {\"actors\": {...}}})")
            return
        msg = (
            f"Imported {result['frames_loaded']} frames, "
            f"{result['actors_loaded']} annotations, "
            f"{len(result['classes_found'])} classes. "
            f"Actors: {', '.join(result.get('actor_ids', [])[:5])}"
        )
        if result.get('warnings'):
            msg += f"  ({len(result['warnings'])} warnings)"
        self.statusBar.showMessage(msg, 8000)

    @log_exceptions
    def viat_detect_and_fix_borders(self):
        """Detect video borders and adjust (remove/clip) labels in the border."""
        if not self.cap and not self.is_image_dataset:
            QMessageBox.warning(self, "Video Borders", "Open a video first.")
            return
        self.statusBar.showMessage("Detecting video borders...")
        QApplication.processEvents()
        self.save_undo_state()
        result = _viat_detect_adjust_borders(self)
        if not result.get('detected', False):
            QMessageBox.information(
                self, "Video Borders",
                "No black/gray borders detected on the left or right edges.",
            )
            self.statusBar.showMessage("No borders detected", 3000)
            return
        det = result['detection']
        adj = result['adjustment']
        self.canvas.update()
        self.update_annotation_list()
        msg = (
            f"Borders: left={det['left_border']}px, right={det['right_border']}px "
            f"(sampled {det['sampled']} frames).\n\n"
            f"Annotations: {adj['removed']} removed (≥80% in border), "
            f"{adj['clipped']} clipped, {adj['unchanged']} unchanged."
        )
        QMessageBox.information(self, "Video Borders Adjusted", msg)
        self.statusBar.showMessage(
            f"Borders fixed: {adj['removed']} removed, {adj['clipped']} clipped", 5000
        )

    @log_exceptions
    def viat_start_object_visibility_mode(self):
        """Enter per-object visibility editing mode (toolbar-based, no dialog)."""
        if not self.frame_annotations:
            QMessageBox.warning(self, "Object Visibility", "No annotations loaded.")
            return
        if self.object_visibility_manager and self.object_visibility_manager.active:
            return
        from utils.object_visibility import ObjectVisibilityManager
        self.object_visibility_manager = ObjectVisibilityManager(self)
        if not self.object_visibility_manager.start():
            QMessageBox.warning(self, "Object Visibility", "No objects found (need actor_id attributes).")
            self.object_visibility_manager = None
            return
        self._viat_show_object_visibility_toolbar()

    @log_exceptions
    def viat_exit_object_visibility_mode(self):
        """Exit per-object visibility editing mode."""
        if self.object_visibility_manager and self.object_visibility_manager.active:
            self.object_visibility_manager.exit()
        self.object_visibility_manager = None
        self._viat_hide_object_visibility_toolbar()
        self.statusBar.showMessage("Exited object visibility mode", 3000)

    def _viat_show_object_visibility_toolbar(self):
        """Create a toolbar with object-visibility controls (no dialog)."""
        from PyQt5.QtWidgets import QToolBar, QLabel, QPushButton, QWidget, QHBoxLayout
        mgr = self.object_visibility_manager
        if not mgr or not mgr.active:
            return

        # Remove old toolbar if it exists
        if hasattr(self, '_viat_visibility_toolbar'):
            self.removeToolBar(self._viat_visibility_toolbar)

        toolbar = QToolBar("Object Visibility", self)
        toolbar.setObjectName("ObjectVisibilityToolbar")
        self.addToolBar(toolbar)

        # Status labels
        self._viat_obj_label = QLabel()
        toolbar.addWidget(self._viat_obj_label)

        toolbar.addSeparator()

        self._viat_range_label = QLabel()
        toolbar.addWidget(self._viat_range_label)

        toolbar.addSeparator()

        # Buttons
        btn_prev_obj = QPushButton("< Prev Obj")
        btn_next_obj = QPushButton("Next Obj >")
        btn_prev_range = QPushButton("< Prev Range")
        btn_next_range = QPushButton("Next Range >")
        btn_set_start = QPushButton("Set Start")
        btn_set_end = QPushButton("Set End")
        btn_del_frame = QPushButton("Delete Frame")
        btn_del_obj = QPushButton("Delete Object")
        btn_finish = QPushButton("FINISH")
        btn_finish.setStyleSheet("font-weight: bold; background-color: #4CAF50; color: white;")
        btn_exit = QPushButton("Exit")

        for btn in [btn_prev_obj, btn_next_obj, btn_prev_range, btn_next_range,
                    btn_set_start, btn_set_end, btn_del_frame, btn_del_obj,
                    btn_finish, btn_exit]:
            toolbar.addWidget(btn)

        # Store widgets for updates
        mgr.toolbar_widgets = {
            'obj_label': self._viat_obj_label,
            'range_label': self._viat_range_label,
            'prev_obj': btn_prev_obj, 'next_obj': btn_next_obj,
            'prev_range': btn_prev_range, 'next_range': btn_next_range,
            'set_start': btn_set_start, 'set_end': btn_set_end,
            'del_frame': btn_del_frame, 'del_obj': btn_del_obj,
            'finish': btn_finish, 'exit': btn_exit,
        }

        def update_labels():
            s = mgr.get_status()
            self._viat_obj_label.setText(
                f"Object: {s['current_object']} ({s['object_index']+1}/{s['total_objects']})"
            )
            r = s['current_range']
            if r:
                self._viat_range_label.setText(
                    f"Range: {r[0]}-{r[1]} ({s['range_index']+1}/{s['total_ranges']})"
                )
            else:
                self._viat_range_label.setText("No ranges")

        def on_prev_obj():
            mgr.prev_object()
            update_labels()

        def on_next_obj():
            mgr.next_object()
            update_labels()

        def on_prev_range():
            mgr.prev_range()
            update_labels()

        def on_next_range():
            mgr.next_range()
            update_labels()

        def on_set_start():
            mgr.trim_current_frame_as_start()
            update_labels()

        def on_set_end():
            mgr.trim_current_frame_as_end()
            update_labels()

        def on_del_frame():
            n = mgr.delete_current_object_on_current_frame()
            update_labels()
            self.statusBar.showMessage(f"Deleted {n} annotation(s) on this frame", 2000)

        def on_del_obj():
            reply = QMessageBox.question(
                self, "Delete Object",
                f"Delete ALL annotations for '{mgr.current_object}' across all frames?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                n = mgr.delete_object()
                if not mgr.sorted_objects:
                    self.viat_exit_object_visibility_mode()
                    return
                update_labels()
                self.statusBar.showMessage(f"Deleted {n} annotations", 3000)

        def on_finish():
            if mgr.next_object():
                update_labels()
            else:
                QMessageBox.information(self, "All Done", "All objects have been processed.")
                self.viat_exit_object_visibility_mode()

        def on_exit():
            self.viat_exit_object_visibility_mode()

        btn_prev_obj.clicked.connect(on_prev_obj)
        btn_next_obj.clicked.connect(on_next_obj)
        btn_prev_range.clicked.connect(on_prev_range)
        btn_next_range.clicked.connect(on_next_range)
        btn_set_start.clicked.connect(on_set_start)
        btn_set_end.clicked.connect(on_set_end)
        btn_del_frame.clicked.connect(on_del_frame)
        btn_del_obj.clicked.connect(on_del_obj)
        btn_finish.clicked.connect(on_finish)
        btn_exit.clicked.connect(on_exit)

        self._viat_visibility_toolbar = toolbar
        self._viat_update_visibility_labels = update_labels
        update_labels()

    def _viat_hide_object_visibility_toolbar(self):
        """Remove the object visibility toolbar."""
        if hasattr(self, '_viat_visibility_toolbar'):
            self.removeToolBar(self._viat_visibility_toolbar)
            self._viat_visibility_toolbar.deleteLater()
            del self._viat_visibility_toolbar

    @log_exceptions
    def viat_seg_video_pick_color(self):
        """Pick a color from the canvas by CLICKING on it (no X/Y dialog).

        Enables canvas color-pick mode. The user clicks on the canvas, the
        color at that point is extracted, and a small non-blocking dialog
        asks for the class name + actor ID.
        """
        if not self.cap and not self.is_image_dataset:
            QMessageBox.warning(self, "Seg Video", "Open a video first.")
            return

        # Enable color-pick mode on the canvas
        self.canvas.color_pick_mode = True
        self.canvas.setCursor(Qt.CrossCursor)
        self.statusBar.showMessage("Click on an object in the canvas to pick its color...", 0)

        def on_color_picked(x, y):
            """Called when the user clicks on the canvas."""
            # Get the current frame
            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    return
            elif self.is_image_dataset and self.image_files:
                frame = cv2.imread(self.image_files[self.current_frame])
            else:
                return

            from utils.seg_video_labeler import SegmentationVideoLabeler
            tmp_labeler = SegmentationVideoLabeler(self)
            color_hsv = tmp_labeler.pick_color_from_frame(frame, x, y)
            self.statusBar.showMessage(f"Picked color at ({x},{y}): HSV={color_hsv}", 5000)

            # Show a small non-blocking dialog for class name + actor ID
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Add Tracked Object (HSV={color_hsv})")
            dialog.setMinimumWidth(350)
            from PyQt5.QtWidgets import QVBoxLayout, QFormLayout, QDialogButtonBox
            layout = QVBoxLayout(dialog)
            form = QFormLayout()

            from PyQt5.QtWidgets import QLineEdit, QSpinBox
            class_edit = QLineEdit()
            class_edit.setPlaceholderText("class name (e.g. Helicopter)")
            form.addRow("Class:", class_edit)

            actor_edit = QLineEdit()
            actor_edit.setPlaceholderText("auto if empty")
            form.addRow("Actor ID:", actor_edit)

            tol_spin = QSpinBox(); tol_spin.setRange(1, 60); tol_spin.setValue(15)
            form.addRow("Tolerance:", tol_spin)

            min_area_spin = QSpinBox(); min_area_spin.setRange(1, 99999); min_area_spin.setValue(100)
            form.addRow("Min area:", min_area_spin)

            from PyQt5.QtWidgets import QCheckBox
            merge_all_cb = QCheckBox("Create one big bounding box (Merge regions)")
            merge_all_cb.setChecked(False) # Default to false since user wants to track instances
            form.addRow("", merge_all_cb)

            connect_spin = QSpinBox(); connect_spin.setRange(0, 500); connect_spin.setValue(10)
            connect_spin.setToolTip("Pixel distance to connect small disconnected regions")
            form.addRow("Connect Threshold:", connect_spin)

            layout.addLayout(form)

            buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)

            if dialog.exec_() == QDialog.Accepted:
                class_name = class_edit.text().strip()
                if not class_name:
                    return
                if self.seg_labeler is None:
                    from utils.seg_video_labeler import SegmentationVideoLabeler
                    self.seg_labeler = SegmentationVideoLabeler(self)
                self.seg_labeler.add_tracked_object(
                    color_hsv=color_hsv,
                    class_name=class_name,
                    actor_id=actor_edit.text().strip() or None,
                    tolerance=tol_spin.value(),
                    min_area=min_area_spin.value(),
                    merge_all=merge_all_cb.isChecked(),
                    connect_threshold=connect_spin.value(),
                )
                self.statusBar.showMessage(
                    f"Added '{class_name}' (HSV={color_hsv}). Total: {len(self.seg_labeler.tracked_objects)}. "
                    f"Click another object or track all.", 5000,
                )

        self.canvas.color_pick_callback = on_color_picked

    @log_exceptions
    def viat_seg_video_track_all(self):
        """Track all picked objects across the video and add as annotations."""
        if self.seg_labeler is None or not self.seg_labeler.tracked_objects:
            QMessageBox.warning(self, "Seg Video", "No tracked objects. Pick a color first.")
            return
        if not self.video_filename and not self.cap:
            QMessageBox.warning(self, "Seg Video", "No video loaded.")
            return
        video_path = self.video_filename
        # Track
        self.statusBar.showMessage(f"Tracking {len(self.seg_labeler.tracked_objects)} objects...")
        QApplication.processEvents()
        result = self.seg_labeler.track_all_objects(
            video_path,
            start_frame=0,
            end_frame=self.total_frames - 1,
            progress_callback=lambda cur, tot: self.statusBar.showMessage(
                f"Tracking... {cur}/{tot} frames", 0
            ),
        )
        # Commit to frame_annotations
        added = self.seg_labeler.commit_to_app(self, BoundingBox)
        self.refresh_class_ui()
        self.update_annotation_list()
        if self.current_frame in self.frame_annotations:
            self.canvas.annotations = self.frame_annotations[self.current_frame]
            self.canvas.update()
        msg = (
            f"Tracked {len(self.seg_labeler.tracked_objects)} objects across "
            f"{result['frames_processed']} frames. Added {added} annotations.\n"
            f"Per object: {result['per_object']}"
        )
        QMessageBox.information(self, "Seg Video Tracking Complete", msg)
        self.statusBar.showMessage(f"Added {added} annotations from seg video", 5000)

    @log_exceptions
    def viat_seg_video_export_json(self):
        """Export tracked seg-video objects to VIAT JSON."""
        if self.seg_labeler is None or not self.seg_labeler.tracked_objects:
            QMessageBox.warning(self, "Seg Video", "No tracked objects.")
            return
        default_name = "seg_annotations.json"
        if self.video_filename:
            base = os.path.splitext(os.path.basename(self.video_filename))[0]
            default_name = base + "_seg.json"
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export Seg Video JSON", default_name,
            "JSON Files (*.json);;All Files (*)",
        )
        if not filename:
            return
        content = self.seg_labeler.to_viat_json()
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        self.statusBar.showMessage(f"Exported seg video JSON to {filename}", 5000)

    @log_exceptions
    def viat_export_json(self):
        """Export all current annotations to VIAT custom JSON format."""
        # Ensure current frame's annotations are synchronized back to self.frame_annotations
        self.frame_annotations[self.current_frame] = self.canvas.annotations.copy()

        # Check if we have any annotations
        has_annotations = any(self.frame_annotations.values())
        if not has_annotations:
            QMessageBox.warning(self, "Export VIAT JSON", "No annotations to export!")
            return

        default_name = "annotations.json"
        default_dir = ""
        if self.video_filename:
            base = os.path.splitext(os.path.basename(self.video_filename))[0]
            default_name = base + "_viat.json"
            default_dir = os.path.dirname(self.video_filename)
        elif hasattr(self, "is_image_dataset") and self.is_image_dataset and self.image_files:
            image_folder = os.path.dirname(self.image_files[0])
            folder_name = os.path.basename(image_folder)
            default_name = folder_name + "_viat.json"
            default_dir = image_folder

        default_path = os.path.join(default_dir, default_name) if default_dir else default_name

        filename, _ = QFileDialog.getSaveFileName(
            self, "Export VIAT JSON", default_path,
            "JSON Files (*.json);;All Files (*)",
        )
        if not filename:
            return

        boxes_by_frame = {}
        for frame_num, annotations in self.frame_annotations.items():
            if not annotations:
                continue
            boxes_by_frame[frame_num] = []
            for ann in annotations:
                box_dict = {
                    "class_name": ann.class_name,
                    "x": ann.rect.x(),
                    "y": ann.rect.y(),
                    "w": ann.rect.width(),
                    "h": ann.rect.height(),
                    "verified": getattr(ann, "verified", False),
                    "segmentation": getattr(ann, "segmentation", None),
                }
                actor_id = ann.attributes.get("actor_id") if hasattr(ann, "attributes") else None
                if actor_id:
                    box_dict["actor_id"] = actor_id
                boxes_by_frame[frame_num].append(box_dict)

        try:
            from utils.label_formats.viat_json import ViatJsonLabelFormat
            fmt = ViatJsonLabelFormat()
            content = fmt.dump(boxes_by_frame, (0, 0), [])
            with open(filename, "w", encoding="utf-8") as f:
                f.write(content)
            self.statusBar.showMessage(f"Exported VIAT JSON to {filename}", 5000)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to export VIAT JSON: {e}")

    @log_exceptions
    def viat_perf_stats(self):
        """Show frame cache statistics."""
        if not hasattr(self, "viat_perf"):
            QMessageBox.information(self, "Performance", "No performance manager.")
            return
        stats = self.viat_perf.get_stats()
        QMessageBox.information(self, "Frame Cache Stats",
            f"Cache: {stats['cache_size']}/{stats['cache_capacity']} frames\n"
            f"Hit rate: {stats['hit_rate']}\n"
            f"Hits: {stats['hits']}, Misses: {stats['misses']}")

    @log_exceptions
    def viat_clear_cache(self):
        """Clear the frame cache."""
        if hasattr(self, "viat_perf"):
            self.viat_perf.clear_cache()
            self.statusBar.showMessage("Frame cache cleared", 2000)

    @log_exceptions
    def viat_auto_import_detections(self):
        """Auto-import a YOLO detections JSON and move flagged images to review_label/.

        Workflow:
          1. Run yolo_detect.py separately (produces _viat_detections.json).
          2. In VIAT: Dataset > Auto-Import Detections... -> select the JSON.
          3. Images WITH detections are moved to review_label/ and the
             detections are added as unverified annotations.
        """
        if not self._viat_ensure_dataset():
            return
        filename, _ = QFileDialog.getOpenFileName(
            self, "Select Detections JSON", "",
            "JSON Files (*.json);;All Files (*)",
        )
        if not filename:
            return

        reply = QMessageBox.question(
            self, "Auto-Import Detections",
            f"Import detections from:\n  {os.path.basename(filename)}\n\n"
            f"Images with detections will be:\n"
            f"  - moved to review_label/\n"
            f"  - detections added as unverified annotations\n\n"
            f"Continue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if reply != QMessageBox.Yes:
            return

        result = _viat_auto_import_detections(
            self, filename,
            move_to_review=True,
            add_as_annotations=False,
            bbox_cls=BoundingBox,
        )

        # Refresh UI
        if self.current_frame >= self.total_frames:
            self.current_frame = max(0, self.total_frames - 1)
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.load_current_image()
        self.update_frame_info()
        self.update_annotation_list()
        self.refresh_class_ui()

        json_info = ""
        if result.get("json_copied_to"):
            json_info = f"\n  Detections JSON copied to: review_label/_viat_detections.json"
        msg = (
            f"Auto-import complete:\n"
            f"  Flagged images: {result['flagged_images']}\n"
            f"  Moved to review_label/: {result['moved_images']}\n"
            f"  Total detections in JSON: {result['total_detections']}"
            f"{json_info}\n\n"
            f"To review: open the review_label/ folder as a dataset, then\n"
            f"use Dataset > Import VIAT JSON Annotations to load the detections."
        )
        QMessageBox.information(self, "Auto-Import Detections", msg)
        self.statusBar.showMessage(
            f"Moved {result['moved_images']} images to review_label/", 5000,
        )

    @log_exceptions
    def viat_merge_dataset(self, source_folder=None, target_folder=None):
        """Merge another dataset into the current (target) dataset."""
        if not target_folder:
            if not self._viat_ensure_dataset():
                return
            target_folder = self._viat_dataset_info.root

        # Select source dataset if not provided
        if not source_folder:
            source_folder = QFileDialog.getExistingDirectory(
                self, "Select Source Dataset to Merge", "", QFileDialog.ShowDirsOnly
            )
            if not source_folder:
                return

        # Default dataset name = source folder name
        default_name = os.path.basename(source_folder)

        # Show dialog: dataset name, split mode, class mapping
        from PyQt5.QtWidgets import (
            QDialog, QVBoxLayout, QFormLayout, QLineEdit, QComboBox,
            QDialogButtonBox, QLabel, QSpinBox, QTableWidget, QTableWidgetItem,
            QHeaderView,
        )
        dialog = QDialog(self)
        dialog.setWindowTitle("Merge Dataset")
        dialog.setMinimumWidth(550)
        layout = QVBoxLayout(dialog)

        form = QFormLayout()
        name_edit = QLineEdit(default_name)
        form.addRow("Dataset name (file prefix):", name_edit)

        split_combo = QComboBox()
        split_combo.addItems([
            "Keep original splits",
            "Random split (specify %)",
            "All in train",
            "All in valid",
            "All in test",
        ])
        form.addRow("Split mode:", split_combo)

        valid_spin = QSpinBox(); valid_spin.setRange(1, 50); valid_spin.setValue(10)
        form.addRow("Valid % (if random):", valid_spin)

        layout.addLayout(form)

        # Check for unmatched classes
        check_result = _viat_find_unmatched_classes(source_folder, target_folder)
        unmatched = check_result['unmatched']
        target_classes = check_result['target_classes']

        if unmatched:
            layout.addWidget(QLabel(
                f"The following source classes have no match in the target dataset.\n"
                f"Map each to a target class (or type a new name to create it):"
            ))
            table = QTableWidget(len(unmatched), 2)
            table.setHorizontalHeaderLabels(["Source class", "Map to (target class)"])
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            for i, src_cls in enumerate(unmatched):
                table.setItem(i, 0, QTableWidgetItem(src_cls))
                combo_text = target_classes[0] if target_classes else src_cls
                table.setItem(i, 1, QTableWidgetItem(combo_text))
            layout.addWidget(table)
        else:
            table = None

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() != QDialog.Accepted:
            return

        dataset_name = name_edit.text().strip() or default_name
        split_mode_map = {
            0: "keep", 1: "random", 2: "all_train", 3: "all_valid", 4: "all_test"
        }
        split_mode = split_mode_map[split_combo.currentIndex()]

        # Build class mapping
        class_mapping = dict(check_result['matched'])
        if table:
            for i in range(table.rowCount()):
                src = table.item(i, 0).text()
                tgt = table.item(i, 1).text()
                class_mapping[src] = tgt

        # Progress
        self.statusBar.showMessage(f"Merging {dataset_name} into target...")
        QApplication.processEvents()

        result = _viat_merge_dataset(
            self, source_folder, target_folder,
            dataset_name=dataset_name,
            split_mode=split_mode,
            random_valid_pct=valid_spin.value(),
            class_mapping=class_mapping,
            progress_callback=lambda cur, tot, msg: self.statusBar.showMessage(f"Merging {cur}/{tot}: {msg}", 0),
        )

        if result.get('error'):
            QMessageBox.warning(self, "Merge Error", result['error'])
            return

        msg = (
            f"Merge complete:\n"
            f"  Images copied: {result['images_copied']}\n"
            f"  Labels copied: {result['labels_copied']}\n"
            f"  Classes mapped: {result['classes_mapped']}\n"
            f"  Skipped: {result['skipped']}\n"
            f"  Dataset name: {result['dataset_name']}"
        )
        QMessageBox.information(self, "Merge Complete", msg)
        self.statusBar.showMessage(
            f"Merged {result['images_copied']} images from {dataset_name}", 5000
        )

    @log_exceptions
    def viat_extract_single_class_dataset(self):
        """Show the single class dataset extraction dialog."""
        from widgets.single_class_extractor_dialog import SingleClassExtractorDialog
        dialog = SingleClassExtractorDialog(self)
        dialog.exec_()

    @log_exceptions
    def viat_launch_batch_prediction_queue(self):
        """Show the batch prediction queue builder dialog."""
        from widgets.batch_prediction_dialog import BatchPredictionDialog
        dialog = BatchPredictionDialog(self)
        dialog.exec_()

    @log_exceptions
    def viat_remove_background_images(self):
        """Show the background remover dialog."""
        from widgets.background_remover_dialog import BackgroundRemoverDialog
        dialog = BackgroundRemoverDialog(self)
        dialog.exec_()

    @log_exceptions
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

    @log_exceptions
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

    #
    # Tracking Methods
    #
    @log_exceptions
    def toggle_tracking_mode(self, enabled):
        self.tracking_mode_enabled = enabled
        if enabled:
            self.add_track_id_to_bboxes()
            self.statusBar.showMessage("Tracking mode enabled: track_id will be auto-assigned", 3000)
        else:
            self.statusBar.showMessage("Tracking mode disabled: track_id hidden from UI", 3000)
        # Update UI to hide/show track_id
        self.update_annotation_list()
        self.canvas.update()

    def add_track_id_to_bboxes(self):
        """
        Add a 'track_id' attribute to all bounding boxes in all frames.
        Uses IoU tracking to maintain consistent IDs across frames, even when
        objects disappear and reappear later.
        """
        # First, count max objects per class in any single frame
        class_counts = {}
        for frame_num, annotations in self.frame_annotations.items():
            frame_class_counts = {}
            for ann in annotations:
                if hasattr(ann, "rect"):  # Only count bounding boxes
                    class_name = getattr(ann, "class_name", "unknown")
                    frame_class_counts[class_name] = frame_class_counts.get(class_name, 0) + 1
            
            # Update the max count for each class
            for class_name, count in frame_class_counts.items():
                class_counts[class_name] = max(class_counts.get(class_name, 0), count)
        
        # Reset all track IDs
        for frame_num, annotations in self.frame_annotations.items():
            for ann in annotations:
                if hasattr(ann, "rect"):
                    if not hasattr(ann, "attributes"):
                        ann.attributes = {}
                    if "track_id" in ann.attributes:
                        del ann.attributes["track_id"]
        
        # Also reset current frame's annotations on the canvas
        for ann in self.canvas.annotations:
            if hasattr(ann, "rect"):
                if not hasattr(ann, "attributes"):
                    ann.attributes = {}
                if "track_id" in ann.attributes:
                    del ann.attributes["track_id"]
        
        # Track object history across frames - keep objects' data even if they temporarily disappear
        # Format: {class_name: [{"track_id": id, "rect": QRect, "last_seen": frame_num, "active": bool}]}
        tracked_objects = {}
        updated_count = 0
        
        # Sort frames to ensure we start from the first frame
        sorted_frames = sorted(self.frame_annotations.keys())
        
        # Memory window: how many frames to remember disappeared objects for potential reappearance
        memory_window = 30  # Remember objects for 30 frames
        
        # Process frames in chronological order
        for frame_num in sorted_frames:
            annotations = self.frame_annotations[frame_num]
            
            # Group annotations by class
            class_annotations = {}
            for ann in annotations:
                if hasattr(ann, "rect"):  # Only process bounding boxes
                    if not hasattr(ann, "attributes"):
                        ann.attributes = {}
                    
                    class_name = getattr(ann, "class_name", "unknown")
                    if class_name not in class_annotations:
                        class_annotations[class_name] = []
                    class_annotations[class_name].append(ann)
                    
                    # Initialize tracking for new classes
                    if class_name not in tracked_objects:
                        tracked_objects[class_name] = []
                        if class_name not in class_counts:
                            class_counts[class_name] = 1  # At least 1 for new classes
            
            # Process each class separately
            for class_name, anns in class_annotations.items():
                # Mark all objects as unmatched initially
                matched_objs = set()
                matched_anns = set()
                
                # First pass: Match current annotations with recently active tracked objects
                # This prioritizes maintaining active tracks
                for i, obj in enumerate(tracked_objects[class_name]):
                    if not obj["active"]:
                        continue  # Skip inactive objects in first pass
                        
                    best_iou = 0.5  # IoU threshold
                    best_ann = None
                    best_ann_idx = -1
                    
                    for j, ann in enumerate(anns):
                        if j in matched_anns:
                            continue  # Skip already matched annotations
                        
                        iou_val = self.iou(ann.rect, obj["rect"])
                        if iou_val > best_iou:
                            best_iou = iou_val
                            best_ann = ann
                            best_ann_idx = j
                    
                    if best_ann is not None:
                        # Match found - update track
                        matched_objs.add(i)
                        matched_anns.add(best_ann_idx)
                        # Assign track_id to annotation
                        best_ann.attributes["track_id"] = obj["track_id"]
                        # Update the tracked object
                        obj["rect"] = best_ann.rect
                        obj["last_seen"] = frame_num
                        obj["active"] = True
                        updated_count += 1
                
                # Deactivate objects not seen in this frame
                for i, obj in enumerate(tracked_objects[class_name]):
                    if i not in matched_objs and obj["active"]:
                        obj["active"] = False
                
                # Second pass: Try to match remaining annotations with inactive objects (recently disappeared)
                for j, ann in enumerate(anns):
                    if j in matched_anns:
                        continue  # Skip already matched annotations
                    
                    best_iou = 0.4  # Lower threshold for reappearing objects
                    best_obj_idx = -1
                    
                    for i, obj in enumerate(tracked_objects[class_name]):
                        if i in matched_objs or obj["active"]:
                            continue  # Skip already matched or active objects
                        
                        # Only consider recently disappeared objects
                        if frame_num - obj["last_seen"] > memory_window:
                            continue
                        
                        iou_val = self.iou(ann.rect, obj["rect"])
                        if iou_val > best_iou:
                            best_iou = iou_val
                            best_obj_idx = i
                    
                    if best_obj_idx != -1:
                        # Found a reappearing object
                        matched_objs.add(best_obj_idx)
                        matched_anns.add(j)
                        # Assign track_id to annotation
                        ann.attributes["track_id"] = tracked_objects[class_name][best_obj_idx]["track_id"]
                        # Update the tracked object
                        tracked_objects[class_name][best_obj_idx]["rect"] = ann.rect
                        tracked_objects[class_name][best_obj_idx]["last_seen"] = frame_num
                        tracked_objects[class_name][best_obj_idx]["active"] = True
                        updated_count += 1
                
                # Third pass: Create new tracks for unmatched annotations
                for j, ann in enumerate(anns):
                    if j in matched_anns:
                        continue  # Skip already matched annotations
                    
                    # Find an available track_id
                    available_ids = set(range(class_counts[class_name]))
                    used_ids = {obj["track_id"] for obj in tracked_objects[class_name]}
                    available_ids -= used_ids
                    
                    if available_ids:
                        # Use first available ID
                        track_id = min(available_ids)
                    elif len(tracked_objects[class_name]) < class_counts[class_name]:
                        # If we haven't reached max objects yet, create a new ID
                        track_id = len(tracked_objects[class_name])
                    else:
                        # Recycle oldest inactive ID if no IDs are available
                        oldest_frame = float('inf')
                        oldest_idx = 0
                        for i, obj in enumerate(tracked_objects[class_name]):
                            if not obj["active"] and obj["last_seen"] < oldest_frame:
                                oldest_frame = obj["last_seen"]
                                oldest_idx = i
                        
                        # If we found an inactive object to recycle its ID
                        if not math.isinf(oldest_frame):
                            track_id = tracked_objects[class_name][oldest_idx]["track_id"]
                            tracked_objects[class_name].pop(oldest_idx)
                        else:
                            # Last resort: use ID 0 (should rarely happen)
                            track_id = 0
                    
                    # Create new tracked object
                    tracked_objects[class_name].append({
                        "track_id": track_id,
                        "rect": ann.rect,
                        "last_seen": frame_num,
                        "active": True
                    })
                    
                    # Assign track_id to annotation
                    ann.attributes["track_id"] = track_id
                    updated_count += 1
        
        # Update current frame's annotations on the canvas to match assigned IDs
        if self.current_frame in self.frame_annotations:
            for ann in self.canvas.annotations:
                if hasattr(ann, "rect"):
                    if not hasattr(ann, "attributes"):
                        ann.attributes = {}
                    
                    # Try to find matching annotation in current frame
                    for frame_ann in self.frame_annotations[self.current_frame]:
                        if hasattr(frame_ann, "rect") and self.iou(ann.rect, frame_ann.rect) > 0.9:
                            if "track_id" in frame_ann.attributes:
                                ann.attributes["track_id"] = frame_ann.attributes["track_id"]
                                break

        # Add IoU calculation method if needed
        if not hasattr(self, 'iou'):
            self.iou = lambda rect1, rect2: self.calculate_iou(rect1, rect2)

        self.update_annotation_list()
        self.canvas.update()
        QMessageBox.information(self, "Track ID", f"Added 'track_id' to {updated_count} bounding boxes using intelligent tracking.")

    def calculate_iou(self, rect1, rect2):
        """
        Calculate Intersection over Union between two QRect objects.
        """
        # Calculate intersection area
        x_left = max(rect1.left(), rect2.left())
        y_top = max(rect1.top(), rect2.top())
        x_right = min(rect1.right(), rect2.right())
        y_bottom = min(rect1.bottom(), rect2.bottom())
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0  # No intersection
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        rect1_area = rect1.width() * rect1.height()
        rect2_area = rect2.width() * rect2.height()
        union_area = rect1_area + rect2_area - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area

    def assign_track_id_to_new_bbox(self, new_bbox):
        """
        Assigns a track_id to new_bbox based on IoU with previous frame's bboxes.
        """
        if not self.tracking_mode_enabled:
            return

        prev_frame = self.current_frame - 1
        if prev_frame < 0 or prev_frame not in self.frame_annotations:
            # No previous frame, assign new id
            new_id = self.get_next_track_id()
            new_bbox.attributes['track_id'] = new_id
            return

        # Get class name of the new bbox
        class_name = getattr(new_bbox, "class_name", "unknown")
        
        # Only consider previous bboxes of the same class
        prev_bboxes = [ann for ann in self.frame_annotations[prev_frame] 
                    if hasattr(ann, 'rect') and getattr(ann, "class_name", "unknown") == class_name]
        
        best_iou = 0
        best_ann = None
        for ann in prev_bboxes:
            iou_val = self.iou(new_bbox.rect, ann.rect)
            if iou_val > best_iou:
                best_iou = iou_val
                best_ann = ann

        if best_iou > 0.5 and best_ann and 'track_id' in best_ann.attributes:
            new_bbox.attributes['track_id'] = best_ann.attributes['track_id']
        else:
            new_bbox.attributes['track_id'] = self.get_next_track_id()

    def get_next_track_id(self):
        # Find the max track_id used so far
        max_id = -1  # Start from -1 so first ID will be 0
        for anns in self.frame_annotations.values():
            for ann in anns:
                tid = ann.attributes.get('track_id') if hasattr(ann, 'attributes') else None
                if tid is not None and isinstance(tid, int):
                    max_id = max(max_id, tid)
        return max_id + 1
    
    def iou(self, rect1, rect2):
        """
        Calculate Intersection over Union between two QRect objects.
        
        Args:
            rect1: First QRect
            rect2: Second QRect
            
        Returns:
            float: IoU value between 0 and 1
        """
        # Calculate intersection area
        x_left = max(rect1.left(), rect2.left())
        y_top = max(rect1.top(), rect2.top())
        x_right = min(rect1.right(), rect2.right())
        y_bottom = min(rect1.bottom(), rect2.bottom())
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0  # No intersection
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        rect1_area = rect1.width() * rect1.height()
        rect2_area = rect2.width() * rect2.height()
        union_area = rect1_area + rect2_area - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area

        # rect: (x, y, w, h)
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

