"""
UI Creator module for VIAT application.

This module contains functions for creating various UI elements of the
Video Image Annotation Tool, including menus, toolbars, and controls.
"""

from PyQt5.QtWidgets import (
    QAction,
    QActionGroup,
    QLabel,
    QComboBox,
    QToolBar,
    QWidgetAction,
    QWidget,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QStatusBar,
    QSpinBox,
    QCheckBox,
    QMainWindow,
    QTabWidget,
    QMenu,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viat.widgets import AnnotationDock, ClassDock, PrimaryAnnotationToolbar
from viat.widgets.sam_interactive_dock import SAMInteractiveDock
from viat.widgets.track_queue_dock import TrackQueueDock
from viat.widgets.empty_frames_dock import EmptyFramesManagerDock
from viat.widgets.uncertain_frames_dock import UncertainFramesManagerDock
from viat.widgets.class_frames_dock import ClassFramesManagerDock
from viat.widgets.video_manager_dock import VideoManagerDock
from viat.widgets.crop_settings_dock import CropSettingsDock
from viat.widgets.clip_cuts_dock import ClipCutsDock
from viat.widgets.evaluation_inspector_dock import EvaluationInspectorDock
from viat.utils import gpu_utils



class UICreator:
    """Class responsible for creating UI elements for the VIAT application."""

    def __init__(self, main_window):
        """
        Initialize the UI creator with a reference to the main window.

        Args:
            main_window: The main application window
        """
        self.main_window = main_window

    def create_menu_bar(self):
        """Create the application menu bar and its actions."""
        menubar = self.main_window.menuBar()

        # File menu
        self.create_file_menu(menubar)

        # Edit menu
        self.create_edit_menu(menubar)

        # View menu
        self.create_view_menu(menubar)

        # Tools menu
        self.create_tools_menu(menubar)

        # Help menu
        self.create_help_menu(menubar)


    def create_file_menu(self, menubar):
        """Create the File menu and its actions."""
        file_menu = menubar.addMenu("File")

        # Open Video action
        open_action = QAction("Open Video", self.main_window)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.main_window.open_video)
        file_menu.addAction(open_action)


        # Open Image Folder action
        open_image_folder_action = QAction("Open Image Folder", self.main_window)
        open_image_folder_action.setShortcut("Ctrl+Shift+O")
        open_image_folder_action.triggered.connect(self.main_window.open_image_folder)
        file_menu.addAction(open_image_folder_action)

        # Open Video Folder action
        open_video_folder_action = QAction("Open Video Folder", self.main_window)
        open_video_folder_action.setShortcut("Ctrl+Alt+O")
        open_video_folder_action.triggered.connect(
            lambda: getattr(self.main_window, "open_video_folder", lambda: None)()
        )
        file_menu.addAction(open_video_folder_action)

        # Save Project action
        save_action = QAction("Save Project", self.main_window)
        save_action.setShortcut("Ctrl+S")
        save_action.triggered.connect(self.main_window.save_project)
        file_menu.addAction(save_action)

        # Save Project As action
        save_as_action = QAction("Save Project As...", self.main_window)
        save_as_action.setShortcut("Ctrl+Shift+S")
        save_as_action.triggered.connect(lambda: self.main_window.save_project(True))
        file_menu.addAction(save_as_action)

        # Load Project action
        load_action = QAction("Load Project", self.main_window)
        load_action.setShortcut("Ctrl+L")
        load_action.triggered.connect(self.main_window.load_project)
        file_menu.addAction(load_action)

        # Recent Projects submenu
        self.main_window.recent_projects_menu = file_menu.addMenu("Recent Projects")
        self.main_window.update_recent_projects_menu()

        # Import Annotations action
        import_action = QAction("Import Annotations", self.main_window)
        import_action.setShortcut("Ctrl+I")
        import_action.triggered.connect(
            lambda: self.main_window.import_annotations(None)
        )
        file_menu.addAction(import_action)

        # Export Annotations action
        export_action = QAction("Export Annotations", self.main_window)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.main_window.export_annotations)
        file_menu.addAction(export_action)

        # Export Clip Cuts action
        export_clip_cuts_action = QAction("Export Clip Cuts", self.main_window)
        export_clip_cuts_action.triggered.connect(
            lambda: getattr(self.main_window, 'export_clip_cuts', lambda: None)()
        )
        file_menu.addAction(export_clip_cuts_action)

        # Export Blurred Media action
        export_blurred_action = QAction("Export Blurred Media (Video/Images)...", self.main_window)
        export_blurred_action.triggered.connect(
            lambda: getattr(self.main_window, 'export_blurred_video', lambda: None)()
        )
        file_menu.addAction(export_blurred_action)

        # Update Original Dataset Labels action
        update_dataset_labels_action = QAction("Update Original Dataset Labels", self.main_window)
        update_dataset_labels_action.triggered.connect(self.main_window.update_original_dataset_labels)
        self.main_window.update_dataset_labels_action = update_dataset_labels_action
        self.main_window.update_dataset_labels_action.setEnabled(False)
        file_menu.addAction(update_dataset_labels_action)

        file_menu.addSeparator()

        # GPU Device submenu (which GPU AI backends should use)
        self.create_gpu_menu(file_menu)

        file_menu.addSeparator()

        # Delete History action
        delete_history_action = QAction("Delete History", self.main_window)
        delete_history_action.triggered.connect(self.main_window.delete_history)
        file_menu.addAction(delete_history_action)

        # Exit action
        exit_action = QAction("Exit", self.main_window)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.main_window.close)
        file_menu.addAction(exit_action)

    def create_edit_menu(self, menubar):
        """Create the Edit menu and its actions."""
        edit_menu = menubar.addMenu("Edit")

        # Undo action
        undo_action = QAction("&Undo", self.main_window)
        undo_action.setShortcut("Ctrl+Z")
        undo_action.setStatusTip("Undo the last action")
        undo_action.triggered.connect(self.main_window.undo)
        edit_menu.addAction(undo_action)

        # Redo action
        redo_action = QAction("&Redo", self.main_window)
        redo_action.setShortcuts(["Ctrl+Y", "Ctrl+Shift+Z"])
        redo_action.setStatusTip("Redo the last undone action")
        redo_action.triggered.connect(self.main_window.redo)
        edit_menu.addAction(redo_action)

        edit_menu.addSeparator()

        # Add Annotation action
        add_action = QAction("Add Annotation", self.main_window)
        add_action.triggered.connect(self.main_window.add_annotation)
        edit_menu.addAction(add_action)

        # Delete Selected action
        delete_selected_action = QAction("Delete Selected", self.main_window)
        delete_selected_action.triggered.connect(self.main_window.delete_selected_annotations)
        edit_menu.addAction(delete_selected_action)

        # Clear Annotations action
        clear_action = QAction("Clear All Annotations", self.main_window)
        clear_action.triggered.connect(self.main_window.clear_annotations)
        edit_menu.addAction(clear_action)

        # Select All action
        select_all_action = QAction("Select &All", self.main_window)
        select_all_action.setShortcut("Ctrl+A")
        select_all_action.triggered.connect(self.main_window.select_all_annotations)
        edit_menu.addAction(select_all_action)

        # Go to Frame action
        goto_frame_action = QAction("Go to Frame...", self.main_window)
        goto_frame_action.setShortcut("Ctrl+G")
        goto_frame_action.setStatusTip("Jump directly to a specific frame number")
        goto_frame_action.triggered.connect(self.main_window.go_to_frame_dialog)
        edit_menu.addAction(goto_frame_action)

        edit_menu.addSeparator()

        # Batch Edit Annotations action
        batch_edit_action = QAction("Batch Edit Annotations", self.main_window)
        batch_edit_action.setShortcut("Ctrl+B")
        batch_edit_action.triggered.connect(
            lambda: self.main_window.annotation_dock.batch_edit_annotations()
        )
        edit_menu.addAction(batch_edit_action)

        # Add Class action
        add_class_action = QAction("Add Class", self.main_window)
        add_class_action.triggered.connect(self.main_window.add_class)
        edit_menu.addAction(add_class_action)

        edit_menu.addSeparator()

        # Preferences / Settings
        preferences_menu = edit_menu.addMenu("Preferences")
        self.create_settings_menu(preferences_menu)

    def create_view_menu(self, menubar):
        """Create the View menu and its actions."""
        view_menu = menubar.addMenu("View")

        # Toggle Side Panel (collapses / restores the entire right panel)
        self.toggle_docks_action = QAction("Toggle Side Panel", self.main_window, checkable=True)
        self.toggle_docks_action.setChecked(True)
        def toggle_side_panel(checked):
            right_docks = [
                getattr(self.main_window, 'annotation_dock', None),
                getattr(self.main_window, 'class_dock', None),
                getattr(self.main_window, 'sam_interactive_dock', None),
                getattr(self.main_window, 'track_queue_dock', None),
                getattr(self.main_window, 'empty_frames_dock', None),
                getattr(self.main_window, 'class_frames_dock', None),
                getattr(self.main_window, 'uncertain_frames_dock', None),
                getattr(self.main_window, 'crop_settings_dock', None),
                getattr(self.main_window, 'clip_cuts_dock', None),
            ]
            if checked:
                if hasattr(self.main_window, 'annotation_dock'):
                    self.main_window.annotation_dock.setVisible(True)
                    self.main_window.annotation_dock.raise_()
                if hasattr(self.main_window, 'class_dock'):
                    self.main_window.class_dock.setVisible(True)
            else:
                for dock in right_docks:
                    if dock:
                        dock.setVisible(False)
        self.toggle_docks_action.triggered.connect(toggle_side_panel)
        view_menu.addAction(self.toggle_docks_action)

        view_menu.addSeparator()

        def _make_right_dock_toggler(dock_attr, on_show=None):
            def handler(checked):
                dock = getattr(self.main_window, dock_attr, None)
                if not dock:
                    return
                if checked:
                    # Tabify into right dock area so it opens as a tab instead of stacking below
                    if hasattr(self.main_window, 'annotation_dock') and self.main_window.annotation_dock is not dock:
                        self.main_window.tabifyDockWidget(self.main_window.annotation_dock, dock)
                    dock.setVisible(True)
                    dock.raise_()
                    if on_show:
                        on_show()
                else:
                    dock.setVisible(False)
            return handler

        # Annotations Dock
        self.toggle_annotations_action = QAction("Annotations Panel", self.main_window, checkable=True)
        self.toggle_annotations_action.setChecked(True)
        self.toggle_annotations_action.triggered.connect(
            _make_right_dock_toggler('annotation_dock')
        )
        view_menu.addAction(self.toggle_annotations_action)

        # Classes Dock
        self.toggle_classes_action = QAction("Classes Panel", self.main_window, checkable=True)
        self.toggle_classes_action.setChecked(True)
        self.toggle_classes_action.triggered.connect(
            _make_right_dock_toggler('class_dock')
        )
        view_menu.addAction(self.toggle_classes_action)

        # SAM Interactive mode is a workflow state rather than a passive dock
        # visibility toggle.  Reuse the action created for the compact
        # toolbar's AI menu so both surfaces stay synchronized.
        self.toggle_sam_action = getattr(
            self.main_window, "action_sam_interactive", None
        )
        if self.toggle_sam_action is None:
            self.toggle_sam_action = QAction(
                "SAM Interactive", self.main_window, checkable=True
            )
            self.toggle_sam_action.setChecked(False)
            self.toggle_sam_action.setToolTip(
                "Open the interactive SAM prompting workflow"
            )
            self.toggle_sam_action.triggered.connect(
                self.main_window.toggle_sam_interactive_mode
            )
            self.main_window.action_sam_interactive = self.toggle_sam_action
        view_menu.addAction(self.toggle_sam_action)

        # Track Queue Dock (batch SAM tracking jobs)
        self.toggle_track_queue_action = getattr(
            self.main_window, "track_queue_toolbar_action", None
        )
        if self.toggle_track_queue_action is None:
            self.toggle_track_queue_action = QAction(
                "Track Queue Panel", self.main_window, checkable=True
            )
            self.toggle_track_queue_action.setChecked(False)
            self.toggle_track_queue_action.triggered.connect(
                _make_right_dock_toggler('track_queue_dock')
            )
        view_menu.addAction(self.toggle_track_queue_action)

        # Empty Frames Manager
        self.toggle_empty_frames_action = QAction("Empty Frames Manager", self.main_window, checkable=True)
        self.toggle_empty_frames_action.setChecked(False)
        self.toggle_empty_frames_action.triggered.connect(
            _make_right_dock_toggler('empty_frames_dock', lambda: getattr(self.main_window, 'refresh_empty_frames_dock', lambda: None)())
        )
        view_menu.addAction(self.toggle_empty_frames_action)
        
        # Class Frames Manager
        self.toggle_class_frames_action = QAction("Class Frames Manager", self.main_window, checkable=True)
        self.toggle_class_frames_action.setChecked(False)
        self.toggle_class_frames_action.triggered.connect(
            _make_right_dock_toggler('class_frames_dock', lambda: getattr(self.main_window, 'refresh_class_frames_dock', lambda: None)())
        )
        view_menu.addAction(self.toggle_class_frames_action)

        # Uncertain Frames Manager
        self.toggle_uncertain_frames_action = QAction("Uncertain Frames Manager", self.main_window, checkable=True)
        self.toggle_uncertain_frames_action.setChecked(False)
        self.toggle_uncertain_frames_action.triggered.connect(
            _make_right_dock_toggler('uncertain_frames_dock', lambda: getattr(self.main_window, 'refresh_uncertain_frames_dock', lambda: None)())
        )
        view_menu.addAction(self.toggle_uncertain_frames_action)

        # Clip Cuts Manager
        self.toggle_clip_cuts_action = getattr(
            self.main_window, "clip_cuts_action", None
        )
        if self.toggle_clip_cuts_action is None:
            self.toggle_clip_cuts_action = QAction(
                "Clip Cuts Manager", self.main_window, checkable=True
            )
            self.toggle_clip_cuts_action.setChecked(False)
            self.toggle_clip_cuts_action.triggered.connect(
                _make_right_dock_toggler('clip_cuts_dock')
            )
        view_menu.addAction(self.toggle_clip_cuts_action)

        # Crop Settings Dock
        self.toggle_crop_settings_action = QAction("Crop Settings Panel", self.main_window, checkable=True)
        self.toggle_crop_settings_action.setChecked(False)
        self.toggle_crop_settings_action.triggered.connect(
            _make_right_dock_toggler('crop_settings_dock')
        )
        view_menu.addAction(self.toggle_crop_settings_action)

        # Evaluation Inspector Dock
        self.toggle_eval_inspector_action = QAction("Evaluation Inspector Panel", self.main_window, checkable=True)
        self.toggle_eval_inspector_action.setChecked(False)
        self.toggle_eval_inspector_action.triggered.connect(
            _make_right_dock_toggler('evaluation_inspector_dock')
        )
        view_menu.addAction(self.toggle_eval_inspector_action)

        view_menu.addSeparator()

        # Toggle Video Manager (Left Area)
        self.toggle_video_manager_action = getattr(
            self.main_window, "video_manager_toolbar_action", None
        )
        if self.toggle_video_manager_action is None:
            self.toggle_video_manager_action = QAction(
                "Video Manager", self.main_window, checkable=True
            )
            self.toggle_video_manager_action.setChecked(False)
            self.toggle_video_manager_action.triggered.connect(
                lambda checked: self._set_dock_visible(
                    "video_manager_dock", checked
                )
            )
        view_menu.addAction(self.toggle_video_manager_action)

        # Toggle Visual Inspector (Bottom Area)
        self.toggle_visual_inspector_action = QAction("Visual Inspector (SHOW)", self.main_window, checkable=True)
        self.toggle_visual_inspector_action.setChecked(False)
        def toggle_visual_inspector(checked):
            if hasattr(self.main_window, 'visual_inspector_dock'):
                self.main_window.visual_inspector_dock.setVisible(checked)
                if checked:
                    self.main_window.visual_inspector_dock.raise_()
        self.toggle_visual_inspector_action.triggered.connect(toggle_visual_inspector)
        view_menu.addAction(self.toggle_visual_inspector_action)

        view_menu.addSeparator()

        # Add Style Menu as a submenu
        self.create_style_menu(view_menu)

    def create_tools_menu_legacy(self, menubar):
        """Create the Tools menu and its actions."""
        tools_menu = menubar.addMenu("Tools")

        # Auto Label action
        auto_label_action = QAction("Auto Label", self.main_window)
        auto_label_action.triggered.connect(self.main_window.auto_label)
        tools_menu.addAction(auto_label_action)

        # Autopilot: unattended video labeling + review panel
        autopilot_action = QAction("Autopilot (Unattended Video Labeling)...", self.main_window)
        autopilot_action.setToolTip("Detect every frame, track objects in/out of the scene, repair tracks and "
                                    "flag only the likely mistakes for review")
        autopilot_action.triggered.connect(self.main_window.open_autopilot)
        tools_menu.addAction(autopilot_action)
        autopilot_review_action = QAction("Autopilot Review Panel", self.main_window)
        autopilot_review_action.triggered.connect(self.main_window.show_autopilot_review)
        tools_menu.addAction(autopilot_review_action)

        # Model Evaluation action
        model_eval_action = QAction("Model Evaluation...", self.main_window)
        model_eval_action.setToolTip("Evaluate detector and tracker metrics (COCO mAP, MOT, Center BBox)")
        model_eval_action.triggered.connect(
            lambda: getattr(self.main_window, 'open_evaluation_dialog', lambda: None)()
        )
        tools_menu.addAction(model_eval_action)



        # Smart Edge Movement action
        smart_edge_action = QAction(
            "Smart Edge Movement", self.main_window, checkable=True
        )
        smart_edge_action.triggered.connect(self.main_window.toggle_smart_edge)
        tools_menu.addAction(smart_edge_action)

        # Add separator
        tools_menu.addSeparator()

        # Create Dataset action (for image datasets)
        create_dataset_action = QAction("Create Dataset...", self.main_window)
        create_dataset_action.setToolTip(
            "Create a new dataset from the current annotations"
        )
        create_dataset_action.triggered.connect(self.main_window.create_dataset)
        tools_menu.addAction(create_dataset_action)

        # Merge Video Dataset action
        merge_video_dataset_action = QAction("Merge Video Dataset to Single Video...", self.main_window)
        merge_video_dataset_action.setToolTip(
            "Merge all annotated videos in a folder into a single video and label file (many2single)"
        )
        merge_video_dataset_action.triggered.connect(
            lambda: getattr(self.main_window, 'open_merge_video_dataset_dialog', lambda: None)()
        )
        tools_menu.addAction(merge_video_dataset_action)

        # Convert Video Dataset to YOLO Dataset action
        convert_video_to_yolo_action = QAction("Convert Video Dataset to YOLO Dataset...", self.main_window)
        convert_video_to_yolo_action.setToolTip(
            "Convert a folder of videos and per-frame annotations into a YOLO image dataset with smart cropping and class balancing"
        )
        convert_video_to_yolo_action.triggered.connect(
            lambda: getattr(self.main_window, 'open_video_to_yolo_dialog', lambda: None)()
        )
        tools_menu.addAction(convert_video_to_yolo_action)
        
        # Zero-Shot Classification Refiner
        zero_shot_refiner_action = QAction("Zero-Shot Classification Refiner...", self.main_window)
        zero_shot_refiner_action.setToolTip("Refine bounding boxes with zero-shot classification")
        if hasattr(self.main_window, "open_zero_shot_refiner_dialog"):
            zero_shot_refiner_action.triggered.connect(self.main_window.open_zero_shot_refiner_dialog)
        tools_menu.addAction(zero_shot_refiner_action)

        # Remove duplicate (overlapping) annotations via NMS
        annotation_nms_action = QAction("Remove Duplicate / Overlapping Annotations (NMS)...", self.main_window)
        annotation_nms_action.setToolTip(
            "Remove boxes that duplicate an existing box of the same object, or that lie inside "
            "ignore-region classes such as 'shadow'")
        annotation_nms_action.triggered.connect(self.main_window.open_annotation_nms_dialog)
        tools_menu.addAction(annotation_nms_action)

        # Add separator
        tools_menu.addSeparator()

        # Detect Duplicate Frames action
        duplicate_frames_action = QAction(
            "Detect Duplicate Frames", self.main_window, checkable=True
        )
        duplicate_frames_action.setToolTip(
            "Automatically detect and propagate annotations to duplicate frames"
        )
        duplicate_frames_action.triggered.connect(
            self.main_window.toggle_duplicate_frames_detection
        )
        duplicate_frames_action.setChecked(
            self.main_window.duplicate_frames_enabled
        )  # Set initial state
        tools_menu.addAction(duplicate_frames_action)
        self.main_window.duplicate_frames_action = duplicate_frames_action

        # Scan for Duplicates action
        scan_action = QAction("Scan Video for Duplicates", self.main_window)
        scan_action.setToolTip("Scan the entire video to identify duplicate frames")
        scan_action.triggered.connect(self.main_window.scan_video_for_duplicates)
        tools_menu.addAction(scan_action)

        # Propagate Annotations action
        propagate_action = QAction("Propagate Annotations...", self.main_window)
        propagate_action.setToolTip("Copy current frame annotations to multiple frames")
        propagate_action.setShortcut("Ctrl+P")
        propagate_action.triggered.connect(self.main_window.propagate_annotations)
        tools_menu.addAction(propagate_action)

        # Propagate to Similar Frames action
        similar_action = QAction("Propagate to Similar Frames...", self.main_window)
        similar_action.setToolTip("Copy annotations to frames that look similar")
        similar_action.triggered.connect(self.main_window.propagate_to_similar_frames)
        tools_menu.addAction(similar_action)

        # Blur & Privacy Tools submenu
        tools_menu.addSeparator()
        privacy_menu = tools_menu.addMenu("Privacy & Blur Tools")

        auto_blur_menu_action = QAction("Auto-Blur New Labels", self.main_window, checkable=True)
        auto_blur_menu_action.setIcon(self.main_window.icon_provider.get_icon("auto-blur"))
        auto_blur_menu_action.setToolTip("Automatically blur all newly added labels (manual, pasted, tracked, detected)")
        auto_blur_menu_action.setChecked(getattr(self.main_window, 'auto_blur_labels', False))
        auto_blur_menu_action.triggered.connect(
            lambda checked: getattr(self.main_window, 'toggle_auto_blur_labels', lambda c: None)(checked)
        )
        privacy_menu.addAction(auto_blur_menu_action)
        self.main_window.auto_blur_menu_action = auto_blur_menu_action

        auto_remove_blur_action = QAction("Auto-Remove BBoxes Under Blur (>70%)", self.main_window, checkable=True)
        auto_remove_blur_action.setToolTip("Automatically remove bounding boxes that are more than 70% covered by a blur region")
        auto_remove_blur_action.setChecked(getattr(self.main_window, 'auto_remove_under_blur', True))
        auto_remove_blur_action.triggered.connect(
            lambda checked: setattr(self.main_window, 'auto_remove_under_blur', checked)
        )
        privacy_menu.addAction(auto_remove_blur_action)

        auto_save_blur_action = QAction("Auto-Save/Burn Blur to Image File on Frame Switch", self.main_window, checkable=True)
        auto_save_blur_action.setToolTip("Automatically burn blur regions into image files when switching frames, saving unblurred versions to unblurred_backups/")
        auto_save_blur_action.setChecked(getattr(self.main_window, 'auto_save_blur_on_switch', False))
        auto_save_blur_action.triggered.connect(
            lambda checked: getattr(self.main_window, 'toggle_auto_save_blur_on_switch', lambda c: None)(checked)
        )
        privacy_menu.addAction(auto_save_blur_action)
        self.main_window.auto_save_blur_action = auto_save_blur_action

        privacy_menu.addSeparator()

        clear_frame_blur_action = QAction("Clear Current Frame Blur", self.main_window)
        clear_frame_blur_action.setToolTip("Remove all blur regions from the current frame")
        clear_frame_blur_action.triggered.connect(
            lambda: getattr(self.main_window, 'clear_current_frame_blur', lambda: None)()
        )
        privacy_menu.addAction(clear_frame_blur_action)

        clear_range_blur_action = QAction("Clear Blur in Range...", self.main_window)
        clear_range_blur_action.setToolTip("Remove blur regions from a range of frames")
        clear_range_blur_action.triggered.connect(
            lambda: getattr(self.main_window, 'clear_blur_range', lambda: None)()
        )
        privacy_menu.addAction(clear_range_blur_action)

        clear_all_blur_action = QAction("Clear All Blurs Across Video", self.main_window)
        clear_all_blur_action.setToolTip("Remove all blur regions from every frame")
        clear_all_blur_action.triggered.connect(
            lambda: getattr(self.main_window, 'clear_all_blurs', lambda: None)()
        )
        privacy_menu.addAction(clear_all_blur_action)

        # Store reference to keep menu and toolbar in sync
        self.main_window.smart_edge_action = smart_edge_action

    def create_tools_menu(self, menubar):
        """Create a grouped Tools menu using the toolbar's shared actions."""
        main_window = self.main_window
        tools_menu = menubar.addMenu("Tools")

        ai_menu = tools_menu.addMenu("AI & Tracking")
        for attribute_name in (
            "auto_label_action",
            "autopilot_action",
            "autopilot_review_action",
            "zero_shot_refiner_action",
            "annotation_nms_action",
            "tracking_mode_action",
            "sam_interactive_action",
            "auto_bbox_action",
            "duplicate_frames_action",
            "scan_duplicate_frames_action",
            "propagate_action",
            "similar_frames_action",
        ):
            action = getattr(main_window, attribute_name, None)
            if action is not None:
                ai_menu.addAction(action)

        smart_edge_action = self._create_action(
            ai_menu,
            "smart_edge_action",
            "Smart Edge Movement",
            main_window.toggle_smart_edge,
            checkable=True,
            tooltip="Snap annotation edges to image features",
        )

        review_menu = tools_menu.addMenu("Review")
        for attribute_name in (
            "verify_mode_action",
            "verify_selected_action",
            "verify_all_action",
            "object_visibility_action",
            "evaluation_action",
        ):
            action = getattr(main_window, attribute_name, None)
            if action is not None:
                review_menu.addAction(action)

        tools_menu.addSeparator()
        dataset_tools_menu = tools_menu.addMenu("Dataset Preparation")

        create_dataset_action = QAction("Create Dataset...", main_window)
        create_dataset_action.setToolTip(
            "Create a new dataset from the current annotations"
        )
        create_dataset_action.triggered.connect(main_window.create_dataset)
        dataset_tools_menu.addAction(create_dataset_action)

        merge_video_dataset_action = QAction(
            "Merge Video Dataset to Single Video...", main_window
        )
        merge_video_dataset_action.setToolTip(
            "Merge annotated videos in a folder into one video and label file"
        )
        merge_video_dataset_action.triggered.connect(
            lambda: getattr(
                main_window, "open_merge_video_dataset_dialog", lambda: None
            )()
        )
        dataset_tools_menu.addAction(merge_video_dataset_action)

        convert_video_to_yolo_action = QAction(
            "Convert Video Dataset to YOLO Dataset...", main_window
        )
        convert_video_to_yolo_action.setToolTip(
            "Convert annotated videos into a YOLO image dataset"
        )
        convert_video_to_yolo_action.triggered.connect(
            lambda: getattr(
                main_window, "open_video_to_yolo_dialog", lambda: None
            )()
        )
        dataset_tools_menu.addAction(convert_video_to_yolo_action)

        tools_menu.addSeparator()
        privacy_menu = tools_menu.addMenu("Privacy & Blur Tools")
        for attribute_name in (
            "auto_blur_action",
            "auto_blur_dialog_action",
            "auto_save_blur_action",
            "clear_frame_blur_action",
            "clear_range_blur_action",
            "clear_all_blur_action",
            "crop_mode_action",
            "clip_cuts_action",
        ):
            action = getattr(main_window, attribute_name, None)
            if action is not None:
                privacy_menu.addAction(action)

        auto_remove_blur_action = QAction(
            "Auto-Remove BBoxes Under Blur (>70%)", main_window, checkable=True
        )
        auto_remove_blur_action.setToolTip(
            "Remove boxes that are more than 70% covered by a blur region"
        )
        auto_remove_blur_action.setChecked(
            getattr(main_window, "auto_remove_under_blur", True)
        )
        auto_remove_blur_action.triggered.connect(
            lambda checked: setattr(main_window, "auto_remove_under_blur", checked)
        )
        privacy_menu.addAction(auto_remove_blur_action)

        main_window.smart_edge_action = smart_edge_action

    def create_style_menu(self, menubar):
        """Create the Style menu and its actions."""
        self.main_window.style_menu = menubar.addMenu("Style")

        # Create action group for styles to make them exclusive
        style_group = QActionGroup(self.main_window)
        style_group.setExclusive(True)

        # Add style options
        for style_name in self.main_window.styles.keys():
            style_action = QAction(style_name, self.main_window, checkable=True)
            if style_name == self.main_window.current_style:
                style_action.setChecked(True)
            style_action.triggered.connect(
                lambda checked, s=style_name: self.main_window.change_style(s)
            )
            style_group.addAction(style_action)
            self.main_window.style_menu.addAction(style_action)

    def create_help_menu(self, menubar):
        """Create the Help menu and its actions."""
        help_menu = menubar.addMenu("Help")

        # About action
        about_action = QAction("About", self.main_window)
        about_action.triggered.connect(self.main_window.show_about)
        help_menu.addAction(about_action)

    def create_toolbar(self):
        """Create the complete, uncollapsed annotation toolbar.

        The second toolbar is intentionally kept as the full working toolbar:
        annotation, tracking, AI, review, privacy, dataset, and temporary
        integration controls are all placed directly on it.  The application
        menus reuse the same actions, so moving an action into a menu does not
        remove it from this toolbar or create a second stateful copy.
        """
        main_window = self.main_window
        toolbar = PrimaryAnnotationToolbar(
            main_window=main_window,
            icon_provider=main_window.icon_provider,
            parent=main_window,
        )
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        main_window.toolbar = toolbar
        main_window.addToolBar(Qt.TopToolBarArea, toolbar)

        # Compatibility aliases used by the canvas, annotation manager, and
        # existing refresh code.
        main_window.class_selector = toolbar.class_selector
        main_window.method_selector = toolbar.method_selector
        main_window.pan_tool_action = toolbar.pan_button

        toolbar.class_selected.connect(self._set_current_class)
        toolbar.add_class_requested.connect(main_window.add_class)
        toolbar.method_selected.connect(main_window.change_annotation_method)
        toolbar.add_annotation_requested.connect(main_window.add_annotation)
        toolbar.edit_annotation_requested.connect(self._edit_selected_annotation)
        toolbar.delete_annotation_requested.connect(
            main_window.delete_selected_annotation
        )
        toolbar.pan_toggled.connect(lambda _checked: main_window.toggle_pan_mode())
        toolbar.zoom_in_requested.connect(main_window.zoom_in)
        toolbar.zoom_out_requested.connect(main_window.zoom_out)
        toolbar.zoom_reset_requested.connect(main_window.reset_zoom)

        self._create_full_toolbar_controls(toolbar)

    def _set_current_class(self, class_name):
        """Apply a class selection from the primary toolbar to the canvas."""
        canvas = getattr(self.main_window, "canvas", None)
        if canvas is not None and hasattr(canvas, "set_current_class"):
            canvas.set_current_class(class_name)

    def _edit_selected_annotation(self):
        """Edit the current canvas selection when one exists."""
        canvas = getattr(self.main_window, "canvas", None)
        selected = getattr(canvas, "selected_annotation", None)
        if selected is not None and hasattr(self.main_window, "edit_annotation"):
            self.main_window.edit_annotation(selected)

    def _create_action(
        self,
        menu,
        attribute_name,
        text,
        callback,
        *,
        checkable=False,
        checked=False,
        icon_name=None,
        tooltip=None,
    ):
        """Create and register one reusable action for a grouped menu."""
        action = QAction(text, self.main_window)
        action.setObjectName(attribute_name)
        action.setCheckable(checkable)
        action.setChecked(checked)
        if tooltip:
            action.setToolTip(tooltip)
            action.setStatusTip(tooltip)
        if icon_name:
            action.icon_name = icon_name
        if callback is not None:
            action.triggered.connect(callback)
        menu.addAction(action)
        setattr(self.main_window, attribute_name, action)
        if icon_name and hasattr(self.main_window.toolbar, "register_icon_action"):
            self.main_window.toolbar.register_icon_action(action, icon_name)
        return action

    def _add_widget_action(self, menu, widget, object_name, label=None):
        """Add a compact widget control to a grouped menu."""
        widget.setObjectName(object_name)
        if label:
            widget.setToolTip(label)
        widget_action = QWidgetAction(menu)
        widget_action.setDefaultWidget(widget)
        menu.addAction(widget_action)
        return widget_action

    def _create_full_toolbar_controls(self, toolbar):
        """Add every working control directly to the second toolbar.

        This deliberately does not create a ``More`` button or hide the
        controls behind submenus.  Some of these actions are also inserted in
        the File/View/Tools menus later; Qt allows one QAction to be displayed
        in several containers while keeping one checked state and one
        callback.
        """
        main_window = self.main_window

        toolbar.addSeparator()

        # Controls that were historically on the full annotation toolbar.
        main_window.tracking_toggle_btn = self._create_action(
            toolbar,
            "tracking_mode_action",
            "Tracking Mode",
            main_window.toggle_tracking_mode,
            checkable=True,
            checked=getattr(main_window, "tracking_mode_enabled", False),
            tooltip="Toggle automatic tracking ID assignment",
        )
        main_window.tracking_toggle_btn.setObjectName("TrackingModeAction")

        main_window.blur_pen_action = self._create_action(
            toolbar,
            "blur_pen_action",
            "Blur Pen",
            self._toggle_blur_pen,
            checkable=True,
            icon_name="blur-pen",
            tooltip="Enable Blur Pen (left-click to draw blur, right-click to erase)",
        )

        pen_size_label = QLabel("Pen Size:", toolbar)
        toolbar.addWidget(pen_size_label)
        main_window.blur_pen_size_spinbox = QSpinBox(toolbar)
        main_window.blur_pen_size_spinbox.setRange(1, 300)
        main_window.blur_pen_size_spinbox.setValue(30)
        main_window.blur_pen_size_spinbox.setSuffix(" px")
        main_window.blur_pen_size_spinbox.setToolTip(
            "Radius of the Blur Pen brush, in image pixels"
        )
        main_window.blur_pen_size_spinbox.valueChanged.connect(
            self._set_blur_pen_size
        )
        toolbar.addWidget(main_window.blur_pen_size_spinbox)

        main_window.auto_blur_action = self._create_action(
            toolbar,
            "auto_blur_action",
            "Auto Blur",
            main_window.toggle_auto_blur_labels,
            checkable=True,
            checked=getattr(main_window, "auto_blur_labels", False),
            icon_name="auto-blur",
            tooltip="Auto-blur all new labels (manual, pasted, tracked, detected)",
        )
        main_window.auto_blur_menu_action = main_window.auto_blur_action

        main_window.auto_bbox_action = self._create_action(
            toolbar,
            "auto_bbox_action",
            "Auto BBox",
            main_window.toggle_auto_bbox_mode,
            checkable=True,
            checked=getattr(main_window, "auto_bbox_mode", False),
            icon_name="magic",
            tooltip="Enable Auto BBox mode (left-click on an object to auto-segment)",
        )

        main_window.crop_mode_action = self._create_action(
            toolbar,
            "crop_mode_action",
            "Crop Mode",
            main_window.toggle_crop_mode,
            checkable=True,
            checked=False,
            icon_name="transform-crop",
            tooltip="Enable Crop Mode (define a crop region to extract a dataset)",
        )

        main_window.clip_cuts_action = self._create_action(
            toolbar,
            "clip_cuts_action",
            "Clip Cuts",
            self._toggle_clip_cuts,
            checkable=True,
            checked=False,
            tooltip="Toggle Clip Cuts Manager",
        )

        model_label = QLabel("SAM Model:", toolbar)
        toolbar.addWidget(model_label)
        main_window.sam_model_selector = QComboBox(toolbar)
        main_window.sam_model_selector.addItems([
            "SAM2 Fast (sam2.1_s.pt)",
            "SAM2 Huge (sam2.1_l.pt)",
            "SAM3 Fast (sam3.1_s.pt)",
            "SAM3 Huge (sam3.1_l.pt)",
            "SAM2 TRT C++ (TensorRT)",
            "SAM2 TRT C++ Tiny (TensorRT)",
        ])
        main_window.sam_model_selector.setToolTip(
            "Select the model to use for Auto BBox"
        )
        main_window.sam_model_selector.currentTextChanged.connect(
            main_window.change_sam_model
        )
        toolbar.addWidget(main_window.sam_model_selector)

        self.save_seg_checkbox = QCheckBox("Save Segmentations", toolbar)
        self.save_seg_checkbox.setChecked(True)
        self.save_seg_checkbox.setToolTip(
            "Save polygon segmentations when using Auto BBox (Magic Wand)"
        )
        main_window.save_seg_checkbox = self.save_seg_checkbox
        toolbar.addWidget(self.save_seg_checkbox)

        toolbar.addSeparator()

        # Temporary loading/integration controls stay on this same toolbar;
        # their actions are hidden until the corresponding workflow is active.
        cancel_button = QPushButton("Cancel Loading", toolbar)
        cancel_button.setStyleSheet(
            "background-color: #F56C6C; color: white; font-weight: bold; "
            "padding: 4px 8px; border-radius: 3px;"
        )
        cancel_button.clicked.connect(main_window.cancel_dataset_loading)
        cancel_action = toolbar.addWidget(cancel_button)
        cancel_action.setVisible(False)
        main_window.cancel_loading_btn = cancel_button
        main_window.cancel_loading_action = cancel_action

        main_window.verify_mode_action = self._create_action(
            toolbar,
            "verify_mode_action",
            "Auto Clean Mode",
            main_window.toggle_verification_mode,
            checkable=True,
            checked=False,
            tooltip=(
                "Delete unverified annotations when changing frames"
            ),
        )
        main_window.verify_selected_action = self._create_action(
            toolbar,
            "verify_selected_action",
            "Verify Selected",
            main_window.verify_selected_annotation,
            tooltip="Mark the selected annotation as verified",
        )
        main_window.verify_all_action = self._create_action(
            toolbar,
            "verify_all_action",
            "Verify All",
            main_window.verify_all_annotations,
            tooltip="Mark all annotations in the current frame as verified",
        )

        finish_separator = toolbar.addSeparator()
        finish_separator.setVisible(False)
        main_window.finish_integration_sep = finish_separator

        finish_button = QPushButton("Finish Integration & Merge", toolbar)
        finish_button.setStyleSheet(
            "background-color: #67C23A; color: white; font-weight: bold; "
            "padding: 4px 8px; border-radius: 3px;"
        )
        finish_button.setToolTip(
            "Finish reviewing and merge the dataset into the main dataset"
        )
        finish_button.clicked.connect(main_window.viat_finish_integration)
        finish_action = toolbar.addWidget(finish_button)
        finish_action.setVisible(False)
        main_window.finish_integration_action = finish_action

        move_button = QPushButton("Move to Review", toolbar)
        move_button.setStyleSheet(
            "background-color: #E6A23C; color: white; font-weight: bold; "
            "padding: 4px 8px; border-radius: 3px;"
        )
        move_button.setToolTip(
            "Move the dataset to a different folder and add details"
        )
        if hasattr(main_window, "viat_finish_integration_move"):
            move_button.clicked.connect(main_window.viat_finish_integration_move)
        move_action = toolbar.addWidget(move_button)
        move_action.setVisible(False)
        main_window.finish_integration_move_action = move_action

        # The remaining actions are kept directly visible as well.  They were
        # previously placed in the experimental More menu, and keeping them
        # here avoids silently changing the user's established workflow.
        toolbar.addSeparator()

        self._create_action(
            toolbar,
            "autopilot_action",
            "Autopilot...",
            main_window.open_autopilot,
            tooltip="Run unattended labeling and flag likely mistakes for review",
        )
        self._create_action(
            toolbar,
            "autopilot_review_action",
            "Autopilot Review",
            main_window.show_autopilot_review,
        )
        self._create_action(
            toolbar,
            "zero_shot_refiner_action",
            "Zero-Shot Refiner...",
            getattr(main_window, "open_zero_shot_refiner_dialog", None),
        )
        self._create_action(
            toolbar,
            "annotation_nms_action",
            "Remove Duplicates (NMS)...",
            main_window.open_annotation_nms_dialog,
        )
        main_window.action_sam_interactive = self._create_action(
            toolbar,
            "sam_interactive_action",
            "SAM Interactive",
            main_window.toggle_sam_interactive_mode,
            checkable=True,
            checked=False,
            tooltip="Open the interactive SAM prompting workflow",
        )
        main_window.action_sam_interactive.setObjectName("SAMInteractiveAction")

        main_window.duplicate_frames_action = self._create_action(
            toolbar,
            "duplicate_frames_action",
            "Detect Duplicate Frames",
            main_window.toggle_duplicate_frames_detection,
            checkable=True,
            checked=getattr(main_window, "duplicate_frames_enabled", True),
            tooltip="Automatically detect and propagate annotations to duplicate frames",
        )
        self._create_action(
            toolbar,
            "scan_duplicate_frames_action",
            "Scan Video for Duplicates",
            main_window.scan_video_for_duplicates,
        )
        propagate_action = self._create_action(
            toolbar,
            "propagate_action",
            "Propagate Annotations...",
            main_window.propagate_annotations,
            tooltip="Copy current frame annotations to multiple frames",
        )
        propagate_action.setShortcut("Ctrl+P")
        self._create_action(
            toolbar,
            "similar_frames_action",
            "Propagate to Similar Frames...",
            main_window.propagate_to_similar_frames,
            tooltip="Copy annotations to frames that look similar",
        )

        self._create_action(
            toolbar,
            "object_visibility_action",
            "Object Visibility...",
            main_window.viat_start_object_visibility_mode,
        )
        self._create_action(
            toolbar,
            "evaluation_action",
            "Model Evaluation...",
            getattr(main_window, "open_evaluation_dialog", None),
        )
        self._create_action(
            toolbar,
            "auto_blur_dialog_action",
            "Auto Blur Labels...",
            getattr(main_window, "open_auto_blur_dialog", None),
        )
        main_window.auto_save_blur_action = self._create_action(
            toolbar,
            "auto_save_blur_action",
            "Auto-Save/Burn Blur on Frame Switch",
            main_window.toggle_auto_save_blur_on_switch,
            checkable=True,
            checked=getattr(main_window, "auto_save_blur_on_switch", False),
        )
        for attribute_name, label, callback in (
            ("clear_frame_blur_action", "Clear Current Frame Blur", "clear_current_frame_blur"),
            ("clear_range_blur_action", "Clear Blur in Range...", "clear_blur_range"),
            ("clear_all_blur_action", "Clear All Blurs Across Video", "clear_all_blurs"),
        ):
            self._create_action(
                toolbar,
                attribute_name,
                label,
                getattr(main_window, callback, None),
            )

        main_window.video_manager_toolbar_action = self._create_action(
            toolbar,
            "video_manager_toolbar_action",
            "Video Manager",
            lambda checked=False: self._set_dock_visible(
                "video_manager_dock", checked
            ),
            checkable=True,
            checked=False,
        )
        main_window.track_queue_toolbar_action = self._create_action(
            toolbar,
            "track_queue_toolbar_action",
            "Track Queue",
            lambda checked=False: self._set_dock_visible(
                "track_queue_dock", checked
            ),
            checkable=True,
            checked=False,
        )


    def _toggle_blur_pen(self, checked):
        canvas = getattr(self.main_window, "canvas", None)
        if canvas is None:
            return
        canvas.blur_pen_active = bool(checked)
        canvas.setCursor(Qt.CrossCursor if checked else Qt.ArrowCursor)

    def _set_blur_pen_size(self, value):
        canvas = getattr(self.main_window, "canvas", None)
        if canvas is not None:
            canvas.blur_pen_size = value
            canvas.update()

    def _toggle_clip_cuts(self, checked):
        dock = getattr(self.main_window, "clip_cuts_dock", None)
        if dock is None:
            return
        if checked:
            annotation_dock = getattr(self.main_window, "annotation_dock", None)
            if annotation_dock is not None:
                self.main_window.tabifyDockWidget(annotation_dock, dock)
            dock.setVisible(True)
            dock.raise_()
        else:
            dock.setVisible(False)

    def _set_dock_visible(self, dock_attribute, visible):
        dock = getattr(self.main_window, dock_attribute, None)
        if dock is not None:
            dock.setVisible(bool(visible))
            if visible:
                dock.raise_()

    def _create_contextual_toolbar(self):
        """Create a hidden bar for loading and integration-only controls."""
        main_window = self.main_window
        toolbar = QToolBar("Contextual Tools", main_window)
        toolbar.setObjectName("ContextualToolsToolbar")
        toolbar.setMovable(False)
        toolbar.setFloatable(False)
        main_window.addToolBar(Qt.TopToolBarArea, toolbar)

        cancel_button = QPushButton("Cancel Loading")
        cancel_button.setStyleSheet(
            "background-color: #F56C6C; color: white; font-weight: bold; "
            "padding: 4px 8px; border-radius: 3px;"
        )
        cancel_button.clicked.connect(main_window.cancel_dataset_loading)
        cancel_action = toolbar.addWidget(cancel_button)

        integration_separator = toolbar.addSeparator()

        finish_button = QPushButton("Finish Integration & Merge")
        finish_button.setStyleSheet(
            "background-color: #67C23A; color: white; font-weight: bold; "
            "padding: 4px 8px; border-radius: 3px;"
        )
        finish_button.setToolTip("Finish reviewing and merge the dataset")
        finish_button.clicked.connect(main_window.viat_finish_integration)
        finish_action = toolbar.addWidget(finish_button)

        move_button = QPushButton("Move to Review")
        move_button.setStyleSheet(
            "background-color: #E6A23C; color: white; font-weight: bold; "
            "padding: 4px 8px; border-radius: 3px;"
        )
        move_button.setToolTip("Move the dataset to a review folder")
        if hasattr(main_window, "viat_finish_integration_move"):
            move_button.clicked.connect(main_window.viat_finish_integration_move)
        move_action = toolbar.addWidget(move_button)

        contextual_actions = [
            cancel_action,
            integration_separator,
            finish_action,
            move_action,
        ]
        for action in contextual_actions:
            action.setVisible(False)
            action.changed.connect(
                lambda _action=None: self._sync_contextual_toolbar(
                    toolbar, contextual_actions
                )
            )

        main_window.cancel_loading_btn = cancel_button
        main_window.cancel_loading_action = cancel_action
        main_window.finish_integration_sep = integration_separator
        main_window.finish_integration_action = finish_action
        main_window.finish_integration_move_action = move_action
        self._contextual_toolbar = toolbar
        self._sync_contextual_toolbar(toolbar, contextual_actions)

    @staticmethod
    def _sync_contextual_toolbar(toolbar, actions):
        toolbar.setVisible(any(action.isVisible() for action in actions))

    def create_dock_widgets(self):
        """Create and set up the dock widgets."""
        # Annotation dock
        self.main_window.annotation_dock = AnnotationDock(self.main_window)
        self.main_window.addDockWidget(
            Qt.RightDockWidgetArea, self.main_window.annotation_dock
        )

        # Class dock
        self.main_window.class_dock = ClassDock(self.main_window)
        self.main_window.addDockWidget(
            Qt.RightDockWidgetArea, self.main_window.class_dock
        )

        # SAM Interactive dock
        self.main_window.sam_interactive_dock = SAMInteractiveDock(self.main_window)
        self.main_window.addDockWidget(
            Qt.RightDockWidgetArea, self.main_window.sam_interactive_dock
        )
        self.main_window.sam_interactive_dock.hide() # Hidden by default

        # Track Queue dock (batch tracking jobs, run sequentially/overnight)
        # Placed on the LEFT, tabbed together with the Video Manager dock
        # below (see tabifyDockWidget after both are created) so they share
        # one panel instead of splitting the left side of the window.
        self.main_window.track_queue_dock = TrackQueueDock(self.main_window)
        self.main_window.addDockWidget(
            Qt.LeftDockWidgetArea, self.main_window.track_queue_dock
        )
        self.main_window.track_queue_dock.hide() # Hidden by default

        # Empty Frames Manager dock
        self.main_window.empty_frames_dock = EmptyFramesManagerDock(self.main_window)
        self.main_window.addDockWidget(
            Qt.RightDockWidgetArea, self.main_window.empty_frames_dock
        )
        self.main_window.empty_frames_dock.hide() # Hidden by default
        
        # Class Frames Manager dock
        self.main_window.class_frames_dock = ClassFramesManagerDock(self.main_window)
        self.main_window.addDockWidget(
            Qt.RightDockWidgetArea, self.main_window.class_frames_dock
        )
        self.main_window.class_frames_dock.hide() # Hidden by default
        
        # Uncertain Frames Manager dock
        self.main_window.uncertain_frames_dock = UncertainFramesManagerDock(self.main_window)
        self.main_window.addDockWidget(
            Qt.RightDockWidgetArea, self.main_window.uncertain_frames_dock
        )
        self.main_window.uncertain_frames_dock.hide() # Hidden by default

        # Video Manager dock
        self.main_window.video_manager_dock = VideoManagerDock(self.main_window)
        self.main_window.addDockWidget(
            Qt.LeftDockWidgetArea, self.main_window.video_manager_dock
        )
        self.main_window.video_manager_dock.hide() # Hidden by default

        # Tab the Track Queue and Video Manager docks together instead of
        # stacking them on top of each other in the left area -- opening
        # both used to split the left panel vertically, which got cramped
        # and hard to read. Tabbed, only one shows at a time and the user
        # switches between them with the tab bar.
        self.main_window.tabifyDockWidget(
            self.main_window.track_queue_dock, self.main_window.video_manager_dock
        )

        # Crop Settings dock
        self.main_window.crop_settings_dock = CropSettingsDock(self.main_window)
        self.main_window.addDockWidget(
            Qt.RightDockWidgetArea, self.main_window.crop_settings_dock
        )
        self.main_window.crop_settings_dock.hide() # Hidden by default
        
        # Clip Cuts dock
        self.main_window.clip_cuts_dock = ClipCutsDock("Clip Cuts", self.main_window)
        self.main_window.addDockWidget(
            Qt.RightDockWidgetArea, self.main_window.clip_cuts_dock
        )
        self.main_window.clip_cuts_dock.hide() # Hidden by default

        # Visual Inspector (SHOW) dock
        from viat.widgets import VisualInspectorDock
        self.main_window.visual_inspector_dock = VisualInspectorDock(self.main_window)
        self.main_window.addDockWidget(
            Qt.BottomDockWidgetArea, self.main_window.visual_inspector_dock
        )
        self.main_window.visual_inspector_dock.hide() # Hidden by default

        # Evaluation Inspector dock
        self.main_window.evaluation_inspector_dock = EvaluationInspectorDock(self.main_window)
        self.main_window.addDockWidget(
            Qt.RightDockWidgetArea, self.main_window.evaluation_inspector_dock
        )
        self.main_window.evaluation_inspector_dock.hide() # Hidden by default

        # Tabify all right side docks together so they share a single tabbed container
        right_docks = [
            self.main_window.annotation_dock,
            self.main_window.class_dock,
            self.main_window.evaluation_inspector_dock,
            self.main_window.sam_interactive_dock,
            self.main_window.empty_frames_dock,
            self.main_window.class_frames_dock,
            self.main_window.uncertain_frames_dock,
            self.main_window.crop_settings_dock,
            self.main_window.clip_cuts_dock,
        ]
        for i in range(len(right_docks) - 1):
            self.main_window.tabifyDockWidget(right_docks[i], right_docks[i + 1])

        # Configure dock area tab positions and options to prevent height ballooning
        self.main_window.setDockOptions(
            QMainWindow.AllowTabbedDocks | QMainWindow.AnimatedDocks
        )
        self.main_window.setTabPosition(Qt.RightDockWidgetArea, QTabWidget.North)
        self.main_window.setTabPosition(Qt.LeftDockWidgetArea, QTabWidget.North)

        self.main_window.annotation_dock.raise_()

        # Connect dock visibility signals to menu actions to keep check states in sync
        self._sync_view_menu_actions()

    def _sync_view_menu_actions(self):
        """Connect dock visibility signals to View menu actions to keep check states in sync."""
        mappings = [
            (getattr(self.main_window, 'annotation_dock', None), getattr(self, 'toggle_annotations_action', None)),
            (getattr(self.main_window, 'class_dock', None), getattr(self, 'toggle_classes_action', None)),
            (getattr(self.main_window, 'evaluation_inspector_dock', None), getattr(self, 'toggle_eval_inspector_action', None)),
            (getattr(self.main_window, 'sam_interactive_dock', None), getattr(self, 'toggle_sam_action', None)),
            (getattr(self.main_window, 'track_queue_dock', None), getattr(self, 'toggle_track_queue_action', None)),
            (getattr(self.main_window, 'empty_frames_dock', None), getattr(self, 'toggle_empty_frames_action', None)),
            (getattr(self.main_window, 'class_frames_dock', None), getattr(self, 'toggle_class_frames_action', None)),
            (getattr(self.main_window, 'uncertain_frames_dock', None), getattr(self, 'toggle_uncertain_frames_action', None)),
            (getattr(self.main_window, 'video_manager_dock', None), getattr(self, 'toggle_video_manager_action', None)),
            (getattr(self.main_window, 'clip_cuts_dock', None), getattr(self, 'toggle_clip_cuts_action', None)),
            (getattr(self.main_window, 'crop_settings_dock', None), getattr(self, 'toggle_crop_settings_action', None)),
            (getattr(self.main_window, 'visual_inspector_dock', None), getattr(self, 'toggle_visual_inspector_action', None)),
        ]
        for dock, action in mappings:
            if dock and action:
                try:
                    dock.visibilityChanged.connect(action.setChecked)
                except Exception:
                    pass

    def create_status_bar(self):
        """Create the status bar."""
        self.main_window.statusBar = QStatusBar()
        self.main_window.setStatusBar(self.main_window.statusBar)
        self.main_window.statusBar.showMessage("Ready")

    def create_playback_controls(self):
        """Create video playback controls."""
        # Create a widget to hold the controls
        playback_widget = QWidget()
        playback_layout = QHBoxLayout(playback_widget)
        playback_layout.setContentsMargins(5, 2, 5, 2)  # Reduce margins

        # Play/Pause button
        self.main_window.play_button = QPushButton()
        self.main_window.play_button.setIcon(
            self.main_window.icon_provider.get_icon("media-playback-start")
        )
        self.main_window.play_button.setToolTip("Play/Pause")
        self.main_window.play_button.clicked.connect(self.main_window.play_pause_video)
        self.main_window.play_button.setMaximumWidth(30)  # Make button smaller
        self.main_window.play_button.setMaximumHeight(24)  # Make button smaller

        # Previous frame button
        prev_button = QPushButton()
        prev_button.setIcon(
            self.main_window.icon_provider.get_icon("media-skip-backward")
        )
        prev_button.setToolTip("Previous Frame")
        prev_button.clicked.connect(self.main_window.prev_frame)
        prev_button.setMaximumWidth(30)  # Make button smaller
        prev_button.setMaximumHeight(24)  # Make button smaller

        # Next frame button
        next_button = QPushButton()
        next_button.setIcon(
            self.main_window.icon_provider.get_icon("media-skip-forward")
        )
        next_button.setToolTip("Next Frame")
        next_button.clicked.connect(self.main_window.next_frame)
        next_button.setMaximumWidth(30)  # Make button smaller
        next_button.setMaximumHeight(24)  # Make button smaller

        # Frame slider
        self.main_window.frame_slider = QSlider(Qt.Horizontal)
        self.main_window.frame_slider.setMinimum(0)
        self.main_window.frame_slider.setMaximum(
            100
        )  # Will be updated when video is loaded
        self.main_window.frame_slider.valueChanged.connect(
            self.main_window.slider_changed
        )
        self.main_window.frame_slider.setMaximumHeight(20)  # Make slider smaller



        # Frame counter label
        self.main_window.frame_label = QLabel("0/0")
        self.main_window.frame_label.setMaximumWidth(80)  # Limit width

        # Add widgets to layout
        playback_layout.addWidget(prev_button)
        playback_layout.addWidget(self.main_window.play_button)
        playback_layout.addWidget(next_button)
        playback_layout.addWidget(self.main_window.frame_slider)
        playback_layout.addWidget(self.main_window.frame_label)

        # Set fixed height for the entire widget
        playback_widget.setMaximumHeight(30)

        return playback_widget

    def setup_playback_timer(self):
        """Set up the timer for video playback."""
        self.main_window.play_timer = QTimer()
        self.main_window.play_timer.timeout.connect(self.main_window.next_frame)

    def create_settings_menu(self, menubar):
        """Create the Settings menu and its actions."""
        # Dark mode toggle
        darkmode_action = QAction("Enable Dark Mode", self.main_window, checkable=True)
        darkmode_action.setChecked(getattr(self.main_window, "dark_mode_enabled", False))
        darkmode_action.triggered.connect(self.main_window.toggle_dark_mode)
        menubar.addAction(darkmode_action)

        # Auto-save toggle
        autosave_action = QAction("Enable Auto-save", self.main_window, checkable=True)
        autosave_action.setChecked(self.main_window.autosave_enabled)
        autosave_action.triggered.connect(self.main_window.toggle_autosave)
        autosave_action.setToolTip(
            "Automatically save the project at regular intervals"
        )
        menubar.addAction(autosave_action)


        # Auto-save interval submenu
        interval_menu = menubar.addMenu("Auto-save Interval")

        # Add interval options
        intervals = [
            ("10 seconds", 10000),
            ("30 seconds", 30000),
            ("1 minutes", 60000),
            ("1.5 minutes", 90000),
            ("3 minutes", 180000),
        ]

        interval_group = QActionGroup(self.main_window)
        interval_group.setExclusive(True)

        for name, ms in intervals:
            action = QAction(name, self.main_window, checkable=True)
            action.setChecked(self.main_window.autosave_interval == ms)
            action.triggered.connect(
                lambda checked, ms=ms: self.main_window.set_autosave_interval(ms)
            )
            interval_group.addAction(action)
            interval_menu.addAction(action)

        # Add separator
        menubar.addSeparator()

        # Annotation attribute settings
        attr_dialog_action = QAction(
            "Show Attribute Dialog for New Annotations",
            self.main_window,
            checkable=True,
        )
        attr_dialog_action.setChecked(self.main_window.auto_show_attribute_dialog)
        attr_dialog_action.triggered.connect(self.main_window.toggle_attribute_dialog)
        attr_dialog_action.setToolTip(
            "Automatically show the attribute dialog when creating a new annotation "
        )

        menubar.addAction(attr_dialog_action)

        prev_attr_action = QAction(
            "Use Previous Annotation Attributes as Default",
            self.main_window,
            checkable=True,
        )
        prev_attr_action.setChecked(self.main_window.use_previous_attributes)
        prev_attr_action.triggered.connect(self.main_window.toggle_previous_attributes)
        prev_attr_action.setToolTip(
            "Use attribute values from previous annotations of the same class as default values"
        )
        menubar.addAction(prev_attr_action)
        menubar.addSeparator()

        slideshow_menu = menubar.addMenu("Slideshow Speed")

        # Add speed options
        speeds = [
            ("0.5x (2 seconds per image)", 0.5),
            ("1x (1 second per image)", 1.0),
            ("2x (0.5 seconds per image)", 2.0),
            ("3x (0.33 seconds per image)", 3.0),
            ("5x (0.2 seconds per image)", 5.0),
        ]

        speed_group = QActionGroup(self.main_window)
        speed_group.setExclusive(True)

        for name, speed in speeds:
            action = QAction(name, self.main_window, checkable=True)
            action.setChecked(speed == 1.0)  # Default to 1x
            action.triggered.connect(
                lambda checked, s=speed: self.main_window.set_slideshow_speed(s)
            )
            speed_group.addAction(action)
            slideshow_menu.addAction(action)

    def create_gpu_menu(self, menubar):
        """Create the GPU Device submenu, listing available GPUs by name with live VRAM usage.

        Picking one only saves a preference for the *next* launch (see
        gpu_utils.py for why a live switch isn't possible), so the menu also
        shows which GPU this running session actually initialized.
        """
        gpu_menu = menubar.addMenu("GPU Device")

        def refresh_gpu_menu():
            gpu_menu.clear()
            gpus = gpu_utils.list_gpus()
            if not gpus:
                no_gpu_action = QAction("No NVIDIA GPU detected (using CPU)", self.main_window)
                no_gpu_action.setEnabled(False)
                gpu_menu.addAction(no_gpu_action)
                return

            active_name = gpu_utils.get_active_gpu_name()
            active_action = QAction(
                f"Active this session: {active_name or 'CPU'}", self.main_window
            )
            active_action.setEnabled(False)
            gpu_menu.addAction(active_action)
            gpu_menu.addSeparator()

            current_index = gpu_utils.get_saved_gpu_index()
            if current_index is None:
                current_index = 0

            gpu_action_group = QActionGroup(self.main_window)
            gpu_action_group.setExclusive(True)
            for gpu in gpus:
                label = f"{gpu['index']}: {gpu['name']}"
                if gpu['total_gb'] is not None:
                    used_gb = gpu['total_gb'] - gpu['free_gb']
                    label += f"  ({used_gb:.1f}/{gpu['total_gb']:.1f} GB used)"
                action = QAction(label, self.main_window, checkable=True)
                action.setChecked(gpu['index'] == current_index)
                action.triggered.connect(
                    lambda checked, idx=gpu['index']: self.main_window.change_gpu_device(idx)
                )
                gpu_action_group.addAction(action)
                gpu_menu.addAction(action)
            # Keep the group alive as long as the menu (Qt parents it to main_window).
            gpu_menu._gpu_action_group = gpu_action_group

        gpu_menu.aboutToShow.connect(refresh_gpu_menu)
        refresh_gpu_menu()

    def create_interpolation_ui(self,):
        """
        Create UI elements for interpolation feature.
        
        Args:
            main_window: The main application window
        """
        # Add interpolation commands to the grouped Review menu.  The
        # temporary toolbar below still appears only while the mode is active.
        interpolation_menu = getattr(
            getattr(self.main_window, "toolbar", None), "review_menu", None
        )
        if interpolation_menu is None:
            menubar = self.main_window.menuBar()
            interpolation_menu = QMenu("&Interpolation", self.main_window)
            help_action = next(
                (
                    action
                    for action in menubar.actions()
                    if action.text().replace("&", "") == "Help"
                ),
                None,
            )
            if help_action is not None:
                menubar.insertMenu(help_action, interpolation_menu)
            else:
                menubar.addMenu(interpolation_menu)
        else:
            interpolation_menu.addSeparator()
        
        # Toggle interpolation mode
        toggle_interpolation_action = QAction("Enable Interpolation Mode", self.main_window)
        toggle_interpolation_action.setCheckable(True)
        toggle_interpolation_action.setChecked(False)
        toggle_interpolation_action.triggered.connect(self.main_window.toggle_interpolation_mode)
        interpolation_menu.addAction(toggle_interpolation_action)
        self.main_window.toggle_interpolation_action = toggle_interpolation_action
        
        # Set interpolation interval
        set_interval_action = QAction("Set Keyframe Interval...", self.main_window)
        set_interval_action.triggered.connect(self.main_window.set_interpolation_interval)
        interpolation_menu.addAction(set_interval_action)
        
        interpolation_menu.addSeparator()
        
        # Perform interpolation
        interpolate_action = QAction("Interpolate Now", self.main_window)
        interpolate_action.triggered.connect(self.main_window.perform_interpolation)
        # Ctrl+I is reserved for importing annotations; keep interpolation
        # discoverable without colliding with the File menu shortcut.
        interpolate_action.setShortcut("Ctrl+Shift+I")
        interpolation_menu.addAction(interpolate_action)
        
        # Add to toolbar
        interpolation_toolbar = self.main_window.addToolBar("Interpolation")
        interpolation_toolbar.setObjectName("interpolationToolbar")
        
        # Add keyframe indicator to toolbar
        keyframe_indicator = QLabel("  ")
        keyframe_indicator.setStyleSheet("background-color: transparent; min-width: 16px;")
        keyframe_indicator.setToolTip("Annotation indicator")
        self.main_window.keyframe_indicator = keyframe_indicator
        interpolation_toolbar.addWidget(keyframe_indicator)
        
        # Add toggle button to toolbar
        toggle_interpolation_button = QPushButton("Interpolation")
        toggle_interpolation_button.setCheckable(True)
        toggle_interpolation_button.setChecked(False)
        toggle_interpolation_button.clicked.connect(self.main_window.toggle_interpolation_mode)
        interpolation_toolbar.addWidget(toggle_interpolation_button)
        
        # Add interval selector
        interval_label = QLabel("Interval:")
        interpolation_toolbar.addWidget(interval_label)
        
        interval_spinner = QSpinBox()
        interval_spinner.setRange(2, 100)
        interval_spinner.setValue(10)  # Default interval
        interval_spinner.valueChanged.connect(lambda v: self.main_window.interpolation_manager.set_interval(v))
        self.main_window.interval_spinner = interval_spinner
        interpolation_toolbar.addWidget(interval_spinner)
        
        # Add interpolate button
        interpolate_button = QPushButton("Interpolate Now")
        interpolate_button.clicked.connect(self.main_window.perform_interpolation)
        interpolation_toolbar.addWidget(interpolate_button)
        
        # Make toolbar movable and floatable
        interpolation_toolbar.setMovable(True)
        interpolation_toolbar.setFloatable(True)
        
        # Initially hide the toolbar until interpolation mode is activated
        interpolation_toolbar.setVisible(False)
        self.main_window.interpolation_toolbar = interpolation_toolbar
