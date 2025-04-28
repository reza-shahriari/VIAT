"""
UI Creator module for VIAT application.

This module contains functions for creating various UI elements of the
Video Image Annotation Tool, including menus, toolbars, and controls.
"""

from PyQt5.QtWidgets import (
    QAction, QActionGroup, QLabel, QComboBox, QWidget, 
    QHBoxLayout, QPushButton, QSlider, QStatusBar
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QIcon
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from viat.widgets import AnnotationDock, ClassDock, AnnotationToolbar


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

        # Tools menu
        self.create_tools_menu(menubar)

        # Settings menu
        self.main_window.settings_menu = menubar.addMenu("Settings")
        self.create_settings_menu(self.main_window.settings_menu)

        # Style menu
        self.create_style_menu(menubar)

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
        open_image_folder_action.setShortcut("Ctrl+I")
        open_image_folder_action.triggered.connect(self.main_window.open_image_folder)
        file_menu.addAction(open_image_folder_action)

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
        import_action.triggered.connect(lambda: self.main_window.import_annotations(None))
        file_menu.addAction(import_action)

        # Export Annotations action
        export_action = QAction("Export Annotations", self.main_window)
        export_action.setShortcut("Ctrl+E")
        export_action.triggered.connect(self.main_window.export_annotations)
        file_menu.addAction(export_action)

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

        # Clear Annotations action
        clear_action = QAction("Clear Annotations", self.main_window)
        clear_action.triggered.connect(self.main_window.clear_annotations)
        edit_menu.addAction(clear_action)

        # Add Annotation action
        add_action = QAction("Add Annotation", self.main_window)
        add_action.triggered.connect(self.main_window.add_annotation)
        edit_menu.addAction(add_action)

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

    def create_tools_menu(self, menubar):
        """Create the Tools menu and its actions."""
        tools_menu = menubar.addMenu("Tools")

        # Auto Label action
        auto_label_action = QAction("Auto Label", self.main_window)
        auto_label_action.triggered.connect(self.main_window.auto_label)
        tools_menu.addAction(auto_label_action)

        # Track Objects action
        track_action = QAction("Track Objects", self.main_window)
        track_action.triggered.connect(self.main_window.track_objects)
        tools_menu.addAction(track_action)

        # Smart Edge Movement action
        smart_edge_action = QAction("Smart Edge Movement", self.main_window, checkable=True)
        smart_edge_action.setShortcut("Ctrl+E")
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

        # Store reference to keep menu and toolbar in sync
        self.main_window.smart_edge_action = smart_edge_action

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
        """Create the annotation toolbar."""
        self.main_window.toolbar = AnnotationToolbar(self.main_window, self.main_window.icon_provider)
        self.main_window.addToolBar(self.main_window.toolbar)
        self.main_window.class_selector = self.main_window.toolbar.class_selector

        # Add annotation method selector
        method_label = QLabel("Method:")
        self.main_window.toolbar.addWidget(method_label)

        self.main_window.method_selector = QComboBox()
        self.main_window.method_selector.addItems(["Drag", "TwoClick"])
        self.main_window.method_selector.setToolTip(
            "Drag: Click and drag to create box\nTwoClick: Click two corners to create box"
        )
        self.main_window.method_selector.currentTextChanged.connect(self.main_window.change_annotation_method)
        self.main_window.toolbar.addWidget(self.main_window.method_selector)

    def create_dock_widgets(self):
        """Create and set up the dock widgets."""
        # Annotation dock
        self.main_window.annotation_dock = AnnotationDock(self.main_window)
        self.main_window.addDockWidget(Qt.RightDockWidgetArea, self.main_window.annotation_dock)

        # Class dock
        self.main_window.class_dock = ClassDock(self.main_window)
        self.main_window.addDockWidget(Qt.RightDockWidgetArea, self.main_window.class_dock)

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
        self.main_window.play_button.setIcon(self.main_window.icon_provider.get_icon("media-playback-start"))
        self.main_window.play_button.setToolTip("Play/Pause")
        self.main_window.play_button.clicked.connect(self.main_window.play_pause_video)
        self.main_window.play_button.setMaximumWidth(30)  # Make button smaller
        self.main_window.play_button.setMaximumHeight(24)  # Make button smaller

        # Previous frame button
        prev_button = QPushButton()
        prev_button.setIcon(self.main_window.icon_provider.get_icon("media-skip-backward"))
        prev_button.setToolTip("Previous Frame")
        prev_button.clicked.connect(self.main_window.prev_frame)
        prev_button.setMaximumWidth(30)  # Make button smaller
        prev_button.setMaximumHeight(24)  # Make button smaller

        # Next frame button
        next_button = QPushButton()
        next_button.setIcon(self.main_window.icon_provider.get_icon("media-skip-forward"))
        next_button.setToolTip("Next Frame")
        next_button.clicked.connect(self.main_window.next_frame)
        next_button.setMaximumWidth(30)  # Make button smaller
        next_button.setMaximumHeight(24)  # Make button smaller

        # Frame slider
        self.main_window.frame_slider = QSlider(Qt.Horizontal)
        self.main_window.frame_slider.setMinimum(0)
        self.main_window.frame_slider.setMaximum(100)  # Will be updated when video is loaded
        self.main_window.frame_slider.valueChanged.connect(self.main_window.slider_changed)
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
            "Show Attribute Dialog for New Annotations", self.main_window, checkable=True
        )
        attr_dialog_action.setChecked(self.main_window.auto_show_attribute_dialog)
        attr_dialog_action.triggered.connect(self.main_window.toggle_attribute_dialog)
        attr_dialog_action.setToolTip(
            "Automatically show the attribute dialog when creating a new annotation (Ctrl+A)"
        )
        attr_dialog_action.setShortcut("Ctrl+A")
        menubar.addAction(attr_dialog_action)

        prev_attr_action = QAction(
            "Use Previous Annotation Attributes as Default", self.main_window, checkable=True
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
