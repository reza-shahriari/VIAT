import os
from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, 
    QListWidget, QPushButton, QLabel, QCheckBox
)
from PyQt5.QtCore import pyqtSignal, Qt

class VideoManagerDock(QDockWidget):
    """Dock widget for managing video sequences."""
    
    video_mode_toggled = pyqtSignal(bool)
    video_selected = pyqtSignal(str)
    prev_video_requested = pyqtSignal()
    next_video_requested = pyqtSignal()
    sam_tracking_toggled = pyqtSignal(bool)
    remove_video_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__("Video Manager", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.setup_ui()
        
    def setup_ui(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Video Mode Toggle
        self.chk_video_mode = QCheckBox("Enable Video Mode")
        self.chk_video_mode.toggled.connect(self.video_mode_toggled.emit)
        layout.addWidget(self.chk_video_mode)
        
        # Navigation Buttons
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("Previous Video")
        self.btn_prev.clicked.connect(self.prev_video_requested.emit)
        self.btn_prev.setEnabled(False)
        nav_layout.addWidget(self.btn_prev)
        
        self.btn_next = QPushButton("Next Video")
        self.btn_next.clicked.connect(self.next_video_requested.emit)
        self.btn_next.setEnabled(False)
        nav_layout.addWidget(self.btn_next)
        
        layout.addLayout(nav_layout)
        
        self.btn_remove = QPushButton("Remove Cut (All frames)")
        self.btn_remove.clicked.connect(self.remove_video_requested.emit)
        self.btn_remove.setEnabled(False)
        layout.addWidget(self.btn_remove)
        
        # Video List
        layout.addWidget(QLabel("Videos:"))
        self.list_widget = QListWidget()
        self.list_widget.currentItemChanged.connect(self.on_selection_changed)
        self.list_widget.setEnabled(False)
        layout.addWidget(self.list_widget)
        
        # SAM Tracking button
        self.btn_sam_track = QPushButton("SAM Tracking")
        self.btn_sam_track.setCheckable(True)
        self.btn_sam_track.setEnabled(False)
        self.btn_sam_track.toggled.connect(self.sam_tracking_toggled.emit)
        layout.addWidget(self.btn_sam_track)
        
        self.setWidget(widget)
        
    def set_videos(self, videos):
        """Update the list of videos."""
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        self.list_widget.addItems(videos)
        self.list_widget.blockSignals(False)
        
    def select_video(self, video_name):
        """Select a specific video in the list without triggering signals."""
        self.list_widget.blockSignals(True)
        items = self.list_widget.findItems(video_name, Qt.MatchExactly)
        if items:
            self.list_widget.setCurrentItem(items[0])
        self.list_widget.blockSignals(False)
        
    def on_selection_changed(self, current, previous):
        if current:
            self.video_selected.emit(current.text())
            
    def set_active(self, is_active):
        """Enable or disable controls based on video mode state."""
        self.list_widget.setEnabled(is_active)
        self.btn_prev.setEnabled(is_active)
        self.btn_next.setEnabled(is_active)
        self.btn_remove.setEnabled(is_active)
        self.btn_sam_track.setEnabled(is_active)
