import os
from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, 
    QListWidget, QListWidgetItem, QPushButton, QLabel, QCheckBox,
    QLineEdit, QComboBox, QFrame
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QIcon, QColor


class VideoManagerDock(QDockWidget):
    """Dock widget for managing video sequences and multi-video datasets."""
    
    video_mode_toggled = pyqtSignal(bool)
    video_selected = pyqtSignal(str)
    prev_video_requested = pyqtSignal()
    next_video_requested = pyqtSignal()
    next_unannotated_requested = pyqtSignal()
    fast_export_requested = pyqtSignal()
    sam_tracking_toggled = pyqtSignal(bool)
    remove_video_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__("Video Manager", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self._is_video_dataset_mode = False
        self._dataset_videos = []  # List of VideoInfo or dicts
        self.setup_ui()
        
    def setup_ui(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        
        # Video Mode Toggle (for image datasets)
        self.chk_video_mode = QCheckBox("Enable Video Sequence Mode")
        self.chk_video_mode.toggled.connect(self.video_mode_toggled.emit)
        layout.addWidget(self.chk_video_mode)
        
        # Dataset Info Header / Stats Label
        self.lbl_stats = QLabel("No videos loaded")
        self.lbl_stats.setStyleSheet("font-size: 11px; color: #888888;")
        layout.addWidget(self.lbl_stats)
        
        # Search & Filter Container (for Video Dataset mode)
        self.filter_container = QWidget()
        filter_layout = QVBoxLayout(self.filter_container)
        filter_layout.setContentsMargins(0, 2, 0, 2)
        filter_layout.setSpacing(4)
        
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("🔍 Filter videos...")
        self.search_edit.textChanged.connect(self.apply_filter)
        filter_layout.addWidget(self.search_edit)
        
        filter_row = QHBoxLayout()
        filter_row.setSpacing(4)
        
        self.cmb_split_filter = QComboBox()
        self.cmb_split_filter.addItems(["All Splits"])
        self.cmb_split_filter.currentIndexChanged.connect(self.apply_filter)
        filter_row.addWidget(self.cmb_split_filter)
        
        self.cmb_status_filter = QComboBox()
        self.cmb_status_filter.addItems(["All Status", "Annotated", "Unannotated"])
        self.cmb_status_filter.currentIndexChanged.connect(self.apply_filter)
        filter_row.addWidget(self.cmb_status_filter)
        
        filter_layout.addLayout(filter_row)
        layout.addWidget(self.filter_container)
        self.filter_container.hide()  # Hidden until video dataset is loaded
        
        # Navigation Buttons Row 1 (Prev / Next)
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("◀ Prev Video")
        self.btn_prev.setToolTip("Previous Video (Shift+Left)")
        self.btn_prev.clicked.connect(self.prev_video_requested.emit)
        self.btn_prev.setEnabled(False)
        nav_layout.addWidget(self.btn_prev)
        
        self.btn_next = QPushButton("Next Video ▶")
        self.btn_next.setToolTip("Next Video (Shift+Right)")
        self.btn_next.clicked.connect(self.next_video_requested.emit)
        self.btn_next.setEnabled(False)
        nav_layout.addWidget(self.btn_next)
        layout.addLayout(nav_layout)
        
        # Navigation Buttons Row 2 (Next Unannotated / Fast Export)
        nav_layout2 = QHBoxLayout()
        self.btn_next_unannotated = QPushButton("Next Unannotated ⏭")
        self.btn_next_unannotated.setToolTip("Jump to next video without annotations")
        self.btn_next_unannotated.clicked.connect(self.next_unannotated_requested.emit)
        self.btn_next_unannotated.setEnabled(False)
        nav_layout2.addWidget(self.btn_next_unannotated)
        
        self.btn_fast_export = QPushButton("⚡ Export & Next")
        self.btn_fast_export.setToolTip("Fast export current video annotations and advance to next")
        self.btn_fast_export.clicked.connect(self.fast_export_requested.emit)
        self.btn_fast_export.setEnabled(False)
        nav_layout2.addWidget(self.btn_fast_export)
        layout.addLayout(nav_layout2)
        
        # Remove video / cut button
        self.btn_remove = QPushButton("🗑 Remove Video / Cut")
        self.btn_remove.clicked.connect(self.remove_video_requested.emit)
        self.btn_remove.setEnabled(False)
        layout.addWidget(self.btn_remove)
        
        # Video List Label
        self.lbl_list_title = QLabel("Videos:")
        layout.addWidget(self.lbl_list_title)
        
        # Video List Widget
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
        
    def set_mode(self, is_video_dataset: bool):
        """Toggle between simple Image Sequence mode and Video Dataset mode."""
        self._is_video_dataset_mode = is_video_dataset
        if is_video_dataset:
            self.chk_video_mode.hide()
            self.filter_container.show()
            self.btn_next_unannotated.show()
            self.btn_fast_export.show()
            self.lbl_list_title.setText("Dataset Videos:")
            self.btn_remove.setText("🗑 Remove Video from Dataset")
            self.set_active(True)
        else:
            self.chk_video_mode.show()
            self.filter_container.hide()
            self.btn_next_unannotated.hide()
            self.btn_fast_export.hide()
            self.lbl_list_title.setText("Videos:")
            self.btn_remove.setText("Remove Cut (All frames)")
            
    def set_videos(self, videos):
        """Update the list of videos (legacy / simple sequence mode)."""
        self.set_mode(False)
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        self.list_widget.addItems(videos)
        self.list_widget.blockSignals(False)
        self.lbl_stats.setText(f"Total sequences: {len(videos)}")
        
    def set_video_dataset(self, video_dataset_info):
        """Populate dock with a scanned VideoDatasetInfo object."""
        self.set_mode(True)
        self._dataset_videos = video_dataset_info.all_videos
        
        # Update split filter options
        self.cmb_split_filter.blockSignals(True)
        self.cmb_split_filter.clear()
        self.cmb_split_filter.addItem("All Splits")
        splits = [s.name for s in video_dataset_info.splits if s.name != "root"]
        if splits:
            for s_name in sorted(set(splits)):
                self.cmb_split_filter.addItem(f"Split: {s_name}")
            self.cmb_split_filter.show()
        else:
            self.cmb_split_filter.hide()
        self.cmb_split_filter.blockSignals(False)
        
        self.apply_filter()
        self.update_stats(video_dataset_info)

    def update_stats(self, video_dataset_info=None):
        """Update the summary stats label."""
        if not self._is_video_dataset_mode:
            return
        total = len(self._dataset_videos)
        annotated = sum(1 for v in self._dataset_videos if getattr(v, 'has_annotations', False))
        unannotated = total - annotated
        self.lbl_stats.setText(f"Total: {total} | 🟢 Annotated: {annotated} | ⚪ Unannotated: {unannotated}")

    def update_video_status(self, video_path: str, is_annotated: bool):
        """Update the annotation status badge of a specific video in the list."""
        # Update underlying VideoInfo object
        for v in self._dataset_videos:
            if v.path == video_path or os.path.abspath(v.path) == os.path.abspath(video_path):
                v.has_annotations = is_annotated
                v.status = "annotated" if is_annotated else "unannotated"
                break
                
        self.apply_filter()
        self.update_stats()

    def apply_filter(self):
        """Filter the list widget based on search query, split, and status."""
        if not self._is_video_dataset_mode:
            return
            
        search_query = self.search_edit.text().strip().lower()
        split_filter_idx = self.cmb_split_filter.currentIndex()
        split_filter = self.cmb_split_filter.currentText().replace("Split: ", "").strip() if split_filter_idx > 0 else None
        status_filter = self.cmb_status_filter.currentText()
        
        # Save current selected video path
        cur_item = self.list_widget.currentItem()
        cur_path = cur_item.data(Qt.UserRole) if cur_item else None
        
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        
        for v in self._dataset_videos:
            # 1. Search text filter
            if search_query and search_query not in v.filename.lower() and search_query not in v.relative_path.lower():
                continue
                
            # 2. Split filter
            if split_filter and v.split != split_filter:
                continue
                
            # 3. Status filter
            if status_filter == "Annotated" and not v.has_annotations:
                continue
            elif status_filter == "Unannotated" and v.has_annotations:
                continue
                
            # Create item
            status_icon = "🟢" if v.has_annotations else "⚪"
            display_text = f"{status_icon} {v.filename}"
            if v.split and v.split != "root":
                display_text += f" [{v.split}]"
                
            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, v.path)
            item.setData(Qt.UserRole + 1, v.filename)
            item.setData(Qt.UserRole + 2, v.split)
            item.setData(Qt.UserRole + 3, v.has_annotations)
            item.setToolTip(f"Path: {v.path}\nStatus: {'Annotated' if v.has_annotations else 'Unannotated'}\nSplit: {v.split}")
            
            self.list_widget.addItem(item)
            if cur_path and v.path == cur_path:
                self.list_widget.setCurrentItem(item)
                
        self.list_widget.blockSignals(False)

    def select_video(self, video_identifier):
        """Select a video by filename or full path without triggering signals."""
        self.list_widget.blockSignals(True)
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item:
                path = item.data(Qt.UserRole)
                name = item.data(Qt.UserRole + 1)
                text = item.text()
                if video_identifier in (path, name, text):
                    self.list_widget.setCurrentItem(item)
                    break
        self.list_widget.blockSignals(False)
        
    def get_selected_video_path(self):
        """Get the full path of the currently selected video."""
        current = self.list_widget.currentItem()
        if not current:
            return None
        if self._is_video_dataset_mode:
            return current.data(Qt.UserRole)
        return current.text()
        
    def on_selection_changed(self, current, previous):
        if current:
            if self._is_video_dataset_mode:
                video_path = current.data(Qt.UserRole)
                if video_path:
                    self.video_selected.emit(video_path)
            else:
                self.video_selected.emit(current.text())
            
    def set_active(self, is_active):
        """Enable or disable controls based on video mode state."""
        self.list_widget.setEnabled(is_active)
        self.btn_prev.setEnabled(is_active)
        self.btn_next.setEnabled(is_active)
        self.btn_next_unannotated.setEnabled(is_active)
        self.btn_fast_export.setEnabled(is_active)
        self.btn_remove.setEnabled(is_active)
        self.btn_sam_track.setEnabled(is_active)
