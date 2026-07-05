from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, 
    QListWidget, QPushButton, QLabel, QListWidgetItem
)
from PyQt5.QtCore import pyqtSignal, Qt

class UncertainFramesManagerDock(QDockWidget):
    """Dock widget for managing frames with uncertain zero-shot classification annotations."""
    
    refresh_requested = pyqtSignal()
    frame_selected = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__("Uncertain Classifications", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.uncertain_frames = []
        self.setup_ui()
        
    def setup_ui(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Info label
        self.info_label = QLabel("Frames with uncertain annotations:")
        layout.addWidget(self.info_label)
        
        # List of frames
        self.list_widget = QListWidget()
        self.list_widget.itemDoubleClicked.connect(self.on_double_clicked)
        layout.addWidget(self.list_widget)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh List")
        self.btn_refresh.clicked.connect(self.refresh_requested.emit)
        btn_layout.addWidget(self.btn_refresh)
        
        # Clear selected uncertainty
        self.btn_clear_uncertain = QPushButton("Mark Verified")
        self.btn_clear_uncertain.clicked.connect(self.on_clear_uncertain)
        btn_layout.addWidget(self.btn_clear_uncertain)
        
        layout.addLayout(btn_layout)
        
        self.setWidget(widget)

    def update_data(self, uncertain_frames):
        """Update the list with frames containing uncertain annotations."""
        self.uncertain_frames = uncertain_frames
        
        self.list_widget.clear()
        for frame_idx in uncertain_frames:
            item = QListWidgetItem(f"Frame {frame_idx}")
            item.setData(Qt.UserRole, frame_idx)
            self.list_widget.addItem(item)
            
        self.info_label.setText(f"Uncertain Frames: {len(uncertain_frames)}")
        
    def on_double_clicked(self, item):
        frame_idx = item.data(Qt.UserRole)
        self.frame_selected.emit(frame_idx)
        
    def on_clear_uncertain(self):
        """
        Mark all uncertain annotations in the currently selected frame as verified (uncertain=False).
        """
        items = self.list_widget.selectedItems()
        if not items:
            return
            
        frame_idx = items[0].data(Qt.UserRole)
        main_window = self.parent()
        if main_window and hasattr(main_window, 'frame_annotations'):
            annotations = main_window.frame_annotations.get(frame_idx, [])
            changed = False
            for bbox in annotations:
                if getattr(bbox, 'uncertain', False):
                    bbox.uncertain = False
                    changed = True
            
            if changed:
                if main_window.current_frame == frame_idx:
                    main_window.canvas.update()
                self.refresh_requested.emit()
