import os
from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, 
    QListWidget, QPushButton, QLabel, QComboBox, QMessageBox,
    QListWidgetItem, QCheckBox, QLineEdit
)
from PyQt5.QtCore import pyqtSignal, Qt

class EmptyFramesManagerDock(QDockWidget):
    """Dock widget for managing and predicting empty frames."""
    
    predict_requested = pyqtSignal(int, int, str)  # target_frame, source_frame, model_type
    predict_all_requested = pyqtSignal(str)  # model_type
    zero_shot_requested = pyqtSignal(str, str)  # prompt, model_type
    refresh_requested = pyqtSignal()
    frame_selected = pyqtSignal(object)
    filter_toggled = pyqtSignal(bool)
    
    def __init__(self, parent=None):
        super().__init__("Empty Frames Manager", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.empty_frames = []
        self.annotated_frames = set()
        self.setup_ui()
        
    def setup_ui(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Info label
        self.info_label = QLabel("Frames without annotations:")
        layout.addWidget(self.info_label)
        
        # Filter checkbox
        self.chk_filter = QCheckBox("Navigate Only Empty Frames")
        self.chk_filter.toggled.connect(self.filter_toggled.emit)
        layout.addWidget(self.chk_filter)
        
        # List of empty frames
        self.list_widget = QListWidget()
        self.list_widget.itemSelectionChanged.connect(self.on_selection_changed)
        self.list_widget.itemDoubleClicked.connect(self.on_double_clicked)
        layout.addWidget(self.list_widget)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh List")
        self.btn_refresh.clicked.connect(self.refresh_requested.emit)
        btn_layout.addWidget(self.btn_refresh)
        layout.addLayout(btn_layout)
        
        # Model Selection
        model_layout = QVBoxLayout()
        model_layout.addWidget(QLabel("Tracking Model:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "SAM2 Fast (sam2.1_s.pt)",
            "SAM2 Huge (sam2.1_l.pt)"
        ])
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)
        
        # Predict actions
        self.btn_predict_prev = QPushButton("Predict from Previous Frame")
        self.btn_predict_prev.setEnabled(False)
        self.btn_predict_prev.clicked.connect(self.on_predict_prev)
        layout.addWidget(self.btn_predict_prev)
        
        self.btn_predict_next = QPushButton("Predict from Next Frame")
        self.btn_predict_next.setEnabled(False)
        self.btn_predict_next.clicked.connect(self.on_predict_next)
        layout.addWidget(self.btn_predict_next)
        
        # Predict all action
        self.btn_predict_all = QPushButton("Predict All Possible Empty Frames")
        self.btn_predict_all.clicked.connect(self.on_predict_all)
        layout.addWidget(self.btn_predict_all)
        
        # Zero-shot detect
        zs_layout = QVBoxLayout()
        zs_layout.addWidget(QLabel("Zero-Shot Model:"))
        self.zs_model_combo = QComboBox()
        self.zs_model_combo.addItems(['yolov8x-worldv2.pt', 'yolo11x.pt'])
        zs_layout.addWidget(self.zs_model_combo)
        
        self.zs_prompt_input = QLineEdit()
        self.zs_prompt_input.setPlaceholderText("Class prompt (e.g. 'car')")
        zs_layout.addWidget(self.zs_prompt_input)
        
        self.btn_zs_detect = QPushButton("Zero-Shot Detect Empty Frames")
        self.btn_zs_detect.clicked.connect(self.on_zs_detect)
        zs_layout.addWidget(self.btn_zs_detect)
        layout.addLayout(zs_layout)
        
        self.setWidget(widget)
        
    def get_selected_model_type(self):
        text = self.model_combo.currentText()
        if "sam2.1_l.pt" in text:
            return "sam2.1_l.pt"
        return "sam2.1_s.pt"

    def update_data(self, empty_frames, annotated_frames, total_frames):
        """Update the list with new data from the main window."""
        self.empty_frames = empty_frames
        self.annotated_frames = set(annotated_frames)
        self.total_frames = total_frames
        
        self.list_widget.clear()
        for item_data in empty_frames:
            if isinstance(item_data, tuple):
                display_text, actual_data = item_data
                item = QListWidgetItem(display_text)
                item.setData(Qt.UserRole, actual_data)
            elif isinstance(item_data, str):
                item = QListWidgetItem(f"Video {item_data}")
                item.setData(Qt.UserRole, item_data)
            else:
                item = QListWidgetItem(f"Frame {item_data}")
                item.setData(Qt.UserRole, item_data)
            self.list_widget.addItem(item)
            
        label_text = "Empty videos:" if (empty_frames and isinstance(empty_frames[0], str)) else "Empty frames:"
        self.info_label.setText(f"{label_text} {len(empty_frames)} / {total_frames}")
        self.on_selection_changed()
        
    def on_selection_changed(self):
        items = self.list_widget.selectedItems()
        if not items:
            self.btn_predict_prev.setEnabled(False)
            self.btn_predict_next.setEnabled(False)
            return
            
        item_data = items[0].data(Qt.UserRole)
        self.frame_selected.emit(item_data)
        
        if isinstance(item_data, str):
            self.btn_predict_prev.setEnabled(False)
            self.btn_predict_next.setEnabled(False)
            self.btn_predict_prev.setText("Predict from Previous")
            self.btn_predict_next.setText("Predict from Next")
            return
            
        frame_idx = item_data
        
        # Check if previous frame has annotations
        prev_has_annotations = (frame_idx - 1) in self.annotated_frames
        self.btn_predict_prev.setEnabled(prev_has_annotations)
        if prev_has_annotations:
            self.btn_predict_prev.setText(f"Predict from Frame {frame_idx - 1}")
        else:
            self.btn_predict_prev.setText("Predict from Previous Frame")
            
        # Check if next frame has annotations
        next_has_annotations = (frame_idx + 1) in self.annotated_frames
        self.btn_predict_next.setEnabled(next_has_annotations)
        if next_has_annotations:
            self.btn_predict_next.setText(f"Predict from Frame {frame_idx + 1}")
        else:
            self.btn_predict_next.setText("Predict from Next Frame")
            
    def on_double_clicked(self, item):
        frame_idx = item.data(Qt.UserRole)
        self.frame_selected.emit(frame_idx)
        
    def on_predict_prev(self):
        items = self.list_widget.selectedItems()
        if items:
            frame_idx = items[0].data(Qt.UserRole)
            self.predict_requested.emit(frame_idx, frame_idx - 1, self.get_selected_model_type())
            
    def on_predict_next(self):
        items = self.list_widget.selectedItems()
        if items:
            frame_idx = items[0].data(Qt.UserRole)
            self.predict_requested.emit(frame_idx, frame_idx + 1, self.get_selected_model_type())

    def on_predict_all(self):
        self.predict_all_requested.emit(self.get_selected_model_type())

    def on_zs_detect(self):
        prompt = self.zs_prompt_input.text().strip()
        if prompt:
            self.zero_shot_requested.emit(prompt, self.zs_model_combo.currentText())
