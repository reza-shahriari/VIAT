import os
from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, 
    QListWidget, QPushButton, QLabel, QComboBox,
    QListWidgetItem, QCheckBox, QSpinBox
)
from PyQt5.QtCore import pyqtSignal, Qt

class ClassFramesManagerDock(QDockWidget):
    """Dock widget for managing and filtering frames by class presence."""
    
    refresh_requested = pyqtSignal()
    frame_selected = pyqtSignal(object)
    delete_labels_requested = pyqtSignal(list, str)  # frame_indices, class_name
    zero_shot_requested = pyqtSignal(str, str)  # prompt (class_name), model_type
    filter_toggled = pyqtSignal(bool, str, int, str)  # is_active, mode, count, class_name
    
    def __init__(self, parent=None):
        super().__init__("Class Frames Manager", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.matching_frames = []
        self.total_frames = 0
        self.setup_ui()
        
    def setup_ui(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Filter Mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Condition:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "More than",
            "Less than",
            "Exactly"
        ])
        self.mode_combo.currentIndexChanged.connect(self._on_filter_criteria_changed)
        mode_layout.addWidget(self.mode_combo)
        
        self.count_spin = QSpinBox()
        self.count_spin.setMinimum(0)
        self.count_spin.setMaximum(9999)
        self.count_spin.setValue(0)
        self.count_spin.valueChanged.connect(self._on_filter_criteria_changed)
        mode_layout.addWidget(self.count_spin)
        
        layout.addLayout(mode_layout)
        
        # Target Class
        class_layout = QHBoxLayout()
        class_layout.addWidget(QLabel("Class:"))
        self.class_combo = QComboBox()
        self.class_combo.currentIndexChanged.connect(self._on_filter_criteria_changed)
        class_layout.addWidget(self.class_combo)
        layout.addLayout(class_layout)
        
        # Filter checkbox
        self.chk_filter = QCheckBox("Navigate Only Matching Frames")
        self.chk_filter.toggled.connect(self._on_filter_toggled)
        layout.addWidget(self.chk_filter)
        
        # Info label
        self.info_label = QLabel("Matching frames: 0 / 0")
        layout.addWidget(self.info_label)
        
        # List of frames
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(QListWidget.ExtendedSelection)
        self.list_widget.itemDoubleClicked.connect(self.on_double_clicked)
        self.list_widget.itemSelectionChanged.connect(self.on_selection_changed)
        layout.addWidget(self.list_widget)
        
        # Action buttons
        btn_layout = QHBoxLayout()
        self.btn_refresh = QPushButton("Refresh List")
        self.btn_refresh.clicked.connect(self.refresh_requested.emit)
        btn_layout.addWidget(self.btn_refresh)
        
        self.btn_delete = QPushButton("Delete Selected Labels")
        self.btn_delete.clicked.connect(self.on_delete_clicked)
        btn_layout.addWidget(self.btn_delete)
        layout.addLayout(btn_layout)
        
        # Zero-shot detect
        zs_layout = QVBoxLayout()
        zs_layout.addWidget(QLabel("Zero-Shot Model:"))
        self.zs_model_combo = QComboBox()
        self.zs_model_combo.addItems(['yolov8x-worldv2.pt', 'yolo11x.pt'])
        zs_layout.addWidget(self.zs_model_combo)
        
        self.btn_zs_detect = QPushButton("Zero-Shot Detect on Selected")
        self.btn_zs_detect.clicked.connect(self.on_zs_detect)
        zs_layout.addWidget(self.btn_zs_detect)
        layout.addLayout(zs_layout)
        
        self.setWidget(widget)
        
    def _on_filter_criteria_changed(self):
        if self.chk_filter.isChecked():
            self._on_filter_toggled(True)
        self.refresh_requested.emit()
            
    def _on_filter_toggled(self, checked):
        mode = self.mode_combo.currentText()
        count = self.count_spin.value()
        class_name = self.class_combo.currentText()
        self.filter_toggled.emit(checked, mode, count, class_name)
        
    def get_filter_state(self):
        """Returns mode, count, and class_name."""
        return self.mode_combo.currentText(), self.count_spin.value(), self.class_combo.currentText()

    def update_classes(self, classes):
        """Update the class combo box with new classes."""
        current = self.class_combo.currentText()
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        self.class_combo.addItems(classes)
        
        # Restore selection if it still exists
        idx = self.class_combo.findText(current)
        if idx >= 0:
            self.class_combo.setCurrentIndex(idx)
            
        self.class_combo.blockSignals(False)
        self.refresh_requested.emit()

    def update_data(self, matching_frames, total_frames):
        """Update the list with new data from the main window."""
        self.matching_frames = matching_frames
        self.total_frames = total_frames
        
        self.list_widget.clear()
        for item_data in matching_frames:
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
            
        label_text = "Matching videos:" if (matching_frames and isinstance(matching_frames[0], str)) else "Matching frames:"
        self.info_label.setText(f"{label_text} {len(matching_frames)} / {total_frames}")
        
    def on_double_clicked(self, item):
        item_data = item.data(Qt.UserRole)
        self.frame_selected.emit(item_data)

    def on_selection_changed(self):
        items = self.list_widget.selectedItems()
        if items:
            item_data = items[0].data(Qt.UserRole)
            self.frame_selected.emit(item_data)
            
    def on_delete_clicked(self):
        items = self.list_widget.selectedItems()
        if items:
            frame_indices = [item.data(Qt.UserRole) for item in items]
        else:
            frame_indices = getattr(self, 'matching_frames', [])
            
        if not frame_indices:
            return
            
        class_name = self.class_combo.currentText()
        if class_name:
            self.delete_labels_requested.emit(frame_indices, class_name)

    def on_zs_detect(self):
        class_name = self.class_combo.currentText()
        if class_name:
            self.zero_shot_requested.emit(class_name, self.zs_model_combo.currentText())
