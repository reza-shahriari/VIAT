from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QComboBox,
    QSpinBox,
    QLabel,
    QDialogButtonBox,
    QMessageBox
)
from PyQt5.QtCore import Qt

class TrackingDialog(QDialog):
    def __init__(self, parent, tracker_manager, current_frame, max_frame, target_class):
        super().__init__(parent)
        self.setWindowTitle("Track Object")
        self.tracker_manager = tracker_manager
        self.current_frame = current_frame
        self.max_frame = max_frame
        self.target_class = target_class
        
        self.selected_tracker_name = None
        self.end_frame = current_frame
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        form_layout = QFormLayout()
        
        # Target Info
        self.target_label = QLabel(f"<b>Target:</b> {self.target_class}")
        form_layout.addRow(self.target_label)
        
        # Tracker Selection
        self.tracker_combo = QComboBox()
        tracker_info = self.tracker_manager.get_tracker_info()
        for name, info in tracker_info.items():
            display_name = name
            if not info["available"]:
                display_name += f" (Unavailable: {info['message']})"
            self.tracker_combo.addItem(display_name, userData=name)
            
            # Disable item if not available
            if not info["available"]:
                model = self.tracker_combo.model()
                item = model.item(self.tracker_combo.count() - 1)
                item.setEnabled(False)
                
        form_layout.addRow("Algorithm:", self.tracker_combo)
        
        # Start Frame (Read Only)
        self.start_frame_spin = QSpinBox()
        self.start_frame_spin.setRange(0, self.max_frame)
        self.start_frame_spin.setValue(self.current_frame)
        self.start_frame_spin.setEnabled(False)
        form_layout.addRow("Start Frame:", self.start_frame_spin)
        
        # End Frame
        self.end_frame_spin = QSpinBox()
        self.end_frame_spin.setRange(self.current_frame + 1, self.max_frame)
        self.end_frame_spin.setValue(self.max_frame)
        form_layout.addRow("End Frame:", self.end_frame_spin)
        
        layout.addLayout(form_layout)
        
        # Buttons
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.validate_and_accept)
        self.button_box.rejected.connect(self.reject)
        self.button_box.button(QDialogButtonBox.Ok).setText("Track")
        layout.addWidget(self.button_box)
        
    def validate_and_accept(self):
        idx = self.tracker_combo.currentIndex()
        if idx < 0:
            QMessageBox.warning(self, "Validation Error", "Please select a tracker.")
            return
            
        tracker_name = self.tracker_combo.itemData(idx)
        info = self.tracker_manager.get_tracker_info().get(tracker_name)
        
        if not info or not info["available"]:
            QMessageBox.warning(self, "Validation Error", f"Tracker {tracker_name} is not available.")
            return
            
        self.selected_tracker_name = tracker_name
        self.end_frame = self.end_frame_spin.value()
        self.accept()
