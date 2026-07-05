import os
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, 
    QPushButton, QFileDialog, QDoubleSpinBox, QMessageBox, QGroupBox, QFormLayout
)
from PyQt5.QtCore import Qt

class ZeroShotClassificationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Zero-Shot Classification Refiner')
        self.setMinimumWidth(500)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # JSON Config File
        json_group = QGroupBox("Configuration File")
        json_layout = QHBoxLayout()
        self.json_path_input = QLineEdit()
        self.json_path_input.setPlaceholderText("Path to hierarchy rules JSON...")
        self.btn_browse = QPushButton("Browse")
        self.btn_browse.clicked.connect(self.browse_json)
        json_layout.addWidget(self.json_path_input)
        json_layout.addWidget(self.btn_browse)
        json_group.setLayout(json_layout)
        layout.addWidget(json_group)

        # Model Settings
        settings_group = QGroupBox("Model Settings")
        form_layout = QFormLayout()

        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14",
            "google/siglip-base-patch16-224",
            "google/siglip-so400m-patch14-384"
        ])
        self.model_combo.setEditable(True)
        form_layout.addRow("Classification Model:", self.model_combo)

        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(0.3)
        self.confidence_spin.setToolTip("Minimum confidence required to change the label.")
        form_layout.addRow("Min Confidence:", self.confidence_spin)

        self.overlap_margin_spin = QDoubleSpinBox()
        self.overlap_margin_spin.setRange(0.0, 1.0)
        self.overlap_margin_spin.setSingleStep(0.01)
        self.overlap_margin_spin.setValue(0.05)
        self.overlap_margin_spin.setToolTip("If the difference between Top-1 and Top-2 is less than this, they might be overlapping.")
        form_layout.addRow("Overlap Margin:", self.overlap_margin_spin)

        settings_group.setLayout(form_layout)
        layout.addWidget(settings_group)

        # Action Buttons
        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton('Run Classification')
        self.btn_run.clicked.connect(self.validate_and_accept)
        self.btn_cancel = QPushButton('Cancel')
        self.btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

    def browse_json(self):
        default_dir = os.path.join(os.getcwd(), "checkpoints", "zero_shot")
        os.makedirs(default_dir, exist_ok=True)
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select JSON Rules File", default_dir, "JSON Files (*.json)"
        )
        if file_path:
            self.json_path_input.setText(file_path)

    def validate_and_accept(self):
        if not os.path.exists(self.json_path_input.text()):
            QMessageBox.warning(self, "Invalid File", "Please select a valid JSON configuration file.")
            return
        self.accept()

    def get_config(self):
        return {
            'json_path': self.json_path_input.text(),
            'model_type': self.model_combo.currentText(),
            'min_confidence': self.confidence_spin.value(),
            'overlap_margin': self.overlap_margin_spin.value()
        }
