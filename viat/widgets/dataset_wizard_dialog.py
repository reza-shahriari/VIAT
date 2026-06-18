from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QFileDialog, QMessageBox, QGroupBox, QLineEdit, QProgressBar,
    QFormLayout
)
from PyQt5.QtCore import Qt, pyqtSignal
import os

class DatasetWizardDialog(QDialog):
    """Setup dialog for the interactive Dataset Integration workflow."""
    
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
        self.setWindowTitle("Dataset Integration Setup")
        self.resize(600, 400)
        self.settings = None
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Step 1: Main Dataset
        main_dataset_group = QGroupBox("1. Main Dataset (Target Merge Destination)")
        main_dataset_layout = QHBoxLayout()
        self.main_dataset_input = QLineEdit()
        self.main_dataset_input.setReadOnly(True)
        # Load from config if available
        from config import DEFAULT_SETTINGS
        saved_path = DEFAULT_SETTINGS.get("main_dataset_path", "")
        self.main_dataset_input.setText(saved_path)
        
        btn_browse_main = QPushButton("Browse...")
        btn_browse_main.clicked.connect(self.browse_main_dataset)
        main_dataset_layout.addWidget(self.main_dataset_input)
        main_dataset_layout.addWidget(btn_browse_main)
        main_dataset_group.setLayout(main_dataset_layout)
        layout.addWidget(main_dataset_group)
        
        # Step 2: New Dataset
        new_dataset_group = QGroupBox("2. New Dataset (Source to Import)")
        new_dataset_layout = QHBoxLayout()
        self.new_dataset_input = QLineEdit()
        self.new_dataset_input.setReadOnly(True)
        btn_browse_new = QPushButton("Browse...")
        btn_browse_new.clicked.connect(self.browse_new_dataset)
        new_dataset_layout.addWidget(self.new_dataset_input)
        new_dataset_layout.addWidget(btn_browse_new)
        new_dataset_group.setLayout(new_dataset_layout)
        layout.addWidget(new_dataset_group)
        
        # Step 3: Auto Tasks
        options_group = QGroupBox("3. Auto-Tasks (Executed upon loading)")
        options_layout = QVBoxLayout()
        
        self.chk_preflight = QCheckBox("Pre-flight check (remove corrupted/0-byte images)")
        self.chk_preflight.setChecked(True)
        
        self.chk_grayscale = QCheckBox("Remove Grayscale Images (move to removed/grayscale)")
        self.chk_grayscale.setChecked(False)
        
        self.chk_duplicates = QCheckBox("Remove Augmentations / Roboflow Duplicates (move to removed/duplicates)")
        self.chk_duplicates.setChecked(False)
        
        self.chk_hash_duplicates = QCheckBox("Remove Exact Duplicates (Hash-based search)")
        self.chk_hash_duplicates.setChecked(True)
        
        self.chk_standardize_format = QCheckBox("Standardize Image Format (Convert all to .jpg)")
        self.chk_standardize_format.setChecked(False)
        
        self.chk_normalize_res = QCheckBox("Normalize Resolution (Downscale to max 1920x1080)")
        self.chk_normalize_res.setChecked(False)
        
        # Auto import JSON
        auto_import_layout = QHBoxLayout()
        self.chk_auto_import = QCheckBox("Auto-Import Detections & Isolate Unmapped to 'review_label'")
        self.chk_auto_import.setChecked(False)
        self.json_path_input = QLineEdit()
        self.json_path_input.setPlaceholderText("Path to _viat_detections.json")
        self.json_path_input.setEnabled(False)
        btn_browse_json = QPushButton("Browse JSON")
        btn_browse_json.setEnabled(False)
        btn_browse_json.clicked.connect(self.browse_json)
        
        self.chk_auto_import.toggled.connect(self.json_path_input.setEnabled)
        self.chk_auto_import.toggled.connect(btn_browse_json.setEnabled)
        
        auto_import_layout.addWidget(self.chk_auto_import)
        auto_import_layout.addWidget(self.json_path_input)
        auto_import_layout.addWidget(btn_browse_json)
        
        options_layout.addWidget(self.chk_preflight)
        options_layout.addWidget(self.chk_grayscale)
        options_layout.addWidget(self.chk_duplicates)
        options_layout.addWidget(self.chk_hash_duplicates)
        options_layout.addWidget(self.chk_standardize_format)
        options_layout.addWidget(self.chk_normalize_res)
        options_layout.addLayout(auto_import_layout)
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("Start Integration Mode")
        self.btn_run.clicked.connect(self.accept_settings)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)
        
    def browse_main_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Main Dataset Directory")
        if folder:
            self.main_dataset_input.setText(folder)
            from config import DEFAULT_SETTINGS
            DEFAULT_SETTINGS["main_dataset_path"] = folder
            
    def browse_new_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, "Select New Dataset Directory")
        if folder:
            self.new_dataset_input.setText(folder)
            
    def browse_json(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Detections JSON", "", "JSON Files (*.json)")
        if file:
            self.json_path_input.setText(file)

    def accept_settings(self):
        main_ds = self.main_dataset_input.text()
        new_ds = self.new_dataset_input.text()
        
        if not main_ds or not os.path.exists(main_ds):
            QMessageBox.warning(self, "Error", "Please select a valid Main Dataset folder.")
            return
            
        if not new_ds or not os.path.exists(new_ds):
            QMessageBox.warning(self, "Error", "Please select a valid New Dataset folder.")
            return
            
        self.settings = {
            "main_dataset": main_ds,
            "new_dataset": new_ds,
            "preflight": self.chk_preflight.isChecked(),
            "remove_grayscale": self.chk_grayscale.isChecked(),
            "remove_duplicates": self.chk_duplicates.isChecked(),
            "remove_hash_duplicates": self.chk_hash_duplicates.isChecked(),
            "standardize_format": self.chk_standardize_format.isChecked(),
            "normalize_res": self.chk_normalize_res.isChecked(),
            "auto_import": self.chk_auto_import.isChecked(),
            "json_path": self.json_path_input.text() if self.chk_auto_import.isChecked() else None
        }
        self.accept()
