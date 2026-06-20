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
        
        # Pre-fill with currently opened dataset if available
        if getattr(self.main_window, "is_image_dataset", False):
            if hasattr(self.main_window, "_viat_dataset_info") and self.main_window._viat_dataset_info:
                self.new_dataset_input.setText(self.main_window._viat_dataset_info.root)
            elif hasattr(self.main_window, "image_files") and self.main_window.image_files:
                self.new_dataset_input.setText(os.path.dirname(self.main_window.image_files[0]))
                
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
        auto_import_layout = QVBoxLayout()
        
        json_row = QHBoxLayout()
        self.chk_auto_import = QCheckBox("Auto-Import Detections & Isolate Unmapped to 'review_label'")
        self.chk_auto_import.setChecked(False)
        self.json_path_input = QLineEdit()
        self.json_path_input.setPlaceholderText("Paths to _viat_detections.json (multiple allowed)")
        self.json_path_input.setEnabled(False)
        btn_browse_json = QPushButton("Browse JSON(s)")
        btn_browse_json.setEnabled(False)
        btn_browse_json.clicked.connect(self.browse_json)
        json_row.addWidget(self.chk_auto_import)
        json_row.addWidget(self.json_path_input)
        json_row.addWidget(btn_browse_json)
        
        target_row = QHBoxLayout()
        target_row.setContentsMargins(20, 0, 0, 0)
        self.target_classes_input = QLineEdit()
        self.target_classes_input.setPlaceholderText("Target Classes (comma-separated, e.g. car, person) - leave empty for all")
        self.target_classes_input.setEnabled(False)
        target_row.addWidget(QLabel("Target Classes:"))
        target_row.addWidget(self.target_classes_input)
        
        self.chk_auto_import.toggled.connect(self.json_path_input.setEnabled)
        self.chk_auto_import.toggled.connect(btn_browse_json.setEnabled)
        self.chk_auto_import.toggled.connect(self.target_classes_input.setEnabled)
        
        auto_import_layout.addLayout(json_row)
        auto_import_layout.addLayout(target_row)
        
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
        files, _ = QFileDialog.getOpenFileNames(self, "Select Detections JSON(s)", "", "JSON Files (*.json)")
        if files:
            self.json_path_input.setText(";".join(files))

    def accept_settings(self):
        main_ds = self.main_dataset_input.text()
        new_ds = self.new_dataset_input.text()
        
        if not main_ds or not os.path.exists(main_ds):
            QMessageBox.warning(self, "Error", "Please select a valid Main Dataset folder.")
            return
            
        if not new_ds or not os.path.exists(new_ds):
            QMessageBox.warning(self, "Error", "Please select a valid New Dataset folder.")
            return
            
        json_paths = []
        if self.chk_auto_import.isChecked() and self.json_path_input.text():
            json_paths = [p.strip() for p in self.json_path_input.text().split(';') if p.strip()]
            
        target_classes = []
        if self.chk_auto_import.isChecked() and self.target_classes_input.text():
            target_classes = [c.strip() for c in self.target_classes_input.text().split(',') if c.strip()]

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
            "json_paths": json_paths,
            "target_classes": target_classes
        }
        self.accept()
