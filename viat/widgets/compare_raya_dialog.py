import os
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QDialogButtonBox,
    QMessageBox,
    QLabel,
    QGridLayout
)

class CompareRayaDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Compare Raya Annotations")
        self.resize(600, 200)
        self.base_file = ""
        self.mod_file = ""
        self.out_md = ""
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        grid = QGridLayout()
        
        # Base Raya File
        grid.addWidget(QLabel("Base Raya File:"), 0, 0)
        self.base_edit = QLineEdit()
        self.base_edit.setPlaceholderText("Select base Raya text file...")
        grid.addWidget(self.base_edit, 0, 1)
        self.base_btn = QPushButton("Browse...")
        self.base_btn.clicked.connect(self.browse_base)
        grid.addWidget(self.base_btn, 0, 2)
        
        # Changed Raya File
        grid.addWidget(QLabel("Changed Raya File:"), 1, 0)
        self.mod_edit = QLineEdit()
        self.mod_edit.setPlaceholderText("Select changed/modified Raya text file...")
        self.mod_edit.textChanged.connect(self.on_mod_changed)
        grid.addWidget(self.mod_edit, 1, 1)
        self.mod_btn = QPushButton("Browse...")
        self.mod_btn.clicked.connect(self.browse_mod)
        grid.addWidget(self.mod_btn, 1, 2)
        
        # Output Markdown File
        grid.addWidget(QLabel("Output Report Path:"), 2, 0)
        self.out_edit = QLineEdit()
        self.out_edit.setPlaceholderText("Select output Markdown report file path...")
        grid.addWidget(self.out_edit, 2, 1)
        self.out_btn = QPushButton("Browse...")
        self.out_btn.clicked.connect(self.browse_out)
        grid.addWidget(self.out_btn, 2, 2)
        
        layout.addLayout(grid)
        
        # Dialog buttons
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.validate_and_accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
        
    def browse_base(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Base Raya File", self.base_edit.text(), "Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            self.base_edit.setText(file_path)
            
    def browse_mod(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Changed Raya File", self.mod_edit.text(), "Text Files (*.txt);;All Files (*)"
        )
        if file_path:
            self.mod_edit.setText(file_path)
            
    def browse_out(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Select Output Report File", self.out_edit.text(), "Markdown Files (*.md);;All Files (*)"
        )
        if file_path:
            self.out_edit.setText(file_path)
            
    def on_mod_changed(self, text):
        current_out = self.out_edit.text().strip()
        if text.strip():
            dir_name = os.path.dirname(text)
            default_out = os.path.join(dir_name, 'comparison_report.md')
            if not current_out or current_out.endswith('comparison_report.md') or not os.path.dirname(current_out):
                self.out_edit.setText(default_out)
                
    def validate_and_accept(self):
        self.base_file = self.base_edit.text().strip()
        self.mod_file = self.mod_edit.text().strip()
        self.out_md = self.out_edit.text().strip()
        
        if not self.base_file or not os.path.exists(self.base_file):
            QMessageBox.warning(self, "Validation Error", "Please select a valid Base Raya File.")
            return
            
        if not self.mod_file or not os.path.exists(self.mod_file):
            QMessageBox.warning(self, "Validation Error", "Please select a valid Changed Raya File.")
            return
            
        if not self.out_md:
            QMessageBox.warning(self, "Validation Error", "Please select a valid Output Report Path.")
            return
            
        self.accept()
