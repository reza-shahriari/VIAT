import os
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QFileDialog,
    QTableWidget,
    QTableWidgetItem,
    QComboBox,
    QMessageBox,
    QDialogButtonBox,
    QProgressDialog,
    QHeaderView,
)
from PyQt5.QtGui import QColor, QBrush
from PyQt5.QtCore import Qt

from utils.import_masks import scan_mask_colors, bgr_to_hex


class ImportMasksDialog(QDialog):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Segmentation Masks")
        self.resize(600, 500)
        self.main_window = main_window
        self.color_mapping = {}  # Store the final mapping

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Folder selection
        folder_layout = QHBoxLayout()
        folder_label = QLabel("Masks Directory:")
        self.folder_edit = QLineEdit()
        self.folder_button = QPushButton("Browse...")
        self.folder_button.clicked.connect(self.browse_folder)
        folder_layout.addWidget(folder_label)
        folder_layout.addWidget(self.folder_edit)
        folder_layout.addWidget(self.folder_button)
        layout.addLayout(folder_layout)

        # Scan button
        self.scan_button = QPushButton("Scan Colors")
        self.scan_button.clicked.connect(self.scan_colors)
        layout.addWidget(self.scan_button)

        # Table for mapping
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Mask Color", "Target Class Name"])
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        layout.addWidget(self.table)

        # Dialog buttons
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Masks Directory")
        if folder:
            self.folder_edit.setText(folder)

    def scan_colors(self):
        masks_dir = self.folder_edit.text().strip()
        if not masks_dir or not os.path.exists(masks_dir):
            QMessageBox.warning(self, "Error", "Please select a valid masks directory.")
            return

        image_files = getattr(self.main_window, "image_files", [])
        if not image_files:
            QMessageBox.warning(self, "Error", "No image dataset loaded to match masks against.")
            return

        progress = QProgressDialog("Scanning for colors...", "Cancel", 0, len(image_files), self)
        progress.setWindowModality(Qt.WindowModal)

        def update_progress(current, total):
            progress.setValue(current)

        try:
            colors = scan_mask_colors(image_files, masks_dir, progress_callback=update_progress)
        except Exception as e:
            progress.close()
            QMessageBox.warning(self, "Error", f"Failed to scan colors: {e}")
            return

        progress.close()

        if progress.wasCanceled():
            return

        if not colors:
            QMessageBox.information(self, "Result", "No valid colors found in the masks.")
            return

        self.populate_table(colors)

    def populate_table(self, colors):
        self.table.setRowCount(0)
        
        # Get existing classes
        existing_classes = []
        if hasattr(self.main_window, "canvas") and hasattr(self.main_window.canvas, "class_colors"):
            existing_classes = list(self.main_window.canvas.class_colors.keys())

        for row, color_bgr in enumerate(colors):
            self.table.insertRow(row)

            # Color item
            hex_color = bgr_to_hex(color_bgr)
            color_item = QTableWidgetItem(hex_color)
            color_item.setFlags(color_item.flags() & ~Qt.ItemIsEditable)
            
            # Set background color
            rgb_color = QColor(color_bgr[2], color_bgr[1], color_bgr[0])
            color_item.setBackground(QBrush(rgb_color))
            
            # Text color for contrast
            luma = 0.299 * color_bgr[2] + 0.587 * color_bgr[1] + 0.114 * color_bgr[0]
            if luma > 128:
                color_item.setForeground(QBrush(QColor(0, 0, 0)))
            else:
                color_item.setForeground(QBrush(QColor(255, 255, 255)))

            self.table.setItem(row, 0, color_item)

            # Combo box for class selection
            combo = QComboBox()
            combo.addItem("-- Skip --")
            combo.addItems(existing_classes)
            combo.setEditable(True)  # Allow user to type new class name

            # Try to pre-select a matching class if it exists (very basic heuristic)
            for cls in existing_classes:
                if cls.lower() in hex_color.lower() or hex_color.lower() in cls.lower():
                    combo.setCurrentText(cls)
                    break
                    
            self.table.setCellWidget(row, 1, combo)
            
            # Store bgr tuple in data for retrieval later
            color_item.setData(Qt.UserRole, color_bgr)

    def get_mapping(self):
        """Returns the dictionary mapping BGR tuples to class names."""
        mapping = {}
        for row in range(self.table.rowCount()):
            color_item = self.table.item(row, 0)
            combo = self.table.cellWidget(row, 1)
            
            if not color_item or not combo:
                continue
                
            color_bgr = color_item.data(Qt.UserRole)
            class_name = combo.currentText().strip()
            
            if class_name and class_name != "-- Skip --":
                mapping[color_bgr] = class_name
                
        return mapping

    def get_masks_dir(self):
        return self.folder_edit.text().strip()
