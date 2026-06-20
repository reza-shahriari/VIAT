import os
import cv2
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QCheckBox, QFileDialog, QMessageBox, QGroupBox, QLineEdit, QProgressBar,
    QScrollArea, QWidget, QGridLayout, QApplication
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from utils.single_class_extractor import scan_main_folder, extract_class_samples, execute_extraction

class ScannerThread(QThread):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(dict)
    
    def __init__(self, dataset_path):
        super().__init__()
        self.dataset_path = dataset_path
        
    def run(self):
        try:
            class_samples = extract_class_samples(self.dataset_path, progress_callback=self._progress_cb)
            self.finished.emit(class_samples)
        except Exception as e:
            self.progress.emit(0, 0, f"Error: {str(e)}")
            self.finished.emit({})
            
    def _progress_cb(self, current, total, msg):
        self.progress.emit(current, total, msg)


class ExtractionThread(QThread):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(dict)
    
    def __init__(self, dataset_path, target_folder, target_class_name, selected_classes):
        super().__init__()
        self.dataset_path = dataset_path
        self.target_folder = target_folder
        self.target_class_name = target_class_name
        self.selected_classes = selected_classes
        
    def run(self):
        result = execute_extraction(
            self.dataset_path,
            self.target_folder,
            self.target_class_name,
            self.selected_classes,
            progress_callback=self._progress_cb
        )
        self.finished.emit(result)
        
    def _progress_cb(self, current, total, msg):
        self.progress.emit(current, total, msg)


class SingleClassExtractorDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Extract Single Class (Sequential)")
        self.resize(800, 600)
        self.dataset_paths = []
        self.current_idx = -1
        self.class_samples = {}
        self.checkboxes = {}
        
        self.total_images_copied = 0
        self.total_labels_copied = 0
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # 1. Main Folder Selection
        group_input = QGroupBox("1. Select Main Folder (containing sub-datasets)")
        layout_input = QHBoxLayout()
        self.input_folder_edit = QLineEdit()
        self.input_folder_edit.setReadOnly(True)
        btn_browse_input = QPushButton("Browse...")
        btn_browse_input.clicked.connect(self.browse_input_folder)
        layout_input.addWidget(self.input_folder_edit)
        layout_input.addWidget(btn_browse_input)
        group_input.setLayout(layout_input)
        layout.addWidget(group_input)
        
        # 2. Output Configuration
        group_output = QGroupBox("2. Target Output")
        layout_output = QVBoxLayout()
        
        row_target_folder = QHBoxLayout()
        self.output_folder_edit = QLineEdit()
        self.output_folder_edit.setReadOnly(True)
        btn_browse_output = QPushButton("Browse...")
        btn_browse_output.clicked.connect(self.browse_output_folder)
        row_target_folder.addWidget(QLabel("Output Folder:"))
        row_target_folder.addWidget(self.output_folder_edit)
        row_target_folder.addWidget(btn_browse_output)
        layout_output.addLayout(row_target_folder)
        
        row_target_class = QHBoxLayout()
        self.target_class_edit = QLineEdit()
        self.target_class_edit.setText("target_class")
        row_target_class.addWidget(QLabel("New Single Class Name:"))
        row_target_class.addWidget(self.target_class_edit)
        layout_output.addLayout(row_target_class)
        
        group_output.setLayout(layout_output)
        layout.addWidget(group_output)
        
        # 3. Class Selection Grid
        group_classes = QGroupBox("3. Current Dataset Classes")
        layout_classes = QVBoxLayout()
        
        layout_class_actions = QHBoxLayout()
        btn_sel_all = QPushButton("Select All")
        btn_sel_all.clicked.connect(self.select_all_classes)
        btn_desel_all = QPushButton("Deselect All")
        btn_desel_all.clicked.connect(self.deselect_all_classes)
        layout_class_actions.addWidget(btn_sel_all)
        layout_class_actions.addWidget(btn_desel_all)
        layout_class_actions.addStretch()
        layout_classes.addLayout(layout_class_actions)
        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.scroll_area.setWidget(self.grid_widget)
        
        layout_classes.addWidget(self.scroll_area)
        group_classes.setLayout(layout_classes)
        layout.addWidget(group_classes)
        
        # 4. Progress and Action
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        self.lbl_status = QLabel("Select folders to start.")
        layout.addWidget(self.lbl_status)
        
        self.btn_start = QPushButton("Start Processing")
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start_workflow)
        layout.addWidget(self.btn_start)
        
        row_actions = QHBoxLayout()
        self.btn_skip = QPushButton("Skip Dataset")
        self.btn_skip.setEnabled(False)
        self.btn_skip.clicked.connect(self.skip_dataset)
        row_actions.addWidget(self.btn_skip)
        
        self.btn_process_next = QPushButton("Process && Next")
        self.btn_process_next.setEnabled(False)
        self.btn_process_next.clicked.connect(self.process_and_next)
        row_actions.addWidget(self.btn_process_next)
        
        layout.addLayout(row_actions)
        
    def browse_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Main Folder with Datasets")
        if folder:
            self.input_folder_edit.setText(folder)
            self.check_ready()
            
    def browse_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Dataset Folder")
        if folder:
            self.output_folder_edit.setText(folder)
            self.check_ready()
            
    def check_ready(self):
        has_input = bool(self.input_folder_edit.text())
        has_output = bool(self.output_folder_edit.text())
        self.btn_start.setEnabled(has_input and has_output)
        
    def start_workflow(self):
        main_folder = self.input_folder_edit.text()
        self.dataset_paths = scan_main_folder(main_folder)
        
        if not self.dataset_paths:
            QMessageBox.warning(self, "No Datasets", "No valid subdirectories found.")
            return
            
        self.btn_start.setEnabled(False)
        self.input_folder_edit.setEnabled(False)
        self.output_folder_edit.setEnabled(False)
        self.target_class_edit.setEnabled(False)
        
        self.total_images_copied = 0
        self.total_labels_copied = 0
        
        self.current_idx = 0
        self.load_current_dataset()
        
    def load_current_dataset(self):
        if self.current_idx >= len(self.dataset_paths):
            self.finish_workflow()
            return
            
        d_path = self.dataset_paths[self.current_idx]
        d_name = os.path.basename(d_path)
        
        self.btn_skip.setEnabled(False)
        self.btn_process_next.setEnabled(False)
        self.progress_bar.setValue(0)
        
        self.lbl_status.setText(f"Dataset {self.current_idx + 1} of {len(self.dataset_paths)}: Scanning {d_name}...")
        
        # Clear layout
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.checkboxes.clear()
        
        self.scanner_thread = ScannerThread(d_path)
        self.scanner_thread.progress.connect(self.update_progress)
        self.scanner_thread.finished.connect(self.on_scan_finished)
        self.scanner_thread.start()
        
    def on_scan_finished(self, class_samples):
        self.class_samples = class_samples
        
        if not self.class_samples:
            # Empty dataset or no classes found, skip automatically
            self.current_idx += 1
            self.load_current_dataset()
            return
            
        self.populate_grid()
        
        d_name = os.path.basename(self.dataset_paths[self.current_idx])
        self.lbl_status.setText(f"Dataset {self.current_idx + 1} of {len(self.dataset_paths)} ({d_name}): Found {len(self.class_samples)} classes. Select to merge.")
        self.progress_bar.setValue(self.progress_bar.maximum())
        
        self.btn_skip.setEnabled(True)
        self.btn_process_next.setEnabled(True)
        
    def populate_grid(self):
        cols = 4
        row = 0
        col = 0
        
        for cls_name, data in self.class_samples.items():
            container = QWidget()
            vbox = QVBoxLayout(container)
            vbox.setAlignment(Qt.AlignCenter)
            
            pixmap = self.create_crop_pixmap(data["img_path"], data["box"])
            lbl_img = QLabel()
            lbl_img.setFixedSize(150, 150)
            lbl_img.setAlignment(Qt.AlignCenter)
            if pixmap:
                lbl_img.setPixmap(pixmap.scaled(150, 150, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                lbl_img.setText("No Image")
            
            chk = QCheckBox(cls_name)
            chk.setChecked(True)
            self.checkboxes[cls_name] = chk
            
            vbox.addWidget(lbl_img)
            vbox.addWidget(chk)
            
            self.grid_layout.addWidget(container, row, col)
            
            col += 1
            if col >= cols:
                col = 0
                row += 1
                
    def select_all_classes(self):
        for chk in self.checkboxes.values():
            chk.setChecked(True)
            
    def deselect_all_classes(self):
        for chk in self.checkboxes.values():
            chk.setChecked(False)
                
    def create_crop_pixmap(self, img_path, box):
        try:
            import numpy as np
            import cv2
            
            # Safe read for Windows unicode paths
            img_array = np.fromfile(img_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                return None
            
            bx = int(box.get("x", 0))
            by = int(box.get("y", 0))
            bw = int(box.get("w", 0))
            bh = int(box.get("h", 0))
            
            h, w = img.shape[:2]
            
            # If width/height are 0, they might be missing or invalid. Use full image as fallback.
            if bw <= 0 or bh <= 0:
                bx, by, bw, bh = 0, 0, w, h
                
            bx = max(0, bx)
            by = max(0, by)
            bw = min(w - bx, bw)
            bh = min(h - by, bh)
            
            crop = img[by:by+bh, bx:bx+bw]
            if crop.size == 0:
                return None
                
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = np.ascontiguousarray(crop)
            
            ch, cw, channels = crop.shape
            bytes_per_line = channels * cw
            
            # Using crop.data directly can sometimes cause memory corruption in PyQt
            # passing it via memoryview or keeping reference helps
            qimg = QImage(crop.data, cw, ch, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            
            # Detach to ensure safe memory
            pixmap.detach()
            return pixmap
        except Exception as e:
            print(f"Error creating crop pixmap for {img_path}: {e}")
            return None
            
    def skip_dataset(self):
        self.current_idx += 1
        self.load_current_dataset()
        
    def process_and_next(self):
        selected_classes = [name for name, chk in self.checkboxes.items() if chk.isChecked()]
        if not selected_classes:
            # Same as skip
            self.skip_dataset()
            return
            
        target_folder = self.output_folder_edit.text()
        target_class_name = self.target_class_edit.text().strip()
        
        self.btn_skip.setEnabled(False)
        self.btn_process_next.setEnabled(False)
        self.progress_bar.setValue(0)
        
        d_path = self.dataset_paths[self.current_idx]
        d_name = os.path.basename(d_path)
        self.lbl_status.setText(f"Dataset {self.current_idx + 1} of {len(self.dataset_paths)}: Extracting from {d_name}...")
        
        self.extraction_thread = ExtractionThread(
            d_path,
            target_folder,
            target_class_name,
            selected_classes
        )
        self.extraction_thread.progress.connect(self.update_progress)
        self.extraction_thread.finished.connect(self.on_extraction_finished)
        self.extraction_thread.start()
        
    def update_progress(self, current, total, msg):
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        
        # Don't overwrite the main status label with file names if we can avoid it, 
        # or append it if needed. Let's just update the status label if needed.
        if "Scanning" in msg or "Error" in msg:
            self.lbl_status.setText(msg)
            
    def on_extraction_finished(self, result):
        if "error" not in result:
            self.total_images_copied += result.get("images_copied", 0)
            self.total_labels_copied += result.get("labels_copied", 0)
            
        self.current_idx += 1
        self.load_current_dataset()
        
    def finish_workflow(self):
        self.progress_bar.setValue(self.progress_bar.maximum())
        msg = f"All datasets processed!\nTotal Images Copied: {self.total_images_copied}\nTotal Labels Copied: {self.total_labels_copied}"
        QMessageBox.information(self, "Finished", msg)
        self.lbl_status.setText("All done.")
        
        # Reset UI
        self.btn_start.setEnabled(True)
        self.input_folder_edit.setEnabled(True)
        self.output_folder_edit.setEnabled(True)
        self.target_class_edit.setEnabled(True)
        self.btn_skip.setEnabled(False)
        self.btn_process_next.setEnabled(False)
