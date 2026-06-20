import os
import cv2
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QMessageBox, QGroupBox, QLineEdit, QProgressBar,
    QScrollArea, QWidget, QGridLayout, QApplication
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, pyqtSignal

# Reuse the ScannerThread and utility functions from the single class extractor
from .single_class_extractor_dialog import ScannerThread
from utils.single_class_extractor import scan_main_folder


class BatchPredictionDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch Prediction Queue Builder")
        self.resize(800, 600)
        
        self.dataset_paths = []
        self.current_idx = -1
        self.class_samples = {}
        
        self.queued_count = 0
        self.queue_file_path = ""
        
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
        group_output = QGroupBox("2. Target Output Queue File")
        layout_output = QHBoxLayout()
        
        self.output_file_edit = QLineEdit()
        self.output_file_edit.setReadOnly(True)
        btn_browse_output = QPushButton("Browse...")
        btn_browse_output.clicked.connect(self.browse_output_file)
        layout_output.addWidget(QLabel("Queue File (.txt):"))
        layout_output.addWidget(self.output_file_edit)
        layout_output.addWidget(btn_browse_output)
        
        group_output.setLayout(layout_output)
        layout.addWidget(group_output)
        
        # 3. Class Selection Grid (Preview only, no checkboxes needed)
        group_classes = QGroupBox("3. Dataset Classes Preview")
        layout_classes = QVBoxLayout()
        
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
        
        self.btn_start = QPushButton("Start Reviewing")
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start_workflow)
        layout.addWidget(self.btn_start)
        
        row_actions = QHBoxLayout()
        self.btn_skip = QPushButton("Skip Dataset")
        self.btn_skip.setEnabled(False)
        self.btn_skip.clicked.connect(self.skip_dataset)
        row_actions.addWidget(self.btn_skip)
        
        self.btn_queue = QPushButton("Queue for Prediction && Next")
        self.btn_queue.setEnabled(False)
        self.btn_queue.clicked.connect(self.queue_and_next)
        row_actions.addWidget(self.btn_queue)
        
        layout.addLayout(row_actions)
        
    def browse_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Main Folder with Datasets")
        if folder:
            self.input_folder_edit.setText(folder)
            self.check_ready()
            
    def browse_output_file(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Prediction Queue", "", "Text Files (*.txt)")
        if file_path:
            self.output_file_edit.setText(file_path)
            self.check_ready()
            
    def check_ready(self):
        has_input = bool(self.input_folder_edit.text())
        has_output = bool(self.output_file_edit.text())
        self.btn_start.setEnabled(has_input and has_output)
        
    def start_workflow(self):
        main_folder = self.input_folder_edit.text()
        self.queue_file_path = self.output_file_edit.text()
        
        self.dataset_paths = scan_main_folder(main_folder)
        
        if not self.dataset_paths:
            QMessageBox.warning(self, "No Datasets", "No valid subdirectories found.")
            return
            
        # Create or clear the output queue file
        try:
            with open(self.queue_file_path, "w", encoding="utf-8") as f:
                f.write("")
        except Exception as e:
            QMessageBox.critical(self, "File Error", f"Cannot write to target queue file:\n{e}")
            return
            
        self.btn_start.setEnabled(False)
        self.input_folder_edit.setEnabled(False)
        self.output_file_edit.setEnabled(False)
        
        self.queued_count = 0
        self.current_idx = 0
        self.load_current_dataset()
        
    def load_current_dataset(self):
        if self.current_idx >= len(self.dataset_paths):
            self.finish_workflow()
            return
            
        d_path = self.dataset_paths[self.current_idx]
        d_name = os.path.basename(d_path)
        
        self.btn_skip.setEnabled(False)
        self.btn_queue.setEnabled(False)
        self.progress_bar.setValue(0)
        
        self.lbl_status.setText(f"Dataset {self.current_idx + 1} of {len(self.dataset_paths)}: Scanning {d_name}...")
        
        # Clear layout
        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
                
        self.scanner_thread = ScannerThread(d_path)
        self.scanner_thread.progress.connect(self.update_progress)
        self.scanner_thread.finished.connect(self.on_scan_finished)
        self.scanner_thread.start()
        
    def on_scan_finished(self, class_samples):
        self.class_samples = class_samples
        
        self.populate_grid()
        
        d_name = os.path.basename(self.dataset_paths[self.current_idx])
        self.lbl_status.setText(f"Dataset {self.current_idx + 1} of {len(self.dataset_paths)} ({d_name}): Found {len(self.class_samples)} classes.")
        self.progress_bar.setValue(self.progress_bar.maximum())
        
        self.btn_skip.setEnabled(True)
        self.btn_queue.setEnabled(True)
        
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
            
            lbl_name = QLabel(cls_name)
            lbl_name.setAlignment(Qt.AlignCenter)
            
            vbox.addWidget(lbl_img)
            vbox.addWidget(lbl_name)
            
            self.grid_layout.addWidget(container, row, col)
            
            col += 1
            if col >= cols:
                col = 0
                row += 1
                
    def create_crop_pixmap(self, img_path, box):
        try:
            import numpy as np
            import cv2
            
            img_array = np.fromfile(img_path, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is None:
                return None
            
            bx = int(box.get("x", 0))
            by = int(box.get("y", 0))
            bw = int(box.get("w", 0))
            bh = int(box.get("h", 0))
            
            h, w = img.shape[:2]
            
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
            
            qimg = QImage(crop.data, cw, ch, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)
            pixmap.detach()
            return pixmap
        except Exception as e:
            print(f"Error creating crop pixmap for {img_path}: {e}")
            return None
            
    def skip_dataset(self):
        self.current_idx += 1
        self.load_current_dataset()
        
    def queue_and_next(self):
        d_path = self.dataset_paths[self.current_idx]
        
        try:
            with open(self.queue_file_path, "a", encoding="utf-8") as f:
                f.write(f"{d_path}\n")
            self.queued_count += 1
        except Exception as e:
            QMessageBox.warning(self, "Write Error", f"Failed to append to queue file:\n{e}")
            
        self.skip_dataset()
        
    def update_progress(self, current, total, msg):
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(current)
        else:
            self.progress_bar.setMaximum(0)
            self.progress_bar.setValue(0)
            
        if "Scanning" in msg or "Error" in msg:
            self.lbl_status.setText(msg)
            
    def finish_workflow(self):
        self.progress_bar.setValue(self.progress_bar.maximum())
        msg = f"All datasets reviewed!\nTotal Queued: {self.queued_count}"
        QMessageBox.information(self, "Finished", msg)
        self.lbl_status.setText("All done.")
        
        # Reset UI
        self.btn_start.setEnabled(True)
        self.input_folder_edit.setEnabled(True)
        self.output_file_edit.setEnabled(True)
        self.btn_skip.setEnabled(False)
        self.btn_queue.setEnabled(False)
