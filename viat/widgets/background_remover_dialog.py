import os
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QMessageBox, QGroupBox, QLineEdit, QProgressBar,
    QSpinBox, QSlider, QRadioButton, QButtonGroup
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from utils.background_remover import execute_background_removal

class BackgroundRemoverThread(QThread):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(dict)
    
    def __init__(self, dataset_path, percentage, action_type):
        super().__init__()
        self.dataset_path = dataset_path
        self.percentage = percentage
        self.action_type = action_type
        
    def run(self):
        try:
            result = execute_background_removal(
                self.dataset_path, 
                self.percentage, 
                self.action_type, 
                progress_callback=self._progress_cb
            )
            self.finished.emit(result)
        except Exception as e:
            self.progress.emit(0, 0, f"Error: {str(e)}")
            self.finished.emit({'error': str(e)})
            
    def _progress_cb(self, current, total, msg):
        self.progress.emit(current, total, msg)


class BackgroundRemoverDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Remove Background Images")
        self.resize(500, 350)
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # 1. Dataset Folder Selection
        group_input = QGroupBox("1. Select Dataset Folder")
        layout_input = QHBoxLayout()
        self.input_folder_edit = QLineEdit()
        self.input_folder_edit.setReadOnly(True)
        btn_browse_input = QPushButton("Browse...")
        btn_browse_input.clicked.connect(self.browse_input_folder)
        layout_input.addWidget(self.input_folder_edit)
        layout_input.addWidget(btn_browse_input)
        group_input.setLayout(layout_input)
        layout.addWidget(group_input)
        
        # 2. Percentage Selection
        group_percentage = QGroupBox("2. Percentage to Remove")
        layout_percentage = QVBoxLayout()
        
        row_percentage = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(50)
        
        self.spinbox = QSpinBox()
        self.spinbox.setRange(0, 100)
        self.spinbox.setValue(50)
        self.spinbox.setSuffix("%")
        
        self.slider.valueChanged.connect(self.spinbox.setValue)
        self.spinbox.valueChanged.connect(self.slider.setValue)
        
        row_percentage.addWidget(self.slider)
        row_percentage.addWidget(self.spinbox)
        layout_percentage.addLayout(row_percentage)
        
        layout_percentage.addWidget(QLabel("Select the percentage of background images to remove."))
        group_percentage.setLayout(layout_percentage)
        layout.addWidget(group_percentage)
        
        # 3. Action Selection
        group_action = QGroupBox("3. Action")
        layout_action = QVBoxLayout()
        
        self.radio_move = QRadioButton("Move to 'removed_backgrounds' folder (Recommended)")
        self.radio_move.setChecked(True)
        self.radio_delete = QRadioButton("Permanently Delete (os.remove)")
        
        self.action_group = QButtonGroup(self)
        self.action_group.addButton(self.radio_move)
        self.action_group.addButton(self.radio_delete)
        
        layout_action.addWidget(self.radio_move)
        layout_action.addWidget(self.radio_delete)
        group_action.setLayout(layout_action)
        layout.addWidget(group_action)
        
        # 4. Progress and Action
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        self.lbl_status = QLabel("Select a dataset folder to start.")
        layout.addWidget(self.lbl_status)
        
        self.btn_start = QPushButton("Start Processing")
        self.btn_start.setEnabled(False)
        self.btn_start.clicked.connect(self.start_workflow)
        layout.addWidget(self.btn_start)
        
    def browse_input_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Dataset Folder")
        if folder:
            self.input_folder_edit.setText(folder)
            self.check_ready()
            
    def check_ready(self):
        has_input = bool(self.input_folder_edit.text())
        self.btn_start.setEnabled(has_input)
        
    def start_workflow(self):
        dataset_path = self.input_folder_edit.text()
        percentage = self.spinbox.value()
        action_type = "move" if self.radio_move.isChecked() else "remove"
        
        if percentage == 0:
            QMessageBox.information(self, "Finished", "0% selected, nothing to do.")
            return
            
        self.btn_start.setEnabled(False)
        self.input_folder_edit.setEnabled(False)
        self.slider.setEnabled(False)
        self.spinbox.setEnabled(False)
        self.radio_move.setEnabled(False)
        self.radio_delete.setEnabled(False)
        
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Starting...")
        
        self.thread = BackgroundRemoverThread(dataset_path, percentage, action_type)
        self.thread.progress.connect(self.update_progress)
        self.thread.finished.connect(self.on_finished)
        self.thread.start()
        
    def update_progress(self, current, total, msg):
        self.lbl_status.setText(msg)
        if total > 0:
            val = int((current / total) * 100)
            self.progress_bar.setValue(min(val, 100))
        else:
            self.progress_bar.setValue(0)
            
    def on_finished(self, result):
        self.btn_start.setEnabled(True)
        self.input_folder_edit.setEnabled(True)
        self.slider.setEnabled(True)
        self.spinbox.setEnabled(True)
        self.radio_move.setEnabled(True)
        self.radio_delete.setEnabled(True)
        self.progress_bar.setValue(100)
        
        if 'error' in result:
            QMessageBox.critical(self, "Error", f"An error occurred: {result['error']}")
            self.lbl_status.setText("Error occurred.")
            return
            
        action_str = "Moved" if result['action'] == "move" else "Deleted"
        msg = (
            f"Process Complete:\n"
            f"Total images in dataset: {result['total_images']}\n"
            f"Total background images found: {result['total_backgrounds']}\n"
            f"{action_str} {result['processed']} background images."
        )
        QMessageBox.information(self, "Process Complete", msg)
        self.lbl_status.setText(f"Completed: {result['processed']} backgrounds processed.")
