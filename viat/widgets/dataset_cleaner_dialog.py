import os
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QCheckBox, QRadioButton,
                             QSpinBox, QMessageBox, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

def is_contained(boxA, boxB):
    # Returns True if boxA is mostly (> 90%) inside boxB
    rectA = boxA.rect
    rectB = boxB.rect
    
    x_left = max(rectA.x(), rectB.x())
    y_top = max(rectA.y(), rectB.y())
    x_right = min(rectA.right(), rectB.right())
    y_bottom = min(rectA.bottom(), rectB.bottom())
    
    if x_right < x_left or y_bottom < y_top:
        return False
        
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)
    areaA = (rectA.width()) * (rectA.height())
    
    if areaA == 0:
        return True
        
    ioa = intersection_area / areaA
    return ioa >= 0.90

class CleanerThread(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished_processing = pyqtSignal(int, int) # total_frames, total_removed

    def __init__(self, main_window, rewrite, start_frame, end_frame, whole_dataset):
        super().__init__()
        self.main_window = main_window
        self.rewrite = rewrite
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.whole_dataset = whole_dataset

    def run(self):
        # We process the open project's annotations
        frame_annotations = self.main_window.frame_annotations
        
        # Determine frames to process
        if self.whole_dataset:
            frames_to_process = list(frame_annotations.keys())
        else:
            frames_to_process = [f for f in frame_annotations.keys() if self.start_frame <= f <= self.end_frame]
            
        total_frames = len(frames_to_process)
        if total_frames == 0:
            self.finished_processing.emit(0, 0)
            return

        total_removed = 0
        
        for i, frame_idx in enumerate(frames_to_process):
            boxes = frame_annotations.get(frame_idx, [])
            if not boxes:
                self.progress.emit(int((i+1)/total_frames * 100))
                continue
                
            to_remove = set()
            for idxA, boxA in enumerate(boxes):
                for idxB, boxB in enumerate(boxes):
                    if idxA != idxB and boxA.class_name == boxB.class_name: # same class
                        if is_contained(boxA, boxB):
                            # If they are exactly the same, only remove one (the latter one)
                            if is_contained(boxB, boxA) and idxB > idxA:
                                to_remove.add(idxB)
                            elif not is_contained(boxB, boxA):
                                to_remove.add(idxA)

            if to_remove:
                total_removed += len(to_remove)
                self.log.emit(f"Found {len(to_remove)} nested boxes in frame {frame_idx}")
                
                if self.rewrite:
                    kept_boxes = [boxes[j] for j in range(len(boxes)) if j not in to_remove]
                    frame_annotations[frame_idx] = kept_boxes
            
            self.progress.emit(int((i+1)/total_frames * 100))
            
        self.finished_processing.emit(total_frames, total_removed)


class DatasetCleanerDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.setWindowTitle("Dataset Cleaner (Nested Boxes)")
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout(self)
        
        # Introduction
        lbl_intro = QLabel("This tool removes labels that are completely nested inside another label of the same class from the current project.")
        lbl_intro.setWordWrap(True)
        layout.addWidget(lbl_intro)
        
        # Current Project Status
        if hasattr(self.main_window, "is_image_dataset") and self.main_window.is_image_dataset:
            lbl_status_proj = QLabel(f"Project loaded: {len(getattr(self.main_window, 'frame_annotations', {}))} frames with annotations.")
        else:
            lbl_status_proj = QLabel("Warning: No image dataset loaded!")
            lbl_status_proj.setStyleSheet("color: red; font-weight: bold;")
        layout.addWidget(lbl_status_proj)
        
        # Scope Selection
        self.rb_whole = QRadioButton("Whole Dataset")
        self.rb_whole.setChecked(True)
        self.rb_range = QRadioButton("Range of frames (by index)")
        
        self.rb_whole.toggled.connect(self.toggle_range)
        layout.addWidget(self.rb_whole)
        layout.addWidget(self.rb_range)
        
        range_layout = QHBoxLayout()
        range_layout.addWidget(QLabel("Start:"))
        self.spin_start = QSpinBox()
        self.spin_start.setRange(0, 9999999)
        self.spin_start.setEnabled(False)
        range_layout.addWidget(self.spin_start)
        
        range_layout.addWidget(QLabel("End:"))
        self.spin_end = QSpinBox()
        self.spin_end.setRange(0, 9999999)
        self.spin_end.setValue(100)
        self.spin_end.setEnabled(False)
        range_layout.addWidget(self.spin_end)
        layout.addLayout(range_layout)
        
        # Options
        self.chk_rewrite = QCheckBox("Rewrite labels (Apply changes to project and .txt files)")
        self.chk_rewrite.setChecked(True) # Checked by default so it updates project & txts
        layout.addWidget(self.chk_rewrite)
        
        # Progress and Log
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        self.lbl_status = QLabel("Ready.")
        layout.addWidget(self.lbl_status)
        
        # Start button
        self.btn_start = QPushButton("Start Cleaning")
        self.btn_start.clicked.connect(self.start_cleaning)
        layout.addWidget(self.btn_start)
        
        if not hasattr(self.main_window, "is_image_dataset") or not getattr(self.main_window, "is_image_dataset", False):
            self.btn_start.setEnabled(False)
            
        self.thread = None

    def toggle_range(self, checked):
        self.spin_start.setEnabled(not checked)
        self.spin_end.setEnabled(not checked)

    def start_cleaning(self):
        # Update frame annotations first to make sure UI changes are in memory
        if hasattr(self.main_window, "update_frame_annotations"):
            self.main_window.update_frame_annotations()
            
        self.btn_start.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Processing...")
        
        self.thread = CleanerThread(
            self.main_window, 
            self.chk_rewrite.isChecked(),
            self.spin_start.value(),
            self.spin_end.value(),
            self.rb_whole.isChecked()
        )
        self.thread.progress.connect(self.progress_bar.setValue)
        self.thread.log.connect(lambda msg: self.lbl_status.setText(msg))
        self.thread.finished_processing.connect(self.on_finished)
        self.thread.start()

    def on_finished(self, total_frames, total_removed):
        self.btn_start.setEnabled(True)
        
        # VERY IMPORTANT: The current frame on the canvas still has the old boxes!
        # If the user moves to the next frame, the canvas's old boxes will overwrite the cleaned frame_annotations.
        # So we MUST reload the canvas annotations from memory now.
        if hasattr(self.main_window, "load_current_frame_annotations"):
            self.main_window.load_current_frame_annotations()
            
        if hasattr(self.main_window, "canvas") and hasattr(self.main_window.canvas, "update"):
            self.main_window.canvas.update()
            
        if self.chk_rewrite.isChecked():
            # Save the project (updates the .txt files using main_window's method)
            if hasattr(self.main_window, "_viat_dataset_info"):
                from utils.dataset_manager import update_dataset_labels
                self.lbl_status.setText("Writing to .txt files...")
                updated, errors = update_dataset_labels(
                    self.main_window._viat_dataset_info,
                    self.main_window.frame_annotations,
                    self.main_window.image_files,
                    current_classes=list(self.main_window.canvas.class_colors.keys())
                )
                if errors:
                    QMessageBox.warning(self, "Warning", f"Processed {total_frames} frames.\nRemoved {total_removed} nested boxes.\nFailed to update some .txt files.")
                else:
                    QMessageBox.information(self, "Done", f"Processed {total_frames} frames.\nRemoved {total_removed} nested boxes and saved to project & .txt files.")
            else:
                QMessageBox.information(self, "Done", f"Processed {total_frames} frames.\nRemoved {total_removed} nested boxes (Saved in memory, but no dataset info to write .txt).")
        else:
            QMessageBox.information(self, "Done", f"Processed {total_frames} frames.\nFound {total_removed} nested boxes (Dry run, not saved).")
        self.lbl_status.setText("Finished.")
