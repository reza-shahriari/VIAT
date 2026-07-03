import os
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QPushButton,
                             QLabel, QCheckBox, QRadioButton,
                             QSpinBox, QMessageBox, QProgressBar)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

def get_source_priority(source_str):
    if not source_str: return 0
    s = source_str.lower()
    if s == 'sam_detected': return 5  # Magic wand
    if s == 'sam_tracked': return 4
    if 'interpolated' in s: return 3
    if s == 'manual': return 2
    if s in ['detected', 'zeroshot', 'imported']: return 1
    return 0

def calculate_iou_and_ioa(boxA, boxB):
    rectA = boxA.rect
    rectB = boxB.rect
    
    x_left = max(rectA.x(), rectB.x())
    y_top = max(rectA.y(), rectB.y())
    x_right = min(rectA.right(), rectB.right())
    y_bottom = min(rectA.bottom(), rectB.bottom())
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0, 0.0, 0.0
        
    intersection_area = (x_right - x_left + 1) * (y_bottom - y_top + 1)
    areaA = (rectA.width()) * (rectA.height())
    areaB = (rectB.width()) * (rectB.height())
    
    if areaA == 0 or areaB == 0:
        return 0.0, 0.0, 0.0
        
    union_area = areaA + areaB - intersection_area
    iou = intersection_area / union_area
    ioa_A = intersection_area / areaA  # how much of A is inside B
    ioa_B = intersection_area / areaB  # how much of B is inside A
    
    return iou, ioa_A, ioa_B

class CleanerThread(QThread):
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished_processing = pyqtSignal(int, int) # total_frames, total_removed

    def __init__(self, main_window, rewrite, start_frame, end_frame, whole_dataset, remove_duplicates, remove_nested):
        super().__init__()
        self.main_window = main_window
        self.rewrite = rewrite
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.whole_dataset = whole_dataset
        self.remove_duplicates = remove_duplicates
        self.remove_nested = remove_nested

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
            for idxA in range(len(boxes)):
                for idxB in range(idxA + 1, len(boxes)):
                    boxA = boxes[idxA]
                    boxB = boxes[idxB]
                    if boxA.class_name == boxB.class_name:
                        iou, ioa_A, ioa_B = calculate_iou_and_ioa(boxA, boxB)
                        
                        if iou > 0.95 and self.remove_duplicates:
                            # Nearly identical - check source priority
                            pA = get_source_priority(getattr(boxA, 'source', ''))
                            pB = get_source_priority(getattr(boxB, 'source', ''))
                            
                            if pA < pB:
                                to_remove.add(idxA)
                            elif pB < pA:
                                to_remove.add(idxB)
                            else:
                                # Tie - remove the latter
                                to_remove.add(idxB)
                        elif self.remove_nested and iou <= 0.95:
                            # Not identical, check if one is nested in the other
                            if ioa_A > 0.90:
                                to_remove.add(idxA)
                            elif ioa_B > 0.90:
                                to_remove.add(idxB)

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
        self.setWindowTitle("Dataset Cleaner (Nested & Duplicate Boxes)")
        self.setMinimumWidth(500)
        
        layout = QVBoxLayout(self)
        
        # Introduction
        lbl_intro = QLabel("This tool removes redundant bounding boxes from the current project.")
        lbl_intro.setWordWrap(True)
        layout.addWidget(lbl_intro)
        
        # Current Project Status
        if hasattr(self.main_window, "is_image_dataset") and self.main_window.is_image_dataset:
            lbl_status_proj = QLabel(f"Project loaded: {len(getattr(self.main_window, 'frame_annotations', {}))} frames with annotations.")
        else:
            lbl_status_proj = QLabel("Warning: No image dataset loaded!")
            lbl_status_proj.setStyleSheet("color: red; font-weight: bold;")
        layout.addWidget(lbl_status_proj)
        
        # Cleaning Options
        layout.addWidget(QLabel("Cleaning Options:"))
        self.chk_remove_duplicates = QCheckBox("Remove near-duplicates (IoU > 95%) based on Source Priority")
        self.chk_remove_duplicates.setChecked(True)
        layout.addWidget(self.chk_remove_duplicates)
        
        self.chk_remove_nested = QCheckBox("Remove nested boxes (IoA > 90%)")
        self.chk_remove_nested.setChecked(True)
        layout.addWidget(self.chk_remove_nested)
        
        # Scope Selection
        layout.addWidget(QLabel("Scope:"))
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
        self.chk_update_txt = QCheckBox("Update .txt label files on disk as well")
        self.chk_update_txt.setChecked(True)
        layout.addWidget(self.chk_update_txt)
        
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
            True, # ALWAYS update the project memory now
            self.spin_start.value(),
            self.spin_end.value(),
            self.rb_whole.isChecked(),
            self.chk_remove_duplicates.isChecked(),
            self.chk_remove_nested.isChecked()
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
            
        if self.chk_update_txt.isChecked():
            # Save the project (updates the .txt files using main_window's method)
            if hasattr(self.main_window, "_viat_dataset_info"):
                from viat.utils.dataset_manager import update_dataset_labels
                self.lbl_status.setText("Writing to .txt files...")
                updated, errors = update_dataset_labels(
                    self.main_window._viat_dataset_info,
                    self.main_window.frame_annotations,
                    self.main_window.image_files,
                    current_classes=list(self.main_window.canvas.class_colors.keys())
                )
                if errors:
                    QMessageBox.warning(self, "Warning", f"Processed {total_frames} frames.\nRemoved {total_removed} boxes.\nFailed to update some .txt files.")
                else:
                    QMessageBox.information(self, "Done", f"Processed {total_frames} frames.\nRemoved {total_removed} boxes and saved to project & .txt files.")
            else:
                QMessageBox.information(self, "Done", f"Processed {total_frames} frames.\nRemoved {total_removed} boxes (Saved in memory, but no dataset info to write .txt).")
        else:
            QMessageBox.information(self, "Done", f"Processed {total_frames} frames.\nRemoved {total_removed} boxes (Saved in project memory, .txt files untouched).")
        self.lbl_status.setText("Finished.")
