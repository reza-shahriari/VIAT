from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QLineEdit, QPushButton, QComboBox, QSpinBox, QGroupBox, QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal

class SAMInteractiveDock(QDockWidget):
    """Dock widget for SAM Interactive Tracking mode."""
    
    preview_requested = pyqtSignal()
    track_requested = pyqtSignal(str, int, int, str) # strategy, start_frame, end_frame, direction
    clear_requested = pyqtSignal()
    undo_requested = pyqtSignal()
    model_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__("Interactive Tracking Menu", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.total_frames = 1
        self.current_frame = 0
        self.setup_ui()

    def setup_ui(self):
        self.widget = QWidget()
        self.layout = QVBoxLayout(self.widget)
        self.layout.setAlignment(Qt.AlignTop)

        # Status Group
        self.status_group = QGroupBox("Prompts Status")
        status_layout = QVBoxLayout()
        self.lbl_points = QLabel("Points: 0 Positive, 0 Negative")
        self.lbl_box = QLabel("Bounding Box: Not Set")
        status_layout.addWidget(self.lbl_points)
        status_layout.addWidget(self.lbl_box)
        self.status_group.setLayout(status_layout)
        self.layout.addWidget(self.status_group)

        # Text Prompt
        self.layout.addWidget(QLabel("Text Prompt (Optional):"))
        self.txt_prompt = QLineEdit()
        self.txt_prompt.setPlaceholderText("e.g. car, person, dog")
        self.layout.addWidget(self.txt_prompt)

        # Model Selection
        self.layout.addWidget(QLabel("SAM Model:"))
        self.cmb_model = QComboBox()
        self.cmb_model.addItems([
            "SAM2 Fast (sam2.1_s.pt)",
            "SAM2 Huge (sam2.1_l.pt)",
            "SAM3.1 Huge (sam3.1_l.pt)",
            "SAM3.1 Fast (sam3.1_s.pt)",
            "SAM2 TRT C++ (TensorRT)",
            "SAM2 TRT C++ Tiny (TensorRT)"
        ])
        self.cmb_model.currentIndexChanged.connect(self.on_model_changed)
        self.layout.addWidget(self.cmb_model)

        # Tracker Engine
        self.layout.addWidget(QLabel("Tracker Engine:"))
        self.cmb_tracker = QComboBox()
        self.cmb_tracker.addItems([
            "SAM (High Accuracy, Mask + Box)",
            "E.T.Track (Fast, Box Only)",
            "OSTrack (Fast, Box Only)",
            "OSTrack TRT (FP16 Accelerated, Box Only)",
            "OSTrack Native TRT (Engine, Box Only)"
        ])
        self.layout.addWidget(self.cmb_tracker)

        # Zero-Shot Detection Model
        self.layout.addWidget(QLabel("Zero-Shot Detection (Optional):"))
        self.cmb_det_model = QComboBox()
        self.cmb_det_model.addItems([
            "None (SAM Pure Tracking/Detection)",
            "YOLOE 11n (yoloe-11n-seg.pt)",
            "YOLOE 11s (yoloe-11s-seg.pt)",
            "YOLOE 11m (yoloe-11m-seg.pt)",
            "YOLOE 11l (yoloe-11l-seg.pt)",
            "YOLOE 11X (yoloe-11x-seg.pt)",
            "YOLOE 26s (yoloe-26s-seg.pt)",
            "YOLOE 26x (yoloe-26x-seg.pt)"
        ])
        self.cmb_det_model.setToolTip(
            "Use YOLOE for frame-by-frame visual prompting. "
            "Draw a box in the first frame, and YOLOE will detect that object in all subsequent frames."
        )
        self.layout.addWidget(self.cmb_det_model)

        # Scope Selection
        self.layout.addWidget(QLabel("Tracking Scope:"))
        self.cmb_scope = QComboBox()
        self.cmb_scope.addItems(["Current Frame Only", "Whole Video", "Custom Range"])
        self.cmb_scope.currentIndexChanged.connect(self.on_scope_changed)
        self.layout.addWidget(self.cmb_scope)

        # Tracking Direction
        self.lbl_direction = QLabel("Tracking Direction:")
        self.layout.addWidget(self.lbl_direction)
        self.cmb_direction = QComboBox()
        self.cmb_direction.addItems(["Forward", "Backward", "Bi-directional"])
        self.cmb_direction.setToolTip(
            "Forward: Track from current/start frame forward.\n"
            "Backward: Track from current frame backward in time.\n"
            "Bi-directional: Track in both forward and backward directions."
        )
        self.cmb_direction.currentIndexChanged.connect(self._update_execute_button_label)
        self.layout.addWidget(self.cmb_direction)
        self.lbl_direction.setVisible(False)
        self.cmb_direction.setVisible(False)

        # Frame-by-Frame Detection checkbox (only visible for multi-frame scopes)
        self.chk_frame_by_frame = QCheckBox("Frame-by-Frame Detection (no tracking)")
        self.chk_frame_by_frame.setChecked(False)
        self.chk_frame_by_frame.setToolTip(
            "When enabled, SAM runs independently on each frame using the same prompt.\n"
            "The object is detected fresh per frame — no temporal tracking state is used.\n"
            "Useful when objects are unrelated across frames or appear/disappear."
        )
        self.chk_frame_by_frame.toggled.connect(self._update_execute_button_label)
        self.chk_frame_by_frame.setVisible(False)  # hidden for "Current Frame Only"
        self.layout.addWidget(self.chk_frame_by_frame)

        # Save Segmentations
        self.chk_save_seg = QCheckBox("Save Segmentations to JSON (Creates large files)")
        self.chk_save_seg.setChecked(False)
        self.layout.addWidget(self.chk_save_seg)

        # Blur Tracked Objects
        self.chk_blur_tracked = QCheckBox("Automatically Blur Tracked Objects")
        self.chk_blur_tracked.setChecked(False)
        self.layout.addWidget(self.chk_blur_tracked)

        # Custom Range
        self.range_widget = QWidget()
        range_layout = QHBoxLayout(self.range_widget)
        range_layout.setContentsMargins(0, 0, 0, 0)
        self.spin_start = QSpinBox()
        self.spin_start.setMinimum(1)
        self.spin_end = QSpinBox()
        self.spin_end.setMinimum(1)
        range_layout.addWidget(QLabel("Start:"))
        range_layout.addWidget(self.spin_start)
        range_layout.addWidget(QLabel("End:"))
        range_layout.addWidget(self.spin_end)
        self.range_widget.setVisible(False)
        self.layout.addWidget(self.range_widget)

        # Action Buttons
        self.btn_undo = QPushButton("Undo Last Track [Ctrl+Z]")
        self.btn_undo.setToolTip("Undo the last tracking action or annotation modification (Shortcut: Ctrl+Z)")
        self.btn_clear = QPushButton("Clear Prompts [X]")
        self.btn_clear.setToolTip("Clear all prompt points and bounding boxes (Shortcut: X)")
        self.btn_preview = QPushButton("Preview Mask [Z]")
        self.btn_preview.setToolTip("Preview SAM segmentation mask on current frame (Shortcut: Z)")
        self.btn_track = QPushButton("Execute Tracking [C]")
        self.btn_track.setToolTip("Execute SAM tracking / detection (Shortcut: C)")
        
        self.btn_track.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        
        self.btn_undo.clicked.connect(self.undo_requested.emit)
        self.btn_clear.clicked.connect(self.clear_requested.emit)
        self.btn_preview.clicked.connect(self.preview_requested.emit)
        self.btn_track.clicked.connect(self.on_track_clicked)
        
        self.layout.addWidget(self.btn_undo)
        self.layout.addWidget(self.btn_clear)
        self.layout.addWidget(self.btn_preview)
        self.layout.addWidget(self.btn_track)
        
        self._update_execute_button_label()
        
        self.layout.addStretch()
        self.setWidget(self.widget)

    def on_scope_changed(self, index):
        is_multi_frame = index > 0  # Whole Video or Custom Range
        self.range_widget.setVisible(index == 2)
        if index == 2:
            self.spin_start.setValue(self.current_frame + 1)
            self.spin_end.setValue(self.total_frames)
        self.chk_frame_by_frame.setVisible(is_multi_frame)
        if hasattr(self, 'cmb_direction'):
            self.lbl_direction.setVisible(is_multi_frame)
            self.cmb_direction.setVisible(is_multi_frame)
        self._update_execute_button_label()

    def get_direction(self):
        if not hasattr(self, 'cmb_direction'):
            return "forward"
        text = self.cmb_direction.currentText().lower()
        if "backward" in text:
            return "backward"
        if "bi-directional" in text or "bidirectional" in text:
            return "bidirectional"
        return "forward"

    def _update_execute_button_label(self):
        """Update the execute button label based on current mode."""
        scope = self.cmb_scope.currentIndex()
        direction = self.get_direction()
        if scope == 0:
            self.btn_track.setText("Execute (Current Frame) [C]")
        elif self.chk_frame_by_frame.isChecked():
            self.btn_track.setText("Run Frame-by-Frame Detection [C]")
        else:
            if direction == "backward":
                self.btn_track.setText("Execute Backward Tracking [C]")
            elif direction == "bidirectional":
                self.btn_track.setText("Execute Bi-directional Tracking [C]")
            else:
                self.btn_track.setText("Execute Tracking [C]")

    def on_model_changed(self, index):
        self.model_changed.emit(self.get_model_type())

    def update_status(self, num_pos, num_neg, has_box):
        self.lbl_points.setText(f"Points: {num_pos} Positive, {num_neg} Negative")
        self.lbl_box.setText("Bounding Box: Set" if has_box else "Bounding Box: Not Set")

    def update_frame_info(self, current_frame, total_frames):
        self.current_frame = current_frame
        self.total_frames = total_frames
        self.spin_start.setMaximum(max(1, total_frames))
        self.spin_end.setMaximum(max(1, total_frames))
        
        if self.cmb_scope.currentIndex() == 2: # If Custom range is active, don't override unless necessary
            pass
        else:
            self.spin_start.setValue(current_frame + 1)
            self.spin_end.setValue(total_frames)

    def on_track_clicked(self):
        scope = self.cmb_scope.currentIndex()
        direction = self.get_direction()
        use_frame_by_frame = self.chk_frame_by_frame.isChecked() and scope > 0

        if scope == 0:
            strategy = "frame"
            start_f = self.current_frame
            end_f = self.current_frame
        elif scope == 1:
            strategy = "detect" if use_frame_by_frame else "video"
            if direction == "backward":
                start_f = self.current_frame
                end_f = 0
            elif direction == "bidirectional":
                start_f = 0
                end_f = self.total_frames - 1
            else: # forward
                start_f = self.current_frame
                end_f = self.total_frames - 1
        else:
            strategy = "detect" if use_frame_by_frame else "range"
            s_val = self.spin_start.value() - 1
            e_val = self.spin_end.value() - 1
            if s_val > e_val:
                QMessageBox.warning(self, "Invalid Range", "Start frame must be <= End frame")
                return
                
            if direction == "backward":
                start_f = self.current_frame
                end_f = s_val
                if start_f < end_f:
                    QMessageBox.warning(self, "Invalid Backward Range", 
                        f"Current frame ({self.current_frame + 1}) must be >= range start ({s_val + 1}) for backward tracking.")
                    return
            elif direction == "bidirectional":
                start_f = s_val
                end_f = e_val
            else: # forward
                start_f = self.current_frame
                end_f = e_val
                if start_f > end_f:
                    QMessageBox.warning(self, "Invalid Forward Range", 
                        f"Current frame ({self.current_frame + 1}) must be <= range end ({e_val + 1}) for forward tracking.")
                    return

        self.track_requested.emit(strategy, start_f, end_f, direction)

    def get_save_segmentation(self):
        return self.chk_save_seg.isChecked()

    def get_blur_tracked_objects(self):
        return self.chk_blur_tracked.isChecked()

    def get_text_prompt(self):
        return self.txt_prompt.text().strip()

    def get_model_type(self):
        text = self.cmb_model.currentText()
        if "sam2.1_s.pt" in text: return "sam2.1_s.pt"
        if "sam2.1_l.pt" in text: return "sam2.1_l.pt"
        if "sam3.1_l.pt" in text: return "sam3.1_l.pt"
        if "sam3.1_s.pt" in text: return "sam3.1_s.pt"
        if "Tiny" in text and "TensorRT" in text: return "sam2_trt_cpp_tiny"
        if "TensorRT" in text: return "sam2_trt_cpp"
        return "sam2.1_s.pt"

    def get_tracker_engine(self):
        text = self.cmb_tracker.currentText()
        if "E.T.Track" in text:
            return "ettrack"
        if "Native TRT" in text:
            return "ostrack_engine"
        if "OSTrack TRT" in text:
            return "ostrack_trt"
        if "OSTrack" in text:
            return "ostrack"
        return "sam"

    def get_det_model_type(self):
        text = self.cmb_det_model.currentText()
        if "None" in text:
            return None
        # Extract the model file name inside parentheses
        import re
        match = re.search(r'\((.*?)\)', text)
        if match:
            return match.group(1)
        return text
