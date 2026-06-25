from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QLineEdit, QPushButton, QComboBox, QSpinBox, QGroupBox, QMessageBox, QCheckBox
)
from PyQt5.QtCore import Qt, pyqtSignal

class SAMInteractiveDock(QDockWidget):
    """Dock widget for SAM Interactive Tracking mode."""
    
    preview_requested = pyqtSignal()
    track_requested = pyqtSignal(str, int, int) # strategy, start_frame, end_frame
    clear_requested = pyqtSignal()
    model_changed = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__("SAM Interactive Tracker", parent)
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
            "SAM3.1 Fast (sam3.1_s.pt)"
        ])
        self.cmb_model.currentIndexChanged.connect(self.on_model_changed)
        self.layout.addWidget(self.cmb_model)

        # Scope Selection
        self.layout.addWidget(QLabel("Tracking Scope:"))
        self.cmb_scope = QComboBox()
        self.cmb_scope.addItems(["Current Frame Only", "Whole Video", "Custom Range"])
        self.cmb_scope.currentIndexChanged.connect(self.on_scope_changed)
        self.layout.addWidget(self.cmb_scope)
        
        # Save Segmentations
        self.chk_save_seg = QCheckBox("Save Segmentations to JSON (Creates large files)")
        self.chk_save_seg.setChecked(False)
        self.layout.addWidget(self.chk_save_seg)

        # Custom Range
        self.range_widget = QWidget()
        range_layout = QHBoxLayout(self.range_widget)
        range_layout.setContentsMargins(0, 0, 0, 0)
        self.spin_start = QSpinBox()
        self.spin_start.setMinimum(0)
        self.spin_end = QSpinBox()
        self.spin_end.setMinimum(0)
        range_layout.addWidget(QLabel("Start:"))
        range_layout.addWidget(self.spin_start)
        range_layout.addWidget(QLabel("End:"))
        range_layout.addWidget(self.spin_end)
        self.range_widget.setVisible(False)
        self.layout.addWidget(self.range_widget)

        # Action Buttons
        self.btn_clear = QPushButton("Clear Prompts")
        self.btn_preview = QPushButton("Preview Mask (Current Frame)")
        self.btn_track = QPushButton("Execute Tracking")
        
        self.btn_track.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        
        self.btn_clear.clicked.connect(self.clear_requested.emit)
        self.btn_preview.clicked.connect(self.preview_requested.emit)
        self.btn_track.clicked.connect(self.on_track_clicked)
        
        self.layout.addWidget(self.btn_clear)
        self.layout.addWidget(self.btn_preview)
        self.layout.addWidget(self.btn_track)
        
        self.layout.addStretch()
        self.setWidget(self.widget)

    def on_scope_changed(self, index):
        self.range_widget.setVisible(index == 2)

    def on_model_changed(self, index):
        self.model_changed.emit(self.get_model_type())

    def update_status(self, num_pos, num_neg, has_box):
        self.lbl_points.setText(f"Points: {num_pos} Positive, {num_neg} Negative")
        self.lbl_box.setText("Bounding Box: Set" if has_box else "Bounding Box: Not Set")

    def update_frame_info(self, current_frame, total_frames):
        self.current_frame = current_frame
        self.total_frames = total_frames
        self.spin_start.setMaximum(max(0, total_frames - 1))
        self.spin_end.setMaximum(max(0, total_frames - 1))
        
        if self.cmb_scope.currentIndex() == 2: # If Custom range is active, don't override unless necessary
            pass
        else:
            self.spin_start.setValue(current_frame)
            self.spin_end.setValue(total_frames - 1)

    def on_track_clicked(self):
        scope = self.cmb_scope.currentIndex()
        if scope == 0:
            strategy = "frame"
            start_f = self.current_frame
            end_f = self.current_frame
        elif scope == 1:
            strategy = "video"
            start_f = 0
            end_f = self.total_frames - 1
        else:
            strategy = "range"
            start_f = self.spin_start.value()
            end_f = self.spin_end.value()
            if start_f > end_f:
                QMessageBox.warning(self, "Invalid Range", "Start frame must be <= End frame")
                return
                
        self.track_requested.emit(strategy, start_f, end_f)

    def get_save_segmentation(self):
        return self.chk_save_seg.isChecked()

    def get_text_prompt(self):
        t = self.txt_prompt.text().strip()
        return t if t else None
        
    def get_model_type(self):
        text = self.cmb_model.currentText()
        print(f"[DEBUG LOG] sam_interactive_dock.get_model_type called. text='{text}'")
        import re
        match = re.search(r'\((.*?)\)', text)
        result = match.group(1) if match else "sam2.1_s.pt"
        print(f"[DEBUG LOG] match={match}, result='{result}'")
        return result
