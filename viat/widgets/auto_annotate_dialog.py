from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QComboBox, QRadioButton, QPushButton, QButtonGroup, QMessageBox
)
from PyQt5.QtCore import Qt
import re

class AutoAnnotateDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto Annotate Dataset")
        self.setMinimumWidth(450)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Classes
        layout.addWidget(QLabel("Classes to detect (comma-separated):"))
        self.classes_input = QLineEdit()
        self.classes_input.setPlaceholderText("e.g. person, car, dog")
        layout.addWidget(self.classes_input)

        # Detection Model
        layout.addWidget(QLabel("Zero-Shot Detection Model:"))
        self.det_model_combo = QComboBox()
        self.det_model_combo.addItems([
            "YOLO-World Small (yolov8s-world.pt)",
            "YOLO-World Large (yolov8l-world.pt)",
            "YOLO-World XLarge (yolov8x-world.pt)",
            "Grounding DINO Tiny (IDEA-Research/grounding-dino-tiny)",
            "Grounding DINO Base (IDEA-Research/grounding-dino-base)",
            "Florence-2 Base (microsoft/Florence-2-base)",
            "Florence-2 Large (microsoft/Florence-2-large)"
        ])
        layout.addWidget(self.det_model_combo)

        # Segmentation Model
        layout.addWidget(QLabel("Segmentation Refiner (Optional - creates polygons):"))
        self.seg_model_combo = QComboBox()
        self.seg_model_combo.addItems([
            "None (Bounding Boxes Only)",
            "SAM2 Fast (sam2_s.pt)",
            "SAM2 Huge (sam2_l.pt)",
            "SAM3 Fast (sam3_s.pt)",
            "SAM3 Huge (sam3_l.pt)"
        ])
        layout.addWidget(self.seg_model_combo)

        # Scope
        layout.addWidget(QLabel("Annotation Scope:"))
        scope_layout = QHBoxLayout()
        self.radio_current = QRadioButton("Current Frame")
        self.radio_all = QRadioButton("All Frames (Entire Video/Dataset)")
        self.radio_current.setChecked(True)
        
        self.scope_group = QButtonGroup()
        self.scope_group.addButton(self.radio_current)
        self.scope_group.addButton(self.radio_all)
        
        scope_layout.addWidget(self.radio_current)
        scope_layout.addWidget(self.radio_all)
        layout.addLayout(scope_layout)

        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_start = QPushButton("Start Auto-Annotation")
        self.btn_start.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_start.clicked.connect(self.accept)
        
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_cancel)
        btn_layout.addWidget(self.btn_start)
        
        layout.addLayout(btn_layout)

    def get_config(self):
        raw_classes = self.classes_input.text()
        classes = [c.strip() for c in raw_classes.split(",") if c.strip()]
        
        det_text = self.det_model_combo.currentText()
        det_match = re.search(r'\((.*?)\)', det_text)
        det_model = det_match.group(1) if det_match else "yolov8s-world.pt"
        
        seg_text = self.seg_model_combo.currentText()
        if "None" in seg_text:
            seg_model = None
        else:
            seg_match = re.search(r'\((.*?)\)', seg_text)
            seg_model = seg_match.group(1) if seg_match else None
            
        scope = "all" if self.radio_all.isChecked() else "current"
        
        return {
            "classes": classes,
            "det_model": det_model,
            "seg_model": seg_model,
            "scope": scope
        }
