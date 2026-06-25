from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QComboBox, QRadioButton, QPushButton, QButtonGroup, QMessageBox, QSpinBox, QWidget,
    QListWidget, QListWidgetItem
)
from PyQt5.QtCore import Qt
import re

class AutoAnnotateDialog(QDialog):
    def __init__(self, current_frame=0, total_frames=1, parent=None):
        super().__init__(parent)
        self.current_frame = current_frame
        self.total_frames = total_frames
        self.setWindowTitle("Auto Annotate Dataset")
        self.setMinimumWidth(500)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        # Classes
        layout.addWidget(QLabel("Classes to detect (comma-separated):"))
        self.classes_input = QLineEdit()
        self.classes_input.setPlaceholderText("e.g. person, car, dog")
        layout.addWidget(self.classes_input)

        # Detection Models
        layout.addWidget(QLabel("Zero-Shot Detection Models (Select one or more):"))
        self.det_model_list = QListWidget()
        self.det_model_list.setMaximumHeight(150)
        models = [
            "Existing Annotations (Hand labeled)",
            "YOLO-World Small (yolov8s-world.pt)",
            "YOLO-World Medium (yolov8m-world.pt)",
            "YOLO-World Large (yolov8l-world.pt)",
            "YOLO-World v2 Small (yolov8s-worldv2.pt)",
            "YOLO-World v2 Medium (yolov8m-worldv2.pt)",
            "YOLO-World v2 Large (yolov8l-worldv2.pt)",
            "YOLO-World v2 XLarge (yolov8x-worldv2.pt)",
            "YOLOv11 World Small (yolo11s-world.pt)",
            "YOLOv11 World Medium (yolo11m-world.pt)",
            "YOLOv11 World Large (yolo11l-world.pt)",
            "YOLOv11 World XLarge (yolo11x-world.pt)",
            "YOLOE 11l (yoloe-11l-seg.pt)",
            "YOLOE 26s (yoloe-26s-seg.pt)",
            "YOLOE 26x (yoloe-26x-seg.pt)",
            "Grounding DINO Tiny (IDEA-Research/grounding-dino-tiny)",
            "Grounding DINO Base (IDEA-Research/grounding-dino-base)",
            "Florence-2 Base (microsoft/Florence-2-base)",
            "Florence-2 Large (microsoft/Florence-2-large)",
            "LocateAnything-3B (nvidia/LocateAnything-3B)",
            "SAM3 Text Prompt (sam3.1_l.pt)"
        ]
        for m in models:
            item = QListWidgetItem(m)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.det_model_list.addItem(item)
        layout.addWidget(self.det_model_list)

        # Threshold for Existing Annotations
        self.threshold_widget = QWidget()
        threshold_layout = QHBoxLayout(self.threshold_widget)
        threshold_layout.setContentsMargins(0, 0, 0, 0)
        self.threshold_label = QLabel("Change Threshold (%):")
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 100)
        self.threshold_spin.setValue(40)
        self.threshold_spin.setToolTip("If the bounding box changes by more than this percentage, a new unverified label is created.")
        threshold_layout.addWidget(self.threshold_label)
        threshold_layout.addWidget(self.threshold_spin)
        threshold_layout.addStretch()
        layout.addWidget(self.threshold_widget)
        
        # Hide threshold by default
        self.threshold_widget.setVisible(False)
        self.det_model_list.itemChanged.connect(self.on_det_model_changed)

        
        # Restore last selected det models
        if hasattr(self.parent(), "last_auto_det_models"):
            last_models = self.parent().last_auto_det_models
            if isinstance(last_models, list):
                for i in range(self.det_model_list.count()):
                    item = self.det_model_list.item(i)
                    for lm in last_models:
                        if lm == "existing_annotations" and "Existing Annotations" in item.text():
                            item.setCheckState(Qt.Checked)
                        elif lm in item.text():
                            item.setCheckState(Qt.Checked)
        elif hasattr(self.parent(), "last_auto_det_model"): # Fallback for old saved state
            for i in range(self.det_model_list.count()):
                item = self.det_model_list.item(i)
                if self.parent().last_auto_det_model and self.parent().last_auto_det_model in item.text():
                    item.setCheckState(Qt.Checked)
                    break

        # Segmentation Model
        layout.addWidget(QLabel("Segmentation Refiner (Optional - creates polygons):"))
        self.seg_model_combo = QComboBox()
        self.seg_model_combo.addItems([
            "None (Bounding Boxes Only)",
            "SAM2 Fast (sam2.1_s.pt)",
            "SAM2 Huge (sam2.1_l.pt)",
            "SAM3 Fast (sam3.1_s.pt)",
            "SAM3 Huge (sam3.1_l.pt)"
        ])
        layout.addWidget(self.seg_model_combo)
        
        # Restore last selected seg model
        if hasattr(self.parent(), "last_auto_seg_model"):
            idx = self.seg_model_combo.findText(self.parent().last_auto_seg_model, Qt.MatchContains)
            if idx >= 0:
                self.seg_model_combo.setCurrentIndex(idx)

        # Save Segmentations Checkbox
        self.chk_save_seg = QCheckBox("Save Segmentations to JSON (Creates large files)")
        self.chk_save_seg.setChecked(False) # Default OFF
        layout.addWidget(self.chk_save_seg)


        # Strategy
        layout.addWidget(QLabel("Annotation Strategy:"))
        strategy_layout = QVBoxLayout()
        self.radio_independent = QRadioButton("Independent Frames (Zero-Shot Only)")
        self.radio_tracking = QRadioButton("Zero-Shot + Video Tracking (SAM2/3)")
        self.radio_independent.setChecked(True)
        
        self.strategy_group = QButtonGroup()
        self.strategy_group.addButton(self.radio_independent)
        self.strategy_group.addButton(self.radio_tracking)
        
        strategy_layout.addWidget(self.radio_independent)
        strategy_layout.addWidget(self.radio_tracking)
        layout.addLayout(strategy_layout)

        # Frame Range
        layout.addWidget(QLabel("Frame Range:"))
        range_layout = QHBoxLayout()
        
        self.start_frame_spin = QSpinBox()
        self.start_frame_spin.setRange(0, max(0, self.total_frames - 1))
        self.start_frame_spin.setValue(self.current_frame)
        
        self.end_frame_spin = QSpinBox()
        self.end_frame_spin.setRange(0, max(0, self.total_frames - 1))
        self.end_frame_spin.setValue(self.total_frames - 1)
        
        range_layout.addWidget(QLabel("Start:"))
        range_layout.addWidget(self.start_frame_spin)
        range_layout.addWidget(QLabel("End:"))
        range_layout.addWidget(self.end_frame_spin)
        layout.addLayout(range_layout)

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

    def on_det_model_changed(self, item):
        is_existing = False
        for i in range(self.det_model_list.count()):
            it = self.det_model_list.item(i)
            if it.checkState() == Qt.Checked and "Existing Annotations" in it.text():
                is_existing = True
                break
        self.threshold_widget.setVisible(is_existing)
        # If refining existing annotations, we must have a segmentation model
        if is_existing and self.seg_model_combo.currentIndex() == 0:
            self.seg_model_combo.setCurrentIndex(1) # Select SAM2 Fast by default

    def get_config(self):
        raw_classes = self.classes_input.text()
        classes = [c.strip() for c in raw_classes.split(",") if c.strip()]
        
        det_models = []
        for i in range(self.det_model_list.count()):
            item = self.det_model_list.item(i)
            if item.checkState() == Qt.Checked:
                det_text = item.text()
                if "Existing Annotations" in det_text:
                    det_models.append("existing_annotations")
                else:
                    det_match = re.search(r'\((.*?)\)', det_text)
                    if det_match:
                        det_models.append(det_match.group(1))
                        
        if not det_models:
            det_models = ["yolov8s-world.pt"]
        
        seg_text = self.seg_model_combo.currentText()
        if "None" in seg_text:
            seg_model = None
        else:
            seg_match = re.search(r'\((.*?)\)', seg_text)
            seg_model = seg_match.group(1) if seg_match else None
            
        strategy = "tracking" if self.radio_tracking.isChecked() else "independent"
        start_frame = self.start_frame_spin.value()
        end_frame = self.end_frame_spin.value()
        
        if end_frame < start_frame:
            end_frame = start_frame
            
        return {
            "classes": classes,
            "det_models": det_models,
            "seg_model": seg_model,
            "save_segmentation": self.chk_save_seg.isChecked(),
            "strategy": strategy,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "threshold": self.threshold_spin.value()
        }
