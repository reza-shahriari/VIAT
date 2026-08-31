import sys
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QDoubleSpinBox, 
    QSpinBox, QCheckBox, QPushButton, QGroupBox, QFormLayout, 
    QDialogButtonBox, QMessageBox
)

class AutoBlurDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto Blur Labels")
        self.setMinimumWidth(400)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Size Settings
        size_group = QGroupBox("Size Filters")
        size_layout = QFormLayout()
        
        self.cb_small = QCheckBox("Blur Small Objects")
        self.spin_small_area = QSpinBox()
        self.spin_small_area.setRange(0, 1000000)
        self.spin_small_area.setValue(1000)
        self.spin_small_area.setSuffix(" pixels^2")
        size_layout.addRow(self.cb_small, self.spin_small_area)

        self.cb_big = QCheckBox("Blur Big Objects")
        self.spin_big_area = QSpinBox()
        self.spin_big_area.setRange(0, 10000000)
        self.spin_big_area.setValue(500000)
        self.spin_big_area.setSuffix(" pixels^2")
        size_layout.addRow(self.cb_big, self.spin_big_area)

        size_group.setLayout(size_layout)
        layout.addWidget(size_group)

        # Aspect Ratio Settings
        ar_group = QGroupBox("Aspect Ratio Filters")
        ar_layout = QFormLayout()
        
        self.cb_ar = QCheckBox("Blur by Aspect Ratio")
        
        ar_hbox = QHBoxLayout()
        self.spin_ar_min = QDoubleSpinBox()
        self.spin_ar_min.setRange(0.0, 100.0)
        self.spin_ar_min.setValue(0.1)
        
        self.spin_ar_max = QDoubleSpinBox()
        self.spin_ar_max.setRange(0.0, 100.0)
        self.spin_ar_max.setValue(10.0)
        
        ar_hbox.addWidget(QLabel("Min:"))
        ar_hbox.addWidget(self.spin_ar_min)
        ar_hbox.addWidget(QLabel("Max:"))
        ar_hbox.addWidget(self.spin_ar_max)
        
        ar_layout.addRow(self.cb_ar, ar_hbox)
        ar_group.setLayout(ar_layout)
        layout.addWidget(ar_group)

        # Position Settings
        pos_group = QGroupBox("Position Filters (Corners)")
        pos_layout = QFormLayout()
        
        self.cb_corner = QCheckBox("Blur Objects near Corners/Edges")
        self.spin_corner_dist = QSpinBox()
        self.spin_corner_dist.setRange(0, 500)
        self.spin_corner_dist.setValue(50)
        self.spin_corner_dist.setSuffix(" px")
        pos_layout.addRow(self.cb_corner, self.spin_corner_dist)
        
        pos_group.setLayout(pos_layout)
        layout.addWidget(pos_group)

        # Attributes Settings
        attr_group = QGroupBox("Attribute & Overlap Settings")
        attr_layout = QVBoxLayout()
        
        self.cb_occluded = QCheckBox("Blur Occluded Objects (if 'occluded' or 'occlusion' attribute is True)")
        self.cb_occluded.setChecked(True)
        attr_layout.addWidget(self.cb_occluded)
        
        self.cb_recursive = QCheckBox("Recursively Blur Overlapping Objects (Union)")
        self.cb_recursive.setChecked(True)
        self.cb_recursive.setToolTip("If Object A is blurred, any Object B overlapping it will also be blurred, recursively.")
        attr_layout.addWidget(self.cb_recursive)
        
        self.cb_remove_bbox = QCheckBox("Remove Bounding Boxes after Blurring")
        self.cb_remove_bbox.setChecked(True)
        attr_layout.addWidget(self.cb_remove_bbox)

        self.cb_all_frames = QCheckBox("Apply to All Frames in Dataset")
        self.cb_all_frames.setChecked(False)
        attr_layout.addWidget(self.cb_all_frames)
        
        attr_group.setLayout(attr_layout)
        layout.addWidget(attr_group)

        # Buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Cancel)
        button_box.button(QDialogButtonBox.Apply).clicked.connect(self.on_apply)
        button_box.button(QDialogButtonBox.Cancel).clicked.connect(self.reject)
        layout.addWidget(button_box)

    def on_apply(self):
        self.settings = {
            "small": {"enabled": self.cb_small.isChecked(), "max_area": self.spin_small_area.value()},
            "big": {"enabled": self.cb_big.isChecked(), "min_area": self.spin_big_area.value()},
            "aspect_ratio": {
                "enabled": self.cb_ar.isChecked(), 
                "min": self.spin_ar_min.value(), 
                "max": self.spin_ar_max.value()
            },
            "corner": {"enabled": self.cb_corner.isChecked(), "dist": self.spin_corner_dist.value()},
            "occluded": self.cb_occluded.isChecked(),
            "recursive": self.cb_recursive.isChecked(),
            "remove_bbox": self.cb_remove_bbox.isChecked(),
            "all_frames": self.cb_all_frames.isChecked()
        }
        self.accept()
