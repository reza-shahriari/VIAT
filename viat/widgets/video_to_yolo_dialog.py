"""
Interactive Dialog for Converting Video Datasets into YOLO Image Datasets.
"""

import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QSpinBox, QDoubleSpinBox, QCheckBox,
    QComboBox, QTabWidget, QWidget, QProgressBar, QMessageBox, QGroupBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from viat.converters.video_to_yolo import convert_video_dataset_to_yolo


class VideoToYoloWorker(QThread):
    """Background thread running the conversion generator."""
    progress_updated = pyqtSignal(int, str)
    finished_success = pyqtSignal(str)
    finished_error = pyqtSignal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            generator = convert_video_dataset_to_yolo(
                source_dir=Path(self.params["source_dir"]),
                output_dir=Path(self.params["output_dir"]),
                yaml_path=Path(self.params["yaml_path"]) if self.params.get("yaml_path") else None,
                dist=self.params.get("dist", 1),
                img_ext=self.params.get("img_ext", ".jpg"),
                remove_padding=self.params.get("remove_padding", False),
                black_thresh=self.params.get("black_thresh", 16),
                bg_remove_percent=self.params.get("bg_remove_percent", 0.0),
                enable_smart_crop=self.params.get("enable_smart_crop", False),
                crop_size=self.params.get("crop_size", (640, 640)),
                max_crops_per_frame=self.params.get("max_crops_per_frame", 3),
                min_visibility=self.params.get("min_visibility", 0.4),
                context_padding=self.params.get("context_padding", 0.2),
                max_instances_per_class=self.params.get("max_instances_per_class"),
                flip_augment_percent=self.params.get("flip_augment_percent", 0.0),
                min_box_size_px=self.params.get("min_box_size_px", 2.0),
                split_mode=self.params.get("split_mode", "single"),
                split_ratios=self.params.get("split_ratios", (0.8, 0.2, 0.0)),
                cancel_callback=lambda: self._is_cancelled,
            )

            last_msg = ""
            for pct, msg in generator:
                if self._is_cancelled:
                    self.finished_error.emit("Conversion cancelled by user.")
                    return
                last_msg = msg
                self.progress_updated.emit(pct, msg)

            self.finished_success.emit(last_msg)
        except Exception as e:
            self.finished_error.emit(str(e))


class VideoToYoloDialog(QDialog):
    """Configuration & execution dialog for Video Dataset -> YOLO conversion."""

    def __init__(self, parent=None, default_source_dir=""):
        super().__init__(parent)
        self.setWindowTitle("Convert Video Dataset to YOLO Dataset")
        self.setMinimumWidth(680)
        self.worker = None

        self._init_ui(default_source_dir)

    def _init_ui(self, default_source_dir):
        main_layout = QVBoxLayout(self)

        # Tab Widget for organized settings
        tabs = QTabWidget()

        # Tab 1: Paths & General
        tab_general = QWidget()
        gen_layout = QVBoxLayout(tab_general)

        # Path selectors
        paths_group = QGroupBox("Directories & Files")
        paths_grid = QGridLayout(paths_group)

        paths_grid.addWidget(QLabel("Source Videos Folder:"), 0, 0)
        self.edit_source = QLineEdit(default_source_dir)
        btn_browse_src = QPushButton("Browse...")
        btn_browse_src.clicked.connect(self._browse_source)
        paths_grid.addWidget(self.edit_source, 0, 1)
        paths_grid.addWidget(btn_browse_src, 0, 2)

        paths_grid.addWidget(QLabel("Output YOLO Folder:"), 1, 0)
        self.edit_output = QLineEdit(os.path.join(default_source_dir, "yolo_dataset") if default_source_dir else "")
        btn_browse_out = QPushButton("Browse...")
        btn_browse_out.clicked.connect(self._browse_output)
        paths_grid.addWidget(self.edit_output, 1, 1)
        paths_grid.addWidget(btn_browse_out, 1, 2)

        paths_grid.addWidget(QLabel("Optional YOLO data.yaml:"), 2, 0)
        self.edit_yaml = QLineEdit("")
        btn_browse_yaml = QPushButton("Browse...")
        btn_browse_yaml.clicked.connect(self._browse_yaml)
        paths_grid.addWidget(self.edit_yaml, 2, 1)
        paths_grid.addWidget(btn_browse_yaml, 2, 2)
        gen_layout.addWidget(paths_group)

        # Extraction parameters
        extract_group = QGroupBox("Extraction & Layout")
        extract_grid = QGridLayout(extract_group)

        extract_grid.addWidget(QLabel("Sampling Stride (DIST):"), 0, 0)
        self.spin_dist = QSpinBox()
        self.spin_dist.setRange(1, 1000)
        self.spin_dist.setValue(1)
        self.spin_dist.setToolTip("Extract every Nth frame (1 = all frames, 2 = every 2nd frame...)")
        extract_grid.addWidget(self.spin_dist, 0, 1)

        extract_grid.addWidget(QLabel("Image Extension:"), 0, 2)
        self.cmb_ext = QComboBox()
        self.cmb_ext.addItems([".jpg", ".png", ".webp"])
        extract_grid.addWidget(self.cmb_ext, 0, 3)

        extract_grid.addWidget(QLabel("Export Layout:"), 1, 0)
        self.cmb_split = QComboBox()
        self.cmb_split.addItem("Single Folder (images/ + labels/)", "single")
        self.cmb_split.addItem("Preserve Subfolder Hierarchy", "preserve")
        self.cmb_split.addItem("Auto Train/Val Split (80% / 20%)", "auto")
        extract_grid.addWidget(self.cmb_split, 1, 1, 1, 3)

        self.chk_padding = QCheckBox("Remove Letterbox / Pillarbox Black Padding")
        self.chk_padding.setChecked(False)
        extract_grid.addWidget(self.chk_padding, 2, 0, 1, 2)

        extract_grid.addWidget(QLabel("Black Threshold:"), 2, 2)
        self.spin_black_thresh = QSpinBox()
        self.spin_black_thresh.setRange(0, 255)
        self.spin_black_thresh.setValue(16)
        extract_grid.addWidget(self.spin_black_thresh, 2, 3)

        gen_layout.addWidget(extract_group)
        gen_layout.addStretch()
        tabs.addTab(tab_general, "General")

        # Tab 2: Smart Cropping (ROI Sub-Images)
        tab_crop = QWidget()
        crop_layout = QVBoxLayout(tab_crop)
        crop_group = QGroupBox("Smart Object-Focused Multi-Cropping")
        crop_grid = QGridLayout(crop_group)

        self.chk_smart_crop = QCheckBox("Enable Smart Multi-Cropping")
        self.chk_smart_crop.setToolTip("Crops high-res sub-images around distant object clusters, multiplying training data without re-labeling")
        self.chk_smart_crop.setChecked(False)
        crop_grid.addWidget(self.chk_smart_crop, 0, 0, 1, 4)

        crop_grid.addWidget(QLabel("Crop Width:"), 1, 0)
        self.spin_crop_w = QSpinBox()
        self.spin_crop_w.setRange(128, 7680)
        self.spin_crop_w.setValue(640)
        crop_grid.addWidget(self.spin_crop_w, 1, 1)

        crop_grid.addWidget(QLabel("Crop Height:"), 1, 2)
        self.spin_crop_h = QSpinBox()
        self.spin_crop_h.setRange(128, 4320)
        self.spin_crop_h.setValue(640)
        crop_grid.addWidget(self.spin_crop_h, 1, 3)

        crop_grid.addWidget(QLabel("Max Crops per Frame:"), 2, 0)
        self.spin_max_crops = QSpinBox()
        self.spin_max_crops.setRange(1, 20)
        self.spin_max_crops.setValue(3)
        crop_grid.addWidget(self.spin_max_crops, 2, 1)

        crop_grid.addWidget(QLabel("Min Visibility Ratio:"), 2, 2)
        self.spin_min_vis = QDoubleSpinBox()
        self.spin_min_vis.setRange(0.05, 1.0)
        self.spin_min_vis.setSingleStep(0.05)
        self.spin_min_vis.setValue(0.40)
        self.spin_min_vis.setToolTip("Keep bounding boxes that have at least this fraction inside the crop")
        crop_grid.addWidget(self.spin_min_vis, 2, 3)

        crop_layout.addWidget(crop_group)
        crop_layout.addStretch()
        tabs.addTab(tab_crop, "Smart Cropping (ROI)")

        # Tab 3: Class Balancing & Filtering
        tab_balance = QWidget()
        balance_layout = QVBoxLayout(tab_balance)
        balance_group = QGroupBox("O(N) Uniform Class Balancing & Thinning")
        balance_grid = QGridLayout(balance_group)

        balance_grid.addWidget(QLabel("Max Instances per Class (e.g. car=25000, bus=20000):"), 0, 0, 1, 2)
        self.edit_class_caps = QLineEdit("")
        self.edit_class_caps.setPlaceholderText("e.g. car=25000, person=20000 (leave blank for no capping)")
        self.edit_class_caps.setToolTip("Samples dominant classes evenly across all videos from beginning to end using O(N) uniform stride")
        balance_grid.addWidget(self.edit_class_caps, 1, 0, 1, 2)

        balance_grid.addWidget(QLabel("Background Frame Drop %:"), 2, 0)
        self.spin_bg_drop = QDoubleSpinBox()
        self.spin_bg_drop.setRange(0, 100)
        self.spin_bg_drop.setValue(0)
        self.spin_bg_drop.setToolTip("Randomly drop empty background frames ([]) to prevent dataset imbalance")
        balance_grid.addWidget(self.spin_bg_drop, 2, 1)

        balance_layout.addWidget(balance_group)
        balance_layout.addStretch()
        tabs.addTab(tab_balance, "Class Balancing")

        # Tab 4: Augmentation & Filters
        tab_aug = QWidget()
        aug_layout = QVBoxLayout(tab_aug)
        aug_group = QGroupBox("Pre-Augmentation & Quality Filters")
        aug_grid = QGridLayout(aug_group)

        aug_grid.addWidget(QLabel("Horizontal Flip %:"), 0, 0)
        self.spin_flip = QDoubleSpinBox()
        self.spin_flip.setRange(0, 100)
        self.spin_flip.setValue(0)
        self.spin_flip.setToolTip("Chance to add horizontally mirrored copies with inverted YOLO coords")
        aug_grid.addWidget(self.spin_flip, 0, 1)

        aug_grid.addWidget(QLabel("Min Box Size (px):"), 1, 0)
        self.spin_min_box = QDoubleSpinBox()
        self.spin_min_box.setRange(0.5, 100.0)
        self.spin_min_box.setValue(2.0)
        aug_grid.addWidget(self.spin_min_box, 1, 1)

        aug_layout.addWidget(aug_group)
        aug_layout.addStretch()
        tabs.addTab(tab_aug, "Augmentation")

        main_layout.addWidget(tabs)

        # Progress Section
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        main_layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("Ready.")
        main_layout.addWidget(self.lbl_status)

        # Buttons
        btn_box = QHBoxLayout()
        self.btn_start = QPushButton("🚀 Start Conversion")
        self.btn_start.clicked.connect(self._start_conversion)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_conversion)
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.reject)

        btn_box.addStretch()
        btn_box.addWidget(self.btn_start)
        btn_box.addWidget(self.btn_cancel)
        btn_box.addWidget(self.btn_close)
        main_layout.addLayout(btn_box)

    def _browse_source(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Video Dataset Folder", self.edit_source.text())
        if folder:
            self.edit_source.setText(folder)
            if not self.edit_output.text():
                self.edit_output.setText(os.path.join(folder, "yolo_dataset"))

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output YOLO Folder", self.edit_output.text())
        if folder:
            self.edit_output.setText(folder)

    def _browse_yaml(self):
        yaml_file, _ = QFileDialog.getOpenFileName(self, "Select Existing data.yaml", "", "YAML Files (*.yaml *.yml)")
        if yaml_file:
            self.edit_yaml.setText(yaml_file)

    def _parse_class_caps(self) -> dict:
        caps = {}
        raw = self.edit_class_caps.text().strip()
        if not raw:
            return caps
        for part in raw.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                try:
                    caps[k.strip().lower()] = int(v.strip())
                except ValueError:
                    pass
            elif ":" in part:
                k, v = part.split(":", 1)
                try:
                    caps[k.strip().lower()] = int(v.strip())
                except ValueError:
                    pass
        return caps

    def _start_conversion(self):
        src = self.edit_source.text().strip()
        out = self.edit_output.text().strip()

        if not src or not os.path.isdir(src):
            QMessageBox.warning(self, "Invalid Path", "Please select a valid source video dataset folder.")
            return
        if not out:
            QMessageBox.warning(self, "Invalid Path", "Please specify an output folder.")
            return

        params = {
            "source_dir": src,
            "output_dir": out,
            "yaml_path": self.edit_yaml.text().strip() or None,
            "dist": self.spin_dist.value(),
            "img_ext": self.cmb_ext.currentText(),
            "remove_padding": self.chk_padding.isChecked(),
            "black_thresh": self.spin_black_thresh.value(),
            "bg_remove_percent": self.spin_bg_drop.value(),
            "enable_smart_crop": self.chk_smart_crop.isChecked(),
            "crop_size": (self.spin_crop_w.value(), self.spin_crop_h.value()),
            "max_crops_per_frame": self.spin_max_crops.value(),
            "min_visibility": self.spin_min_vis.value(),
            "max_instances_per_class": self._parse_class_caps() or None,
            "flip_augment_percent": self.spin_flip.value(),
            "min_box_size_px": self.spin_min_box.value(),
            "split_mode": self.cmb_split.currentData(),
        }

        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_close.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Starting conversion...")

        self.worker = VideoToYoloWorker(params)
        self.worker.progress_updated.connect(self._on_progress)
        self.worker.finished_success.connect(self._on_finished_success)
        self.worker.finished_error.connect(self._on_finished_error)
        self.worker.start()

    def _cancel_conversion(self):
        if self.worker and self.worker.isRunning():
            self.lbl_status.setText("Cancelling...")
            self.worker.cancel()

    def _on_progress(self, pct, msg):
        self.progress_bar.setValue(pct)
        self.lbl_status.setText(msg)

    def _on_finished_success(self, msg):
        self.progress_bar.setValue(100)
        self.lbl_status.setText("Complete!")
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_close.setEnabled(True)
        QMessageBox.information(self, "Conversion Complete", msg)

    def _on_finished_error(self, err_msg):
        self.lbl_status.setText("Error / Cancelled")
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_close.setEnabled(True)
        QMessageBox.warning(self, "Conversion Info", err_msg)
