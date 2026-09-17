"""
Interactive Dialog for Converting Video Datasets into YOLO Image Datasets.
"""

import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QSpinBox, QDoubleSpinBox, QCheckBox,
    QComboBox, QTabWidget, QWidget, QProgressBar, QMessageBox, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QPlainTextEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from viat.converters.video_to_yolo import (
    convert_video_dataset_to_yolo, scan_dataset_statistics, build_plan, load_yaml_classes,
)

FATE_LABELS = [
    ("map", "Map → include"),
    ("skip", "Skip (drop boxes, keep frame)"),
    ("soft", "Soft-delete (drop boxes, drop frame if empty)"),
    ("purge", "Purge (drop whole frame)"),
]
FATE_KEYS = [k for k, _ in FATE_LABELS]


class ScanWorker(QThread):
    """Background thread running class discovery (scan_dataset_statistics)."""
    finished_success = pyqtSignal(dict)
    finished_error = pyqtSignal(str)

    def __init__(self, source_dir: str, yaml_path: str):
        super().__init__()
        self.source_dir = source_dir
        self.yaml_path = yaml_path

    def run(self):
        try:
            stats = scan_dataset_statistics(
                source_dir=Path(self.source_dir),
                yaml_path=Path(self.yaml_path) if self.yaml_path else None,
            )
            self.finished_success.emit(stats)
        except Exception as e:
            self.finished_error.emit(str(e))


class PreviewWorker(QThread):
    """Background thread running a plan-only dry-run with current settings."""
    finished_success = pyqtSignal(dict)
    finished_error = pyqtSignal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self):
        try:
            plan = build_plan(
                source_dir=Path(self.params["source_dir"]),
                yaml_path=Path(self.params["yaml_path"]) if self.params.get("yaml_path") else None,
                class_config=self.params.get("class_config"),
                dist=self.params.get("dist", 1),
                bg_remove_percent=self.params.get("bg_remove_percent", 0.0),
                balance_mode=self.params.get("balance_mode", "none"),
                manual_caps=self.params.get("max_instances_per_class"),
                classes_in_balance=self.params.get("classes_in_balance"),
                min_frame_gap=self.params.get("min_frame_gap", 0),
                max_frames_per_video=self.params.get("max_frames_per_video"),
            )
            self.finished_success.emit(plan)
        except Exception as e:
            self.finished_error.emit(str(e))


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
                min_crop_size=self.params.get("min_crop_size", (320, 320)),
                max_crops_per_frame=self.params.get("max_crops_per_frame", 3),
                max_bg_crops_per_frame=self.params.get("max_bg_crops_per_frame", 1),
                min_visibility=self.params.get("min_visibility", 0.7),
                context_padding=self.params.get("context_padding", 0.2),
                overlap_iou_threshold=self.params.get("overlap_iou_threshold", 0.5),
                square_crops=self.params.get("square_crops", False),
                default_size_crop_chance=self.params.get("default_size_crop_chance", 0.3),
                wide_position=self.params.get("wide_position", True),
                include_default_frame=self.params.get("include_default_frame", True),
                default_frame_chance=self.params.get("default_frame_chance", 1.0),
                max_instances_per_class=self.params.get("max_instances_per_class"),
                balance_mode=self.params.get("balance_mode", "none"),
                classes_in_balance=self.params.get("classes_in_balance"),
                min_frame_gap=self.params.get("min_frame_gap", 0),
                max_frames_per_video=self.params.get("max_frames_per_video"),
                enable_dedup=self.params.get("enable_dedup", False),
                dedup_hamming_threshold=self.params.get("dedup_hamming_threshold", 4),
                flip_augment_percent=self.params.get("flip_augment_percent", 0.0),
                min_box_size_px=self.params.get("min_box_size_px", 2.0),
                class_config=self.params.get("class_config"),
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

    COL_NAME, COL_INSTANCES, COL_FRAMES, COL_FATE, COL_TARGET, COL_BALANCE, COL_CAP = range(7)

    def __init__(self, parent=None, default_source_dir=""):
        super().__init__(parent)
        self.setWindowTitle("Convert Video Dataset to YOLO Dataset")
        self.setMinimumWidth(820)
        self.setMinimumHeight(640)
        self.worker = None
        self.scan_worker = None
        self.preview_worker = None
        self.last_scan = None  # cached scan_dataset_statistics() result
        self.yaml_global_classes = []  # class names loaded from the existing data.yaml, if any

        self._init_ui(default_source_dir)

    # ------------------------------------------------------------------ #
    # UI construction
    # ------------------------------------------------------------------ #
    def _init_ui(self, default_source_dir):
        main_layout = QVBoxLayout(self)

        tabs = QTabWidget()

        tabs.addTab(self._build_general_tab(default_source_dir), "General")
        tabs.addTab(self._build_classes_tab(), "Classes")
        tabs.addTab(self._build_crop_tab(), "Smart Cropping (ROI)")
        tabs.addTab(self._build_balance_tab(), "Balancing & Dedup")
        tabs.addTab(self._build_aug_tab(), "Augmentation")

        main_layout.addWidget(tabs)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        main_layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("Ready. Pick a source folder, then click \"Scan Classes\".")
        self.lbl_status.setWordWrap(True)
        main_layout.addWidget(self.lbl_status)

        btn_box = QHBoxLayout()
        self.btn_start = QPushButton("\U0001F680 Start Conversion")
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

    def _build_general_tab(self, default_source_dir):
        tab_general = QWidget()
        gen_layout = QVBoxLayout(tab_general)

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

        mismatch_group = QGroupBox("Annotation / Video Mismatches (from last scan)")
        mismatch_layout = QVBoxLayout(mismatch_group)
        self.txt_mismatches = QPlainTextEdit()
        self.txt_mismatches.setReadOnly(True)
        self.txt_mismatches.setPlaceholderText(
            "Run \"Scan Classes\" (Classes tab) to check whether any video's frame "
            "count disagrees with its .txt annotation file's line count."
        )
        self.txt_mismatches.setMaximumHeight(90)
        mismatch_layout.addWidget(self.txt_mismatches)
        gen_layout.addWidget(mismatch_group)

        gen_layout.addStretch()
        return tab_general

    def _build_classes_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        row = QHBoxLayout()
        self.btn_scan = QPushButton("\U0001F50D Scan Classes")
        self.btn_scan.setToolTip("Parse every annotation file in the source folder and list the classes found")
        self.btn_scan.clicked.connect(self._scan_classes)
        row.addWidget(self.btn_scan)
        self.lbl_scan_status = QLabel("Not scanned yet.")
        row.addWidget(self.lbl_scan_status)
        row.addStretch()
        layout.addLayout(row)

        self.table_classes = QTableWidget(0, 7)
        self.table_classes.setHorizontalHeaderLabels(
            ["Class", "Instances", "Frames", "Fate", "Map to (rename)", "Balance", "Manual cap"]
        )
        self.table_classes.verticalHeader().setVisible(False)
        self.table_classes.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_classes.setEditTriggers(QAbstractItemView.NoEditTriggers)
        # Columns that hold free-form text (class/target names) grow with the
        # dialog; columns that hold a fixed-size widget (combo/checkbox/spin
        # box) size to that widget instead of being left at an arbitrary
        # one-time width, so the table keeps scaling sensibly as the dialog
        # is resized rather than clipping or leaving dead space.
        header = self.table_classes.horizontalHeader()
        header.setSectionResizeMode(self.COL_NAME, QHeaderView.Stretch)
        header.setSectionResizeMode(self.COL_INSTANCES, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_FRAMES, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_FATE, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_TARGET, QHeaderView.Stretch)
        header.setSectionResizeMode(self.COL_BALANCE, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_CAP, QHeaderView.ResizeToContents)
        layout.addWidget(self.table_classes)

        legend = QLabel(
            "Map = written to the final dataset (optionally renamed/merged into another class).  "
            "Skip = boxes dropped, frame kept as background.  "
            "Soft-delete = boxes dropped, frame dropped too if nothing else remains in it.  "
            "Purge = the entire frame is dropped whenever this class appears in it."
        )
        legend.setWordWrap(True)
        layout.addWidget(legend)
        return tab

    def _build_crop_tab(self):
        tab_crop = QWidget()
        crop_layout = QVBoxLayout(tab_crop)
        crop_group = QGroupBox("Smart Object-Focused Multi-Cropping")
        crop_grid = QGridLayout(crop_group)

        self.chk_smart_crop = QCheckBox("Enable Smart Multi-Cropping")
        self.chk_smart_crop.setToolTip(
            "Crops high-res sub-images around distant/small object clusters, adapting crop size to "
            "the objects it frames, instead of shipping a whole downscaled 4K frame"
        )
        self.chk_smart_crop.setChecked(False)
        crop_grid.addWidget(self.chk_smart_crop, 0, 0, 1, 4)

        crop_grid.addWidget(QLabel("Max Crop Width:"), 1, 0)
        self.spin_crop_w = QSpinBox()
        self.spin_crop_w.setRange(128, 7680)
        self.spin_crop_w.setValue(640)
        crop_grid.addWidget(self.spin_crop_w, 1, 1)

        crop_grid.addWidget(QLabel("Max Crop Height:"), 1, 2)
        self.spin_crop_h = QSpinBox()
        self.spin_crop_h.setRange(128, 4320)
        self.spin_crop_h.setValue(640)
        crop_grid.addWidget(self.spin_crop_h, 1, 3)

        crop_grid.addWidget(QLabel("Min Crop Width:"), 2, 0)
        self.spin_min_crop_w = QSpinBox()
        self.spin_min_crop_w.setRange(32, 7680)
        self.spin_min_crop_w.setValue(320)
        self.spin_min_crop_w.setToolTip("Crop size adapts down to this floor around small/isolated objects, instead of always using the max size")
        crop_grid.addWidget(self.spin_min_crop_w, 2, 1)

        crop_grid.addWidget(QLabel("Min Crop Height:"), 2, 2)
        self.spin_min_crop_h = QSpinBox()
        self.spin_min_crop_h.setRange(32, 4320)
        self.spin_min_crop_h.setValue(320)
        crop_grid.addWidget(self.spin_min_crop_h, 2, 3)

        crop_grid.addWidget(QLabel("Max Object Crops per Frame:"), 3, 0)
        self.spin_max_crops = QSpinBox()
        self.spin_max_crops.setRange(1, 20)
        self.spin_max_crops.setValue(3)
        self.spin_max_crops.setToolTip(
            "Crops taken from a frame that HAS object(s) in it. With few/small object clusters, budget left "
            "over after one crop per cluster is used for extra varied crops of the same cluster(s) instead of "
            "going unused - so even a single isolated object can yield several different crops of it. Set "
            "separately from Max Background Crops below, since an empty tile has no object diversity to gain "
            "from taking as many of them as a busy foreground frame."
        )
        crop_grid.addWidget(self.spin_max_crops, 3, 1)

        crop_grid.addWidget(QLabel("Min Visibility Ratio:"), 3, 2)
        self.spin_min_vis = QDoubleSpinBox()
        self.spin_min_vis.setRange(0.05, 1.0)
        self.spin_min_vis.setSingleStep(0.05)
        self.spin_min_vis.setValue(0.70)
        self.spin_min_vis.setToolTip(
            "Keep a box only if at least this fraction of it survives the crop; also the bar used to "
            "decide a crop is \"clean\" (see below). Set high - a box may already be a labeled sliver "
            "of an object that's mostly hidden behind something unlabeled, and cropping it further "
            "compounds that, e.g. an already-30%-visible object cropped down to 40% of its box leaves "
            "only ~12% of the real object standing in for the whole class."
        )
        crop_grid.addWidget(self.spin_min_vis, 3, 3)

        crop_grid.addWidget(QLabel("Context Padding:"), 4, 0)
        self.spin_context_pad = QDoubleSpinBox()
        self.spin_context_pad.setRange(0.0, 1.0)
        self.spin_context_pad.setSingleStep(0.05)
        self.spin_context_pad.setValue(0.20)
        crop_grid.addWidget(self.spin_context_pad, 4, 1)

        crop_grid.addWidget(QLabel("Max Crop Overlap (IoU):"), 4, 2)
        self.spin_overlap = QDoubleSpinBox()
        self.spin_overlap.setRange(0.0, 1.0)
        self.spin_overlap.setSingleStep(0.05)
        self.spin_overlap.setValue(0.50)
        self.spin_overlap.setToolTip("Crops that overlap an already-chosen crop by more than this are skipped as near-duplicates")
        crop_grid.addWidget(self.spin_overlap, 4, 3)

        self.chk_square_crops = QCheckBox("Force Square Crops")
        self.chk_square_crops.setToolTip(
            "Crop windows are square (side = the larger of width/height, clamped to the min/max crop "
            "size) instead of an arbitrary rectangle"
        )
        self.chk_square_crops.setChecked(False)
        crop_grid.addWidget(self.chk_square_crops, 5, 0, 1, 2)

        self.chk_wide_position = QCheckBox("Vary Object Position in Crop")
        self.chk_wide_position.setToolTip(
            "Let the object land anywhere in the crop (not just near-center) as long as it stays fully "
            "visible - how far it can actually move is naturally limited by how much smaller the object "
            "is than the crop and by the frame's edges. When another object is nearby, only a position "
            "that leaves it either cleanly included or cleanly excluded is accepted - never half-cut; if "
            "no such clean position exists, that crop is skipped rather than truncating anything (the "
            "object stays fully labeled in the whole-frame \"_df\" image instead)."
        )
        self.chk_wide_position.setChecked(True)
        crop_grid.addWidget(self.chk_wide_position, 6, 0, 1, 4)

        crop_grid.addWidget(QLabel("Default-Size Crop Chance:"), 5, 2)
        self.spin_default_size_chance = QDoubleSpinBox()
        self.spin_default_size_chance.setRange(0.0, 1.0)
        self.spin_default_size_chance.setSingleStep(0.05)
        self.spin_default_size_chance.setValue(0.30)
        self.spin_default_size_chance.setToolTip(
            "Chance that a given object-focused crop uses the fixed max crop size instead of shrinking "
            "to hug the object cluster, so the object crops themselves mix tightly-zoomed and "
            "default-context sizes"
        )
        crop_grid.addWidget(self.spin_default_size_chance, 5, 3)

        self.chk_default_frame = QCheckBox("Also Save Whole Frame (\"_df\")")
        self.chk_default_frame.setToolTip(
            "Alongside the object-focused crop(s) (\"_c0\", \"_c1\", ...), also save one extra image of "
            "the whole frame with all its labels - suffixed \"_df\" - so the model also sees full-scene "
            "context, not only zoomed-in sub-crops. Only applies when a frame is actually being smart-"
            "cropped; a frame that already fits the crop size is the default view already."
        )
        self.chk_default_frame.setChecked(True)
        crop_grid.addWidget(self.chk_default_frame, 7, 0, 1, 2)

        crop_grid.addWidget(QLabel("Whole-Frame Chance %:"), 7, 2)
        self.spin_default_frame_chance = QDoubleSpinBox()
        self.spin_default_frame_chance.setRange(0, 100)
        self.spin_default_frame_chance.setSingleStep(5)
        self.spin_default_frame_chance.setValue(100)
        self.spin_default_frame_chance.setToolTip("How often the extra whole-frame \"_df\" image is added (100% = almost always)")
        crop_grid.addWidget(self.spin_default_frame_chance, 7, 3)

        crop_grid.addWidget(QLabel("Max Background Crops per Frame:"), 8, 0)
        self.spin_max_bg_crops = QSpinBox()
        self.spin_max_bg_crops.setRange(1, 20)
        self.spin_max_bg_crops.setValue(1)
        self.spin_max_bg_crops.setToolTip(
            "Crops taken from a frame that has NO objects in it (empty/background). Kept separate and "
            "usually much lower than Max Object Crops above - an empty tile is just filler diversity, not "
            "worth multiplying the way a busy object frame is."
        )
        crop_grid.addWidget(self.spin_max_bg_crops, 8, 1)

        crop_layout.addWidget(crop_group)
        crop_layout.addStretch()
        return tab_crop

    def _build_balance_tab(self):
        tab_balance = QWidget()
        balance_layout = QVBoxLayout(tab_balance)

        balance_group = QGroupBox("Class Balancing (evenly-spaced, not random, downsampling)")
        balance_grid = QGridLayout(balance_group)

        balance_grid.addWidget(QLabel("Balance Mode:"), 0, 0)
        self.cmb_balance_mode = QComboBox()
        self.cmb_balance_mode.addItem("Off - keep every frame", "none")
        self.cmb_balance_mode.addItem("Manual per-class caps (set in Classes tab)", "manual")
        self.cmb_balance_mode.addItem("Auto: balance every included class to the rarest one", "auto_min")
        self.cmb_balance_mode.setToolTip(
            "Auto mode caps every class checked \"Balance\" in the Classes tab to the smallest "
            "instance count among them - it can only remove frames from over-represented classes, "
            "it cannot invent more frames for an already-scarce class."
        )
        balance_grid.addWidget(self.cmb_balance_mode, 0, 1, 1, 3)

        balance_grid.addWidget(QLabel("Min Frame Gap (frames):"), 1, 0)
        self.spin_min_gap = QSpinBox()
        self.spin_min_gap.setRange(0, 10000)
        self.spin_min_gap.setValue(0)
        self.spin_min_gap.setToolTip(
            "When thinning an over-represented class, keep at least this many frames of "
            "separation between two kept frames of that class instead of possibly keeping "
            "near-adjacent, near-duplicate frames"
        )
        balance_grid.addWidget(self.spin_min_gap, 1, 1)

        balance_grid.addWidget(QLabel("Background Frame Drop %:"), 1, 2)
        self.spin_bg_drop = QDoubleSpinBox()
        self.spin_bg_drop.setRange(0, 100)
        self.spin_bg_drop.setValue(0)
        self.spin_bg_drop.setToolTip("Randomly drop empty background frames ([]) to prevent dataset imbalance")
        balance_grid.addWidget(self.spin_bg_drop, 1, 3)

        balance_grid.addWidget(QLabel("Max Frames per Video (0 = no cap):"), 2, 0)
        self.spin_max_per_video = QSpinBox()
        self.spin_max_per_video.setRange(0, 10_000_000)
        self.spin_max_per_video.setValue(0)
        self.spin_max_per_video.setToolTip("Caps how many frames any single video can contribute, so one long/dense clip can't dominate the dataset")
        balance_grid.addWidget(self.spin_max_per_video, 2, 1)

        balance_layout.addWidget(balance_group)

        dedup_group = QGroupBox("Near-Duplicate Frame Detection")
        dedup_grid = QGridLayout(dedup_group)
        self.chk_dedup = QCheckBox("Skip near-duplicate frames (perceptual hash)")
        self.chk_dedup.setToolTip("Compares each candidate frame against recently kept frames from the same video and skips near-identical ones")
        dedup_grid.addWidget(self.chk_dedup, 0, 0, 1, 2)

        dedup_grid.addWidget(QLabel("Similarity Threshold (lower = stricter):"), 1, 0)
        self.spin_dedup_thresh = QSpinBox()
        self.spin_dedup_thresh.setRange(0, 64)
        self.spin_dedup_thresh.setValue(4)
        dedup_grid.addWidget(self.spin_dedup_thresh, 1, 1)

        balance_layout.addWidget(dedup_group)

        preview_row = QHBoxLayout()
        self.btn_preview = QPushButton("\U0001F4CA Preview Planned Output")
        self.btn_preview.setToolTip("Runs the planning pass only (no video decode) and reports projected image counts under the current settings")
        self.btn_preview.clicked.connect(self._preview_plan)
        preview_row.addWidget(self.btn_preview)
        preview_row.addStretch()
        balance_layout.addLayout(preview_row)

        self.txt_preview = QPlainTextEdit()
        self.txt_preview.setReadOnly(True)
        self.txt_preview.setMaximumHeight(120)
        self.txt_preview.setPlaceholderText("Run a scan first, then click Preview to see projected per-class output counts.")
        balance_layout.addWidget(self.txt_preview)

        balance_layout.addStretch()
        return tab_balance

    def _build_aug_tab(self):
        tab_aug = QWidget()
        aug_layout = QVBoxLayout(tab_aug)
        aug_group = QGroupBox("Pre-Augmentation & Quality Filters")
        aug_grid = QGridLayout(aug_group)

        aug_grid.addWidget(QLabel("Horizontal Flip %:"), 0, 0)
        self.spin_flip = QDoubleSpinBox()
        self.spin_flip.setRange(0, 100)
        self.spin_flip.setValue(0)
        self.spin_flip.setToolTip(
            "Chance to add horizontally mirrored copies with inverted YOLO coords. Note: most YOLO "
            "trainers already apply a live horizontal-flip augmentation every epoch for free - this "
            "offline option mainly matters if you are NOT using such a trainer."
        )
        aug_grid.addWidget(self.spin_flip, 0, 1)

        aug_grid.addWidget(QLabel("Min Box Size (px):"), 1, 0)
        self.spin_min_box = QDoubleSpinBox()
        self.spin_min_box.setRange(0.5, 100.0)
        self.spin_min_box.setValue(2.0)
        aug_grid.addWidget(self.spin_min_box, 1, 1)

        aug_layout.addWidget(aug_group)
        aug_layout.addStretch()
        return tab_aug

    # ------------------------------------------------------------------ #
    # Browsing
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    # Class scanning / table population
    # ------------------------------------------------------------------ #
    def _scan_classes(self):
        src = self.edit_source.text().strip()
        if not src or not os.path.isdir(src):
            QMessageBox.warning(self, "Invalid Path", "Please select a valid source video dataset folder first.")
            return

        self.btn_scan.setEnabled(False)
        self.lbl_scan_status.setText("Scanning...")
        self.scan_worker = ScanWorker(src, self.edit_yaml.text().strip())
        self.scan_worker.finished_success.connect(self._on_scan_success)
        self.scan_worker.finished_error.connect(self._on_scan_error)
        self.scan_worker.start()

    def _on_scan_success(self, stats: dict):
        self.btn_scan.setEnabled(True)
        self.last_scan = stats
        n_videos = stats.get("total_videos", 0)
        n_classes = len(stats.get("classes", {}))

        # Load the *existing* yaml class list directly (not stats["global_names"],
        # which already has classes auto-created from unmatched local names mixed
        # in) so the "Map to" picker offers exactly what's already in the yaml.
        yaml_path = self.edit_yaml.text().strip()
        self.yaml_global_classes = []
        if yaml_path and os.path.isfile(yaml_path):
            try:
                self.yaml_global_classes, _ = load_yaml_classes(Path(yaml_path))
            except Exception:
                self.yaml_global_classes = []

        status = f"Found {n_classes} class(es) across {n_videos} video(s)."
        if yaml_path:
            status += f" {len(self.yaml_global_classes)} existing class(es) loaded from yaml."
        self.lbl_scan_status.setText(status)
        self._populate_class_table(stats)
        self._populate_mismatches(stats)

    def _on_scan_error(self, msg: str):
        self.btn_scan.setEnabled(True)
        self.lbl_scan_status.setText("Scan failed.")
        QMessageBox.warning(self, "Scan Failed", msg)

    def _populate_class_table(self, stats: dict):
        classes = stats.get("classes", {})
        self.table_classes.setRowCount(0)
        self.table_classes.setRowCount(len(classes))

        for row, (cname, info) in enumerate(sorted(classes.items(), key=lambda kv: -kv[1]["instance_count"])):
            name_item = QTableWidgetItem(cname)
            name_item.setData(Qt.UserRole, cname)
            self.table_classes.setItem(row, self.COL_NAME, name_item)
            self.table_classes.setItem(row, self.COL_INSTANCES, QTableWidgetItem(str(info["instance_count"])))
            self.table_classes.setItem(row, self.COL_FRAMES, QTableWidgetItem(str(info["frame_count"])))

            # A local class only defaults to "map" when it actually matches an
            # existing yaml class by name. Anything else defaults to "skip"
            # (ignored) instead of silently being queued up to create a new
            # same-named class - creating a class is something the user has to
            # opt into per row, not something that happens by default.
            matched_yaml_name = next(
                (n for n in self.yaml_global_classes if n.strip().lower() == cname.strip().lower()),
                None,
            )
            # "Ignore by default when it's not in the yaml" only makes sense
            # when there IS a yaml to check against. With no yaml loaded at
            # all, there's nothing for a class to fail to match, so every
            # class defaults to Map (create it) same as before - otherwise
            # running without a yaml would silently skip every single class.
            has_yaml = bool(self.yaml_global_classes)
            default_fate = "map" if (matched_yaml_name or not has_yaml) else "skip"

            fate_combo = QComboBox()
            for key, label in FATE_LABELS:
                fate_combo.addItem(label, key)
            fate_combo.setCurrentIndex(FATE_KEYS.index(default_fate))

            target_combo = QComboBox()
            target_combo.setEditable(True)
            # Offer every class already defined in the yaml so the user can map
            # a differently-named local class (e.g. "human") onto an existing
            # global one (e.g. "person") instead of only ever matching by exact
            # name or silently creating a brand-new class.
            existing_options = sorted({*self.yaml_global_classes, cname})
            target_combo.addItems(existing_options)
            if matched_yaml_name:
                target_combo.setCurrentText(matched_yaml_name)
                target_combo.setEnabled(True)
                target_combo.setToolTip(
                    "Global class this local class's boxes are written as. Pick an existing yaml class to "
                    "merge into it, or type a new name to create one."
                )
            elif not has_yaml:
                target_combo.setCurrentText(cname)
                target_combo.setEnabled(True)
                target_combo.setToolTip(
                    "No data.yaml loaded, so there's nothing to merge into yet - this will be created as a "
                    "new class. Load a data.yaml on the General tab first if you meant to merge into one."
                )
            else:
                target_combo.setCurrentIndex(-1)
                target_combo.lineEdit().setPlaceholderText("IGNORE — not in yaml")
                target_combo.setEnabled(False)
                target_combo.setToolTip(
                    "No matching class found in the loaded data.yaml, so this class is ignored by default. "
                    "Switch Fate to \"Map\" and pick/type a target here to include it instead."
                )
            self.table_classes.setCellWidget(row, self.COL_TARGET, target_combo)

            def _on_fate_changed(_index, _fate_combo=fate_combo, _target_combo=target_combo, _cname=cname, _matched=matched_yaml_name):
                is_map = _fate_combo.currentData() == "map"
                _target_combo.setEnabled(is_map)
                if is_map and _target_combo.currentIndex() == -1 and not _target_combo.currentText().strip():
                    _target_combo.setCurrentText(_matched or _cname)

            fate_combo.currentIndexChanged.connect(_on_fate_changed)
            self.table_classes.setCellWidget(row, self.COL_FATE, fate_combo)

            balance_chk = QCheckBox()
            balance_chk.setChecked(True)
            balance_chk.setToolTip("Include this class when \"Auto: balance to the rarest class\" is selected")
            self.table_classes.setCellWidget(row, self.COL_BALANCE, balance_chk)

            cap_spin = QSpinBox()
            cap_spin.setRange(0, 100_000_000)
            cap_spin.setValue(0)
            cap_spin.setToolTip("Manual instance cap for this class (0 = uncapped). Used when Balance Mode = Manual.")
            self.table_classes.setCellWidget(row, self.COL_CAP, cap_spin)

    def _populate_mismatches(self, stats: dict):
        mismatches = stats.get("mismatches", [])
        if not mismatches:
            self.txt_mismatches.setPlainText("No mismatches - every video's decoded frame count matches its annotation file.")
            return
        lines = [
            f"{Path(m['video']).name}: annotation has {m['annotated_lines']} line(s), "
            f"video has {m['actual_frames']} frame(s) (diff {m['diff']:+d})"
            for m in mismatches
        ]
        self.txt_mismatches.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------ #
    # Reading the class table back into engine params
    # ------------------------------------------------------------------ #
    def _read_class_config(self):
        """Returns (class_config, max_instances_per_class, classes_in_balance)."""
        class_config = {}
        manual_caps = {}
        classes_in_balance = []

        for row in range(self.table_classes.rowCount()):
            name_item = self.table_classes.item(row, self.COL_NAME)
            if name_item is None:
                continue
            cname = name_item.data(Qt.UserRole)

            fate_combo = self.table_classes.cellWidget(row, self.COL_FATE)
            fate_key = fate_combo.currentData() if fate_combo else "map"

            target_combo = self.table_classes.cellWidget(row, self.COL_TARGET)
            target = (target_combo.currentText().strip() if target_combo else "") or cname

            balance_chk = self.table_classes.cellWidget(row, self.COL_BALANCE)
            is_balanced = balance_chk.isChecked() if balance_chk else True

            cap_spin = self.table_classes.cellWidget(row, self.COL_CAP)
            cap_val = cap_spin.value() if cap_spin else 0

            class_config[cname] = {"fate": fate_key, "target": target}

            if fate_key == "map":
                if is_balanced:
                    classes_in_balance.append(target)
                if cap_val > 0:
                    manual_caps[target] = cap_val

        return class_config, (manual_caps or None), (classes_in_balance or None)

    # ------------------------------------------------------------------ #
    # Preview (plan-only dry run)
    # ------------------------------------------------------------------ #
    def _collect_common_params(self):
        class_config, manual_caps, classes_in_balance = self._read_class_config()
        return {
            "source_dir": self.edit_source.text().strip(),
            "yaml_path": self.edit_yaml.text().strip() or None,
            "dist": self.spin_dist.value(),
            "bg_remove_percent": self.spin_bg_drop.value(),
            "balance_mode": self.cmb_balance_mode.currentData(),
            "max_instances_per_class": manual_caps,
            "classes_in_balance": classes_in_balance,
            "min_frame_gap": self.spin_min_gap.value(),
            "max_frames_per_video": self.spin_max_per_video.value() or None,
            "class_config": class_config or None,
        }

    def _preview_plan(self):
        src = self.edit_source.text().strip()
        if not src or not os.path.isdir(src):
            QMessageBox.warning(self, "Invalid Path", "Please select a valid source video dataset folder first.")
            return

        self.btn_preview.setEnabled(False)
        self.txt_preview.setPlainText("Planning...")
        self.preview_worker = PreviewWorker(self._collect_common_params())
        self.preview_worker.finished_success.connect(self._on_preview_success)
        self.preview_worker.finished_error.connect(self._on_preview_error)
        self.preview_worker.start()

    def _on_preview_success(self, plan: dict):
        self.btn_preview.setEnabled(True)
        if plan.get("cancelled"):
            self.txt_preview.setPlainText("Preview cancelled.")
            return

        final_plan = plan.get("final_plan", {})
        total_frames = sum(len(v) for v in final_plan.values())
        lines = [
            f"Projected output: {total_frames} image(s) across {len(final_plan)} video(s)",
            f"  DELETED frames dropped: {plan.get('deleted_total', 0)}",
            f"  Background frames thinned: {plan.get('bg_dropped_total', 0)}",
            f"  Frames dropped for class balance: {plan.get('balance_dropped_total', 0)}",
        ]
        counts = plan.get("resolved_instance_counts", {})
        targets = plan.get("targets", {})
        if counts:
            lines.append("Per-class instance counts (target if capped):")
            for name in sorted(counts.keys(), key=lambda n: -counts[n]):
                t = targets.get(name)
                lines.append(f"  {name}: {counts[name]}" + (f"  → target {t}" if t is not None else ""))
        if plan.get("mismatches"):
            lines.append(f"{len(plan['mismatches'])} video(s) have annotation/frame-count mismatches (see General tab after a scan).")

        self.txt_preview.setPlainText("\n".join(lines))

    def _on_preview_error(self, msg: str):
        self.btn_preview.setEnabled(True)
        self.txt_preview.setPlainText(f"Preview failed: {msg}")

    # ------------------------------------------------------------------ #
    # Conversion
    # ------------------------------------------------------------------ #
    def _start_conversion(self):
        src = self.edit_source.text().strip()
        out = self.edit_output.text().strip()

        if not src or not os.path.isdir(src):
            QMessageBox.warning(self, "Invalid Path", "Please select a valid source video dataset folder.")
            return
        if not out:
            QMessageBox.warning(self, "Invalid Path", "Please specify an output folder.")
            return

        params = self._collect_common_params()
        params.update({
            "output_dir": out,
            "img_ext": self.cmb_ext.currentText(),
            "remove_padding": self.chk_padding.isChecked(),
            "black_thresh": self.spin_black_thresh.value(),
            "enable_smart_crop": self.chk_smart_crop.isChecked(),
            "crop_size": (self.spin_crop_w.value(), self.spin_crop_h.value()),
            "min_crop_size": (self.spin_min_crop_w.value(), self.spin_min_crop_h.value()),
            "max_crops_per_frame": self.spin_max_crops.value(),
            "max_bg_crops_per_frame": self.spin_max_bg_crops.value(),
            "min_visibility": self.spin_min_vis.value(),
            "context_padding": self.spin_context_pad.value(),
            "overlap_iou_threshold": self.spin_overlap.value(),
            "square_crops": self.chk_square_crops.isChecked(),
            "default_size_crop_chance": self.spin_default_size_chance.value(),
            "wide_position": self.chk_wide_position.isChecked(),
            "include_default_frame": self.chk_default_frame.isChecked(),
            "default_frame_chance": self.spin_default_frame_chance.value() / 100.0,
            "enable_dedup": self.chk_dedup.isChecked(),
            "dedup_hamming_threshold": self.spin_dedup_thresh.value(),
            "flip_augment_percent": self.spin_flip.value(),
            "min_box_size_px": self.spin_min_box.value(),
            "split_mode": self.cmb_split.currentData(),
        })

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
