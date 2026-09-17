"""
Interactive Dialog for Class-Balancing an existing YOLO Image Dataset.

Same evenly-spaced, non-random balancing logic as the Video -> YOLO
converter's "Balancing & Dedup" tab, applied to a dataset that's already
just images/ + labels/ (optionally split into train/val/test) - regardless
of whether it came from this tool's video converter or any other source.
"""

import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QSpinBox, QCheckBox, QComboBox, QWidget,
    QProgressBar, QMessageBox, QGroupBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QPlainTextEdit
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from viat.converters.image_balance import (
    balance_image_dataset, scan_image_dataset_statistics, build_balance_plan,
)


class ImageScanWorker(QThread):
    finished_success = pyqtSignal(dict)
    finished_error = pyqtSignal(str)

    def __init__(self, source_dir: str, yaml_path: str):
        super().__init__()
        self.source_dir = source_dir
        self.yaml_path = yaml_path

    def run(self):
        try:
            stats = scan_image_dataset_statistics(
                source_dir=Path(self.source_dir),
                yaml_path=Path(self.yaml_path) if self.yaml_path else None,
            )
            self.finished_success.emit(stats)
        except Exception as e:
            self.finished_error.emit(str(e))


class ImagePreviewWorker(QThread):
    finished_success = pyqtSignal(dict)
    finished_error = pyqtSignal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self):
        try:
            plan = build_balance_plan(
                source_dir=Path(self.params["source_dir"]),
                yaml_path=Path(self.params["yaml_path"]) if self.params.get("yaml_path") else None,
                balance_mode=self.params.get("balance_mode", "auto_min"),
                manual_caps=self.params.get("manual_caps"),
                classes_in_balance=self.params.get("classes_in_balance"),
                min_gap=self.params.get("min_gap", 0),
            )
            self.finished_success.emit(plan)
        except Exception as e:
            self.finished_error.emit(str(e))


class ImageBalanceWorker(QThread):
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
            generator = balance_image_dataset(
                source_dir=Path(self.params["source_dir"]),
                output_dir=Path(self.params["output_dir"]),
                yaml_path=Path(self.params["yaml_path"]) if self.params.get("yaml_path") else None,
                balance_mode=self.params.get("balance_mode", "auto_min"),
                manual_caps=self.params.get("manual_caps"),
                classes_in_balance=self.params.get("classes_in_balance"),
                min_gap=self.params.get("min_gap", 0),
                copy_mode=self.params.get("copy_mode", "copy"),
                cancel_callback=lambda: self._is_cancelled,
            )
            last_msg = ""
            for pct, msg in generator:
                if self._is_cancelled:
                    self.finished_error.emit("Balance cancelled by user.")
                    return
                last_msg = msg
                self.progress_updated.emit(pct, msg)
            self.finished_success.emit(last_msg)
        except Exception as e:
            self.finished_error.emit(str(e))


class ImageBalanceDialog(QDialog):
    """Configuration & execution dialog for balancing an existing YOLO image dataset."""

    COL_NAME, COL_INSTANCES, COL_IMAGES, COL_BALANCE, COL_CAP = range(5)

    def __init__(self, parent=None, default_source_dir=""):
        super().__init__(parent)
        self.setWindowTitle("Balance YOLO Image Dataset")
        self.setMinimumWidth(760)
        self.setMinimumHeight(600)
        self.worker = None
        self.scan_worker = None
        self.preview_worker = None

        self._init_ui(default_source_dir)

    def _init_ui(self, default_source_dir):
        main_layout = QVBoxLayout(self)

        paths_group = QGroupBox("Directories & Files")
        paths_grid = QGridLayout(paths_group)

        paths_grid.addWidget(QLabel("Source Image Dataset (images/ + labels/):"), 0, 0)
        self.edit_source = QLineEdit(default_source_dir)
        btn_browse_src = QPushButton("Browse...")
        btn_browse_src.clicked.connect(self._browse_source)
        paths_grid.addWidget(self.edit_source, 0, 1)
        paths_grid.addWidget(btn_browse_src, 0, 2)

        paths_grid.addWidget(QLabel("Output Folder (balanced subset):"), 1, 0)
        self.edit_output = QLineEdit(os.path.join(default_source_dir, "balanced") if default_source_dir else "")
        btn_browse_out = QPushButton("Browse...")
        btn_browse_out.clicked.connect(self._browse_output)
        paths_grid.addWidget(self.edit_output, 1, 1)
        paths_grid.addWidget(btn_browse_out, 1, 2)

        paths_grid.addWidget(QLabel("Optional data.yaml (class names):"), 2, 0)
        self.edit_yaml = QLineEdit("")
        btn_browse_yaml = QPushButton("Browse...")
        btn_browse_yaml.clicked.connect(self._browse_yaml)
        paths_grid.addWidget(self.edit_yaml, 2, 1)
        paths_grid.addWidget(btn_browse_yaml, 2, 2)
        main_layout.addWidget(paths_group)

        scan_row = QHBoxLayout()
        self.btn_scan = QPushButton("\U0001F50D Scan Classes")
        self.btn_scan.clicked.connect(self._scan_classes)
        scan_row.addWidget(self.btn_scan)
        self.lbl_scan_status = QLabel("Not scanned yet.")
        scan_row.addWidget(self.lbl_scan_status)
        scan_row.addStretch()
        main_layout.addLayout(scan_row)

        self.table_classes = QTableWidget(0, 5)
        self.table_classes.setHorizontalHeaderLabels(
            ["Class", "Instances", "Images", "Balance", "Manual cap"]
        )
        self.table_classes.verticalHeader().setVisible(False)
        self.table_classes.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_classes.setEditTriggers(QAbstractItemView.NoEditTriggers)
        header = self.table_classes.horizontalHeader()
        header.setSectionResizeMode(self.COL_NAME, QHeaderView.Stretch)
        header.setSectionResizeMode(self.COL_INSTANCES, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_IMAGES, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_BALANCE, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(self.COL_CAP, QHeaderView.ResizeToContents)
        main_layout.addWidget(self.table_classes)

        settings_group = QGroupBox("Balance Settings")
        settings_grid = QGridLayout(settings_group)

        settings_grid.addWidget(QLabel("Balance Mode:"), 0, 0)
        self.cmb_balance_mode = QComboBox()
        self.cmb_balance_mode.addItem("Auto: balance every included class to the rarest one", "auto_min")
        self.cmb_balance_mode.addItem("Manual per-class caps (set in table)", "manual")
        self.cmb_balance_mode.setToolTip(
            "Auto mode caps every class checked \"Balance\" to the smallest image-count among them - it "
            "can only remove images from over-represented classes, it cannot invent more for an "
            "already-scarce one."
        )
        settings_grid.addWidget(self.cmb_balance_mode, 0, 1, 1, 3)

        settings_grid.addWidget(QLabel("Min Image Gap:"), 1, 0)
        self.spin_min_gap = QSpinBox()
        self.spin_min_gap.setRange(0, 10000)
        self.spin_min_gap.setValue(0)
        self.spin_min_gap.setToolTip(
            "When thinning an over-represented class, keep at least this many positions of separation "
            "(in the path-sorted image list) between two kept images of that class, instead of possibly "
            "keeping near-adjacent frames extracted from the same moment"
        )
        settings_grid.addWidget(self.spin_min_gap, 1, 1)

        settings_grid.addWidget(QLabel("File Mode:"), 1, 2)
        self.cmb_copy_mode = QComboBox()
        self.cmb_copy_mode.addItem("Copy (source untouched)", "copy")
        self.cmb_copy_mode.addItem("Symlink (saves disk space)", "symlink")
        self.cmb_copy_mode.addItem("Move (source loses kept files)", "move")
        self.cmb_copy_mode.setToolTip("How kept image+label pairs are placed into the output folder")
        settings_grid.addWidget(self.cmb_copy_mode, 1, 3)

        main_layout.addWidget(settings_group)

        preview_row = QHBoxLayout()
        self.btn_preview = QPushButton("\U0001F4CA Preview Planned Output")
        self.btn_preview.clicked.connect(self._preview_plan)
        preview_row.addWidget(self.btn_preview)
        preview_row.addStretch()
        main_layout.addLayout(preview_row)

        self.txt_preview = QPlainTextEdit()
        self.txt_preview.setReadOnly(True)
        self.txt_preview.setMaximumHeight(110)
        self.txt_preview.setPlaceholderText("Run a scan first, then click Preview to see projected kept/dropped counts.")
        main_layout.addWidget(self.txt_preview)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        main_layout.addWidget(self.progress_bar)

        self.lbl_status = QLabel("Ready. Pick a source folder, then click \"Scan Classes\".")
        self.lbl_status.setWordWrap(True)
        main_layout.addWidget(self.lbl_status)

        btn_box = QHBoxLayout()
        self.btn_start = QPushButton("\U0001F680 Start Balancing")
        self.btn_start.clicked.connect(self._start_balance)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self._cancel_balance)
        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.reject)
        btn_box.addStretch()
        btn_box.addWidget(self.btn_start)
        btn_box.addWidget(self.btn_cancel)
        btn_box.addWidget(self.btn_close)
        main_layout.addLayout(btn_box)

    # ------------------------------------------------------------------ #
    def _browse_source(self):
        folder = QFileDialog.getExistingDirectory(self, "Select YOLO Image Dataset Folder", self.edit_source.text())
        if folder:
            self.edit_source.setText(folder)
            if not self.edit_output.text():
                self.edit_output.setText(os.path.join(folder, "balanced"))

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", self.edit_output.text())
        if folder:
            self.edit_output.setText(folder)

    def _browse_yaml(self):
        yaml_file, _ = QFileDialog.getOpenFileName(self, "Select data.yaml", "", "YAML Files (*.yaml *.yml)")
        if yaml_file:
            self.edit_yaml.setText(yaml_file)

    # ------------------------------------------------------------------ #
    def _scan_classes(self):
        src = self.edit_source.text().strip()
        if not src or not os.path.isdir(src):
            QMessageBox.warning(self, "Invalid Path", "Please select a valid source image dataset folder first.")
            return
        self.btn_scan.setEnabled(False)
        self.lbl_scan_status.setText("Scanning...")
        self.scan_worker = ImageScanWorker(src, self.edit_yaml.text().strip())
        self.scan_worker.finished_success.connect(self._on_scan_success)
        self.scan_worker.finished_error.connect(self._on_scan_error)
        self.scan_worker.start()

    def _on_scan_success(self, stats: dict):
        self.btn_scan.setEnabled(True)
        n_images = stats.get("total_images", 0)
        n_classes = len(stats.get("classes", {}))
        self.lbl_scan_status.setText(f"Found {n_classes} class(es) across {n_images} image(s).")
        self._populate_class_table(stats)

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
            self.table_classes.setItem(row, self.COL_IMAGES, QTableWidgetItem(str(info["image_count"])))

            balance_chk = QCheckBox()
            balance_chk.setChecked(True)
            balance_chk.setToolTip("Include this class when \"Auto: balance to the rarest class\" is selected")
            self.table_classes.setCellWidget(row, self.COL_BALANCE, balance_chk)

            cap_spin = QSpinBox()
            cap_spin.setRange(0, 100_000_000)
            cap_spin.setValue(0)
            cap_spin.setToolTip("Manual image-count cap for this class (0 = uncapped). Used when Balance Mode = Manual.")
            self.table_classes.setCellWidget(row, self.COL_CAP, cap_spin)

    # ------------------------------------------------------------------ #
    def _read_balance_config(self):
        manual_caps, classes_in_balance = {}, []
        for row in range(self.table_classes.rowCount()):
            name_item = self.table_classes.item(row, self.COL_NAME)
            if name_item is None:
                continue
            cname = name_item.data(Qt.UserRole)
            balance_chk = self.table_classes.cellWidget(row, self.COL_BALANCE)
            cap_spin = self.table_classes.cellWidget(row, self.COL_CAP)
            if balance_chk and balance_chk.isChecked():
                classes_in_balance.append(cname)
            if cap_spin and cap_spin.value() > 0:
                manual_caps[cname] = cap_spin.value()
        return (manual_caps or None), (classes_in_balance or None)

    def _collect_common_params(self):
        manual_caps, classes_in_balance = self._read_balance_config()
        return {
            "source_dir": self.edit_source.text().strip(),
            "yaml_path": self.edit_yaml.text().strip() or None,
            "balance_mode": self.cmb_balance_mode.currentData(),
            "manual_caps": manual_caps,
            "classes_in_balance": classes_in_balance,
            "min_gap": self.spin_min_gap.value(),
        }

    def _preview_plan(self):
        src = self.edit_source.text().strip()
        if not src or not os.path.isdir(src):
            QMessageBox.warning(self, "Invalid Path", "Please select a valid source image dataset folder first.")
            return
        self.btn_preview.setEnabled(False)
        self.txt_preview.setPlainText("Planning...")
        self.preview_worker = ImagePreviewWorker(self._collect_common_params())
        self.preview_worker.finished_success.connect(self._on_preview_success)
        self.preview_worker.finished_error.connect(self._on_preview_error)
        self.preview_worker.start()

    def _on_preview_success(self, plan: dict):
        self.btn_preview.setEnabled(True)
        if plan.get("cancelled"):
            self.txt_preview.setPlainText("Preview cancelled.")
            return
        lines = [
            f"Projected: {len(plan['kept'])} image(s) kept, {len(plan['dropped'])} dropped "
            f"out of {plan['total_images']} total.",
            "Per-class image counts (target if capped):",
        ]
        counts, targets = plan.get("resolved_instance_counts", {}), plan.get("targets", {})
        for name in sorted(counts.keys(), key=lambda n: -counts[n]):
            t = targets.get(name)
            lines.append(f"  {name}: {counts[name]}" + (f"  → target {t}" if t is not None else ""))
        self.txt_preview.setPlainText("\n".join(lines))

    def _on_preview_error(self, msg: str):
        self.btn_preview.setEnabled(True)
        self.txt_preview.setPlainText(f"Preview failed: {msg}")

    # ------------------------------------------------------------------ #
    def _start_balance(self):
        src = self.edit_source.text().strip()
        out = self.edit_output.text().strip()
        if not src or not os.path.isdir(src):
            QMessageBox.warning(self, "Invalid Path", "Please select a valid source image dataset folder.")
            return
        if not out:
            QMessageBox.warning(self, "Invalid Path", "Please specify an output folder.")
            return
        if os.path.abspath(out) == os.path.abspath(src):
            QMessageBox.warning(self, "Invalid Path", "Output folder must be different from the source folder.")
            return

        params = self._collect_common_params()
        params["output_dir"] = out
        params["copy_mode"] = self.cmb_copy_mode.currentData()

        if params["copy_mode"] == "move":
            reply = QMessageBox.question(
                self, "Confirm Move",
                "\"Move\" removes the kept files from the source folder. This cannot be undone. Continue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                return

        self.btn_start.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        self.btn_close.setEnabled(False)
        self.progress_bar.setValue(0)
        self.lbl_status.setText("Starting...")

        self.worker = ImageBalanceWorker(params)
        self.worker.progress_updated.connect(self._on_progress)
        self.worker.finished_success.connect(self._on_finished_success)
        self.worker.finished_error.connect(self._on_finished_error)
        self.worker.start()

    def _cancel_balance(self):
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
        QMessageBox.information(self, "Balance Complete", msg)

    def _on_finished_error(self, err_msg):
        self.lbl_status.setText("Error / Cancelled")
        self.btn_start.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.btn_close.setEnabled(True)
        QMessageBox.warning(self, "Balance Info", err_msg)
