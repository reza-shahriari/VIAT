"""
Evaluation Inspector Dock for VIAT Main Window

Provides interactive controls directly inside the main UI for:
- Toggling Evaluation Inspection Mode
- Real-time confidence score threshold adjustments
- IoU threshold adjustments
- Error filtering (Show All, Errors Only, FP Only, FN Only, TP Only)
- Next/Previous error frame jumping with keyboard shortcuts
- Live precision, recall, F1, TP, FP, FN metrics
- One-click promotion of False Positives to Ground Truth annotations
"""

import os
import json
from PyQt5.QtWidgets import (
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QPushButton,
    QSlider,
    QDoubleSpinBox,
    QComboBox,
    QGroupBox,
    QFrame,
    QListWidget,
    QListWidgetItem,
    QFileDialog,
    QMessageBox,
    QToolButton,
    QApplication,
    QShortcut
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QKeySequence, QFont, QColor, QIcon


class EvaluationInspectorDock(QDockWidget):
    """Dock widget for in-app visual evaluation error inspection."""

    eval_mode_toggled = pyqtSignal(bool)
    conf_threshold_changed = pyqtSignal(float)
    iou_threshold_changed = pyqtSignal(float)
    filter_changed = pyqtSignal(str)
    jump_to_frame_requested = pyqtSignal(int)
    promote_fp_requested = pyqtSignal(dict, int)  # (prediction_dict, frame_idx)
    load_predictions_requested = pyqtSignal()
    video_selected = pyqtSignal(str)              # video_name
    save_ground_truth_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Evaluation Inspector", parent)
        self.main_window = parent
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self.predictions_by_frame = {}
        self.current_video_name = ""
        self.dataset_videos = []
        self.error_frames = []
        self.current_error_idx = -1

        self._init_ui()
        self._setup_shortcuts()

    def _init_ui(self):
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # 1. Mode Header & Toggle
        top_box = QGroupBox("Inspection Mode")
        top_box.setStyleSheet("QGroupBox { font-weight: bold; }")
        top_layout = QVBoxLayout(top_box)

        self.btn_toggle_mode = QPushButton("👁️ Enable Evaluation View")
        self.btn_toggle_mode.setCheckable(True)
        self.btn_toggle_mode.setChecked(False)
        self.btn_toggle_mode.setMinimumHeight(34)
        self.btn_toggle_mode.setStyleSheet("""
            QPushButton {
                background-color: #2b2b2b;
                color: #e0e0e0;
                font-weight: bold;
                border: 1px solid #444;
                border-radius: 4px;
            }
            QPushButton:checked {
                background-color: #007acc;
                color: white;
                border: 1px solid #0098ff;
            }
        """)
        self.btn_toggle_mode.toggled.connect(self._on_mode_toggled)
        top_layout.addWidget(self.btn_toggle_mode)

        # Multi-video selector
        vid_hdr = QLabel("Select Evaluated Video / Sequence:")
        vid_hdr.setStyleSheet("font-size: 11px; font-weight: normal; color: #ccc; margin-top: 4px;")
        top_layout.addWidget(vid_hdr)

        self.combo_video = QComboBox()
        self.combo_video.setMinimumHeight(28)
        self.combo_video.currentIndexChanged.connect(self._on_video_combo_changed)
        top_layout.addWidget(self.combo_video)

        self.lbl_dataset_info = QLabel("No predictions loaded")
        self.lbl_dataset_info.setStyleSheet("color: #888; font-size: 11px;")
        self.lbl_dataset_info.setWordWrap(True)
        top_layout.addWidget(self.lbl_dataset_info)

        layout.addWidget(top_box)

        # 2. Live Metrics Card
        self.metrics_box = QGroupBox("Live Metrics")
        self.metrics_box.setStyleSheet("QGroupBox { font-weight: bold; }")
        m_layout = QGridLayout(self.metrics_box)
        m_layout.setContentsMargins(6, 6, 6, 6)

        self.lbl_tp = QLabel("TP: <b>0</b>")
        self.lbl_tp.setStyleSheet("color: #00e5ff;")
        self.lbl_fp = QLabel("FP: <b>0</b>")
        self.lbl_fp.setStyleSheet("color: #ff334b;")
        self.lbl_fn = QLabel("FN: <b>0</b>")
        self.lbl_fn.setStyleSheet("color: #ff9900;")

        self.lbl_precision = QLabel("Precision: <b>0.0%</b>")
        self.lbl_recall = QLabel("Recall: <b>0.0%</b>")
        self.lbl_f1 = QLabel("F1: <b>0.00</b>")

        m_layout.addWidget(self.lbl_tp, 0, 0)
        m_layout.addWidget(self.lbl_fp, 0, 1)
        m_layout.addWidget(self.lbl_fn, 0, 2)
        m_layout.addWidget(self.lbl_precision, 1, 0)
        m_layout.addWidget(self.lbl_recall, 1, 1)
        m_layout.addWidget(self.lbl_f1, 1, 2)

        layout.addWidget(self.metrics_box)

        # 3. Confidence & IoU Sliders
        thr_box = QGroupBox("Thresholds (Real-time)")
        thr_box.setStyleSheet("QGroupBox { font-weight: bold; }")
        thr_layout = QVBoxLayout(thr_box)

        # Confidence Score
        conf_hdr = QHBoxLayout()
        conf_hdr.addWidget(QLabel("Confidence Score:"))
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.0, 1.0)
        self.spin_conf.setSingleStep(0.05)
        self.spin_conf.setValue(0.50)
        self.spin_conf.setDecimals(2)
        self.spin_conf.setFixedWidth(65)
        conf_hdr.addWidget(self.spin_conf)
        thr_layout.addLayout(conf_hdr)

        self.slider_conf = QSlider(Qt.Horizontal)
        self.slider_conf.setRange(0, 100)
        self.slider_conf.setValue(50)
        self.slider_conf.valueChanged.connect(self._on_conf_slider_changed)
        self.spin_conf.valueChanged.connect(self._on_conf_spin_changed)
        thr_layout.addWidget(self.slider_conf)

        # IoU Threshold
        iou_hdr = QHBoxLayout()
        iou_hdr.addWidget(QLabel("Match IoU Thr:"))
        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setRange(0.1, 0.95)
        self.spin_iou.setSingleStep(0.05)
        self.spin_iou.setValue(0.50)
        self.spin_iou.setDecimals(2)
        self.spin_iou.setFixedWidth(65)
        iou_hdr.addWidget(self.spin_iou)
        thr_layout.addLayout(iou_hdr)

        self.slider_iou = QSlider(Qt.Horizontal)
        self.slider_iou.setRange(10, 95)
        self.slider_iou.setValue(50)
        self.slider_iou.valueChanged.connect(self._on_iou_slider_changed)
        self.spin_iou.valueChanged.connect(self._on_iou_spin_changed)
        thr_layout.addWidget(self.slider_iou)

        layout.addWidget(thr_box)

        # 4. View Filter
        filter_box = QGroupBox("Filter View")
        filter_box.setStyleSheet("QGroupBox { font-weight: bold; }")
        flt_layout = QVBoxLayout(filter_box)

        self.combo_filter = QComboBox()
        self.combo_filter.addItems([
            "Show All (GT + Predictions)",
            "Only Errors (FP + FN)",
            "Only False Positives (FP)",
            "Only False Negatives (FN)",
            "Only True Positives (TP)"
        ])
        self.combo_filter.currentIndexChanged.connect(self._on_filter_changed)
        flt_layout.addWidget(self.combo_filter)

        # Color legend
        legend = QLabel(
            '<div style="font-size:11px; line-height: 1.6;">'
            '<span style="color:#00e5ff;">■ Cyan:</span> <b>True Positive (TP)</b><br>'
            '<span style="color:#ff334b;">■ Red:</span> <b>False Positive (FP)</b><br>'
            '<span style="color:#ff9900;">■ Orange:</span> <b>False Negative (FN / Missed)</b><br>'
            '<span style="color:#00ff78;">■ Green:</span> <b>Ground Truth (GT)</b>'
            '</div>'
        )
        flt_layout.addWidget(legend)

        layout.addWidget(filter_box)

        # 5. Error Navigation (Jumping)
        nav_box = QGroupBox("Error Jump Navigation")
        nav_box.setStyleSheet("QGroupBox { font-weight: bold; }")
        nav_layout = QVBoxLayout(nav_box)

        btn_jump_row = QHBoxLayout()
        self.btn_prev_err = QPushButton("◄ Prev Error [Ctrl+[]")
        self.btn_prev_err.clicked.connect(self.jump_prev_error)
        self.btn_next_err = QPushButton("Next Error ► [Ctrl+]]")
        self.btn_next_err.clicked.connect(self.jump_next_error)
        btn_jump_row.addWidget(self.btn_prev_err)
        btn_jump_row.addWidget(self.btn_next_err)
        nav_layout.addLayout(btn_jump_row)

        self.lbl_error_count = QLabel("Error Frames: 0")
        self.lbl_error_count.setStyleSheet("color: #aaa; font-size: 11px;")
        nav_layout.addWidget(self.lbl_error_count)

        self.list_error_frames = QListWidget()
        self.list_error_frames.setMaximumHeight(120)
        self.list_error_frames.itemClicked.connect(self._on_error_item_clicked)
        nav_layout.addWidget(self.list_error_frames)

        layout.addWidget(nav_box)

        # 6. Action Buttons
        btn_load = QPushButton("📁 Load Prediction (.txt / .json)")
        btn_load.clicked.connect(self.load_predictions_requested.emit)
        layout.addWidget(btn_load)

        self.btn_promote_fp = QPushButton("✨ Convert Selected FP to GT")
        self.btn_promote_fp.setStyleSheet("background-color: #2e5b38; color: white; font-weight: bold;")
        self.btn_promote_fp.clicked.connect(self._on_promote_fp_clicked)
        layout.addWidget(self.btn_promote_fp)

        self.btn_save_gt = QPushButton("💾 Save Changes to Ground Truth")
        self.btn_save_gt.setStyleSheet("""
            QPushButton {
                background-color: #1b5e20;
                color: #ffffff;
                font-weight: bold;
                font-size: 12px;
                padding: 6px;
                border-radius: 4px;
                border: 1px solid #2e7d32;
            }
            QPushButton:hover {
                background-color: #2e7d32;
            }
        """)
        self.btn_save_gt.clicked.connect(self.save_ground_truth_requested.emit)
        layout.addWidget(self.btn_save_gt)

        layout.addStretch()
        self.setWidget(container)

    def _setup_shortcuts(self):
        # Shortcuts for jumping between errors
        self.shortcut_prev = QShortcut(QKeySequence("Ctrl+["), self)
        self.shortcut_prev.activated.connect(self.jump_prev_error)
        self.shortcut_next = QShortcut(QKeySequence("Ctrl+]"), self)
        self.shortcut_next.activated.connect(self.jump_next_error)

    def _on_video_combo_changed(self, idx):
        if idx >= 0:
            vid_name = self.combo_video.itemData(idx) or self.combo_video.currentText()
            if vid_name and vid_name != self.current_video_name:
                self.video_selected.emit(vid_name)

    def set_dataset_videos(self, video_names, current_video=None):
        """Populates the video selector with list of evaluated videos."""
        self.dataset_videos = list(video_names)
        self.combo_video.blockSignals(True)
        self.combo_video.clear()
        for v in self.dataset_videos:
            self.combo_video.addItem(v, v)
        if current_video and current_video in self.dataset_videos:
            idx = self.dataset_videos.index(current_video)
            self.combo_video.setCurrentIndex(idx)
            self.current_video_name = current_video
        elif self.dataset_videos:
            self.current_video_name = self.dataset_videos[0]
            self.combo_video.setCurrentIndex(0)
        self.combo_video.blockSignals(False)

    def _on_mode_toggled(self, checked):
        if checked:
            self.btn_toggle_mode.setText("👁️ Evaluation View ACTIVE")
        else:
            self.btn_toggle_mode.setText("👁️ Enable Evaluation View")
        self.eval_mode_toggled.emit(checked)

    def _on_conf_slider_changed(self, val):
        conf = val / 100.0
        self.spin_conf.blockSignals(True)
        self.spin_conf.setValue(conf)
        self.spin_conf.blockSignals(False)
        self.conf_threshold_changed.emit(conf)

    def _on_conf_spin_changed(self, val):
        self.slider_conf.blockSignals(True)
        self.slider_conf.setValue(int(round(val * 100)))
        self.slider_conf.blockSignals(False)
        self.conf_threshold_changed.emit(val)

    def _on_iou_slider_changed(self, val):
        iou = val / 100.0
        self.spin_iou.blockSignals(True)
        self.spin_iou.setValue(iou)
        self.spin_iou.blockSignals(False)
        self.iou_threshold_changed.emit(iou)

    def _on_iou_spin_changed(self, val):
        self.slider_iou.blockSignals(True)
        self.slider_iou.setValue(int(round(val * 100)))
        self.slider_iou.blockSignals(False)
        self.iou_threshold_changed.emit(val)

    def _on_filter_changed(self, idx):
        filters = ["ALL", "ERRORS", "FP", "FN", "TP"]
        f_val = filters[idx] if 0 <= idx < len(filters) else "ALL"
        self.filter_changed.emit(f_val)

    def _on_error_item_clicked(self, item):
        try:
            txt = item.text()
            frame_num = int(txt.split()[1]) - 1  # 1-indexed display to 0-indexed frame
            self.jump_to_frame_requested.emit(frame_num)
        except Exception:
            pass

    def _on_promote_fp_clicked(self):
        if hasattr(self.main_window, 'canvas') and getattr(self.main_window.canvas, 'eval_selected_prediction', None):
            pred = self.main_window.canvas.eval_selected_prediction
            curr_frame = getattr(self.main_window, 'current_frame', 0)
            self.promote_fp_requested.emit(pred, curr_frame)
        else:
            QMessageBox.information(
                self,
                "Promote to Ground Truth",
                "Please click or select a False Positive (Red) box on the canvas first to promote it to Ground Truth."
            )

    def set_evaluation_info(self, video_name, total_predictions, default_conf=0.5):
        self.current_video_name = video_name
        self.lbl_dataset_info.setText(
            f"<b>Video:</b> {video_name}<br><b>Detections:</b> {total_predictions} boxes"
        )
        self.spin_conf.setValue(default_conf)

    def update_metrics_display(self, tp, fp, fn):
        self.lbl_tp.setText(f"TP: <b>{tp}</b>")
        self.lbl_fp.setText(f"FP: <b>{fp}</b>")
        self.lbl_fn.setText(f"FN: <b>{fn}</b>")

        precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall) / 100.0) if (precision + recall) > 0 else 0.0

        self.lbl_precision.setText(f"Precision: <b>{precision:.1f}%</b>")
        self.lbl_recall.setText(f"Recall: <b>{recall:.1f}%</b>")
        self.lbl_f1.setText(f"F1: <b>{f1:.2f}</b>")

    def update_error_frames_list(self, error_frames_set):
        self.error_frames = sorted(list(error_frames_set))
        self.lbl_error_count.setText(f"Error Frames: <b>{len(self.error_frames)}</b>")

        self.list_error_frames.clear()
        for f_idx in self.error_frames:
            item = QListWidgetItem(f"Frame {f_idx + 1}")
            item.setData(Qt.UserRole, f_idx)
            self.list_error_frames.addItem(item)

    def jump_next_error(self):
        if not self.error_frames:
            return
        curr_frame = getattr(self.main_window, 'current_frame', 0)
        # Find first error frame strictly after current frame
        next_frames = [f for f in self.error_frames if f > curr_frame]
        target = next_frames[0] if next_frames else self.error_frames[0]
        self.jump_to_frame_requested.emit(target)

    def jump_prev_error(self):
        if not self.error_frames:
            return
        curr_frame = getattr(self.main_window, 'current_frame', 0)
        # Find last error frame strictly before current frame
        prev_frames = [f for f in self.error_frames if f < curr_frame]
        target = prev_frames[-1] if prev_frames else self.error_frames[-1]
        self.jump_to_frame_requested.emit(target)
