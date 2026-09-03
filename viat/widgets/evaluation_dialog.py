"""
Comprehensive Evaluation Dialog for VIAT

Supports:
- Multi-format Ground Truth & Predictions (Raya format, Image datasets, YOLO data.yaml).
- Interactive Class Mapping & Assignment.
- Class Merging Impact Simulator (Before vs After Merge mAP comparison).
- Per-Class & Per-Size (Small, Medium, Large) Detailed Metrics & Matplotlib Charts.
- Visual Error Inspector ("SHOW") with GT/TP/FP/FN Overlays & Timeline Sync.
- Non-blocking Background Worker Thread.
"""
import os
import sys
import glob
import numpy as np
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
    QMenu,
    QGroupBox,
    QProgressBar,
    QPlainTextEdit,
    QMessageBox,
    QFileDialog,
    QSplitter,
    QWidget,
    QFrame,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QComboBox,
    QScrollArea,
    QInputDialog,
    QListWidget,
    QListWidgetItem
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QIcon, QFont, QTextCursor, QPixmap

from viat.evaluation.utils.yaml_parser import parse_yolo_yaml, scan_dataset_classes, scan_dataset_videos
from viat.evaluation.utils.class_merger import DetailedAnalyticsEngine
from viat.evaluation.utils.advanced_diagnostics import AdvancedDiagnosticsEngine
from viat.evaluation.visualization.visual_inspector import VisualInspectorWidget
from viat.evaluation.inference.model_runner import ModelRunner


class VideoMultiSelectDialog(QDialog):
    """Modern checkable multi-select dialog for video sequences."""

    def __init__(self, all_videos, selected_videos=None, title="Select Videos for Evaluation", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(460, 520)
        self.setStyleSheet("""
            QDialog { background-color: #1e1e1e; color: #ffffff; }
            QLabel { color: #e0e0e0; font-size: 12px; }
            QLineEdit {
                background-color: #2b2b2b; color: #ffffff; border: 1px solid #444444;
                border-radius: 4px; padding: 6px 10px; font-size: 12px;
            }
            QLineEdit:focus { border: 1px solid #409EFF; }
            QListWidget {
                background-color: #252525; border: 1px solid #3d3d3d;
                border-radius: 6px; padding: 4px;
            }
            QListWidget::item {
                padding: 6px 8px; color: #ffffff; border-radius: 4px;
            }
            QListWidget::item:hover { background-color: rgba(64, 158, 255, 0.15); }
            QPushButton {
                background-color: #333333; color: #ffffff; border: 1px solid #555555;
                border-radius: 4px; padding: 6px 12px; font-size: 11px;
            }
            QPushButton:hover { background-color: #444444; border-color: #409EFF; }
        """)

        layout = QVBoxLayout(self)

        lbl_desc = QLabel(
            "<b>Select Target Videos:</b><br>"
            "<span style='color:#aaaaaa; font-size:11px;'>By default, all detected videos are selected. Deselect any videos you wish to exclude.</span>"
        )
        lbl_desc.setWordWrap(True)
        layout.addWidget(lbl_desc)

        # Search bar
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("🔍 Search / Filter videos...")
        self.search_edit.textChanged.connect(self.filter_items)
        layout.addWidget(self.search_edit)

        # Action buttons
        btn_bar = QHBoxLayout()
        btn_all = QPushButton("☑ Select All")
        btn_all.clicked.connect(self.select_all)
        btn_none = QPushButton("☐ Deselect All")
        btn_none.clicked.connect(self.deselect_all)
        btn_invert = QPushButton("🔀 Invert")
        btn_invert.clicked.connect(self.invert_selection)

        self.lbl_count = QLabel("")
        self.lbl_count.setStyleSheet("color: #409EFF; font-weight: bold; font-size: 11px;")

        btn_bar.addWidget(btn_all)
        btn_bar.addWidget(btn_none)
        btn_bar.addWidget(btn_invert)
        btn_bar.addStretch()
        btn_bar.addWidget(self.lbl_count)
        layout.addLayout(btn_bar)

        # Checkable list
        self.list_widget = QListWidget()
        self.all_videos = list(all_videos) if all_videos else []

        if selected_videos is None:
            self.selected_set = set(self.all_videos)
        else:
            self.selected_set = set(selected_videos)

        for vid in self.all_videos:
            item = QListWidgetItem(vid, self.list_widget)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            if vid in self.selected_set or (selected_videos is None):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)

        self.list_widget.itemChanged.connect(self.update_count)
        layout.addWidget(self.list_widget)
        self.update_count()

        # Bottom buttons
        bottom_box = QHBoxLayout()
        bottom_box.addStretch()
        btn_ok = QPushButton("Apply Selection")
        btn_ok.setStyleSheet("background-color: #007acc; color: white; font-weight: bold; padding: 7px 16px;")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        bottom_box.addWidget(btn_ok)
        bottom_box.addWidget(btn_cancel)
        layout.addLayout(bottom_box)

    def select_all(self):
        self.list_widget.blockSignals(True)
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.Checked)
        self.list_widget.blockSignals(False)
        self.update_count()

    def deselect_all(self):
        self.list_widget.blockSignals(True)
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.Unchecked)
        self.list_widget.blockSignals(False)
        self.update_count()

    def invert_selection(self):
        self.list_widget.blockSignals(True)
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked)
        self.list_widget.blockSignals(False)
        self.update_count()

    def filter_items(self, text):
        query = text.lower().strip()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            item.setHidden(bool(query and query not in item.text().lower()))

    def update_count(self):
        sel = len(self.get_selected_videos())
        tot = self.list_widget.count()
        self.lbl_count.setText(f"{sel}/{tot} Selected")

    def get_selected_videos(self):
        selected = []
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.checkState() == Qt.Checked:
                selected.append(item.text())
        return selected




class StreamRedirector(QObject):
    """Redirects stdout/stderr to Qt signal for live log streaming."""
    text_written = pyqtSignal(str)

    def __init__(self, original_stream=None):
        super().__init__()
        self.original_stream = original_stream

    def write(self, text):
        if text:
            self.text_written.emit(str(text))
            if self.original_stream:
                try:
                    self.original_stream.write(text)
                except Exception:
                    pass

    def flush(self):
        if self.original_stream:
            try:
                self.original_stream.flush()
            except Exception:
                pass


class EvaluateWorkerThread(QThread):
    """Background worker thread executing evaluation & generating analytics."""
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str, dict)

    def __init__(self, evaluate_instance, class_map=None, merge_groups=None, results_dir=None):
        super().__init__()
        self.evaluate_instance = evaluate_instance
        self.class_map = class_map or {}
        self.merge_groups = merge_groups or {}
        self.results_dir = results_dir or "/tmp/eval_results"

    def run(self):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirector = StreamRedirector(old_stdout)
        redirector.text_written.connect(self.log_signal.emit)
        sys.stdout = redirector
        sys.stderr = redirector

        results = {}
        try:
            self.log_signal.emit("[INFO] Starting Evaluation Pipeline...\n")
            self.evaluate_instance.evaluate_all()
            self.log_signal.emit("\n[INFO] Generating Per-Class & Size Analytics...\n")

            # Initialize defaults
            results['unmerged_mAP'] = 0.0
            results['merged_mAP'] = 0.0
            results['class_metrics'] = {}

            results['size_metrics'] = {
                'Small': {'ap50': 0.0},
                'Medium': {'ap50': 0.0},
                'Large': {'ap50': 0.0}
            }

            video_metrics = []
            det_dir = self.evaluate_instance.det_path[0] if hasattr(self.evaluate_instance, 'det_path') and self.evaluate_instance.det_path else ""
            eval_result_dir = os.path.join(det_dir, 'evaluation_result') if det_dir else ""
            eval_det_csv = os.path.join(eval_result_dir, 'eval_detection.csv')

            if os.path.exists(eval_det_csv):
                self.results_dir = os.path.join(eval_result_dir, 'plots')
                import csv
                with open(eval_det_csv, 'r') as f:
                    reader = csv.reader(f)
                    headers = next(reader, [])
                    for row in reader:
                        if not row: continue
                        name = row[0]
                        metrics = dict(zip(headers[1:], row[1:]))

                        # Robust case-insensitive metric extraction
                        def _get_val(keys, default=0.0):
                            for k in keys:
                                for mk, mv in metrics.items():
                                    if mk.lower().replace('@', '').replace('_', '').replace(' ', '') == k.lower().replace('@', '').replace('_', '').replace(' ', ''):
                                        try:
                                            return float(mv) if mv not in (None, '', '-') else default
                                        except (ValueError, TypeError):
                                            pass
                            return default

                        ap50 = _get_val(['ap50', 'map50', 'map050', 'ap'])
                        f1 = _get_val(['f1', 'f1score', 'fscore'])
                        p_val = _get_val(['precision', 'prec', 'p'])
                        r_val = _get_val(['recall', 'rec', 'r'])

                        if name == 'all_video':
                            results['unmerged_mAP'] = ap50
                            results['merged_mAP'] = ap50
                            continue
                        elif name.endswith('_all_video'):
                            continue

                        video_metrics.append({
                            'name': name,
                            'video': name,
                            'metrics': metrics,
                            'ap50': ap50,
                            'f1': f1,
                            'precision': p_val,
                            'recall': r_val
                        })

            # Real per-class & per-size metrics written by the engine
            per_class_file = os.path.join(eval_result_dir, 'per_class_metrics.json') if eval_result_dir else ""
            if per_class_file and os.path.exists(per_class_file):
                try:
                    import json as _json
                    with open(per_class_file, 'r') as f:
                        pc_data = _json.load(f)
                    results['class_metrics'] = {
                        c_name: {
                            'ap50': float(c.get('AP50') or 0),
                            'ap': float(c.get('AP') or 0),
                            'tp': int(c.get('TP') or 0),
                            'fp': int(c.get('FP') or 0),
                            'fn': int(c.get('FN') or 0),
                        }
                        for c_name, c in pc_data.get('per_class_metrics', {}).items()
                    }
                    size_by_prefix = {}
                    for s_name, s_val in pc_data.get('per_size_metrics', {}).items():
                        prefix = str(s_name).split(' ')[0]
                        if s_val is not None:
                            try:
                                size_by_prefix[prefix] = float(s_val)
                            except (TypeError, ValueError):
                                pass
                    if size_by_prefix:
                        results['size_metrics'] = {
                            'Small': {'ap50': size_by_prefix.get('Small', 0.0)},
                            'Medium': {'ap50': size_by_prefix.get('Medium', 0.0)},
                            'Large': {'ap50': size_by_prefix.get('Large', 0.0)},
                        }
                except Exception as parse_err:
                    self.log_signal.emit(f"\n[WARNING] Could not parse per_class_metrics.json: {parse_err}\n")

            # Graceful fallback only if the engine produced no per-class data
            if not results['class_metrics']:
                cats = [c for c in (self.evaluate_instance.target_classes or ['object'])
                        if c != '__IGNORE__'] or ['object']
                for cat in cats:
                    results['class_metrics'][cat] = {'ap50': 0, 'ap': 0, 'tp': 0, 'fp': 0, 'fn': 0}

            results['video_metrics'] = video_metrics

            # Generate matplotlib plot images into results_dir
            os.makedirs(self.results_dir, exist_ok=True)
            bar_path = os.path.join(self.results_dir, "class_ap50_bar.png")
            size_path = os.path.join(self.results_dir, "size_breakdown_bar.png")
            merge_path = os.path.join(self.results_dir, "class_merge_comparison.png")
            pr_path = os.path.join(self.results_dir, "diag_pr_curves.png")
            f1_path = os.path.join(self.results_dir, "diag_f1_vs_confidence.png")
            conf_dist_path = os.path.join(self.results_dir, "diag_confidence_distribution.png")
            err_breakdown_path = os.path.join(self.results_dir, "diag_error_breakdown.png")
            cm_path = os.path.join(self.results_dir, "diag_confusion_matrix.png")
            calib_path = os.path.join(self.results_dir, "diag_calibration_ece.png")
            iou_path = os.path.join(self.results_dir, "diag_iou_distribution.png")
            ar_path = os.path.join(self.results_dir, "diag_aspect_ratio_bias.png")
            per_video_path = os.path.join(self.results_dir, "diag_per_video_comparison.png")
            track_path = os.path.join(self.results_dir, "diag_tracking_taxonomy.png")
            spatial_path = os.path.join(self.results_dir, "diag_spatial_error_map.png")

            try:
                DetailedAnalyticsEngine.generate_per_class_bar_chart(results['class_metrics'], bar_path)
                DetailedAnalyticsEngine.generate_size_breakdown_chart(results['size_metrics'], size_path)
                if self.merge_groups:
                    DetailedAnalyticsEngine.generate_class_merge_comparison_chart(results['unmerged_mAP'], results['merged_mAP'], merge_path)
            except Exception as chart_err:
                self.log_signal.emit(f"\n[WARNING] Could not generate basic analytics charts: {str(chart_err)}\n")

            # Generate Advanced Diagnostic Plots from the engine's real per-detection data
            try:
                import numpy as np
                diag_file = os.path.join(eval_result_dir, 'diagnostics.json') if eval_result_dir else ""
                diag = None
                if diag_file and os.path.exists(diag_file):
                    try:
                        import json as _json
                        with open(diag_file, 'r') as f:
                            diag = _json.load(f)
                    except Exception as diag_parse_err:
                        self.log_signal.emit(f"\n[WARNING] Could not parse diagnostics.json: {str(diag_parse_err)}\n")

                if diag:
                    classes = diag.get('classes', [])
                    cm = np.array(diag.get('confusion', []), dtype=float)
                    if len(classes) and cm.size:
                        AdvancedDiagnosticsEngine.generate_confusion_matrix_plot(
                            cm=cm,
                            class_names=classes,
                            save_path=cm_path
                        )
                    calib = diag.get('calibration', {})
                    if calib.get('confidences'):
                        AdvancedDiagnosticsEngine.generate_calibration_plot(
                            confidences=np.array(calib['confidences']),
                            precisions=np.array(calib['precisions']),
                            recalls=np.array(calib['recalls']),
                            ece_score=calib.get('ece_score', 0.0),
                            optimal_thr=calib.get('optimal_thr', 0.5),
                            save_path=calib_path
                        )
                    if diag.get('per_class_curves'):
                        AdvancedDiagnosticsEngine.generate_pr_curves_plot(
                            per_class_curves=diag['per_class_curves'],
                            save_path=pr_path
                        )
                        AdvancedDiagnosticsEngine.generate_f1_confidence_plot(
                            per_class_curves=diag['per_class_curves'],
                            save_path=f1_path
                        )
                    if diag.get('conf_tp') or diag.get('conf_fp'):
                        AdvancedDiagnosticsEngine.generate_confidence_distribution_plot(
                            conf_tp=diag.get('conf_tp', []),
                            conf_fp=diag.get('conf_fp', []),
                            save_path=conf_dist_path
                        )
                    if diag.get('error_breakdown'):
                        AdvancedDiagnosticsEngine.generate_error_breakdown_plot(
                            error_breakdown=diag['error_breakdown'],
                            save_path=err_breakdown_path
                        )
                    if diag.get('ious'):
                        AdvancedDiagnosticsEngine.generate_iou_distribution_plot(
                            ious=np.array(diag['ious']),
                            save_path=iou_path
                        )
                    ar = diag.get('aspect_ratio', {})
                    if ar.get('ratios'):
                        AdvancedDiagnosticsEngine.generate_aspect_ratio_plot(
                            aspect_ratios=ar['ratios'],
                            error_rates=ar['error_rates'],
                            save_path=ar_path
                        )
                    fp_coords = diag.get('fp_coords', [])
                    fn_coords = diag.get('fn_coords', [])
                    if fp_coords or fn_coords:
                        AdvancedDiagnosticsEngine.generate_spatial_error_heatmap(
                            fp_coords=fp_coords,
                            fn_coords=fn_coords,
                            canvas_size=tuple(diag.get('canvas_size', (1920, 1080))),
                            save_path=spatial_path
                        )
                    if results.get('video_metrics'):
                        AdvancedDiagnosticsEngine.generate_per_video_comparison_plot(
                            video_metrics=results['video_metrics'],
                            save_path=per_video_path
                        )
                else:
                    self.log_signal.emit("\n[WARNING] No diagnostics.json produced by the engine; advanced diagnostic plots were skipped.\n")
            except Exception as diag_err:
                self.log_signal.emit(f"\n[WARNING] Could not generate advanced diagnostic charts: {str(diag_err)}\n")

            results['bar_path'] = bar_path if os.path.exists(bar_path) else None
            results['size_path'] = size_path if os.path.exists(size_path) else None
            results['merge_path'] = merge_path if (self.merge_groups and os.path.exists(merge_path)) else None
            results['pr_path'] = pr_path if os.path.exists(pr_path) else None
            results['f1_path'] = f1_path if os.path.exists(f1_path) else None
            results['conf_dist_path'] = conf_dist_path if os.path.exists(conf_dist_path) else None
            results['err_breakdown_path'] = err_breakdown_path if os.path.exists(err_breakdown_path) else None
            results['cm_path'] = cm_path if os.path.exists(cm_path) else None
            results['calib_path'] = calib_path if os.path.exists(calib_path) else None
            results['iou_path'] = iou_path if os.path.exists(iou_path) else None
            results['ar_path'] = ar_path if os.path.exists(ar_path) else None
            results['per_video_path'] = per_video_path if os.path.exists(per_video_path) else None
            results['track_path'] = track_path if os.path.exists(track_path) else None
            results['spatial_path'] = spatial_path if os.path.exists(spatial_path) else None
            results['diagnostics_data'] = diag

            self.log_signal.emit("\n[SUCCESS] Evaluation & Analytics Finished Successfully!\n")
            self.finished_signal.emit(True, "Evaluation completed!", results)


        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            self.log_signal.emit(f"\n[ERROR] Evaluation failed:\n{err_msg}\n")
            self.finished_signal.emit(False, str(e), {})
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


class SingleSetWidget(QFrame):
    """Widget managing GT path, DT path, YAML config, and Category tag."""
    remove_requested = pyqtSignal(object)

    def __init__(self, index=1, parent=None):
        super().__init__(parent)
        self.index = index
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("""
            SingleSetWidget {
                background-color: rgba(255, 255, 255, 0.03);
                border: 1px solid rgba(255, 255, 255, 0.1);
                border-radius: 6px;
                padding: 6px;
            }
        """)
        self.setup_ui()

    def setup_ui(self):
        layout = QGridLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header_label = QLabel(f"<b>Dataset Bundle #{self.index}</b>")
        layout.addWidget(header_label, 0, 0)

        if self.index > 1:
            self.btn_remove = QPushButton("✕ Remove")
            self.btn_remove.setFixedWidth(80)
            self.btn_remove.setStyleSheet("color: #ff5555; background: transparent; border: none;")
            self.btn_remove.clicked.connect(lambda: self.remove_requested.emit(self))
            layout.addWidget(self.btn_remove, 0, 2, Qt.AlignRight)

        # GT Row
        layout.addWidget(QLabel("Ground Truth (GT):"), 1, 0)
        self.edit_gt = QLineEdit()
        self.edit_gt.setPlaceholderText("Select GT folder (Raya, Image dataset, or YOLO)...")
        layout.addWidget(self.edit_gt, 1, 1)

        gt_btns = QHBoxLayout()
        self.btn_gt = QPushButton("Browse...")
        self.btn_gt.clicked.connect(self.browse_gt)
        gt_btns.addWidget(self.btn_gt)

        self.btn_active_gt = QPushButton("Use Active VIAT")
        self.btn_active_gt.clicked.connect(self.use_active_project_gt)
        gt_btns.addWidget(self.btn_active_gt)
        layout.addLayout(gt_btns, 1, 2)

        # YOLO YAML Row
        layout.addWidget(QLabel("YOLO data.yaml (Optional):"), 2, 0)
        self.edit_yaml = QLineEdit()
        self.edit_yaml.setPlaceholderText("Select data.yaml for YOLO class name mapping...")
        layout.addWidget(self.edit_yaml, 2, 1)
        self.btn_yaml = QPushButton("Load YAML...")
        self.btn_yaml.clicked.connect(self.browse_yaml)
        layout.addWidget(self.btn_yaml, 2, 2)

        # DT Row Mode Toggle
        self.chk_use_weights = QCheckBox("Auto-run Model Inference from Weights (.pt / .onnx / .engine)")
        self.chk_use_weights.toggled.connect(self.toggle_weights_mode)
        layout.addWidget(self.chk_use_weights, 3, 0, 1, 3)

        # Standard DT Folder Row
        self.lbl_dt = QLabel("Detections (DT):")
        layout.addWidget(self.lbl_dt, 4, 0)
        self.edit_dt = QLineEdit()
        self.edit_dt.setPlaceholderText("Select Detection output folder...")
        layout.addWidget(self.edit_dt, 4, 1)
        self.btn_dt = QPushButton("Browse...")
        self.btn_dt.clicked.connect(self.browse_dt)
        layout.addWidget(self.btn_dt, 4, 2)

        # Auto Weights Fields (Hidden by default)
        self.lbl_weights = QLabel("Model Weights (.pt):")
        layout.addWidget(self.lbl_weights, 5, 0)
        self.edit_weights = QLineEdit()
        self.edit_weights.setPlaceholderText("Select model weights file (.pt, .onnx, .engine)...")
        layout.addWidget(self.edit_weights, 5, 1)
        self.btn_weights = QPushButton("Browse Weights...")
        self.btn_weights.clicked.connect(self.browse_weights)
        layout.addWidget(self.btn_weights, 5, 2)

        self.lbl_media = QLabel("Media Source:")
        self.lbl_media.setVisible(False)
        layout.addWidget(self.lbl_media, 6, 0)
        self.edit_media = QLineEdit()
        self.edit_media.setPlaceholderText("Select video file OR dataset folder (images/videos)...")
        self.edit_media.setVisible(False)
        layout.addWidget(self.edit_media, 6, 1)

        media_btns = QHBoxLayout()
        self.btn_media = QPushButton("Browse Media...")
        self.btn_media.clicked.connect(self.browse_media)
        media_btns.addWidget(self.btn_media)
        self.btn_active_media = QPushButton("Use Active VIAT Video")
        self.btn_active_media.clicked.connect(self.use_active_viat_media)
        media_btns.addWidget(self.btn_active_media)
        layout.addLayout(media_btns, 6, 2)

        self.toggle_weights_mode(False)

        # Category Tag
        layout.addWidget(QLabel("Category Tag:"), 7, 0)
        self.edit_cat = QLineEdit("All" if self.index == 1 else f"Category_{self.index}")
        layout.addWidget(self.edit_cat, 7, 1, 1, 2)

    def toggle_weights_mode(self, checked):
        # Hide standard DT folder if weights mode is checked
        self.lbl_dt.setVisible(not checked)
        self.edit_dt.setVisible(not checked)
        self.btn_dt.setVisible(not checked)

        self.lbl_weights.setVisible(checked)
        self.edit_weights.setVisible(checked)
        self.btn_weights.setVisible(checked)

        self.lbl_media.setVisible(checked)
        self.edit_media.setVisible(checked)
        self.btn_media.setVisible(checked)
        self.btn_active_media.setVisible(checked)

    def browse_weights(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model Weights File", "", "Model Weights (*.pt *.onnx *.engine *.tflite)")
        if file_path:
            self.edit_weights.setText(file_path)

    def browse_media(self):
        menu = QMenu(self)
        action_file = menu.addAction("Select Single Video File...")
        action_dir = menu.addAction("Select Dataset Folder (Images or Videos)...")
        
        # Position menu under the button
        action = menu.exec_(self.btn_media.mapToGlobal(self.btn_media.rect().bottomLeft()))
        
        if action == action_file:
            file_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mkv *.mov)")
            if file_path:
                self.edit_media.setText(file_path)
        elif action == action_dir:
            dir_path = QFileDialog.getExistingDirectory(self, "Select Dataset Folder (Images or Videos)")
            if dir_path:
                self.edit_media.setText(dir_path)

    def use_active_viat_media(self):
        main_win = self.parent()
        if main_win and hasattr(main_win, 'video_path') and main_win.video_path:
            self.edit_media.setText(main_win.video_path)
        else:
            self.edit_media.setText("[ACTIVE_VIAT_VIDEO]")


    def browse_gt(self):
        path = QFileDialog.getExistingDirectory(self, f"Select GT Folder (Bundle #{self.index})")
        if path:
            self.edit_gt.setText(path)
            # Auto-check for data.yaml inside GT folder
            yaml_match = glob.glob(os.path.join(path, "*.yaml")) + glob.glob(os.path.join(path, "*.yml"))
            if yaml_match:
                self.edit_yaml.setText(yaml_match[0])

    def browse_yaml(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select YOLO data.yaml", "", "YAML Files (*.yaml *.yml)")
        if file_path:
            self.edit_yaml.setText(file_path)

    def browse_dt(self):
        path = QFileDialog.getExistingDirectory(self, f"Select DT Folder (Bundle #{self.index})")
        if path:
            self.edit_dt.setText(path)

    def use_active_project_gt(self):
        self.edit_gt.setText("[ACTIVE_VIAT_PROJECT]")

    def get_gt_path(self):
        return self.edit_gt.text().strip()

    def get_yaml_path(self):
        return self.edit_yaml.text().strip()

    def get_dt_path(self):
        return self.edit_dt.text().strip()

    def get_category(self):
        return self.edit_cat.text().strip()

    def get_paths(self):
        """Returns tuple of (gt_path, dt_path, category) for this bundle."""
        return self.get_gt_path(), self.get_dt_path(), self.get_category()


class EvaluationDialog(QDialog):
    """Multi-Tab Advanced Evaluation Dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Model & Dataset Evaluation Suite")
        self.resize(1000, 800)
        self.setMinimumSize(850, 650)

        self.bundle_widgets = []
        self.eval_worker = None
        self.last_results_dir = "/tmp/eval_results"
        self.class_map = {}
        self.merge_groups = {}

        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)

        # Header Info
        header_frame = QFrame()
        header_frame.setStyleSheet("background-color: rgba(64, 158, 255, 0.1); border-radius: 6px; padding: 8px;")
        header_layout = QVBoxLayout(header_frame)
        title_lbl = QLabel("<b>VIAT Model Evaluation & Analytics Suite</b>")
        title_lbl.setStyleSheet("font-size: 14px; color: #409EFF;")
        desc_lbl = QLabel(
            "Benchmark Ground Truth vs Detections across COCO, MOT, and Center metrics.\n"
            "Features multi-format support (Raya, Image, YOLO data.yaml), Class Assignment/Remapping, Merge Impact Simulator, Per-Class/Size Analytics, and Interactive Error Visualizer (SHOW)."
        )
        desc_lbl.setWordWrap(True)
        header_layout.addWidget(title_lbl)
        header_layout.addWidget(desc_lbl)
        main_layout.addWidget(header_frame)

        # Main Tab Widget
        self.tabs = QTabWidget()

        # Tab 1: Datasets, Settings & Class Mapping
        self.tab_datasets = QWidget()
        self.setup_datasets_tab()
        self.tabs.addTab(self.tab_datasets, "📁 Setup & Metrics")

        # Tab 1.5: Video Categories
        self.tab_video_categories = QWidget()
        self.setup_video_categories_tab()
        self.tabs.addTab(self.tab_video_categories, "🎥 Video Categories")

        # Tab 2: Class Merging Simulator
        self.tab_merge = QWidget()
        self.setup_merge_tab()
        self.tabs.addTab(self.tab_merge, "🔀 Class Merge Simulator")

        # Tab 3: Analytics & Charts
        self.tab_analytics = QWidget()
        self.setup_analytics_tab()
        self.tabs.addTab(self.tab_analytics, "📊 Analytics & Plots")

        # Tab 4: Console Log
        self.tab_log = QWidget()
        self.setup_log_tab()
        self.tabs.addTab(self.tab_log, "💻 Console Log")

        self.tabs.currentChanged.connect(self.on_tab_changed)

        main_layout.addWidget(self.tabs)

        # Bottom Execution Bar
        bottom_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.setVisible(False)
        bottom_layout.addWidget(self.progress_bar)

        self.btn_open_results = QPushButton("📁 Open Results Folder")
        self.btn_open_results.setEnabled(False)
        self.btn_open_results.clicked.connect(self.open_results_folder)
        bottom_layout.addWidget(self.btn_open_results)

        self.btn_load_profile = QPushButton("📂 Load Profile")
        self.btn_load_profile.clicked.connect(self.load_profile)
        bottom_layout.addWidget(self.btn_load_profile)

        self.btn_save_profile = QPushButton("💾 Save Profile")
        self.btn_save_profile.clicked.connect(self.save_profile)
        bottom_layout.addWidget(self.btn_save_profile)

        bottom_layout.addStretch()

        btn_layout = QHBoxLayout()
        self.btn_evaluate = QPushButton("▶ Run Evaluation")
        self.btn_evaluate.setMinimumHeight(40)
        self.btn_evaluate.setStyleSheet("background-color: #007acc; color: white; font-weight: bold; font-size: 13px;")
        self.btn_evaluate.clicked.connect(self.start_evaluation)
        btn_layout.addWidget(self.btn_evaluate)
        
        self.btn_inspect_main = QPushButton("👁️ Inspect in Main Canvas")
        self.btn_inspect_main.setMinimumHeight(40)
        self.btn_inspect_main.setStyleSheet("background-color: #2e5b38; color: white; font-weight: bold; font-size: 13px;")
        self.btn_inspect_main.clicked.connect(self.inspect_in_main_canvas)
        btn_layout.addWidget(self.btn_inspect_main)
        
        main_layout.addLayout(btn_layout)

        self.btn_close = QPushButton("Close")
        self.btn_close.clicked.connect(self.close)
        bottom_layout.addWidget(self.btn_close)

        main_layout.addLayout(bottom_layout)

    def setup_datasets_tab(self):
        main_tab_layout = QVBoxLayout(self.tab_datasets)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)

        # Bundles Group
        bundle_group = QGroupBox("Dataset Bundles (Ground Truth & Detections)")
        self.bundle_layout = QVBoxLayout(bundle_group)
        self.bundles_container_layout = QVBoxLayout()
        self.bundle_layout.addLayout(self.bundles_container_layout)

        btn_add = QPushButton("+ Add Dataset Bundle")
        btn_add.clicked.connect(self.add_bundle_set)
        self.bundle_layout.addWidget(btn_add, 0, Qt.AlignLeft)
        layout.addWidget(bundle_group)

        # Metric Settings integrated here
        settings_group = QGroupBox("Evaluation Metrics & Settings")
        settings_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        set_layout = QGridLayout(settings_group)
        set_layout.setSpacing(10)

        # Thresholds row
        thr_layout = QHBoxLayout()
        thr_layout.addWidget(QLabel("Size Filter Threshold (%):"))
        self.spin_size_thr = QSpinBox()
        self.spin_size_thr.setRange(0, 100)
        thr_layout.addWidget(self.spin_size_thr)

        thr_layout.addWidget(QLabel("Quality Filter Threshold (%):"))
        self.spin_quality_thr = QSpinBox()
        self.spin_quality_thr.setRange(0, 100)
        thr_layout.addWidget(self.spin_quality_thr)

        thr_layout.addWidget(QLabel("Confidence Threshold:"))
        self.spin_conf_thr = QDoubleSpinBox()
        self.spin_conf_thr.setRange(0.01, 1.0)
        self.spin_conf_thr.setSingleStep(0.05)
        self.spin_conf_thr.setValue(0.25)
        thr_layout.addWidget(self.spin_conf_thr)

        thr_layout.addStretch()
        set_layout.addLayout(thr_layout, 0, 0, 1, 2)

        set_layout.addWidget(QLabel("<b>Select Metrics & Evaluation Engines:</b>"), 1, 0, 1, 2)

        # Modern high-contrast metric card stylesheet
        card_style = """
            QCheckBox {
                color: #ffffff;
                font-size: 11.5px;
                font-weight: bold;
                padding: 10px 14px;
                background-color: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.16);
                border-radius: 6px;
                min-height: 42px;
            }
            QCheckBox:hover {
                background-color: rgba(64, 158, 255, 0.14);
                border: 1px solid #409EFF;
            }
            QCheckBox::indicator {
                width: 20px;
                height: 20px;
                border-radius: 4px;
                border: 2px solid #888888;
                background-color: #222222;
            }
            QCheckBox::indicator:hover {
                border-color: #409EFF;
            }
            QCheckBox::indicator:checked {
                background-color: #007acc;
                border-color: #409EFF;
            }
        """

        self.chk_detection = QCheckBox("🎯 COCO Detection Benchmark\n   (Computes mAP@0.50, mAP@[0.40:0.95], AP75, and PR Curves)")
        self.chk_detection.setStyleSheet(card_style)
        self.chk_detection.setChecked(True)
        set_layout.addWidget(self.chk_detection, 2, 0)

        self.chk_tracking = QCheckBox("📊 MOT Tracking Evaluation\n   (Computes HOTA, MOTA, IDF1, Track Loss, Frag, and ID Swaps)")
        self.chk_tracking.setStyleSheet(card_style)
        self.chk_tracking.setChecked(True)
        set_layout.addWidget(self.chk_tracking, 2, 1)

        self.chk_speed = QCheckBox("⚡ Speed Profile Segmentation\n   (Segmented accuracy breakdown for Slow, Medium, and Fast objects)")
        self.chk_speed.setStyleSheet(card_style)
        set_layout.addWidget(self.chk_speed, 3, 0)

        self.chk_center = QCheckBox("📍 Center Bounding Box Accuracy\n   (Measures center point displacement and localization offsets)")
        self.chk_center.setStyleSheet(card_style)
        set_layout.addWidget(self.chk_center, 3, 1)

        self.chk_visualize = QCheckBox("🎬 Generate Error Review Videos\n   (Exports MP4 video sequences highlighting False Positives and False Negatives)")
        self.chk_visualize.setStyleSheet(card_style)
        set_layout.addWidget(self.chk_visualize, 4, 0, 1, 2)

        self.chk_ignore_all = QCheckBox("🚫 Ignore Non-Evaluated Categories\n   (Automatically filters out classes not explicitly mapped)")
        self.chk_ignore_all.setStyleSheet(card_style)
        self.chk_ignore_all.setChecked(True)
        set_layout.addWidget(self.chk_ignore_all, 5, 0, 1, 2)

        layout.addWidget(settings_group)

        # Class Assignment / Remapping Table Group (Collapsible)
        self.btn_toggle_mapping = QPushButton("▶ Show Class Assignment & Remapping")
        self.btn_toggle_mapping.setCheckable(True)
        self.btn_toggle_mapping.clicked.connect(self.toggle_class_mapping)
        layout.addWidget(self.btn_toggle_mapping)

        self.mapping_group = QGroupBox()
        self.mapping_group.setVisible(False)
        mapping_layout = QVBoxLayout(self.mapping_group)

        video_filter_box = QHBoxLayout()
        self.lbl_global_videos = QLabel("🎥 Active Video Sequences:")
        self.lbl_global_videos.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.btn_manage_videos = QPushButton("🎬 All Detected Videos Included (Click to Deselect)")
        self.btn_manage_videos.setStyleSheet("background-color: #2b2b2b; color: #409EFF; border: 1px solid #409EFF; font-weight: bold; border-radius: 4px; padding: 6px 12px; font-size: 12px;")
        self.btn_manage_videos.clicked.connect(self.open_global_video_selector)
        self.global_selected_videos = None # None means all selected
        video_filter_box.addWidget(self.lbl_global_videos)
        video_filter_box.addWidget(self.btn_manage_videos)
        video_filter_box.addStretch()
        mapping_layout.addLayout(video_filter_box)

        btn_scan = QPushButton("🔍 Scan Classes from GT Bundles & YAMLs")
        btn_scan.clicked.connect(self.scan_and_populate_classes)
        mapping_layout.addWidget(btn_scan, 0, Qt.AlignLeft)

        self.table_classes = QTableWidget(0, 3)
        self.table_classes.setMinimumHeight(300)
        self.table_classes.setHorizontalHeaderLabels([
            "Original Class (ID / Name)",
            "Target Evaluation Class (Select [IGNORE] to exclude)",
            "Target Video Filter (Click to Select / Deselect)"
        ])
        self.table_classes.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        mapping_layout.addWidget(self.table_classes)

        layout.addWidget(self.mapping_group)
        
        scroll.setWidget(container)
        main_tab_layout.addWidget(scroll)
        
        self.add_bundle_set()

    def toggle_class_mapping(self, checked):
        self.mapping_group.setVisible(checked)
        if checked:
            self.btn_toggle_mapping.setText("▼ Hide Class Assignment & Remapping")
        else:
            self.btn_toggle_mapping.setText("▶ Show Class Assignment & Remapping")

    def on_tab_changed(self, index):
        current_widget = self.tabs.widget(index)
        if current_widget == self.tab_video_categories:
            self.refresh_video_categories_tab_videos()
        elif current_widget == self.tab_merge:
            self.refresh_merge_classes()

    def get_all_detected_classes(self):
        """Returns sorted unique class names from GT and YAML across bundles."""
        classes = []
        for b in self.bundle_widgets:
            yaml_p = b.get_yaml_path()
            if yaml_p and os.path.exists(yaml_p):
                y_dict = parse_yolo_yaml(yaml_p)
                for v in y_dict.values():
                    if str(v) not in classes:
                        classes.append(str(v))
            gt_p = b.get_gt_path()
            if gt_p and os.path.exists(gt_p) and gt_p != "[ACTIVE_VIAT_PROJECT]":
                cls_list = scan_dataset_classes(gt_p)
                for c in cls_list:
                    if str(c) not in classes:
                        classes.append(str(c))
        if hasattr(self, 'table_classes'):
            for r in range(self.table_classes.rowCount()):
                c_item = self.table_classes.item(r, 0)
                if c_item and c_item.text().strip() and c_item.text().strip() not in classes:
                    classes.append(c_item.text().strip())
        return sorted(classes) if classes else ["object"]

    def setup_merge_tab(self):
        layout = QVBoxLayout(self.tab_merge)
        layout.setSpacing(10)

        header = QLabel(
            "<b>🔀 Class Merging Impact Simulator</b><br>"
            "<span style='color: #aaaaaa; font-size: 11.5px;'>Select classes from the list below to combine into a super-category (e.g. check <i>car, bus, truck</i> and merge into <i>Vehicle</i>). "
            "The benchmark will evaluate both the original baseline and the merged group to measure accuracy gain.</span>"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        splitter = QSplitter(Qt.Horizontal)

        # Left Panel: Available Classes with Checkboxes
        left_group = QGroupBox("1. Select Classes to Merge")
        left_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        left_layout = QVBoxLayout(left_group)

        self.edit_search_classes = QLineEdit()
        self.edit_search_classes.setPlaceholderText("🔍 Filter classes...")
        self.edit_search_classes.textChanged.connect(self.filter_merge_classes)
        left_layout.addWidget(self.edit_search_classes)

        c_btn_box = QHBoxLayout()
        btn_c_all = QPushButton("Select All")
        btn_c_none = QPushButton("Deselect All")
        btn_c_refresh = QPushButton("🔄 Refresh")
        btn_c_all.clicked.connect(self.select_all_merge_classes)
        btn_c_none.clicked.connect(self.deselect_all_merge_classes)
        btn_c_refresh.clicked.connect(self.refresh_merge_classes)
        c_btn_box.addWidget(btn_c_all)
        c_btn_box.addWidget(btn_c_none)
        c_btn_box.addWidget(btn_c_refresh)
        left_layout.addLayout(c_btn_box)

        self.list_merge_classes = QListWidget()
        left_layout.addWidget(self.list_merge_classes)
        splitter.addWidget(left_group)

        # Right Panel: Merge Rule Creator & Table
        right_group = QGroupBox("2. Merge Rules Definition")
        right_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        right_layout = QVBoxLayout(right_group)

        form_box = QHBoxLayout()
        form_box.addWidget(QLabel("Target Group Name:"))
        self.edit_merge_group = QLineEdit()
        self.edit_merge_group.setPlaceholderText("e.g. Vehicle, Pedestrian, Animal")
        form_box.addWidget(self.edit_merge_group)

        btn_add_rule = QPushButton("🔀 Create Merge Rule from Checked Classes")
        btn_add_rule.setStyleSheet("background-color: #007acc; color: white; font-weight: bold; padding: 6px 14px;")
        btn_add_rule.clicked.connect(self.add_merge_rule_from_selection)
        form_box.addWidget(btn_add_rule)
        right_layout.addLayout(form_box)

        self.table_merge = QTableWidget(0, 3)
        self.table_merge.setHorizontalHeaderLabels(["Merged Group Name", "Combined Classes", "Action"])
        self.table_merge.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table_merge.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_merge.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        right_layout.addWidget(self.table_merge)

        splitter.addWidget(right_group)
        splitter.setSizes([320, 580])
        layout.addWidget(splitter)

    def refresh_merge_classes(self):
        classes = self.get_all_detected_classes()
        self.list_merge_classes.clear()
        for c in classes:
            item = QListWidgetItem(str(c), self.list_merge_classes)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)

    def select_all_merge_classes(self):
        for i in range(self.list_merge_classes.count()):
            item = self.list_merge_classes.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.Checked)

    def deselect_all_merge_classes(self):
        for i in range(self.list_merge_classes.count()):
            item = self.list_merge_classes.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.Unchecked)

    def filter_merge_classes(self, text):
        query = text.lower().strip()
        for i in range(self.list_merge_classes.count()):
            item = self.list_merge_classes.item(i)
            item.setHidden(bool(query and query not in item.text().lower()))

    def add_merge_rule_from_selection(self):
        group_name = self.edit_merge_group.text().strip()
        checked_classes = []
        for i in range(self.list_merge_classes.count()):
            item = self.list_merge_classes.item(i)
            if item.checkState() == Qt.Checked:
                checked_classes.append(item.text().strip())

        if not group_name:
            QMessageBox.warning(self, "Missing Name", "Please enter a target group name (e.g. 'Vehicle').")
            return
        if not checked_classes:
            QMessageBox.warning(self, "No Classes Checked", "Please check at least one class from the list on the left to merge.")
            return

        classes_str = ", ".join(checked_classes)
        row = self.table_merge.rowCount()
        self.table_merge.insertRow(row)
        self.table_merge.setItem(row, 0, QTableWidgetItem(group_name))
        self.table_merge.setItem(row, 1, QTableWidgetItem(classes_str))
        
        btn_remove = QPushButton("Remove")
        btn_remove.clicked.connect(lambda: self.remove_merge_rule_at_btn(btn_remove))
        self.table_merge.setCellWidget(row, 2, btn_remove)

        self.merge_groups[group_name] = checked_classes
        self.edit_merge_group.clear()
        self.deselect_all_merge_classes()

    def remove_merge_rule_at_btn(self, btn):
        for r in range(self.table_merge.rowCount()):
            if self.table_merge.cellWidget(r, 2) == btn:
                g_item = self.table_merge.item(r, 0)
                if g_item:
                    g_name = g_item.text()
                    if g_name in self.merge_groups:
                        del self.merge_groups[g_name]
                self.table_merge.removeRow(r)
                break

    def add_merge_rule(self):
        self.add_merge_rule_from_selection()

    def setup_video_categories_tab(self):
        layout = QVBoxLayout(self.tab_video_categories)
        layout.setSpacing(10)

        header = QLabel(
            "<b>🎥 Video Category Subsets Assignment</b><br>"
            "<span style='color: #aaaaaa; font-size: 11.5px;'>Categorize videos into evaluation subsets (e.g. <i>Urban, Night, Highway, Stadium, Thermal</i>). "
            "Select videos from the list on the left, choose a category tag, and click Assign.</span>"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        splitter = QSplitter(Qt.Horizontal)

        # Left: Video List with Checkboxes
        left_group = QGroupBox("1. Detected Video Sequences")
        left_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        left_layout = QVBoxLayout(left_group)

        self.edit_search_vids = QLineEdit()
        self.edit_search_vids.setPlaceholderText("🔍 Filter videos...")
        self.edit_search_vids.textChanged.connect(self.filter_category_tab_videos)
        left_layout.addWidget(self.edit_search_vids)

        v_btn_box = QHBoxLayout()
        btn_v_all = QPushButton("Select All")
        btn_v_none = QPushButton("Deselect All")
        btn_v_refresh = QPushButton("🔄 Refresh")
        btn_v_all.clicked.connect(self.select_all_category_videos)
        btn_v_none.clicked.connect(self.deselect_all_category_videos)
        btn_v_refresh.clicked.connect(self.refresh_video_categories_tab_videos)
        v_btn_box.addWidget(btn_v_all)
        v_btn_box.addWidget(btn_v_none)
        v_btn_box.addWidget(btn_v_refresh)
        left_layout.addLayout(v_btn_box)

        self.list_category_videos = QListWidget()
        left_layout.addWidget(self.list_category_videos)
        splitter.addWidget(left_group)

        # Right: Category Assignment & Mapping Table
        right_group = QGroupBox("2. Category Tagging & Summary")
        right_group.setStyleSheet("QGroupBox { font-weight: bold; }")
        right_layout = QVBoxLayout(right_group)

        form_box = QHBoxLayout()
        form_box.addWidget(QLabel("Category Tag:"))
        self.combo_category_tag = QComboBox()
        self.combo_category_tag.setEditable(True)
        self.combo_category_tag.addItems(["Urban", "Highway", "Night", "Day", "Thermal", "Rain", "Aerial", "Stadium", "Perimeter"])
        form_box.addWidget(self.combo_category_tag)

        btn_assign_cat = QPushButton("➕ Assign to Checked Videos")
        btn_assign_cat.setStyleSheet("background-color: #007acc; color: white; font-weight: bold; padding: 6px 12px;")
        btn_assign_cat.clicked.connect(self.assign_category_to_checked_videos)
        form_box.addWidget(btn_assign_cat)

        btn_remove_cat = QPushButton("➖ Remove Tag from Checked")
        btn_remove_cat.clicked.connect(self.remove_category_from_checked_videos)
        form_box.addWidget(btn_remove_cat)

        right_layout.addLayout(form_box)

        self.table_video_maps = QTableWidget(0, 3)
        self.table_video_maps.setHorizontalHeaderLabels(["Video Name", "Assigned Categories", "Action"])
        self.table_video_maps.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table_video_maps.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table_video_maps.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        right_layout.addWidget(self.table_video_maps)

        splitter.addWidget(right_group)
        splitter.setSizes([320, 580])
        layout.addWidget(splitter)

    def refresh_video_categories_tab_videos(self):
        vids = self.get_all_detected_videos()
        self.list_category_videos.clear()
        
        # Existing video mapping lookup
        existing_table_vids = {}
        for r in range(self.table_video_maps.rowCount()):
            v_item = self.table_video_maps.item(r, 0)
            c_item = self.table_video_maps.item(r, 1)
            if v_item:
                existing_table_vids[v_item.text()] = c_item.text() if c_item else ""

        for v in vids:
            item = QListWidgetItem(v, self.list_category_videos)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            
            # Ensure every video exists in the summary table
            if v not in existing_table_vids:
                row = self.table_video_maps.rowCount()
                self.table_video_maps.insertRow(row)
                self.table_video_maps.setItem(row, 0, QTableWidgetItem(v))
                self.table_video_maps.setItem(row, 1, QTableWidgetItem(""))
                btn_remove = QPushButton("Clear")
                btn_remove.clicked.connect(lambda _, r=row: self.clear_video_category_row(r))
                self.table_video_maps.setCellWidget(row, 2, btn_remove)

    def select_all_category_videos(self):
        for i in range(self.list_category_videos.count()):
            item = self.list_category_videos.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.Checked)

    def deselect_all_category_videos(self):
        for i in range(self.list_category_videos.count()):
            item = self.list_category_videos.item(i)
            if not item.isHidden():
                item.setCheckState(Qt.Unchecked)

    def filter_category_tab_videos(self, text):
        query = text.lower().strip()
        for i in range(self.list_category_videos.count()):
            item = self.list_category_videos.item(i)
            item.setHidden(bool(query and query not in item.text().lower()))

    def assign_category_to_checked_videos(self):
        cat = self.combo_category_tag.currentText().strip()
        if not cat:
            QMessageBox.warning(self, "Missing Category", "Please enter or select a category tag.")
            return

        checked_vids = []
        for i in range(self.list_category_videos.count()):
            item = self.list_category_videos.item(i)
            if item.checkState() == Qt.Checked:
                checked_vids.append(item.text().strip())

        if not checked_vids:
            QMessageBox.warning(self, "No Videos Checked", "Please check at least one video from the list on the left.")
            return

        # Update table rows
        for v in checked_vids:
            found = False
            for r in range(self.table_video_maps.rowCount()):
                v_item = self.table_video_maps.item(r, 0)
                if v_item and v_item.text().strip() == v:
                    found = True
                    c_item = self.table_video_maps.item(r, 1)
                    curr_cats = [c.strip() for c in c_item.text().split(',') if c.strip()] if c_item else []
                    if cat not in curr_cats:
                        curr_cats.append(cat)
                    self.table_video_maps.setItem(r, 1, QTableWidgetItem(", ".join(curr_cats)))
                    break
            if not found:
                row = self.table_video_maps.rowCount()
                self.table_video_maps.insertRow(row)
                self.table_video_maps.setItem(row, 0, QTableWidgetItem(v))
                self.table_video_maps.setItem(row, 1, QTableWidgetItem(cat))
                btn_remove = QPushButton("Clear")
                btn_remove.clicked.connect(lambda _, r=row: self.clear_video_category_row(r))
                self.table_video_maps.setCellWidget(row, 2, btn_remove)

    def remove_category_from_checked_videos(self):
        cat = self.combo_category_tag.currentText().strip()
        if not cat:
            return
        checked_vids = []
        for i in range(self.list_category_videos.count()):
            item = self.list_category_videos.item(i)
            if item.checkState() == Qt.Checked:
                checked_vids.append(item.text().strip())

        for v in checked_vids:
            for r in range(self.table_video_maps.rowCount()):
                v_item = self.table_video_maps.item(r, 0)
                if v_item and v_item.text().strip() == v:
                    c_item = self.table_video_maps.item(r, 1)
                    curr_cats = [c.strip() for c in c_item.text().split(',') if c.strip()] if c_item else []
                    if cat in curr_cats:
                        curr_cats.remove(cat)
                    self.table_video_maps.setItem(r, 1, QTableWidgetItem(", ".join(curr_cats)))
                    break

    def clear_video_category_row(self, row):
        if row < self.table_video_maps.rowCount():
            self.table_video_maps.setItem(row, 1, QTableWidgetItem(""))

    def add_video_mapping_row(self):
        self.assign_category_to_checked_videos()

    def setup_analytics_tab(self):
        main_tab_layout = QVBoxLayout(self.tab_analytics)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(12)

        # 1. Aesthetic & Replot Toolbar
        aesthetic_box = QGroupBox("🎨 Chart Aesthetics & Theme Customization")
        aesthetic_box.setStyleSheet("QGroupBox { font-weight: bold; }")
        a_layout = QHBoxLayout(aesthetic_box)
        a_layout.setContentsMargins(10, 10, 10, 10)
        a_layout.setSpacing(12)

        a_layout.addWidget(QLabel("<b>Target Scope:</b>"))
        self.combo_scope = QComboBox()
        self.combo_scope.setMinimumWidth(200)
        self.combo_scope.addItem("🌐 All Videos (Aggregated)")
        self.combo_scope.currentIndexChanged.connect(self.on_scope_changed)
        a_layout.addWidget(self.combo_scope)

        a_layout.addWidget(QLabel("<b>Theme:</b>"))
        self.combo_theme = QComboBox()
        self.combo_theme.addItems(["Dark", "Light", "Publication"])
        a_layout.addWidget(self.combo_theme)

        a_layout.addWidget(QLabel("<b>Palette:</b>"))
        self.combo_palette = QComboBox()
        self.combo_palette.addItems(["Vibrant", "Viridis", "Cool Ocean", "Warm Sunset", "Monochrome"])
        a_layout.addWidget(self.combo_palette)

        a_layout.addWidget(QLabel("<b>DPI:</b>"))
        self.combo_dpi = QComboBox()
        self.combo_dpi.addItem("100 DPI (Standard)", 100)
        self.combo_dpi.addItem("150 DPI (Crisp HD)", 150)
        self.combo_dpi.addItem("300 DPI (Ultra / Print)", 300)
        self.combo_dpi.setCurrentIndex(1)
        a_layout.addWidget(self.combo_dpi)

        self.chk_show_grid = QCheckBox("Show Grid")
        self.chk_show_grid.setChecked(True)
        a_layout.addWidget(self.chk_show_grid)

        self.btn_replot = QPushButton("🔄 Re-plot & Refresh Charts")
        self.btn_replot.setStyleSheet("""
            QPushButton {
                background-color: #007acc;
                color: white;
                font-weight: bold;
                font-size: 12px;
                padding: 6px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #0098ff;
            }
        """)
        self.btn_replot.clicked.connect(lambda: self.replot_diagnostics_with_aesthetics(show_toast=True))
        a_layout.addWidget(self.btn_replot)
        a_layout.addStretch()

        layout.addWidget(aesthetic_box)

        # 2. Basic Charts Group
        group_basic = QGroupBox("1. Class AP50 & Size Breakdown Diagnostics")
        group_basic.setStyleSheet("QGroupBox { font-weight: bold; }")
        layout_basic = QHBoxLayout(group_basic)

        self.lbl_plot_bar = QLabel("Run evaluation to generate per-class AP50 bar chart.")
        self.lbl_plot_bar.setAlignment(Qt.AlignCenter)
        layout_basic.addWidget(self.lbl_plot_bar)

        self.lbl_plot_size = QLabel("Run evaluation to generate size breakdown chart.")
        self.lbl_plot_size.setAlignment(Qt.AlignCenter)
        layout_basic.addWidget(self.lbl_plot_size)

        self.lbl_plot_merge = QLabel("Run evaluation with merge groups to see accuracy comparison.")
        self.lbl_plot_merge.setAlignment(Qt.AlignCenter)
        layout_basic.addWidget(self.lbl_plot_merge)
        layout.addWidget(group_basic)

        # 3. Advanced Diagnostic Plots Grid (11 Comprehensive Deep Analytics)
        group_diag = QGroupBox("2. Comprehensive Model Diagnostics Dashboard (11 Deep Analytics)")
        group_diag.setStyleSheet("QGroupBox { font-weight: bold; }")
        diag_grid = QGridLayout(group_diag)
        diag_grid.setSpacing(12)

        # Row 0: PR Curves & F1 vs Confidence
        self.lbl_diag_pr = QLabel("Precision-Recall Curves (AUC)")
        self.lbl_diag_pr.setAlignment(Qt.AlignCenter)
        diag_grid.addWidget(self.lbl_diag_pr, 0, 0)

        self.lbl_diag_f1 = QLabel("F1-Score vs Confidence Threshold")
        self.lbl_diag_f1.setAlignment(Qt.AlignCenter)
        diag_grid.addWidget(self.lbl_diag_f1, 0, 1)

        # Row 1: Confusion Matrix & Calibration Curve
        self.lbl_diag_cm = QLabel("Confusion Matrix Heatmap")
        self.lbl_diag_cm.setAlignment(Qt.AlignCenter)
        diag_grid.addWidget(self.lbl_diag_cm, 1, 0)

        self.lbl_diag_calib = QLabel("Confidence Calibration Curve")
        self.lbl_diag_calib.setAlignment(Qt.AlignCenter)
        diag_grid.addWidget(self.lbl_diag_calib, 1, 1)

        # Row 2: TP vs FP Confidence Separation & Error Breakdown Donut
        self.lbl_diag_conf_dist = QLabel("TP vs FP Confidence Distribution")
        self.lbl_diag_conf_dist.setAlignment(Qt.AlignCenter)
        diag_grid.addWidget(self.lbl_diag_conf_dist, 2, 0)

        self.lbl_diag_err_breakdown = QLabel("Error Taxonomy Breakdown Donut")
        self.lbl_diag_err_breakdown.setAlignment(Qt.AlignCenter)
        diag_grid.addWidget(self.lbl_diag_err_breakdown, 2, 1)

        # Row 3: Localization IoU Histogram & Aspect Ratio Bias
        self.lbl_diag_iou = QLabel("Localization IoU Histogram")
        self.lbl_diag_iou.setAlignment(Qt.AlignCenter)
        diag_grid.addWidget(self.lbl_diag_iou, 3, 0)

        self.lbl_diag_ar = QLabel("Aspect Ratio Geometry Bias")
        self.lbl_diag_ar.setAlignment(Qt.AlignCenter)
        diag_grid.addWidget(self.lbl_diag_ar, 3, 1)

        # Row 4: Spatial 2D Error Heatmap & Cross-Video Comparison
        self.lbl_diag_spatial = QLabel("Spatial 2D Error Heatmap")
        self.lbl_diag_spatial.setAlignment(Qt.AlignCenter)
        diag_grid.addWidget(self.lbl_diag_spatial, 4, 0)

        self.lbl_diag_per_video = QLabel("Multi-Video Performance Comparison")
        self.lbl_diag_per_video.setAlignment(Qt.AlignCenter)
        diag_grid.addWidget(self.lbl_diag_per_video, 4, 1)

        # Row 5: MOT Tracking Failure Taxonomy
        self.lbl_diag_track = QLabel("MOT Tracking Error Taxonomy")
        self.lbl_diag_track.setAlignment(Qt.AlignCenter)
        diag_grid.addWidget(self.lbl_diag_track, 5, 0)

        layout.addWidget(group_diag)

        # 4. Detailed Stats Table
        layout.addWidget(QLabel("<b>3. Detailed Per-Class Performance Summary Table:</b>"))
        self.table_stats = QTableWidget(0, 6)
        self.table_stats.setHorizontalHeaderLabels(["Class Name", "AP50", "AP", "TP", "FP", "FN"])
        self.table_stats.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_stats.setMinimumHeight(200)
        layout.addWidget(self.table_stats)

        # 5. Per-Video Table
        layout.addWidget(QLabel("<b>4. Per-Video Detection Metrics:</b>"))
        self.table_video_stats = QTableWidget(0, 6)
        self.table_video_stats.setHorizontalHeaderLabels(["Video", "Precision", "Recall", "F1", "AP", "AP50"])
        self.table_video_stats.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_video_stats.setMinimumHeight(200)
        self.table_video_stats.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_video_stats.customContextMenuRequested.connect(self.show_video_context_menu)
        layout.addWidget(self.table_video_stats)

        scroll.setWidget(container)
        main_tab_layout.addWidget(scroll)

    def show_video_context_menu(self, pos):
        item = self.table_video_stats.itemAt(pos)
        if not item: return
        row = item.row()
        video_name = self.table_video_stats.item(row, 0).text()

        menu = QMenu(self)
        action_ignore = menu.addAction(f"Ignore '{video_name}' and Recompute")
        action_merge = menu.addAction(f"Add Merge Rule for '{video_name}' and Recompute")

        action = menu.exec_(self.table_video_stats.mapToGlobal(pos))
        if action == action_ignore:
            curr = self.edit_ignored_videos.text().strip()
            if curr:
                self.edit_ignored_videos.setText(curr + f", {video_name}")
            else:
                self.edit_ignored_videos.setText(video_name)
            self.tabs.setCurrentWidget(self.tab_datasets)
            self.start_evaluation()
        elif action == action_merge:
            from viat.widgets.class_info_dialog import ClassInfoDialog
            # Simple prompt for original and target
            import PyQt5.QtWidgets as qtw
            orig, ok1 = qtw.QInputDialog.getText(self, "Add Merge Rule", f"[{video_name}]\nEnter original class to merge:")
            if ok1 and orig:
                target, ok2 = qtw.QInputDialog.getText(self, "Add Merge Rule", f"[{video_name}]\nEnter target class for '{orig}':")
                if ok2 and target:
                    r = self.table_classes.rowCount()
                    self.table_classes.insertRow(r)
                    self.table_classes.setItem(r, 0, QTableWidgetItem(orig.strip()))
                    self.table_classes.setItem(r, 1, QTableWidgetItem(target.strip()))
                    self.table_classes.setItem(r, 2, QTableWidgetItem(video_name))
                    self.tabs.setCurrentWidget(self.tab_datasets)
                    self.start_evaluation()
    def setup_log_tab(self):
        layout = QVBoxLayout(self.tab_log)
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Monospace", 9))
        self.log_text.setStyleSheet("background-color: #1e1e1e; color: #00ff66; border-radius: 4px;")
        layout.addWidget(self.log_text)

    def add_bundle_set(self):
        if len(self.bundle_widgets) >= 4:
            return
        index = len(self.bundle_widgets) + 1
        bundle_w = SingleSetWidget(index=index, parent=self)
        bundle_w.remove_requested.connect(self.remove_bundle_set)
        self.bundles_container_layout.addWidget(bundle_w)
        self.bundle_widgets.append(bundle_w)

    def remove_bundle_set(self, widget):
        if widget in self.bundle_widgets:
            self.bundle_widgets.remove(widget)
            self.bundles_container_layout.removeWidget(widget)
            widget.deleteLater()
            for i, w in enumerate(self.bundle_widgets, 1):
                w.index = i

    def get_all_detected_videos(self):
        """Scans all bundle GT and DT directories and returns unique video sequence names."""
        all_vids = []
        for b in self.bundle_widgets:
            gt_p = b.get_gt_path()
            dt_p = b.get_dt_path()
            if gt_p and os.path.exists(gt_p) and gt_p != "[ACTIVE_VIAT_PROJECT]":
                vids = scan_dataset_videos(gt_p)
                for v in vids:
                    if v not in all_vids:
                        all_vids.append(v)
            if dt_p and os.path.exists(dt_p):
                vids = scan_dataset_videos(dt_p)
                for v in vids:
                    if v not in all_vids:
                        all_vids.append(v)
        return sorted(all_vids)

    def open_global_video_selector(self):
        all_vids = self.get_all_detected_videos()
        if not all_vids:
            QMessageBox.information(self, "No Videos Found", "No video sequences were detected in the dataset folders yet. Please set GT/DT folder paths first.")
            return
        dlg = VideoMultiSelectDialog(all_vids, self.global_selected_videos, title="Filter Included Videos for Evaluation", parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self.global_selected_videos = dlg.get_selected_videos()
            if len(self.global_selected_videos) == len(all_vids) or not self.global_selected_videos:
                self.btn_manage_videos.setText(f"🎬 All Detected Videos Included ({len(all_vids)}/{len(all_vids)})")
                self.btn_manage_videos.setStyleSheet("background-color: #2b2b2b; color: #409EFF; border: 1px solid #409EFF; font-weight: bold; border-radius: 4px; padding: 6px 12px; font-size: 12px;")
            else:
                self.btn_manage_videos.setText(f"🎬 {len(self.global_selected_videos)}/{len(all_vids)} Videos Selected (Click to change)")
                self.btn_manage_videos.setStyleSheet("background-color: rgba(230, 162, 60, 0.15); color: #E6A23C; border: 1px solid #E6A23C; font-weight: bold; border-radius: 4px; padding: 6px 12px; font-size: 12px;")

    def open_class_video_selector(self, btn, class_name):
        all_vids = self.get_all_detected_videos()
        if not all_vids:
            QMessageBox.information(self, "No Videos Found", "No video sequences were detected in the dataset folders yet. Please set GT/DT folder paths first.")
            return
        dlg = VideoMultiSelectDialog(all_vids, btn.selected_videos, title=f"Select Target Videos for Class '{class_name}'", parent=self)
        if dlg.exec_() == QDialog.Accepted:
            selected = dlg.get_selected_videos()
            if len(selected) == len(all_vids) or not selected:
                btn.selected_videos = None
                btn.setText("All Videos (Default) ▾")
                btn.setStyleSheet("background-color: #2b2b2b; color: #cccccc; border: 1px solid #444444; border-radius: 4px; padding: 4px 8px; font-size: 11px;")
                btn.setToolTip("Applies to all videos globally")
            else:
                btn.selected_videos = selected
                btn.setText(f"{len(selected)}/{len(all_vids)} Videos ▾")
                btn.setStyleSheet("background-color: rgba(64, 158, 255, 0.15); color: #409EFF; border: 1px solid #409EFF; font-weight: bold; border-radius: 4px; padding: 4px 8px; font-size: 11px;")
                btn.setToolTip(f"Applies only to: {', '.join(selected)}")

    def scan_and_populate_classes(self):
        global_yaml_dict = {}
        global_yaml_classes = []
        local_gt_classes = []
        
        for b in self.bundle_widgets:
            gt_p = b.get_gt_path()
            yaml_p = b.get_yaml_path()

            if yaml_p and os.path.exists(yaml_p):
                yaml_dict = parse_yolo_yaml(yaml_p)
                for k in sorted(yaml_dict.keys()):
                    val_str = str(yaml_dict[k])
                    global_yaml_dict[k] = val_str
                    if val_str not in global_yaml_classes:
                        global_yaml_classes.append(val_str)
            
            if gt_p and os.path.exists(gt_p) and gt_p != "[ACTIVE_VIAT_PROJECT]":
                cls_list = scan_dataset_classes(gt_p)
                for c in cls_list:
                    str_c = str(c)
                    if str_c not in local_gt_classes:
                        local_gt_classes.append(str_c)

        if not local_gt_classes and not global_yaml_classes:
            local_gt_classes = ["0"]

        self.table_classes.setRowCount(0)
        global_yaml_lower = {g.lower(): g for g in global_yaml_classes}

        # If GT classes found, show all original GT classes. Otherwise show YAML class indices
        if local_gt_classes:
            classes_to_process = local_gt_classes
        else:
            classes_to_process = [str(k) for k in sorted(global_yaml_dict.keys())] if global_yaml_dict else global_yaml_classes

        for c in classes_to_process:
            row = self.table_classes.rowCount()
            self.table_classes.insertRow(row)

            orig_class = str(c)
            matched_yaml_class = None

            # 1. Check if orig_class is an integer index matching global_yaml_dict (e.g. '0' -> 'person', '1' -> 'car')
            try:
                c_int = int(orig_class)
                if c_int in global_yaml_dict:
                    matched_yaml_class = global_yaml_dict[c_int]
            except ValueError:
                pass

            # 2. Check if orig_class name matches a YAML class name case-insensitively
            if not matched_yaml_class and orig_class.lower() in global_yaml_lower:
                matched_yaml_class = global_yaml_lower[orig_class.lower()]

            if global_yaml_classes:
                if matched_yaml_class:
                    target_class = matched_yaml_class
                else:
                    # Unmatched in YAML -> Default to [IGNORE]
                    target_class = "[IGNORE] - Exclude from evaluation"
            else:
                target_class = orig_class

            self.table_classes.setItem(row, 0, QTableWidgetItem(orig_class))
            
            # Use QComboBox for target classes with [IGNORE] option
            combo_target = QComboBox()
            combo_target.setEditable(True)
            
            options = ["[IGNORE] - Exclude from evaluation"]
            if global_yaml_classes:
                for g in global_yaml_classes:
                    if g not in options:
                        options.append(g)
            else:
                if orig_class not in options and orig_class != "__IGNORE__":
                    options.append(orig_class)

            combo_target.addItems(options)
            
            # Find index if matched
            idx = -1
            for i in range(combo_target.count()):
                if combo_target.itemText(i).lower() == target_class.lower():
                    idx = i
                    break
            
            if idx >= 0:
                combo_target.setCurrentIndex(idx)
            else:
                combo_target.setCurrentIndex(0)
                
            self.table_classes.setCellWidget(row, 1, combo_target)
            
            # Interactive Multi-Select Button for Target Video Filter
            btn_vid = QPushButton("All Videos (Default) ▾")
            btn_vid.setStyleSheet("background-color: #2b2b2b; color: #cccccc; border: 1px solid #444444; border-radius: 4px; padding: 4px 8px; font-size: 11px;")
            btn_vid.selected_videos = None
            btn_vid.clicked.connect(lambda _, b=btn_vid, r=orig_class: self.open_class_video_selector(b, r))
            self.table_classes.setCellWidget(row, 2, btn_vid)

    def add_merge_rule(self):
        group_name = self.edit_merge_group.text().strip()
        classes_str = self.edit_merge_classes.text().strip()

        if not group_name or not classes_str:
            QMessageBox.warning(self, "Invalid Rule", "Please enter both group name and class list.")
            return

        row = self.table_merge.rowCount()
        self.table_merge.insertRow(row)
        self.table_merge.setItem(row, 0, QTableWidgetItem(group_name))
        self.table_merge.setItem(row, 1, QTableWidgetItem(classes_str))

        self.merge_groups[group_name] = [c.strip() for c in classes_str.split(',') if c.strip()]
        self.edit_merge_group.clear()
        self.edit_merge_classes.clear()

    def append_log(self, text):
        self.log_text.moveCursor(QTextCursor.End)
        self.log_text.insertPlainText(text)
        self.log_text.moveCursor(QTextCursor.End)

    def sync_frame_with_main_window(self, frame_idx):
        main_win = self.parent()
        if main_win and hasattr(main_win, 'frame_slider'):
            try:
                main_win.frame_slider.setValue(frame_idx)
            except Exception:
                pass

    def export_active_project_gt(self):
        main_win = self.parent()
        if not main_win or not hasattr(main_win, 'annotations'):
            return None

        out_dir = "/tmp/viat_eval_gt_active"
        os.makedirs(out_dir, exist_ok=True)
        video_name = "active_video"
        if hasattr(main_win, 'video_path') and main_win.video_path:
            video_name = os.path.splitext(os.path.basename(main_win.video_path))[0]

        txt_path = os.path.join(out_dir, f"{video_name}.txt")
        lines = []

        total_frames = getattr(main_win, 'total_frames', 0)
        if hasattr(main_win, 'annotations') and main_win.annotations:
            max_annotated = max(main_win.annotations.keys()) if main_win.annotations else 0
            total_frames = max(total_frames, max_annotated + 1)

        for f_idx in range(total_frames):
            frame_anns = main_win.annotations.get(f_idx, [])
            line_parts = []
            for ann in frame_anns:
                if isinstance(ann, dict):
                    cls_id, x1, y1, w, h = ann.get('class_id', 0), ann.get('x1', 0), ann.get('y1', 0), ann.get('width', 0), ann.get('height', 0)
                elif isinstance(ann, (list, tuple)) and len(ann) >= 5:
                    cls_id, x1, y1, w, h = ann[0], ann[1], ann[2], ann[3], ann[4]
                else:
                    continue
                line_parts.append(f"[{cls_id}, {x1}, {y1}, {w}, {h}, 100, 100, 0]")
            lines.append(";".join(line_parts) + ";\n")

        with open(txt_path, "w") as f:
            f.writelines(lines)
        return out_dir

    def inspect_in_main_canvas(self):
        """Loads evaluated video and predictions directly into the main VIAT canvas using VIAT's video dataset loader."""
        main_win = self.parent() or getattr(self, 'main_window', None)
        if not main_win:
            QMessageBox.warning(self, "No Main Window", "Main VIAT application window was not found.")
            return

        gt_paths = []
        dt_paths = []
        for b in self.bundle_widgets:
            if hasattr(b, 'get_paths'):
                gt, dt, _ = b.get_paths()
            else:
                gt = b.get_gt_path() if hasattr(b, 'get_gt_path') else ""
                dt = b.get_dt_path() if hasattr(b, 'get_dt_path') else ""
            if gt and os.path.exists(gt):
                gt_paths.append(gt)
            if dt and os.path.exists(dt):
                dt_paths.append(dt)

        gt_dir = gt_paths[0] if gt_paths else ""
        det_dir = dt_paths[0] if dt_paths else ""

        if not dt_paths and hasattr(self, 'last_results_dir') and self.last_results_dir:
            parent_det = os.path.dirname(os.path.abspath(self.last_results_dir))
            if os.path.basename(parent_det) == 'evaluation_result':
                parent_det = os.path.dirname(parent_det)
            if os.path.exists(parent_det):
                det_dir = parent_det

        if not gt_dir or not os.path.exists(gt_dir):
            QMessageBox.warning(self, "Missing Ground Truth", "Ground Truth directory was not found. Please specify a valid GT folder.")
            return

        # Prioritize scanning GT directory for videos / sequences
        video_names = scan_dataset_videos(gt_dir)
        if not video_names and det_dir and os.path.exists(det_dir):
            video_names = scan_dataset_videos(det_dir)

        # Filter out evaluation engine aggregate output names — these are COCO JSON
        # artifacts written by the engine (all_video.json, fast_all_video.json, etc.)
        # and are NOT real video sequences with per-frame GT.
        _AGGREGATE_NAMES = {
            'all_video', 'fast_all_video', 'medium_all_video', 'slow_all_video',
            'all_videos', 'combined', 'aggregated',
        }
        video_names = [
            v for v in video_names
            if v not in _AGGREGATE_NAMES and not v.endswith('_all_video')
        ]

        if not video_names:
            QMessageBox.warning(self, "No Videos Found", f"No video files or annotation sequences found in Ground Truth directory:\n{gt_dir}")
            return

        initial_video = video_names[0]

        # 1. Use VIAT's built-in video dataset loader if available and gt_dir contains video files
        if hasattr(main_win, 'load_video_dataset_path') and os.path.isdir(gt_dir):
            try:
                from viat.utils.video_dataset_manager import scan_video_dataset
                info = scan_video_dataset(gt_dir)
                if info and info.all_videos:
                    main_win.load_video_dataset_path(gt_dir)
            except Exception as e:
                logger.warning(f"Could not load via built-in video dataset loader: {e}")

        # Collect class mapping and target classes
        cls_map = {}
        tgt_classes = []
        for row in range(self.table_classes.rowCount()):
            item_orig = self.table_classes.item(row, 0)
            orig_name = item_orig.text().strip() if item_orig else ""
            try:
                target_name = self.table_classes.cellWidget(row, 1).currentText().strip()
            except AttributeError:
                item_tgt = self.table_classes.item(row, 1)
                target_name = item_tgt.text().strip() if item_tgt else ""
            if not target_name or target_name.upper().startswith("[IGNORE]") or target_name in ("__IGNORE__", "IGNORE"):
                target_name = "__IGNORE__"
            else:
                if target_name not in tgt_classes:
                    tgt_classes.append(target_name)
            if orig_name:
                cls_map[orig_name] = target_name
                try:
                    cls_map[int(orig_name)] = target_name
                except ValueError:
                    pass

        # 2. Attach evaluation context and inspector
        if hasattr(main_win, 'load_evaluation_dataset_into_inspector'):
            main_win.load_evaluation_dataset_into_inspector(
                gt_dir, det_dir, video_names,
                initial_video=initial_video,
                class_mapping=cls_map,
                target_classes=tgt_classes
            )
            self.hide()
        elif hasattr(main_win, 'load_predictions_file_into_inspector'):
            exts = ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.MOV']
            for ext in exts:
                candidate = os.path.join(gt_dir, initial_video + ext)
                if os.path.exists(candidate) and hasattr(main_win, 'open_video'):
                    main_win.open_video(candidate)
                    break
            pred_file = os.path.join(det_dir, f"{initial_video}.txt")
            if not os.path.exists(pred_file):
                pred_file = os.path.join(det_dir, f"{initial_video}.json")
            if os.path.exists(pred_file):
                main_win.load_predictions_file_into_inspector(pred_file, initial_video)
            self.hide()

    def open_advanced_visualizer(self):
        try:
            import viat.widgets.visualizer as vis
            self.vis_window = vis.EvaluatorVisualizer()
            self.vis_window.show()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not open visualizer: {str(e)}")

    def start_evaluation(self):
        gt_paths = []
        dt_paths = []
        
        # We need to extract the global YAML classes to pass to the evaluator
        global_yaml_classes = []
        for b in self.bundle_widgets:
            yaml_p = b.get_yaml_path()
            if yaml_p and os.path.exists(yaml_p):
                from viat.evaluation.utils.yaml_parser import parse_yolo_yaml
                y_dict = parse_yolo_yaml(yaml_p)
                for k, v in y_dict.items():
                    if str(v) not in global_yaml_classes:
                        global_yaml_classes.append(str(v))
                        
        if not global_yaml_classes:
            # If no YAML, extract the unique target strings from the table in order
            for row in range(self.table_classes.rowCount()):
                try:
                    target_name = self.table_classes.cellWidget(row, 1).currentText()
                except AttributeError:
                    target_name = self.table_classes.item(row, 1).text()
                if target_name and target_name not in global_yaml_classes:
                    global_yaml_classes.append(target_name)

        target_classes = global_yaml_classes
        category_names = []

        for b in self.bundle_widgets:
            gt = b.get_gt_path()
            cat = b.get_category()

            if gt == "[ACTIVE_VIAT_PROJECT]":
                gt = self.export_active_project_gt()
                if not gt:
                    QMessageBox.warning(self, "No Active Annotations", "No annotations found in active project.")
                    return
            elif not gt or not os.path.exists(gt):
                QMessageBox.warning(self, "Invalid Path", f"GT folder path for Bundle #{b.index} is invalid or empty.")
                return

            if hasattr(b, 'chk_use_weights') and b.chk_use_weights.isChecked():
                weights_p = b.edit_weights.text().strip()
                media_p = b.edit_media.text().strip()

                if not weights_p or not os.path.exists(weights_p):
                    QMessageBox.warning(self, "Invalid Model Weights", f"Model weights file for Bundle #{b.index} is invalid or empty.")
                    return

                if media_p == "[ACTIVE_VIAT_VIDEO]":
                    main_w = self.parent()
                    if main_w and hasattr(main_w, 'video_path') and main_w.video_path:
                        media_p = main_w.video_path
                    else:
                        QMessageBox.warning(self, "No Active Video", "No active video loaded in VIAT main window.")
                        return
                elif not media_p or not os.path.exists(media_p):
                    QMessageBox.warning(self, "Invalid Media Source", f"Media source (video/images) for Bundle #{b.index} is invalid or empty.")
                    return

                try:
                    self.append_log(f"[INFO] Running Model Inference on '{media_p}' using weights '{weights_p}'...\n")
                    out_dt_dir = f"/tmp/viat_eval_auto_dt_{b.index}"
                    conf_val = self.spin_conf_thr.value()
                    dt = ModelRunner.run_inference(weights_p, media_p, conf_thr=conf_val, output_dir=out_dt_dir)
                    self.append_log(f"[SUCCESS] Model Predictions generated and saved to '{dt}'\n")
                except Exception as ex:
                    QMessageBox.critical(self, "Inference Error", f"Model inference failed:\n{str(ex)}")
                    return
            else:
                dt = b.get_dt_path()
                if not dt or not os.path.exists(dt):
                    QMessageBox.warning(self, "Invalid Path", f"DT folder path for Bundle #{b.index} is invalid or empty.")
                    return

            gt_paths.append(gt)
            dt_paths.append(dt)
            category_names.append(cat)

        # Parse Video Category Mappings
        self.video_category_mappings = {}
        if hasattr(self, 'table_video_maps'):
            for row in range(self.table_video_maps.rowCount()):
                v_name = self.table_video_maps.item(row, 0).text().strip()
                c_str = self.table_video_maps.item(row, 1).text().strip()
                if v_name and c_str:
                    self.video_category_mappings[v_name] = [c.strip() for c in c_str.split(',') if c.strip()]

        # Parse Class Mappings & Video Mappings
        self.class_map = {}
        self.video_class_mappings = {}
        all_vids = self.get_all_detected_videos()

        for row in range(self.table_classes.rowCount()):
            orig_name = self.table_classes.item(row, 0).text().strip()
            try:
                target_name = self.table_classes.cellWidget(row, 1).currentText().strip()
            except AttributeError:
                target_name = self.table_classes.item(row, 1).text().strip()

            btn_vid = self.table_classes.cellWidget(row, 2)
            if hasattr(btn_vid, 'selected_videos'):
                selected_vids = btn_vid.selected_videos
            else:
                item_vid = self.table_classes.item(row, 2)
                raw_txt = item_vid.text().strip() if item_vid else ""
                selected_vids = [v.strip() for v in raw_txt.split(',') if v.strip()] if raw_txt else None

            if not target_name or target_name.upper().startswith("[IGNORE]") or target_name in ("__IGNORE__", "IGNORE"):
                target_name = "__IGNORE__"

            if selected_vids and all_vids and len(selected_vids) < len(all_vids):
                for vid in selected_vids:
                    if vid not in self.video_class_mappings:
                        self.video_class_mappings[vid] = {}
                    self.video_class_mappings[vid][orig_name] = target_name
                    try:
                        self.video_class_mappings[vid][int(orig_name)] = target_name
                    except ValueError: pass
            else:
                self.class_map[orig_name] = target_name
                try:
                    self.class_map[int(orig_name)] = target_name
                except ValueError: pass

        # Global ignored videos calculation from global_selected_videos
        if self.global_selected_videos is not None and all_vids:
            ignored_videos = [v for v in all_vids if v not in self.global_selected_videos]
        else:
            ignored_videos = []

        try:
            try:
                from viat.evaluation.engine import Evaluate
            except Exception as imp_err:
                import traceback
                QMessageBox.critical(
                    self,
                    "Evaluation Engine Error",
                    f"Failed to import Evaluation Engine:\n{str(imp_err)}\n\nPlease ensure required dependencies are installed.\n\nDetails:\n{traceback.format_exc()}"
                )
                return

            try:
                evaluate_inst = Evaluate(
                    gt_path=gt_paths,
                    det_path=dt_paths,
                    category_names=category_names,
                    target_classes=target_classes,
                    quality_thr=self.spin_quality_thr.value(),
                    size_thr=self.spin_size_thr.value(),
                    margin_check=False,
                    margin_value=0,
                    center_check=self.chk_center.isChecked(),
                    detection_check=self.chk_detection.isChecked(),
                    tracking_check=self.chk_tracking.isChecked(),
                    speed_check=self.chk_speed.isChecked(),
                    visualize_check=self.chk_visualize.isChecked(),
                    combine=True,
                    visualize_iou=0.5,
                    ignore_all=self.chk_ignore_all.isChecked(),
                    class_mapping=self.class_map,
                    ignored_videos=ignored_videos,
                    video_class_mappings=self.video_class_mappings,
                    video_category_mappings=self.video_category_mappings
                )
            except Exception as init_err:
                import traceback
                QMessageBox.critical(
                    self,
                    "Evaluation Setup Error",
                    f"Failed to initialize evaluation parameters:\n{str(init_err)}\n\nDetails:\n{traceback.format_exc()}"
                )
                return

            self.btn_evaluate.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.tabs.setCurrentWidget(self.tab_log)
            self.log_text.clear()

            self.eval_worker = EvaluateWorkerThread(evaluate_inst, self.class_map, self.merge_groups, self.last_results_dir)
            self.eval_worker.log_signal.connect(self.append_log)
            self.eval_worker.finished_signal.connect(self.on_evaluation_finished)
            self.eval_worker.start()
        except Exception as general_err:
            import traceback
            QMessageBox.critical(
                self,
                "Evaluation Error",
                f"An unexpected error occurred while starting evaluation:\n{str(general_err)}\n\nDetails:\n{traceback.format_exc()}"
            )
            self.btn_evaluate.setEnabled(True)
            self.progress_bar.setVisible(False)

    def refresh_plot_displays(self, results=None):
        """Updates all 11 chart labels from results dictionary."""
        if not results:
            results = getattr(self, 'last_eval_results', {})
        if not results:
            return

        # Basic Charts
        if results.get('bar_path') and os.path.exists(results['bar_path']):
            self.lbl_plot_bar.setPixmap(QPixmap(results['bar_path']).scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if results.get('size_path') and os.path.exists(results['size_path']):
            self.lbl_plot_size.setPixmap(QPixmap(results['size_path']).scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if results.get('merge_path') and os.path.exists(results['merge_path']):
            self.lbl_plot_merge.setPixmap(QPixmap(results['merge_path']).scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # Advanced Diagnostic Plots (11 Plots)
        diag_mappings = [
            ('pr_path', self.lbl_diag_pr, 500, 340),
            ('f1_path', self.lbl_diag_f1, 500, 340),
            ('cm_path', self.lbl_diag_cm, 480, 360),
            ('calib_path', self.lbl_diag_calib, 550, 320),
            ('conf_dist_path', self.lbl_diag_conf_dist, 500, 320),
            ('err_breakdown_path', self.lbl_diag_err_breakdown, 480, 340),
            ('iou_path', self.lbl_diag_iou, 480, 320),
            ('ar_path', self.lbl_diag_ar, 480, 320),
            ('spatial_path', self.lbl_diag_spatial, 550, 320),
            ('per_video_path', self.lbl_diag_per_video, 520, 320),
            ('track_path', self.lbl_diag_track, 480, 320),
        ]
        for key, label_widget, w, h in diag_mappings:
            path = results.get(key)
            if path and os.path.exists(path):
                label_widget.setPixmap(QPixmap(path).scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def populate_scope_selector(self, diag):
        if not diag:
            return
        curr_text = self.combo_scope.currentText()
        self.combo_scope.blockSignals(True)
        self.combo_scope.clear()
        self.combo_scope.addItem("🌐 All Videos (Aggregated)")

        # Categories
        by_category = diag.get('by_category', {})
        for cat in sorted(by_category.keys()):
            self.combo_scope.addItem(f"📁 Category: {cat}")

        # Videos
        by_video = diag.get('by_video', {})
        for vid in sorted(by_video.keys()):
            self.combo_scope.addItem(f"🎬 Video: {vid}")

        idx = self.combo_scope.findText(curr_text)
        if idx >= 0:
            self.combo_scope.setCurrentIndex(idx)
        else:
            self.combo_scope.setCurrentIndex(0)
        self.combo_scope.blockSignals(False)

    def on_scope_changed(self):
        if hasattr(self, 'last_eval_results') and self.last_eval_results:
            self.replot_diagnostics_with_aesthetics(show_toast=False)

    def replot_diagnostics_with_aesthetics(self, show_toast=True):
        """Re-plots all active charts with the user-selected scope (All, Category, or Video), theme, palette, and DPI."""
        results = getattr(self, 'last_eval_results', None)
        if not results:
            if show_toast:
                QMessageBox.information(self, "No Evaluation Data", "Please run an evaluation first before customizing charts.")
            return

        theme = self.combo_theme.currentText()
        palette = self.combo_palette.currentText()
        dpi = self.combo_dpi.currentData() or 150
        show_grid = self.chk_show_grid.isChecked()

        results_dir = getattr(self, 'results_dir', None) or getattr(self, 'last_results_dir', None)
        if not results_dir:
            return

        # 1. Load root diagnostics data
        diag = results.get('diagnostics_data')
        if not diag:
            diag_file = os.path.join(results_dir, 'diagnostics.json')
            if os.path.exists(diag_file):
                import json as _json
                with open(diag_file, 'r') as f:
                    diag = _json.load(f)
                    results['diagnostics_data'] = diag

        # 2. Determine target scope (All Videos, Category, or Video)
        target_scope = self.combo_scope.currentText()
        active_diag = diag or {}
        scope_tag = "all"
        scope_title = "All Videos (Aggregated)"

        if target_scope.startswith("📁 Category: "):
            cat_name = target_scope.replace("📁 Category: ", "").strip()
            if diag and 'by_category' in diag and cat_name in diag['by_category']:
                active_diag = diag['by_category'][cat_name]
                scope_tag = f"cat_{cat_name}"
                scope_title = f"Category: {cat_name}"
        elif target_scope.startswith("🎬 Video: "):
            vid_name = target_scope.replace("🎬 Video: ", "").strip()
            if diag and 'by_video' in diag and vid_name in diag['by_video']:
                active_diag = diag['by_video'][vid_name]
                scope_tag = f"vid_{vid_name}"
                scope_title = f"Video: {vid_name}"

        # 3. Re-plot basic class AP bar chart for this scope
        from viat.evaluation.utils.plotter import plot_map_by_class, plot_map_by_size
        bar_path = os.path.join(results_dir, f"class_ap50_bar_{scope_tag}.png")
        if active_diag and active_diag.get('per_class_curves'):
            named_m = {c: {'AP50': v.get('ap', 0), 'AP': v.get('ap', 0)} for c, v in active_diag['per_class_curves'].items()}
            plot_map_by_class(named_m, bar_path, theme, palette, dpi)
            results['bar_path'] = bar_path
        elif 'class_metrics' in results and results['class_metrics']:
            named_m = {c: {'AP50': v.get('ap50', 0), 'AP': v.get('ap', 0)} for c, v in results['class_metrics'].items()}
            plot_map_by_class(named_m, bar_path, theme, palette, dpi)
            results['bar_path'] = bar_path

        if 'size_metrics' in results and results['size_metrics']:
            size_path = os.path.join(results_dir, "size_breakdown_bar.png")
            named_s = {k: v.get('ap50', 0) for k, v in results['size_metrics'].items()}
            plot_map_by_size(named_s, size_path, theme, palette, dpi)
            results['size_path'] = size_path

        # 4. Re-plot advanced diagnostic charts from active_diag
        if active_diag:
            classes = active_diag.get('classes', [])
            cm = np.array(active_diag.get('confusion', []), dtype=float)
            if len(classes) and cm.size:
                cm_path = os.path.join(results_dir, f"diag_confusion_matrix_{scope_tag}.png")
                AdvancedDiagnosticsEngine.generate_confusion_matrix_plot(cm, classes, cm_path, theme, palette, dpi)
                results['cm_path'] = cm_path

            calib = active_diag.get('calibration', {})
            if calib.get('confidences'):
                calib_path = os.path.join(results_dir, f"diag_calibration_ece_{scope_tag}.png")
                AdvancedDiagnosticsEngine.generate_calibration_plot(
                    np.array(calib['confidences']), np.array(calib['precisions']), np.array(calib['recalls']),
                    calib.get('ece_score', 0.0), calib.get('optimal_thr', 0.5), calib_path,
                    theme, palette, dpi, show_grid=show_grid
                )
                results['calib_path'] = calib_path

            if active_diag.get('per_class_curves'):
                pr_path = os.path.join(results_dir, f"diag_pr_curves_{scope_tag}.png")
                AdvancedDiagnosticsEngine.generate_pr_curves_plot(
                    active_diag['per_class_curves'], pr_path, theme, palette, dpi, show_grid=show_grid
                )
                results['pr_path'] = pr_path

                f1_path = os.path.join(results_dir, f"diag_f1_vs_confidence_{scope_tag}.png")
                AdvancedDiagnosticsEngine.generate_f1_confidence_plot(
                    active_diag['per_class_curves'], f1_path, theme, palette, dpi, show_grid=show_grid
                )
                results['f1_path'] = f1_path

            if active_diag.get('conf_tp') or active_diag.get('conf_fp'):
                conf_dist_path = os.path.join(results_dir, f"diag_confidence_distribution_{scope_tag}.png")
                AdvancedDiagnosticsEngine.generate_confidence_distribution_plot(
                    active_diag.get('conf_tp', []), active_diag.get('conf_fp', []), conf_dist_path,
                    theme, palette, dpi, show_grid=show_grid
                )
                results['conf_dist_path'] = conf_dist_path

            if active_diag.get('error_breakdown'):
                err_breakdown_path = os.path.join(results_dir, f"diag_error_breakdown_{scope_tag}.png")
                AdvancedDiagnosticsEngine.generate_error_breakdown_plot(
                    active_diag['error_breakdown'], err_breakdown_path, theme, palette, dpi
                )
                results['err_breakdown_path'] = err_breakdown_path

            if active_diag.get('ious'):
                iou_path = os.path.join(results_dir, f"diag_iou_distribution_{scope_tag}.png")
                AdvancedDiagnosticsEngine.generate_iou_distribution_plot(
                    active_diag['ious'], iou_path, theme, palette, dpi, show_grid=show_grid
                )
                results['iou_path'] = iou_path

            ar = active_diag.get('aspect_ratio', {})
            if ar.get('ratios'):
                ar_path = os.path.join(results_dir, f"diag_aspect_ratio_bias_{scope_tag}.png")
                AdvancedDiagnosticsEngine.generate_aspect_ratio_plot(
                    ar['ratios'], ar['error_rates'], ar_path, theme, palette, dpi, show_grid=show_grid
                )
                results['ar_path'] = ar_path

            fp_coords = active_diag.get('fp_coords', [])
            fn_coords = active_diag.get('fn_coords', [])
            if fp_coords or fn_coords:
                spatial_path = os.path.join(results_dir, f"diag_spatial_error_map_{scope_tag}.png")
                AdvancedDiagnosticsEngine.generate_spatial_error_heatmap(
                    fp_coords, fn_coords, tuple(active_diag.get('canvas_size', (1920, 1080))), spatial_path,
                    theme, palette, dpi
                )
                results['spatial_path'] = spatial_path

            v_metrics = active_diag.get('video_metrics') or results.get('video_metrics') or (diag.get('video_metrics') if diag else None)
            if v_metrics:
                per_video_path = os.path.join(results_dir, f"diag_per_video_{scope_tag}.png")
                AdvancedDiagnosticsEngine.generate_per_video_comparison_plot(
                    v_metrics, per_video_path, theme, palette, dpi
                )
                results['per_video_path'] = per_video_path

        # 5. Refresh UI displays & Tables
        self.refresh_plot_displays(results)

        # Update stats table for active scope
        if active_diag and active_diag.get('per_class_curves'):
            self.table_stats.setRowCount(0)
            for c_name, c_crv in active_diag['per_class_curves'].items():
                r = self.table_stats.rowCount()
                self.table_stats.insertRow(r)
                self.table_stats.setItem(r, 0, QTableWidgetItem(c_name))
                self.table_stats.setItem(r, 1, QTableWidgetItem(f"{c_crv.get('ap', 0)*100:.1f}%"))
                self.table_stats.setItem(r, 2, QTableWidgetItem(f"{c_crv.get('ap', 0)*100:.1f}%"))
                self.table_stats.setItem(r, 3, QTableWidgetItem(f"Peak F1: {c_crv.get('peak_f1', 0)*100:.1f}%"))
                self.table_stats.setItem(r, 4, QTableWidgetItem(f"Opt Thr: {c_crv.get('optimal_thr', 0.5):.2f}"))
                self.table_stats.setItem(r, 5, QTableWidgetItem(scope_title))

        if show_toast:
            QMessageBox.information(self, "Plots Updated", f"Successfully re-rendered diagnostic charts for [{scope_title}] with '{theme}' theme and '{palette}' palette at {dpi} DPI.")

    def on_evaluation_finished(self, success, msg, results):
        self.btn_evaluate.setEnabled(True)
        self.progress_bar.setVisible(False)

        if success and results:
            self.last_eval_results = results
            self.results_dir = self.last_results_dir
            self.btn_open_results.setEnabled(True)

            # Populate Scope Selector (All, Categories, Videos)
            diag = results.get('diagnostics_data')
            if not diag:
                diag_file = os.path.join(self.results_dir, 'diagnostics.json')
                if os.path.exists(diag_file):
                    import json as _json
                    with open(diag_file, 'r') as f:
                        diag = _json.load(f)
                        results['diagnostics_data'] = diag
            self.populate_scope_selector(diag)

            # Update Analytics Plots
            self.replot_diagnostics_with_aesthetics(show_toast=False)

            # Populate Stats Table
            class_metrics = results.get('class_metrics', {})
            self.table_stats.setRowCount(0)
            for c_name, c_data in class_metrics.items():
                r = self.table_stats.rowCount()
                self.table_stats.insertRow(r)
                self.table_stats.setItem(r, 0, QTableWidgetItem(c_name))
                self.table_stats.setItem(r, 1, QTableWidgetItem(f"{c_data.get('ap50', 0)*100:.1f}%"))
                self.table_stats.setItem(r, 2, QTableWidgetItem(f"{c_data.get('ap', 0)*100:.1f}%"))
                self.table_stats.setItem(r, 3, QTableWidgetItem(str(c_data.get('tp', 0))))
                self.table_stats.setItem(r, 4, QTableWidgetItem(str(c_data.get('fp', 0))))
                self.table_stats.setItem(r, 5, QTableWidgetItem(str(c_data.get('fn', 0))))

            # Populate Video Stats Table
            video_metrics = results.get('video_metrics', [])
            self.table_video_stats.setRowCount(0)
            for v_data in video_metrics:
                r = self.table_video_stats.rowCount()
                self.table_video_stats.insertRow(r)
                self.table_video_stats.setItem(r, 0, QTableWidgetItem(v_data['name']))
                mets = v_data['metrics']
                
                try:
                    p_val = f"{float(mets.get('Precision', 0))*100:.1f}%"
                except: p_val = "-"
                
                try:
                    r_val = f"{float(mets.get('Recall', 0))*100:.1f}%"
                except: r_val = "-"
                
                try:
                    f1_val = f"{float(mets.get('F1', 0))*100:.1f}%"
                except: f1_val = "-"
                
                try:
                    ap_val = f"{float(mets.get('AP', 0))*100:.1f}%"
                except: ap_val = "-"
                
                try:
                    ap50_val = f"{float(mets.get('AP50', 0))*100:.1f}%"
                except: ap50_val = "-"

                self.table_video_stats.setItem(r, 1, QTableWidgetItem(p_val))
                self.table_video_stats.setItem(r, 2, QTableWidgetItem(r_val))
                self.table_video_stats.setItem(r, 3, QTableWidgetItem(f1_val))
                self.table_video_stats.setItem(r, 4, QTableWidgetItem(ap_val))
                self.table_video_stats.setItem(r, 5, QTableWidgetItem(ap50_val))

            # Switch to Analytics Tab
            self.tabs.setCurrentWidget(self.tab_analytics)

            QMessageBox.information(self, "Evaluation Complete", f"Evaluation & Analytics Completed Successfully!\nResults saved to: {self.last_results_dir}")
        else:
            QMessageBox.warning(self, "Evaluation Failed", f"Evaluation failed: {msg}")

    def open_results_folder(self):
        if self.last_results_dir and os.path.exists(self.last_results_dir):
            try:
                import subprocess
                subprocess.Popen(['xdg-open', self.last_results_dir])
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to open results folder:\n{str(e)}")

    def save_profile(self):
        import time
        default_dir = ""
        if self.bundle_widgets:
            first_gt = self.bundle_widgets[0].get_gt_path()
            if first_gt and os.path.exists(first_gt):
                default_dir = first_gt
        default_name = f"profile_{time.strftime('%Y%m%d_%H%M%S')}.json"
        default_path = os.path.join(default_dir, default_name)
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Evaluation Profile", default_path, "JSON Files (*.json)")
        if not file_path: return
        if not file_path.endswith('.json'): file_path += '.json'

        profile = {
            "bundles": [],
            "settings": {
                "size_thr": self.spin_size_thr.value(),
                "quality_thr": self.spin_quality_thr.value(),
                "conf_thr": self.spin_conf_thr.value(),
                "ignore_all": self.chk_ignore_all.isChecked(),
                "detection": self.chk_detection.isChecked(),
                "speed": self.chk_speed.isChecked(),
                "tracking": self.chk_tracking.isChecked(),
                "center": self.chk_center.isChecked(),
                "visualize": self.chk_visualize.isChecked(),
            },
            "class_mappings": []
        }

        for b in self.bundle_widgets:
            profile["bundles"].append({
                "gt": b.edit_gt.text(),
                "dt": b.edit_dt.text(),
                "yaml": b.edit_yaml.text(),
                "cat": b.edit_cat.text(),
                "use_weights": hasattr(b, 'chk_use_weights') and b.chk_use_weights.isChecked(),
                "weights": b.edit_weights.text() if hasattr(b, 'edit_weights') else "",
                "media": b.edit_media.text() if hasattr(b, 'edit_media') else ""
            })

        profile["global_selected_videos"] = self.global_selected_videos
        for row in range(self.table_classes.rowCount()):
            orig_name = self.table_classes.item(row, 0).text()
            try:
                target_name = self.table_classes.cellWidget(row, 1).currentText()
            except AttributeError:
                target_name = self.table_classes.item(row, 1).text()
            btn_vid = self.table_classes.cellWidget(row, 2)
            selected_vids = getattr(btn_vid, 'selected_videos', None) if btn_vid else None
            profile["class_mappings"].append({"orig": orig_name, "target": target_name, "selected_videos": selected_vids})

        profile["video_mappings"] = []
        if hasattr(self, 'table_video_maps'):
            for row in range(self.table_video_maps.rowCount()):
                v_name = self.table_video_maps.item(row, 0).text()
                c_str = self.table_video_maps.item(row, 1).text()
                profile["video_mappings"].append({"video": v_name, "categories": c_str})

        import json
        with open(file_path, 'w') as f:
            json.dump(profile, f, indent=4)
        QMessageBox.information(self, "Success", "Evaluation profile saved successfully!")

    def load_profile(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Evaluation Profile", "", "JSON Files (*.json)")
        if not file_path: return

        import json
        with open(file_path, 'r') as f:
            profile = json.load(f)

        settings = profile.get("settings", {})
        self.spin_size_thr.setValue(settings.get("size_thr", 0))
        self.spin_quality_thr.setValue(settings.get("quality_thr", 0))
        if "conf_thr" in settings:
            self.spin_conf_thr.setValue(settings["conf_thr"])
        self.chk_ignore_all.setChecked(settings.get("ignore_all", True))
        self.chk_detection.setChecked(settings.get("detection", True))
        self.chk_speed.setChecked(settings.get("speed", False))
        self.chk_tracking.setChecked(settings.get("tracking", True))
        self.chk_center.setChecked(settings.get("center", False))
        self.chk_visualize.setChecked(settings.get("visualize", False))

        for widget in list(self.bundle_widgets):
            self.remove_bundle_set(widget)
        
        for b_data in profile.get("bundles", []):
            self.add_bundle_set()
            b = self.bundle_widgets[-1]
            b.edit_gt.setText(b_data.get("gt", ""))
            b.edit_dt.setText(b_data.get("dt", ""))
            b.edit_yaml.setText(b_data.get("yaml", ""))
            b.edit_cat.setText(b_data.get("cat", ""))
            if hasattr(b, 'chk_use_weights'):
                b.chk_use_weights.setChecked(b_data.get("use_weights", False))
                b.edit_weights.setText(b_data.get("weights", ""))
                b.edit_media.setText(b_data.get("media", ""))

        self.global_selected_videos = profile.get("global_selected_videos", None)
        all_vids = self.get_all_detected_videos()
        if self.global_selected_videos is not None and all_vids:
            if len(self.global_selected_videos) == len(all_vids):
                self.btn_manage_videos.setText(f"🎬 All Detected Videos Included ({len(all_vids)}/{len(all_vids)})")
                self.btn_manage_videos.setStyleSheet("background-color: #2b2b2b; color: #409EFF; border: 1px solid #409EFF; font-weight: bold; border-radius: 4px; padding: 6px 12px; font-size: 12px;")
            else:
                self.btn_manage_videos.setText(f"🎬 {len(self.global_selected_videos)}/{len(all_vids)} Videos Selected (Click to change)")
                self.btn_manage_videos.setStyleSheet("background-color: rgba(230, 162, 60, 0.15); color: #E6A23C; border: 1px solid #E6A23C; font-weight: bold; border-radius: 4px; padding: 6px 12px; font-size: 12px;")

        mappings = profile.get("class_mappings", [])
        self.table_classes.setRowCount(0)
        for m in mappings:
            row = self.table_classes.rowCount()
            self.table_classes.insertRow(row)
            orig_name = m.get("orig", "")
            self.table_classes.setItem(row, 0, QTableWidgetItem(orig_name))
            
            combo = QComboBox()
            combo.setEditable(True)
            combo.addItem("[IGNORE] - Exclude from evaluation")
            target_val = m.get("target", "")
            if target_val and target_val not in ["[IGNORE] - Exclude from evaluation", "__IGNORE__"]:
                combo.addItem(target_val)
            if target_val in ["__IGNORE__", "[IGNORE] - Exclude from evaluation", "IGNORE"] or m.get("include") is False:
                combo.setCurrentIndex(0)
            else:
                combo.setCurrentText(target_val)
            self.table_classes.setCellWidget(row, 1, combo)

            # Recreate button
            btn_vid = QPushButton("All Videos (Default) ▾")
            btn_vid.setStyleSheet("background-color: #2b2b2b; color: #cccccc; border: 1px solid #444444; border-radius: 4px; padding: 4px 8px; font-size: 11px;")
            sel_vids = m.get("selected_videos") or (m.get("video").split(',') if m.get("video") else None)
            if sel_vids and all_vids and len(sel_vids) < len(all_vids):
                btn_vid.selected_videos = sel_vids
                btn_vid.setText(f"{len(sel_vids)}/{len(all_vids)} Videos ▾")
                btn_vid.setStyleSheet("background-color: rgba(64, 158, 255, 0.15); color: #409EFF; border: 1px solid #409EFF; font-weight: bold; border-radius: 4px; padding: 4px 8px; font-size: 11px;")
                btn_vid.setToolTip(f"Applies only to: {', '.join(sel_vids)}")
            else:
                btn_vid.selected_videos = None
            btn_vid.clicked.connect(lambda _, b=btn_vid, r=orig_name: self.open_class_video_selector(b, r))
            self.table_classes.setCellWidget(row, 2, btn_vid)
            
        video_mappings = profile.get("video_mappings", [])
        if hasattr(self, 'table_video_maps'):
            self.table_video_maps.setRowCount(0)
            for m in video_mappings:
                row = self.table_video_maps.rowCount()
                self.table_video_maps.insertRow(row)
                self.table_video_maps.setItem(row, 0, QTableWidgetItem(m.get("video", "")))
                self.table_video_maps.setItem(row, 1, QTableWidgetItem(m.get("categories", "")))
                btn_remove = QPushButton("Remove")
                btn_remove.clicked.connect(lambda: self.table_video_maps.removeRow(self.table_video_maps.currentRow()))
                self.table_video_maps.setCellWidget(row, 2, btn_remove)
