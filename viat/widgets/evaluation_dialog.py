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
    QInputDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QIcon, QFont, QTextCursor, QPixmap

from viat.evaluation.utils.yaml_parser import parse_yolo_yaml, scan_dataset_classes
from viat.evaluation.utils.class_merger import DetailedAnalyticsEngine
from viat.evaluation.utils.advanced_diagnostics import AdvancedDiagnosticsEngine
from viat.evaluation.visualization.visual_inspector import VisualInspectorWidget
from viat.evaluation.inference.model_runner import ModelRunner




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
                        try:
                            ap50 = float(metrics.get('AP50', 0) or 0)
                        except Exception:
                            ap50 = 0

                        if name == 'all_video':
                            results['unmerged_mAP'] = ap50
                            results['merged_mAP'] = ap50
                            continue
                        elif name.endswith('_all_video'):
                            continue

                        video_metrics.append({'name': name, 'metrics': metrics})

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
            cm_path = os.path.join(self.results_dir, "diag_confusion_matrix.png")
            calib_path = os.path.join(self.results_dir, "diag_calibration_ece.png")
            iou_path = os.path.join(self.results_dir, "diag_iou_distribution.png")
            ar_path = os.path.join(self.results_dir, "diag_aspect_ratio_bias.png")
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
                else:
                    self.log_signal.emit("\n[WARNING] No diagnostics.json produced by the engine; advanced diagnostic plots were skipped.\n")
            except Exception as diag_err:
                self.log_signal.emit(f"\n[WARNING] Could not generate advanced diagnostic charts: {str(diag_err)}\n")

            results['bar_path'] = bar_path
            results['size_path'] = size_path
            results['merge_path'] = merge_path if self.merge_groups else None
            results['cm_path'] = cm_path
            results['calib_path'] = calib_path
            results['iou_path'] = iou_path
            results['ar_path'] = ar_path
            results['track_path'] = track_path
            results['spatial_path'] = spatial_path

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
        self.btn_evaluate = QPushButton("Evaluate")
        self.btn_evaluate.setMinimumHeight(40)
        self.btn_evaluate.clicked.connect(self.start_evaluation)
        btn_layout.addWidget(self.btn_evaluate)
        
        self.btn_visualize = QPushButton("Advanced Visualizer")
        self.btn_visualize.setMinimumHeight(40)
        self.btn_visualize.clicked.connect(self.open_advanced_visualizer)
        btn_layout.addWidget(self.btn_visualize)
        
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
        set_layout = QGridLayout(settings_group)
        set_layout.addWidget(QLabel("Size Threshold (%):"), 0, 0)
        self.spin_size_thr = QSpinBox()
        self.spin_size_thr.setRange(0, 100)
        set_layout.addWidget(self.spin_size_thr, 0, 1)

        set_layout.addWidget(QLabel("Quality Threshold (%):"), 0, 2)
        self.spin_quality_thr = QSpinBox()
        self.spin_quality_thr.setRange(0, 100)
        set_layout.addWidget(self.spin_quality_thr, 0, 3)

        set_layout.addWidget(QLabel("Confidence Threshold (Auto-Inference):"), 1, 0)
        self.spin_conf_thr = QDoubleSpinBox()
        self.spin_conf_thr.setRange(0.01, 1.0)
        self.spin_conf_thr.setSingleStep(0.05)
        self.spin_conf_thr.setValue(0.25)
        set_layout.addWidget(self.spin_conf_thr, 1, 1)

        self.chk_ignore_all = QCheckBox("Ignore non-evaluated categories")
        self.chk_ignore_all.setChecked(True)
        set_layout.addWidget(self.chk_ignore_all, 1, 2, 1, 2)

        set_layout.addWidget(QLabel("<b>Metrics & Diagnostics to Run:</b>"), 2, 0, 1, 4)

        self.chk_detection = QCheckBox("COCO Detection Evaluation (mAP, AP50, AP75)")
        self.chk_detection.setChecked(True)
        set_layout.addWidget(self.chk_detection, 3, 0, 1, 2)

        self.chk_speed = QCheckBox("Speed Profile Segmentation (Slow / Med / Fast)")
        set_layout.addWidget(self.chk_speed, 3, 2, 1, 2)

        self.chk_tracking = QCheckBox("MOT Tracking Evaluation (MOTA, IDF1, HOTA)")
        set_layout.addWidget(self.chk_tracking, 4, 0, 1, 2)

        self.chk_center = QCheckBox("Center Bounding Box Accuracy Check")
        set_layout.addWidget(self.chk_center, 4, 2, 1, 2)

        self.chk_visualize = QCheckBox("Generate MP4 Error Videos (FP/FN Visualizer)")
        set_layout.addWidget(self.chk_visualize, 5, 0, 1, 2)
        layout.addWidget(settings_group)

        # Class Assignment / Remapping Table Group (Collapsible)
        self.btn_toggle_mapping = QPushButton("▶ Show Class Assignment & Remapping")
        self.btn_toggle_mapping.setCheckable(True)
        self.btn_toggle_mapping.clicked.connect(self.toggle_class_mapping)
        layout.addWidget(self.btn_toggle_mapping)

        self.mapping_group = QGroupBox()
        self.mapping_group.setVisible(False)
        mapping_layout = QVBoxLayout(self.mapping_group)

        ignore_layout = QHBoxLayout()
        ignore_layout.addWidget(QLabel("Ignored Videos (comma separated):"))
        self.edit_ignored_videos = QLineEdit()
        self.edit_ignored_videos.setPlaceholderText("e.g. video1, sequence3")
        ignore_layout.addWidget(self.edit_ignored_videos)
        mapping_layout.addLayout(ignore_layout)

        btn_scan = QPushButton("🔍 Scan Classes from GT Bundles & YAMLs")
        btn_scan.clicked.connect(self.scan_and_populate_classes)
        mapping_layout.addWidget(btn_scan, 0, Qt.AlignLeft)

        self.table_classes = QTableWidget(0, 4)
        self.table_classes.setMinimumHeight(300)
        self.table_classes.setHorizontalHeaderLabels(["Original Class ID / Name", "Target Evaluation Name", "Include in Eval", "Target Video (Empty=All)"])
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

    def setup_merge_tab(self):
        layout = QVBoxLayout(self.tab_merge)
        layout.addWidget(QLabel(
            "<b>Class Merging Impact Simulator</b><br>"
            "Define merge groups (e.g. merge 'car' + 'bus' + 'truck' into 'vehicle'). "
            "The evaluator will run both baseline and merged passes to measure accuracy gain."
        ))

        form_layout = QHBoxLayout()
        form_layout.addWidget(QLabel("Target Merged Group Name:"))
        self.edit_merge_group = QLineEdit()
        self.edit_merge_group.setPlaceholderText("e.g. Vehicle")
        form_layout.addWidget(self.edit_merge_group)

        form_layout.addWidget(QLabel("Classes to Merge (comma separated):"))
        self.edit_merge_classes = QLineEdit()
        self.edit_merge_classes.setPlaceholderText("e.g. car, bus, truck")
        form_layout.addWidget(self.edit_merge_classes)

        btn_add_group = QPushButton("+ Add Merge Rule")
        btn_add_group.clicked.connect(self.add_merge_rule)
        form_layout.addWidget(btn_add_group)
        layout.addLayout(form_layout)

        self.table_merge = QTableWidget(0, 2)
        self.table_merge.setHorizontalHeaderLabels(["Merged Group Name", "Combined Classes"])
        self.table_merge.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.table_merge)

    def setup_video_categories_tab(self):
        layout = QVBoxLayout(self.tab_video_categories)
        layout.addWidget(QLabel("Map specific videos to evaluation subsets/categories (e.g., stadium, night).\nUse this when a single GT/DT folder contains videos from multiple subsets."))
        
        form = QHBoxLayout()
        self.edit_vid_map_name = QLineEdit()
        self.edit_vid_map_name.setPlaceholderText("Video Name (e.g., video_1)")
        self.edit_vid_map_cats = QLineEdit()
        self.edit_vid_map_cats.setPlaceholderText("Categories (comma-separated, e.g., stadium, night)")
        btn_add_vid_map = QPushButton("Add Mapping")
        btn_add_vid_map.clicked.connect(self.add_video_mapping_row)
        form.addWidget(self.edit_vid_map_name)
        form.addWidget(self.edit_vid_map_cats)
        form.addWidget(btn_add_vid_map)
        layout.addLayout(form)
        
        self.table_video_maps = QTableWidget()
        self.table_video_maps.setColumnCount(3)
        self.table_video_maps.setHorizontalHeaderLabels(["Video Name", "Categories", "Action"])
        self.table_video_maps.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table_video_maps)

    def add_video_mapping_row(self):
        vid_name = self.edit_vid_map_name.text().strip()
        cats = self.edit_vid_map_cats.text().strip()
        if not vid_name or not cats:
            return
        row = self.table_video_maps.rowCount()
        self.table_video_maps.insertRow(row)
        self.table_video_maps.setItem(row, 0, QTableWidgetItem(vid_name))
        self.table_video_maps.setItem(row, 1, QTableWidgetItem(cats))
        btn_remove = QPushButton("Remove")
        btn_remove.clicked.connect(lambda: self.table_video_maps.removeRow(self.table_video_maps.currentRow()))
        self.table_video_maps.setCellWidget(row, 2, btn_remove)
        self.edit_vid_map_name.clear()
        self.edit_vid_map_cats.clear()

    def setup_analytics_tab(self):
        main_tab_layout = QVBoxLayout(self.tab_analytics)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        layout = QVBoxLayout(container)

        # Basic Charts Group
        group_basic = QGroupBox("1. Class AP50 & Size Breakdown Diagnostics")
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

        # Advanced Diagnostic Plots Grid (6 Plots)
        group_diag = QGroupBox("2. Advanced Model Diagnostic Dashboard (6 Deep Analytics)")
        diag_grid = QGridLayout(group_diag)

        self.lbl_diag_cm = QLabel("Confusion Matrix Heatmap")
        self.lbl_diag_cm.setAlignment(Qt.AlignCenter)
        diag_grid.addWidget(self.lbl_diag_cm, 0, 0)

        self.lbl_diag_calib = QLabel("Confidence Calibration Curve")
        self.lbl_diag_calib.setAlignment(Qt.AlignCenter)
        diag_grid.addWidget(self.lbl_diag_calib, 0, 1)

        self.lbl_diag_iou = QLabel("Localization IoU Histogram")
        self.lbl_diag_iou.setAlignment(Qt.AlignCenter)
        diag_grid.addWidget(self.lbl_diag_iou, 1, 0)

        self.lbl_diag_ar = QLabel("Aspect Ratio Geometry Bias")
        self.lbl_diag_ar.setAlignment(Qt.AlignCenter)
        diag_grid.addWidget(self.lbl_diag_ar, 1, 1)

        self.lbl_diag_track = QLabel("MOT Tracking Error Taxonomy")
        self.lbl_diag_track.setAlignment(Qt.AlignCenter)
        diag_grid.addWidget(self.lbl_diag_track, 2, 0)

        self.lbl_diag_spatial = QLabel("Spatial 2D Error Heatmap")
        self.lbl_diag_spatial.setAlignment(Qt.AlignCenter)
        diag_grid.addWidget(self.lbl_diag_spatial, 2, 1)

        layout.addWidget(group_diag)

        # Detailed Stats Table
        layout.addWidget(QLabel("<b>3. Detailed Per-Class Performance Summary Table:</b>"))
        self.table_stats = QTableWidget(0, 6)
        self.table_stats.setHorizontalHeaderLabels(["Class Name", "AP50", "AP", "TP", "FP", "FN"])
        self.table_stats.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table_stats.setMinimumHeight(200)
        layout.addWidget(self.table_stats)

        # Add Per-Video Table
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
                    chk = QCheckBox()
                    chk.setChecked(True)
                    self.table_classes.setCellWidget(r, 2, chk)
                    self.table_classes.setItem(r, 3, QTableWidgetItem(video_name))
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

    def scan_and_populate_classes(self):
        global_yaml_classes = []
        local_gt_classes = []
        
        for b in self.bundle_widgets:
            gt_p = b.get_gt_path()
            yaml_p = b.get_yaml_path()

            if yaml_p and os.path.exists(yaml_p):
                yaml_dict = parse_yolo_yaml(yaml_p)
                for k, v in yaml_dict.items():
                    global_yaml_classes.append(str(v))
            
            if gt_p and os.path.exists(gt_p):
                cls_list = scan_dataset_classes(gt_p)
                for c in cls_list:
                    str_c = str(c)
                    if str_c not in local_gt_classes:
                        local_gt_classes.append(str_c)

        if not local_gt_classes and not global_yaml_classes:
            local_gt_classes = ["0"]

        self.table_classes.setRowCount(0)
        global_yaml_lower = {g.lower(): g for g in global_yaml_classes}

        # If they provided a YAML, but no GT classes were found, just show the YAML classes
        classes_to_process = local_gt_classes if local_gt_classes else global_yaml_classes

        for c in classes_to_process:
            row = self.table_classes.rowCount()
            self.table_classes.insertRow(row)

            orig_class = str(c)
            target_class = orig_class
            include_in_eval = True

            # Auto-match case-insensitive against YAML if we have one
            if global_yaml_classes:
                if orig_class.lower() in global_yaml_lower:
                    target_class = global_yaml_lower[orig_class.lower()]
                else:
                    # Unmatched class -> leave target_class as original name, 
                    # user can edit it manually in the table if they want to merge it.
                    pass

            self.table_classes.setItem(row, 0, QTableWidgetItem(orig_class))
            
            # Use QComboBox for target classes to easily select from YAML classes
            combo_target = QComboBox()
            combo_target.setEditable(True)
            if global_yaml_classes:
                # Deduplicate and sort, but keep exact cases
                combo_target.addItems(sorted(list(set(global_yaml_classes))))
            
            # Find index if it matches (case insensitive)
            idx = -1
            if global_yaml_classes:
                for i in range(combo_target.count()):
                    if combo_target.itemText(i).lower() == target_class.lower():
                        idx = i
                        break
            
            if idx >= 0:
                combo_target.setCurrentIndex(idx)
            else:
                if global_yaml_classes:
                    # Unmatched with YAML present, leave it blank to force user choice
                    combo_target.setCurrentText("")
                    include_in_eval = False # Uncheck by default so it doesn't break if they forget
                else:
                    # No YAML at all, just default to itself
                    combo_target.setCurrentText(target_class)
                
            self.table_classes.setCellWidget(row, 1, combo_target)

            chk = QCheckBox()
            chk.setChecked(include_in_eval)
            self.table_classes.setCellWidget(row, 2, chk)
            self.table_classes.setItem(row, 3, QTableWidgetItem(""))

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

    def open_advanced_visualizer(self):
        try:
            import viat.widgets.visualizer as vis
            # Keep reference to avoid garbage collection
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
        for row in range(self.table_classes.rowCount()):
            orig_name = self.table_classes.item(row, 0).text()
            try:
                target_name = self.table_classes.cellWidget(row, 1).currentText()
            except AttributeError:
                target_name = self.table_classes.item(row, 1).text()
            include = self.table_classes.cellWidget(row, 2).isChecked()
            video_name_item = self.table_classes.item(row, 3)
            video_name = video_name_item.text().strip() if video_name_item else ""

            if not include:
                target_name = "__IGNORE__"

            if video_name:
                if video_name not in self.video_class_mappings:
                    self.video_class_mappings[video_name] = {}
                self.video_class_mappings[video_name][orig_name] = target_name
                try:
                    self.video_class_mappings[video_name][int(orig_name)] = target_name
                except ValueError: pass
            else:
                self.class_map[orig_name] = target_name
                try:
                    self.class_map[int(orig_name)] = target_name
                except ValueError: pass

        ignored_videos_str = self.edit_ignored_videos.text().strip()
        ignored_videos = [v.strip() for v in ignored_videos_str.split(',')] if ignored_videos_str else []

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

    def on_evaluation_finished(self, success, msg, results):
        self.btn_evaluate.setEnabled(True)
        self.progress_bar.setVisible(False)

        if success and results:
            self.btn_open_results.setEnabled(True)

            # Update Analytics Plots
            if 'bar_path' in results and os.path.exists(results['bar_path']):
                self.lbl_plot_bar.setPixmap(QPixmap(results['bar_path']).scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            if 'size_path' in results and os.path.exists(results['size_path']):
                self.lbl_plot_size.setPixmap(QPixmap(results['size_path']).scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            if results.get('merge_path') and os.path.exists(results['merge_path']):
                self.lbl_plot_merge.setPixmap(QPixmap(results['merge_path']).scaled(400, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation))

            # Set 6 Advanced Diagnostic Pixmaps
            diag_mappings = [
                ('cm_path', self.lbl_diag_cm, 450, 350),
                ('calib_path', self.lbl_diag_calib, 550, 320),
                ('iou_path', self.lbl_diag_iou, 450, 320),
                ('ar_path', self.lbl_diag_ar, 450, 320),
                ('track_path', self.lbl_diag_track, 450, 320),
                ('spatial_path', self.lbl_diag_spatial, 550, 320),
            ]
            for key, label_widget, w, h in diag_mappings:
                if key in results and os.path.exists(results[key]):
                    label_widget.setPixmap(QPixmap(results[key]).scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation))


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

        for row in range(self.table_classes.rowCount()):
            orig_name = self.table_classes.item(row, 0).text()
            try:
                target_name = self.table_classes.cellWidget(row, 1).currentText()
            except AttributeError:
                target_name = self.table_classes.item(row, 1).text()
            include = self.table_classes.cellWidget(row, 2).isChecked()
            profile["class_mappings"].append({"orig": orig_name, "target": target_name, "include": include})

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

        mappings = profile.get("class_mappings", [])
        self.table_classes.setRowCount(0)
        for m in mappings:
            row = self.table_classes.rowCount()
            self.table_classes.insertRow(row)
            self.table_classes.setItem(row, 0, QTableWidgetItem(m.get("orig", "")))
            
            combo = QComboBox()
            combo.setEditable(True)
            combo.setCurrentText(m.get("target", ""))
            self.table_classes.setCellWidget(row, 1, combo)
            
            chk = QCheckBox()
            chk.setChecked(m.get("include", True))
            self.table_classes.setCellWidget(row, 2, chk)
            
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
