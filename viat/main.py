"""
Video Annotation Tool (VAT) - Main Application

This module contains the main application window and program entry point for the
Video Annotation Tool. It provides the UI framework and coordinates between the
different components of the application.
"""
try:
    import torch
except ImportError:
    pass
import os
import random
import math
import cv2
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel, QAction, QFileDialog, QStatusBar, QComboBox, QMessageBox, QListWidget, QListWidgetItem, QDialog, QFormLayout, QSpinBox, QDialogButtonBox, QLineEdit, QColorDialog, QActionGroup, QGroupBox, QDoubleSpinBox, QApplication, QProgressBar, QCheckBox, QTextEdit, QProgressDialog, QPlainTextEdit
from PyQt5.QtCore import Qt, QTimer, QRect, QDateTime, QEvent, QThread, pyqtSignal,QRectF
from PyQt5.QtGui import QColor, QIcon, QImage, QPixmap
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from .canvas import VideoCanvas
from .annotation import BoundingBox, AnnotationManager, ClassManager
from .widgets import AnnotationDock, StyleManager, ClassDock, AnnotationToolbar
from .interpolation import InterpolationManager
from .logger import VIATLogger, log_exceptions, logger
from .widgets.auto_annotate_dialog import AutoAnnotateDialog
from .widgets.scene_detect_dialog import SceneDetectDialog
from .utils.zero_shot_manager import ZeroShotManager
from .tracking.nossort import NOCSORT
from .tracking.manager import TrackerManager
from .widgets.tracking_dialog import TrackingDialog
from .utils.ui_creator import UICreator
from .utils.sam_manager import SamManager
from .utils.sam3_native_manager import Sam3NativeManager
from .utils.sam2_trt_manager import Sam2TrtManager
import numpy as np
import json
from viat.utils import save_project, load_project, export_annotations, get_config_directory, get_recent_projects, get_last_project, save_last_state, load_last_state, export_image_dataset_pascal_voc, export_image_dataset_yolo, export_image_dataset_coco, export_standard_annotations, mse_similarity, calculate_frame_hash, create_thumbnail, import_annotations, UICreator, export_dataset_dialog, export_dataset, import_dataset_dialog, load_dataset, PerfomanceManger, load_project_with_backup, backup_before_save
from viat.utils.dataset_manager import detect_folder_type as _viat_detect_folder_type, scan_dataset, load_dataset_into_app, DatasetInfo, SplitInfo
from viat.utils.dataset_ops import remove_bad_frames as _viat_remove_bad_frames, remap_class as _viat_remap_class, merge_classes as _viat_merge_classes
from viat.utils.dataset_ops import move_frames_to as _viat_move_frames_to, remove_bad_frames as _viat_remove_bad_frames_v2, move_to_removed as _viat_move_to_removed, move_to_review_label as _viat_move_to_review_label, remove_grayscale_images as _viat_remove_grayscale, remove_duplicate_groups as _viat_remove_dup_groups, remove_class_and_images as _viat_remove_class_and_images, remap_class as _viat_remap_class_v2, merge_classes as _viat_merge_classes_v2, auto_import_detections as _viat_auto_import_detections
from viat.utils.dataset_log import init_dataset_log as _viat_init_dataset_log, append_dataset_log as _viat_append_dataset_log
from viat.utils.dataset_manager import load_viat_json_for_video as _viat_load_json_video
from viat.utils.video_border import detect_and_adjust_borders as _viat_detect_adjust_borders
from viat.utils.video_border import detect_video_borders as _viat_detect_borders
from viat.utils.object_visibility import ObjectVisibilityManager as _ViatObjectVisibilityManager
from viat.utils.performance import PerformanceManager as _ViatPerformanceManager
from viat.utils.seg_video_labeler import SegmentationVideoLabeler as _ViatSegLabeler
from viat.utils.dataset_merger import merge_dataset_into_target as _viat_merge_dataset, find_unmatched_classes as _viat_find_unmatched_classes
from viat.utils.icon_provider import IconProvider
from viat.utils.sam_manager import SamManager
from natsort import natsorted
from copy import deepcopy
from pathlib import Path


def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    if float(boxAArea + boxBArea - interArea) == 0:
        return 0.0
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def apply_nms(detections, iou_threshold=0.5, class_agnostic=False,
    class_priority=None):
    if len(detections) == 0:
        return []
    if class_agnostic:
        if class_priority:
            priority_map = {c.lower(): i for i, c in enumerate(class_priority)}
            dets = sorted(detections, key=lambda x: (-priority_map.get(x[
                'class_name'].lower(), 999), x['score']), reverse=True)
        else:
            dets = sorted(detections, key=lambda x: x['score'], reverse=True)
        keep = []
        while len(dets) > 0:
            best = dets.pop(0)
            keep.append(best)
            remaining = []
            for other in dets:
                iou = calculate_iou(best['box'], other['box'])
                if iou < iou_threshold:
                    remaining.append(other)
            dets = remaining
        return keep
    by_class = {}
    for det in detections:
        c = det['class_name']
        if c not in by_class:
            by_class[c] = []
        by_class[c].append(det)
    final_detections = []
    for c, dets in by_class.items():
        dets = sorted(dets, key=lambda x: x['score'], reverse=True)
        keep = []
        while len(dets) > 0:
            best = dets.pop(0)
            keep.append(best)
            remaining = []
            for other in dets:
                iou = calculate_iou(best['box'], other['box'])
                if iou < iou_threshold:
                    remaining.append(other)
            dets = remaining
        final_detections.extend(keep)
    return final_detections


class AutoLabelWorker(QThread):
    progress_updated = pyqtSignal(int)
    frame_processed = pyqtSignal(int, list)
    frame_started = pyqtSignal(int)
    finished_processing = pyqtSignal()
    error_occurred = pyqtSignal(str)

    def __init__(self, config, is_image_dataset, image_files,
        video_filename, zero_shot_manager, sam_manager, main_window=None):
        super().__init__()
        self.config = config
        self.is_image_dataset = is_image_dataset
        self.image_files = image_files
        self.video_filename = video_filename
        self.zero_shot_manager = zero_shot_manager
        self.sam_manager = sam_manager
        self.main_window = main_window
        self.is_cancelled = False
        self.sam3_native_manager = getattr(main_window,
            'sam3_native_manager', None) if main_window else None

    def run(self):
        try:
            start_frame = self.config.get('start_frame', 0)
            end_frame = self.config.get('end_frame', 0)
            strategy = self.config.get('strategy', 'independent')
            seg_model = self.config.get('seg_model')
            helper_classes = self.config.get('helper_classes', [])
            # Also check classes_config for helper entries in case helper_classes
            # wasn't populated (e.g. config printed before run_auto_label_dataset ran)
            if not helper_classes:
                helper_classes = [c for c in self.config.get('classes_config', [])
                                  if 'helper' in c.get('action', '').lower()]
            # Determine if we have a SAM3 manager ready for helper refinement
            _sam3_mgr = self.sam3_native_manager
            _has_sam3 = _sam3_mgr is not None and _sam3_mgr.is_available() and _sam3_mgr.video_predictor is not None

            def calculate_iou(boxA, boxB):
                xA = max(boxA[0], boxB[0])
                yA = max(boxA[1], boxB[1])
                xB = min(boxA[2], boxB[2])
                yB = min(boxA[3], boxB[3])
                interArea = max(0, xB - xA) * max(0, yB - yA)
                boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
                boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
                iou = interArea / float(boxAArea + boxBArea - interArea
                    ) if boxAArea + boxBArea - interArea > 0 else 0
                return iou

            def get_frame_generator():
                if self.is_image_dataset:
                    for f_idx in range(start_frame, end_frame + 1):
                        if self.is_cancelled:
                            break
                        if f_idx < len(self.image_files):
                            frame = cv2.imread(self.image_files[f_idx])
                            if frame is not None:
                                yield f_idx, frame
                else:
                    # Open VideoCapture ONCE and read sequentially —
                    # avoids the massive overhead of open+seek+close per frame.
                    cap = cv2.VideoCapture(self.video_filename)
                    try:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                        for f_idx in range(start_frame, end_frame + 1):
                            if self.is_cancelled:
                                break
                            ret, frame = cap.read()
                            if ret:
                                yield f_idx, frame
                            else:
                                break
                    finally:
                        cap.release()
            if strategy == 'tracking':
                pass
            frame_detections = {f_idx: [] for f_idx in range(start_frame, end_frame + 1)}
            min_score = self.config.get('min_score', 0.05)
            dedup_iou_thresh = self.config.get('dedup_iou', 0.7)
            classes_config = self.config.get('classes_config', [])
            det_models = self.config.get('det_models', [])

            dedup_rules = {}
            for c in classes_config:
                if not isinstance(c, dict):
                    continue
                if c.get('action') == 'Detect (Zero-Shot)':
                    rule = c.get('dedup_against', '').strip().lower()
                    if rule == '*':
                        dedup_rules[c['name']] = '*'
                    elif rule:
                        dedup_rules[c['name']] = [x.strip() for x in rule.split(',')]
                    else:
                        dedup_rules[c['name']] = [c['name'].lower()]

            # Build detect prompts once (text labels to search for with zero-shot models).
            # Use extract_prompt if set, otherwise class name.
            # Only include Detect-action classes; pure Helper classes are excluded.
            detect_prompts = []
            for _c in classes_config:
                if isinstance(_c, dict):
                    action = _c.get('action', '')
                    if 'detect' in action.lower() or not action:
                        prompt = _c.get('extract_prompt', '').strip() or _c.get('name', '').strip()
                        if prompt:
                            detect_prompts.append(prompt)
                elif isinstance(_c, str) and _c.strip():
                    detect_prompts.append(_c.strip())

            processed_count = 0
            for f_idx, frame in get_frame_generator():
                if self.is_cancelled:
                    break
                self.frame_started.emit(f_idx)
                
                current_detections = []

                # --- Helper class refinement (SAM3-based) ---
                # Activates when helper_classes exist AND a SAM3 manager is ready.
                # Does NOT require seg_model to be set.
                _use_helper = bool(helper_classes) and (_has_sam3 or (seg_model and 'sam3' in (seg_model or '').lower()))
                if _use_helper:
                    existing_anns = self.config.get('existing_annotations_data', {}).get(f_idx, [])
                    for i, ann in enumerate(existing_anns):
                        matching_configs = [hc for hc in helper_classes if ann.class_name == hc['name']]
                        if matching_configs:
                            box = [ann.rect.x(), ann.rect.y(),
                                   ann.rect.x() + ann.rect.width(),
                                   ann.rect.y() + ann.rect.height()]
                            current_detections.append({
                                'box': box,
                                'class_name': ann.class_name,
                                'score': 1.0,
                                'original_ann_idx': i,
                                'is_helper': True,
                                'helper_configs': matching_configs
                            })

                for model_name in det_models:
                    if self.is_cancelled:
                        break
                    if model_name == 'existing_annotations':
                        # Load existing frame annotations as detections to be
                        # refined by the seg_model (SAM3 / SAM2).
                        # If classes_config has Detect/Helper entries, only load
                        # those classes; otherwise load ALL existing annotations.
                        refine_class_filter = set()
                        for _c in classes_config:
                            if _c.get('action', '') not in ('Ignore', 'Remove Labels'):
                                refine_class_filter.add(_c['name'])
                        ex_anns = self.config.get('existing_annotations_data', {}).get(f_idx, [])
                        for i, ann in enumerate(ex_anns):
                            if refine_class_filter and ann.class_name not in refine_class_filter:
                                continue
                            b = [ann.rect.x(), ann.rect.y(),
                                 ann.rect.x() + ann.rect.width(),
                                 ann.rect.y() + ann.rect.height()]
                            # Guard: skip zero-area boxes (bad QRect rounding)
                            if b[2] <= b[0] or b[3] <= b[1]:
                                # Try float rect
                                try:
                                    rf = ann.rect
                                    b = [rf.x(), rf.y(),
                                         rf.x() + rf.width(),
                                         rf.y() + rf.height()]
                                except Exception:
                                    pass
                            if b[2] > b[0] and b[3] > b[1]:
                                current_detections.append({
                                    'box': [float(v) for v in b],
                                    'class_name': ann.class_name,
                                    'score': 1.0,
                                    'original_ann_idx': i,
                                    'is_helper': False,  # SAM3 refines in-place
                                    'is_existing': True,
                                })
                        continue  # don't try to load as a zero-shot model
                    # Skip zero-shot models when there are no detect-action classes
                    if not detect_prompts:
                        continue
                    success, _ = self.zero_shot_manager.load_model(model_name)
                    if success:
                        self.zero_shot_manager.set_classes(detect_prompts)
                        # Pass min_score so the manager does NOT silently discard
                        # detections that pass the user-configured threshold.
                        model_dets = self.zero_shot_manager.predict(frame, score_threshold=min_score)
                        for det in model_dets:
                            det['is_helper'] = False
                        current_detections.extend(model_dets)

                existing_anns = self.config.get('existing_annotations_data', {}).get(f_idx, [])
                helpers = [d for d in current_detections if d.get('is_helper')]
                detected = [d for d in current_detections if not d.get('is_helper') and d.get('score', 1.0) >= min_score]
                detected = apply_nms(detected, iou_threshold=0.5, class_agnostic=True, class_priority=self.config.get('classes', []))
                
                filtered_detected = []
                for d in detected:
                    # is_existing detections are existing annotation boxes being
                    # refined by SAM3 - they naturally have IOU ~1.0 with originals
                    # and must bypass dedup so SAM3 can process them.
                    if d.get('is_existing'):
                        filtered_detected.append(d)
                        continue
                    is_duplicate = False
                    c_name = d['class_name']
                    rule = dedup_rules.get(c_name, [c_name.lower()])
                    for ann in existing_anns:
                        ann_class_lower = ann.class_name.lower()
                        should_check = False
                        if rule == '*':
                            should_check = True
                        elif ann_class_lower in rule:
                            should_check = True
                        if should_check:
                            boxA = d['box']
                            boxB = [ann.rect.x(), ann.rect.y(), ann.rect.x() + ann.rect.width(), ann.rect.y() + ann.rect.height()]
                            iou = calculate_iou(boxA, boxB)
                            if iou > dedup_iou_thresh:
                                is_duplicate = True
                                break
                    if not is_duplicate:
                        filtered_detected.append(d)
                
                final_detections = helpers + filtered_detected
                frame_anns = []
                
                for det in final_detections:
                    box = det['box']
                    c_name = det['class_name']
                    score = det['score']
                    polygon = det.get('segmentation')
                    is_helper = det.get('is_helper', False)
                    
                    if not polygon:
                        if is_helper:
                            # Helper refinement: use SAM3 native with box prompt first,
                            # then optionally text prompt. Send box and text SEPARATELY
                            # because combining them in one add_prompt call can produce
                            # empty masks in some SAM3 versions.
                            matching_configs = det.get('helper_configs', [])
                            success_polygon = None
                            if _has_sam3:
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                for hc in matching_configs:
                                    extract_prompt = hc.get('extract_prompt', '').strip()
                                    ignore_prompt = hc.get('ignore_prompt', '').strip()
                                    rename_to = hc.get('rename_to', '').strip()
                                    
                                    final_text_prompt = extract_prompt
                                    if extract_prompt and ignore_prompt:
                                        final_text_prompt = f"{extract_prompt}, but not {ignore_prompt}"
                                        
                                    # Try box + text first
                                    polygon_mask = _sam3_mgr.predict_mask_from_prompt(
                                        frame_rgb,
                                        box=box,
                                        text_prompt=final_text_prompt if final_text_prompt else None
                                    )
                                    if polygon_mask and isinstance(polygon_mask, list) and len(polygon_mask) > 0:
                                        success_polygon = [(float(pt[0]), float(pt[1])) for pt in polygon_mask]
                                        if rename_to:
                                            c_name = rename_to
                                        break
                                    elif final_text_prompt:
                                        # Fallback: box alone without text
                                        polygon_mask = _sam3_mgr.predict_mask_from_prompt(
                                            frame_rgb, box=box
                                        )
                                        if polygon_mask and isinstance(polygon_mask, list) and len(polygon_mask) > 0:
                                            success_polygon = [(float(pt[0]), float(pt[1])) for pt in polygon_mask]
                                            if rename_to:
                                                c_name = rename_to
                                            break
                            elif seg_model and 'sam3' in (seg_model or '').lower() and self.sam3_native_manager:
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                for hc in matching_configs:
                                    extract_prompt = hc.get('extract_prompt', '').strip()
                                    ignore_prompt = hc.get('ignore_prompt', '').strip()
                                    rename_to = hc.get('rename_to', '').strip()
                                    
                                    final_text_prompt = extract_prompt
                                    if extract_prompt and ignore_prompt:
                                        final_text_prompt = f"{extract_prompt}, but not {ignore_prompt}"
                                        
                                    polygon_mask = self.sam3_native_manager.predict_mask_from_prompt(
                                        frame_rgb, box=box,
                                        text_prompt=final_text_prompt if final_text_prompt else None
                                    )
                                    if polygon_mask and isinstance(polygon_mask, list) and len(polygon_mask) > 0:
                                        success_polygon = [(float(pt[0]), float(pt[1])) for pt in polygon_mask]
                                        if rename_to:
                                            c_name = rename_to
                                        break
                            elif seg_model and self.sam_manager:
                                success_polygon = self.sam_manager.predict_mask_from_box(frame, box)
                            polygon = success_polygon
                        elif seg_model:
                            # Existing annotation refinement (is_existing=True) or
                            # plain detected box refinement via seg model
                            if 'sam3' in (seg_model or '').lower() and _has_sam3:
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                # Validate box before sending to SAM3
                                if box and box[2] > box[0] and box[3] > box[1]:
                                    polygon_mask = _sam3_mgr.predict_mask_from_prompt(
                                        frame_rgb, box=box, text_prompt=None
                                    )
                                    if polygon_mask and isinstance(polygon_mask, list) and len(polygon_mask) > 0:
                                        polygon = [(float(pt[0]), float(pt[1])) for pt in polygon_mask]
                            else:
                                polygon = self.sam_manager.predict_mask_from_box(frame, box)
                            
                        if polygon and len(polygon) > 0:
                            x_coords = [pt[0] for pt in polygon]
                            y_coords = [pt[1] for pt in polygon]
                            box = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
                            
                    frame_anns.append({'box': list(box), 'class_name': c_name, 'score': score, 'segmentation': polygon, 'source': 'refined' if is_helper else 'detected', 'original_ann_idx': det.get('original_ann_idx')})
                    
                self.frame_processed.emit(f_idx, frame_anns)
                processed_count += 1
                self.progress_updated.emit(processed_count)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error_occurred.emit(str(e))
        finally:
            self.finished_processing.emit()

    def cancel(self):
        self.is_cancelled = True


class VideoAnnotationTool(QMainWindow):
    """
    Main application window for the Video Annotation Tool.

    This class manages the UI components, video playback, and annotation functionality.
    It serves as the central coordinator between different parts of the application.
    """

    def __init__(self):
        """Initialize the main application window and its components."""
        super().__init__()
        self.setWindowTitle('Video Annotation Tool')
        self.setGeometry(100, 100, 1200, 800)
        self.init_properties()
        self.video_filename = ''
        self.setup_ui()
        self.canvas.smart_edge_enabled = False
        self.setup_autosave()
        self.locate_anything_manager = None
        self.sam_manager = None
        self.sam3_native_manager = None
        self.sam2_trt_manager = None
        self.init_managers()
        self.ui_creator.create_interpolation_ui()
        QApplication.instance().installEventFilter(self)
        self.dark_mode_enabled = False
        QTimer.singleShot(100, self.load_last_project)

    def toggle_dark_mode(self, enabled: bool):
        self.dark_mode_enabled = enabled
        if enabled:
            try:
                import os
                style_path = os.path.join(os.path.dirname(__file__),
                    'styles.qss')
                with open(style_path, 'r') as f:
                    self.setStyleSheet(f.read())
            except Exception as e:
                print(f'Failed to load stylesheet: {e}')
        else:
            self.setStyleSheet('')

    @log_exceptions
    def init_managers(self):
        self.annotation_manager = AnnotationManager(self, self.canvas)
        self.class_manager = ClassManager(self)
        self.interpolation_manager = InterpolationManager(self)
        self.performance_manager = PerfomanceManger(self, cache_capacity=200)
        from viat.utils.blur_manager import BlurManager
        self.blur_manager = BlurManager()
        self.viat_perf = self.performance_manager
        self.tracker_manager = TrackerManager()
        self.sam_manager = SamManager()
        self.sam3_native_manager = Sam3NativeManager()
        self.sam2_trt_manager = Sam2TrtManager()
        from viat.utils.fast_tracker_manager import FastTrackerManager
        self.fast_tracker_manager = FastTrackerManager()

    @log_exceptions
    def load_last_project(self):
        """Load the last project that was open."""
        if self.load_application_state():
            return
        last_project = get_last_project()
        if last_project and os.path.exists(last_project):
            self.load_project(last_project)

    @log_exceptions
    def init_properties(self):
        """Initialize the application properties and state variables."""
        self.auto_blur_labels = False
        self.auto_save_blur_on_switch = False
        self.duplicate_frames_enabled = True
        self.duplicate_frames_cache = {}
        self.frame_hashes = {}
        self.integration_mode = False
        self.integration_main_dataset = ''
        self.undo_stack = []
        self.redo_stack = []
        self.max_undo_steps = 20
        self.max_redo_steps = 20
        self.styles = {}
        self.icon_provider = IconProvider()
        self._class_refresh_scheduled = False
        self.setFocusPolicy(Qt.StrongFocus)
        for style_name in StyleManager.get_available_styles():
            method_name = f"set_{style_name.lower().replace(' ', '_')}_style"
            if hasattr(StyleManager, method_name):
                self.styles[style_name] = getattr(StyleManager, method_name)
            elif style_name.lower() == 'default':
                self.styles[style_name] = StyleManager.set_darkmodern_style
        self.annotation_methods = {'Rectangle':
            'Draw rectangular bounding boxes', 'Polygon':
            'Draw polygon shapes', 'Point': 'Mark specific points'}
        self.current_annotation_method = 'Rectangle'
        self.current_style = 'DarkModern'
        self.playback_speed = 1.0
        self.cap = None
        self.is_playing = False
        self.current_frame = 0
        self.total_frames = 0
        self.zoom_level = 1.0
        self.auto_bbox_mode = False
        self.sam_model_type = 'sam2.1_s.pt'
        self.zero_shot_manager = None
        self.sam_manager = SamManager()
        self.auto_show_attribute_dialog = True
        self.use_previous_attributes = True
        self.last_used_attributes = {}
        self.frame_annotations = {}
        self.frame_crops = {}
        self.canvas_class_attributes = {'Quad': {'Size': {'type': 'int',
            'default': -1, 'min': 0, 'max': 100}, 'Quality': {'type': 'int',
            'default': -1, 'min': 0, 'max': 100}}}
        self.class_thresholds = {}
        self.project_file = None
        self.project_modified = False
        self.autosave_timer = None
        self.autosave_enabled = True
        self.autosave_interval = 180000
        self.autosave_file = None
        self._annotations_imported = set()
        self.last_autosave_time = None
        self.tracking_mode_enabled = False
        self.verification_mode = False
        self.remove_unverified = False
        self.object_visibility_manager = None
        self.performance_manager = None
        self.seg_labeler = None
        self.is_image_dataset = False
        self.image_files = []
        self.deleted_frames = set()
        self.deleted_annotations = {}
        self.labeler_analytics = {
            "prompts": [],
            "tool_usage": {
                "zero_shot": 0,
                "tracking": 0,
                "interpolation": 0,
                "magic_wand": 0,
            }
        }

    @log_exceptions
    def setup_ui(self):
        """Set up the user interface."""
        self.ui_creator = UICreator(self)
        self.ui_creator.create_menu_bar()
        self.ui_creator.create_toolbar()
        self.ui_creator.create_dock_widgets()
        self.ui_creator.create_status_bar()
        self.ui_creator.setup_playback_timer()
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        self.integration_banner = QWidget()
        banner_layout = QHBoxLayout(self.integration_banner)
        banner_layout.setContentsMargins(5, 5, 5, 5)
        self.integration_label = QLabel(
            '<b>Integration Mode Active:</b> Please review the dataset and apply labels.'
            )
        self.integration_label.setStyleSheet('color: #E6A23C;')
        self.btn_finish_integration = QPushButton('Finish & Merge')
        self.btn_finish_integration.setStyleSheet(
            'background-color: #67C23A; color: white; font-weight: bold; padding: 5px;'
            )
        self.btn_finish_integration.clicked.connect(self.
            viat_finish_integration)
        banner_layout.addWidget(self.integration_label)
        banner_layout.addStretch()
        banner_layout.addWidget(self.btn_finish_integration)
        self.integration_banner.setVisible(False)
        layout.addWidget(self.integration_banner)
        self.canvas = VideoCanvas(self)
        layout.addWidget(self.canvas)
        if hasattr(self.canvas, 'annotationChanged'):
            self.canvas.annotationChanged.connect(self.save_undo_state)
        if hasattr(self.canvas, 'annotationMoved'):
            self.canvas.annotationMoved.connect(self.save_undo_state)
        if hasattr(self.canvas, 'annotationResized'):
            self.canvas.annotationResized.connect(self.save_undo_state)
        if hasattr(self.canvas, 'cropRectChanged'):
            self.canvas.cropRectChanged.connect(self.handle_crop_rect_changed)
        playback_controls = self.ui_creator.create_playback_controls()
        layout.addWidget(playback_controls)
        layout.setContentsMargins(5, 5, 5, 5)
        self.resize(1200, 800)
        self.setup_sam_interactive()
        self.setup_empty_frames_manager()
        self.setup_uncertain_frames_manager()
        self.setup_class_frames_manager()
        self.setup_video_manager()
        self.setup_evaluation_inspector()
        self.setWindowTitle('VIAT - Video Image Annotation Tool')
        self.setWindowIcon(self.icon_provider.get_icon('app-icon'))
        self.viat_setup_extra_menus()

    @log_exceptions
    def handle_crop_rect_changed(self, rect):
        """Handle changes to the crop rectangle from the canvas."""
        if rect:
            self.frame_crops[self.current_frame] = rect
            if hasattr(self, 'crop_settings_dock'):
                # Block signals to avoid infinite loops if spinboxes are connected
                self.crop_settings_dock.width_spin.blockSignals(True)
                self.crop_settings_dock.height_spin.blockSignals(True)
                self.crop_settings_dock.width_spin.setValue(rect.width())
                self.crop_settings_dock.height_spin.setValue(rect.height())
                self.crop_settings_dock.width_spin.blockSignals(False)
                self.crop_settings_dock.height_spin.blockSignals(False)

    @log_exceptions
    def viat_setup_extra_menus(self):
        """Inject menu items for all new VIAT features into the menu bar.

        This is called from setup_ui() AFTER UICreator has built the
        existing menus, so we just add a new 'Dataset' menu and items to
        the 'Edit' and 'View' menus without touching UICreator.
        """
        menubar = self.menuBar()
        file_menu = None
        for action in menubar.actions():
            if action.text().replace('&', '') == 'File':
                file_menu = action.menu()
                break
                
        if file_menu:
            file_menu.addSeparator()
            
            self.auto_save_project_on_fast_export_act = file_menu.addAction('Save Project on Fast Export')
            self.auto_save_project_on_fast_export_act.setCheckable(True)
            self.auto_save_project_on_fast_export_act.setChecked(True)
            
            act = file_menu.addAction('Fast Export Video & Next')
            act.setShortcut('Ctrl+Shift+E')
            act.triggered.connect(lambda: self.fast_export_video_and_next())
            
            act = file_menu.addAction('Delete Video & Next')
            act.setShortcut('Ctrl+Shift+D')
            act.triggered.connect(lambda: self.delete_video_and_next())
            
            act = file_menu.addAction('Compare Raya Annotations...')
            act.triggered.connect(lambda: self.compare_raya_annotations())
            
            act = file_menu.addAction('View Labeler Analytics from JSON...')
            act.triggered.connect(lambda: self.view_labeler_analytics())
            
            act = file_menu.addAction('Model Evaluation...')
            act.triggered.connect(lambda: self.open_evaluation_dialog())

            
        dataset_menu = menubar.addMenu('&Dataset')
        img_menu = dataset_menu.addMenu('Image Dataset Operations')
        act = img_menu.addAction('Export Image Dataset...')
        act.triggered.connect(lambda : self.export_image_dataset())
        act = img_menu.addAction('Import Segmentation Masks (to BBoxes)...')
        act.triggered.connect(lambda : self.viat_import_segmentation_masks())
        act = img_menu.addAction('Convert Segmentation to BBox Project')
        act.triggered.connect(lambda : self.viat_convert_segmentation_to_bbox_project())
        img_menu.addSeparator()
        act = img_menu.addAction('Rotate Clockwise 90Â°')
        act.triggered.connect(lambda : self.rotate_image_dataset('cw'))
        act = img_menu.addAction('Rotate Counter-Clockwise 90Â°')
        act.triggered.connect(lambda : self.rotate_image_dataset('ccw'))
        img_menu.addSeparator()
        act = img_menu.addAction('Remove Current Image')
        act.setShortcut('Shift+X')
        act.triggered.connect(lambda : self.viat_move_current_to_removed())
        act = img_menu.addAction('Move to Review Label (CHANGE LABEL)')
        act.setShortcut('Shift+R')
        act.triggered.connect(lambda : self.viat_move_current_to_review_label()
            )
        img_menu.addSeparator()
        act = img_menu.addAction('Remove Grayscale Images...')
        act.triggered.connect(lambda : self.viat_remove_grayscale())
        act = img_menu.addAction('Remove Roboflow Duplicates...')
        act.triggered.connect(lambda : self.viat_remove_duplicates())
        act = img_menu.addAction('Remove Class + Images...')
        act.triggered.connect(lambda : self.
            viat_remove_class_and_images_dialog())
        img_menu.addSeparator()
        act = img_menu.addAction('Remove Bad Frames (batch)...')
        act.triggered.connect(lambda : self.remove_bad_frames_dialog())
        act = img_menu.addAction('Remap Class...')
        act.triggered.connect(lambda : self.remap_class_dialog())
        act = img_menu.addAction('Merge Classes...')
        act.triggered.connect(lambda : self.merge_classes_dialog())
        img_menu.addSeparator()
        act = img_menu.addAction('Dataset Statistics...')
        act.triggered.connect(lambda : self.viat_dataset_stats())
        act = img_menu.addAction('View Dataset Log (DATASET_LOG.md)')
        act.triggered.connect(lambda : self.viat_view_dataset_log())
        img_menu.addSeparator()
        act = img_menu.addAction(
            'Auto-Import Detections (move to review_label)...')
        act.triggered.connect(lambda : self.viat_auto_import_detections())
        img_menu.addSeparator()
        act = img_menu.addAction('Merge Dataset into Current...')
        act.triggered.connect(lambda : self.viat_merge_dataset())
        img_menu.addSeparator()
        act = img_menu.addAction('Extract Single Class from Datasets...')
        act.triggered.connect(lambda : self.viat_extract_single_class_dataset()
            )
        act = img_menu.addAction('Batch Prediction Queue Builder...')
        act.triggered.connect(lambda : self.
            viat_launch_batch_prediction_queue())
        act = img_menu.addAction('Remove Background Images (Percentage)...')
        act.triggered.connect(lambda : self.viat_remove_background_images())
        act = img_menu.addAction('Dataset Cleaner (Remove Nested Labels)...')
        act.triggered.connect(lambda : self.open_dataset_cleaner_dialog())
        act = img_menu.addAction('Dataset Integration Wizard (Roadmap)...')
        act.triggered.connect(lambda : self.viat_launch_integration_wizard())
        img_menu.addSeparator()
        act = img_menu.addAction('Auto Blur Labels...')
        act.triggered.connect(lambda : self.open_auto_blur_dialog())
        vid_menu = dataset_menu.addMenu('Video Annotation Operations')
        act = vid_menu.addAction('Auto Blur Labels...')
        act.triggered.connect(lambda : self.open_auto_blur_dialog())
        act = vid_menu.addAction('Split Video by Scene Cuts...')
        act.triggered.connect(lambda : self.viat_split_video_scenes())
        act = vid_menu.addAction('Import VIAT JSON Annotations...')
        act.triggered.connect(lambda : self.viat_import_video_json())
        act = vid_menu.addAction('Fix Video Borders (remove/clip labels)...')
        act.triggered.connect(lambda : self.viat_detect_and_fix_borders())
        act = vid_menu.addAction('Object Visibility Mode...')
        act.triggered.connect(lambda : self.viat_start_object_visibility_mode()
            )
        vid_menu.addSeparator()
        seg_menu = vid_menu.addMenu('Segmentation Video')
        act = seg_menu.addAction('Pick Object Color...')
        act.triggered.connect(lambda : self.viat_seg_video_pick_color())
        act = seg_menu.addAction('Track All Objects')
        act.triggered.connect(lambda : self.viat_seg_video_track_all())
        act = seg_menu.addAction('Export Seg Video JSON...')
        act.triggered.connect(lambda : self.viat_seg_video_export_json())
        vid_menu.addSeparator()
        act = vid_menu.addAction('Export VIAT JSON...')
        act.triggered.connect(lambda : self.viat_export_json())
        act = vid_menu.addAction('Merge Videos in Folder...')
        act.triggered.connect(lambda : self.viat_merge_videos_in_folder())
        act = vid_menu.addAction('Toggle Remove Unverified Labels')
        act.setShortcut('Shift+U')
        act.triggered.connect(lambda : self.viat_toggle_remove_unverified())
        act = vid_menu.addAction('Frame Cache Stats...')
        act.triggered.connect(lambda : self.viat_perf_stats())
        act = vid_menu.addAction('Clear Frame Cache')
        act.triggered.connect(lambda : self.viat_clear_cache())
        view_menu = None
        for action in menubar.actions():
            if action.text() in ('&View', 'View'):
                view_menu = action.menu()
                break
        if view_menu is None:
            view_menu = menubar.addMenu('&View')
        view_menu.addSeparator()
        act = view_menu.addAction('Toggle Segmentation Display')
        act.setShortcut('Shift+S')
        act.triggered.connect(lambda : self.viat_toggle_segmentation())
        act = view_menu.addAction('Toggle Attribute Display')
        act.setShortcut('Shift+A')
        act.triggered.connect(lambda : self.viat_toggle_attribute_display())
        act = view_menu.addAction('Dataset Statistics...')
        act.triggered.connect(lambda : self.viat_dataset_stats())
        act = view_menu.addAction('View Dataset Log...')
        act.triggered.connect(lambda : self.viat_view_dataset_log())
        self._viat_extra_menus = dataset_menu

    @log_exceptions
    def viat_merge_videos_in_folder(self):
        """Prompt user to select a folder and merge all videos in it."""
        from PyQt5.QtWidgets import QFileDialog, QMessageBox, QApplication
        import sys
        import os
        from pathlib import Path
        
        default_dir = os.path.dirname(self.video_filename) if getattr(self, 'video_filename', None) else ''
        folder_path = QFileDialog.getExistingDirectory(self, 'Select Folder to Merge Videos', default_dir)
        
        if not folder_path:
            return
            
        # Optional: warn the user that this might take time
        reply = QMessageBox.question(self, 'Confirm Merge',
            f'Are you sure you want to merge all annotated videos in:\n{folder_path}?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            
        if reply != QMessageBox.Yes:
            return
            
        # Ensure current video cap is released if it is inside the folder to avoid locking issues (mostly Windows, but safe everywhere)
        if getattr(self, 'cap', None) and getattr(self, 'video_filename', None) and self.video_filename.startswith(folder_path):
            self.cap.release()
            self.cap = None

        try:
            root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            if root_path not in sys.path:
                sys.path.insert(0, root_path)
                
            from many2single import merge_dataset_programmatic
            
            folder = Path(folder_path)
            out_video = folder / 'outvideo.mp4'
            out_labels = folder / 'outvideo.txt'
            
            self.statusBar.showMessage('Merging videos... This may take a while.')
            QApplication.processEvents()
            
            success, msg = merge_dataset_programmatic(folder, out_video, out_labels)
            
            if success:
                QMessageBox.information(self, "Merge Successful", msg)
            self.statusBar.showMessage('Merge complete', 5000)
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Merge Error", f"Failed to merge videos:\n{e}")

    @log_exceptions
    def open_dataset_cleaner_dialog(self):
        """Launches the Dataset Cleaner Dialog."""
        from .widgets.dataset_cleaner_dialog import DatasetCleanerDialog
        dialog = DatasetCleanerDialog(self)
        dialog.exec_()

    @log_exceptions
    def open_auto_blur_dialog(self):
        """Launches the Auto Blur Dialog."""
        from .widgets.auto_blur_dialog import AutoBlurDialog
        from PyQt5.QtWidgets import QDialog
        dialog = AutoBlurDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            self.apply_auto_blur(dialog.settings)

    @log_exceptions
    def apply_auto_blur(self, settings):
        from PyQt5.QtWidgets import QProgressDialog, QMessageBox
        from PyQt5.QtCore import Qt
        
        frames_to_process = []
        if settings.get("all_frames", False):
            if hasattr(self, 'is_image_dataset') and self.is_image_dataset:
                frames_to_process = list(range(self.total_frames))
            else:
                frames_to_process = list(self.frame_annotations.keys())
                if not frames_to_process and self.total_frames > 0:
                    frames_to_process = list(range(self.total_frames))
        else:
            frames_to_process = [self.current_frame]
            
        if not frames_to_process:
            return
            
        progress = QProgressDialog("Applying Auto Blur...", "Cancel", 0, len(frames_to_process), self)
        progress.setWindowModality(Qt.WindowModal)
        
        blurred_count = 0
        removed_count = 0
        blur_kernel = getattr(self.canvas, 'blur_kernel', 151)
        w_frame = self.canvas.pixmap.width() if self.canvas.pixmap else 1920
        h_frame = self.canvas.pixmap.height() if self.canvas.pixmap else 1080
        
        for i, f_idx in enumerate(frames_to_process):
            if progress.wasCanceled():
                break
            progress.setValue(i)
            
            if f_idx not in self.frame_annotations:
                continue
                
            annots = self.frame_annotations[f_idx]
            to_blur = []
            
            for ann in annots:
                rect = ann.rect
                area = rect.width() * rect.height()
                
                if settings["small"]["enabled"] and area <= settings["small"]["max_area"]:
                    to_blur.append(ann)
                    continue
                if settings["big"]["enabled"] and area >= settings["big"]["min_area"]:
                    to_blur.append(ann)
                    continue
                    
                if settings["aspect_ratio"]["enabled"]:
                    if rect.height() > 0:
                        ar = rect.width() / float(rect.height())
                        if ar < settings["aspect_ratio"]["min"] or ar > settings["aspect_ratio"]["max"]:
                            to_blur.append(ann)
                            continue
                            
                if settings["corner"]["enabled"]:
                    dist = settings["corner"]["dist"]
                    if rect.left() <= dist or rect.right() >= (w_frame - dist) or rect.top() <= dist or rect.bottom() >= (h_frame - dist):
                        to_blur.append(ann)
                        continue
                        
                if settings["occluded"]:
                    is_occ = False
                    for k, v in ann.attributes.items():
                        if str(k).lower() in ["occluded", "occlusion"]:
                            if str(v).lower() in ["true", "1", "yes"]:
                                is_occ = True
                                break
                    if is_occ:
                        to_blur.append(ann)
                        continue
                        
            if settings["recursive"] and to_blur:
                added_new = True
                while added_new:
                    added_new = False
                    for ann in annots:
                        if ann not in to_blur:
                            for b_ann in to_blur:
                                if ann.rect.intersects(b_ann.rect):
                                    to_blur.append(ann)
                                    added_new = True
                                    break
                                    
            if to_blur:
                if not hasattr(self, 'blur_manager') or self.blur_manager is None:
                    from viat.utils.blur_manager import BlurManager
                    self.blur_manager = BlurManager()
                    
                for ann in to_blur:
                    self.blur_manager.add_bbox_region(f_idx, ann.rect, blur_kernel)
                    blurred_count += 1
                    
                if settings["remove_bbox"]:
                    new_annots = [a for a in annots if a not in to_blur]
                    removed_count += (len(annots) - len(new_annots))
                    self.frame_annotations[f_idx] = new_annots
                    if f_idx == self.current_frame:
                        self.canvas.annotations = new_annots
                        
        progress.setValue(len(frames_to_process))
        if blurred_count > 0:
            self.project_modified = True
            if hasattr(self, '_refresh_blur_display'):
                self._refresh_blur_display()
            self.canvas.update()
            if settings["remove_bbox"]:
                self.update_annotation_list()
            QMessageBox.information(self, "Auto Blur Complete", f"Blurred {blurred_count} objects.\\nRemoved {removed_count} bounding boxes.")
        else:
            QMessageBox.information(self, "Auto Blur Complete", "No objects matched the criteria.")

    @log_exceptions
    def viat_launch_integration_wizard(self):
        """Launches the Dataset Integration Roadmap Setup."""
        from PyQt5.QtWidgets import QApplication, QDialog, QMessageBox
        from widgets.dataset_wizard_dialog import DatasetWizardDialog
        self.update_frame_annotations()
        if getattr(self, 'is_image_dataset', False) and hasattr(self,
            '_viat_dataset_info'):
            reply = QMessageBox.question(self, 'Save Current Changes',
                'You are currently working on a dataset. Do you want to save your current annotations to disk before starting the integration wizard?'
                , QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
                QMessageBox.Yes)
            if reply == QMessageBox.Cancel:
                return
            if reply == QMessageBox.Yes:
                from viat.utils.dataset_manager import update_dataset_labels
                self.statusBar.showMessage('Updating dataset labels...')
                QApplication.processEvents()
                updated, errors = update_dataset_labels(self.
                    _viat_dataset_info, self.frame_annotations, self.
                    image_files, current_classes=list(self.canvas.
                    class_colors.keys()))
                self.statusBar.showMessage(f'Updated {updated} dataset labels.'
                    )
        elif getattr(self, 'project_modified', False):
            if not self.check_unsaved_changes():
                return
        dialog = DatasetWizardDialog(self, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            settings = dialog.settings
            if not settings:
                return
                
            if dialog.is_video_mode:
                # Video integration execution logic
                import tempfile
                import shutil
                from viat.utils.dataset_manager import export_dataset
                from viat.utils.dataset_merger import merge_dataset_into_target
                
                temp_dir = tempfile.mkdtemp(prefix="viat_video_integration_")
                try:
                    # Config for export
                    export_config = {
                        "output_dir": temp_dir,
                        "format": "yolo",
                        "make_splits": False,
                        "valid_pct": 0,
                        "video_width": 1920 if settings.get("normalize_res") else None,
                        "video_height": 1080 if settings.get("normalize_res") else None,
                        "resize_mode": "pad" if settings.get("remove_padding") else None,
                        "include_classes": True
                    }
                    
                    # Virtual image files
                    video_base = os.path.splitext(os.path.basename(self.video_filename))[0]
                    image_files = [f"{video_base}_frame_{i:06d}.jpg" for i in range(self.total_frames)]
                    
                    # Extract frames
                    from viat.utils.task_runner import run_task_with_progress
                    run_task_with_progress(self, 'Exporting Video Frames',
                        'Extracting frames for integration...', export_dataset, self, export_config,
                        image_files, self.frame_annotations, self.canvas.class_colors,
                        maximum=100)
                        
                    # Merge temp dataset into main target dataset
                    self.statusBar.showMessage('Merging video frames into target dataset...')
                    QApplication.processEvents()
                    
                    result = merge_dataset_into_target(
                        self,
                        source_folder=temp_dir,
                        target_folder=settings['main_dataset'],
                        dataset_name=video_base,
                        split_mode=settings.get('split_mode', 'keep'),
                        random_valid_pct=settings.get('valid_pct', 10),
                        class_mapping=settings['class_mapping'],
                        progress_callback=lambda cur, tot, msg: self.statusBar.showMessage(f'Merging {cur}/{tot}: {msg}', 0)
                    )
                    
                    if result.get('error'):
                        QMessageBox.warning(self, 'Merge Error', result['error'])
                    else:
                        msg = f"""Video integration complete:
  Frames integrated: {result['images_copied']}
  Labels integrated: {result['labels_copied']}
  Classes mapped: {result['classes_mapped']}
  Target Dataset: {settings['main_dataset']}"""
                        QMessageBox.information(self, 'Integration Complete', msg)
                        
                        reply = QMessageBox.question(self, 'Open Dataset',
                            'Would you like to open the main dataset folder now?'
                            , QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                        if reply == QMessageBox.Yes:
                            self.load_image_dataset_path(settings['main_dataset'])
                except Exception as e:
                    QMessageBox.warning(self, 'Integration Error', f'Failed during video integration: {str(e)}')
                finally:
                    # Clean up
                    try:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    except Exception:
                        pass
                return

            try:
                from managers.dataset_integration import DatasetIntegrationManager
                manager = DatasetIntegrationManager(self)
                if settings.get('preflight'):
                    manager.run_preflight_check(settings['new_dataset'])
                if settings.get('remove_hash_duplicates'):
                    manager.remove_duplicates(settings['new_dataset'])
                if settings.get('standardize_format'):
                    manager.standardize_format(settings['new_dataset'])
                if settings.get('normalize_res'):
                    manager.normalize_resolution(settings['new_dataset'])
            except Exception as e:
                QMessageBox.warning(self, 'Offline Task Error',
                    f"""Error during preflight/hash checks:
{e}""")
            self.load_image_dataset_path(settings['new_dataset'])
            self.integration_mode = True
            self.integration_main_dataset = settings['main_dataset']
            if settings['remove_grayscale']:
                self.viat_remove_grayscale()
            if settings['remove_duplicates']:
                self.viat_remove_duplicates()
            if settings['auto_import'] and settings.get('json_paths'):
                try:
                    from viat.utils.dataset_ops import auto_import_detections
                    auto_import_detections(self, settings['json_paths'],
                        target_classes=settings.get('target_classes'))
                    if self.current_frame >= self.total_frames:
                        self.current_frame = max(0, self.total_frames - 1)
                    self.frame_slider.blockSignals(True)
                    self.frame_slider.setMaximum(max(0, self.total_frames - 1))
                    self.frame_slider.setValue(self.current_frame)
                    self.frame_slider.blockSignals(False)
                    self.load_current_image()
                    self.update_frame_info()
                    self.update_annotation_list()
                except Exception as e:
                    QMessageBox.warning(self, 'Auto-Import Error',
                        f"""Error applying auto-import:
{e}""")
            if hasattr(self, 'finish_integration_action'):
                self.finish_integration_action.setVisible(True)
            if hasattr(self, 'finish_integration_move_action'):
                self.finish_integration_move_action.setVisible(True)
            if hasattr(self, 'finish_integration_sep'):
                self.finish_integration_sep.setVisible(True)
            self.statusBar.showMessage(
                'Integration Mode active. Please review the dataset.', 10000)
            self.canvas.update()

    def setup_sam_interactive(self):
        """Set up the SAM Interactive Dock signals and menu action."""
        if hasattr(self, 'sam_interactive_dock'):
            self.sam_interactive_dock.preview_requested.connect(self.
                on_sam_preview_requested)
            self.sam_interactive_dock.track_requested.connect(self.
                on_sam_track_requested)
            self.sam_interactive_dock.undo_requested.connect(self.undo)
            self.sam_interactive_dock.clear_requested.connect(self.
                on_sam_clear_requested)
            self.sam_interactive_dock.model_changed.connect(self.
                on_sam_model_changed)
            view_menu = self.menuBar().addMenu('&SAM Tracking')
            self.action_sam_interactive = view_menu.addAction(
                'Toggle SAM Interactive Mode')
            self.action_sam_interactive.setCheckable(True)
            self.action_sam_interactive.triggered.connect(self.
                toggle_sam_interactive_mode)

    def setup_evaluation_inspector(self):
        """Set up Evaluation Inspector Dock signals and menu actions."""
        if hasattr(self, 'evaluation_inspector_dock') and self.evaluation_inspector_dock:
            dock = self.evaluation_inspector_dock
            dock.eval_mode_toggled.connect(self.on_eval_mode_toggled)
            dock.conf_threshold_changed.connect(self.on_eval_conf_changed)
            dock.iou_threshold_changed.connect(self.on_eval_iou_changed)
            dock.filter_changed.connect(self.on_eval_filter_changed)
            dock.jump_to_frame_requested.connect(self.set_current_frame)
            dock.promote_fp_requested.connect(self.on_eval_promote_fp)
            dock.load_predictions_requested.connect(self.on_eval_load_predictions)
            dock.video_selected.connect(self.on_eval_video_selected)
            dock.save_ground_truth_requested.connect(self.save_evaluation_ground_truth)

    def on_eval_mode_toggled(self, checked):
        """Toggle evaluation overlay on canvas."""
        if hasattr(self, 'canvas'):
            self.canvas.set_eval_mode(checked)
        if hasattr(self, 'evaluation_inspector_dock') and self.evaluation_inspector_dock:
            self.evaluation_inspector_dock.btn_toggle_mode.blockSignals(True)
            self.evaluation_inspector_dock.btn_toggle_mode.setChecked(checked)
            self.evaluation_inspector_dock.btn_toggle_mode.setText("👁️ Evaluation View ACTIVE" if checked else "👁️ Enable Evaluation View")
            self.evaluation_inspector_dock.btn_toggle_mode.blockSignals(False)
        self.statusBar.showMessage(f"Evaluation View Mode {'Enabled' if checked else 'Disabled'}", 4000)

    def on_eval_conf_changed(self, conf):
        if hasattr(self, 'canvas'):
            self.canvas.set_eval_conf_threshold(conf)

    def on_eval_iou_changed(self, iou):
        if hasattr(self, 'canvas'):
            self.canvas.set_eval_iou_threshold(iou)

    def on_eval_filter_changed(self, filter_mode):
        if hasattr(self, 'canvas'):
            self.canvas.set_eval_filter(filter_mode)

    def on_eval_promote_fp(self, pred, frame_idx):
        """Converts a model prediction into a permanent Ground Truth annotation."""
        if not pred or 'bbox' not in pred:
            return
        pb = pred['bbox']
        rect = QRect(int(pb[0]), int(pb[1]), int(pb[2]), int(pb[3]))
        cls_name = pred.get('class_name') or self.canvas.current_class

        self.save_undo_state()

        bbox = BoundingBox(rect, cls_name)
        bbox.color = self.canvas.class_colors.get(cls_name, QColor(0, 255, 0))
        bbox.source = "promoted_prediction"

        if hasattr(self, 'frame_annotations'):
            if frame_idx not in self.frame_annotations:
                self.frame_annotations[frame_idx] = []
            self.frame_annotations[frame_idx].append(bbox)

        if frame_idx == self.current_frame:
            self.canvas.annotations.append(bbox)

        self.canvas.recompute_eval_matches()
        self.canvas.update()
        self.statusBar.showMessage(f"Promoted [{cls_name}] prediction to Ground Truth on Frame {frame_idx + 1}", 5000)

    def on_eval_load_predictions(self):
        """Prompt user to select a prediction file (.txt or .json) and load into Evaluation Inspector."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Model Predictions File", "", "Prediction Files (*.txt *.json);;All Files (*)"
        )
        if not file_path:
            return
        self.load_predictions_file_into_inspector(file_path)

    def load_predictions_file_into_inspector(self, file_path, video_name=None):
        """Parses prediction file and loads it into canvas & inspector dock."""
        if not os.path.exists(file_path):
            QMessageBox.warning(self, "File Not Found", f"Prediction file does not exist: {file_path}")
            return

        if not video_name:
            video_name = os.path.splitext(os.path.basename(file_path))[0]

        preds_by_frame = {}
        try:
            if file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                anns = data.get('annotations', data) if isinstance(data, dict) else data
                images = {img.get('id'): idx for idx, img in enumerate(data.get('images', []))} if isinstance(data, dict) else {}
                categories = {cat.get('id'): cat.get('name') for cat in data.get('categories', [])} if isinstance(data, dict) else {}

                for ann in anns:
                    img_id = ann.get('image_id', 1)
                    frame_idx = images.get(img_id, img_id - 1 if isinstance(img_id, int) and img_id > 0 else 0)
                    if frame_idx not in preds_by_frame:
                        preds_by_frame[frame_idx] = []
                    cat_id = ann.get('category_id', 1)
                    cls_name = categories.get(cat_id, str(cat_id))
                    preds_by_frame[frame_idx].append({
                        'bbox': ann.get('bbox', [0, 0, 0, 0]),
                        'score': float(ann.get('score', ann.get('confidence', 1.0))),
                        'class_name': cls_name,
                        'class_id': cat_id
                    })
            else:
                # Text format (e.g. [[class_id, x1, y1, x2, y2, score]];)
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                for f_idx, line in enumerate(lines):
                    clean = line.strip().rstrip(';').strip()
                    if not clean:
                        continue
                    try:
                        raw_list = eval(clean)
                    except Exception:
                        continue
                    if not isinstance(raw_list, list):
                        continue
                    if len(raw_list) > 0 and not isinstance(raw_list[0], list):
                        raw_list = [raw_list]

                    frame_preds = []
                    for box in raw_list:
                        if len(box) >= 5:
                            cid = box[0]
                            cls_name = str(cid)
                            # Check classes from active GT or class manager
                            if hasattr(self, 'eval_current_classes') and self.eval_current_classes:
                                if isinstance(cid, int) and 0 <= cid < len(self.eval_current_classes):
                                    cls_name = self.eval_current_classes[cid]
                            elif hasattr(self, 'class_manager') and hasattr(self.class_manager, 'classes'):
                                if isinstance(cid, int) and 0 <= cid < len(self.class_manager.classes):
                                    cls_name = self.class_manager.classes[cid]

                            x, y, w, h = box[1], box[2], box[3], box[4]
                            score = float(box[5]) if len(box) > 5 else 1.0
                            frame_preds.append({
                                'bbox': [x, y, w, h],
                                'score': score,
                                'class_name': cls_name,
                                'class_id': cid
                            })
                    if frame_preds:
                        preds_by_frame[f_idx] = frame_preds
        except Exception as e:
            QMessageBox.critical(self, "Parse Error", f"Failed to parse predictions file:\n{str(e)}")
            return

        total_boxes = sum(len(v) for v in preds_by_frame.values())
        self.activate_evaluation_inspection(video_name, preds_by_frame)
        self.statusBar.showMessage(f"Loaded {total_boxes} predictions for {video_name}", 5000)

    def activate_evaluation_inspection(self, video_name, predictions_dict, default_conf=0.5, iou_thr=0.5):
        """Activates inspection mode, sets predictions on canvas, updates dock, and shows dock."""
        self.eval_current_video_name = video_name
        if hasattr(self, 'canvas'):
            self.canvas.set_eval_predictions(predictions_dict, default_conf, iou_thr)
            self.canvas.set_eval_mode(True)

        if hasattr(self, 'evaluation_inspector_dock') and self.evaluation_inspector_dock:
            dock = self.evaluation_inspector_dock
            total_boxes = sum(len(v) for v in predictions_dict.values())
            dock.set_evaluation_info(video_name, total_boxes, default_conf)
            dock.btn_toggle_mode.blockSignals(True)
            dock.btn_toggle_mode.setChecked(True)
            dock.btn_toggle_mode.setText("👁️ Evaluation View ACTIVE")
            dock.btn_toggle_mode.blockSignals(False)
            dock.show()
            dock.raise_()

    def load_evaluation_dataset_into_inspector(self, gt_dir, det_dir, video_names, initial_video=None):
        """Loads a multi-video evaluation dataset into the inspector with full video switching support."""
        self.eval_dataset_context = {
            'gt_dir': gt_dir,
            'det_dir': det_dir,
            'video_names': list(video_names)
        }
        target_video = initial_video or (video_names[0] if video_names else None)
        if hasattr(self, 'evaluation_inspector_dock') and self.evaluation_inspector_dock:
            self.evaluation_inspector_dock.set_dataset_videos(video_names, current_video=target_video)

        if target_video:
            self.load_eval_video_sequence(target_video)

    def on_eval_video_selected(self, video_name):
        """Switches to the selected video sequence in Evaluation Inspection Mode."""
        self.load_eval_video_sequence(video_name)

    def load_eval_video_sequence(self, video_name):
        """Loads a specific video sequence, its ground truth, and its predictions into VIAT."""
        if not hasattr(self, 'eval_dataset_context') or not self.eval_dataset_context:
            return

        gt_dir = self.eval_dataset_context.get('gt_dir', '')
        det_dir = self.eval_dataset_context.get('det_dir', '')
        self.eval_current_video_name = video_name

        # 1. Locate and open video file
        exts = ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.MOV', '.m4v']
        video_file = None
        for ext in exts:
            c1 = os.path.join(gt_dir, video_name + ext)
            c2 = os.path.join(det_dir, video_name + ext)
            if os.path.exists(c1):
                video_file = c1
                break
            elif os.path.exists(c2):
                video_file = c2
                break

        if video_file and hasattr(self, 'open_video'):
            self.open_video(video_file)

        # 2. Locate and import Ground Truth
        self.frame_annotations = {}
        if hasattr(self, 'canvas'):
            self.canvas.annotations = []

        gt_candidates = [
            os.path.join(gt_dir, f"{video_name}.txt"),
            os.path.join(gt_dir, f"{video_name}.json"),
        ]
        self.eval_current_gt_path = None
        for gtc in gt_candidates:
            if os.path.exists(gtc):
                self.eval_current_gt_path = gtc
                try:
                    from viat.utils.file_operations import import_annotations as import_annotations_func, detect_annotation_format, extract_raya_classes, import_raya_with_classes_annotations
                    fmt = detect_annotation_format(gtc)
                    if fmt == "Raya with classes":
                        cls_list = extract_raya_classes(gtc) or []
                        self.eval_current_classes = cls_list
                        cls_mapping = {i: c for i, c in enumerate(cls_list)}
                        f_anns, _ = import_raya_with_classes_annotations(gtc, BoundingBox, cls_mapping)
                        self.frame_annotations = f_anns
                    else:
                        self.eval_current_classes = list(getattr(self.canvas, 'class_colors', {}).keys())
                        f_anns, _ = import_annotations_func(gtc, BoundingBox, class_colors=getattr(self.canvas, 'class_colors', None))
                        self.frame_annotations = f_anns

                    curr_f = getattr(self, 'current_frame', 0)
                    if hasattr(self, 'canvas') and curr_f in self.frame_annotations:
                        self.canvas.annotations = list(self.frame_annotations[curr_f])
                except Exception as e:
                    logger.warning(f"Could not import GT file {gtc}: {e}")
                break

        # 3. Locate and load Predictions
        det_candidates = [
            os.path.join(det_dir, f"{video_name}.txt"),
            os.path.join(det_dir, f"{video_name}.json"),
        ]
        for dtc in det_candidates:
            if os.path.exists(dtc):
                self.load_predictions_file_into_inspector(dtc, video_name)
                break

    def save_evaluation_ground_truth(self, show_dialog=True):
        """Saves current annotations (including promoted FPs and modifications) to the ground truth file."""
        gt_file = getattr(self, 'eval_current_gt_path', None)
        if not gt_file:
            if hasattr(self, 'eval_dataset_context') and self.eval_dataset_context:
                gt_dir = self.eval_dataset_context.get('gt_dir', '')
                vid_name = getattr(self, 'eval_current_video_name', '')
                if gt_dir and vid_name:
                    candidate = os.path.join(gt_dir, f"{vid_name}.txt")
                    gt_file = candidate

        if not gt_file:
            gt_file, _ = QFileDialog.getSaveFileName(
                self, "Save Ground Truth Annotations", "", "Text Files (*.txt);;JSON Files (*.json);;All Files (*)"
            )
            if not gt_file:
                return

        # Collect all annotations from all frames
        all_annotations = []
        for f_idx, anns in self.frame_annotations.items():
            for ann in anns:
                import copy
                c_ann = copy.copy(ann)
                c_ann.frame = f_idx
                all_annotations.append(c_ann)

        curr_f = getattr(self, 'current_frame', 0)
        if curr_f not in self.frame_annotations and self.canvas.annotations:
            for ann in self.canvas.annotations:
                import copy
                c_ann = copy.copy(ann)
                c_ann.frame = curr_f
                all_annotations.append(c_ann)

        try:
            from viat.utils.file_operations import export_raya_with_classes_annotations
            classes = list(self.canvas.class_colors.keys())
            deleted = getattr(self, 'deleted_frames', set())
            total_f = getattr(self, 'total_frames', None)

            export_raya_with_classes_annotations(
                gt_file,
                all_annotations,
                classes=classes,
                deleted_frames=deleted,
                total_frames=total_f
            )
            self.eval_current_gt_path = gt_file
            self.statusBar.showMessage(f"Ground truth saved to: {os.path.basename(gt_file)} ({len(all_annotations)} annotations)", 6000)
            if show_dialog and self.isVisible():
                QMessageBox.information(
                    self,
                    "Ground Truth Saved",
                    f"Successfully saved {len(all_annotations)} annotations to:\n{gt_file}"
                )
        except Exception as e:
            if show_dialog and self.isVisible():
                QMessageBox.critical(self, "Save Error", f"Failed to save ground truth annotations:\n{str(e)}")
            else:
                logger.error(f"Failed to save ground truth: {e}")


    def on_dock_item_selected(self, item_data):
        if isinstance(item_data, str):
            target_frame = None
            if hasattr(self, 'class_frames_dock') and hasattr(self, 'video_groups'):
                mode, count, class_name = self.class_frames_dock.get_filter_state()
                if class_name and item_data in self.video_groups:
                    for idx in self.video_groups[item_data]:
                        annots = self.frame_annotations.get(idx, [])
                        target_count = sum(1 for ann in annots if ann.class_name == class_name)
                        if (mode == "More than" and target_count > count) or \
                           (mode == "Less than" and target_count < count) or \
                           (mode == "Exactly" and target_count == count) or \
                           (mode == "Frames With Class" and target_count > 0) or \
                           (mode == "Frames Without Class" and target_count == 0):
                            target_frame = idx
                            break
            
            if target_frame is not None:
                self.set_current_frame(target_frame)
            elif hasattr(self, 'video_combo'):
                self.video_combo.setCurrentText(item_data)
        else:
            self.set_current_frame(item_data)

    def setup_empty_frames_manager(self):
        self.only_show_empty_frames = False
        if hasattr(self, 'empty_frames_dock'):
            self.empty_frames_dock.frame_selected.connect(self.on_dock_item_selected)
            self.empty_frames_dock.refresh_requested.connect(self.
                refresh_empty_frames_dock)
            self.empty_frames_dock.predict_requested.connect(self.
                on_empty_frame_predict_requested)
            self.empty_frames_dock.predict_all_requested.connect(self.
                on_empty_frame_predict_all_requested)
            self.empty_frames_dock.filter_toggled.connect(self.
                on_empty_frames_filter_toggled)
            self.empty_frames_dock.zero_shot_requested.connect(self.
                on_zero_shot_empty_requested)

    def setup_uncertain_frames_manager(self):
        if hasattr(self, 'uncertain_frames_dock'):
            self.uncertain_frames_dock.frame_selected.connect(self.on_dock_item_selected)
            self.uncertain_frames_dock.refresh_requested.connect(self.refresh_uncertain_frames_dock)
                
    def setup_class_frames_manager(self):
        self.nav_class_filter_active = False
        self.nav_class_filter_mode = ""
        self.nav_class_filter_count = 0
        self.nav_class_filter_target = ""
        if hasattr(self, 'class_frames_dock'):
            self.class_frames_dock.frame_selected.connect(self.on_dock_item_selected)
            self.class_frames_dock.refresh_requested.connect(self.refresh_class_frames_dock)
            self.class_frames_dock.filter_toggled.connect(self.on_class_frames_filter_toggled)
            self.class_frames_dock.delete_labels_requested.connect(self.on_class_labels_delete_requested)
            self.class_frames_dock.zero_shot_requested.connect(self.on_zero_shot_class_requested)

    def on_class_frames_filter_toggled(self, is_active, mode, count, class_name):
        self.nav_class_filter_active = is_active
        self.nav_class_filter_mode = mode
        self.nav_class_filter_count = count
        self.nav_class_filter_target = class_name
        
    @log_exceptions
    def on_class_labels_delete_requested(self, frame_indices, class_name):
        if not frame_indices or not class_name:
            return
            
        frame_indices = self._resolve_frame_indices(frame_indices)
            
        reply = QMessageBox.question(self, 'Delete Labels',
            f"Are you sure you want to delete all '{class_name}' labels from {len(frame_indices)} selected frame(s)?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            
        if reply == QMessageBox.Yes:
            deleted_count = 0
            for frame_idx in frame_indices:
                if frame_idx in self.frame_annotations:
                    annots = self.frame_annotations[frame_idx]
                    original_len = len(annots)
                    # Keep annotations that are NOT the target class
                    annots = [ann for ann in annots if ann.class_name != class_name]
                    if len(annots) < original_len:
                        self.frame_annotations[frame_idx] = annots
                        deleted_count += (original_len - len(annots))
                        
            if deleted_count > 0:
                self.project_modified = True
                if self.current_frame in frame_indices:
                    self.load_current_frame_annotations()
                    self.update_frame_display()
                self.refresh_class_frames_dock()
                self.refresh_empty_frames_dock()
                self.statusBar.showMessage(f"Deleted {deleted_count} '{class_name}' label(s) from selected frames.", 5000)
            else:
                self.statusBar.showMessage(f"No '{class_name}' labels found in selected frames.", 5000)
        
    def refresh_class_frames_dock(self):
        # Avoid recursive refresh
        if getattr(self, '_refreshing_class_frames', False):
            return
        self._refreshing_class_frames = True
        try:
            mode, count, class_name = self.class_frames_dock.get_filter_state()
            if not class_name:
                self.class_frames_dock.update_data([], self.total_frames)
                return
                
            matching_items = []
            
            if getattr(self, 'video_mode', False) and hasattr(self, 'video_groups'):
                current_vid = None
                for vid_name, indices in self.video_groups.items():
                    if self.current_frame in indices:
                        current_vid = vid_name
                        break
                if current_vid:
                    indices = self.video_groups[current_vid]
                    for local_idx, absolute_idx in enumerate(indices):
                        annots = self.frame_annotations.get(absolute_idx, [])
                        target_count = sum(1 for ann in annots if ann.class_name == class_name)
                        if (mode == "More than" and target_count > count) or \
                           (mode == "Less than" and target_count < count) or \
                           (mode == "Exactly" and target_count == count) or \
                           (mode == "Frames With Class" and target_count > 0) or \
                           (mode == "Frames Without Class" and target_count == 0):
                            matching_items.append((f"Frame {local_idx + 1}", absolute_idx))
                    total_items = len(indices)
                else:
                    total_items = 0
            else:
                for i in range(self.total_frames):
                    annots = self.frame_annotations.get(i, [])
                    target_count = sum(1 for ann in annots if ann.class_name == class_name)
                    if (mode == "More than" and target_count > count) or \
                       (mode == "Less than" and target_count < count) or \
                       (mode == "Exactly" and target_count == count) or \
                       (mode == "Frames With Class" and target_count > 0) or \
                       (mode == "Frames Without Class" and target_count == 0):
                        matching_items.append(i)
                total_items = self.total_frames
                    
            self.class_frames_dock.update_data(matching_items, total_items)
        finally:
            self._refreshing_class_frames = False

    def on_zero_shot_empty_requested(self, prompt, model_type):
        if hasattr(self, 'empty_frames_dock'):
            items = self.empty_frames_dock.list_widget.selectedItems()
            if items:
                frame_indices = [item.data(Qt.UserRole) for item in items]
            else:
                frame_indices = self.empty_frames_dock.empty_frames
            
            frame_indices = self._resolve_frame_indices(frame_indices)
            self._run_zero_shot_batch(frame_indices, prompt, model_type)

    def on_zero_shot_class_requested(self, prompt, model_type):
        if hasattr(self, 'class_frames_dock'):
            items = self.class_frames_dock.list_widget.selectedItems()
            if items:
                frame_indices = [item.data(Qt.UserRole) for item in items]
            else:
                frame_indices = getattr(self.class_frames_dock, 'matching_frames', [])
                
            if not frame_indices:
                QMessageBox.information(self, "No Selection", "No frames found to detect on.")
                return
                
            frame_indices = self._resolve_frame_indices(frame_indices)
            self._run_zero_shot_batch(frame_indices, prompt, model_type)

    @log_exceptions
    def _run_zero_shot_batch(self, frame_indices, prompt, model_type):
        if not hasattr(self, 'image_files') and not hasattr(self, 'video_filename'):
            return
        if not frame_indices:
            QMessageBox.information(self, "No Frames", "No frames to process.")
            return
            
        progress = QProgressDialog("Loading Zero-Shot model...", "Cancel", 0, len(frame_indices), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setWindowTitle("Zero-Shot Detect")
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if hasattr(self, 'labeler_analytics'):
                self.labeler_analytics['tool_usage']['zero_shot'] += 1
                self.labeler_analytics['prompts'].append({
                    "prompt": prompt,
                    "model": model_type,
                    "frames_count": len(frame_indices)
                })
                
            if not hasattr(self, 'zero_shot_manager') or self.zero_shot_manager is None:
                from .utils.zero_shot_manager import ZeroShotManager
                self.zero_shot_manager = ZeroShotManager()
                
            success, msg = self.zero_shot_manager.load_model(model_type)
            if not success:
                QApplication.restoreOverrideCursor()
                progress.close()
                QMessageBox.warning(self, 'Model Load Error', msg)
                return
                
            zs_model = self.zero_shot_manager.detector
            if zs_model:
                zs_model.set_classes([prompt])
                
            predictions_made = 0
            
            for i, target_idx in enumerate(frame_indices):
                if progress.wasCanceled():
                    break
                    
                progress.setValue(i)
                progress.setLabelText(f"Detecting '{prompt}' in frame {target_idx}...")
                self.set_current_frame(target_idx)
                QApplication.processEvents()
                
                # Load frame
                frame = None
                if hasattr(self, 'is_image_dataset') and self.is_image_dataset:
                    if 0 <= target_idx < len(self.image_files):
                        frame = cv2.imread(self.image_files[target_idx])
                else:
                    if hasattr(self, 'video_filename'):
                        cap = cv2.VideoCapture(self.video_filename)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                        ret, bgr = cap.read()
                        if ret:
                            frame = bgr
                        cap.release()
                
                if frame is None:
                    continue
                    
                detections = []
                if zs_model:
                    detections = zs_model.predict(frame)
                    
                if detections:
                    if target_idx not in self.frame_annotations:
                        self.frame_annotations[target_idx] = []
                    
                    for det in detections:
                        box = det['box']
                        rect = QRectF(box[0], box[1], box[2] - box[0], box[3] - box[1])
                        
                        # Generate color
                        if prompt not in self.class_colors:
                            self.class_colors[prompt] = QColor(
                                __import__('random').randint(0, 255),
                                __import__('random').randint(0, 255),
                                __import__('random').randint(0, 255)
                            )
                        class_color = self.class_colors[prompt]
                        
                        from .annotation import BoundingBox
                        new_ann = BoundingBox(rect=rect, class_name=prompt, color=class_color)
                        new_ann.is_unverified = True
                        self.frame_annotations[target_idx].append(new_ann)
                        
                    predictions_made += len(detections)
                    self.load_current_frame_annotations()
                    self.update_frame_display()
                    
            progress.setValue(len(frame_indices))
            
            if predictions_made > 0:
                if prompt not in self.project_classes:
                    self.project_classes.append(prompt)
                    self.save_project_config()
                    if hasattr(self, 'class_list'):
                        self.class_list.update_classes(self.project_classes)
                    if hasattr(self, 'class_frames_dock'):
                        self.class_frames_dock.update_classes(self.project_classes)
                        
                self.project_modified = True
                if self.current_frame in frame_indices:
                    self.load_current_frame_annotations()
                    self.update_frame_display()
                if hasattr(self, 'refresh_class_frames_dock'):
                    self.refresh_class_frames_dock()
                if hasattr(self, 'refresh_empty_frames_dock'):
                    self.refresh_empty_frames_dock()
                self.statusBar.showMessage(f'Zero-shot complete. Added {predictions_made} "{prompt}" labels.', 5000)
            else:
                self.statusBar.showMessage(f'Zero-shot complete. No "{prompt}" labels were detected.', 5000)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, 'Error', f'Error during zero-shot detect: {e}')
        finally:
            QApplication.restoreOverrideCursor()

    def on_empty_frames_filter_toggled(self, checked):
        self.only_show_empty_frames = checked

    def refresh_uncertain_frames_dock(self):
        if not hasattr(self, 'uncertain_frames_dock'):
            return
            
        uncertain_frames = []
        for frame_idx, annots in self.frame_annotations.items():
            if any(getattr(ann, 'uncertain', False) for ann in annots):
                uncertain_frames.append(frame_idx)
                
        uncertain_frames.sort()
        self.uncertain_frames_dock.update_data(uncertain_frames)

    def refresh_empty_frames_dock(self):
        if not hasattr(self, 'empty_frames_dock'):
            return
        if not hasattr(self, 'image_files') or not self.image_files:
            self.empty_frames_dock.update_data([], [], 0)
            return
        empty_items = []
        annotated_items = []
        self.update_frame_annotations()
        
        if getattr(self, 'video_mode', False) and hasattr(self, 'video_groups'):
            current_vid = None
            for vid_name, indices in self.video_groups.items():
                if self.current_frame in indices:
                    current_vid = vid_name
                    break
            if current_vid:
                indices = self.video_groups[current_vid]
                for local_idx, absolute_idx in enumerate(indices):
                    annots = self.frame_annotations.get(absolute_idx, [])
                    if not annots:
                        empty_items.append((f"Frame {local_idx + 1}", absolute_idx))
                    else:
                        annotated_items.append(absolute_idx)
                total_items = len(indices)
            else:
                total_items = 0
        else:
            for idx in range(len(self.image_files)):
                annots = self.frame_annotations.get(idx, [])
                if not annots:
                    empty_items.append(idx)
                else:
                    annotated_items.append(idx)
            total_items = len(self.image_files)
            
        self.empty_frames_dock.update_data(empty_items, annotated_items, total_items)

    def on_empty_frame_predict_requested(self, target_idx, source_idx,
        model_type):
        if not hasattr(self, 'image_files') or not self.image_files:
            return
        source_annots = self.frame_annotations.get(source_idx, [])
        if not source_annots:
            QMessageBox.warning(self, 'No Annotations',
                f'Source frame {source_idx} has no annotations to track from.')
            return
        self.statusBar.showMessage(f'Loading {model_type} for tracking...')
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if 'sam3' in model_type.lower():
                success, msg = self.sam3_native_manager.load_model(model_type)
                tracker = self.sam3_native_manager
            elif 'trt' in model_type.lower():
                success, msg = self.sam2_trt_manager.load_model(model_type)
                tracker = self.sam2_trt_manager
            else:
                success, msg = self.sam_manager.load_model(model_type)
                tracker = self.sam_manager
            if not success:
                QMessageBox.warning(self, 'Model Load Error', msg)
                return
            bboxes = []
            for ann in source_annots:
                bboxes.append([ann.rect.left(), ann.rect.top(), ann.rect.
                    right(), ann.rect.bottom()])
            source_img = cv2.imread(self.image_files[source_idx])
            target_img = cv2.imread(self.image_files[target_idx])

            def frame_gen():
                yield cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
                yield cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
            self.statusBar.showMessage(
                f'Tracking objects from frame {source_idx} to {target_idx}...')
            results = list(tracker.track_video_from_boxes(frame_gen(),
                bboxes, model_type))
            if len(results) >= 2:
                success, target_result = results[1]
                if success:
                    new_annots = []
                    polygons = target_result.get('polygons', [])
                    boxes = target_result.get('boxes', [])
                    for i in range(len(boxes)):
                        if i < len(source_annots):
                            src_ann = source_annots[i]
                            b = boxes[i]
                            rect = QRectF(b[0], b[1], b[2] - b[0], b[3] - b[1])
                            poly = polygons[i] if i < len(polygons
                                ) and polygons[i] is not None else None
                            new_ann = BoundingBox(rect=rect, class_name=
                                src_ann.class_name, attributes=src_ann.
                                attributes.copy(), color=src_ann.color)
                            if poly:
                                new_ann.segmentation = poly
                            new_ann.verified = False
                            new_annots.append(new_ann)
                    self.frame_annotations[target_idx] = new_annots
                    self.refresh_empty_frames_dock()
                    self.seek_to_frame(target_idx)
                    self.statusBar.showMessage(
                        f'Tracking successful. {len(new_annots)} objects predicted.'
                        , 5000)
                else:
                    QMessageBox.warning(self, 'Tracking Failed', str(
                        target_result))
            else:
                QMessageBox.warning(self, 'Tracking Failed',
                    'Not enough frames yielded by tracker.')
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, 'Error', f'Error during tracking: {e}')
        finally:
            QApplication.restoreOverrideCursor()

    @log_exceptions
    def on_empty_frame_predict_all_requested(self, model_type):
        if not hasattr(self, 'image_files') or not self.image_files:
            return
        if not hasattr(self, 'empty_frames_dock'):
            return
            
        empty_frames = self.empty_frames_dock.empty_frames
        if not empty_frames:
            QMessageBox.information(self, "No Empty Frames", "There are no empty frames to predict.")
            return
            
        # Initialize progress dialog
        progress = QProgressDialog("Loading model for batch prediction...", "Cancel", 0, len(empty_frames), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setWindowTitle("Predict All")
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()
        
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if 'sam3' in model_type.lower():
                success, msg = self.sam3_native_manager.load_model(model_type)
                tracker = self.sam3_native_manager
            elif 'trt' in model_type.lower():
                success, msg = self.sam2_trt_manager.load_model(model_type)
                tracker = self.sam2_trt_manager
            else:
                success, msg = self.sam_manager.load_model(model_type)
                tracker = self.sam_manager
                
            if not success:
                QApplication.restoreOverrideCursor()
                progress.close()
                QMessageBox.warning(self, 'Model Load Error', msg)
                return
                
            predictions_made = 0
            
            for i, target_idx in enumerate(empty_frames):
                if progress.wasCanceled():
                    break
                    
                progress.setValue(i)
                self.set_current_frame(target_idx)
                QApplication.processEvents()
                
                # Check neighbors
                source_idx = None
                if (target_idx - 1) in self.frame_annotations and self.frame_annotations[target_idx - 1]:
                    source_idx = target_idx - 1
                elif (target_idx + 1) in self.frame_annotations and self.frame_annotations[target_idx + 1]:
                    source_idx = target_idx + 1
                    
                if source_idx is None:
                    # Skip if no adjacent frame is annotated
                    continue
                    
                progress.setLabelText(f"Predicting frame {target_idx} from frame {source_idx}...")
                QApplication.processEvents()
                
                source_annots = self.frame_annotations[source_idx]
                
                bboxes = []
                for ann in source_annots:
                    bboxes.append([ann.rect.left(), ann.rect.top(), ann.rect.right(), ann.rect.bottom()])
                    
                source_img = cv2.imread(self.image_files[source_idx])
                target_img = cv2.imread(self.image_files[target_idx])
                
                def frame_gen():
                    yield cv2.cvtColor(source_img, cv2.COLOR_BGR2RGB)
                    yield cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
                    
                results = list(tracker.track_video_from_boxes(frame_gen(), bboxes, model_type))
                if len(results) >= 2:
                    success_track, target_result = results[1]
                    if success_track:
                        new_annots = []
                        polygons = target_result.get('polygons', [])
                        boxes = target_result.get('boxes', [])
                        for j in range(len(boxes)):
                            if j < len(source_annots):
                                src_ann = source_annots[j]
                                b = boxes[j]
                                rect = QRectF(b[0], b[1], b[2] - b[0], b[3] - b[1])
                                poly = polygons[j] if j < len(polygons) and polygons[j] is not None else None
                                new_ann = BoundingBox(rect=rect, class_name=src_ann.class_name, 
                                                      attributes=src_ann.attributes.copy(), color=src_ann.color)
                                if poly:
                                    new_ann.segmentation = poly
                                new_ann.verified = False
                                new_annots.append(new_ann)
                        self.frame_annotations[target_idx] = new_annots
                        predictions_made += 1
                        self.load_current_frame_annotations()
                        self.update_frame_display()
                        
            progress.setValue(len(empty_frames))
            
            if predictions_made > 0:
                self.project_modified = True
                self.refresh_empty_frames_dock()
                self.load_current_frame_annotations()
                self.update_frame_display()
                self.statusBar.showMessage(f'Batch prediction complete. Predicted {predictions_made} frames.', 5000)
            else:
                self.statusBar.showMessage('Batch prediction complete. No frames were predicted.', 5000)
                
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, 'Error', f'Error during batch tracking: {e}')
        finally:
            QApplication.restoreOverrideCursor()

    def on_sam_model_changed(self, model_type):
        self.statusBar.showMessage(f'Loading {model_type}...')
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            if 'sam3' in model_type.lower():
                success, msg = self.sam3_native_manager.load_model(model_type)
            elif 'trt' in model_type.lower():
                success, msg = self.sam2_trt_manager.load_model(model_type)
            else:
                success, msg = self.sam_manager.load_model(model_type)
            if success:
                self.statusBar.showMessage(f'{model_type} loaded successfully.'
                    , 3000)
            else:
                QMessageBox.warning(self, 'Model Load Error', msg)
                self.statusBar.showMessage('Model loading failed.', 3000)
        finally:
            QApplication.restoreOverrideCursor()

    def toggle_sam_interactive_mode(self, checked):
        if not hasattr(self, 'sam_interactive_dock'):
            return
            
        if hasattr(self, 'action_sam_interactive') and self.action_sam_interactive.isChecked() != checked:
            self.action_sam_interactive.blockSignals(True)
            self.action_sam_interactive.setChecked(checked)
            self.action_sam_interactive.blockSignals(False)
            
        if hasattr(self, 'btn_sam_track') and self.btn_sam_track.isChecked() != checked:
            self.btn_sam_track.blockSignals(True)
            self.btn_sam_track.setChecked(checked)
            self.btn_sam_track.blockSignals(False)
            
        self.canvas.sam_interactive_mode = checked
        if checked:
            if hasattr(self, 'annotation_dock') and hasattr(self, 'tabifyDockWidget'):
                self.tabifyDockWidget(self.annotation_dock, self.sam_interactive_dock)
            self.sam_interactive_dock.show()
            self.sam_interactive_dock.raise_()
            model_type = self.sam_interactive_dock.get_model_type()
            
            self.statusBar.showMessage(f'Loading {model_type}...')
            QApplication.setOverrideCursor(Qt.WaitCursor)
            QApplication.processEvents()
            
            try:
                if 'sam3' in model_type.lower():
                    success, msg = self.sam3_native_manager.load_model(model_type)
                    if not success:
                        from PyQt5.QtWidgets import QMessageBox
                        QMessageBox.warning(self, 'SAM3 Load Error', msg)
                elif 'trt' in model_type.lower():
                    success, msg = self.sam2_trt_manager.load_model(model_type)
                    if not success:
                        from PyQt5.QtWidgets import QMessageBox
                        QMessageBox.warning(self, 'SAM2 TRT Load Error', msg)
                else:
                    success, msg = self.sam_manager.load_model(model_type)
                    if not success:
                        from PyQt5.QtWidgets import QMessageBox
                        QMessageBox.warning(self, 'SAM2 Load Error', msg)
            finally:
                QApplication.restoreOverrideCursor()
                
            self.canvas.sam_prompt_points = []
            self.canvas.sam_prompt_labels = []
            self.canvas.sam_prompt_box = None
            self.sam_interactive_dock.update_status(0, 0, False)
            self.update_frame_info()
            self.statusBar.showMessage(
                'SAM Interactive Mode Enabled: Left click for pos point, drag for box, right click for neg point.'
                )
        else:
            self.sam_interactive_dock.hide()
            self.statusBar.showMessage('SAM Interactive Mode Disabled')
        self.canvas.update()

    def on_sam_clear_requested(self):
        self.canvas.sam_prompt_points = []
        self.canvas.sam_prompt_labels = []
        self.canvas.sam_prompt_box = None
        self.canvas.sam_preview_polygon = None
        self.canvas.sam_preview_rect = None
        self.canvas.sam_preview_class = None
        
        if hasattr(self.canvas, 'annotations'):
            self.canvas.annotations = [a for a in self.canvas.annotations if getattr(a, 'is_sam_preview', False) == False]
            
        self.sam_interactive_dock.update_status(0, 0, False)
        
        # Clear sessions in all managers
        if hasattr(self, 'sam3_native_manager') and self.sam3_native_manager is not None:
            try:
                self.sam3_native_manager.clear_session()
            except Exception as e:
                print(f"Failed to clear SAM3 native session: {e}")
        if hasattr(self, 'sam2_trt_manager') and self.sam2_trt_manager is not None:
            try:
                self.sam2_trt_manager.clear_session()
            except Exception as e:
                print(f"Failed to clear SAM2 TRT session: {e}")
        if hasattr(self, 'sam_manager') and self.sam_manager is not None:
            try:
                self.sam_manager.clear_session()
            except Exception as e:
                print(f"Failed to clear SAM session: {e}")
                
        if hasattr(self, 'fast_tracker_manager') and self.fast_tracker_manager is not None:
            try:
                self.fast_tracker_manager.clear_session()
            except Exception as e:
                print(f"Failed to clear Fast Tracker session: {e}")
                
        self.canvas.update()
        if hasattr(self, 'canvas') and self.canvas:
            self.canvas.setFocus()

    def on_sam_preview_requested(self):
        if self.canvas.current_frame_array is None:
            return
        points = [[p.x(), p.y()] for p in self.canvas.sam_prompt_points
            ] if self.canvas.sam_prompt_points else None
        labels = (self.canvas.sam_prompt_labels if self.canvas.
            sam_prompt_labels else None)
        box = [self.canvas.sam_prompt_box.x(), self.canvas.sam_prompt_box.y
            (), self.canvas.sam_prompt_box.x() + self.canvas.sam_prompt_box
            .width(), self.canvas.sam_prompt_box.y() + self.canvas.
            sam_prompt_box.height()] if self.canvas.sam_prompt_box else None
        text_prompt = self.sam_interactive_dock.get_text_prompt()
        if not points and not box and not text_prompt:
            QMessageBox.warning(self, 'No Prompts',
                'Please add at least one point, bounding box, or text prompt.')
            return
        model_type = self.sam_interactive_dock.get_model_type()
        if 'sam3' in model_type.lower():
            manager = self.sam3_native_manager
        elif 'trt' in model_type.lower():
            manager = self.sam2_trt_manager
        else:
            manager = self.sam_manager
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            polygon = manager.predict_mask_from_prompt(self.canvas.
                current_frame_array, points=points, labels=labels, box=box,
                text_prompt=text_prompt)
        finally:
            QApplication.restoreOverrideCursor()
        if polygon:
            x_coords = [p[0] for p in polygon]
            y_coords = [p[1] for p in polygon]
            preview_rect = None
            if x_coords and y_coords:
                preview_rect = QRect(int(min(x_coords)), int(min(y_coords)),
                    int(max(x_coords) - min(x_coords)), int(max(y_coords) -
                    min(y_coords)))
            self.canvas.sam_preview_polygon = polygon
            self.canvas.sam_preview_rect = preview_rect
            self.canvas.sam_preview_class = self.canvas.current_class
            # Remove any lingering preview from canvas annotations
            self.canvas.annotations = [a for a in self.canvas.annotations if getattr(a, 'is_sam_preview', False) == False]
            self.canvas.update()
            self.statusBar.showMessage('Preview generated successfully.', 3000)
        else:
            self.canvas.sam_preview_polygon = None
            self.canvas.sam_preview_rect = None
            self.canvas.sam_preview_class = None
            self.canvas.update()
            self.statusBar.showMessage('No mask generated.', 3000)
        if hasattr(self, 'canvas') and self.canvas:
            self.canvas.setFocus()

    def on_sam_track_requested(self, strategy, start_f, end_f, direction="forward"):
        if hasattr(self, 'is_image_dataset') and self.is_image_dataset:
            if not hasattr(self, 'image_files') or not self.image_files:
                return
        elif not hasattr(self, 'cap') or not self.cap or not self.cap.isOpened():
            return

        if direction == "bidirectional" and strategy != "frame":
            saved_points = list(self.canvas.sam_prompt_points) if self.canvas.sam_prompt_points else None
            saved_labels = list(self.canvas.sam_prompt_labels) if self.canvas.sam_prompt_labels else None
            saved_box = self.canvas.sam_prompt_box
            prompt_frame = self.current_frame

            # 1. Forward Pass (prompt_frame -> max(start_f, end_f))
            fwd_end = max(start_f, end_f)
            if prompt_frame < fwd_end:
                self.on_sam_track_requested(strategy, prompt_frame, fwd_end, direction="forward")

            # Restore prompt state on prompt_frame for backward pass
            self.canvas.sam_prompt_points = saved_points if saved_points else []
            self.canvas.sam_prompt_labels = saved_labels if saved_labels else []
            self.canvas.sam_prompt_box = saved_box
            self.sam_interactive_dock.update_status(
                len([l for l in (saved_labels or []) if l == 1]),
                len([l for l in (saved_labels or []) if l == 0]),
                saved_box is not None
            )

            # 2. Backward Pass (prompt_frame -> min(start_f, end_f))
            bwd_end = min(start_f, end_f)
            if prompt_frame > bwd_end:
                self.on_sam_track_requested(strategy, prompt_frame, bwd_end, direction="backward")
            return

        is_backward = (direction == "backward") or (start_f > end_f)
        step = -1 if is_backward else 1

        auto_blur_global = getattr(self, 'auto_blur_labels', False)
        sam_blur_checked = self.sam_interactive_dock.get_blur_tracked_objects()
        should_blur = False
        
        if sam_blur_checked:
            should_blur = True
        elif auto_blur_global:
            reply = QMessageBox.question(self, "Blur Tracking Results?",
                "Global 'Auto Blur All New Labels' is enabled, but the 'Automatically Blur Tracked Objects' setting in the SAM panel is not checked.\n\n"
                "Do you want to blur the tracked objects?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                should_blur = True
            elif reply == QMessageBox.No:
                should_blur = False
            else:
                return

        blur_shape = getattr(self.sam_interactive_dock, 'get_blur_shape', lambda: 'segmentation')()
        blur_margin = getattr(self.sam_interactive_dock, 'get_blur_margin', lambda: 0)()

        def _apply_sam_blur(frame_idx, poly, b_rect):
            if blur_shape == "segmentation" and poly:
                self.blur_manager.add_polygon_region(frame_idx, poly, self.canvas.blur_kernel, margin=blur_margin)
            else:
                self.blur_manager.add_bbox_region(frame_idx, b_rect, self.canvas.blur_kernel, margin=blur_margin)
            if getattr(self, 'auto_remove_under_blur', False) and hasattr(self, 'remove_annotations_under_blur'):
                self.remove_annotations_under_blur(frame_idx)
            
        if getattr(self, 'video_mode', False) and hasattr(self, 'video_groups') and hasattr(self, 'video_manager_dock'):
            current_vid = None
            for vid_name, indices in self.video_groups.items():
                if self.current_frame in indices:
                    current_vid = vid_name
                    break
            if current_vid:
                indices = self.video_groups[current_vid]
                start_f = indices[start_f] if start_f < len(indices) else indices[-1]
                end_f = indices[end_f] if end_f < len(indices) else indices[-1]
        points = [[p.x(), p.y()] for p in self.canvas.sam_prompt_points
            ] if self.canvas.sam_prompt_points else None
        labels = (self.canvas.sam_prompt_labels if self.canvas.
            sam_prompt_labels else None)
        box = [self.canvas.sam_prompt_box.x(), self.canvas.sam_prompt_box.y
            (), self.canvas.sam_prompt_box.x() + self.canvas.sam_prompt_box
            .width(), self.canvas.sam_prompt_box.y() + self.canvas.
            sam_prompt_box.height()] if self.canvas.sam_prompt_box else None
            
        if not box:
            if getattr(self.canvas, 'sam_preview_rect', None):
                pr = self.canvas.sam_preview_rect
                box = [pr.left(), pr.top(), pr.right(), pr.bottom()]
            else:
                for ann in self.canvas.annotations:
                    if getattr(ann, 'is_sam_preview', False):
                        box = [ann.rect.left(), ann.rect.top(), ann.rect.right(), ann.rect.bottom()]
                        break
        text_prompt = self.sam_interactive_dock.get_text_prompt()
        if not points and not box and not text_prompt:
            QMessageBox.warning(self, 'No Prompts',
                'Please add at least one point, bounding box, or text prompt.')
            return
            
        tracker_engine = getattr(self.sam_interactive_dock, 'get_tracker_engine', lambda: 'sam')()
        model_type = self.sam_interactive_dock.get_model_type()
        
        initial_polygon = None
        if tracker_engine in ['ettrack', 'ostrack', 'ostrack_trt', 'ostrack_engine']:
            if not box and (points or text_prompt):
                if 'sam3' in model_type.lower():
                    sam_mgr = self.sam3_native_manager
                elif 'trt' in model_type.lower():
                    sam_mgr = self.sam2_trt_manager
                else:
                    sam_mgr = self.sam_manager
                
                self.statusBar.showMessage(f'Generating initial bounding box using {model_type}...')
                QApplication.processEvents()
                ok_sam, msg_sam = sam_mgr.load_model(model_type)
                if ok_sam:
                    frame_start_rgb = None
                    if hasattr(self, 'is_image_dataset') and self.is_image_dataset:
                        if 0 <= start_f < len(self.image_files):
                            bgr_init = cv2.imread(self.image_files[start_f])
                            if bgr_init is not None:
                                frame_start_rgb = cv2.cvtColor(bgr_init, cv2.COLOR_BGR2RGB)
                    else:
                        cap_init = cv2.VideoCapture(self.video_filename)
                        cap_init.set(cv2.CAP_PROP_POS_FRAMES, start_f)
                        ret_init, bgr_init = cap_init.read()
                        if ret_init:
                            frame_start_rgb = cv2.cvtColor(bgr_init, cv2.COLOR_BGR2RGB)
                        cap_init.release()
                    
                    if frame_start_rgb is not None:
                        QApplication.setOverrideCursor(Qt.WaitCursor)
                        try:
                            initial_polygon = sam_mgr.predict_mask_from_prompt(
                                frame_start_rgb, points=points, labels=labels, box=None, text_prompt=text_prompt
                            )
                        except Exception as e:
                            logger.error(f"Failed to generate initial mask from SAM: {e}")
                            initial_polygon = None
                        finally:
                            QApplication.restoreOverrideCursor()
                        
                        if initial_polygon:
                            x_coords = [p[0] for p in initial_polygon]
                            y_coords = [p[1] for p in initial_polygon]
                            if x_coords and y_coords:
                                box = [int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))]

            manager = self.fast_tracker_manager
            if tracker_engine == 'ettrack':
                model_type = "E.T.Track"
            elif tracker_engine == 'ostrack_engine':
                model_type = "OSTrack Native TRT"
            elif tracker_engine == 'ostrack_trt':
                model_type = "OSTrack TRT"
            else:
                model_type = "OSTrack"
            success, msg = manager.load_model(model_type)
            if not success:
                QMessageBox.warning(self, "Tracker Error", msg)
                return
        else:
            if 'sam3' in model_type.lower():
                manager = self.sam3_native_manager
            elif 'trt' in model_type.lower():
                manager = self.sam2_trt_manager
            else:
                manager = self.sam_manager
        classes = list(self.canvas.class_colors.keys())
        if not classes:
            QMessageBox.warning(self, 'No Classes',
                'Please define at least one class before tracking.')
            return
        current_idx = classes.index(self.canvas.current_class
            ) if self.canvas.current_class in classes else 0
        from PyQt5.QtWidgets import QInputDialog
        target_class, ok = QInputDialog.getItem(self,
            'Tracked Object Class', 'Select class for the tracked object:',
            classes, current_idx, False)
        if not ok or not target_class:
            return
        self.save_undo_state(range(min(start_f, end_f), max(start_f, end_f) + 1))
        self.set_active_class(target_class)
        self.canvas.annotations = [a for a in self.canvas.annotations if 
            getattr(a, 'is_sam_preview', False) == False]
        progress_label = ('Detecting object frame by frame...' if strategy ==
            'detect' else ('Tracking object backward...' if is_backward else 'Tracking object...'))
        
        progress = None
        cancelled = False
        try:
            total_steps = abs(end_f - start_f) + 1
            progress = QProgressDialog(progress_label, 'Cancel', 0, total_steps, self)
            progress.setWindowModality(Qt.WindowModal)
            progress.show()
            QApplication.processEvents()

            if start_f == end_f:
                if tracker_engine in ['ettrack', 'ostrack', 'ostrack_trt', 'ostrack_engine']:
                    if tracker_engine == 'ettrack':
                        tracker_title = "E.T.Track"
                    elif tracker_engine == 'ostrack_engine':
                        tracker_title = "OSTrack Native TRT"
                    elif tracker_engine == 'ostrack_trt':
                        tracker_title = "OSTrack TRT"
                    else:
                        tracker_title = "OSTrack"
                    QMessageBox.warning(self, 'Invalid Scope', f'{tracker_title} cannot be used for a single frame. Please select a range or Whole Video.')
                    return
                self.statusBar.showMessage('Generating mask for single frame...')
                QApplication.setOverrideCursor(Qt.WaitCursor)
                frame = None
                try:
                    if hasattr(self, 'is_image_dataset') and self.is_image_dataset:
                        if 0 <= start_f < len(self.image_files):
                            frame_bgr = cv2.imread(self.image_files[start_f])
                            if frame_bgr is not None:
                                frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    else:
                        cap = cv2.VideoCapture(self.video_filename)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
                        ret, frame_bgr = cap.read()
                        if ret:
                            frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        cap.release()
                finally:
                    QApplication.restoreOverrideCursor()

                if frame is None:
                    QMessageBox.warning(self, 'Error', 'Could not read the frame.')
                    return
                polygon = manager.predict_mask_from_prompt(frame, points=points,
                    labels=labels, box=box, text_prompt=text_prompt)
                if polygon:
                    if start_f not in self.frame_annotations:
                        self.frame_annotations[start_f] = []
                    ann_rect = QRect(0, 0, 0, 0)
                    if box:
                        ann_rect = QRect(box[0], box[1], box[2] - box[0], box[3
                            ] - box[1])
                    else:
                        x_coords = [p[0] for p in polygon]
                        y_coords = [p[1] for p in polygon]
                        if x_coords and y_coords:
                            ann_rect = QRect(int(min(x_coords)), int(min(
                                y_coords)), int(max(x_coords) - min(x_coords)),
                                int(max(y_coords) - min(y_coords)))
                    default_attributes = {'Size': -1, 'Quality': -1}
                    if hasattr(self, 'get_default_attributes_for_class'):
                        default_attributes = self.get_default_attributes_for_class(
                            target_class)
                    ann = BoundingBox(ann_rect, target_class, attributes=
                        default_attributes, color=self.canvas.class_colors.get(
                        target_class, QColor(0, 255, 0)), source='sam_tracked',
                        segmentation=polygon if self.sam_interactive_dock.
                        get_save_segmentation() else None)
                    if should_blur:
                        _apply_sam_blur(start_f, polygon, ann_rect)
                    else:
                        self.frame_annotations[start_f].append(ann)
                else:
                    QMessageBox.warning(self, 'Tracking Error',
                        'No object detected.')
                return
            if strategy == 'detect':
                if tracker_engine in ['ettrack', 'ostrack', 'ostrack_trt', 'ostrack_engine']:
                    if tracker_engine == 'ettrack':
                        tracker_title = "E.T.Track"
                    elif tracker_engine == 'ostrack_engine':
                        tracker_title = "OSTrack Native TRT"
                    elif tracker_engine == 'ostrack_trt':
                        tracker_title = "OSTrack TRT"
                    else:
                        tracker_title = "OSTrack"
                    QMessageBox.warning(self, 'Invalid Strategy', f'{tracker_title} is a temporal tracker and cannot be used for frame-by-frame detection.')
                    return
                det_model_type = self.sam_interactive_dock.get_det_model_type()
                zero_shot_model = None
                if det_model_type:
                    if not hasattr(self, 'zero_shot_manager'
                        ) or self.zero_shot_manager is None:
                        from .utils.zero_shot_manager import ZeroShotManager
                        self.zero_shot_manager = ZeroShotManager()
                    prog_load = QProgressDialog(f'Loading {det_model_type}...',
                        'Cancel', 0, 0, self)
                    prog_load.setWindowModality(Qt.WindowModal)
                    prog_load.show()
                    QApplication.processEvents()
                    checkpoints_dir = os.path.join(os.path.dirname(os.path.
                        abspath(__file__)), '..', 'checkpoints')
                    success, msg = self.zero_shot_manager.load_model(det_model_type
                        , checkpoints_dir)
                    prog_load.close()
                    if not success:
                        QMessageBox.warning(self, 'Error',
                            f'Failed to load Zero-Shot Model:\n{msg}')
                        return
                    zero_shot_model = self.zero_shot_manager.detector
                    if text_prompt:
                        zero_shot_model.set_classes([text_prompt])
                self.statusBar.showMessage(
                    f'Frame-by-Frame Detection using {model_type}...')
                detected_count = 0
                source_iter = list(range(start_f, end_f - 1, -1)) if is_backward else list(range(start_f, end_f + 1))
                if hasattr(self, 'is_image_dataset') and self.is_image_dataset:
                    def _read_frame_detect(f_idx):
                        if 0 <= f_idx < len(self.image_files):
                            bgr = cv2.imread(self.image_files[f_idx])
                            if bgr is not None:
                                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        return None
                else:
                    _detect_cap = cv2.VideoCapture(self.video_filename)

                    def _read_frame_detect(f_idx):
                        _detect_cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                        ret, bgr = _detect_cap.read()
                        if ret:
                            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                        return None
                try:
                    processed_count = 0
                    for f_idx in source_iter:
                        if progress.wasCanceled():
                            cancelled = True
                            break
                        processed_count += 1
                        progress.setValue(processed_count)
                        self.set_current_frame(f_idx)
                        QApplication.processEvents()
                        frame = _read_frame_detect(f_idx)
                        if frame is None:
                            continue
                        try:
                            current_box = box
                            if zero_shot_model:
                                visual_prompts = None
                                if box and not text_prompt:
                                    visual_prompts = {'bboxes': [[box[0], box[1],
                                        box[2], box[3]]], 'cls': [0]}
                                detections = zero_shot_model.predict(frame,
                                    visual_prompts=visual_prompts)
                                if not detections:
                                    continue
                                best_det = max(detections, key=lambda x: x['score'])
                                current_box = best_det['box']
                            polygon = manager.predict_mask_from_prompt(frame,
                                points=points, labels=labels, box=current_box,
                                text_prompt=text_prompt)
                        except Exception:
                            continue
                        if not polygon:
                            continue
                        x_coords = [p[0] for p in polygon]
                        y_coords = [p[1] for p in polygon]
                        ann_rect = QRect(int(min(x_coords)), int(min(y_coords)),
                            int(max(x_coords) - min(x_coords)), int(max(y_coords) -
                            min(y_coords)))
                        if f_idx not in self.frame_annotations:
                            self.frame_annotations[f_idx] = []
                        default_attributes = {'Size': -1, 'Quality': -1}
                        if hasattr(self, 'get_default_attributes_for_class'):
                            default_attributes = self.get_default_attributes_for_class(
                                target_class)
                        ann = BoundingBox(ann_rect, target_class, attributes=
                            default_attributes, color=self.canvas.class_colors.get(
                            target_class, QColor(0, 255, 0)), source='sam_detected',
                            segmentation=polygon if self.sam_interactive_dock.
                            get_save_segmentation() else None)
                        if should_blur:
                            _apply_sam_blur(f_idx, polygon, ann_rect)
                        else:
                            self.frame_annotations[f_idx].append(ann)
                        detected_count += 1
                        self.load_current_frame_annotations()
                        self.update_frame_display()
                finally:
                    if not (hasattr(self, 'is_image_dataset') and self.is_image_dataset):
                        _detect_cap.release()
                return
            self.statusBar.showMessage(f'Tracking using {model_type}...')
            
            is_sam_tracker = manager in [getattr(self, 'sam_manager', None), getattr(self, 'sam2_trt_manager', None)]
            CHUNK_SIZE = 400 if is_sam_tracker else (abs(end_f - start_f) + 1)
            
            current_chunk_start = start_f
            chunk_points = points
            chunk_labels = labels
            chunk_box = box
            chunk_text_prompt = text_prompt

            def _has_more_chunks():
                return current_chunk_start >= end_f if is_backward else current_chunk_start <= end_f

            processed_count = 0
            
            while _has_more_chunks():
                if is_backward:
                    current_chunk_end = max(current_chunk_start - CHUNK_SIZE + 1, end_f)
                else:
                    current_chunk_end = min(current_chunk_start + CHUNK_SIZE - 1, end_f)

                if progress.wasCanceled():
                    cancelled = True
                    break
                    
                if manager == getattr(self, 'sam3_native_manager', None):
                    res_path = getattr(self, 'video_filename', None)
                    if hasattr(self, 'is_image_dataset') and self.is_image_dataset:
                        res_path = os.path.dirname(self.image_files[0]) if self.image_files else None
                    results_generator = manager.track_video_from_prompt(
                        resource_path=res_path, start_f=current_chunk_start, end_f=current_chunk_end, 
                        points=chunk_points, labels=chunk_labels, box=chunk_box, text_prompt=chunk_text_prompt)
                else:
                    def chunk_frame_generator():
                        if hasattr(self, 'is_image_dataset') and self.is_image_dataset:
                            end_bound = current_chunk_end - 1 if is_backward else current_chunk_end + 1
                            for f_idx in range(current_chunk_start, end_bound, step):
                                if 0 <= f_idx < len(self.image_files):
                                    frame = cv2.imread(self.image_files[f_idx])
                                    if frame is not None:
                                        yield cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        else:
                            cap = cv2.VideoCapture(self.video_filename)
                            end_bound = current_chunk_end - 1 if is_backward else current_chunk_end + 1
                            for f_idx in range(current_chunk_start, end_bound, step):
                                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                                ret, frame = cap.read()
                                if not ret:
                                    break
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                yield frame_rgb
                            cap.release()

                    source_input = chunk_frame_generator()
                    if not is_backward and not getattr(self, 'is_image_dataset', False) and hasattr(self, 'video_filename') and self.video_filename:
                        if manager == getattr(self, 'fast_tracker_manager', None) or current_chunk_start == 0:
                            source_input = self.video_filename

                    if manager == getattr(self, 'fast_tracker_manager', None):
                        results_generator = manager.track_video_from_prompt(source_input, points=chunk_points, labels=chunk_labels, box=chunk_box, text_prompt=chunk_text_prompt, model_type=model_type, initial_polygon=initial_polygon, start_f=current_chunk_start, end_f=current_chunk_end)
                    else:
                        results_generator = manager.track_video_from_prompt(source_input, points=chunk_points, labels=chunk_labels, box=chunk_box, text_prompt=chunk_text_prompt, model_type=model_type, start_f=current_chunk_start, end_f=current_chunk_end)
                
                current_f = current_chunk_start
                last_box = None
                
                for success, track_res in results_generator:
                    if progress.wasCanceled():
                        cancelled = True
                        break
                    self.set_current_frame(current_f)
                    QApplication.processEvents()
                    if not success:
                        QMessageBox.warning(self, 'Tracking Error', track_res)
                        break
                    tracked_boxes = track_res['boxes']
                    tracked_polygons = track_res['polygons']
                    if len(tracked_boxes) > 0:
                        t_box = tracked_boxes[0]
                        last_box = t_box
                        polygon = tracked_polygons[0] if len(tracked_polygons) > 0 else None
                        rect = QRect(t_box[0], t_box[1], t_box[2] - t_box[0], t_box[3] - t_box[1])
                        if current_f not in self.frame_annotations:
                            self.frame_annotations[current_f] = []
                        default_attributes = {'Size': -1, 'Quality': -1}
                        if hasattr(self, 'get_default_attributes_for_class'):
                            default_attributes = self.get_default_attributes_for_class(target_class)
                        ann = BoundingBox(rect, target_class, attributes=default_attributes, color=self.canvas.class_colors.get(target_class, QColor(0, 255, 0)), source='sam_tracked', segmentation=polygon if self.sam_interactive_dock.get_save_segmentation() else None)
                        if should_blur:
                            _apply_sam_blur(current_f, polygon, rect)
                        else:
                            self.frame_annotations[current_f].append(ann)
                    self.load_current_frame_annotations()
                    self.update_frame_display()
                    processed_count += 1
                    progress.setValue(processed_count)
                    current_f += step

                if progress.wasCanceled():
                    cancelled = True
                    break
                    
                chunk_done = (current_chunk_end <= end_f) if is_backward else (current_chunk_end >= end_f)
                if is_sam_tracker and not chunk_done:
                    if hasattr(manager, 'clear_session'):
                        manager.clear_session()
                    if last_box:
                        chunk_box = [last_box[0], last_box[1], last_box[2], last_box[3]]
                        chunk_points = None
                        chunk_labels = None
                        chunk_text_prompt = None
                    else:
                        QMessageBox.warning(self, 'Tracking Lost', f'Lost object at frame {current_f-step}. Aborting remaining frames.')
                        break
                        
                current_chunk_start = current_chunk_end + step
        except Exception as e:
            logger.error(f"Error during SAM tracking: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Tracking Error", f"An error occurred during tracking:\n{e}")
        finally:
            if progress is not None:
                try:
                    progress.close()
                    progress.deleteLater()
                except Exception:
                    pass
            QApplication.restoreOverrideCursor()
            self.on_sam_clear_requested()
            self.seek_to_frame(self.current_frame)
            if cancelled:
                self.statusBar.showMessage('SAM Tracking cancelled by user.', 4000)
            else:
                self.statusBar.showMessage('SAM processing completed.', 5000)
            if hasattr(self, 'canvas') and self.canvas:
                self.canvas.setFocus()

    @log_exceptions
    def viat_finish_integration(self):
        """Finalizes the integration mode and merges the dataset."""
        from PyQt5.QtWidgets import QMessageBox, QInputDialog
        import cv2
        if not self.integration_mode or not self.integration_main_dataset:
            return
        reply = QMessageBox.question(self, 'Finish Integration',
            'Are you done reviewing? This will merge the current dataset into the Main Dataset.'
            , QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            source_folder = ''
            if hasattr(self, '_viat_dataset_info') and self._viat_dataset_info:
                source_folder = self._viat_dataset_info.root
            elif self.image_files:
                source_folder = os.path.dirname(self.image_files[0])
            if not source_folder:
                QMessageBox.warning(self, 'Error',
                    'Could not determine current dataset path.')
                return
            self.update_frame_annotations()
            if getattr(self, 'is_image_dataset', False) and hasattr(self,
                '_viat_dataset_info'):
                from viat.utils.dataset_manager import update_dataset_labels
                from PyQt5.QtWidgets import QApplication
                self.statusBar.showMessage(
                    'Updating dataset labels before merge...')
                QApplication.processEvents()
                update_dataset_labels(self._viat_dataset_info, self.
                    frame_annotations, self.image_files, current_classes=
                    list(self.canvas.class_colors.keys()))
            else:
                self.save_project()
            text, ok = QInputDialog.getText(self, 'Resize Dataset',
                'Resize all images before merging? Enter W,H (or clear to skip):'
                , text='640,640')
            if ok and text.strip():
                try:
                    parts = text.split(',')
                    w, h = int(parts[0].strip()), int(parts[1].strip())
                    resized_count = 0
                    for root, dirs, files in os.walk(source_folder):
                        if 'removed' in root or 'review' in root:
                            continue
                        for file in files:
                            ext = os.path.splitext(file)[1].lower()
                            if ext in {'.jpg', '.jpeg', '.png', '.bmp',
                                '.tiff', '.webp'}:
                                img_path = os.path.join(root, file)
                                img = cv2.imread(img_path)
                                if img is not None:
                                    img_resized = cv2.resize(img, (w, h))
                                    cv2.imwrite(img_path, img_resized)
                                    resized_count += 1
                    self.statusBar.showMessage(
                        f'Resized {resized_count} images to {w}x{h}', 5000)
                except Exception as e:
                    QMessageBox.warning(self, 'Resize Error',
                        f'Failed to resize images:\n{e}')
            self.viat_merge_dataset(source_folder=source_folder,
                target_folder=self.integration_main_dataset)
            self.integration_mode = False
            self.integration_main_dataset = ''
            if hasattr(self, 'finish_integration_action'):
                self.finish_integration_action.setVisible(False)
            if hasattr(self, 'finish_integration_move_action'):
                self.finish_integration_move_action.setVisible(False)
            if hasattr(self, 'finish_integration_sep'):
                self.finish_integration_sep.setVisible(False)

    @log_exceptions
    def viat_finish_integration_move(self):
        """Moves the dataset to a review folder instead of merging."""
        from PyQt5.QtWidgets import QMessageBox, QInputDialog, QFileDialog
        import shutil
        import os
        from datetime import datetime
        if not self.integration_mode:
            return
        reply = QMessageBox.question(self, 'Move Dataset',
            'Are you sure you want to move this dataset to a review folder instead of merging?'
            , QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            source_folder = ''
            if hasattr(self, '_viat_dataset_info') and self._viat_dataset_info:
                source_folder = self._viat_dataset_info.root
            elif self.image_files:
                source_folder = os.path.dirname(self.image_files[0])
            if not source_folder:
                QMessageBox.warning(self, 'Error',
                    'Could not determine current dataset path.')
                return
            target_folder = QFileDialog.getExistingDirectory(self,
                'Select Review/Destination Folder', '', QFileDialog.
                ShowDirsOnly)
            if not target_folder:
                return
            details, ok = QInputDialog.getMultiLineText(self,
                'Dataset Details',
                'Enter details for this dataset (will be saved to details.md):'
                )
            if ok:
                try:
                    dataset_name = os.path.basename(source_folder)
                    details_file = os.path.join(target_folder, 'details.md')
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    with open(details_file, 'a', encoding='utf-8') as f:
                        f.write(f'\n## Dataset: {dataset_name} ({timestamp})\n'
                            )
                        if details.strip():
                            f.write(f'{details.strip()}\n')
                        else:
                            f.write('No details provided.\n')
                    self.reset_media_state()
                    dest_path = os.path.join(target_folder, dataset_name)
                    if os.path.exists(dest_path):
                        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                        dataset_name = f'{dataset_name}_{timestamp}'
                        dest_path = os.path.join(target_folder, dataset_name)
                    shutil.move(source_folder, dest_path)
                    self.statusBar.showMessage(f'Moved dataset to {dest_path}',
                        5000)
                    self.integration_mode = False
                    self.integration_main_dataset = ''
                    if hasattr(self, 'finish_integration_action'):
                        self.finish_integration_action.setVisible(False)
                    if hasattr(self, 'finish_integration_move_action'):
                        self.finish_integration_move_action.setVisible(False)
                    if hasattr(self, 'finish_integration_sep'):
                        self.finish_integration_sep.setVisible(False)
                except Exception as e:
                    QMessageBox.warning(self, 'Move Error',
                        f'Failed to move dataset:\n{e}')

    @log_exceptions
    def setup_playback_timer(self):
        """Set up the timer for video playback."""
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.next_frame)

    @log_exceptions
    def setup_autosave(self):
        """Set up auto-save functionality."""
        self.autosave_timer = QTimer()
        self.autosave_timer.timeout.connect(self.perform_autosave)
        if self.autosave_enabled:
            self.autosave_timer.start(self.autosave_interval)
            if hasattr(self, 'statusBar') and self.statusBar:
                self.statusBar.showMessage('Auto-save enabled', 3000)

    @log_exceptions
    def open_video(self):
        """Open a video file."""
        if not self.check_unsaved_changes():
            return
        filename, _ = QFileDialog.getOpenFileName(self, 'Open Video', '',
            'Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)')
        if filename:
            self.project_file = None
            self.image_dataset_info = None
            self.is_image_dataset = False
            self.canvas.annotations = []
            self.frame_annotations = {}
            self.deleted_frames = set()
            self.deleted_annotations = {}
            self.current_frame = 0
            self.frame_hashes = {}
            self.duplicate_frames_cache = {}
            self.load_video_file(filename)

    @log_exceptions
    def fast_export_video_and_next(self):
        """Fast export to Raya with classes TXT and load next video."""
        if not hasattr(self, 'video_filename') or not self.video_filename:
            return

        has_blurs = hasattr(self, 'blur_manager') and bool(self.blur_manager.blur_regions)

        # 1. Export Raya with classes
        default_dir = os.path.dirname(self.video_filename)
        default_filename = os.path.splitext(os.path.basename(self.video_filename))[0]
        
        if has_blurs:
            export_path = os.path.join(default_dir, default_filename + '_blurred.txt')
        else:
            export_path = os.path.join(default_dir, default_filename + '.txt')

        from viat.utils.file_operations import export_raya_with_classes_annotations
        all_annotations = []
        for frame_num, annotations in self.frame_annotations.items():
            for annotation in annotations:
                import copy
                annotation_copy = copy.copy(annotation)
                annotation_copy.frame = frame_num
                all_annotations.append(annotation_copy)
                
        if not all_annotations and self.canvas.annotations:
            all_annotations = self.canvas.annotations
            
        classes = list(self.canvas.class_colors.keys())
        try:
            deleted = getattr(self, 'deleted_frames', set())
            total_f = getattr(self, 'total_frames', None)
            export_raya_with_classes_annotations(export_path, all_annotations, classes, deleted_frames=deleted, total_frames=total_f)
            self.statusBar.showMessage(f'Fast Exported to {os.path.basename(export_path)}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to fast export: {str(e)}')
            import traceback
            traceback.print_exc()
            return
            
        if hasattr(self, 'auto_save_project_on_fast_export_act') and self.auto_save_project_on_fast_export_act.isChecked():
            target_json = os.path.join(default_dir, default_filename + '.json')
            self.save_project(target_json)
            self.project_file = target_json
            
        if hasattr(self, 'clip_cuts_dock'):
            self.export_clip_cuts(auto_export_dir=default_dir)
            self.clip_cuts_dock.clear_all()
            
        if has_blurs:
            self.export_blurred_video(interactive=False)
            
        # 2. Find next video
        video_exts = ['.mp4', '.avi', '.mov', '.mkv']
        all_files = os.listdir(default_dir)
        video_files = [f for f in all_files if os.path.splitext(f)[1].lower() in video_exts]
        video_files = natsorted(video_files)
        
        current_idx = -1
        current_basename = os.path.basename(self.video_filename)
        if current_basename in video_files:
            current_idx = video_files.index(current_basename)
            
        next_video_path = None
        for i in range(current_idx + 1, len(video_files)):
            candidate_name = video_files[i]
            candidate_base = os.path.splitext(candidate_name)[0]
            candidate_txt = candidate_base + '.txt'
            if candidate_txt in all_files:
                continue
            else:
                next_video_path = os.path.join(default_dir, candidate_name)
                break
                
        if next_video_path:
            self.project_file = None
            self.image_dataset_info = None
            self.is_image_dataset = False
            self.canvas.annotations = []
            self.frame_annotations = {}
            self.deleted_frames = set()
            self.deleted_annotations = {}
            self.current_frame = 0
            self.frame_hashes = {}
            self.duplicate_frames_cache = {}
            self.load_video_file(next_video_path)
        else:
            self._prompt_merge_and_open_next(default_dir)

    @log_exceptions
    def delete_video_and_next(self):
        """Delete the current video (and its .txt annotation if present) then load the next unannotated video."""
        if not hasattr(self, 'video_filename') or not self.video_filename:
            return

        video_path = self.video_filename
        default_dir = os.path.dirname(video_path)
        default_filename = os.path.splitext(os.path.basename(video_path))[0]
        txt_path = os.path.join(default_dir, default_filename + '.txt')

        # Confirm deletion
        msg = f'Delete video file:\n{os.path.basename(video_path)}'
        if os.path.isfile(txt_path):
            msg += f'\n\nAssociated annotation file:\n{os.path.basename(txt_path)}\n\nwill also be deleted.'
        reply = QMessageBox.question(
            self, 'Delete Video & Next', msg,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        # Find next video BEFORE deleting (so the file listing is still intact)
        video_exts = ['.mp4', '.avi', '.mov', '.mkv']
        all_files = os.listdir(default_dir)
        video_files = natsorted([f for f in all_files if os.path.splitext(f)[1].lower() in video_exts])

        current_basename = os.path.basename(video_path)
        current_idx = video_files.index(current_basename) if current_basename in video_files else -1

        next_video_path = None
        for i in range(current_idx + 1, len(video_files)):
            candidate_name = video_files[i]
            candidate_base = os.path.splitext(candidate_name)[0]
            candidate_txt = candidate_base + '.txt'
            if candidate_txt in all_files:
                continue
            next_video_path = os.path.join(default_dir, candidate_name)
            break

        # Reset state before deleting
        self.image_dataset_info = None
        self.is_image_dataset = False
        self.canvas.annotations = []
        self.frame_annotations = {}
        self.deleted_frames = set()
        self.deleted_annotations = {}
        self.current_frame = 0
        self.frame_hashes = {}
        self.duplicate_frames_cache = {}
        self.video_filename = None
        self.project_file = None
        
        # Release the video capture object so the file is not locked
        if getattr(self, 'cap', None):
            self.cap.release()
            self.cap = None

        # Delete files
        try:
            if os.path.isfile(txt_path):
                os.remove(txt_path)
            os.remove(video_path)
            self.statusBar.showMessage(f'Deleted {os.path.basename(video_path)}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to delete file(s): {str(e)}')
            return

        # Load next video
        if next_video_path:
            self.load_video_file(next_video_path)
        else:
            self._prompt_merge_and_open_next(default_dir)

    @log_exceptions
    def _prompt_merge_and_open_next(self, default_dir):
        reply = QMessageBox.question(self, 'End of Folder',
            'No more unannotated videos found in this folder.\nDo you want to merge all annotated videos in this folder?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        
        if reply == QMessageBox.Yes:
            # Release current video capture to free resources
            if getattr(self, 'cap', None):
                self.cap.release()
                self.cap = None

            try:
                import sys
                from pathlib import Path
                
                # Make sure the root directory is in sys.path
                root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                if root_path not in sys.path:
                    sys.path.insert(0, root_path)
                    
                from many2single import merge_dataset_programmatic
                
                folder = Path(default_dir)
                out_video = folder / 'outvideo.mp4'
                out_labels = folder / 'outvideo.txt'
                
                self.statusBar.showMessage('Merging videos... This may take a while.')
                QApplication.processEvents()
                
                success, msg = merge_dataset_programmatic(folder, out_video, out_labels)
                
                if success:
                    QMessageBox.information(self, "Merge Successful", msg)
                self.statusBar.showMessage('Merge complete', 5000)
            except Exception as e:
                import traceback
                traceback.print_exc()
                QMessageBox.critical(self, "Merge Error", f"Failed to merge videos:\n{e}")

        # Ask for a new video
        filename, _ = QFileDialog.getOpenFileName(self, 'Open Video', default_dir, 'Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)')
        if filename:
            self.project_file = None
            self.image_dataset_info = None
            self.is_image_dataset = False
            self.canvas.annotations = []
            self.frame_annotations = {}
            self.deleted_frames = set()
            self.deleted_annotations = {}
            self.current_frame = 0
            self.frame_hashes = {}
            self.duplicate_frames_cache = {}
            self.load_video_file(filename)

    @log_exceptions
    def open_scene_detect_for_video(self):
        """Open scene detection dialog for the current loaded video."""
        if not self.video_filename:
            return
            
        dialog = SceneDetectDialog(self, video_path=self.video_filename)
        if dialog.exec_() == QDialog.Accepted:
            self.video_mode = True
            self.custom_video_groups = dialog.video_groups
            self.custom_single_images = dialog.single_images
            self.update_video_groups()
            
            if hasattr(self, 'video_manager_dock'):
                self.video_manager_dock.set_active(True)
                self.video_manager_dock.chk_video_mode.blockSignals(True)
                self.video_manager_dock.chk_video_mode.setChecked(True)
                self.video_manager_dock.chk_video_mode.blockSignals(False)
                # Select first video cut
                if self.video_groups:
                    first_vid = list(self.video_groups.keys())[0]
                    self.video_manager_dock.select_video(first_vid)
                    self.on_video_selected(first_vid)

    @log_exceptions
    def load_video_file(self, filename):
        """Load a video file and display the first frame."""
        if self.cap:
            self.cap.release()
            
        if hasattr(self, 'performance_manager') and self.performance_manager:
            self.performance_manager.clear_cache()
            
        if not getattr(self, '_loading_from_project', False):
            if hasattr(self, 'blur_manager') and self.blur_manager is not None:
                self.blur_manager.clear_all()
            
        self.cap = cv2.VideoCapture(filename, cv2.CAP_ANY)
        if not self.cap.isOpened():
            QMessageBox.critical(self, 'Error', 'Could not open video file!')
            self.cap = None
            return False
        self.video_filename = filename
        
        # Check for metadata file
        self.video_metadata = None
        metadata_path = os.path.join(os.path.dirname(filename), "dataset_video_metadata.json")
        if os.path.exists(metadata_path):
            try:
                import json
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    self.video_metadata = json.load(f)
            except Exception:
                pass

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        ret, frame = self.cap.read()
        if ret:
            frame = self._process_frame_metadata(frame, 0)
            self.canvas.set_frame(frame)
            self.update_frame_info()
            self.statusBar.showMessage(
                f'Loaded video: {os.path.basename(filename)}')
            self.video_filename = filename
            video_base = os.path.dirname(filename)
            video_name = os.path.splitext(os.path.basename(filename))[0]
            auto_save_folder = os.path.join(video_base, 'autosaves')
            os.makedirs(auto_save_folder, exist_ok=True)
            self.autosave_file = os.path.join(auto_save_folder, video_name +
                '_autosave.json')
            if self.autosave_enabled and not self.autosave_timer.isActive():
                self.autosave_timer.start(self.autosave_interval)
            if self.duplicate_frames_enabled and not self.frame_hashes:
                reply = QMessageBox.question(self,
                    'Duplicate Frame Detection',
                    """Would you like to scan this video for duplicate frames?
(This will help automatically propagate annotations)"""
                    , QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if reply == QMessageBox.Yes:
                    QTimer.singleShot(500, self.scan_video_for_duplicates)
            elif self.duplicate_frames_enabled and self.frame_hashes:
                duplicate_count = sum(len(frames) - 1 for frames in self.
                    duplicate_frames_cache.values() if len(frames) > 1)
                self.statusBar.showMessage(
                    f'Loaded {duplicate_count} duplicate frames from project file'
                    , 5000)
            if not hasattr(self, '_loading_from_project'
                ) or not self._loading_from_project:
                self.check_for_annotation_files(filename)
                
            return True
        else:
            QMessageBox.critical(self, 'Error', 'Could not read video frame!')
            self.cap.release()
            self.cap = None
            return False

    def _process_frame_metadata(self, frame, frame_num):
        """Process the frame according to loaded metadata (e.g., cropping padded regions)."""
        if hasattr(self, 'video_metadata') and self.video_metadata and self.video_metadata.get("resize_mode") == "pad":
            sizes = self.video_metadata.get("original_sizes", {})
            orig_size = sizes.get(str(frame_num))
            if orig_size and len(orig_size) == 2:
                orig_w, orig_h = orig_size
                frame = frame[:orig_h, :orig_w]
        return frame

    @log_exceptions
    def load_video_from_project(self, video_path, current_frame):
        """Load a video from project information."""
        original_check_method = self.check_for_annotation_files
        self.check_for_annotation_files = lambda x: None
        success = self.load_video_file(video_path)
        self.check_for_annotation_files = original_check_method
        if success:
            if current_frame > 0 and current_frame < self.total_frames:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = self.cap.read()
                if ret:
                    frame = self._process_frame_metadata(frame, current_frame)
                    self.current_frame = current_frame
                    self.canvas.set_frame(frame)
                    self.update_frame_info()
                    self.load_current_frame_annotations()
        return success

    @log_exceptions
    def open_image_folder(self):
        """Open a folder of images OR a labeled dataset via file dialog."""
        if not self.check_unsaved_changes():
            return
        folder_path = QFileDialog.getExistingDirectory(self,
            'Open Image Folder / Dataset', '', QFileDialog.ShowDirsOnly)
        if not folder_path:
            return
        self.load_image_dataset_path(folder_path)

    @log_exceptions
    def load_image_dataset_path(self, folder_path):
        """Load a folder of images OR a labeled dataset directly from a path."""
        self.reset_media_state()
        info = scan_dataset(folder_path)
        self._viat_dataset_info = info
        if hasattr(self, 'update_dataset_labels_action'):
            self.update_dataset_labels_action.setEnabled(info.layout !=
                'simple')
        if info.image_count == 0:
            QMessageBox.warning(self, 'Open Image Folder',
                'No image files found in the selected folder!')
            return
        self.is_image_dataset = True
        if hasattr(self, 'video_manager_dock'):
            self.video_manager_dock.show()
            
        self.current_frame = 0
        self.statusBar.showMessage('Loading dataset in background...', 5000)
        result = load_dataset_into_app(self, info, BoundingBox)
        folder_name = os.path.basename(folder_path)
        self.setWindowTitle(
            f'VIAT - {folder_name} [{info.layout}] (Loading...)')
        if hasattr(self, 'play_button'):
            self.play_button.setEnabled(True)
            self.play_button.setIcon(self.icon_provider.get_icon(
                'media-playback-start'))
        self.autosave_file = os.path.join(folder_path,
            f'{folder_name}_autosave.json')
        if self.autosave_enabled and not self.autosave_timer.isActive():
            self.autosave_timer.start(self.autosave_interval)
        if info.classes_conflict:
            QMessageBox.warning(self, 'Class Name Conflict', info.
                classes_conflict)
        try:
            _viat_init_dataset_log(self, info)
        except Exception:
            pass

    @log_exceptions
    def open_simple_image_folder(self, folder_path):
        """Open a simple folder of images."""
        if not self.check_unsaved_changes():
            return
        self.reset_media_state()
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        image_files = []
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(os.path.join(folder_path, file))
        if not image_files:
            QMessageBox.warning(self, 'Open Image Folder',
                'No image files found in the selected folder!')
            return
        image_files.sort()
        self.image_files = image_files
        self.total_frames = len(image_files)
        self.current_frame = 0
        self.is_image_dataset = True
        
        if hasattr(self, 'video_manager_dock'):
            self.video_manager_dock.show()
        self.load_current_image()
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self.update_frame_info()
        folder_name = os.path.basename(folder_path)
        self.setWindowTitle(
            f'Video Annotation Tool - Image Folder: {folder_name}')
        if hasattr(self, 'play_button'):
            self.play_button.setEnabled(True)
            self.play_button.setIcon(self.icon_provider.get_icon(
                'media-playback-start'))
        self.autosave_file = os.path.join(folder_path,
            f'{folder_name}_autosave.json')
        if self.autosave_enabled and not self.autosave_timer.isActive():
            self.autosave_timer.start(self.autosave_interval)
        self.check_for_image_annotation_files(folder_path, folder_name)

    @log_exceptions
    def open_image_dataset(self, folder_path=None):
        """Open a labeled dataset (Roboflow/splits/etc). Now delegates
        to the unified open_image_folder() flow."""
        self.open_image_folder()
        if hasattr(self, 'chk_video_mode'):
            self.chk_video_mode.setVisible(True)

    @log_exceptions
    def load_image_dataset_from_project(self, image_dataset_info,
        current_frame=None):
        """Load an image dataset from a saved project / autosave.

        current_frame is optional (defaults to self.current_frame, which
        load_project has just restored). Re-scans the dataset folder so
        that _viat_dataset_info, _viat_frame_to_split, and labels are
        populated, then refreshes the class UI.
        """
        if current_frame is None:
            current_frame = getattr(self, 'current_frame', 0)
        base_folder = image_dataset_info.get('base_folder', '')
        if not os.path.exists(base_folder):
            reply = QMessageBox.question(self, 'Folder Not Found',
                f"""The original image folder '{base_folder}' was not found.
Would you like to locate it?"""
                , QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                new_base = QFileDialog.getExistingDirectory(self,
                    'Locate Image Folder', '', QFileDialog.ShowDirsOnly)
                if new_base:
                    base_folder = new_base
                else:
                    return False
            else:
                return False
        from viat.utils.dataset_manager import scan_dataset, load_dataset_into_app
        info = scan_dataset(base_folder)
        self._viat_dataset_info = info
        relative_paths = image_dataset_info.get('image_files', [])
        if relative_paths:
            image_files = []
            for rel in relative_paths:
                ap = os.path.join(base_folder, rel)
                if os.path.exists(ap):
                    image_files.append(ap)
        else:
            image_files = info.all_images
        if not image_files:
            QMessageBox.critical(self, 'Error',
                'No image files could be loaded.')
            return False
        saved_annotations = getattr(self, 'frame_annotations', {})
        saved_class_colors = getattr(self.canvas, 'class_colors', {})
        saved_class_attrs = getattr(self.canvas, 'class_attributes', {})
        saved_blur_regions = self.blur_manager.to_dict() if hasattr(self, 'blur_manager') and self.blur_manager else None
        self.is_image_dataset = True
        self.current_frame = current_frame
        if hasattr(self, 'chk_video_mode'):
            self.chk_video_mode.setVisible(True)
        load_dataset_into_app(self, info, BoundingBox)
        if hasattr(self, '_dataset_loader') and self._dataset_loader:

            def apply_project_overlay(stats):
                for frame_idx, anns in saved_annotations.items():
                    self.frame_annotations[frame_idx] = anns
                self.canvas.class_colors.update(saved_class_colors)
                self.canvas.class_attributes.update(saved_class_attrs)
                self.class_colors = self.canvas.class_colors
                self.class_attributes = self.canvas.class_attributes
                if saved_blur_regions and hasattr(self, 'blur_manager') and self.blur_manager:
                    self.blur_manager.from_dict(saved_blur_regions)
                self.current_frame = min(self.current_frame, max(0, self.
                    total_frames - 1))
                self.frame_slider.blockSignals(True)
                self.frame_slider.setMinimum(0)
                self.frame_slider.setMaximum(max(0, self.total_frames - 1))
                self.frame_slider.setValue(self.current_frame)
                self.frame_slider.blockSignals(False)
                self.load_current_image()
                self.update_frame_info()
            self._dataset_loader.finishedLoading.connect(apply_project_overlay)
        folder_name = os.path.basename(base_folder)
        self.setWindowTitle(f'VIAT - Image Dataset: {folder_name} (Loading...)'
            )
        if hasattr(self, 'play_button'):
            self.play_button.setEnabled(True)
            self.play_button.setIcon(self.icon_provider.get_icon(
                'media-playback-start'))
        try:
            _viat_init_dataset_log(self, info)
        except Exception:
            pass
        self.refresh_class_ui()
        self.update_annotation_list()
        return True

    @log_exceptions
    def load_current_image(self):
        """Load the current image from the image dataset."""
        if not hasattr(self, 'image_files') or not self.image_files:
            return
        if 0 <= self.current_frame < len(self.image_files):
            image_path = self.image_files[self.current_frame]
            try:
                frame = cv2.imread(image_path)
            except Exception as e:
                logger.error(f'Failed to load image {image_path}: {e}')
                frame = None
            if frame is not None:
                self.canvas.set_frame(frame)
                self.load_current_frame_annotations()
                self.update_frame_info()
                return True
            else:
                self.statusBar.showMessage(
                    f'Error loading image: {os.path.basename(image_path)}')
                QMessageBox.warning(self, 'Image Load Error',
                    f"""Cannot load image:
{image_path}

The file might be corrupted or have an unsupported format."""
                    )
                return False

    @log_exceptions
    def cancel_dataset_loading(self):
        if hasattr(self, '_dataset_loader') and self._dataset_loader:
            self._dataset_loader.cancel()
            self.statusBar.showMessage('Loading cancelled.', 3000)
            if hasattr(self, 'cancel_loading_action'):
                self.cancel_loading_action.setVisible(False)

    @log_exceptions
    def reset_media_state(self):
        """Reset all state related to the current media (video or image dataset)"""
        self.cancel_dataset_loading()
        if hasattr(self, 'canvas'):
            self.canvas.annotations = []
            self.canvas.selected_annotation = None
            self.canvas.update()
        self.frame_annotations = {}
        self.deleted_frames = set()
        self.deleted_annotations = {}
        self.current_frame = 0
        self.video_path = None
        self.image_dataset_info = None
        self.is_image_dataset = False
        if hasattr(self, 'update_dataset_labels_action'):
            self.update_dataset_labels_action.setEnabled(False)
        self.frame_hashes = {}
        self.duplicate_frames_cache = {}
        if hasattr(self, 'performance_manager') and self.performance_manager:
            self.performance_manager.clear_cache()
        if hasattr(self, 'blur_manager') and self.blur_manager is not None:
            self.blur_manager.clear_all()
        if hasattr(self, 'annotation_dock'):
            self.annotation_dock.update_annotation_list()
        self.statusBar.showMessage('Ready')

    @log_exceptions
    def setup_playback_timer(self):
        """Set up the timer for video playback or image slideshow."""
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.next_frame)
        self.slideshow_speed = 1.0

    @log_exceptions
    def check_for_image_annotation_files(self, folder_path, folder_name):
        """
        Check if annotation files exist for this image dataset.

        Args:
            folder_path (str): Path to the image folder
            folder_name (str): Name of the image folder
        """
        autosave_file = os.path.join(folder_path,
            f'{folder_name}_autosave.json')
        if os.path.exists(autosave_file):
            reply = QMessageBox.question(self, 'Auto-Save Found',
                """An auto-save file was found for this image dataset.
Would you like to load it?"""
                , QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                try:
                    self._loading_from_project = True
                    self.load_project(autosave_file)
                    self._loading_from_project = False
                    return
                except Exception as e:
                    self._loading_from_project = False
                    QMessageBox.warning(self, 'Auto-Save Error',
                        f'Error loading auto-save file: {str(e)}')
        annotation_patterns = [f'{folder_name}_annotations.json',
            f'{folder_name}_annotations.txt', 'annotations.json']
        annotation_files = []
        for pattern in annotation_patterns:
            potential_file = os.path.join(folder_path, pattern)
            if os.path.exists(potential_file
                ) and potential_file != autosave_file:
                annotation_files.append(potential_file)
        classes_file = os.path.join(folder_path, 'classes.txt')
        if os.path.exists(classes_file):
            has_txt_annotations = False
            for image_path in self.image_files[:10]:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                txt_file = os.path.join(folder_path, f'{base_name}.txt')
                if os.path.exists(txt_file):
                    has_txt_annotations = True
                    break
            if has_txt_annotations:
                annotation_files.append(classes_file)
        for i, file_path in enumerate(annotation_files[:]):
            if file_path.endswith('.json'):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        if 'viat_project_identifier' in data:
                            annotation_files.remove(file_path)
                except:
                    pass
        if annotation_files:
            message = 'Found the following annotation file(s):\n\n'
            for file in annotation_files:
                message += f'- {os.path.basename(file)}\n'
            message += (
                '\nWould you like to import annotations from one of these files?'
                )
            reply = QMessageBox.question(self, 'Annotation Files Found',
                message, QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                if len(annotation_files) > 1:
                    self.show_annotation_file_selection_dialog(annotation_files
                        )
                else:
                    self.import_annotations(annotation_files[0])

    @log_exceptions
    def update_frame_info(self):
        """Update frame information in the UI."""
        sam_active = hasattr(self, 'sam_interactive_dock') and self.sam_interactive_dock.isVisible()
        
        if hasattr(self, 'is_image_dataset') and self.is_image_dataset:
            total = len(self.image_files) if self.image_files else 0
            
            if getattr(self, 'video_mode', False) and hasattr(self, 'video_groups') and hasattr(self, 'video_manager_dock'):
                current_vid = None
                for vid_name, indices in self.video_groups.items():
                    if self.current_frame in indices:
                        current_vid = vid_name
                        break
                if current_vid:
                    indices = self.video_groups[current_vid]
                    local_idx = indices.index(self.current_frame)
                    if sam_active:
                        self.sam_interactive_dock.update_frame_info(local_idx, len(indices))
                    self.frame_label.setText(f'{local_idx + 1}/{len(indices)}')
                    self.frame_slider.blockSignals(True)
                    self.frame_slider.setValue(local_idx)
                    self.frame_slider.blockSignals(False)
                    if 0 <= self.current_frame < len(self.image_files):
                        import os
                        self.statusBar.showMessage(
                            f'Image: {os.path.basename(self.image_files[self.current_frame])}'
                            )
                    return
            
            if sam_active:
                self.sam_interactive_dock.update_frame_info(self.current_frame, total)
            self.frame_label.setText(f'{self.current_frame + 1}/{total}')
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(self.current_frame)
            self.frame_slider.blockSignals(False)
            if 0 <= self.current_frame < len(self.image_files):
                import os
                self.statusBar.showMessage(
                    f'Image: {os.path.basename(self.image_files[self.current_frame])}'
                    )
        elif self.cap and self.cap.isOpened():
            if sam_active:
                self.sam_interactive_dock.update_frame_info(self.current_frame, self.total_frames)
            self.frame_label.setText(
                f'{self.current_frame}/{self.total_frames}')
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(self.current_frame)
            self.frame_slider.blockSignals(False)

    @log_exceptions
    def slider_changed(self, value):
        """Handle slider value changes (user drag only -- programmatic
        setValue blocks signals, so this only fires on genuine user
        interaction)."""
        if getattr(self, 'video_mode', False) and hasattr(self, 'video_groups') and hasattr(self, 'video_manager_dock'):
            current_item = self.video_manager_dock.list_widget.currentItem()
            if current_item:
                vid_name = current_item.text()
                if vid_name in self.video_groups:
                    indices = self.video_groups[vid_name]
                    if value >= 0 and value < len(indices):
                        global_frame = indices[value]
                        self.set_current_frame(global_frame)
                        return
                        
        if (hasattr(self, 'interpolation_manager') and self.
            interpolation_manager.is_active and value != self.current_frame):
            self.interpolation_manager.reset_cycle()
        if (hasattr(self, 'object_visibility_manager') and self.
            object_visibility_manager and self.object_visibility_manager.active
            ):
            visible_frames = (self.object_visibility_manager.
                get_visible_frame_numbers())
            if visible_frames and value not in visible_frames:
                self.frame_slider.blockSignals(True)
                self.frame_slider.setValue(self.current_frame)
                self.frame_slider.blockSignals(False)
                return
        if hasattr(self, 'is_image_dataset') and self.is_image_dataset:
            if 0 <= value < len(self.image_files):
                if value != self.current_frame:
                    self.set_current_frame(value)
        elif self.cap and self.cap.isOpened():
            frame_number = int(value)
            if frame_number != self.current_frame:
                self.set_current_frame(frame_number)

    @log_exceptions
    def _refresh_blur_display(self):
        """Refresh the canvas display to show updated blur regions."""
        if hasattr(self, 'is_image_dataset') and self.is_image_dataset:
            self.load_current_image()
        elif self.cap and self.cap.isOpened():
            # seek_to_frame re-reads the frame and sets it on canvas
            self.seek_to_frame(self.current_frame)

    @log_exceptions
    def seek_to_frame(self, frame_number):
        """Seek the video to an exact frame and refresh the display."""
        if not self.cap or not self.cap.isOpened():
            return False
        if frame_number < 0:
            frame_number = 0
        elif frame_number >= self.total_frames:
            frame_number = self.total_frames - 1
            
        if not self._should_show_frame(frame_number):
            nxt = frame_number + 1
            while nxt < self.total_frames and not self._should_show_frame(nxt):
                nxt += 1
            if nxt < self.total_frames:
                frame_number = nxt
            else:
                prv = frame_number - 1
                while prv >= 0 and not self._should_show_frame(prv):
                    prv -= 1
                if prv >= 0:
                    frame_number = prv
        if hasattr(self, 'performance_manager') and self.performance_manager:
            frame = self.performance_manager.seek_frame(frame_number)
            ret = frame is not None
        elif frame_number == self.current_frame + 1:
            ret, frame = self.cap.read()
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = self.cap.read()
        if not ret:
            return False
        frame = self._process_frame_metadata(frame, frame_number)
        self.current_frame = frame_number
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_frame)
        self.frame_slider.blockSignals(False)
        self.canvas.set_frame(frame)
        self.update_frame_info()
        self.load_current_frame_annotations()
        self.update_frame_display()
        return True

    @log_exceptions
    def go_to_frame_dialog(self):
        """Open a dialog allowing the user to jump directly to a specific frame number (Ctrl+G)."""
        from PyQt5.QtWidgets import QInputDialog
        
        total = 0
        if hasattr(self, 'is_image_dataset') and self.is_image_dataset and getattr(self, 'image_files', None):
            total = len(self.image_files)
        elif self.cap and self.cap.isOpened():
            total = self.total_frames
            
        if total <= 0:
            if hasattr(self, 'statusBar') and self.statusBar:
                self.statusBar.showMessage("No active video or image dataset loaded.")
            return

        current_display = self.current_frame + 1  # 1-indexed for display
        frame_num, ok = QInputDialog.getInt(
            self,
            "Go to Frame",
            f"Enter frame number (1 - {total}):",
            value=current_display,
            min=1,
            max=total,
            step=1
        )

        if ok:
            target_idx = frame_num - 1
            self.set_current_frame(target_idx)

    def _should_show_frame(self, frame_idx):
        if getattr(self, 'only_show_empty_frames', False):
            if self.frame_annotations.get(frame_idx, []):
                return False
                
        if getattr(self, 'nav_class_filter_active', False):
            mode = getattr(self, 'nav_class_filter_mode', '')
            target = getattr(self, 'nav_class_filter_target', '')
            count = getattr(self, 'nav_class_filter_count', 0)
            target_count = sum(1 for ann in self.frame_annotations.get(frame_idx, []) if ann.class_name == target)
            
            if mode == "More than" and not (target_count > count):
                return False
            if mode == "Less than" and not (target_count < count):
                return False
            if mode == "Exactly" and not (target_count == count):
                return False
            if mode == "Frames With Class" and not (target_count > 0):
                return False
            if mode == "Frames Without Class" and not (target_count == 0):
                return False
                
        return True

    @log_exceptions
    def prev_frame(self):
        """Go to the previous frame. ALWAYS steps back exactly one frame,
        regardless of interpolation mode (per user requirement)."""
        if self.is_playing:
            return
        self.handle_unverified_annotations()
        
        filter_active = getattr(self, 'only_show_empty_frames', False) or getattr(self, 'nav_class_filter_active', False)
        if filter_active and hasattr(self, 'image_files'):
            for i in range(self.current_frame - 1, -1, -1):
                if self._should_show_frame(i):
                    self.set_current_frame(i)
                    return
            self.statusBar.showMessage("No previous matching frames.")
            return
            
        if hasattr(self, 'is_image_dataset') and self.is_image_dataset:
            if self.current_frame > 0:
                old_frame = self.current_frame
                self.current_frame -= 1
                self._handle_auto_save_blur_on_frame_change(old_frame)
                self.frame_slider.blockSignals(True)
                self.frame_slider.setValue(self.current_frame)
                self.frame_slider.blockSignals(False)
                self.load_current_image()
                self.update_frame_info()
                self.load_current_frame_annotations()
                self.update_frame_display()
        else:
            prev = self.current_frame - 1
            min_idx = 0
            if getattr(self, 'video_mode', False) and hasattr(self, 'video_groups') and hasattr(self, 'video_manager_dock'):
                current_item = self.video_manager_dock.list_widget.currentItem()
                if current_item:
                    current_vid = current_item.text()
                    indices = self.video_groups.get(current_vid, [])
                    if indices:
                        min_idx = indices[0]
            
            while prev >= min_idx and not self._should_show_frame(prev):
                prev -= 1
                
            if prev >= min_idx:
                if self.seek_to_frame(prev):
                    self.update_frame_display()

    @log_exceptions
    def next_frame(self):
        """Go to the next frame, or drive the interpolation workflow when
        interpolation mode is active (and not during playback)."""
        self.handle_unverified_annotations()
        
        filter_active = getattr(self, 'only_show_empty_frames', False) or getattr(self, 'nav_class_filter_active', False)
        if filter_active and hasattr(self, 'image_files') and not self.is_playing:
            for i in range(self.current_frame + 1, len(self.image_files)):
                if self._should_show_frame(i):
                    self.set_current_frame(i)
                    return
            self.statusBar.showMessage("No more matching frames.")
            return

        if hasattr(self, 'is_image_dataset') and self.is_image_dataset:
            if self.current_frame < len(self.image_files) - 1:
                old_frame = self.current_frame
                self.current_frame += 1
                self._handle_auto_save_blur_on_frame_change(old_frame)
                self.frame_slider.blockSignals(True)
                self.frame_slider.setValue(self.current_frame)
                self.frame_slider.blockSignals(False)
                self.load_current_image()
                self.update_frame_info()
                self.load_current_frame_annotations()
                self.update_frame_display()
            elif self.is_playing:
                old_frame = self.current_frame
                self.current_frame = 0
                self._handle_auto_save_blur_on_frame_change(old_frame)
                self.frame_slider.blockSignals(True)
                self.frame_slider.setValue(self.current_frame)
                self.frame_slider.blockSignals(False)
                self.load_current_image()
                self.update_frame_info()
                self.load_current_frame_annotations()
                self.statusBar.showMessage(
                    'Looping back to start of image dataset')
                self.update_frame_display()
            else:
                self.statusBar.showMessage('End of image dataset')
            return
        next_frame_number = None
        if hasattr(self, 'interpolation_manager'
            ) and self.interpolation_manager.is_active and not self.is_playing:
            next_frame_number = self.interpolation_manager.get_next_frame(self
                .current_frame)
        elif self.cap and self.cap.isOpened():
            if getattr(self, 'video_mode', False) and hasattr(self, 'video_groups') and hasattr(self, 'video_manager_dock'):
                current_item = self.video_manager_dock.list_widget.currentItem()
                if current_item:
                    current_vid = current_item.text()
                    indices = self.video_groups.get(current_vid, [])
                    if indices:
                        if self.current_frame < indices[-1]:
                            next_frame_number = self.current_frame + 1
                        elif self.is_playing:
                            next_frame_number = indices[0]
                            self.statusBar.showMessage(f'Looping back to start of {current_vid}')
                        else:
                            self.statusBar.showMessage(f'End of {current_vid}')
                            return
                            
            if next_frame_number is None:
                next_frame_number = self.current_frame + 1
                while next_frame_number < self.total_frames and not self._should_show_frame(next_frame_number):
                    next_frame_number += 1
                    
                if next_frame_number >= self.total_frames:
                    self.play_timer.stop()
                    self.is_playing = False
                    self.statusBar.showMessage('End of video')
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    first_valid = 0
                    while first_valid < self.total_frames and not self._should_show_frame(first_valid):
                        first_valid += 1
                    if first_valid < self.total_frames:
                        self.seek_to_frame(first_valid)
                    else:
                        self.seek_to_frame(0)
                    return
        if next_frame_number is not None and self.seek_to_frame(
            next_frame_number):
            self.update_frame_display()
            if (self.duplicate_frames_enabled and self.current_frame in
                self.frame_hashes):
                current_hash = self.frame_hashes[self.current_frame]
                if current_hash in self.duplicate_frames_cache and len(self
                    .duplicate_frames_cache[current_hash]) > 1:
                    self.propagate_annotations_to_duplicate(current_hash)

    @log_exceptions
    def play_pause_video(self):
        """Toggle between playing and pausing the video or image slideshow."""
        if hasattr(self, 'is_image_dataset') and self.is_image_dataset:
            if self.is_playing:
                self.play_timer.stop()
                self.is_playing = False
                self.play_button.setIcon(self.icon_provider.get_icon(
                    'media-playback-start'))
                self.statusBar.showMessage('Slideshow paused')
            else:
                self.play_timer.start(1000)
                self.is_playing = True
                self.play_button.setIcon(self.icon_provider.get_icon(
                    'media-playback-pause'))
                self.statusBar.showMessage('Slideshow playing')
            return
        if not self.cap:
            return
        if self.is_playing:
            self.play_timer.stop()
            self.is_playing = False
            self.play_button.setIcon(self.icon_provider.get_icon(
                'media-playback-start'))
            self.statusBar.showMessage('Paused')
        else:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30
            interval = max(1, int(1000 / (fps * self.playback_speed)))
            self.play_timer.start(interval)
            self.is_playing = True
            self.play_button.setIcon(self.icon_provider.get_icon(
                'media-playback-pause'))
            self.statusBar.showMessage(f'Playing at {fps:.1f} FPS')
    def setup_video_manager(self):
        if hasattr(self, 'video_manager_dock'):
            self.video_manager_dock.video_mode_toggled.connect(self.toggle_video_mode)
            self.video_manager_dock.video_selected.connect(self.on_video_selected)
            self.video_manager_dock.prev_video_requested.connect(self.on_prev_video_requested)
            self.video_manager_dock.next_video_requested.connect(self.on_next_video_requested)
            self.video_manager_dock.sam_tracking_toggled.connect(self.toggle_sam_interactive_mode)
            self.video_manager_dock.remove_video_requested.connect(self.remove_current_cut)

    @log_exceptions
    def remove_current_cut(self):
        if not getattr(self, 'video_mode', False) or not hasattr(self, 'video_manager_dock'):
            return
            
        current_item = self.video_manager_dock.list_widget.currentItem()
        if not current_item:
            return
            
        current_vid = current_item.text()
        if not hasattr(self, 'video_groups') or current_vid not in self.video_groups:
            return
            
        indices = self.video_groups[current_vid]
        
        # Move all frames in this cut to removed/
        result = _viat_move_to_removed(self, indices)
        
        # The total frames have changed, so we need to update slider
        if self.current_frame >= self.total_frames:
            self.current_frame = max(0, self.total_frames - 1)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.frame_slider.setValue(self.current_frame)
        self.frame_slider.blockSignals(False)
        
        # We need to rebuild video groups because all frame indices have shifted
        self.update_video_groups()
        if hasattr(self, 'video_manager_dock'):
            self.video_manager_dock.set_videos(list(self.video_groups.keys()))
            
            # Select another cut
            row = self.video_manager_dock.list_widget.row(current_item)
            if self.video_manager_dock.list_widget.count() > 0:
                new_row = min(row, self.video_manager_dock.list_widget.count() - 1)
                self.video_manager_dock.list_widget.setCurrentRow(new_row)
        
        self.load_current_image()
        self.update_frame_info()
        self.update_annotation_list()
        
        self.statusBar.showMessage(
            f"Removed cut {current_vid} ({result['moved_images']} imgs moved to {result['dest_dir']})"
        )
        self.update_frame_display()

    def on_video_selected(self, video_name):
        if not video_name or video_name not in getattr(self, 'video_groups', {}):
            return
            
        indices = self.video_groups[video_name]
        start_idx = indices[0]
        end_idx = indices[-1]
        
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(max(0, len(indices) - 1))
        self.frame_slider.blockSignals(False)
        
        # Navigate to first frame of the video
        self.set_current_frame(start_idx)
        
    def on_prev_video_requested(self):
        if not getattr(self, 'video_mode', False) or not hasattr(self, 'video_manager_dock'):
            return
        current_item = self.video_manager_dock.list_widget.currentItem()
        if current_item:
            row = self.video_manager_dock.list_widget.row(current_item)
            if row > 0:
                self.video_manager_dock.list_widget.setCurrentRow(row - 1)

    def on_next_video_requested(self):
        if not getattr(self, 'video_mode', False) or not hasattr(self, 'video_manager_dock'):
            return
        current_item = self.video_manager_dock.list_widget.currentItem()
        if current_item:
            row = self.video_manager_dock.list_widget.row(current_item)
            if row < self.video_manager_dock.list_widget.count() - 1:
                self.video_manager_dock.list_widget.setCurrentRow(row + 1)

    def toggle_video_mode(self, checked):
        self.video_mode = checked
        if hasattr(self, 'video_manager_dock'):
            self.video_manager_dock.set_active(checked)
            
        if checked:
            self.update_video_groups()
            if hasattr(self, 'video_manager_dock'):
                self.video_manager_dock.set_videos(list(self.video_groups.keys()))
            
            # Select the video containing the current frame, if possible
            current_vid = None
            for vid_name, indices in self.video_groups.items():
                if self.current_frame in indices:
                    current_vid = vid_name
                    break
            
            if current_vid and hasattr(self, 'video_manager_dock'):
                self.video_manager_dock.select_video(current_vid)
            elif self.video_groups and hasattr(self, 'video_manager_dock'):
                self.video_manager_dock.list_widget.setCurrentRow(0)
        else:
            if hasattr(self, 'video_manager_dock'):
                self.video_manager_dock.set_videos([])
                
            self.frame_slider.blockSignals(True)
            self.frame_slider.setMinimum(0)
            self.frame_slider.setMaximum(max(0, getattr(self, 'total_frames', 1) - 1))
            self.frame_slider.setValue(self.current_frame)
            self.frame_slider.blockSignals(False)
            self.update_frame_info()
            
        if hasattr(self, 'refresh_empty_frames_dock'):
            self.refresh_empty_frames_dock()
        
    def open_zero_shot_refiner_dialog(self):
        from .widgets.zero_shot_classification_dialog import ZeroShotClassificationDialog
        from .utils.zero_shot_classifier import ZeroShotClassifierManager

        # Collect current dataset class names to pre-populate the dialog.
        # The canonical class registry in VIAT is canvas.class_colors (keys = class names).
        known_classes = []
        if hasattr(self, 'canvas') and hasattr(self.canvas, 'class_colors') and self.canvas.class_colors:
            known_classes = sorted(self.canvas.class_colors.keys())
        elif hasattr(self, 'frame_annotations'):
            seen = set()
            for anns in self.frame_annotations.values():
                for a in anns:
                    if hasattr(a, 'class_name') and a.class_name:
                        seen.add(a.class_name)
            known_classes = sorted(seen)

        dialog = ZeroShotClassificationDialog(self, known_classes=known_classes)
        if dialog.exec_() != QDialog.Accepted:
            if hasattr(self, 'refresh_class_frames_dock'):
                self.refresh_class_frames_dock()
            return

        config = dialog.get_config()

        # Resolve the absolute checkpoints directory the same way ZeroShotManager does,
        # so the cached model files land in the right place.
        import sys as _sys
        if getattr(_sys, 'frozen', False):
            _project_root = os.path.dirname(_sys.executable)
        else:
            _project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _chkp_dir = os.path.join(_project_root, "checkpoints")

        # ── Load model ────────────────────────────────────────────────
        manager = ZeroShotClassifierManager(checkpoints_dir=_chkp_dir)
        success, msg = manager.load_model(config['model'])
        if not success:
            QMessageBox.critical(self, "Model Error", msg)
            return

        # ── Build rules_config from UI values (JSON import is already merged) ──
        rules_config = ZeroShotClassifierManager.build_config(
            mode=config['mode'],
            rules=config['rules'],
            classes=config['classes'],
            overlap_groups=config.get('overlap_groups', []),
            global_fallback=config.get('global_fallback', []),
        )

        total_frames = len(self.frame_annotations)
        if total_frames == 0:
            QMessageBox.information(self, "Info", "No annotations to refine.")
            return

        progress = QProgressDialog(
            "Refining annotations using Zero-Shot Classification…",
            "Cancel", 0, total_frames, self
        )
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        frames_processed = 0
        uncertain_count = 0

        for frame_idx, annotations in self.frame_annotations.items():
            if progress.wasCanceled():
                break

            if not annotations:
                frames_processed += 1
                progress.setValue(frames_processed)
                continue

            if not hasattr(self, 'image_files') or frame_idx >= len(self.image_files):
                frames_processed += 1
                progress.setValue(frames_processed)
                continue

            image_array = cv2.imread(self.image_files[frame_idx])
            if image_array is not None:
                before = sum(1 for a in annotations if getattr(a, 'uncertain', False))
                manager.refine_annotations(
                    image_array,
                    annotations,
                    rules_config,
                    min_confidence=config['min_confidence'],
                    overlap_margin=config['overlap_margin'],
                )
                after = sum(1 for a in annotations if getattr(a, 'uncertain', False))
                uncertain_count += (after - before)

            self.set_current_frame(frame_idx)
            self.load_current_frame_annotations()
            self.update_frame_display()
            frames_processed += 1
            progress.setValue(frames_processed)
            QApplication.processEvents()

        progress.setValue(total_frames)

        self.project_modified = True
        self.load_current_frame_annotations()
        self.update_frame_display()
        if hasattr(self, 'refresh_uncertain_frames_dock'):
            self.refresh_uncertain_frames_dock()
        if hasattr(self, 'refresh_class_ui'):
            self.refresh_class_ui()

        mode_label = {
            "correctness": "Correctness Check",
            "mislabel": "Mislabel Check",
            "both": "Correctness + Mislabel Check",
        }.get(config['mode'], config['mode'])

        QMessageBox.information(
            self, "Complete",
            f"Zero-Shot Classification Refiner finished.\n\n"
            f"Mode: {mode_label}\n"
            f"New uncertain annotations flagged: {uncertain_count}\n\n"
            "Review flagged items in the Uncertain Frames panel."
        )

        if hasattr(self, 'refresh_class_frames_dock'):
            self.refresh_class_frames_dock()

    def update_video_groups(self):
        self.video_groups = {}
        self.single_images = []
        
        if not getattr(self, 'is_image_dataset', False):
            return
            
        if hasattr(self, 'custom_video_groups') and hasattr(self, 'custom_single_images'):
            self.video_groups = self.custom_video_groups.copy()
            self.single_images = self.custom_single_images.copy()
            return
            
        import re
        import os
        
        temp_groups = {}
        for idx, path in enumerate(self.image_files):
            filename = os.path.basename(path)
            
            # Clean up Roboflow format: remove .rf.HASH and _jpg/_png before extension
            clean_filename = re.sub(r'\.rf\.[a-f0-9]+\.', '.', filename)
            clean_filename = re.sub(r'_(jpg|jpeg|png)\.', '.', clean_filename)
            
            match = re.match(r'^([a-zA-Z_]+)[-_\s]*(\d+)\.[^.]+$', clean_filename)
            if match:
                prefix = match.group(1).strip()
            else:
                match = re.match(r'^(.*?)[-_\s]*(\d+)\.[^.]+$', clean_filename)
                if match:
                    prefix = match.group(1).strip()
                else:
                    prefix = ""
            if not prefix:
                prefix = "Sequence"
            
            if prefix not in temp_groups:
                temp_groups[prefix] = []
            temp_groups[prefix].append(idx)
            
        for prefix, indices in temp_groups.items():
            if len(indices) < 2:
                self.single_images.extend(indices)
            else:
                start_idx = indices[0]
                prev = start_idx
                current_chunk = [start_idx]
                chunk_num = 1
                for i in range(1, len(indices)):
                    curr = indices[i]
                    if curr == prev + 1:
                        current_chunk.append(curr)
                    else:
                        if len(current_chunk) < 2:
                            self.single_images.extend(current_chunk)
                        else:
                            name = f"{prefix}_{chunk_num}" if chunk_num > 1 else prefix
                            self.video_groups[name] = current_chunk
                            chunk_num += 1
                        current_chunk = [curr]
                    prev = curr
                
                if len(current_chunk) < 2:
                    self.single_images.extend(current_chunk)
                else:
                    name = f"{prefix}_{chunk_num}" if chunk_num > 1 else prefix
                    self.video_groups[name] = current_chunk
                    
        if self.single_images:
            self.video_groups["Single Images"] = self.single_images

    @log_exceptions
    def set_current_frame(self, frame):
        """Programmatically set the current frame"""
        if getattr(self, 'current_frame', -1) == frame:
            return
            
        old_frame = getattr(self, 'current_frame', -1)
        if old_frame >= 0 and old_frame != frame:
            self._handle_auto_save_blur_on_frame_change(old_frame)
            
        self.current_frame = frame
        
        self.frame_slider.blockSignals(True)
        if getattr(self, 'video_mode', False) and hasattr(self, 'video_groups') and hasattr(self, 'video_manager_dock'):
            vid_name = None
            for v, indices in self.video_groups.items():
                if frame in indices:
                    vid_name = v
                    break
            if vid_name:
                self.video_manager_dock.select_video(vid_name)
                indices = self.video_groups[vid_name]
                local_idx = indices.index(frame)
                
                self.frame_slider.setMinimum(0)
                self.frame_slider.setMaximum(max(0, len(indices) - 1))
                self.frame_slider.setValue(local_idx)
            else:
                self.frame_slider.setValue(frame)
        else:
            self.frame_slider.setValue(frame)
        self.frame_slider.blockSignals(False)
        
        if hasattr(self, 'is_image_dataset') and self.is_image_dataset:
            self.load_current_image()
            self.update_frame_info()
            self.load_current_frame_annotations()
            self.update_frame_display()
        elif self.cap and self.cap.isOpened():
            self.seek_to_frame(frame)
            self.update_frame_display()

    def _resolve_frame_indices(self, mixed_items):
        resolved = []
        if not mixed_items:
            return resolved
        if getattr(self, 'video_mode', False) and hasattr(self, 'video_groups'):
            for item in mixed_items:
                if isinstance(item, str):
                    resolved.extend(self.video_groups.get(item, []))
                else:
                    resolved.append(item)
        else:
            for item in mixed_items:
                resolved.append(int(item))
        return resolved

    @log_exceptions
    def set_slideshow_speed(self, speed_factor):
        """Set the speed of the image slideshow.

        Args:
            speed_factor (float): Speed multiplier (1.0 = 1 second per image)
        """
        if hasattr(self, 'is_image_dataset') and self.is_image_dataset:
            interval = int(1000 / speed_factor)
            if self.is_playing:
                self.play_timer.stop()
                self.play_timer.start(interval)
            self.statusBar.showMessage(f'Slideshow speed: {speed_factor}x')

    @log_exceptions
    def sync_annotation_selection(self, annotation):
        """
        Synchronize annotation selection between canvas and dock.

        Args:
            annotation: The annotation to select
        """
        if hasattr(self, 'canvas'):
            old_block_state = self.canvas.blockSignals(True)
            self.canvas.selected_annotation = annotation
            self.canvas.update()
            self.canvas.blockSignals(old_block_state)
        if hasattr(self, 'annotation_dock'):
            self.annotation_dock.select_annotation_in_list(annotation)

    @log_exceptions
    def update_frame_annotations(self):
        """Update annotations for the current frame."""
        if hasattr(self.canvas, 'annotations') and self.canvas.annotations:
            valid_annots = [a for a in self.canvas.annotations if not getattr(a, 'is_sam_preview', False)]
            if valid_annots:
                self.frame_annotations[self.current_frame] = valid_annots.copy()
            elif self.current_frame in self.frame_annotations:
                del self.frame_annotations[self.current_frame]
        if self.current_frame in self.frame_annotations:
            self.canvas.annotations = self.frame_annotations[self.current_frame]
        else:
            self.canvas.annotations = []
        self.update_annotation_list()
        self.canvas.update()

    @log_exceptions
    def load_current_frame_annotations(self):
        """Load annotations for the current frame into the canvas."""
        if hasattr(self.canvas, 'sam_preview_polygon'):
            self.canvas.sam_preview_polygon = None
            self.canvas.sam_preview_rect = None
            self.canvas.sam_preview_class = None
        mgr = getattr(self, 'object_visibility_manager', None)
        if mgr and getattr(mgr, 'auto_erase_mode', False) and not getattr(self,
            '_in_auto_erase', False):
            self._in_auto_erase = True
            mgr.delete_current_object_on_current_frame()
            if getattr(self, '_viat_update_visibility_labels', None):
                self._viat_update_visibility_labels()
            self._in_auto_erase = False
        self.canvas.selected_annotation = None
        if self.current_frame in self.frame_annotations:
            self.canvas.annotations = self.frame_annotations[self.current_frame]
        else:
            self.canvas.annotations = []
            
        # Load the latest crop box for this frame or earlier
        if hasattr(self, 'frame_crops') and self.frame_crops:
            closest_frame = -1
            for f in self.frame_crops:
                if f <= self.current_frame and f > closest_frame:
                    closest_frame = f
            if closest_frame >= 0:
                # Use a copy of the rect to prevent modifying the past frame's rect unintentionally
                from PyQt5.QtCore import QRect
                self.canvas.crop_rect = QRect(self.frame_crops[closest_frame])
                
                # Update dock spinboxes without emitting signals to avoid infinite loops
                if hasattr(self, 'crop_settings_dock'):
                    self.crop_settings_dock.width_spin.blockSignals(True)
                    self.crop_settings_dock.height_spin.blockSignals(True)
                    self.crop_settings_dock.width_spin.setValue(self.canvas.crop_rect.width())
                    self.crop_settings_dock.height_spin.setValue(self.canvas.crop_rect.height())
                    self.crop_settings_dock.width_spin.blockSignals(False)
                    self.crop_settings_dock.height_spin.blockSignals(False)
                    
        # Update crop box if tracking is enabled
        if hasattr(self, 'crop_settings_dock') and self.crop_settings_dock.track_object_cb.isChecked():
            if self.canvas.crop_rect is not None and self.canvas.selected_annotation is not None:
                # Get the tracked object's center
                tracked_ann = self.canvas.selected_annotation
                
                # Check if it's the same object (by ID) or just keep tracking the selected one
                # Usually selected_annotation points to the object in the current frame
                # if the tracking engine updated it.
                if tracked_ann in self.canvas.annotations:
                    center = tracked_ann.rect.center()
                    self.canvas.crop_rect.moveCenter(center)
                    # Keep crop box inside frame bounds
                    if self.canvas.pixmap:
                        if self.canvas.crop_rect.left() < 0: self.canvas.crop_rect.moveLeft(0)
                        if self.canvas.crop_rect.top() < 0: self.canvas.crop_rect.moveTop(0)
                        if self.canvas.crop_rect.right() > self.canvas.pixmap.width(): self.canvas.crop_rect.moveRight(self.canvas.pixmap.width())
                        if self.canvas.crop_rect.bottom() > self.canvas.pixmap.height(): self.canvas.crop_rect.moveBottom(self.canvas.pixmap.height())
                    # Since tracking changed the rect, save it to the current frame
                    from PyQt5.QtCore import QRect
                    self.frame_crops[self.current_frame] = QRect(self.canvas.crop_rect)
        if hasattr(self, 'annotation_dock'):
            self.annotation_dock.update_annotation_list()
        self.canvas.update()

    @log_exceptions
    def edit_annotation(self, annotation, focus_first_field=False):
        """
        Edit the properties of an annotation.

        Args:
            annotation: The annotation to edit
            focus_first_field: Whether to focus on the first attribute field
        """
        self.save_undo_state()
        self.annotation_manager.edit_annotation(annotation, focus_first_field)

    @log_exceptions
    def track_deleted_annotation(self, annotation):
        """Track a deleted loaded/imported/detected annotation to keep it in the project file."""
        if hasattr(self, 'deleted_annotations'):
            orig_src = getattr(annotation, 'original_source', getattr(annotation, 'source', 'manual'))
            if orig_src in ('loaded', 'imported', 'detected'):
                frame_num = self.current_frame
                if frame_num not in self.deleted_annotations:
                    self.deleted_annotations[frame_num] = []
                if annotation not in self.deleted_annotations[frame_num]:
                    self.deleted_annotations[frame_num].append(annotation)

    @log_exceptions
    def delete_annotation(self, annotation):
        """Delete the specified annotation."""
        if hasattr(self, 'canvas') and annotation in self.canvas.annotations:
            self.save_undo_state()
            self.track_deleted_annotation(annotation)
            if self.annotation_manager.delete_annotation(annotation):
                self.project_modified = True
                self.update_annotation_list()

    @log_exceptions
    def delete_selected_annotations(self):
        """Delete all currently selected annotations."""
        if not hasattr(self.canvas, 'selected_annotations'
            ) or not self.canvas.selected_annotations:
            self.delete_selected_annotation()
            return
        self.save_undo_state()
        annotations_to_delete = self.canvas.selected_annotations.copy()
        for annotation in annotations_to_delete:
            if annotation in self.canvas.annotations:
                self.track_deleted_annotation(annotation)
                self.canvas.annotations.remove(annotation)
        self.canvas.selected_annotation = None
        self.canvas.selected_annotations = []
        self.canvas.update()
        self.project_modified = True
        self.frame_annotations[self.current_frame
            ] = self.canvas.annotations.copy()
        self.update_annotation_list()
        count = len(annotations_to_delete)
        self.statusBar.showMessage(f'Deleted {count} annotations', 3000)

    @log_exceptions
    def delete_selected_annotation(self):
        """Delete the currently selected annotation."""
        if hasattr(self.canvas, 'selected_annotation'
            ) and self.canvas.selected_annotation:
            if self.canvas.selected_annotation in self.canvas.annotations:
                self.save_undo_state()
                self.track_deleted_annotation(self.canvas.selected_annotation)
                self.canvas.annotations.remove(self.canvas.selected_annotation)
                self.canvas.selected_annotation = None
                self.canvas.update()
                self.project_modified = True
                self.frame_annotations[self.current_frame
                    ] = self.canvas.annotations.copy()
                if hasattr(self, 'update_annotation_list'):
                    self.update_annotation_list()
            else:
                self.canvas.selected_annotation = None

    @log_exceptions
    def add_empty_annotation(self):
        """Add a new empty annotation with default values."""
        self.save_undo_state()
        self.annotation_manager.add_empty_annotation()

    @log_exceptions
    def update_annotation_list(self):
        """Update the annotation list in the UI and handle interpolation."""
        if hasattr(self, 'annotation_dock'):
            self.annotation_dock.update_annotation_list()
        self.frame_annotations[self.current_frame
            ] = self.canvas.annotations.copy()
        if (self.duplicate_frames_enabled and self.current_frame in self.
            frame_hashes):
            current_hash = self.frame_hashes[self.current_frame]
            if current_hash in self.duplicate_frames_cache and len(self.
                duplicate_frames_cache[current_hash]) > 1:
                self.propagate_to_duplicate_frames(current_hash)
        self.perform_autosave()

    @log_exceptions
    def update_annotation_attributes(self, annotation, class_attributes):
        """
        Update annotation attributes based on class configuration.

        Args:
            annotation: The annotation to update
            class_attributes: The class attribute configuration
        """
        self.annotation_manager.update_annotation_attributes(annotation,
            class_attributes)

    @log_exceptions
    def clear_annotations(self):
        """Clear all annotations."""
        self.save_undo_state()
        self.annotation_manager.clear_annotations()

    @log_exceptions
    def add_annotation(self):
        """Add annotation manually."""
        self.save_undo_state()
        self.annotation_manager.add_annotation()

    @log_exceptions
    def create_annotation_dialog(self):
        """Create a dialog for adding or editing annotations."""
        return self.annotation_manager.create_annotation_dialog()

    @log_exceptions
    def parse_attributes(self, text):
        """Parse attributes from text input."""
        return self.annotation_manager.parse_attributes(text)

    @log_exceptions
    def select_all_annotations(self):
        """Select all annotations in the current frame."""
        current_frame = self.current_frame
        if current_frame in self.frame_annotations and self.frame_annotations[
            current_frame]:
            self.canvas.selected_annotations = self.frame_annotations[
                current_frame].copy()
            if (not self.canvas.selected_annotation and self.canvas.
                selected_annotations):
                self.canvas.selected_annotation = (self.canvas.
                    selected_annotations[0])
            self.canvas.update()
            if hasattr(self, 'annotation_dock'):
                self.annotation_dock.select_annotation_in_list(None)
                self.annotation_dock.select_all_in_list()
            count = len(self.frame_annotations[current_frame])
            self.statusBar.showMessage(
                f'Selected all {count} annotations in this frame', 3000)
        else:
            self.statusBar.showMessage('No annotations in this frame', 2000)

    @log_exceptions
    def cycle_annotation_selection(self):
        """Cycle through annotations in the current frame or deselect if only one exists."""
        current_frame = self.current_frame
        if (current_frame not in self.frame_annotations or not self.
            frame_annotations[current_frame]):
            self.statusBar.showMessage('No annotations in this frame', 2000)
            return
        annotations = self.frame_annotations[current_frame]
        if not self.canvas.selected_annotation:
            self.canvas.selected_annotation = annotations[0]
            self.canvas.update()
            self.statusBar.showMessage(
                f'Selected annotation: {self.canvas.selected_annotation.class_name}'
                , 2000)
            return
        try:
            current_index = annotations.index(self.canvas.selected_annotation)
            if current_index == len(annotations) - 1:
                self.canvas.selected_annotation = None
                self.canvas.update()
                self.statusBar.showMessage('Annotation deselected', 2000)
            else:
                next_index = current_index + 1
                self.canvas.selected_annotation = annotations[next_index]
                self.canvas.update()
                self.statusBar.showMessage(
                    f'Selected annotation: {self.canvas.selected_annotation.class_name}'
                    , 2000)
        except ValueError:
            self.canvas.selected_annotation = annotations[0]
            self.canvas.update()
            self.statusBar.showMessage(
                f'Selected annotation: {self.canvas.selected_annotation.class_name}'
                , 2000)

    @log_exceptions
    def get_previous_annotation_attributes(self, class_name):
        """
        Find the most recent annotation of the same class and return its attributes.

        Args:
            class_name (str): The class name to match

        Returns:
            dict: Attributes dictionary or None if no previous annotation found
        """
        if not self.use_previous_attributes:
            return None
        for annotation in reversed(self.canvas.annotations):
            if annotation.class_name == class_name:
                return annotation.attributes.copy()
        for frame_num in sorted(self.frame_annotations.keys(), reverse=True):
            if frame_num >= self.current_frame:
                continue
            for annotation in reversed(self.frame_annotations[frame_num]):
                if annotation.class_name == class_name:
                    return annotation.attributes.copy()
        return None

    @log_exceptions
    def add_class(self, class_name=None, color=None):
        self.save_undo_state()
        self.class_manager.add_class(class_name, color)

    @log_exceptions
    def import_classes_from_yolo_yaml(self):
        """Import annotation classes from a YOLO dataset YAML file."""
        filename, _ = QFileDialog.getOpenFileName(self,
            'Import Classes from YOLO YAML', '',
            'YAML Files (*.yaml *.yml);;All Files (*)')
        if filename:
            self.save_undo_state()
            self.class_manager.import_classes_from_yolo_yaml(filename)

    @log_exceptions
    def refresh_class_lists(self):
        """Refresh class lists in all docks with debouncing"""
        if self._class_refresh_scheduled:
            return
        self._class_refresh_scheduled = True

        def do_refresh():
            if hasattr(self, 'class_dock'):
                self.class_dock.update_class_list()
            if hasattr(self, 'class_frames_dock'):
                self.class_frames_dock.update_classes(list(self.canvas.class_colors.keys()))
            if hasattr(self, 'annotation_dock'):
                self.annotation_dock.update_class_selector()
            if hasattr(self, 'toolbar') and hasattr(self.toolbar,
                'update_class_selector'):
                self.toolbar.update_class_selector()
            self._class_refresh_scheduled = False
        QTimer.singleShot(100, do_refresh)

    @log_exceptions
    def edit_selected_class(self):
        """Edit the selected class with option to convert to another class."""
        self.save_undo_state('all')
        self.class_manager.edit_selected_class()

    @log_exceptions
    def convert_class_with_attributes(self, old_class, new_class,
        keep_original=False):
        """
        Convert all annotations from one class to another with attribute handling.

        Args:
            old_class (str): The original class name
            new_class (str): The target class name
            keep_original (bool): Whether to keep original attributes or use target class defaults
        """
        self.save_undo_state('all')
        self.class_manager.convert_class_with_attributes(old_class,
            new_class, keep_original)

    @log_exceptions
    def convert_class_with_attribute_mapping(self, old_class, new_class):
        """
        Convert class with custom attribute mapping.

        Args:
            old_class (str): The original class name
            new_class (str): The target class name
        """
        self.save_undo_state('all')
        self.class_manager.convert_class_with_attribute_mapping(old_class,
            new_class)

    @log_exceptions
    def refresh_class_ui(self):
        """Refresh all UI components that display class information"""
        if hasattr(self, 'class_dock'):
            self.class_dock.update_class_list()
        if hasattr(self, 'class_frames_dock'):
            self.class_frames_dock.update_classes(list(self.canvas.class_colors.keys()))
        if hasattr(self, 'annotation_dock'):
            self.annotation_dock.update_class_selector()
        if hasattr(self, 'class_selector'):
            self.class_selector.blockSignals(True)
            current_text = self.class_selector.currentText()
            self.class_selector.clear()
            self.class_selector.addItems(sorted(self.canvas.class_colors.
                keys()))
            if current_text in self.canvas.class_colors:
                self.class_selector.setCurrentText(current_text)
            elif self.canvas.current_class in self.canvas.class_colors:
                self.class_selector.setCurrentText(self.canvas.current_class)
            self.class_selector.blockSignals(False)
        self.canvas.update()

    @log_exceptions
    def set_active_class(self, class_name):
        """Set the active class and update all UI controls (canvas, toolbar, dock)."""
        if not class_name or not hasattr(self, 'canvas'
            ) or class_name not in self.canvas.class_colors:
            return
        self.canvas.set_current_class(class_name)
        if hasattr(self, 'class_selector'):
            self.class_selector.blockSignals(True)
            self.class_selector.setCurrentText(class_name)
            self.class_selector.blockSignals(False)
        if hasattr(self, 'class_dock'):
            self.class_dock.blockSignals(True)
            self.class_dock.classes_list.blockSignals(True)
            items = self.class_dock.classes_list.findItems(class_name, Qt.
                MatchExactly)
            if items:
                self.class_dock.classes_list.setCurrentItem(items[0])
            self.class_dock.classes_list.blockSignals(False)
            self.class_dock.blockSignals(False)
            self.class_dock.update_attribute_info(class_name)
        self.canvas.update()

    @log_exceptions
    def get_default_attributes_for_class(self, class_name):
        """
        Get the default/last used attributes for a given class.
        Checks the last_used_attributes cache first, then falls back to previous annotations,
        and finally class default configurations.
        """
        if not hasattr(self, 'last_used_attributes'):
            self.last_used_attributes = {}
        if class_name in self.last_used_attributes:
            return self.last_used_attributes[class_name].copy()
        if self.use_previous_attributes:
            prev = self.get_previous_annotation_attributes(class_name)
            if prev:
                return prev.copy()
        if hasattr(self, 'canvas') and hasattr(self.canvas, 'class_attributes'
            ):
            class_attrs = self.canvas.class_attributes.get(class_name, {})
            if class_attrs:
                defaults = {}
                for attr_name, attr_config in class_attrs.items():
                    defaults[attr_name] = attr_config.get('default', -1)
                return defaults
        return {'Size': -1, 'Quality': -1}

    @log_exceptions
    def convert_class(self, old_class, new_class):
        """Convert all annotations from one class to another."""
        self.save_undo_state('all')
        self.class_manager.convert_class(old_class, new_class)

    @log_exceptions
    def update_class(self, old_name, new_name, color):
        """Update a class with new name and color."""
        self.save_undo_state('all')
        self.class_manager.update_class(old_name, new_name, color)

    @log_exceptions
    def delete_selected_class(self):
        """Delete the selected class."""
        self.save_undo_state('all')
        self.class_manager.delete_selected_class()

    @log_exceptions
    def blur_and_delete_selected_class(self):
        """Blur all bounding boxes of the selected class and delete the class."""
        self.class_manager.blur_and_delete_selected_class()

    @log_exceptions
    def save_project(self, filename=False):
        """Save the current project."""
        if not filename and self.project_file:
            filename = self.project_file
        elif isinstance(filename, str) and filename:
            pass
        else:
            default_name = ""
            if self.project_file:
                default_name = self.project_file
            elif hasattr(self, 'video_filename') and self.video_filename:
                base = os.path.splitext(os.path.basename(self.video_filename))[0]
                default_name = os.path.join(os.path.dirname(self.video_filename), f"{base}.json")
            elif hasattr(self, 'image_files') and self.image_files and len(self.image_files) > 0:
                base_folder = os.path.dirname(self.image_files[0])
                folder_name = os.path.basename(base_folder) or "project"
                default_name = os.path.join(base_folder, f"{folder_name}.json")
            else:
                default_name = "project.json"

            filename, _ = QFileDialog.getSaveFileName(self, 'Save Project',
                default_name, 'JSON Files (*.json);;All Files (*)')
        if filename:
            if isinstance(filename, str) and not filename.lower().endswith('.json'):
                filename += '.json'
            video_path = None
            image_dataset_info = None
            if hasattr(self, 'is_image_dataset') and self.is_image_dataset:
                if self.image_files:
                    base_folder = os.path.dirname(self.image_files[0])
                    image_dataset_info = {'is_image_dataset': True,
                        'base_folder': base_folder, 'image_files': self.
                        get_image_files_relative()}
            else:
                video_path = getattr(self, 'video_filename', None)
            class_attributes = getattr(self.canvas, 'class_attributes', {})
            # backup_before_save(filename)
            from viat.utils.task_runner import run_task_with_progress
            from viat.utils.file_operations import save_project_generator
            frame_annotations_copy = {k: list(v) for k, v in self.
                frame_annotations.items()}
            deleted_annotations_copy = {k: list(v) for k, v in self.deleted_annotations.items()} if hasattr(self, 'deleted_annotations') else None
            class_colors_copy = dict(self.canvas.class_colors)
            annotations_copy = list(self.canvas.annotations)
            blur_regions = self.blur_manager.to_dict() if hasattr(self, 'blur_manager') and self.blur_manager else None
            result = run_task_with_progress(self, 'Saving Project',
                'Saving project to disk...', save_project_generator,
                filename, annotations_copy, class_colors_copy, video_path=
                video_path, current_frame=self.current_frame,
                frame_annotations=frame_annotations_copy, class_attributes=
                class_attributes, current_style=self.current_style,
                auto_show_attribute_dialog=self.auto_show_attribute_dialog,
                use_previous_attributes=self.use_previous_attributes,
                duplicate_frames_enabled=self.duplicate_frames_enabled,
                frame_hashes=dict(self.frame_hashes) if hasattr(self,
                'frame_hashes') else None, duplicate_frames_cache=dict(self
                .duplicate_frames_cache) if hasattr(self,
                'duplicate_frames_cache') else None, image_dataset_info=
                image_dataset_info, tracking_mode_enabled=self.
                tracking_mode_enabled, interpolation_mode_active=self.
                interpolation_manager.is_active, verification_mode_enabled=
                self.verification_mode, annotations_imported_list=list(self
                ._annotations_imported) if hasattr(self,
                '_annotations_imported') else [], class_thresholds=dict(
                self.class_thresholds) if getattr(self, 'class_thresholds',
                None) is not None else {}, deleted_frames=self.deleted_frames if hasattr(self, 'deleted_frames') else set(), labeler_analytics=self.labeler_analytics if hasattr(self, 'labeler_analytics') else None, deleted_annotations=deleted_annotations_copy, blur_regions=blur_regions, maximum=100)
            self.project_file = filename
            self.project_modified = False
            self.statusBar.showMessage(
                f'Project saved to {os.path.basename(filename)}')
            self.update_recent_projects_menu()
            self.save_application_state()
            video_path = getattr(self, 'video_filename', None)

    @log_exceptions
    def delete_history(self):
        """Delete all application history and reset to initial state."""
        reply = QMessageBox.question(self, 'Delete History',
            """This will delete all recent projects, saved settings, and application history.

The application will return to its initial state after installation.

Are you sure you want to continue?"""
            , QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        try:
            config_dir = get_config_directory()
            files_to_delete = ['recent_projects.json', 'last_state.json',
                'settings.json']
            for filename in files_to_delete:
                file_path = os.path.join(config_dir, filename)
                if os.path.exists(file_path):
                    os.remove(file_path)
            self.update_recent_projects_menu()
            self.reset_application_state()
            if hasattr(self, '_annotations_imported'):
                self._annotations_imported = set()
            QMessageBox.information(self, 'History Deleted',
                """Application history has been deleted successfully.

The application has been reset to its initial state."""
                )
        except Exception as e:
            QMessageBox.critical(self, 'Error',
                f'An error occurred while deleting history: {str(e)}')

    @log_exceptions
    def load_project(self, filename=None):
        """Load a project from a file."""
        if not self.check_unsaved_changes():
            return
        if not filename:
            filename, _ = QFileDialog.getOpenFileName(self, 'Load Project',
                '', 'JSON Files (*.json);;All Files (*)')
            if not filename:
                return False
        self._loading_from_project = True
        try:
            project_data = load_project_with_backup(filename)
            if not project_data:
                QMessageBox.critical(self, 'Error Loading Project',
                    'Failed to load project file and no valid backups found.')
                return False
            (annotations, class_colors, video_path, current_frame,
                frame_annotations, class_attributes, current_style,
                auto_show_attribute_dialog, use_previous_attributes,
                duplicate_frames_enabled, frame_hashes,
                duplicate_frames_cache, image_dataset_info,
                tracking_mode_enabled, interpolation_mode_active,
                verification_mode_enabled, annotations_imported_list,
                class_thresholds, deleted_annotations, blur_regions) = load_project(filename, BoundingBox)
            self.canvas.annotations = annotations
            self.class_colors = class_colors
            self.canvas.class_colors = class_colors
            self.current_frame = current_frame
            self.frame_annotations = frame_annotations
            self.deleted_annotations = deleted_annotations
            self.class_attributes = class_attributes
            self.canvas.class_attributes = self.class_attributes
            self.canvas.class_thresholds = self.class_thresholds
            self.current_style = current_style
            self.auto_show_attribute_dialog = auto_show_attribute_dialog
            self.use_previous_attributes = use_previous_attributes
            self.duplicate_frames_enabled = duplicate_frames_enabled
            self.class_thresholds = class_thresholds
            self.canvas.class_thresholds = class_thresholds
            if annotations_imported_list:
                self._annotations_imported = set(annotations_imported_list)
            else:
                self._annotations_imported = set()
            if frame_hashes:
                self.frame_hashes = frame_hashes
            if duplicate_frames_cache:
                self.duplicate_frames_cache = duplicate_frames_cache
            if image_dataset_info:
                self.image_dataset_info = image_dataset_info
                
            self.deleted_frames = set(project_data.get('deleted_frames', []))
            
            # Initialize or restore analytics
            loaded_analytics = project_data.get('labeler_analytics', {})
            self.labeler_analytics = {
                "prompts": loaded_analytics.get("prompts", []),
                "tool_usage": loaded_analytics.get("tool_usage", {
                    "zero_shot": 0, "tracking": 0, "interpolation": 0, "magic_wand": 0
                }),
                "base_annotations": loaded_analytics.get("base_annotations", {})
            }
            
            # Load media (video or image dataset)
            if video_path:
                if not os.path.exists(video_path):
                    project_dir = os.path.dirname(filename)
                    rel_path1 = os.path.join(project_dir, os.path.basename(video_path))
                    rel_path2 = os.path.join(project_dir, video_path)
                    if os.path.exists(rel_path1):
                        video_path = rel_path1
                    elif os.path.exists(rel_path2):
                        video_path = rel_path2
                    else:
                        reply = QMessageBox.question(
                            self, 'Video File Not Found',
                            f"The video file '{video_path}' was not found.\nWould you like to locate it?",
                            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
                        )
                        if reply == QMessageBox.Yes:
                            new_video_path, _ = QFileDialog.getOpenFileName(
                                self, 'Locate Video File', project_dir,
                                'Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;All Files (*)'
                            )
                            if new_video_path and os.path.exists(new_video_path):
                                video_path = new_video_path
                if video_path and os.path.exists(video_path):
                    self.load_video_from_project(video_path, current_frame)
            elif image_dataset_info:
                self.load_image_dataset_from_project(image_dataset_info, current_frame)

            # Restore blur regions into BlurManager
            if hasattr(self, 'blur_manager') and self.blur_manager is not None:
                self.blur_manager.from_dict(blur_regions or {})

            self.toggle_tracking_mode(tracking_mode_enabled)
            if hasattr(self.interpolation_manager, 'set_active'):
                self.interpolation_manager.set_active(interpolation_mode_active)
            self.verification_mode = verification_mode_enabled

            if hasattr(self, '_refresh_blur_display'):
                self._refresh_blur_display()
            else:
                self.update_frame_display()

            self.update_annotation_list()
            self.project_path = filename
            self.project_file = filename
            self.setWindowTitle(
                f'Video Annotation Tool - {os.path.basename(filename)}')
            self.project_modified = False
            self.statusBar.showMessage(f'Project loaded from {filename}', 5000)
            return True
        except Exception as e:
            QMessageBox.critical(self, 'Error Loading Project',
                f'Could not load project:\n{str(e)}')
            return False
        finally:
            self._loading_from_project = False

    @log_exceptions
    def save_application_state(self):
        """Save the current application state."""
        if not hasattr(self, 'project_file') or not self.project_file:
            return
        state = {'project_path': self.project_path, 'video_filename': self.
            video_filename, 'current_frame': self.current_frame,
            'zoom_level': self.canvas.zoom_level if hasattr(self.canvas,
            'zoom_level') else 1.0, 'tracking_mode_enabled': self.
            tracking_mode_enabled, 'interpolation_mode_active': self.
            interpolation_manager.is_active if hasattr(self.
            interpolation_manager, 'is_active') else False,
            'verification_mode_enabled': self.verification_mode if hasattr(
            self, 'verification_mode') else False,
            'show_attributes': getattr(self.canvas, 'show_attributes', False),
            'show_segmentation': getattr(self.canvas, 'show_segmentation', False)}
        save_last_state(state)

    @log_exceptions
    def load_application_state(self):
        """Load the last application state."""
        state = load_last_state()
        if not state:
            return False
            
        if 'show_attributes' in state:
            self.canvas.show_attributes = state.get('show_attributes', False)
        if 'show_segmentation' in state:
            self.canvas.show_segmentation = state.get('show_segmentation', False)
            
        last_project = state.get('last_project')
        if not last_project:
            last_project = state.get('project_path')
            
        if last_project and os.path.exists(last_project):
            self.has_autosave = True
            self.load_project(last_project)
            return True
        return False

    @log_exceptions
    def update_recent_projects_menu(self):
        """Update the recent projects menu with the latest projects."""
        self.recent_projects_menu.clear()
        recent_projects = get_recent_projects()
        if not recent_projects:
            no_recent = QAction('No Recent Projects', self)
            no_recent.setEnabled(False)
            self.recent_projects_menu.addAction(no_recent)
            return
        for project_path in recent_projects:
            project_name = os.path.basename(project_path)
            action = QAction(project_name, self)
            action.setData(project_path)
            action.triggered.connect(lambda checked, path=project_path:
                self.load_project(path))
            self.recent_projects_menu.addAction(action)
        self.recent_projects_menu.addSeparator()
        clear_action = QAction('Clear Recent Projects', self)
        clear_action.triggered.connect(self.clear_recent_projects)
        self.recent_projects_menu.addAction(clear_action)

    @log_exceptions
    def clear_recent_projects(self):
        """Clear the list of recent projects."""
        config_dir = get_config_directory()
        recent_projects_file = os.path.join(config_dir, 'recent_projects.json')
        with open(recent_projects_file, 'w') as f:
            json.dump([], f)
        self.update_recent_projects_menu()
        self.statusBar.showMessage('Recent projects cleared', 3000)

    @log_exceptions
    def reset_application_state(self):
        """Reset the application to its initial state."""
        self.cancel_dataset_loading()
        self.project_file = None
        self.project_modified = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.video_filename = ''
        self.current_frame = 0
        self.total_frames = 0
        self.is_playing = False
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(0)
        self.frame_slider.setMaximum(100)
        self.frame_slider.blockSignals(False)
        self.frame_label.setText('0/0')
        self.canvas.annotations = []
        self.frame_annotations = {}
        self.canvas.class_colors = {'Quad': QColor(0, 255, 255)}
        if hasattr(self.canvas, 'class_attributes'):
            self.canvas.class_attributes = {}
        self.canvas.current_class = 'Quad'
        self.duplicate_frames_enabled = False
        self.frame_hashes = {}
        self.duplicate_frames_cache = {}
        if hasattr(self, 'duplicate_frames_action'):
            self.duplicate_frames_action.setChecked(False)
        self.canvas.pixmap = None
        self.canvas.update()
        self.update_annotation_list()
        self.refresh_class_ui()
        self.change_style('DarkModern')
        self.auto_show_attribute_dialog = True
        self.use_previous_attributes = True
        self.autosave_enabled = True
        self.autosave_interval = 180000
        self.update_settings_menu_actions()
        self.statusBar.showMessage('Application reset to initial state')

    @log_exceptions
    def clear_recent_projects(self):
        """Clear the list of recent projects."""
        config_dir = get_config_directory()
        recent_projects_file = os.path.join(config_dir, 'recent_projects.json')
        with open(recent_projects_file, 'w') as f:
            json.dump([], f)
        self.update_recent_projects_menu()
        self.statusBar.showMessage('Recent projects cleared', 3000)

    @log_exceptions
    def update_original_dataset_labels(self):
        """Update the original dataset label files with current annotations."""
        if not getattr(self, 'is_image_dataset', False) or not hasattr(self,
            '_viat_dataset_info'):
            QMessageBox.warning(self, 'Update Labels',
                'No structured image dataset is currently loaded.')
            return
        reply = QMessageBox.question(self, 'Update Original Labels',
            """This will overwrite the original label files (e.g. YOLO .txt files) with the current annotations.

Are you sure you want to proceed?"""
            , QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        from viat.utils.dataset_manager import update_dataset_labels
        self.statusBar.showMessage('Updating dataset labels...')
        QApplication.processEvents()
        updated, errors = update_dataset_labels(self._viat_dataset_info,
            self.frame_annotations, self.image_files, current_classes=list(
            self.canvas.class_colors.keys()))
        if errors:
            QMessageBox.warning(self, 'Update Labels',
                f"""Updated {updated} labels, but encountered {len(errors)} errors.
First error: {errors[0]}"""
                )
        else:
            QMessageBox.information(self, 'Update Labels',
                f'Successfully updated {updated} label files.')
        self.statusBar.showMessage(f'Updated {updated} dataset labels.')

    @log_exceptions
    def export_clip_cuts(self, auto_export_dir=None):
        """Export defined clip cuts to separate videos and annotation files."""
        if not hasattr(self, 'clip_cuts_dock'):
            return
            
        cuts = self.clip_cuts_dock.get_cuts()
        if not cuts:
            if not auto_export_dir:
                QMessageBox.warning(self, "Export Clip Cuts", "No cuts defined to export.")
            return
            
        if not self.video_filename or not self.cap:
            if not auto_export_dir:
                QMessageBox.warning(self, "Export Clip Cuts", "Please load a video first.")
            return

        total_frames = self.total_frames
        
        # Validate cuts
        for cut in cuts:
            if cut['start'] < 0 or cut['end'] >= total_frames or cut['start'] >= cut['end']:
                QMessageBox.warning(self, "Invalid Cut", f"Cut '{cut['name']}' has invalid start/end frames.")
                return

        default_dir = os.path.dirname(self.video_filename)
        base_filename = os.path.splitext(os.path.basename(self.video_filename))[0]
        
        if auto_export_dir:
            export_dir = auto_export_dir
        else:
            export_dir = QFileDialog.getExistingDirectory(self, "Select Directory for Clip Cuts", default_dir, QFileDialog.ShowDirsOnly)
            if not export_dir:
                return

        progress = QProgressDialog("Exporting Clip Cuts...", "Cancel", 0, len(cuts), self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        try:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0: fps = 30
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            for i, cut in enumerate(cuts):
                if progress.wasCanceled():
                    break
                    
                progress.setLabelText(f"Exporting cut: {cut['name']}")
                
                cut_name = cut['name']
                start_f = cut['start']
                end_f = cut['end']
                
                out_vid_path = os.path.join(export_dir, f"{base_filename}_{cut_name}.mp4")
                out_ann_path = os.path.join(export_dir, f"{base_filename}_{cut_name}.txt")
                
                # Setup VideoWriter
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out_writer = cv2.VideoWriter(out_vid_path, fourcc, fps, (width, height))
                
                # Temporarily open a new cap to avoid messing up main player state
                cap = cv2.VideoCapture(self.video_filename)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
                
                frames_to_read = end_f - start_f + 1
                for _ in range(frames_to_read):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    out_writer.write(frame)
                    
                cap.release()
                out_writer.release()
                
                # Process annotations
                cut_annotations = {}
                for f_idx in range(start_f, end_f + 1):
                    if f_idx in self.frame_annotations and self.frame_annotations[f_idx]:
                        # create copies and adjust frame idx
                        import copy
                        new_anns = []
                        for ann in self.frame_annotations[f_idx]:
                            new_ann = copy.copy(ann)
                            new_ann.frame = f_idx - start_f
                            new_anns.append(new_ann)
                        cut_annotations[f_idx - start_f] = new_anns
                        
                from viat.utils.file_operations import export_raya_with_classes_annotations
                all_cut_annotations = []
                for f_num, anns in cut_annotations.items():
                    all_cut_annotations.extend(anns)
                classes = list(self.canvas.class_colors.keys())
                export_raya_with_classes_annotations(
                    out_ann_path,
                    all_cut_annotations,
                    classes,
                    deleted_frames=set(),
                    total_frames=frames_to_read
                )
                
                progress.setValue(i + 1)
                QApplication.processEvents()
                
            if not auto_export_dir:
                QMessageBox.information(self, "Export Complete", "Successfully exported all clip cuts.")
        except Exception as e:
            logger.error(f"Error exporting clip cuts: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Export Error", f"An error occurred: {str(e)}")
        finally:
            progress.close()

    @log_exceptions
    def export_annotations(self):
        """Export annotations to various formats."""
        has_annotations = bool(self.canvas.annotations) or any(self.
            frame_annotations.values())
        if not has_annotations:
            QMessageBox.warning(self, 'Export Annotations',
                'No annotations to export!')
            return
        dialog = self.create_export_dialog()
        if dialog.exec_() == QDialog.Accepted:
            format_combo = dialog.findChild(QComboBox)
            format_type = format_combo.currentText()
            if format_type == 'Raya Video':
                self.export_image_dataset()
                return
            self.export_annotations_with_format(format_type)

    @log_exceptions
    def create_export_dialog(self):
        """Create a dialog for export options."""
        dialog = QDialog(self)
        dialog.setWindowTitle('Export Annotations')
        dialog.setMinimumWidth(300)
        layout = QVBoxLayout(dialog)
        format_label = QLabel('Export Format:')
        format_combo = QComboBox()
        if hasattr(self, 'is_image_dataset') and self.is_image_dataset:
            format_combo.addItems(['COCO JSON', 'YOLO TXT',
                'Pascal VOC XML', 'Raya TXT', 'Raya with classes TXT',
                'Raya Video'])
        else:
            format_combo.addItems(['Raya TXT', 'Raya with classes TXT',
                'COCO JSON', 'YOLO TXT', 'Pascal VOC XML'])
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.
            Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(format_label)
        layout.addWidget(format_combo)
        layout.addWidget(buttons)
        return dialog

    @log_exceptions
    def export_annotations_with_format(self, format_type):
        """Export annotations with the specified format."""
        default_dir = ''
        default_filename = ''
        if hasattr(self, 'is_image_dataset'
            ) and self.is_image_dataset and self.image_files:
            image_folder = os.path.dirname(self.image_files[0])
            folder_name = os.path.basename(image_folder)
            default_dir = image_folder
            default_filename = folder_name + '_annotations'
        elif hasattr(self, 'video_filename') and self.video_filename:
            default_dir = os.path.dirname(self.video_filename)
            default_filename = os.path.splitext(os.path.basename(self.
                video_filename))[0]
        if format_type == 'COCO JSON':
            default_path = os.path.join(default_dir, default_filename + '.json'
                ) if default_filename else ''
            filename, _ = QFileDialog.getSaveFileName(self,
                'Export Annotations', default_path,
                'JSON Files (*.json);;All Files (*)')
            export_format = 'coco'
        elif format_type == 'YOLO TXT':
            if hasattr(self, 'is_image_dataset') and self.is_image_dataset:
                default_path = os.path.join(default_dir, default_filename +
                    '_yolo')
                export_dir = QFileDialog.getExistingDirectory(self,
                    'Select Directory for YOLO Export', default_path,
                    QFileDialog.ShowDirsOnly)
                if export_dir:
                    from viat.utils.task_runner import run_task_with_progress
                    run_task_with_progress(self, 'Exporting YOLO',
                        'Exporting to YOLO format...',
                        export_image_dataset_yolo, export_dir, self.
                        image_files, self.frame_annotations, self.canvas.
                        class_colors, maximum=100)
                    self.statusBar.showMessage(
                        f'Annotations exported to YOLO format in {os.path.basename(export_dir)}'
                        )
                return
            else:
                default_path = os.path.join(default_dir, default_filename +
                    '.txt') if default_filename else ''
                filename, _ = QFileDialog.getSaveFileName(self,
                    'Export Annotations', default_path,
                    'Text Files (*.txt);;All Files (*)')
                export_format = 'yolo'
        elif format_type == 'Pascal VOC XML':
            if hasattr(self, 'is_image_dataset') and self.is_image_dataset:
                default_path = os.path.join(default_dir, default_filename +
                    '_voc')
                export_dir = QFileDialog.getExistingDirectory(self,
                    'Select Directory for Pascal VOC Export', default_path,
                    QFileDialog.ShowDirsOnly)
                if export_dir:
                    from viat.utils.task_runner import run_task_with_progress
                    run_task_with_progress(self, 'Exporting Pascal VOC',
                        'Exporting to Pascal VOC format...',
                        export_image_dataset_pascal_voc, export_dir, self.
                        image_files, self.frame_annotations, self.canvas.
                        pixmap, maximum=100)
                    self.statusBar.showMessage(
                        f'Annotations exported to Pascal VOC format in {os.path.basename(export_dir)}'
                        )
                return
            else:
                default_path = os.path.join(default_dir, default_filename +
                    '.xml') if default_filename else ''
                filename, _ = QFileDialog.getSaveFileName(self,
                    'Export Annotations', default_path,
                    'XML Files (*.xml);;All Files (*)')
                export_format = 'pascal_voc'
        elif format_type == 'Raya TXT':
            default_path = os.path.join(default_dir, default_filename + '.txt'
                ) if default_filename else ''
            filename, _ = QFileDialog.getSaveFileName(self,
                'Export Annotations', default_path,
                'Text Files (*.txt);;All Files (*)')
            export_format = 'raya'
        elif format_type == 'Raya with classes TXT':
            default_path = os.path.join(default_dir, default_filename + '.txt'
                ) if default_filename else ''
            filename, _ = QFileDialog.getSaveFileName(self,
                'Export Annotations', default_path,
                'Text Files (*.txt);;All Files (*)')
            export_format = 'raya_with_classes'
        else:
            return
        if filename:
            try:
                image_width = self.canvas.pixmap.width(
                    ) if self.canvas.pixmap else 640
                image_height = self.canvas.pixmap.height(
                    ) if self.canvas.pixmap else 480
                if hasattr(self, 'is_image_dataset'
                    ) and self.is_image_dataset and export_format == 'coco':
                    from viat.utils.task_runner import run_task_with_progress
                    run_task_with_progress(self, 'Exporting COCO',
                        'Exporting to COCO format...',
                        export_image_dataset_coco, filename, self.
                        image_files, self.frame_annotations, self.canvas.
                        class_colors, image_width, image_height, maximum=100)
                elif export_format == 'raya_with_classes':
                    from viat.utils.file_operations import export_raya_with_classes_annotations
                    all_annotations = []
                    for frame_num, annotations in self.frame_annotations.items(
                        ):
                        for annotation in annotations:
                            annotation_copy = annotation
                            annotation_copy.frame = frame_num
                            all_annotations.append(annotation_copy)
                    if not all_annotations and self.canvas.annotations:
                        all_annotations = self.canvas.annotations
                    classes = list(self.canvas.class_colors.keys())
                    deleted = getattr(self, 'deleted_frames', set())
                    total_f = getattr(self, 'total_frames', None)
                    export_raya_with_classes_annotations(filename,
                        all_annotations, classes, deleted_frames=deleted, total_frames=total_f)
                else:
                    deleted = getattr(self, 'deleted_frames', set())
                    total_f = getattr(self, 'total_frames', None)
                    export_standard_annotations(filename, self.
                        frame_annotations, self.canvas.annotations,
                        export_format, image_width, image_height, deleted_frames=deleted, total_frames=total_f)
                self.statusBar.showMessage(
                    f'Annotations exported to {os.path.basename(filename)}')
            except Exception as e:
                QMessageBox.critical(self, 'Error',
                    f'Failed to export annotations: {str(e)}')
                import traceback
                traceback.print_exc()

    @log_exceptions
    @log_exceptions
    def export_blurred_video(self, interactive=True):
        """Export the current video or image dataset with blurs baked into the frames/files."""
        is_video = hasattr(self, 'video_filename') and self.video_filename and getattr(self, 'cap', None) is not None
        is_image_ds = getattr(self, 'is_image_dataset', False) or (hasattr(self, 'image_files') and bool(self.image_files))

        if not is_video and not is_image_ds:
            if interactive:
                QMessageBox.warning(self, 'Export Blurred Media', 'Please open a video file or an image dataset first!')
            return
            
        if not hasattr(self, 'blur_manager') or not self.blur_manager.blur_regions:
            if interactive:
                QMessageBox.information(self, 'Export Blurred Media', 'There are no blur regions to export!')
            return

        if is_image_ds and not is_video:
            self.export_blurred_image_dataset(interactive=interactive)
            return

        base, ext = os.path.splitext(self.video_filename)
        output_filename = f"{base}_blurred{ext}"
        
        if interactive:
            reply = QMessageBox.question(self, 'Export Blurred Video',
                f"This will export a new video with all blur regions baked in.\n\nSave to: {output_filename}\n\nContinue?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                
            if reply != QMessageBox.Yes:
                return
            
        progress = QProgressDialog("Exporting blurred video...", "Cancel", 0, self.total_frames, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        
        # Get video properties
        orig_cap = cv2.VideoCapture(self.video_filename)
        fps = orig_cap.get(cv2.CAP_PROP_FPS)
        width = int(orig_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(orig_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') if ext.lower() == '.mp4' else int(orig_cap.get(cv2.CAP_PROP_FOURCC))
        
        writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
        
        for i in range(self.total_frames):
            if progress.wasCanceled():
                break
            ret, frame = orig_cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.blur_manager.has_blur(i):
                frame_rgb = self.blur_manager.apply_blur_to_frame(frame_rgb, i)
                
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            writer.write(frame_bgr)
            
            if i % 10 == 0:
                progress.setValue(i)
                QApplication.processEvents()
                
        writer.release()
        orig_cap.release()
        progress.close()
        
        if not progress.wasCanceled() and interactive:
            QMessageBox.information(self, 'Success', f'Blurred video successfully exported to:\n{output_filename}')

    @log_exceptions
    def export_blurred_image_dataset(self, interactive=True):
        """Export or burn blur regions into the image dataset files on disk."""
        if not hasattr(self, 'image_files') or not self.image_files:
            if interactive:
                QMessageBox.warning(self, 'Export Blurred Images', 'No image files found in current dataset!')
            return
            
        if not hasattr(self, 'blur_manager') or not self.blur_manager.blur_regions:
            if interactive:
                QMessageBox.information(self, 'Export Blurred Images', 'There are no blur regions to export!')
            return

        mode = "new_dir"
        output_dir = ""

        if interactive:
            msg_box = QMessageBox(self)
            msg_box.setWindowTitle("Export Blurred Image Dataset")
            msg_box.setText(
                "Do you want to export the blurred images to a NEW directory, "
                "or OVERWRITE the original image files in place?\n\n"
                "Note: Overwriting original images is permanent and cannot be undone."
            )
            btn_new = msg_box.addButton("Export to New Directory", QMessageBox.AcceptRole)
            btn_overwrite = msg_box.addButton("Overwrite Original Images", QMessageBox.DestructiveRole)
            btn_cancel = msg_box.addButton(QMessageBox.Cancel)
            
            msg_box.exec_()
            clicked_btn = msg_box.clickedButton()

            if clicked_btn == btn_cancel or clicked_btn is None:
                return
            elif clicked_btn == btn_overwrite:
                confirm = QMessageBox.question(
                    self,
                    "Confirm Overwrite",
                    f"Are you sure you want to permanently apply blurs to {len(self.image_files)} original image files in place?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if confirm != QMessageBox.Yes:
                    return
                mode = "overwrite"
            else:
                mode = "new_dir"
                default_dir = os.path.dirname(self.image_files[0]) if self.image_files else ""
                parent_dir = os.path.dirname(default_dir)
                folder_name = os.path.basename(default_dir) + "_blurred"
                suggested_path = os.path.join(parent_dir, folder_name)

                output_dir = QFileDialog.getExistingDirectory(
                    self,
                    "Select Output Directory for Blurred Images",
                    suggested_path,
                    QFileDialog.ShowDirsOnly
                )
                if not output_dir:
                    return

        total_files = len(self.image_files)
        progress = QProgressDialog("Applying blur regions to images...", "Cancel", 0, total_files, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        processed_count = 0
        blurred_count = 0

        for i, img_path in enumerate(self.image_files):
            if progress.wasCanceled():
                break

            has_blur = self.blur_manager.has_blur(i)
            
            if mode == "new_dir":
                img_name = os.path.basename(img_path)
                dest_path = os.path.join(output_dir, img_name)
                
                # Preserve dataset split subdirectory if available
                if hasattr(self, '_viat_frame_to_split') and i < len(self._viat_frame_to_split):
                    split_name = self._viat_frame_to_split[i]
                    if split_name and split_name != "root":
                        dest_dir = os.path.join(output_dir, split_name, "images")
                        os.makedirs(dest_dir, exist_ok=True)
                        dest_path = os.path.join(dest_dir, img_name)

                if has_blur:
                    img_obj = cv2.imread(img_path)
                    if img_obj is not None:
                        img_rgb = cv2.cvtColor(img_obj, cv2.COLOR_BGR2RGB)
                        blurred_rgb = self.blur_manager.apply_blur_to_frame(img_rgb, i)
                        blurred_bgr = cv2.cvtColor(blurred_rgb, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(dest_path, blurred_bgr)
                        blurred_count += 1
                else:
                    try:
                        shutil.copy2(img_path, dest_path)
                    except OSError:
                        pass
            elif mode == "overwrite":
                if has_blur:
                    img_obj = cv2.imread(img_path)
                    if img_obj is not None:
                        img_rgb = cv2.cvtColor(img_obj, cv2.COLOR_BGR2RGB)
                        blurred_rgb = self.blur_manager.apply_blur_to_frame(img_rgb, i)
                        blurred_bgr = cv2.cvtColor(blurred_rgb, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(img_path, blurred_bgr)
                        blurred_count += 1

            processed_count += 1
            if i % 5 == 0:
                progress.setValue(i)
                QApplication.processEvents()

        progress.close()

        if mode == "overwrite" and blurred_count > 0:
            self.load_current_image()

        if not progress.wasCanceled() and interactive:
            if mode == "new_dir":
                QMessageBox.information(
                    self, 'Success',
                    f'Blurred images exported successfully ({blurred_count} blurred images) to:\n{output_dir}'
                )
            else:
                QMessageBox.information(
                    self, 'Success',
                    f'Successfully applied/burned blurs into {blurred_count} original image files.'
                )
        self.statusBar.showMessage(f'Applied blurs to {blurred_count} images.', 5000)

    @log_exceptions
    def export_image_dataset(self):
        """Export the current image dataset with advanced options."""
        has_video = hasattr(self, 'video_filename') and self.video_filename
        has_image_dataset = hasattr(self, 'is_image_dataset') and self.is_image_dataset
        
        if not has_image_dataset and not has_video:
            QMessageBox.warning(self, 'Export Image Dataset',
                'Please open an image dataset or a video file first!')
            return
            
        if has_image_dataset:
            image_files = self.image_files
        else:
            video_base = os.path.splitext(os.path.basename(self.video_filename))[0]
            image_files = [f"{video_base}_frame_{i:06d}.jpg" for i in range(self.total_frames)]

        config = export_dataset_dialog(self, image_files, self.
            frame_annotations)
        if not config:
            return
        from viat.utils.task_runner import run_task_with_progress
        result = run_task_with_progress(self, 'Exporting Dataset',
            'Initializing export...', export_dataset, self, config,
            image_files, self.frame_annotations, self.canvas.class_colors,
            maximum=100)
        if result:
            self.statusBar.showMessage('Export complete', 5000)
            
            # Suggest merging with another dataset
            reply = QMessageBox.question(self, 'Merge Dataset',
                'Export complete! Would you like to merge this exported dataset with another existing image dataset?'
                , QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                self.viat_merge_dataset(source_folder=config["output_dir"])

    @log_exceptions
    def create_dataset(self):
        """Create a new dataset from the current annotations."""
        has_video = hasattr(self, 'video_filename') and self.video_filename
        has_image_dataset = hasattr(self, 'is_image_dataset') and self.is_image_dataset
        
        if not has_image_dataset and not has_video:
            QMessageBox.warning(self, 'Create Dataset',
                'This feature is only available for image datasets or open video files.')
            return
        has_annotations = any(self.frame_annotations.values())
        if not has_annotations:
            QMessageBox.warning(self, 'Create Dataset',
                'No annotations to export!')
            return
            
        if has_image_dataset:
            image_files = self.image_files
        else:
            video_base = os.path.splitext(os.path.basename(self.video_filename))[0]
            image_files = [f"{video_base}_frame_{i:06d}.jpg" for i in range(self.total_frames)]
            
        from viat.utils.dataset_manager import create_dataset_dialog, create_dataset
        config = create_dataset_dialog(self, image_files, self.
            frame_annotations, self.canvas.class_colors)
        if config:
            success = create_dataset(self, config, image_files, self.
                frame_annotations, self.canvas.class_colors)
            if success:
                QMessageBox.information(self, 'Dataset Created',
                    f"Dataset created successfully in {config['output_dir']}")
                
                # Suggest merging with another dataset
                reply = QMessageBox.question(self, 'Merge Dataset',
                    'Dataset created! Would you like to merge this dataset with another existing image dataset?'
                    , QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if reply == QMessageBox.Yes:
                    self.viat_merge_dataset(source_folder=config["output_dir"])

    @log_exceptions
    def update_class_ui_after_import(self):
        """Update the class-related UI components after importing annotations."""
        if hasattr(self, 'toolbar') and hasattr(self.toolbar,
            'update_class_selector'):
            self.toolbar.update_class_selector()
        if hasattr(self, 'class_dock') and hasattr(self.class_dock,
            'update_class_list'):
            self.class_dock.update_class_list()
        if hasattr(self, 'class_frames_dock'):
            self.class_frames_dock.update_classes(list(self.canvas.class_colors.keys()))
        if self.canvas.class_colors and hasattr(self.canvas,
            'set_current_class'):
            first_class = next(iter(self.canvas.class_colors))
            self.canvas.set_current_class(first_class)
            if hasattr(self, 'class_selector') and self.class_selector.count(
                ) > 0:
                self.class_selector.setCurrentText(first_class)

    @log_exceptions
    def import_annotations(self, filename=None):
        """
        Import annotations from a file.

        Args:
            filename (str, optional): Path to the annotation file. If None, a file dialog will be shown.
        """
        if filename is None:
            filename, _ = QFileDialog.getOpenFileName(self,
                'Import Annotations', '',
                'All Files (*);;JSON Files (*.json);;Text Files (*.txt);;XML Files (*.xml)'
                )
            if not filename:
                return
        self.save_undo_state('all')
        self._annotations_imported.add(filename)
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
                if 'viat_project_identifier' in data:
                    QMessageBox.information(self, 'Project File Detected',
                        f"{os.path.basename(filename)} is a VIAT project file, not an annotation file. Please use 'Open Project' to load this file."
                        )
                    return
        except:
            pass
        if self.canvas.pixmap:
            image_width = self.canvas.pixmap.width()
            image_height = self.canvas.pixmap.height()
        else:
            image_width = 640
            image_height = 480
        try:
            from viat.utils.file_operations import import_annotations as import_annotations_func, detect_annotation_format, extract_raya_classes
            format_type = detect_annotation_format(filename)
            class_mapping = None
            if format_type == 'Raya with classes':
                imported_classes = extract_raya_classes(filename)
                existing_classes = list(self.canvas.class_colors.keys())
                if imported_classes and existing_classes:
                    dialog = ClassMappingDialog(imported_classes,
                        existing_classes, self)
                    if dialog.exec_() == QDialog.Accepted:
                        class_mapping = dialog.get_mapping()
                        for imp_cls, tgt_cls in dialog.mapping.items():
                            if tgt_cls not in existing_classes:
                                self.add_class(tgt_cls)
                    else:
                        return
                elif imported_classes:
                    class_mapping = {}
                    for idx, cls in enumerate(imported_classes):
                        class_mapping[idx] = cls
                        self.add_class(cls)
            elif format_type == 'CSV_XYWH_NO_CLASS':
                from PyQt5.QtWidgets import QInputDialog
                text, ok = QInputDialog.getText(self, 'Input Class', 'Enter class name for imported labels:')
                if ok and text:
                    class_name = text.strip()
                    if not class_name:
                        class_name = "Object"
                else:
                    return
                
                existing_classes = list(self.canvas.class_colors.keys())
                if class_name not in existing_classes:
                    self.add_class(class_name)
                    
                class_mapping = {0: class_name}
            res = import_annotations_func(
                filename, BoundingBox, image_width,
                image_height, self.canvas.class_colors, class_mapping=class_mapping
            )
            format_type, annotations, imported_frame_annotations = res[0], res[1], res[2]
            imported_deleted_frames = res[3] if len(res) > 3 else set()

            for frame_num, anns in imported_frame_annotations.items():
                if frame_num not in self.frame_annotations:
                    self.frame_annotations[frame_num] = []
                self.frame_annotations[frame_num].extend(anns)
                
            if imported_deleted_frames:
                if not hasattr(self, 'deleted_frames'):
                    self.deleted_frames = set()
                self.deleted_frames.update(imported_deleted_frames)
                
            # Store base annotations in labeler_analytics
            if hasattr(self, 'labeler_analytics'):
                if 'base_annotations' not in self.labeler_analytics:
                    self.labeler_analytics['base_annotations'] = {}
                for frame_num, anns in imported_frame_annotations.items():
                    if str(frame_num) not in self.labeler_analytics['base_annotations']:
                        self.labeler_analytics['base_annotations'][str(frame_num)] = []
                    self.labeler_analytics['base_annotations'][str(frame_num)].extend([a.to_dict() for a in anns])
            if self.current_frame in imported_frame_annotations:
                self.canvas.annotations.extend(imported_frame_annotations[
                    self.current_frame])
                self.canvas.update()
            self.annotation_dock.update_annotation_list()
            QMessageBox.information(self, 'Import Successful',
                f'Successfully imported annotations from {os.path.basename(filename)} ({format_type} format).'
                )
        except Exception as e:
            QMessageBox.critical(self, 'Import Error',
                f'Error importing annotations: {str(e)}')

    @log_exceptions
    def check_for_annotation_files(self, video_filename):
        """
        Check if annotation files with the same base name as the video exist.
        If found, ask the user if they want to import them.

        Args:
            video_filename (str): Path to the video file
        """
        extensions = ['.txt', '.json', '.xml', '.csv']
        file_basename, _ = os.path.splitext(video_filename)
        for vid in self._annotations_imported:
            for ext in extensions:
                if Path(file_basename + ext) == Path(vid):
                    return
        directory = os.path.join(os.path.dirname(video_filename))
        save_path = os.path.join(directory, 'auto_save')
        base_name = os.path.splitext(os.path.basename(video_filename))[0]
        autosave_file = os.path.join(save_path, f'{base_name}_autosave.json')
        if os.path.exists(autosave_file) and not hasattr(self,
            '_loading_from_project'):
            if not hasattr(self, '_autosave_prompted'):
                self._autosave_prompted = set()
            if video_filename not in self._autosave_prompted:
                self._autosave_prompted.add(video_filename)
                reply = QMessageBox.question(self, 'Auto-Save Found',
                    """An auto-save file was found for this video.
Would you like to load it?"""
                    , QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if reply == QMessageBox.Yes:
                    try:
                        self._loading_from_project = True
                        self.load_project(autosave_file)
                        self._loading_from_project = False
                        return
                    except Exception as e:
                        self._loading_from_project = False
                        QMessageBox.warning(self, 'Auto-Save Error',
                            f'Error loading auto-save file: {str(e)}')
        annotation_files = []
        for ext in extensions:
            potential_file = os.path.join(directory, base_name + ext)
            if os.path.exists(potential_file
                ) and potential_file != autosave_file:
                annotation_files.append(potential_file)
        for an in annotation_files:
            if an.endswith('.json'):
                with open(an, 'r') as f:
                    data = json.load(f)
                    if 'viat_project_identifier' in data:
                        annotation_files.remove(an)
        if annotation_files:
            message = 'Found the following annotation file(s):\n\n'
            for file in annotation_files:
                message += f'- {os.path.basename(file)}\n'
            message += (
                '\nWould you like to import annotations from one of these files?'
                )
            reply = QMessageBox.question(self, 'Annotation Files Found',
                message, QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply == QMessageBox.Yes:
                if len(annotation_files) > 1:
                    self.show_annotation_file_selection_dialog(annotation_files
                        )
                else:
                    self.import_annotations(annotation_files[0])

    @log_exceptions
    def show_annotation_file_selection_dialog(self, annotation_files):
        """
        Show a dialog for the user to select which annotation file(s) to import.

        Args:
            annotation_files (list): List of annotation file paths
        """
        dialog = QDialog(self)
        dialog.setWindowTitle('Select Annotation Files')
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout(dialog)
        label = QLabel(
            'Multiple annotation files found. Please select which ones to import:'
            )
        layout.addWidget(label)
        list_widget = QListWidget()
        for file in annotation_files:
            item = QListWidgetItem(os.path.basename(file))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            list_widget.addItem(item)
        layout.addWidget(list_widget)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.
            Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec_() == QDialog.Accepted:
            selected_files = []
            for i in range(list_widget.count()):
                item = list_widget.item(i)
                if item.checkState() == Qt.Checked:
                    selected_files.append(annotation_files[i])
            if selected_files:
                self.import_multiple_annotations(selected_files)

    @log_exceptions
    def import_multiple_annotations(self, annotation_files):
        """
        Import annotations from multiple files.

        Args:
            annotation_files (list): List of annotation file paths
        """
        progress = QDialog(self)
        progress.setWindowTitle('Importing Annotations')
        progress.setFixedSize(400, 100)
        progress_layout = QVBoxLayout(progress)
        status_label = QLabel('Importing annotations...')
        progress_layout.addWidget(status_label)
        progress_bar = QProgressBar()
        progress_bar.setRange(0, len(annotation_files))
        progress_layout.addWidget(progress_bar)
        progress.setModal(False)
        progress.show()
        QApplication.processEvents()
        for i, file_path in enumerate(annotation_files):
            status_label.setText(f'Importing {os.path.basename(file_path)}...')
            progress_bar.setValue(i)
            QApplication.processEvents()
            try:
                self.import_annotations(file_path)
            except Exception as e:
                print(f'Error importing {file_path}: {str(e)}')
        progress.close()
        QMessageBox.information(self, 'Import Complete',
            f'Successfully imported annotations from {len(annotation_files)} files.'
            )

    @log_exceptions
    def copy_selected_annotation(self):
        """Copy the currently selected annotation."""
        if hasattr(self, 'canvas') and self.canvas.selected_annotation:
            self.clipboard_annotation = self.canvas.selected_annotation.copy()
            self.statusBar.showMessage('Annotation copied', 2000)

    @log_exceptions
    def paste_annotation(self):
        """Paste the copied annotation to the current frame."""
        if hasattr(self, 'clipboard_annotation') and self.clipboard_annotation:
            self.save_undo_state()
            new_annotation = self.clipboard_annotation.copy()
            current_frame = self.current_frame
            if getattr(self, 'auto_blur_labels', False):
                if hasattr(self, 'blur_manager') and self.blur_manager is not None:
                    self.blur_manager.add_bbox_region(current_frame, new_annotation.rect, getattr(self.canvas, 'blur_kernel', 151))
                    if hasattr(self, '_refresh_blur_display'):
                        self._refresh_blur_display()
                    self.statusBar.showMessage('Annotation pasted as blur region', 2000)
                    return
            if current_frame not in self.frame_annotations:
                self.frame_annotations[current_frame] = []
            self.frame_annotations[current_frame].append(new_annotation)
            self.canvas.annotations = self.frame_annotations.get(current_frame,
                [])
            self.canvas.selected_annotation = new_annotation
            self.canvas.update()
            if hasattr(self, 'annotation_dock'):
                self.annotation_dock.update_annotation_list()
            self.statusBar.showMessage('Annotation pasted', 2000)

    @log_exceptions
    def toggle_auto_blur_labels(self, checked=None):
        """Toggle automatic blurring of newly added labels."""
        if checked is None:
            self.auto_blur_labels = not getattr(self, 'auto_blur_labels', False)
        else:
            self.auto_blur_labels = bool(checked)

        if hasattr(self, 'auto_blur_action') and self.auto_blur_action.isChecked() != self.auto_blur_labels:
            self.auto_blur_action.blockSignals(True)
            self.auto_blur_action.setChecked(self.auto_blur_labels)
            self.auto_blur_action.blockSignals(False)

        if hasattr(self, 'auto_blur_menu_action') and self.auto_blur_menu_action.isChecked() != self.auto_blur_labels:
            self.auto_blur_menu_action.blockSignals(True)
            self.auto_blur_menu_action.setChecked(self.auto_blur_labels)
            self.auto_blur_menu_action.blockSignals(False)

        state_str = "enabled" if self.auto_blur_labels else "disabled"
        self.statusBar.showMessage(f"Auto-blur new labels {state_str}", 3000)

    @log_exceptions
    def toggle_auto_save_blur_on_switch(self, checked=None):
        """Toggle automatic saving/burning of blur regions into image files when switching frames."""
        if checked is None:
            self.auto_save_blur_on_switch = not getattr(self, 'auto_save_blur_on_switch', False)
        else:
            self.auto_save_blur_on_switch = bool(checked)

        if hasattr(self, 'auto_save_blur_action') and self.auto_save_blur_action.isChecked() != self.auto_save_blur_on_switch:
            self.auto_save_blur_action.blockSignals(True)
            self.auto_save_blur_action.setChecked(self.auto_save_blur_on_switch)
            self.auto_save_blur_action.blockSignals(False)

        state_str = "enabled" if self.auto_save_blur_on_switch else "disabled"
        self.statusBar.showMessage(f"Auto-save/burn blur on frame switch {state_str}", 3000)

    @log_exceptions
    def _handle_auto_save_blur_on_frame_change(self, old_frame):
        """Helper to trigger auto-saving of blur regions for an image dataset when changing frames."""
        if not getattr(self, 'auto_save_blur_on_switch', False):
            return
        if not getattr(self, 'is_image_dataset', False) or not getattr(self, 'image_files', None):
            return
        if old_frame < 0 or old_frame >= len(self.image_files):
            return
        if not hasattr(self, 'blur_manager') or not self.blur_manager.has_blur(old_frame):
            return

        self._backup_and_save_blurred_image(old_frame)

    @log_exceptions
    def _backup_and_save_blurred_image(self, frame_idx):
        """Bake blur regions for frame_idx into the image file on disk, backing up previous version(s) to unblurred_backups/."""
        if not hasattr(self, 'image_files') or not self.image_files:
            return False
        if frame_idx < 0 or frame_idx >= len(self.image_files):
            return False
        if not hasattr(self, 'blur_manager') or not self.blur_manager.has_blur(frame_idx):
            return False

        image_path = self.image_files[frame_idx]
        if not os.path.exists(image_path):
            return False

        try:
            image_dir = os.path.dirname(image_path)
            backup_dir = os.path.join(image_dir, "unblurred_backups")
            os.makedirs(backup_dir, exist_ok=True)

            base_name, ext = os.path.splitext(os.path.basename(image_path))
            backup_filename = f"{base_name}_unblurred_1{ext}"
            backup_path = os.path.join(backup_dir, backup_filename)

            counter = 1
            while os.path.exists(backup_path):
                counter += 1
                backup_filename = f"{base_name}_unblurred_{counter}{ext}"
                backup_path = os.path.join(backup_dir, backup_filename)

            # Copy current version to backup folder before modifying original file on disk
            shutil.copy2(image_path, backup_path)

            # Read current image, apply blur, write back to original image_path
            img_bgr = cv2.imread(image_path)
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                blurred_rgb = self.blur_manager.apply_blur_to_frame(img_rgb, frame_idx)
                blurred_bgr = cv2.cvtColor(blurred_rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(image_path, blurred_bgr)

                # Clear stored blur regions for this frame in blur_manager so we don't double-blur later
                self.blur_manager.clear_frame(frame_idx)

                msg = f"Auto-saved blur for {os.path.basename(image_path)} (Backup #{counter} in unblurred_backups/)"
                self.statusBar.showMessage(msg, 5000)
                return True
        except Exception as e:
            logger.error(f"Error auto-saving blurred image for frame {frame_idx}: {e}")
        return False

    @log_exceptions
    def clear_current_frame_blur(self):
        """Remove all blur regions from the current frame."""
        if hasattr(self, 'blur_manager') and self.blur_manager is not None:
            self.save_undo_state()
            self.blur_manager.clear_frame(self.current_frame)
            if hasattr(self, '_refresh_blur_display'):
                self._refresh_blur_display()
            self.statusBar.showMessage(f"Cleared blur for frame {self.current_frame}", 3000)

    @log_exceptions
    def clear_blur_range(self):
        """Open dialog to remove blur regions from a range of frames."""
        if not hasattr(self, 'blur_manager') or self.blur_manager is None:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Clear Blur in Range")
        layout = QVBoxLayout(dialog)

        form_layout = QFormLayout()
        total_f = getattr(self, 'total_frames', 1)
        max_f = max(0, total_f - 1)

        start_spin = QSpinBox()
        start_spin.setRange(0, max_f)
        start_spin.setValue(self.current_frame)

        end_spin = QSpinBox()
        end_spin.setRange(0, max_f)
        end_spin.setValue(min(self.current_frame + 10, max_f))

        form_layout.addRow("Start Frame:", start_spin)
        form_layout.addRow("End Frame:", end_spin)
        layout.addLayout(form_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() == QDialog.Accepted:
            start_f = start_spin.value()
            end_f = end_spin.value()
            if start_f > end_f:
                start_f, end_f = end_f, start_f

            self.save_undo_state(range(start_f, end_f + 1))
            self.blur_manager.clear_range(start_f, end_f)
            if hasattr(self, '_refresh_blur_display'):
                self._refresh_blur_display()
            self.statusBar.showMessage(f"Cleared blur for frames {start_f} to {end_f}", 4000)

    @log_exceptions
    def apply_blur_to_annotation_range(self, annotation):
        """Apply blur to the region defined by an annotation over a range of frames."""
        if not hasattr(self, 'blur_manager') or self.blur_manager is None:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Blur Region in Range")
        layout = QVBoxLayout(dialog)
        
        label = QLabel("Apply blur to this region across the following frames:")
        layout.addWidget(label)

        form_layout = QFormLayout()
        total_f = getattr(self, 'total_frames', 1)
        max_f = max(0, total_f - 1)

        start_spin = QSpinBox()
        start_spin.setRange(0, max_f)
        start_spin.setValue(self.current_frame)

        end_spin = QSpinBox()
        end_spin.setRange(0, max_f)
        end_spin.setValue(min(self.current_frame + 10, max_f))

        form_layout.addRow("Start Frame:", start_spin)
        form_layout.addRow("End Frame:", end_spin)
        layout.addLayout(form_layout)
        
        options_layout = QVBoxLayout()
        delete_check = QCheckBox("Delete annotation after applying blur")
        delete_check.setChecked(True)
        options_layout.addWidget(delete_check)
        
        remove_under_check = QCheckBox("Remove other annotations covered by this blur (>70%)")
        remove_under_check.setChecked(False)
        options_layout.addWidget(remove_under_check)
        
        layout.addLayout(options_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec_() == QDialog.Accepted:
            start_f = start_spin.value()
            end_f = end_spin.value()
            delete_after = delete_check.isChecked()
            remove_under = remove_under_check.isChecked()
            
            if start_f > end_f:
                start_f, end_f = end_f, start_f

            self.save_undo_state(range(start_f, end_f + 1))
            
            kernel = getattr(self.canvas, 'blur_kernel', 151)
            has_seg = hasattr(annotation, 'segmentation') and annotation.segmentation
            
            for f in range(start_f, end_f + 1):
                if has_seg:
                    self.blur_manager.add_polygon_region(f, annotation.segmentation, kernel)
                else:
                    self.blur_manager.add_bbox_region(f, annotation.rect, kernel)
                    
                if remove_under:
                    self.remove_annotations_under_blur(f, threshold=0.7)
            
            if delete_after:
                self.delete_selected_annotation()
                
            if hasattr(self, '_refresh_blur_display'):
                self._refresh_blur_display()
            else:
                self.canvas.update()
                
            self.statusBar.showMessage(f"Applied blur region to frames {start_f} to {end_f}", 4000)

    @log_exceptions
    def remove_annotations_under_blur(self, frame_idx, threshold=0.7):
        """Remove annotations in a frame that are mostly covered by all blur regions combined."""
        if frame_idx not in self.frame_annotations:
            return
            
        blur_mgr = getattr(self, 'blur_manager', None)
        if not blur_mgr or not blur_mgr.has_blur(frame_idx):
            return
            
        annotations = self.frame_annotations[frame_idx]
        if not annotations:
            return
            
        # Get frame dimensions
        if not self.canvas.pixmap or self.canvas.pixmap.isNull():
            return
            
        w, h = self.canvas.pixmap.width(), self.canvas.pixmap.height()
        
        # Create a blank mask
        blur_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Draw all blur regions onto the mask
        for region in blur_mgr.get_regions(frame_idx):
            x1 = int(max(0, region["x"]))
            y1 = int(max(0, region["y"]))
            x2 = int(min(w, x1 + region["w"]))
            y2 = int(min(h, y1 + region["h"]))
            if x2 <= x1 or y2 <= y1:
                continue
                
            if region.get("type") == "polygon" and "points" in region:
                points = np.array(region["points"], dtype=np.int32)
                shifted = points - [x1, y1]
                patch_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
                cv2.fillPoly(patch_mask, [shifted], 255)
                np.maximum(blur_mask[y1:y2, x1:x2], patch_mask, out=blur_mask[y1:y2, x1:x2])
            else:
                blur_mask[y1:y2, x1:x2] = 255
                
        annotations_to_remove = []
        for ann in annotations:
            a_rect = ann.rect
            ann_x = int(a_rect.x())
            ann_y = int(a_rect.y())
            ann_w = int(a_rect.width())
            ann_h = int(a_rect.height())
            
            x1 = int(max(0, ann_x))
            y1 = int(max(0, ann_y))
            x2 = int(min(w, ann_x + ann_w))
            y2 = int(min(h, ann_y + ann_h))
            
            if x2 <= x1 or y2 <= y1:
                continue
                
            a_area = a_rect.width() * a_rect.height()
            if a_area <= 0:
                continue
                
            intersect_mask = blur_mask[y1:y2, x1:x2]
            overlap_area = np.count_nonzero(intersect_mask)
            overlap_ratio = overlap_area / a_area
            
            if overlap_ratio >= threshold:
                annotations_to_remove.append(ann)
                    
        for ann in annotations_to_remove:
            self.frame_annotations[frame_idx].remove(ann)
            
        if frame_idx == self.current_frame and annotations_to_remove:
            self.canvas.annotations = self.frame_annotations.get(self.current_frame, [])
            self.canvas.update()
            if hasattr(self, 'annotation_dock'):
                self.annotation_dock.update_annotation_list()

    @log_exceptions
    def clear_all_blurs(self):
        """Remove all blur regions across the entire dataset."""
        if not hasattr(self, 'blur_manager') or self.blur_manager is None:
            return

        reply = QMessageBox.question(
            self, "Clear All Blurs",
            "Are you sure you want to remove all blur regions from all frames?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self.save_undo_state('all')
            self.blur_manager.clear_all()
            if hasattr(self, '_refresh_blur_display'):
                self._refresh_blur_display()
            self.statusBar.showMessage("Cleared all blur regions across video", 4000)

    @log_exceptions
    def cut_selected_annotation(self):
        """Cut (copy and delete) the selected annotation."""
        if hasattr(self, 'canvas') and self.canvas.selected_annotation:
            self.save_undo_state()
            self.clipboard_annotation = self.canvas.selected_annotation.copy()
            current_frame = self.current_frame
            if current_frame in self.frame_annotations:
                if self.canvas.selected_annotation in self.frame_annotations[
                    current_frame]:
                    self.frame_annotations[current_frame].remove(self.
                        canvas.selected_annotation)
            self.canvas.annotations = self.frame_annotations.get(current_frame,
                [])
            self.canvas.selected_annotation = None
            self.canvas.update()
            if hasattr(self, 'annotation_dock'):
                self.annotation_dock.update_annotation_list()
            self.statusBar.showMessage('Annotation cut', 2000)

    @log_exceptions
    def toggle_duplicate_frames_detection(self):
        """Toggle automatic duplicate frame detection and annotation propagation."""
        self.duplicate_frames_enabled = self.duplicate_frames_action.isChecked(
            )
        if self.duplicate_frames_enabled:
            if not self.frame_hashes:
                reply = QMessageBox.question(self,
                    'Duplicate Frame Detection',
                    """This will automatically propagate annotations to duplicate frames.

Do you want to scan the entire video now for duplicate frames?
(This may take some time for long videos)"""
                    , QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
                if reply == QMessageBox.Yes:
                    self.scan_video_for_duplicates()
            self.statusBar.showMessage(
                'Duplicate frame detection enabled - annotations will be propagated automatically'
                )
        else:
            self.statusBar.showMessage('Duplicate frame detection disabled')
        self.project_modified = True

    @log_exceptions
    def scan_video_for_duplicates(self):
        """Scan the entire video to identify duplicate frames."""
        if not self.cap or not self.cap.isOpened():
            QMessageBox.warning(self, 'Scan Video',
                'Please open a video first!')
            return
        progress = QDialog(self)
        progress.setWindowTitle('Scanning Video')
        progress.setFixedSize(300, 100)
        layout = QVBoxLayout(progress)
        label = QLabel('Scanning for duplicate frames...')
        layout.addWidget(label)
        progress_bar = QProgressBar()
        progress_bar.setRange(0, self.total_frames)
        layout.addWidget(progress_bar)
        progress.setModal(False)
        progress.show()
        QApplication.processEvents()
        current_pos = self.current_frame
        self.duplicate_frames_cache = {}
        self.frame_hashes = {}
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for frame_num in range(self.total_frames):
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = self._process_frame_metadata(frame, frame_num)
            progress_bar.setValue(frame_num)
            if frame_num % 10 == 0:
                QApplication.processEvents()
            frame_hash = calculate_frame_hash(frame)
            self.frame_hashes[frame_num] = frame_hash
            if frame_hash in self.duplicate_frames_cache:
                self.duplicate_frames_cache[frame_hash].append(frame_num)
            else:
                self.duplicate_frames_cache[frame_hash] = [frame_num]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        ret, frame = self.cap.read()
        if ret:
            frame = self._process_frame_metadata(frame, current_pos)
            self.canvas.set_frame(frame)
        self.frame_hashes, self.duplicate_frames_cache = (self.
            performance_manager.optimize_frame_hashes(self.frame_hashes,
            self.duplicate_frames_cache))
        progress.close()
        duplicate_count = sum(len(frames) - 1 for frames in self.
            duplicate_frames_cache.values() if len(frames) > 1)
        QMessageBox.information(self, 'Scan Complete',
            f'Found {duplicate_count} duplicate frames in {self.total_frames} total frames.'
            )

    @log_exceptions
    def propagate_annotations_to_duplicate(self, frame_hash):
        """
        Propagate annotations from other frames with the same hash to the current frame.

        Args:
            frame_hash (str): The hash of the current frame
        """
        duplicate_frames = self.duplicate_frames_cache[frame_hash]
        if duplicate_frames[0] == self.current_frame:
            return
        for frame_num in duplicate_frames:
            if (frame_num != self.current_frame and frame_num in self.
                frame_annotations):
                if not self.frame_annotations.get(self.current_frame):
                    self.save_undo_state(self.current_frame)
                    self.frame_annotations[self.current_frame] = [self.
                        clone_annotation(ann) for ann in self.
                        frame_annotations[frame_num]]
                    self.statusBar.showMessage(
                        f'Automatically copied annotations from duplicate frame {frame_num}'
                        , 3000)
                    return

    @log_exceptions
    def clone_annotation(self, annotation):
        """
        Create a deep copy of an annotation.

        Args:
            annotation: The annotation to clone

        Returns:
            A new annotation object with the same properties
        """
        return deepcopy(annotation)

    @log_exceptions
    def propagate_to_duplicate_frames(self, frame_hash):
        """
        Propagate current frame annotations to all duplicate frames with the same hash.

        Args:
            frame_hash (str): The hash of the current frame
        """
        if not frame_hash or frame_hash not in self.duplicate_frames_cache:
            return
        duplicate_frames = self.duplicate_frames_cache[frame_hash]
        if len(duplicate_frames) <= 1:
            return
        self.save_undo_state(duplicate_frames)
        current_annotations = [self.clone_annotation(ann) for ann in self.
            canvas.annotations]
        update_count = 0
        for frame_num in duplicate_frames:
            if frame_num != self.current_frame:
                self.frame_annotations[frame_num] = [self.clone_annotation(
                    ann) for ann in current_annotations]
                update_count += 1
        if update_count > 0:
            self.statusBar.showMessage(
                f'Automatically propagated annotations to {update_count} duplicate frames'
                , 3000)

    @log_exceptions
    def scan_images_for_duplicates(self):
        """Scan all images in the dataset to identify duplicates."""
        if not hasattr(self, 'image_files') or not self.image_files:
            return
        progress = QDialog(self)
        progress.setWindowTitle('Scanning Images')
        progress.setFixedSize(300, 100)
        layout = QVBoxLayout(progress)
        label = QLabel('Scanning for duplicate images...')
        layout.addWidget(label)
        progress_bar = QProgressBar()
        progress_bar.setRange(0, len(self.image_files))
        layout.addWidget(progress_bar)
        progress.setModal(False)
        progress.show()
        QApplication.processEvents()
        self.duplicate_frames_cache = {}
        self.frame_hashes = {}
        for frame_num, image_path in enumerate(self.image_files):
            progress_bar.setValue(frame_num)
            if frame_num % 5 == 0:
                QApplication.processEvents()
            frame = cv2.imread(image_path)
            if frame is None:
                continue
            frame_hash = calculate_frame_hash(frame)
            self.frame_hashes[frame_num] = frame_hash
            if frame_hash in self.duplicate_frames_cache:
                self.duplicate_frames_cache[frame_hash].append(frame_num)
            else:
                self.duplicate_frames_cache[frame_hash] = [frame_num]
        progress.close()
        duplicate_count = sum(len(frames) - 1 for frames in self.
            duplicate_frames_cache.values() if len(frames) > 1)
        QMessageBox.information(self, 'Scan Complete',
            f'Found {duplicate_count} duplicate images in {len(self.image_files)} total images.'
            )

    @log_exceptions
    def propagate_annotations(self):
        """Propagate current frame annotations to a range of frames."""
        if not self.canvas.annotations:
            QMessageBox.warning(self, 'Propagate Annotations',
                'No annotations in current frame to propagate!')
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('Propagate Annotations')
        dialog.setMinimumWidth(300)
        layout = QVBoxLayout(dialog)
        range_group = QGroupBox('Frame Range')
        range_layout = QFormLayout(range_group)
        start_spin = QSpinBox()
        start_spin.setRange(0, self.total_frames - 1)
        start_spin.setValue(self.current_frame)
        end_spin = QSpinBox()
        end_spin.setRange(0, self.total_frames - 1)
        end_spin.setValue(min(self.current_frame + 10, self.total_frames - 1))
        range_layout.addRow('Start Frame:', start_spin)
        range_layout.addRow('End Frame:', end_spin)
        options_group = QGroupBox('Options')
        options_layout = QVBoxLayout(options_group)
        overwrite_check = QCheckBox('Overwrite existing annotations')
        overwrite_check.setChecked(False)
        smart_check = QCheckBox('Smart propagation (skip duplicate frames)')
        smart_check.setChecked(True)
        smart_check.setEnabled(self.duplicate_frames_enabled)
        options_layout.addWidget(overwrite_check)
        options_layout.addWidget(smart_check)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.
            Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(range_group)
        layout.addWidget(options_group)
        layout.addWidget(buttons)
        if dialog.exec_() == QDialog.Accepted:
            start_frame = start_spin.value()
            end_frame = end_spin.value()
            self.save_undo_state(range(start_frame, end_frame + 1))
            overwrite = overwrite_check.isChecked()
            smart = smart_check.isChecked() and self.duplicate_frames_enabled
            if start_frame > end_frame:
                start_frame, end_frame = end_frame, start_frame
            current_annotations = [self.clone_annotation(ann) for ann in
                self.canvas.annotations]
            progress = QDialog(self)
            progress.setWindowTitle('Propagating Annotations')
            progress.setFixedSize(300, 100)
            progress_layout = QVBoxLayout(progress)
            label = QLabel(
                f'Propagating annotations to frames {start_frame}-{end_frame}...'
                )
            progress_layout.addWidget(label)
            progress_bar = QProgressBar()
            progress_bar.setRange(start_frame, end_frame)
            progress_layout.addWidget(progress_bar)
            progress.setModal(False)
            progress.show()
            QApplication.processEvents()
            processed_hashes = set()
            for frame_num in range(start_frame, end_frame + 1):
                if frame_num == self.current_frame:
                    continue
                progress_bar.setValue(frame_num)
                if frame_num % 5 == 0:
                    QApplication.processEvents()
                if (not overwrite and frame_num in self.frame_annotations and
                    self.frame_annotations[frame_num]):
                    continue
                if smart and frame_num in self.frame_hashes:
                    frame_hash = self.frame_hashes[frame_num]
                    if frame_hash in processed_hashes:
                        continue
                    processed_hashes.add(frame_hash)
                if getattr(self, 'auto_blur_labels', False):
                    for ann in current_annotations:
                        self.blur_manager.add_bbox_region(frame_num, ann.rect, getattr(self.canvas, 'blur_kernel', 151))
                else:
                    self.frame_annotations[frame_num] = [self.clone_annotation(
                        ann) for ann in current_annotations]
            progress.close()
            if start_frame <= self.current_frame <= end_frame:
                if getattr(self, 'auto_blur_labels', False) and hasattr(self, '_refresh_blur_display'):
                    self._refresh_blur_display()
                else:
                    self.load_current_frame_annotations()
            self.statusBar.showMessage(
                f'Annotations propagated to frames {start_frame}-{end_frame}',
                5000)

    @log_exceptions
    def detect_similar_frames(self, reference_frame, similarity_threshold=0.9):
        """
        Detect frames that are similar to the reference frame.

        Args:
            reference_frame (int): The reference frame number
            similarity_threshold (float): Threshold for considering frames similar (0-1)

        Returns:
            list: List of frame numbers that are similar to the reference frame
        """
        if not self.cap or not self.cap.isOpened():
            return []
        current_pos = self.current_frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, reference_frame)
        ret, ref_frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
            return []
        ref_frame = self._process_frame_metadata(ref_frame, reference_frame)
        progress = QDialog(self)
        progress.setWindowTitle('Finding Similar Frames')
        progress.setFixedSize(300, 100)
        layout = QVBoxLayout(progress)
        label = QLabel('Scanning for similar frames...')
        layout.addWidget(label)
        progress_bar = QProgressBar()
        progress_bar.setRange(0, self.total_frames)
        layout.addWidget(progress_bar)
        progress.setModal(False)
        progress.show()
        QApplication.processEvents()
        similar_frames = [reference_frame]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        for frame_num in range(self.total_frames):
            if frame_num == reference_frame:
                continue
            progress_bar.setValue(frame_num)
            if frame_num % 10 == 0:
                QApplication.processEvents()
            ret, frame = self.cap.read()
            if not ret:
                break
            frame = self._process_frame_metadata(frame, frame_num)
            similarity = mse_similarity(ref_frame, frame)
            if similarity >= similarity_threshold:
                similar_frames.append(frame_num)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        progress.close()
        return similar_frames

    @log_exceptions
    def propagate_to_similar_frames(self):
        """Propagate current frame annotations to similar frames."""
        if not self.canvas.annotations:
            QMessageBox.warning(self, 'Propagate Annotations',
                'No annotations in current frame to propagate!')
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('Propagate to Similar Frames')
        dialog.setMinimumWidth(350)
        layout = QVBoxLayout(dialog)
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel('Similarity Threshold:')
        threshold_slider = QSlider(Qt.Horizontal)
        threshold_slider.setRange(50, 99)
        threshold_slider.setValue(90)
        threshold_value = QLabel('0.90')

        def update_threshold_label(value):
            threshold_value.setText(f'{value / 100:.2f}')
        threshold_slider.valueChanged.connect(update_threshold_label)
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(threshold_slider)
        threshold_layout.addWidget(threshold_value)
        options_group = QGroupBox('Options')
        options_layout = QVBoxLayout(options_group)
        overwrite_check = QCheckBox('Overwrite existing annotations')
        overwrite_check.setChecked(False)
        preview_check = QCheckBox('Preview similar frames before propagating')
        preview_check.setChecked(True)
        options_layout.addWidget(overwrite_check)
        options_layout.addWidget(preview_check)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.
            Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addLayout(threshold_layout)
        layout.addWidget(options_group)
        layout.addWidget(buttons)
        if dialog.exec_() == QDialog.Accepted:
            similarity_threshold = threshold_slider.value() / 100.0
            overwrite = overwrite_check.isChecked()
            preview = preview_check.isChecked()
            self.statusBar.showMessage(
                'Finding similar frames... This may take a moment.')
            QApplication.processEvents()
            similar_frames = self.detect_similar_frames(self.current_frame,
                similarity_threshold)
            self.save_undo_state(similar_frames)
            if len(similar_frames) <= 1:
                QMessageBox.information(self, 'No Similar Frames',
                    'No similar frames were found with the current threshold.')
                return
            if preview:
                preview_result = self.preview_similar_frames(similar_frames)
                if not preview_result:
                    return
            current_annotations = [self.clone_annotation(ann) for ann in
                self.canvas.annotations]
            propagated_count = 0
            for frame_num in similar_frames:
                if frame_num == self.current_frame:
                    continue
                if (not overwrite and frame_num in self.frame_annotations and
                    self.frame_annotations[frame_num]):
                    continue
                self.frame_annotations[frame_num] = [self.clone_annotation(
                    ann) for ann in current_annotations]
                propagated_count += 1
            self.statusBar.showMessage(
                f'Annotations propagated to {propagated_count} similar frames',
                5000)

    @log_exceptions
    def preview_similar_frames(self, frame_numbers):
        """
        Show a preview of similar frames and let the user select which ones to include.

        Args:
            frame_numbers (list): List of frame numbers to preview

        Returns:
            bool: True if user confirmed, False if cancelled
        """
        if not frame_numbers or len(frame_numbers) <= 1:
            return False
        dialog = QDialog(self)
        dialog.setWindowTitle('Preview Similar Frames')
        dialog.setMinimumWidth(600)
        dialog.setMinimumHeight(400)
        layout = QVBoxLayout(dialog)
        instructions = QLabel('Select frames to propagate annotations to:')
        layout.addWidget(instructions)
        frame_list = QListWidget()
        frame_list.setSelectionMode(QListWidget.MultiSelection)
        layout.addWidget(frame_list)
        current_pos = self.current_frame
        for frame_num in frame_numbers:
            if frame_num == self.current_frame:
                continue
            item = QListWidgetItem(f'Frame {frame_num}')
            item.setData(Qt.UserRole, frame_num)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = self.cap.read()
            if ret:
                frame = self._process_frame_metadata(frame, frame_num)
                thumbnail = create_thumbnail(frame, (160, 90))
                h, w, c = thumbnail.shape
                qimg = QImage(thumbnail.data, w, h, w * c, QImage.Format_RGB888
                    )
                pixmap = QPixmap.fromImage(qimg)
                item.setIcon(QIcon(pixmap))
            frame_list.addItem(item)
            item.setSelected(True)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
        button_layout = QHBoxLayout()
        select_all_btn = QPushButton('Select All')
        select_all_btn.clicked.connect(lambda : [frame_list.item(i).
            setSelected(True) for i in range(frame_list.count())])
        deselect_all_btn = QPushButton('Deselect All')
        deselect_all_btn.clicked.connect(lambda : [frame_list.item(i).
            setSelected(False) for i in range(frame_list.count())])
        button_layout.addWidget(select_all_btn)
        button_layout.addWidget(deselect_all_btn)
        layout.addLayout(button_layout)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.
            Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec_() == QDialog.Accepted:
            selected_frames = [self.current_frame]
            for i in range(frame_list.count()):
                item = frame_list.item(i)
                if item.isSelected():
                    selected_frames.append(item.data(Qt.UserRole))
            frame_numbers.clear()
            frame_numbers.extend(selected_frames)
            return True
        else:
            return False

    @log_exceptions
    def toggle_interpolation_mode(self):
        """Toggle interpolation mode on/off."""
        is_active = self.toggle_interpolation_action.isChecked()
        self.interpolation_manager.set_active(is_active)
        if hasattr(self, 'interpolation_toolbar'):
            self.interpolation_toolbar.setVisible(is_active)
        self.update_frame_display()

    @log_exceptions
    def set_interpolation_interval(self):
        """Open dialog to set interpolation interval."""
        dialog = QDialog(self)
        dialog.setWindowTitle('Set Keyframe Interval')
        layout = QVBoxLayout(dialog)
        form_layout = QFormLayout()
        interval_spinner = QSpinBox()
        interval_spinner.setRange(2, 100)
        interval_spinner.setValue(self.interpolation_manager.interval)
        form_layout.addRow('Frames between keyframes:', interval_spinner)
        layout.addLayout(form_layout)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.
            Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec_() == QDialog.Accepted:
            new_interval = interval_spinner.value()
            self.interpolation_manager.set_interval(new_interval)
            if hasattr(self, 'interval_spinner'):
                self.interval_spinner.setValue(new_interval)

    @log_exceptions
    def perform_interpolation(self):
        """Manually trigger interpolation between annotated frames."""
        if not self.interpolation_manager.is_active:
            QMessageBox.warning(self, 'Interpolation',
                'Interpolation mode is not active.')
            return
        self.interpolation_manager.perform_pending_interpolation()
        if hasattr(self, 'labeler_analytics'):
            self.labeler_analytics['tool_usage']['interpolation'] += 1

    @log_exceptions
    def update_frame_display(self):
        """Update the keyframe/interpolation indicator for the current frame.

        Uses the interpolation cycle (anchor/target) when active, so the
        indicator reflects the actual workflow state instead of a stale
        'last_annotated_frame' value.
        """
        if not (hasattr(self, 'interpolation_manager') and self.
            interpolation_manager.is_active):
            if hasattr(self, 'canvas'):
                self.canvas.setStyleSheet('')
            return
        has_annotations = self.current_frame in self.frame_annotations and len(
            self.frame_annotations[self.current_frame]) > 0
        is_keyframe = self.interpolation_manager.is_keyframe()
        if hasattr(self, 'keyframe_indicator'):
            if is_keyframe:
                self.keyframe_indicator.setStyleSheet(
                    'background-color: #FF5555; min-width: 16px;')
                self.keyframe_indicator.setToolTip(
                    'Current frame is a keyframe')
            elif has_annotations:
                self.keyframe_indicator.setStyleSheet(
                    'background-color: #55AAFF; min-width: 16px;')
                self.keyframe_indicator.setToolTip(
                    'Current frame has interpolated annotations')
            else:
                self.keyframe_indicator.setStyleSheet(
                    'background-color: transparent; min-width: 16px;')
                self.keyframe_indicator.setToolTip(
                    'Current frame has no annotations')
        if hasattr(self, 'canvas'):
            if is_keyframe:
                self.canvas.setStyleSheet('border: 2px solid #FF5555;')
            elif has_annotations:
                self.canvas.setStyleSheet('border: 2px solid #55AAFF;')
            else:
                self.canvas.setStyleSheet('')

    @log_exceptions
    def toggle_verification_mode(self):
        """Toggle verification mode for annotations."""
        if not hasattr(self, 'verification_mode_enabled'):
            self.verification_mode = False
        self.verification_mode = not self.verification_mode
        if self.verification_mode:
            self.statusBar.showMessage(
                'Verification mode enabled - unverified annotations will be deleted when changing frames'
                , 5000)
        else:
            self.statusBar.showMessage('Verification mode disabled', 3000)
        if hasattr(self, 'verify_mode_action'):
            self.verify_mode_action.setChecked(self.verification_mode)

    @log_exceptions
    def verify_selected_annotation(self):
        """Mark the selected annotation as verified."""
        if hasattr(self.canvas, 'selected_annotation'
            ) and self.canvas.selected_annotation:
            self.save_undo_state()
            self.canvas.selected_annotation.verified = True
            self.canvas.update()
            self.update_annotation_list()
            self.statusBar.showMessage('Annotation verified', 2000)

    @log_exceptions
    def verify_all_annotations(self):
        """Mark all annotations in the current frame as verified."""
        if self.canvas.annotations:
            self.save_undo_state()
            for annotation in self.canvas.annotations:
                annotation.verified = True
            self.canvas.update()
            self.update_annotation_list()
            self.statusBar.showMessage(
                f'All {len(self.canvas.annotations)} annotations verified',
                2000)

    @log_exceptions
    def handle_unverified_annotations(self):
        """Handle unverified annotations when changing frames.

        Only removes unverified annotations if self.remove_unverified is True
        (toggle, default OFF). When OFF, unverified labels are kept.
        """
        if not getattr(self, 'remove_unverified', False):
            return
        if self.current_frame in self.frame_annotations:
            unverified = [ann for ann in self.frame_annotations[self.
                current_frame] if not getattr(ann, 'verified', False)]
            for ann in unverified:
                if ann.source == 'interpolated':
                    ann.source = 'manual'
                    ann.verified = True
            if unverified:
                self.frame_annotations[self.current_frame] = [ann for ann in
                    self.frame_annotations[self.current_frame] if getattr(
                    ann, 'verified', False)]
                self.canvas.annotations = self.frame_annotations[self.
                    current_frame].copy()
                self.canvas.update()
                self.statusBar.showMessage(
                    f'Removed {len(unverified)} unverified annotations', 3000)

    @log_exceptions
    def change_style(self, style_name):
        """Change the application style."""
        if style_name in self.styles:
            self.styles[style_name]()
            self.current_style = style_name
            if style_name == 'Dark':
                self.icon_provider.set_theme('dark')
                self.refresh_icons()
                self.canvas.setStyleSheet('background-color: #151515;')
            elif style_name == 'Light':
                self.icon_provider.set_theme('light')
                self.refresh_icons()
                self.canvas.setStyleSheet('background-color: #FFFFFF;')
            elif style_name == 'Blue':
                self.icon_provider.set_theme('light')
                self.refresh_icons()
                self.canvas.setStyleSheet('background-color: #E5F0FF;')
            elif style_name == 'Green':
                self.icon_provider.set_theme('light')
                self.refresh_icons()
                self.canvas.setStyleSheet('background-color: #E5FFE5;')
            elif 'dark' in style_name:
                self.icon_provider.set_theme('dark')
                self.refresh_icons()
                self.canvas.setStyleSheet('')
            else:
                self.icon_provider.set_theme('light')
                self.refresh_icons()
                self.canvas.setStyleSheet('')
            if hasattr(self, 'annotation_dock'):
                if style_name == 'Dark':
                    self.annotation_dock.setStyleSheet(
                        """
                        QListWidget {
                            background-color: #252525;
                            color: #FFFFFF;
                            border: 1px solid #555555;
                        }
                    """
                        )
                else:
                    self.annotation_dock.setStyleSheet('')
            if hasattr(self, 'class_dock'):
                if style_name == 'Dark':
                    self.class_dock.setStyleSheet(
                        """
                        QListWidget {
                            background-color: #252525;
                            color: #FFFFFF;
                            border: 1px solid #555555;
                        }
                    """
                        )
                else:
                    self.class_dock.setStyleSheet('')
            self.statusBar.showMessage(f'Style changed to {style_name}')

    @log_exceptions
    def refresh_icons(self):
        """Refresh all icons in the UI to match the current theme."""
        if hasattr(self, 'play_button'):
            icon_name = ('media-playback-pause' if self.is_playing else
                'media-playback-start')
            self.play_button.setIcon(self.icon_provider.get_icon(icon_name))
        if hasattr(self, 'prev_button'):
            self.prev_button.setIcon(self.icon_provider.get_icon(
                'media-skip-backward'))
        if hasattr(self, 'next_button'):
            self.next_button.setIcon(self.icon_provider.get_icon(
                'media-skip-forward'))
        if hasattr(self, 'toolbar') and hasattr(self.toolbar, 'refresh_icons'):
            self.toolbar.refresh_icons()

    @log_exceptions
    def toggle_attribute_dialog(self):
        """Toggle automatic attribute dialog display."""
        self.auto_show_attribute_dialog = not self.auto_show_attribute_dialog
        self.statusBar.showMessage(
            f"Attribute dialog for new annotations {'enabled' if self.auto_show_attribute_dialog else 'disabled'}"
            , 3000)

    @log_exceptions
    def update_settings_menu_actions(self):
        """Update the settings menu actions to reflect current settings."""
        if not hasattr(self, 'settings_menu'):
            return
        for action in self.settings_menu.actions():
            if action.text() == 'Enable Auto-save':
                action.setChecked(self.autosave_enabled)
            elif action.text() == 'Show Attribute Dialog for New Annotations':
                action.setChecked(self.auto_show_attribute_dialog)
            elif action.text(
                ) == 'Use Previous Annotation Attributes as Default':
                action.setChecked(self.use_previous_attributes)

    @log_exceptions
    def toggle_previous_attributes(self):
        """Toggle using previous annotation attributes as default."""
        self.use_previous_attributes = not self.use_previous_attributes
        self.statusBar.showMessage(
            f"Using previous annotation attributes as default {'enabled' if self.use_previous_attributes else 'disabled'}"
            , 3000)

    @log_exceptions
    def toggle_autosave(self):
        """Toggle auto-save functionality."""
        self.autosave_enabled = not self.autosave_enabled
        if self.autosave_enabled:
            self.autosave_timer.start(self.autosave_interval)
            self.statusBar.showMessage('Auto-save enabled', 3000)
        else:
            self.autosave_timer.stop()
            self.statusBar.showMessage('Auto-save disabled', 3000)

    @log_exceptions
    def toggle_smart_edge(self):
        """Toggle smart edge movement functionality."""
        is_active = self.smart_edge_action.isChecked()
        self.canvas.smart_edge_enabled = is_active
        if is_active:
            self.statusBar.showMessage(
                'Smart Edge Movement enabled - edges will snap to image features'
                )
        else:
            self.statusBar.showMessage('Smart Edge Movement disabled')

    @log_exceptions
    def toggle_pan_mode(self):
        """Toggle pan mode for the canvas"""
        enabled = self.pan_tool_action.isChecked()
        self.canvas.set_pan_mode(enabled)
        if enabled:
            self.canvas.setCursor(Qt.OpenHandCursor)
            self.statusBar.showMessage(
                'Pan mode enabled. Left-click and drag to pan the canvas.',
                3000)
        else:
            self.canvas.setCursor(Qt.ArrowCursor)
            self.statusBar.showMessage('Pan mode disabled.', 3000)

    @log_exceptions
    def toggle_auto_bbox_mode(self):
        """Toggle Auto BBox mode using AI."""
        print(
            f'[DEBUG LOG] toggle_auto_bbox_mode called. auto_bbox_action isChecked: {self.auto_bbox_action.isChecked()}'
            )
        if not hasattr(self, 'auto_bbox_action'):
            return
        self.auto_bbox_mode = self.auto_bbox_action.isChecked()
        print(f'[DEBUG LOG] auto_bbox_mode set to: {self.auto_bbox_mode}')
        if self.auto_bbox_mode:
            model_type = getattr(self, 'sam_model_type', 'sam2.1_s.pt')
            print(f'[DEBUG LOG] Selected model type: {model_type}')
            manager = self.sam3_native_manager if 'sam3' in model_type.lower(
                ) else self.sam_manager
            print(f'[DEBUG LOG] Selected manager: {manager.__class__.__name__}'
                )
            if not manager.is_available():
                print(
                    f'[DEBUG LOG] Manager {manager.__class__.__name__} is NOT available.'
                    )
                QMessageBox.warning(self, 'Error',
                    f"{'SAM3 Native' if 'sam3' in model_type.lower() else 'Ultralytics'} package is not installed."
                    )
                self.action_sam_interactive.setChecked(False)
                return
            self.statusBar.showMessage('Loading SAM model... Please wait.')
            QApplication.setOverrideCursor(Qt.WaitCursor)
            print(f'[DEBUG LOG] Calling manager.load_model with: {model_type}')
            success, msg = manager.load_model(model_type)
            print(
                f'[DEBUG LOG] manager.load_model returned: success={success}, msg={msg}'
                )
            QApplication.restoreOverrideCursor()
            if not success:
                QMessageBox.warning(self, 'Model Error', msg)
                self.auto_bbox_mode = False
                self.auto_bbox_action.setChecked(False)
                self.canvas.setCursor(Qt.ArrowCursor)
        else:
            self.statusBar.showMessage('Auto BBox disabled.', 3000)
            self.canvas.setCursor(Qt.ArrowCursor)

    @log_exceptions
    def toggle_crop_mode(self):
        """Toggle Crop Mode."""
        if not hasattr(self, 'crop_mode_action'):
            return
            
        is_cropping = self.crop_mode_action.isChecked()
        self.canvas.is_cropping_mode = is_cropping
        
        if is_cropping:
            self.statusBar.showMessage('Crop Mode enabled. Draw a box to define the crop region.', 3000)
            self.canvas.setCursor(Qt.CrossCursor)
            if hasattr(self, 'crop_settings_dock'):
                if hasattr(self, 'annotation_dock') and hasattr(self, 'tabifyDockWidget'):
                    self.tabifyDockWidget(self.annotation_dock, self.crop_settings_dock)
                self.crop_settings_dock.show()
                self.crop_settings_dock.raise_()
        else:
            self.statusBar.showMessage('Crop Mode disabled.', 3000)
            self.canvas.setCursor(Qt.ArrowCursor)
            if hasattr(self, 'crop_settings_dock'):
                self.crop_settings_dock.hide()
        self.canvas.update()

    @log_exceptions
    def change_sam_model(self, model_display_name):
        """Change the active SAM model."""
        if 'sam2.1_s' in model_display_name:
            self.sam_model_type = 'sam2.1_s.pt'
        elif 'sam2.1_l' in model_display_name:
            self.sam_model_type = 'sam2.1_l.pt'
        elif 'sam3.1_s' in model_display_name:
            self.sam_model_type = 'sam3.1_s.pt'
        elif 'sam3.1_l' in model_display_name:
            self.sam_model_type = 'sam3.1_l.pt'
        if self.auto_bbox_mode:
            self.statusBar.showMessage(
                f'Loading model {self.sam_model_type}...', 3000)
            QApplication.processEvents()
            manager = (self.sam3_native_manager if 'sam3' in self.
                sam_model_type.lower() else self.sam_manager)
            success, msg = manager.load_model(self.sam_model_type)
            if not success:
                QMessageBox.warning(self, 'Model Error', msg)
            else:
                self.statusBar.showMessage(
                    f'Model {self.sam_model_type} loaded successfully.', 3000)

    @log_exceptions
    def set_autosave_interval(self, interval_ms):
        """Set the auto-save interval."""
        self.autosave_interval = interval_ms
        if self.autosave_enabled and self.autosave_timer.isActive():
            self.autosave_timer.stop()
            self.autosave_timer.start(self.autosave_interval)
        minutes = interval_ms / 60000
        self.statusBar.showMessage(
            f"Auto-save interval set to {minutes} minute{'s' if minutes != 1 else ''}"
            , 3000)

    def get_image_files_relative(self):
        """Get relative paths of image files, caching the result to avoid slow path calculations."""
        if not hasattr(self, 'image_files') or not self.image_files:
            return []
        if hasattr(self, '_cached_image_files'
            ) and self._cached_image_files is self.image_files and hasattr(self
            , '_cached_image_files_relative'):
            return self._cached_image_files_relative
        if hasattr(self, '_cached_image_files') and len(self.
            _cached_image_files) == len(self.image_files) and (len(self.
            image_files) == 0 or self._cached_image_files[0] == self.
            image_files[0]) and hasattr(self, '_cached_image_files_relative'):
            return self._cached_image_files_relative
        base_folder = os.path.dirname(self.image_files[0])
        relative_paths = [os.path.relpath(f, base_folder) for f in self.
            image_files]
        self._cached_image_files = self.image_files
        self._cached_image_files_relative = relative_paths
        return relative_paths

    @log_exceptions
    def perform_autosave(self):
        """Perform auto-save of the current project."""
        if not self.autosave_enabled:
            return
        if not hasattr(self, 'project_file') or not self.project_file:
            if not self.autosave_file:
                if hasattr(self, 'is_image_dataset'
                    ) and self.is_image_dataset and self.image_files:
                    image_folder = os.path.dirname(self.image_files[0])
                    folder_name = os.path.basename(image_folder)
                    self.autosave_file = os.path.join(image_folder,
                        f'{folder_name}_autosave.json')
                elif hasattr(self, 'video_filename') and self.video_filename:
                    video_base = os.path.dirname(self.video_filename)
                    video_name = os.path.splitext(os.path.basename(self.
                        video_filename))[0]
                    auto_save_folder = os.path.join(video_base, 'autosaves')
                    os.makedirs(auto_save_folder, exist_ok=True)
                    self.autosave_file = os.path.join(auto_save_folder, 
                        video_name + '_autosave.json')
                else:
                    return
        else:
            self.autosave_file = self.project_file
        if hasattr(self, '_autosave_thread'
            ) and self._autosave_thread.isRunning():
            return
        try:
            video_path = None
            image_dataset_info = None
            if hasattr(self, 'is_image_dataset') and self.is_image_dataset:
                if self.image_files:
                    base_folder = os.path.dirname(self.image_files[0])
                    image_dataset_info = {'is_image_dataset': True,
                        'base_folder': base_folder, 'image_files': self.
                        get_image_files_relative()}
            else:
                video_path = getattr(self, 'video_filename', None)
            class_attributes = getattr(self.canvas, 'class_attributes', {})
            frame_annotations_copy = {k: list(v) for k, v in self.
                frame_annotations.items()}
            deleted_annotations_copy = {k: list(v) for k, v in self.deleted_annotations.items()} if hasattr(self, 'deleted_annotations') else None
            blur_regions = self.blur_manager.to_dict() if hasattr(self, 'blur_manager') and self.blur_manager else None
            project_data_args = {'annotations': list(self.canvas.
                annotations), 'class_colors': dict(self.canvas.class_colors
                ), 'video_path': video_path, 'current_frame': self.
                current_frame, 'frame_annotations': frame_annotations_copy,
                'class_attributes': dict(class_attributes), 'current_style':
                self.current_style, 'auto_show_attribute_dialog': self.
                auto_show_attribute_dialog, 'use_previous_attributes': self
                .use_previous_attributes, 'duplicate_frames_enabled': self.
                duplicate_frames_enabled, 'frame_hashes': dict(self.
                frame_hashes) if hasattr(self, 'frame_hashes') else None,
                'duplicate_frames_cache': dict(self.duplicate_frames_cache) if
                hasattr(self, 'duplicate_frames_cache') else None,
                'image_dataset_info': image_dataset_info,
                'tracking_mode_enabled': getattr(self,
                'tracking_mode_enabled', False),
                'interpolation_mode_active': self.interpolation_manager.
                is_active if hasattr(self, 'interpolation_manager') else 
                False, 'verification_mode_enabled': getattr(self,
                'verification_mode', False), 'annotations_imported_list': 
                list(self._annotations_imported) if hasattr(self,
                '_annotations_imported') else [],
                'deleted_frames': self.deleted_frames if hasattr(self, 'deleted_frames') else set(),
                'labeler_analytics': self.labeler_analytics if hasattr(self, 'labeler_analytics') else None,
                'deleted_annotations': deleted_annotations_copy,
                'blur_regions': blur_regions}
            from viat.utils.task_runner import AutoSaveThread
            self._autosave_thread = AutoSaveThread(self.autosave_file,
                project_data_args, self)

            def on_autosave_finished(success, message):
                if success:
                    from PyQt5.QtCore import QDateTime
                    import os
                    self.last_autosave_time = QDateTime.currentDateTime()
                    self.statusBar.showMessage(
                        f'Auto-saved to {os.path.basename(self.autosave_file)}'
                        , 3000)
                else:
                    print(f'Auto-save failed: {message}')
            self._autosave_thread.finished_autosave.connect(
                on_autosave_finished)
            self._autosave_thread.start()
        except Exception as e:
            print(f'Auto-save preparation failed: {str(e)}')

    @log_exceptions
    def zoom_in(self):
        """Zoom in on the canvas."""
        self.zoom_level *= 1.2
        self.canvas.set_zoom(self.zoom_level)

    @log_exceptions
    def zoom_out(self):
        """Zoom out on the canvas."""
        self.zoom_level /= 1.2
        self.canvas.set_zoom(self.zoom_level)

    @log_exceptions
    def reset_zoom(self):
        """Reset zoom to default level."""
        self.zoom_level = 1.0
        self.canvas.set_zoom(self.zoom_level)
        self.canvas.reset_pan()

    @log_exceptions
    def auto_label(self):
        """Auto-label objects using zero-shot detection."""
        if not self.canvas.pixmap:
            QMessageBox.warning(self, 'Auto Annotate',
                'Please open a video or image first!')
            return
        dialog = AutoAnnotateDialog(self.current_frame, self.total_frames, self)
        if dialog.exec_():
            config = dialog.get_config()
            if config is None:
                return
            # If the user clicked "Test on Current Frame", force single-frame scope
            if dialog.is_test_mode():
                config['start_frame'] = self.current_frame
                config['end_frame']   = self.current_frame
            self.run_auto_label_dataset(config)

    @log_exceptions
    def run_auto_label_dataset(self, config):
        det_models = config.get('det_models', [])
        sam3_models = [m for m in det_models if 'sam3' in m.lower()]
        if sam3_models:
            other_models = [m for m in det_models if 'sam3' not in m.lower()]
            det_models = other_models + sam3_models
            config['det_models'] = det_models
            config['seg_model'] = sam3_models[-1]
        seg_model = config.get('seg_model')
        strategy = config.get('strategy', 'independent')
        start_frame = config.get('start_frame', 0)
        end_frame = config.get('end_frame', self.total_frames - 1)
        classes_config = config.get('classes_config', [])
        if not classes_config and 'existing_annotations' not in det_models:
            QMessageBox.warning(self, 'Auto Annotate',
                'Please configure at least one class.')
            return
        detect_classes = [c['name'] for c in classes_config if c['action'] ==
            'Detect (Zero-Shot)']
        
        classes_info = []
        for c in classes_config:
            if c['action'] == 'Detect (Zero-Shot)':
                prompt = c.get('extract_prompt', '').strip()
                if not prompt:
                    prompt = c['name']
                classes_info.append({'name': c['name'], 'prompt': prompt})

        helper_classes = [c for c in classes_config if 'Helper' in c['action']]
        remove_classes = [c['name'] for c in classes_config if 'Remove' in
            c['action']]
        config['classes'] = detect_classes
        config['classes_info'] = classes_info
        config['helper_classes'] = helper_classes
        config['remove_classes'] = remove_classes
        if remove_classes:
            for f in range(start_frame, end_frame + 1):
                if f in self.frame_annotations:
                    self.frame_annotations[f] = [ann for ann in self.
                        frame_annotations[f] if ann.class_name not in
                        remove_classes]
            self.canvas.update()
        self.save_undo_state(range(start_frame, end_frame + 1))
        config['existing_annotations_data'] = {}
        for f in range(start_frame, end_frame + 1):
            config['existing_annotations_data'][f
                ] = self.frame_annotations.get(f, [])
        self.last_auto_det_models = det_models
        self.last_auto_seg_model = seg_model
        if self.zero_shot_manager is None:
            self.zero_shot_manager = ZeroShotManager()
        if not hasattr(self, 'sam_manager') or self.sam_manager is None:
            self.sam_manager = SamManager()
        if not hasattr(self, 'sam3_native_manager'
            ) or self.sam3_native_manager is None:
            self.sam3_native_manager = Sam3NativeManager()
        if not hasattr(self, 'sam2_trt_manager') or self.sam2_trt_manager is None:
            self.sam2_trt_manager = Sam2TrtManager()
        if not self.zero_shot_manager.is_available():
            QMessageBox.warning(self, 'Auto Annotate Error',
                'Ultralytics package is missing. Please run: pip install ultralytics'
                )
            return
        total_to_process = end_frame - start_frame + 1
        from PyQt5.QtWidgets import QWidget, QHBoxLayout, QPushButton, QProgressBar
        self.auto_label_widget = QWidget()
        layout = QHBoxLayout(self.auto_label_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        self.auto_label_progress = QProgressBar(self)
        self.auto_label_progress.setRange(0, total_to_process)
        self.auto_label_progress.setValue(0)
        layout.addWidget(self.auto_label_progress)
        self.auto_label_cancel_btn = QPushButton('Cancel AI')
        self.auto_label_cancel_btn.clicked.connect(self.cancel_auto_label)
        layout.addWidget(self.auto_label_cancel_btn)
        self.statusBar.addPermanentWidget(self.auto_label_widget)
        self.statusBar.showMessage('Starting Background AI Annotator...')
        QApplication.processEvents()
        classes_added = False
        all_classes = detect_classes + [c['name'] for c in helper_classes]
        for c in all_classes:
            if c not in self.canvas.class_colors:
                import random
                from PyQt5.QtGui import QColor
                r = random.randint(0, 255)
                g = random.randint(0, 255)
                b = random.randint(0, 255)
                self.canvas.class_colors[c] = QColor(r, g, b)
                if not hasattr(self.canvas, 'class_attributes'
                    ) or self.canvas.class_attributes is None:
                    self.canvas.class_attributes = {}
                self.canvas.class_attributes[c] = {}
                classes_added = True
        if classes_added:
            self.class_attributes = self.canvas.class_attributes
            self.refresh_class_ui()
        if seg_model:
            if 'sam3' in seg_model.lower():
                self.statusBar.showMessage(
                    f'Loading SAM3 Native Refiner {seg_model}...')
                QApplication.processEvents()
                s_success, s_msg = self.sam3_native_manager.load_model(
                    seg_model)
                if not s_success:
                    QMessageBox.warning(self, 'Auto Annotate Error',
                        f'SAM3 Native Model failed: {s_msg}')
                    config['seg_model'] = None
                else:
                    active_sam_manager = self.sam3_native_manager
            elif 'trt' in seg_model.lower():
                self.statusBar.showMessage(
                    f'Loading SAM2 TRT Refiner {seg_model}...')
                QApplication.processEvents()
                s_success, s_msg = self.sam2_trt_manager.load_model(seg_model)
                if not s_success:
                    QMessageBox.warning(self, 'Auto Annotate Error',
                        f'SAM2 TRT Model failed: {s_msg}')
                    config['seg_model'] = None
                else:
                    active_sam_manager = self.sam2_trt_manager
            else:
                self.statusBar.showMessage(
                    f'Loading Segmentation Refiner {seg_model}...')
                QApplication.processEvents()
                s_success, s_msg = self.sam_manager.load_model(seg_model)
                if not s_success:
                    QMessageBox.warning(self, 'Auto Annotate Error',
                        f'SAM Model failed: {s_msg}')
                    config['seg_model'] = None
                else:
                    active_sam_manager = self.sam_manager
        else:
            active_sam_manager = self.sam_manager
            
        self.statusBar.showMessage(
            f'Running Background AI Annotator ({total_to_process} frames)...')
        self.auto_label_worker = AutoLabelWorker(self.config if hasattr(
            self, 'config') else config, hasattr(self, 'is_image_dataset') and
            self.is_image_dataset, getattr(self, 'image_files', []), self.
            video_filename, self.zero_shot_manager, active_sam_manager, self)
        self.auto_label_worker.config = config
        self.auto_label_worker.frame_started.connect(self.
            on_auto_label_frame_started)
        self.auto_label_worker.frame_processed.connect(self.
            on_auto_label_frame_processed)
        self.auto_label_worker.progress_updated.connect(self.
            auto_label_progress.setValue)
        self.auto_label_worker.error_occurred.connect(lambda e: QMessageBox
            .warning(self, 'AI Worker Error', e))
        self.auto_label_worker.finished_processing.connect(self.
            on_auto_label_finished)
        self.auto_label_worker.start()

    def cancel_auto_label(self):
        if hasattr(self, 'auto_label_worker'):
            self.auto_label_worker.cancel()
            self.statusBar.showMessage('Cancelling AI...', 3000)

    def on_auto_label_frame_started(self, f_idx):
        self.ai_processing_frame = f_idx
        if self.current_frame == f_idx:
            self.canvas.update()

    def on_auto_label_frame_processed(self, f_idx, annotations_list):
        if f_idx not in self.frame_annotations:
            self.frame_annotations[f_idx] = []
        threshold = 40
        save_seg = False
        if hasattr(self, 'auto_label_worker'
            ) and self.auto_label_worker and hasattr(self.auto_label_worker,
            'config'):
            threshold = self.auto_label_worker.config.get('threshold', 40)
            save_seg = self.auto_label_worker.config.get('save_segmentation',
                False)
        for ann_dict in annotations_list:
            box = ann_dict['box']
            rect = QRect(int(box[0]), int(box[1]), int(box[2] - box[0]),
                int(box[3] - box[1]))
            if getattr(self, 'auto_blur_labels', False):
                if hasattr(self, 'blur_manager') and self.blur_manager is not None:
                    self.blur_manager.add_bbox_region(f_idx, rect, getattr(self.canvas, 'blur_kernel', 151))
                continue

            color = self.canvas.class_colors.get(ann_dict['class_name'],
                QColor(0, 255, 0))
            original_ann_idx = ann_dict.get('original_ann_idx')
            original_ann = None
            if original_ann_idx is not None and original_ann_idx < len(self
                .frame_annotations[f_idx]):
                original_ann = self.frame_annotations[f_idx][original_ann_idx]
            if original_ann:
                # --- Helper refinement with rename (e.g. bm21 -> truck) ---
                # When the output class name differs from the source annotation
                # (a rename was configured, e.g. rename_to='truck'),
                # ALWAYS append a NEW annotation and leave the original UNTOUCHED.
                is_renamed_helper = (
                    ann_dict.get('source') == 'refined' and
                    ann_dict['class_name'] != original_ann.class_name
                )
                if is_renamed_helper:
                    # Keep original annotation (bm21) intact;
                    # add the renamed result (truck) as a brand-new annotation.
                    default_attributes = {'Size': -1, 'Quality': -1}
                    if hasattr(self, 'get_default_attributes_for_class'):
                        default_attributes = self.get_default_attributes_for_class(
                            ann_dict['class_name'])
                    new_color = self.canvas.class_colors.get(
                        ann_dict['class_name'], color)
                    annotation = BoundingBox(
                        rect=rect,
                        class_name=ann_dict['class_name'],
                        attributes=default_attributes,
                        color=new_color,
                        source=ann_dict['source'],
                        score=ann_dict['score'],
                        segmentation=ann_dict['segmentation'] if save_seg else None
                    )
                    annotation.verified = False
                    self.frame_annotations[f_idx].append(annotation)
                    # Register the new class if needed
                    if hasattr(self, 'project_classes') and ann_dict['class_name'] not in self.project_classes:
                        self.project_classes.append(ann_dict['class_name'])
                else:
                    # Same-class update: use dedup_iou threshold to decide
                    # in-place update (high overlap) vs new annotation (low overlap).
                    # Using dedup_iou (default 0.7) so the behaviour is consistent
                    # with the user-configured IOU setting, not the change-percent
                    # threshold which at value=1 would always create a new annotation.
                    iou_update_thresh = 0.7
                    if hasattr(self, 'auto_label_worker') and self.auto_label_worker:
                        iou_update_thresh = self.auto_label_worker.config.get('dedup_iou', 0.7)
                    orig_box = [original_ann.rect.x(), original_ann.rect.y(),
                        original_ann.rect.x() + original_ann.rect.width(),
                        original_ann.rect.y() + original_ann.rect.height()]
                    xA = max(box[0], orig_box[0])
                    yA = max(box[1], orig_box[1])
                    xB = min(box[2], orig_box[2])
                    yB = min(box[3], orig_box[3])
                    interArea = max(0, xB - xA) * max(0, yB - yA)
                    boxAArea = (box[2] - box[0]) * (box[3] - box[1])
                    origArea = (orig_box[2] - orig_box[0]) * (orig_box[3] - orig_box[1])
                    iou = interArea / float(boxAArea + origArea - interArea) \
                        if boxAArea + origArea - interArea > 0 else 0
                    if iou < iou_update_thresh:
                        # Low overlap → SAM3 found a meaningfully different region → new annotation
                        default_attributes = {'Size': -1, 'Quality': -1}
                        if hasattr(self, 'get_default_attributes_for_class'):
                            default_attributes = self.get_default_attributes_for_class(
                                ann_dict['class_name'])
                        annotation = BoundingBox(
                            rect=rect,
                            class_name=ann_dict['class_name'],
                            attributes=default_attributes,
                            color=color,
                            source=ann_dict['source'],
                            score=ann_dict['score'],
                            segmentation=ann_dict['segmentation'] if save_seg else None
                        )
                        annotation.verified = False
                        self.frame_annotations[f_idx].append(annotation)
                    else:
                        # High overlap → update original in-place (update box + segmentation)
                        original_ann.rect = rect
                        if ann_dict.get('segmentation'):
                            original_ann.segmentation = ann_dict['segmentation'] if save_seg else None
                        original_ann.source = ann_dict['source']
            else:
                default_attributes = {'Size': -1, 'Quality': -1}
                if hasattr(self, 'get_default_attributes_for_class'):
                    default_attributes = self.get_default_attributes_for_class(
                        ann_dict['class_name'])
                annotation = BoundingBox(rect=rect, class_name=ann_dict[
                    'class_name'], attributes=default_attributes, color=
                    color, source=ann_dict['source'], score=ann_dict[
                    'score'], segmentation=ann_dict['segmentation'] if
                    save_seg else None)
                self.frame_annotations[f_idx].append(annotation)
        if self.current_frame == f_idx:
            if getattr(self, 'auto_blur_labels', False) and hasattr(self, '_refresh_blur_display'):
                self._refresh_blur_display()
            else:
                self.canvas.update()

    def on_auto_label_finished(self):
        self.ai_processing_frame = -1
        self.statusBar.removeWidget(self.auto_label_widget)
        self.statusBar.showMessage('Auto-Labeling completed successfully.',
            5000)
        self.canvas.update()
        if hasattr(self, 'project_file') and self.project_file:
            self.save_project()
        self.update_frame_display()
        self.update_frame_annotations()

    @log_exceptions
    def track_objects(self):
        """Track objects across frames."""
        if not hasattr(self.canvas, 'pixmap'
            ) or not self.canvas.pixmap or not self.canvas.annotations:
            QMessageBox.warning(self, 'Track Objects',
                'Please open a video and create annotations first!')
            return
        if not hasattr(self.canvas, 'selected_annotation'
            ) or not self.canvas.selected_annotation:
            QMessageBox.warning(self, 'Track Objects',
                'Please select an annotation to track.')
            return
        target_ann = self.canvas.selected_annotation
        target_class = target_ann.class_name
        dialog = TrackingDialog(self, self.tracker_manager, self.
            current_frame, self.total_frames - 1, target_class)
        if dialog.exec_() != QDialog.Accepted:
            return
        tracker_name = dialog.selected_tracker_name
        end_frame = dialog.end_frame
        self.perform_tracking(target_ann, tracker_name, self.current_frame,
            end_frame)

    @log_exceptions
    def perform_tracking(self, target_ann, tracker_name, start_frame, end_frame
        ):
        if hasattr(self, 'labeler_analytics'):
            self.labeler_analytics['tool_usage']['tracking'] += 1
        self.save_undo_state(range(start_frame + 1, end_frame + 1))
        try:
            tracker = self.tracker_manager.create_tracker(tracker_name)
        except ValueError as e:
            QMessageBox.critical(self, 'Tracker Error', str(e))
            return
        if hasattr(self, 'is_image_dataset') and self.is_image_dataset:
            image_path = self.image_files[start_frame]
            frame = cv2.imread(image_path)
            ret = frame is not None
        else:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            ret, frame = self.cap.read()
            if ret:
                frame = self._process_frame_metadata(frame, start_frame)
        if not ret:
            QMessageBox.critical(self, 'Tracking Error',
                f'Failed to read start frame {start_frame}')
            return
        bbox = target_ann.rect.x(), target_ann.rect.y(), target_ann.rect.width(
            ), target_ann.rect.height()
        if not tracker.init(frame, bbox):
            QMessageBox.critical(self, 'Tracking Error',
                'Failed to initialize tracker on the selected bounding box.')
            return
        from PyQt5.QtWidgets import QProgressDialog
        progress = QProgressDialog('Tracking object...', 'Cancel', 
            start_frame + 1, end_frame, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.show()
        last_successful_frame = start_frame
        f_idx = start_frame
        success = True
        for f_idx in range(start_frame + 1, end_frame + 1):
            if progress.wasCanceled():
                break
            progress.setValue(f_idx)
            QApplication.processEvents()
            if hasattr(self, 'is_image_dataset') and self.is_image_dataset:
                image_path = self.image_files[f_idx]
                frame = cv2.imread(image_path)
                ret = frame is not None
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = self.cap.read()
                if ret:
                    frame = self._process_frame_metadata(frame, f_idx)
            if not ret:
                break
            success, new_bbox = tracker.update(frame)
            if not success:
                QMessageBox.information(self, 'Tracking Lost',
                    f'Tracking lost at frame {f_idx}. Please adjust the bounding box and resume tracking.'
                    )
                break
            x, y, w, h = map(int, new_bbox)
            x = max(0, min(x, frame.shape[1] - 1))
            y = max(0, min(y, frame.shape[0] - 1))
            w = max(1, min(w, frame.shape[1] - x))
            h = max(1, min(h, frame.shape[0] - y))
            new_rect = QRect(x, y, w, h)
            default_attributes = target_ann.attributes.copy(
                ) if target_ann.attributes else {}
            if not default_attributes and hasattr(self,
                'get_default_attributes_for_class'):
                default_attributes = self.get_default_attributes_for_class(
                    target_ann.class_name)
            new_ann = BoundingBox(rect=new_rect, class_name=target_ann.
                class_name, attributes=default_attributes, color=target_ann
                .color, source='tracked')
            if f_idx not in self.frame_annotations:
                self.frame_annotations[f_idx] = []
            self.frame_annotations[f_idx].append(new_ann)
            last_successful_frame = f_idx
        progress.setValue(end_frame)
        target_nav_frame = f_idx if not success else last_successful_frame
        if hasattr(self, 'is_image_dataset') and self.is_image_dataset:
            self.current_frame = target_nav_frame
            self.frame_slider.blockSignals(True)
            self.frame_slider.setValue(self.current_frame)
            self.frame_slider.blockSignals(False)
            self.load_current_image()
        else:
            self.seek_to_frame(target_nav_frame)
        self.update_frame_annotations()

    @log_exceptions
    def change_annotation_method(self, method_name):
        """Change the current annotation method."""
        if method_name in ['Drag', 'TwoClick']:
            self.canvas.set_annotation_method(method_name)
            if method_name == 'TwoClick':
                self.statusBar.showMessage(
                    'Two-click mode: Click first corner, then click second corner to create box. Press ESC to cancel.'
                    )
            else:
                self.statusBar.showMessage(
                    f'Annotation method changed to {method_name}')

    def _copy_annotations_state(self, modified_frames=None):
        """Helper to create a copy of all frame annotations, optimizing unchanged frames."""
        if modified_frames is None:
            frames_to_clone = {self.current_frame}
            if getattr(self, 'duplicate_frames_enabled', False) and hasattr(
                self, 'frame_hashes'
                ) and self.current_frame in self.frame_hashes:
                current_hash = self.frame_hashes[self.current_frame]
                if hasattr(self, 'duplicate_frames_cache'
                    ) and current_hash in self.duplicate_frames_cache:
                    frames_to_clone.update(self.duplicate_frames_cache[
                        current_hash])
        elif modified_frames == 'all':
            frames_to_clone = None
        else:
            try:
                frames_to_clone = set(modified_frames)
            except TypeError:
                frames_to_clone = {modified_frames}
        all_frame_annotations = {}
        for frame_num, annotations in self.frame_annotations.items():
            if frames_to_clone is None or frame_num in frames_to_clone:
                all_frame_annotations[frame_num] = [self.clone_annotation(
                    ann) for ann in annotations]
            else:
                all_frame_annotations[frame_num] = list(annotations)
        return all_frame_annotations

    @log_exceptions
    def save_undo_state(self, modified_frames=None, **extra_state):
        """Save the current state for undo functionality."""
        all_frame_annotations = self._copy_annotations_state(modified_frames)
        class_colors = {}
        for class_name, color in self.canvas.class_colors.items():
            class_colors[class_name] = QColor(color)
        class_attributes = None
        if hasattr(self.canvas, 'class_attributes'):
            class_attributes = deepcopy(self.canvas.class_attributes)
        blur_regions = self.blur_manager.to_dict() if hasattr(self, 'blur_manager') and self.blur_manager else None
        undo_state = {'frame': self.current_frame, 'all_annotations':
            all_frame_annotations, 'current_annotations': [self.
            clone_annotation(ann) for ann in self.canvas.annotations] if
            self.canvas.annotations else [], 'class_colors': class_colors,
            'class_attributes': class_attributes, 'current_class': self.
            canvas.current_class if hasattr(self.canvas, 'current_class') else
            None, 'blur_regions': blur_regions}
        undo_state.update(extra_state)
        self.undo_stack.append(undo_state)
        self.redo_stack.clear()
        if len(self.undo_stack) > self.max_undo_steps:
            self.undo_stack.pop(0)

    @log_exceptions
    def save_undo_state_without_clearing_redo(self, modified_frames=None, **extra_state):
        """Save the current state for undo functionality without clearing the redo stack."""
        all_frame_annotations = self._copy_annotations_state(modified_frames)
        class_colors = {}
        for class_name, color in self.canvas.class_colors.items():
            class_colors[class_name] = QColor(color)
        class_attributes = None
        if hasattr(self.canvas, 'class_attributes'):
            class_attributes = deepcopy(self.canvas.class_attributes)
        blur_regions = self.blur_manager.to_dict() if hasattr(self, 'blur_manager') and self.blur_manager else None
        undo_state = {'frame': self.current_frame, 'all_annotations':
            all_frame_annotations, 'current_annotations': [self.
            clone_annotation(ann) for ann in self.canvas.annotations] if
            self.canvas.annotations else [], 'class_colors': class_colors,
            'class_attributes': class_attributes, 'current_class': self.
            canvas.current_class if hasattr(self.canvas, 'current_class') else
            None, 'blur_regions': blur_regions}
        undo_state.update(extra_state)
        self.undo_stack.append(undo_state)
        if len(self.undo_stack) > self.max_undo_steps:
            self.undo_stack.pop(0)

    @log_exceptions
    def undo(self):
        """Undo the last annotation or class change."""
        # If in SAM interactive mode and there are prompt points/box, undo the last prompt first
        if getattr(getattr(self, 'canvas', None), 'sam_interactive_mode', False):
            has_prompt_pts = bool(getattr(self.canvas, 'sam_prompt_points', None))
            has_prompt_box = getattr(self.canvas, 'sam_prompt_box', None) is not None
            if has_prompt_pts or has_prompt_box:
                if has_prompt_pts:
                    self.canvas.sam_prompt_points.pop()
                    if hasattr(self.canvas, 'sam_prompt_labels') and self.canvas.sam_prompt_labels:
                        self.canvas.sam_prompt_labels.pop()
                elif has_prompt_box:
                    self.canvas.sam_prompt_box = None
                
                num_pos = sum(1 for l in getattr(self.canvas, 'sam_prompt_labels', []) if l == 1)
                num_neg = sum(1 for l in getattr(self.canvas, 'sam_prompt_labels', []) if l == 0)
                has_b = getattr(self.canvas, 'sam_prompt_box', None) is not None
                if hasattr(self, 'sam_interactive_dock'):
                    self.sam_interactive_dock.update_status(num_pos, num_neg, has_b)
                if not num_pos and not num_neg and not has_b:
                    self.canvas.sam_preview_polygon = None
                    self.canvas.sam_preview_rect = None
                self.canvas.update()
                self.statusBar.showMessage('Prompt undone.', 2000)
                return

        if not self.undo_stack:
            self.statusBar.showMessage('Nothing to undo', 3000)
            return
        blur_regions = self.blur_manager.to_dict() if hasattr(self, 'blur_manager') and self.blur_manager else None
        current_state = {'frame': self.current_frame, 'all_annotations':
            self._copy_annotations_state(), 'current_annotations': [self.
            clone_annotation(ann) for ann in self.canvas.annotations] if
            self.canvas.annotations else [], 'class_colors': {class_name:
            QColor(color) for class_name, color in self.canvas.class_colors
            .items()}, 'class_attributes': deepcopy(self.canvas.
            class_attributes) if hasattr(self.canvas, 'class_attributes') else
            None, 'current_class': self.canvas.current_class if hasattr(
            self.canvas, 'current_class') else None, 'blur_regions': blur_regions}
        self.redo_stack.append(current_state)
        if len(self.redo_stack) > self.max_redo_steps:
            self.redo_stack.pop(0)
        last_state = self.undo_stack.pop()
        frame = last_state['frame']
        if 'blur_regions' in last_state and hasattr(self, 'blur_manager') and self.blur_manager:
            self.blur_manager.from_dict(last_state['blur_regions'])
            if hasattr(self, '_refresh_blur_display'):
                self._refresh_blur_display()
        if 'class_colors' in last_state and last_state['class_colors']:
            self.canvas.class_colors = last_state['class_colors']
        if 'class_attributes' in last_state and last_state['class_attributes']:
            self.canvas.class_attributes = last_state['class_attributes']
        if 'current_class' in last_state and last_state['current_class']:
            self.canvas.current_class = last_state['current_class']
        if 'all_annotations' in last_state and last_state['all_annotations']:
            self.frame_annotations = last_state['all_annotations']
        if frame == self.current_frame:
            if 'current_annotations' in last_state:
                self.canvas.annotations = last_state['current_annotations']
            else:
                self.canvas.annotations = self.frame_annotations.get(frame, [])
            self.canvas.selected_annotation = None
            self.canvas.update()
            self.update_annotation_list()
            self.refresh_class_ui()
            self.statusBar.showMessage('Undo successful', 3000)
        else:
            self.statusBar.showMessage(
                f'Undo refers to frame {frame}, navigating there first', 3000)
            if hasattr(self, 'is_image_dataset') and self.is_image_dataset:
                self.current_frame = frame
                self.frame_slider.setValue(frame)
                self.load_current_image()
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ret, frame_img = self.cap.read()
                if ret:
                    frame_img = self._process_frame_metadata(frame_img, frame)
                    self.current_frame = frame
                    self.frame_slider.setValue(frame)
                    self.canvas.set_frame(frame_img)
            if 'current_annotations' in last_state:
                self.canvas.annotations = last_state['current_annotations']
            else:
                self.canvas.annotations = self.frame_annotations.get(frame, [])
            self.canvas.selected_annotation = None
            self.canvas.update()
            self.update_annotation_list()
            self.refresh_class_ui()
            self.statusBar.showMessage('Undo successful', 3000)

        if last_state.get('action_type') == 'blur_and_delete_class':
            class_name = last_state.get('action_class', '')
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(
                self,
                "Undo Class Blur & Delete",
                f"Warning: Undo has restored the class '{class_name}' and its bounding boxes, and reverted the blurred regions."
            )

    @log_exceptions
    def redo(self):
        """Redo the last undone action."""
        if not self.redo_stack:
            self.statusBar.showMessage('Nothing to redo', 3000)
            return
        self.save_undo_state_without_clearing_redo()
        redo_state = self.redo_stack.pop()
        frame = redo_state['frame']
        if 'blur_regions' in redo_state and hasattr(self, 'blur_manager') and self.blur_manager:
            self.blur_manager.from_dict(redo_state['blur_regions'])
            if hasattr(self, '_refresh_blur_display'):
                self._refresh_blur_display()
        if 'class_colors' in redo_state and redo_state['class_colors']:
            self.canvas.class_colors = redo_state['class_colors']
        if 'class_attributes' in redo_state and redo_state['class_attributes']:
            self.canvas.class_attributes = redo_state['class_attributes']
        if 'current_class' in redo_state and redo_state['current_class']:
            self.canvas.current_class = redo_state['current_class']
        if 'all_annotations' in redo_state and redo_state['all_annotations']:
            self.frame_annotations = redo_state['all_annotations']
        if frame == self.current_frame:
            if 'current_annotations' in redo_state:
                self.canvas.annotations = redo_state['current_annotations']
            else:
                self.canvas.annotations = self.frame_annotations.get(frame, [])
            self.canvas.selected_annotation = None
            self.canvas.update()
            self.update_annotation_list()
            self.refresh_class_ui()
            self.statusBar.showMessage('Redo successful', 3000)
        else:
            self.statusBar.showMessage(
                f'Redo refers to frame {frame}, navigating there first', 3000)
            if hasattr(self, 'is_image_dataset') and self.is_image_dataset:
                self.current_frame = frame
                self.frame_slider.setValue(frame)
                self.load_current_image()
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
                ret, frame_img = self.cap.read()
                if ret:
                    frame_img = self._process_frame_metadata(frame_img, frame)
                    self.current_frame = frame
                    self.frame_slider.setValue(frame)
                    self.canvas.set_frame(frame_img)
            if 'current_annotations' in redo_state:
                self.canvas.annotations = redo_state['current_annotations']
            else:
                self.canvas.annotations = self.frame_annotations.get(frame, [])
            self.canvas.selected_annotation = None
            self.canvas.update()
            self.update_annotation_list()
            self.refresh_class_ui()
            self.statusBar.showMessage('Redo successful', 3000)

    @log_exceptions
    def eventFilter(self, obj, event):
        """Global event filter for frame-navigation shortcuts.

        Arrow keys (without Ctrl) step frames; Space toggles play/pause;
        Tab cycles annotation selection when the canvas has focus. These
        are handled HERE and only here, so there is a single owner of
        frame navigation and no double-stepping between the event filter
        and keyPressEvent.
        """
        if event.type() == QEvent.KeyPress:
            from PyQt5.QtWidgets import QLineEdit, QPlainTextEdit, QTextEdit
            focused = QApplication.focusWidget()
            typing = isinstance(focused, (QLineEdit, QPlainTextEdit, QTextEdit))
            mods = QApplication.keyboardModifiers()
            if event.key(
                ) == Qt.Key_Right and not typing and not mods & Qt.ControlModifier:
                self.next_frame()
                return True
            if event.key(
                ) == Qt.Key_Left and not typing and not mods & Qt.ControlModifier:
                self.prev_frame()
                return True
            if event.key() == Qt.Key_Space and not typing:
                self.play_pause_video()
                return True
            if event.key() == Qt.Key_Tab:
                if isinstance(focused, VideoCanvas):
                    self.cycle_annotation_selection()
                    event.accept()
                    return True
            if event.key() in (Qt.Key_Delete, Qt.Key_Backspace) and not typing:
                if hasattr(self, 'canvas'):
                    if hasattr(self.canvas, 'selected_annotations'
                        ) and self.canvas.selected_annotations:
                        self.delete_selected_annotations()
                        return True
                    elif hasattr(self.canvas, 'selected_annotation'
                        ) and self.canvas.selected_annotation:
                        self.delete_selected_annotation()
                        return True

            # SAM Interactive Dock Shortcuts (Z: Preview, X: Clear Prompts, C: Execute)
            sam_active = (hasattr(self, 'sam_interactive_dock') and self.sam_interactive_dock.isVisible()) or getattr(getattr(self, 'canvas', None), 'sam_interactive_mode', False)
            if sam_active and not typing:
                if event.key() == Qt.Key_Z and not (mods & Qt.ControlModifier):
                    self.sam_interactive_dock.btn_preview.click()
                    return True
                if event.key() == Qt.Key_X and not (mods & (Qt.ControlModifier | Qt.ShiftModifier)):
                    self.sam_interactive_dock.btn_clear.click()
                    return True
                if event.key() == Qt.Key_C and not (mods & Qt.ControlModifier):
                    self.sam_interactive_dock.btn_track.click()
                    return True
        return super().eventFilter(obj, event)

    @log_exceptions
    def delete_current_frame(self):
        """Toggle frame deletion status (mark as REMOVED or RESTORE)."""
        if not hasattr(self, 'deleted_frames'):
            self.deleted_frames = set()
        if not hasattr(self, 'deleted_annotations'):
            self.deleted_annotations = {}

        if self.current_frame in self.deleted_frames:
            # UN-DELETE / RESTORE FRAME
            self.deleted_frames.remove(self.current_frame)
            if self.current_frame in self.deleted_annotations:
                self.frame_annotations[self.current_frame] = self.deleted_annotations.pop(self.current_frame)
                self.canvas.annotations = list(self.frame_annotations[self.current_frame])
            self.statusBar.showMessage(f'Frame {self.current_frame} RESTORED to dataset.', 4000)
            self.update_frame_display()
            self.canvas.update()
        else:
            # MARK FRAME AS REMOVED
            self.deleted_frames.add(self.current_frame)
            if self.current_frame in self.frame_annotations:
                self.deleted_annotations[self.current_frame] = self.frame_annotations[self.current_frame]
                del self.frame_annotations[self.current_frame]
            self.canvas.annotations.clear()
            self.statusBar.showMessage(f'Frame {self.current_frame} marked as REMOVED (Press Shift+X to restore).', 4000)
            self.update_frame_display()
            self.canvas.update()
            # Auto-advance to the next frame
            self.next_frame()

    @log_exceptions
    def keyPressEvent(self, event):
        """Handle keyboard shortcuts that are NOT frame navigation.
        Arrow keys are handled globally in eventFilter so there is a
        single owner of frame stepping (no double jumps)."""
        if event.key() == Qt.Key_M:
            current_index = self.method_selector.currentIndex()
            new_index = (current_index + 1) % self.method_selector.count()
            self.method_selector.setCurrentIndex(new_index)
            return
        if event.key() == Qt.Key_B:
            if hasattr(self, 'annotation_dock'):
                self.annotation_dock.batch_edit_annotations()
            return
        if event.key() == Qt.Key_P and event.modifiers() & Qt.ControlModifier:
            self.propagate_annotations()
            return
        if event.key() == Qt.Key_X and event.modifiers() & Qt.ShiftModifier:
            self.delete_current_frame()
            return
        if event.key() == Qt.Key_C and event.modifiers() & Qt.ControlModifier:
            self.copy_selected_annotation()
            return
        if event.key() == Qt.Key_V and event.modifiers() & Qt.ControlModifier:
            self.paste_annotation()
            return
        if event.key() == Qt.Key_X and event.modifiers() & Qt.ControlModifier:
            self.cut_selected_annotation()
            return
        if event.key() == Qt.Key_A and event.modifiers() & Qt.ControlModifier:
            self.select_all_annotations()
            return
        if event.key() == Qt.Key_G and event.modifiers() & Qt.ControlModifier:
            self.go_to_frame_dialog()
            return
        if event.key() == Qt.Key_Z and event.modifiers(
            ) & Qt.ControlModifier and not event.modifiers(
            ) & Qt.ShiftModifier:
            self.undo()
            return
        if event.key() == Qt.Key_Y and event.modifiers(
            ) & Qt.ControlModifier or event.key(
            ) == Qt.Key_Z and event.modifiers(
            ) & Qt.ControlModifier and event.modifiers() & Qt.ShiftModifier:
            self.redo()
            return
        super().keyPressEvent(event)

    def check_unsaved_changes(self):
        """Prompt user to save if project is modified. Returns False if cancelled."""
        if getattr(self, 'project_modified', False):
            reply = QMessageBox.question(self, 'Save Project',
                'The project has been modified. Do you want to save changes?',
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel,
                QMessageBox.Save)
            if reply == QMessageBox.Save:
                self.save_project()
                return True
            elif reply == QMessageBox.Cancel:
                return False
            else:
                return True
        return True

    @log_exceptions
    def closeEvent(self, event):
        """Handle application close event."""
        if getattr(self, 'auto_save_blur_on_switch', False) and getattr(self, 'is_image_dataset', False):
            self._handle_auto_save_blur_on_frame_change(getattr(self, 'current_frame', -1))
        if not self.check_unsaved_changes():
            event.ignore()
            return
        event.accept()
        self.save_application_state()
        if self.autosave_enabled:
            if hasattr(self, 'project_file') and self.project_file or hasattr(
                self, 'is_image_dataset') and self.is_image_dataset or hasattr(
                self, 'video_filename') and self.video_filename:
                self.perform_autosave()
            self.perform_autosave()

    def _viat_ensure_dataset(self, show_warning=True):
        """Return the loaded DatasetInfo or show a warning."""
        info = getattr(self, '_viat_dataset_info', None)
        if info is None or not getattr(self, 'is_image_dataset', False):
            if show_warning:
                QMessageBox.warning(self, 'Dataset Operation',
                    'Open an image dataset first (File > Open Image Folder).')
            return None
        return info

    def viat_current_split(self):
        """Return the split name of the current frame, or 'root'."""
        f2s = getattr(self, '_viat_frame_to_split', None)
        if f2s and 0 <= self.current_frame < len(f2s):
            return f2s[self.current_frame]
        return 'root'

    @log_exceptions
    def remove_bad_frame(self):
        """Discard the current frame: move image + label to discarded/."""
        if not self._viat_ensure_dataset():
            return
        if not 0 <= self.current_frame < self.total_frames:
            return
        fname = os.path.basename(self.image_files[self.current_frame])
        reply = QMessageBox.question(self, 'Remove Bad Frame',
            f"""Move this frame to the 'discarded' folder?

  {fname} (split: {self.viat_current_split()})

Image + label will be moved on disk (reversible)."""
            , QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        result = _viat_remove_bad_frames(self, [self.current_frame])
        if self.current_frame >= self.total_frames:
            self.current_frame = max(0, self.total_frames - 1)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.frame_slider.setValue(self.current_frame)
        self.frame_slider.blockSignals(False)
        self.load_current_image()
        self.update_frame_info()
        self.update_annotation_list()
        self.statusBar.showMessage(
            f"Discarded 1 frame -> {result['discarded_dir']} ({result['moved_images']} img, {result['moved_labels']} lbl)"
            , 5000)

    @log_exceptions
    def remove_bad_frames_dialog(self):
        """Discard a range of frames (batch)."""
        if not self._viat_ensure_dataset():
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('Remove Bad Frames (batch)')
        dialog.setMinimumWidth(360)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        start_spin = QSpinBox()
        start_spin.setRange(0, max(0, self.total_frames - 1))
        start_spin.setValue(self.current_frame)
        end_spin = QSpinBox()
        end_spin.setRange(0, max(0, self.total_frames - 1))
        end_spin.setValue(self.current_frame)
        form.addRow('Start frame:', start_spin)
        form.addRow('End frame:', end_spin)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.
            Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return
        lo, hi = sorted([start_spin.value(), end_spin.value()])
        frames = list(range(lo, hi + 1))
        reply = QMessageBox.question(self, 'Remove Bad Frames',
            f'Move {len(frames)} frames ({lo}..{hi}) to discarded/?', 
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        result = _viat_remove_bad_frames(self, frames)
        if self.current_frame >= self.total_frames:
            self.current_frame = max(0, self.total_frames - 1)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.frame_slider.setValue(self.current_frame)
        self.frame_slider.blockSignals(False)
        self.load_current_image()
        self.update_frame_info()
        self.update_annotation_list()
        self.statusBar.showMessage(
            f"Discarded {result['moved_images']} images / {result['moved_labels']} labels -> {result['discarded_dir']}"
            , 5000)

    @log_exceptions
    def remap_class_dialog(self):
        """Rename a class everywhere (in-memory + on-disk labels)."""
        if not self._viat_ensure_dataset():
            return
        classes = sorted(getattr(self.canvas, 'class_colors', {}).keys())
        if not classes:
            QMessageBox.information(self, 'Remap Class', 'No classes loaded.')
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('Remap Class')
        dialog.setMinimumWidth(380)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        old_combo = QComboBox()
        old_combo.addItems(classes)
        new_edit = QLineEdit()
        new_edit.setPlaceholderText('new class name')
        rewrite_check = QCheckBox('Also rewrite label files on disk')
        rewrite_check.setChecked(True)
        form.addRow('Old class:', old_combo)
        form.addRow('New name:', new_edit)
        form.addRow('', rewrite_check)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.
            Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return
        old = old_combo.currentText()
        new = new_edit.text().strip()
        if not new or old == new:
            return
        result = _viat_remap_class(self, old, new, rewrite_disk=
            rewrite_check.isChecked())
        self.refresh_class_ui()
        self.update_annotation_list()
        self.statusBar.showMessage(
            f"Remapped '{old}' -> '{new}': {result['changed_boxes']} boxes in {result['changed_frames']} frames, {result['disk_files_rewritten']} files rewritten."
            , 6000)

    @log_exceptions
    def merge_classes_dialog(self):
        """Merge several classes into one (e.g. car+truck -> vehicle)."""
        if not self._viat_ensure_dataset():
            return
        classes = sorted(getattr(self.canvas, 'class_colors', {}).keys())
        if len(classes) < 2:
            QMessageBox.information(self, 'Merge Classes',
                'Need at least 2 classes.')
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('Merge Classes')
        dialog.setMinimumWidth(420)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel('Select classes to merge INTO a single class:')
            )
        checks = []
        for c in classes:
            cb = QCheckBox(c)
            checks.append((c, cb))
            layout.addWidget(cb)
        form = QFormLayout()
        new_edit = QLineEdit()
        new_edit.setPlaceholderText('target class name')
        form.addRow('Merge into:', new_edit)
        rewrite_check = QCheckBox('Also rewrite label files on disk')
        rewrite_check.setChecked(True)
        form.addRow('', rewrite_check)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.
            Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return
        old_names = [c for c, cb in checks if cb.isChecked()]
        new = new_edit.text().strip()
        if not new or not old_names:
            return
        if new in old_names:
            old_names = [c for c in old_names if c != new]
        result = _viat_merge_classes(self, old_names, new, rewrite_disk=
            rewrite_check.isChecked())
        self.refresh_class_ui()
        self.update_annotation_list()
        self.statusBar.showMessage(
            f"Merged {len(old_names)} classes into '{new}': {result['changed_boxes']} boxes, {result['disk_files_rewritten']} files rewritten."
            , 6000)

    @log_exceptions
    def viat_dataset_stats(self):
        """Show a quick dataset statistics dialog."""
        info = self._viat_ensure_dataset()
        if not info:
            return
        per_split = {}
        per_class = {}
        f2s = getattr(self, '_viat_frame_to_split', []) or []
        for fidx, anns in self.frame_annotations.items():
            sp = f2s[fidx] if fidx < len(f2s) else 'root'
            per_split[sp] = per_split.get(sp, 0) + 1
            for a in anns:
                per_class[a.class_name] = per_class.get(a.class_name, 0) + 1
        lines = [f'Dataset: {os.path.basename(info.root)}',
            f'Layout: {info.layout}', '']
        lines.append('Splits (annotated frames):')
        for s in info.splits:
            lines.append(
                f'  {s.name}: {per_split.get(s.name, 0)}/{len(s.images)} frames, format={s.label_format}'
                )
        lines.append('')
        lines.append('Classes (box counts):')
        for c, n in sorted(per_class.items(), key=lambda x: -x[1]):
            lines.append(f'  {c}: {n}')
        lines.append('')
        lines.append(f'Classes source: {info.classes_source}')
        if info.classes_conflict:
            lines.append(f'âš  {info.classes_conflict}')
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle('Dataset Statistics')
        msg_box.setText('\n'.join(lines))
        msg_box.setIcon(QMessageBox.Information)
        plot_btn = msg_box.addButton('Plot Details', QMessageBox.ActionRole)
        msg_box.addButton(QMessageBox.Ok)
        msg_box.exec_()
        if msg_box.clickedButton() == plot_btn:
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                classes = list(per_class.keys())
                counts = list(per_class.values())
                plt.bar(classes, counts, color='skyblue')
                plt.xlabel('Classes')
                plt.ylabel('Box Count')
                plt.title('Annotations per Class')
                plt.xticks(rotation=45, ha='right')
                plt.subplot(1, 2, 2)
                splits = list(per_split.keys())
                split_counts = list(per_split.values())
                plt.bar(splits, split_counts, color='lightgreen')
                plt.xlabel('Splits')
                plt.ylabel('Annotated Frames')
                plt.title('Frames per Split')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.show()
            except ImportError:
                QMessageBox.warning(self, 'Plotting Error',
                    'Matplotlib is not installed. Cannot plot details.')
            except Exception as e:
                QMessageBox.warning(self, 'Plotting Error',
                    f'An error occurred while plotting: {e}')

    @log_exceptions
    def rotate_image_dataset(self, direction='cw'):
        """Rotate the current image in the dataset clockwise or counter-clockwise, and update its annotations."""
        if not hasattr(self, 'is_image_dataset') or not self.is_image_dataset:
            QMessageBox.warning(self, 'Rotate Image',
                'This feature is only available for image datasets.')
            return
        if not 0 <= self.current_frame < len(getattr(self, 'image_files', [])):
            return
        image_path = self.image_files[self.current_frame]
        img = cv2.imread(image_path)
        if img is None:
            QMessageBox.warning(self, 'Rotate Image',
                f'Failed to load image: {image_path}')
            return
        H, W = img.shape[:2]
        if direction == 'cw':
            img_rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif direction == 'ccw':
            img_rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            return
        cv2.imwrite(image_path, img_rotated)
        annotations = self.frame_annotations.get(self.current_frame, [])
        for ann in annotations:
            x = ann.rect.x()
            y = ann.rect.y()
            w = ann.rect.width()
            h = ann.rect.height()
            if direction == 'cw':
                new_x = H - y - h
                new_y = x
                new_w = h
                new_h = w
                if ann.segmentation:
                    ann.segmentation = [[H - py, px] for px, py in ann.
                        segmentation]
            else:
                new_x = y
                new_y = W - x - w
                new_w = h
                new_h = w
                if ann.segmentation:
                    ann.segmentation = [[py, W - px] for px, py in ann.
                        segmentation]
            ann.rect = QRect(int(new_x), int(new_y), int(new_w), int(new_h))
        self.canvas.annotations = annotations
        self.update_annotation_list()
        self.seek_to_frame(self.current_frame)
        self.perform_autosave()
        self.statusBar.showMessage(
            f'Rotated image {direction.upper()} and updated annotations.')

    @log_exceptions
    def viat_move_current_to_removed(self):
        """Move the current frame to removed/ (image + label on disk).

        No confirmation dialog -- the user pressed Shift+X / Ctrl+X intentionally.
        """
        if not self._viat_ensure_dataset(show_warning=False):
            # If it's a video file, fallback to deleting the frame
            self.delete_current_frame()
            return
        if not 0 <= self.current_frame < self.total_frames:
            return
        result = _viat_move_to_removed(self, [self.current_frame])
        if self.current_frame >= self.total_frames:
            self.current_frame = max(0, self.total_frames - 1)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.frame_slider.setValue(self.current_frame)
        self.frame_slider.blockSignals(False)
        self.load_current_image()
        self.update_frame_info()
        self.update_annotation_list()
        self.statusBar.showMessage(
            f"Moved to {result['dest_dir']} ({result['moved_images']} img, {result['moved_labels']} lbl)"
            , 3000)

    @log_exceptions
    def viat_move_current_to_review_label(self):
        """Move the current frame to review_label/ (the CHANGE LABEL queue)."""
        if not self._viat_ensure_dataset():
            return
        if not 0 <= self.current_frame < self.total_frames:
            return
        fname = os.path.basename(self.image_files[self.current_frame])
        result = _viat_move_to_review_label(self, [self.current_frame])
        if self.current_frame >= self.total_frames:
            self.current_frame = max(0, self.total_frames - 1)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.frame_slider.setValue(self.current_frame)
        self.frame_slider.blockSignals(False)
        self.load_current_image()
        self.update_frame_info()
        self.update_annotation_list()
        self.statusBar.showMessage(
            f'Moved to review_label/ for label review: {fname}', 5000)

    @log_exceptions
    def viat_remove_grayscale(self):
        """Detect and move all grayscale images to removed/grayscale/."""
        if not self._viat_ensure_dataset():
            return
        reply = QMessageBox.question(self, 'Remove Grayscale Images',
            """Scan all images and move grayscale ones to removed/grayscale/?

This may take a moment for large datasets."""
            , QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        from viat.utils.task_runner import run_task_with_progress
        result = run_task_with_progress(self, 'Removing Grayscale Images',
            'Scanning for grayscale images...', _viat_remove_grayscale,
            self, maximum=100)
        if result is None:
            return
        if self.current_frame >= self.total_frames:
            self.current_frame = max(0, self.total_frames - 1)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.frame_slider.setValue(self.current_frame)
        self.frame_slider.blockSignals(False)
        self.load_current_image()
        self.update_frame_info()
        self.update_annotation_list()
        self.statusBar.showMessage(
            f"Removed {result['moved_images']} grayscale images -> {result['dest_dir']}"
            , 5000)

    @log_exceptions
    def viat_remove_duplicates(self):
        """Remove Roboflow duplicate groups (keep 1 per .rf. group)."""
        if not self._viat_ensure_dataset():
            return
        reply = QMessageBox.question(self, 'Remove .rf. Duplicates',
            """Find Roboflow augmentations (e.g. image_001.rf.abc.jpg) and keep only one random image per group?

Others are moved to removed/duplicates/."""
            , QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        from viat.utils.task_runner import run_task_with_progress
        result = run_task_with_progress(self, 'Removing Duplicate Images',
            'Scanning for duplicates...', _viat_remove_dup_groups, self,
            maximum=100)
        if result is None:
            return
        if self.current_frame >= self.total_frames:
            self.current_frame = max(0, self.total_frames - 1)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.frame_slider.setValue(self.current_frame)
        self.frame_slider.blockSignals(False)
        self.load_current_image()
        self.update_frame_info()
        self.update_annotation_list()
        self.statusBar.showMessage(
            f"Removed {result['moved_images']} duplicates from {result['groups_processed']} groups"
            , 5000)

    @log_exceptions
    def viat_remove_class_and_images_dialog(self):
        """Move all frames whose labels contain a selected class, or just remove the labels."""
        if not self._viat_ensure_dataset():
            return
        classes = sorted(getattr(self.canvas, 'class_colors', {}).keys())
        if not classes:
            QMessageBox.information(self, 'Remove Class', 'No classes loaded.')
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('Remove Class')
        dialog.setMinimumWidth(380)
        layout = QVBoxLayout(dialog)
        layout.addWidget(QLabel('Select classes to remove:'))
        checks = []
        for c in classes:
            cb = QCheckBox(c)
            checks.append((c, cb))
            layout.addWidget(cb)
        remove_images_cb = QCheckBox(
            'Remove images as well (move frames to removed/class_filtered/)')
        remove_images_cb.setChecked(True)
        remove_images_cb.setStyleSheet(
            """
            QCheckBox {
                font-weight: bold;
                color: #e74c3c;
                margin-top: 10px;
                margin-bottom: 5px;
            }
        """
            )
        layout.addWidget(remove_images_cb)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.
            Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return
        selected = [c for c, cb in checks if cb.isChecked()]
        if not selected:
            return
        remove_images = remove_images_cb.isChecked()
        if remove_images:
            reply = QMessageBox.question(self, 'Remove Class + Images',
                f'Move all frames containing classes {selected} to removed/class_filtered/?'
                , QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        else:
            reply = QMessageBox.question(self, 'Remove Class Labels',
                f'Remove labels for classes {selected} from all frames (keep images)?'
                , QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        result = _viat_remove_class_and_images(self, selected,
            remove_images=remove_images)
        if self.current_frame >= self.total_frames:
            self.current_frame = max(0, self.total_frames - 1)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.frame_slider.setValue(self.current_frame)
        self.frame_slider.blockSignals(False)
        self.load_current_image()
        self.update_frame_info()
        self.update_annotation_list()
        if remove_images:
            self.statusBar.showMessage(
                f"Moved {result.get('moved_images', 0)} frames (classes={selected}) -> {result.get('dest_dir', '')}"
                , 5000)
        else:
            self.statusBar.showMessage(
                f"Removed {result.get('removed_boxes', 0)} labels for {selected} in {result.get('affected_frames', 0)} frames."
                , 5000)

    @log_exceptions
    def viat_toggle_segmentation(self):
        """Toggle showing segmentation polygon outlines on the canvas."""
        self.canvas.show_segmentation = not getattr(self.canvas,
            'show_segmentation', False)
        self.canvas.update()
        state = 'ON' if self.canvas.show_segmentation else 'OFF'
        self.statusBar.showMessage(f'Segmentation display: {state}', 3000)

    @log_exceptions
    def viat_toggle_attribute_display(self):
        """Toggle showing attributes on the canvas."""
        self.canvas.show_attributes = not getattr(self.canvas,
            'show_attributes', False)
        self.canvas.update()
        state = 'ON' if self.canvas.show_attributes else 'OFF'
        self.statusBar.showMessage(f'Attribute display: {state}', 3000)

    @log_exceptions
    def viat_view_dataset_log(self):
        """Open the DATASET_LOG.md file in the system text editor."""
        info = getattr(self, '_viat_dataset_info', None)
        if not info:
            QMessageBox.warning(self, 'Dataset Log', 'No dataset loaded.')
            return
        log_path = os.path.join(info.root, 'DATASET_LOG.md')
        if not os.path.isfile(log_path):
            try:
                _viat_init_dataset_log(self, info)
            except Exception:
                pass
        if os.path.isfile(log_path):
            content = open(log_path, 'r', encoding='utf-8').read()
            dialog = QDialog(self)
            dialog.setWindowTitle(
                f'Dataset Log â€” {os.path.basename(info.root)}')
            dialog.setMinimumWidth(600)
            dialog.setMinimumHeight(500)
            layout = QVBoxLayout(dialog)
            text_widget = QPlainTextEdit()
            text_widget.setPlainText(content)
            text_widget.setReadOnly(True)
            layout.addWidget(text_widget)
            buttons = QDialogButtonBox(QDialogButtonBox.Close)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            dialog.exec_()
        else:
            QMessageBox.warning(self, 'Dataset Log',
                'Could not create DATASET_LOG.md')

    @log_exceptions
    def viat_split_video_scenes(self):
        """Detect scene cuts in the currently open video and split it into separate clips."""
        if not self.video_filename or self.is_image_dataset:
            QMessageBox.warning(self, "Split Video", "Please open a video file first.")
            return

        from PyQt5.QtWidgets import QInputDialog, QFileDialog
        
        # Get threshold from user
        threshold, ok = QInputDialog.getDouble(
            self, 'Split Video by Scene Cuts',
            'Enter the cut detection threshold (Adaptive default = 3.0, lower = more cuts):',
            3.0, 1.0, 100.0, 1
        )
        if not ok:
            return

        # Default session dir
        base_dir = os.path.dirname(self.video_filename)
        base_name = os.path.splitext(os.path.basename(self.video_filename))[0]
        default_session_dir = os.path.join(base_dir, f"{base_name}_clips")

        session_dir = QFileDialog.getExistingDirectory(
            self, "Select Output Directory for Clips", default_session_dir
        )
        if not session_dir:
            return

        # Prepare to close current video to avoid file lock issues when removing
        video_path_to_split = self.video_filename
        
        # We need to make sure we don't have pending unsaved changes before closing
        if not self.check_unsaved_changes():
            return
            
        # Close the video
        if hasattr(self, 'video_manager'):
            self.video_manager.close_video()
        else:
            if self.cap:
                self.cap.release()
                self.cap = None

        def split_generator():
            yield 0, "Detecting scenes and splitting (this may take a while)..."
            from viat.utils.scene_splitter import split_video_by_scenes
            # This handles the detection, ffmpeg splitting, and original deletion
            res_dir, num = split_video_by_scenes(video_path_to_split, session_dir, threshold)
            yield res_dir, num

        from viat.utils.task_runner import run_task_with_progress
        
        try:
            result = run_task_with_progress(
                self, "Splitting Video", "Detecting scenes...", 
                split_generator, maximum=0
            )
            
            if result:
                res_dir, num = result
                msg = f"Successfully split video into {num} clips.\nOutput directory: {res_dir}\nOriginal video was removed."
                QMessageBox.information(self, "Split Video Complete", msg)
                
                # Optional: prompt user to open the new clips directory as image dataset or let them do it manually
                reply = QMessageBox.question(self, "Open Clips", "Would you like to open the output directory as a sequence?",
                                           QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.Yes:
                    self.open_image_dataset(res_dir)
                    
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to split video:\n{str(e)}")

    @log_exceptions
    def viat_import_video_json(self):
        """Import a VIAT custom JSON annotation file.

        Works for both video and image-dataset projects. If no media is
        open, infers total_frames from the JSON's max frame key.
        """
        filename, _ = QFileDialog.getOpenFileName(self,
            'Import VIAT JSON Annotations', '',
            'JSON Files (*.json);;All Files (*)')
        if not filename:
            return
        if not self.cap and not self.is_image_dataset:
            try:
                import json
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict) and data:
                    max_frame = max(int(k) for k in data.keys() if str(k).
                        isdigit())
                    self.total_frames = max_frame + 1
                    self.frame_slider.blockSignals(True)
                    self.frame_slider.setMaximum(max(0, self.total_frames - 1))
                    self.frame_slider.setValue(self.current_frame)
                    self.frame_slider.blockSignals(False)
            except Exception as e:
                QMessageBox.warning(self, 'Import JSON',
                    f"""Could not read JSON file:
{e}

Make sure it's a VIAT JSON (frame-number keys with 'actors')."""
                    )
                return
        self.save_undo_state('all')
        try:
            result = _viat_load_json_video(self, filename, BoundingBox)
        except Exception as e:
            QMessageBox.critical(self, 'Import JSON',
                f"""Failed to parse JSON:
{e}

The file must be in VIAT JSON format:  "0000": {{"actors": {{...}}}}"""
                )
            return
        self.refresh_class_ui()
        self.update_annotation_list()
        if self.current_frame in self.frame_annotations:
            self.canvas.annotations = self.frame_annotations[self.current_frame
                ]
            self.canvas.update()
        if result['frames_loaded'] == 0:
            QMessageBox.warning(self, 'Import JSON',
                """No frames were loaded. The file may not be in VIAT JSON format
(expected: {"0000": {"actors": {...}}})"""
                )
            return
        msg = (
            f"Imported {result['frames_loaded']} frames, {result['actors_loaded']} annotations, {len(result['classes_found'])} classes. Actors: {', '.join(result.get('actor_ids', [])[:5])}"
            )
        if result.get('warnings'):
            msg += f"  ({len(result['warnings'])} warnings)"
        self.statusBar.showMessage(msg, 8000)

    @log_exceptions
    def viat_detect_and_fix_borders(self):
        """Detect video borders and adjust (remove/clip) labels in the border."""
        if not self.cap and not self.is_image_dataset:
            QMessageBox.warning(self, 'Video Borders', 'Open a video first.')
            return
        self.statusBar.showMessage('Detecting video borders...')
        QApplication.processEvents()
        self.save_undo_state('all')
        result = _viat_detect_adjust_borders(self)
        if not result.get('detected', False):
            QMessageBox.information(self, 'Video Borders',
                'No black/gray borders detected on the left or right edges.')
            self.statusBar.showMessage('No borders detected', 3000)
            return
        det = result['detection']
        adj = result['adjustment']
        self.canvas.update()
        self.update_annotation_list()
        msg = f"""Borders: left={det['left_border']}px, right={det['right_border']}px (sampled {det['sampled']} frames).

Annotations: {adj['removed']} removed (â‰¥80% in border), {adj['clipped']} clipped, {adj['unchanged']} unchanged."""
        QMessageBox.information(self, 'Video Borders Adjusted', msg)
        self.statusBar.showMessage(
            f"Borders fixed: {adj['removed']} removed, {adj['clipped']} clipped"
            , 5000)

    @log_exceptions
    def viat_start_object_visibility_mode(self):
        """Enter per-object visibility editing mode (toolbar-based, no dialog)."""
        if not self.frame_annotations:
            QMessageBox.warning(self, 'Object Visibility',
                'No annotations loaded.')
            return
        if (self.object_visibility_manager and self.
            object_visibility_manager.active):
            return
        from viat.utils.object_visibility import ObjectVisibilityManager
        self.object_visibility_manager = ObjectVisibilityManager(self)
        if not self.object_visibility_manager.start():
            QMessageBox.warning(self, 'Object Visibility',
                'No objects found (need actor_id attributes).')
            self.object_visibility_manager = None
            return
        self._viat_show_object_visibility_toolbar()

    @log_exceptions
    def viat_exit_object_visibility_mode(self):
        """Exit per-object visibility editing mode."""
        if (self.object_visibility_manager and self.
            object_visibility_manager.active):
            self.object_visibility_manager.exit()
        self.object_visibility_manager = None
        self._viat_hide_object_visibility_toolbar()
        self.statusBar.showMessage('Exited object visibility mode', 3000)

    def _viat_show_object_visibility_toolbar(self):
        """Create a toolbar with object-visibility controls (no dialog)."""
        from PyQt5.QtWidgets import QToolBar, QLabel, QPushButton, QWidget, QHBoxLayout
        from PyQt5.QtCore import Qt
        mgr = self.object_visibility_manager
        if not mgr or not mgr.active:
            return
        if hasattr(self, '_viat_visibility_toolbar'):
            self.removeToolBar(self._viat_visibility_toolbar)
        toolbar = QToolBar('Object Visibility', self)
        toolbar.setObjectName('ObjectVisibilityToolbar')
        self.addToolBar(Qt.BottomToolBarArea, toolbar)
        self._viat_obj_label = QLabel()
        toolbar.addWidget(self._viat_obj_label)
        toolbar.addSeparator()
        self._viat_range_label = QLabel()
        toolbar.addWidget(self._viat_range_label)
        toolbar.addSeparator()
        btn_prev_obj = QPushButton('< Prev Obj')
        btn_next_obj = QPushButton('Next Obj >')
        btn_prev_range = QPushButton('< Prev Range')
        btn_next_range = QPushButton('Next Range >')
        btn_set_start = QPushButton('Set Start')
        btn_set_end = QPushButton('Set End')
        btn_del_frame = QPushButton('Delete Frame')
        btn_del_range = QPushButton('Delete Range')
        btn_del_obj = QPushButton('Delete Object')
        btn_auto_erase = QPushButton('Auto Erase: OFF')
        btn_auto_erase.setCheckable(True)

        def on_auto_erase_toggled(checked):
            mgr.auto_erase_mode = checked
            btn_auto_erase.setText('Auto Erase: ON' if checked else
                'Auto Erase: OFF')
            if checked:
                btn_auto_erase.setStyleSheet(
                    'font-weight: bold; background-color: #f44336; color: white;'
                    )
            else:
                btn_auto_erase.setStyleSheet('')
        btn_auto_erase.toggled.connect(on_auto_erase_toggled)
        btn_finish = QPushButton('FINISH')
        btn_finish.setStyleSheet(
            'font-weight: bold; background-color: #4CAF50; color: white;')
        btn_exit = QPushButton('Exit')
        for btn in [btn_prev_obj, btn_next_obj, btn_prev_range,
            btn_next_range, btn_set_start, btn_set_end, btn_del_frame,
            btn_del_range, btn_del_obj, btn_auto_erase, btn_finish, btn_exit]:
            toolbar.addWidget(btn)
        mgr.toolbar_widgets = {'obj_label': self._viat_obj_label,
            'range_label': self._viat_range_label, 'prev_obj': btn_prev_obj,
            'next_obj': btn_next_obj, 'prev_range': btn_prev_range,
            'next_range': btn_next_range, 'set_start': btn_set_start,
            'set_end': btn_set_end, 'del_frame': btn_del_frame, 'del_range':
            btn_del_range, 'del_obj': btn_del_obj, 'finish': btn_finish,
            'exit': btn_exit}

        def update_labels():
            s = mgr.get_status()
            self._viat_obj_label.setText(
                f"Object: {s['current_object']} ({s['object_index'] + 1}/{s['total_objects']})"
                )
            r = s['current_range']
            if r:
                self._viat_range_label.setText(
                    f"Range: {r[0]}-{r[1]} ({s['range_index'] + 1}/{s['total_ranges']})"
                    )
                self.frame_slider.blockSignals(True)
                self.frame_slider.setMinimum(r[0])
                self.frame_slider.setMaximum(r[1])
                self.frame_slider.blockSignals(False)
            else:
                self._viat_range_label.setText('No ranges')
                self.frame_slider.blockSignals(True)
                self.frame_slider.setMinimum(0)
                self.frame_slider.setMaximum(max(0, self.total_frames - 1))
                self.frame_slider.blockSignals(False)

        def on_prev_obj():
            mgr.prev_object()
            update_labels()

        def on_next_obj():
            mgr.next_object()
            update_labels()

        def on_prev_range():
            mgr.prev_range()
            update_labels()

        def on_next_range():
            mgr.next_range()
            update_labels()

        def on_set_start():
            mgr.trim_current_frame_as_start()
            update_labels()

        def on_set_end():
            mgr.trim_current_frame_as_end()
            update_labels()

        def on_del_frame():
            n = mgr.delete_current_object_on_current_frame()
            update_labels()
            self.statusBar.showMessage(
                f'Deleted {n} annotation(s) on this frame', 2000)

        def on_del_range():
            r = mgr.get_current_range()
            if not r:
                return
            reply = QMessageBox.question(self, 'Delete Range',
                f"Delete ALL annotations for '{mgr.current_object}' in range {r[0]}-{r[1]}?"
                , QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                n = mgr.remove_current_range()
                if not mgr.sorted_objects:
                    self.viat_exit_object_visibility_mode()
                    return
                if mgr.current_object not in mgr.sorted_objects:
                    mgr.current_object_index = min(mgr.current_object_index,
                        len(mgr.sorted_objects) - 1)
                    mgr.current_object = mgr.sorted_objects[mgr.
                        current_object_index]
                    mgr.current_range_index = 0
                    mgr._apply_filter()
                update_labels()
                self.statusBar.showMessage(f'Deleted {n} annotations in range',
                    3000)

        def on_del_obj():
            reply = QMessageBox.question(self, 'Delete Object',
                f"Delete ALL annotations for '{mgr.current_object}' across all frames?"
                , QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.Yes:
                n = mgr.delete_object()
                if not mgr.sorted_objects:
                    self.viat_exit_object_visibility_mode()
                    return
                update_labels()
                self.statusBar.showMessage(f'Deleted {n} annotations', 3000)

        def on_finish():
            if mgr.next_range():
                update_labels()
            elif mgr.next_object():
                update_labels()
            else:
                QMessageBox.information(self, 'All Done',
                    'All objects have been processed.')
                self.viat_exit_object_visibility_mode()

        def on_exit():
            self.viat_exit_object_visibility_mode()
        btn_prev_obj.clicked.connect(on_prev_obj)
        btn_next_obj.clicked.connect(on_next_obj)
        btn_prev_range.clicked.connect(on_prev_range)
        btn_next_range.clicked.connect(on_next_range)
        btn_set_start.clicked.connect(on_set_start)
        btn_set_end.clicked.connect(on_set_end)
        btn_del_frame.clicked.connect(on_del_frame)
        btn_del_range.clicked.connect(on_del_range)
        btn_del_obj.clicked.connect(on_del_obj)
        btn_finish.clicked.connect(on_finish)
        btn_exit.clicked.connect(on_exit)
        self._viat_visibility_toolbar = toolbar
        self._viat_update_visibility_labels = update_labels
        update_labels()

    def _viat_hide_object_visibility_toolbar(self):
        """Remove the object visibility toolbar."""
        if hasattr(self, '_viat_visibility_toolbar'):
            self.removeToolBar(self._viat_visibility_toolbar)
            self._viat_visibility_toolbar.deleteLater()
            del self._viat_visibility_toolbar
        if hasattr(self, 'frame_slider'):
            self.frame_slider.blockSignals(True)
            self.frame_slider.setMinimum(0)
            self.frame_slider.setMaximum(max(0, getattr(self,
                'total_frames', 1) - 1))
            self.frame_slider.blockSignals(False)

    @log_exceptions
    def viat_seg_video_pick_color(self):
        """Pick a color from the canvas by CLICKING on it (no X/Y dialog).

        Enables canvas color-pick mode. The user clicks on the canvas, the
        color at that point is extracted, and a small non-blocking dialog
        asks for the class name + actor ID.
        """
        if not self.cap and not self.is_image_dataset:
            QMessageBox.warning(self, 'Seg Video', 'Open a video first.')
            return
        self.canvas.color_pick_mode = True
        self.canvas.setCursor(Qt.CrossCursor)
        self.statusBar.showMessage(
            'Click on an object in the canvas to pick its color...', 0)

        def on_color_picked(x, y):
            """Called when the user clicks on the canvas."""
            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    return
                frame = self._process_frame_metadata(frame, self.current_frame)
            elif self.is_image_dataset and self.image_files:
                frame = cv2.imread(self.image_files[self.current_frame])
            else:
                return
            from viat.utils.seg_video_labeler import SegmentationVideoLabeler
            tmp_labeler = SegmentationVideoLabeler(self)
            color_hsv = tmp_labeler.pick_color_from_frame(frame, x, y)
            self.statusBar.showMessage(
                f'Picked color at ({x},{y}): HSV={color_hsv}', 5000)
            dialog = QDialog(self)
            dialog.setWindowTitle(f'Add Tracked Object (HSV={color_hsv})')
            dialog.setMinimumWidth(350)
            from PyQt5.QtWidgets import QVBoxLayout, QFormLayout, QDialogButtonBox
            layout = QVBoxLayout(dialog)
            form = QFormLayout()
            from PyQt5.QtWidgets import QLineEdit, QSpinBox
            class_edit = QLineEdit()
            class_edit.setPlaceholderText('class name (e.g. Helicopter)')
            form.addRow('Class:', class_edit)
            actor_edit = QLineEdit()
            actor_edit.setPlaceholderText('auto if empty')
            form.addRow('Actor ID:', actor_edit)
            tol_spin = QSpinBox()
            tol_spin.setRange(1, 60)
            tol_spin.setValue(15)
            form.addRow('Tolerance:', tol_spin)
            min_area_spin = QSpinBox()
            min_area_spin.setRange(1, 99999)
            min_area_spin.setValue(100)
            form.addRow('Min area:', min_area_spin)
            from PyQt5.QtWidgets import QCheckBox
            merge_all_cb = QCheckBox(
                'Create one big bounding box (Merge regions)')
            merge_all_cb.setChecked(False)
            form.addRow('', merge_all_cb)
            connect_spin = QSpinBox()
            connect_spin.setRange(0, 500)
            connect_spin.setValue(10)
            connect_spin.setToolTip(
                'Pixel distance to connect small disconnected regions')
            form.addRow('Connect Threshold:', connect_spin)
            layout.addLayout(form)
            buttons = QDialogButtonBox(QDialogButtonBox.Ok |
                QDialogButtonBox.Cancel)
            buttons.accepted.connect(dialog.accept)
            buttons.rejected.connect(dialog.reject)
            layout.addWidget(buttons)
            if dialog.exec_() == QDialog.Accepted:
                class_name = class_edit.text().strip()
                if not class_name:
                    return
                if self.seg_labeler is None:
                    from viat.utils.seg_video_labeler import SegmentationVideoLabeler
                    self.seg_labeler = SegmentationVideoLabeler(self)
                self.seg_labeler.add_tracked_object(color_hsv=color_hsv,
                    class_name=class_name, actor_id=actor_edit.text().strip
                    () or None, tolerance=tol_spin.value(), min_area=
                    min_area_spin.value(), merge_all=merge_all_cb.isChecked
                    (), connect_threshold=connect_spin.value())
                self.statusBar.showMessage(
                    f"Added '{class_name}' (HSV={color_hsv}). Total: {len(self.seg_labeler.tracked_objects)}. Click another object or track all."
                    , 5000)
        self.canvas.color_pick_callback = on_color_picked

    @log_exceptions
    def viat_seg_video_track_all(self):
        """Track all picked objects across the video and add as annotations."""
        if self.seg_labeler is None or not self.seg_labeler.tracked_objects:
            QMessageBox.warning(self, 'Seg Video',
                'No tracked objects. Pick a color first.')
            return
        if not self.video_filename and not self.cap:
            QMessageBox.warning(self, 'Seg Video', 'No video loaded.')
            return
        video_path = self.video_filename
        self.statusBar.showMessage(
            f'Tracking {len(self.seg_labeler.tracked_objects)} objects...')
        QApplication.processEvents()
        result = self.seg_labeler.track_all_objects(video_path, start_frame
            =0, end_frame=self.total_frames - 1, progress_callback=lambda
            cur, tot: self.statusBar.showMessage(
            f'Tracking... {cur}/{tot} frames', 0))
        added = self.seg_labeler.commit_to_app(self, BoundingBox)
        self.refresh_class_ui()
        self.update_annotation_list()
        if self.current_frame in self.frame_annotations:
            self.canvas.annotations = self.frame_annotations[self.current_frame
                ]
            self.canvas.update()
        msg = f"""Tracked {len(self.seg_labeler.tracked_objects)} objects across {result['frames_processed']} frames. Added {added} annotations.
Per object: {result['per_object']}"""
        QMessageBox.information(self, 'Seg Video Tracking Complete', msg)
        self.statusBar.showMessage(f'Added {added} annotations from seg video',
            5000)

    @log_exceptions
    def viat_seg_video_export_json(self):
        """Export tracked seg-video objects to VIAT JSON."""
        if self.seg_labeler is None or not self.seg_labeler.tracked_objects:
            QMessageBox.warning(self, 'Seg Video', 'No tracked objects.')
            return
        default_name = 'seg_annotations.json'
        if self.video_filename:
            base = os.path.splitext(os.path.basename(self.video_filename))[0]
            default_name = base + '_seg.json'
        filename, _ = QFileDialog.getSaveFileName(self,
            'Export Seg Video JSON', default_name,
            'JSON Files (*.json);;All Files (*)')
        if not filename:
            return
        content = self.seg_labeler.to_viat_json()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        self.statusBar.showMessage(f'Exported seg video JSON to {filename}',
            5000)

    @log_exceptions
    def viat_export_json(self):
        """Export all current annotations to VIAT custom JSON format."""
        self.frame_annotations[self.current_frame
            ] = self.canvas.annotations.copy()
        has_annotations = any(self.frame_annotations.values())
        if not has_annotations:
            QMessageBox.warning(self, 'Export VIAT JSON',
                'No annotations to export!')
            return
        default_name = 'annotations.json'
        default_dir = ''
        if self.video_filename:
            base = os.path.splitext(os.path.basename(self.video_filename))[0]
            default_name = base + '_viat.json'
            default_dir = os.path.dirname(self.video_filename)
        elif hasattr(self, 'is_image_dataset'
            ) and self.is_image_dataset and self.image_files:
            image_folder = os.path.dirname(self.image_files[0])
            folder_name = os.path.basename(image_folder)
            default_name = folder_name + '_viat.json'
            default_dir = image_folder
        default_path = os.path.join(default_dir, default_name
            ) if default_dir else default_name
        filename, _ = QFileDialog.getSaveFileName(self, 'Export VIAT JSON',
            default_path, 'JSON Files (*.json);;All Files (*)')
        if not filename:
            return
        boxes_by_frame = {}
        for frame_num, annotations in self.frame_annotations.items():
            if not annotations:
                continue
            boxes_by_frame[frame_num] = []
            for ann in annotations:
                box_dict = {'class_name': ann.class_name, 'x': ann.rect.x(),
                    'y': ann.rect.y(), 'w': ann.rect.width(), 'h': ann.rect
                    .height(), 'verified': getattr(ann, 'verified', False),
                    'segmentation': getattr(ann, 'segmentation', None)}
                actor_id = ann.attributes.get('actor_id') if hasattr(ann,
                    'attributes') else None
                if actor_id:
                    box_dict['actor_id'] = actor_id
                boxes_by_frame[frame_num].append(box_dict)
        try:
            from viat.utils.label_formats.viat_json import ViatJsonLabelFormat
            fmt = ViatJsonLabelFormat()
            content = fmt.dump(boxes_by_frame, (0, 0), [])
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
            self.statusBar.showMessage(f'Exported VIAT JSON to {filename}',
                5000)
        except Exception as e:
            QMessageBox.critical(self, 'Error',
                f'Failed to export VIAT JSON: {e}')

    @log_exceptions
    def viat_perf_stats(self):
        """Show frame cache statistics."""
        if not hasattr(self, 'viat_perf'):
            QMessageBox.information(self, 'Performance',
                'No performance manager.')
            return
        stats = self.viat_perf.get_stats()
        QMessageBox.information(self, 'Frame Cache Stats',
            f"""Cache: {stats['cache_size']}/{stats['cache_capacity']} frames
Hit rate: {stats['hit_rate']}
Hits: {stats['hits']}, Misses: {stats['misses']}"""
            )

    @log_exceptions
    def viat_clear_cache(self):
        """Clear the frame cache."""
        if hasattr(self, 'viat_perf'):
            self.viat_perf.clear_cache()
            self.statusBar.showMessage('Frame cache cleared', 2000)

    @log_exceptions
    def viat_auto_import_detections(self):
        """Auto-import a YOLO detections JSON and move flagged images to review_label/.

        Workflow:
          1. Run yolo_detect.py separately (produces _viat_detections.json).
          2. In VIAT: Dataset > Auto-Import Detections... -> select the JSON.
          3. Images WITH detections are moved to review_label/ and the
             detections are added as unverified annotations.
        """
        if not self._viat_ensure_dataset():
            return
        filename, _ = QFileDialog.getOpenFileName(self,
            'Select Detections JSON', '', 'JSON Files (*.json);;All Files (*)')
        if not filename:
            return
        reply = QMessageBox.question(self, 'Auto-Import Detections',
            f"""Import detections from:
  {os.path.basename(filename)}

Images with detections will be:
  - moved to review_label/
  - detections added as unverified annotations

Continue?"""
            , QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if reply != QMessageBox.Yes:
            return
        from viat.utils.task_runner import run_task_with_progress
        result = run_task_with_progress(self, 'Auto-Import Detections',
            'Importing detections...', _viat_auto_import_detections, self,
            filename, move_to_review=True, add_as_annotations=False,
            bbox_cls=BoundingBox, maximum=100)
        if result is None:
            return
        if self.current_frame >= self.total_frames:
            self.current_frame = max(0, self.total_frames - 1)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMaximum(max(0, self.total_frames - 1))
        self.frame_slider.setValue(self.current_frame)
        self.frame_slider.blockSignals(False)
        self.load_current_image()
        self.update_frame_info()
        self.update_annotation_list()
        self.refresh_class_ui()
        json_info = ''
        if result.get('json_copied_to'):
            json_info = (
                f'\n  Detections JSON copied to: review_label/_viat_detections.json'
                )
        msg = f"""Auto-import complete:
  Flagged images: {result['flagged_images']}
  Moved to review_label/: {result['moved_images']}
  Total detections in JSON: {result['total_detections']}{json_info}

To review: open the review_label/ folder as a dataset, then
use Dataset > Import VIAT JSON Annotations to load the detections."""
        QMessageBox.information(self, 'Auto-Import Detections', msg)
        self.statusBar.showMessage(
            f"Moved {result['moved_images']} images to review_label/", 5000)

    @log_exceptions
    def viat_merge_dataset(self, source_folder=None, target_folder=None):
        """Merge another dataset into the current (target) dataset."""
        if not target_folder:
            info = getattr(self, '_viat_dataset_info', None)
            if info is not None and getattr(self, 'is_image_dataset', False):
                target_folder = info.root
            else:
                target_folder = QFileDialog.getExistingDirectory(self,
                    'Select Target Dataset to Merge Into', '', QFileDialog.ShowDirsOnly)
                if not target_folder:
                    return
        if not source_folder:
            source_folder = QFileDialog.getExistingDirectory(self,
                'Select Source Dataset to Merge', '', QFileDialog.ShowDirsOnly)
            if not source_folder:
                return
        default_name = os.path.basename(source_folder)
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QLineEdit, QComboBox, QDialogButtonBox, QLabel, QSpinBox, QTableWidget, QTableWidgetItem, QHeaderView
        dialog = QDialog(self)
        dialog.setWindowTitle('Merge Dataset')
        dialog.setMinimumWidth(550)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        name_edit = QLineEdit(default_name)
        form.addRow('Dataset name (file prefix):', name_edit)
        split_combo = QComboBox()
        split_combo.addItems(['Keep original splits',
            'Random split (specify %)', 'All in train', 'All in valid',
            'All in test'])
        form.addRow('Split mode:', split_combo)
        valid_spin = QSpinBox()
        valid_spin.setRange(1, 50)
        valid_spin.setValue(10)
        form.addRow('Valid % (if random):', valid_spin)
        layout.addLayout(form)
        check_result = _viat_find_unmatched_classes(source_folder,
            target_folder)
        matched = check_result['matched']
        target_classes = check_result['target_classes']
        source_classes = check_result['source_classes']
        if source_classes:
            layout.addWidget(QLabel(
                f"""Map source dataset classes to target dataset classes.
Select 'DONT MERGE' to skip, or type a new name to create a new class:"""
                ))
            table = QTableWidget(len(source_classes), 2)
            table.setHorizontalHeaderLabels(['Source class',
                'Map to (target class)'])
            table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            for i, src_cls in enumerate(source_classes):
                item_src = QTableWidgetItem(src_cls)
                item_src.setFlags(item_src.flags() & ~Qt.ItemIsEditable)
                table.setItem(i, 0, item_src)
                combo = QComboBox()
                combo.setEditable(True)
                combo.addItem('DONT MERGE')
                if target_classes:
                    combo.addItems(target_classes)
                if src_cls in matched:
                    combo.setCurrentText(matched[src_cls])
                else:
                    default_text = target_classes[0
                        ] if target_classes else src_cls
                    combo.setCurrentText(default_text)
                table.setCellWidget(i, 1, combo)
            layout.addWidget(table)
        else:
            table = None
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.
            Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return
        dataset_name = name_edit.text().strip() or default_name
        split_mode_map = {(0): 'keep', (1): 'random', (2): 'all_train', (3):
            'all_valid', (4): 'all_test'}
        split_mode = split_mode_map[split_combo.currentIndex()]
        class_mapping = dict(check_result['matched'])
        if table:
            for i in range(table.rowCount()):
                src = table.item(i, 0).text()
                combo = table.cellWidget(i, 1)
                if combo:
                    tgt = combo.currentText()
                else:
                    tgt = table.item(i, 1).text()
                class_mapping[src] = tgt
        self.statusBar.showMessage(f'Merging {dataset_name} into target...')
        QApplication.processEvents()
        result = _viat_merge_dataset(self, source_folder, target_folder,
            dataset_name=dataset_name, split_mode=split_mode,
            random_valid_pct=valid_spin.value(), class_mapping=
            class_mapping, progress_callback=lambda cur, tot, msg: self.
            statusBar.showMessage(f'Merging {cur}/{tot}: {msg}', 0))
        if result.get('error'):
            QMessageBox.warning(self, 'Merge Error', result['error'])
            return
        msg = f"""Merge complete:
  Images copied: {result['images_copied']}
  Labels copied: {result['labels_copied']}
  Classes mapped: {result['classes_mapped']}
  Skipped: {result['skipped']}
  Dataset name: {result['dataset_name']}"""
        QMessageBox.information(self, 'Merge Complete', msg)
        self.statusBar.showMessage(
            f"Merged {result['images_copied']} images from {dataset_name}",
            5000)

    @log_exceptions
    def viat_extract_single_class_dataset(self):
        """Show the single class dataset extraction dialog."""
        from widgets.single_class_extractor_dialog import SingleClassExtractorDialog
        dialog = SingleClassExtractorDialog(self)
        dialog.exec_()

    @log_exceptions
    def viat_launch_batch_prediction_queue(self):
        """Show the batch prediction queue builder dialog."""
        from widgets.batch_prediction_dialog import BatchPredictionDialog
        dialog = BatchPredictionDialog(self)
        dialog.exec_()

    @log_exceptions
    def viat_remove_background_images(self):
        """Show the background remover dialog."""
        from widgets.background_remover_dialog import BackgroundRemoverDialog
        dialog = BackgroundRemoverDialog(self)
        dialog.exec_()

    @log_exceptions
    def show_about(self):
        """Show about dialog."""
        QMessageBox.about(self, 'About Video Annotation Tool',
            """Video Annotation Tool (VAT)

A tool for annotating objects in videos for computer vision tasks.

Features:
- Bounding box annotations with edge movement for precise adjustments
- Multiple object classes with customizable colors
- Export to common formats (COCO, YOLO, Pascal VOC)
- Project saving and loading
- Right-click context menu for quick editing

Created as a demonstration of PyQt5 capabilities."""
            )

    @log_exceptions
    def clear_application_history():
        """Clear all application history files."""
        config_dir = get_config_directory()
        files_to_delete = ['recent_projects.json', 'last_state.json',
            'settings.json']
        deleted_files = []
        for filename in files_to_delete:
            file_path = os.path.join(config_dir, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                deleted_files.append(filename)
        return deleted_files

    @log_exceptions
    def toggle_tracking_mode(self, enabled):
        self.tracking_mode_enabled = enabled
        if hasattr(self, 'tracking_toggle_btn'
            ) and self.tracking_toggle_btn is not None:
            self.tracking_toggle_btn.blockSignals(True)
            self.tracking_toggle_btn.setChecked(enabled)
            self.tracking_toggle_btn.blockSignals(False)
        if enabled:
            self.add_track_id_to_bboxes()
            self.statusBar.showMessage(
                'Tracking mode enabled: track_id will be auto-assigned', 3000)
        else:
            self.statusBar.showMessage(
                'Tracking mode disabled: track_id hidden from UI', 3000)
        self.update_annotation_list()
        self.canvas.update()

    def add_track_id_to_bboxes(self):
        """
        Add a 'track_id' attribute to all bounding boxes in all frames.
        Uses IoU tracking to maintain consistent IDs across frames, even when
        objects disappear and reappear later.
        """
        class_counts = {}
        for frame_num, annotations in self.frame_annotations.items():
            frame_class_counts = {}
            for ann in annotations:
                if hasattr(ann, 'rect'):
                    class_name = getattr(ann, 'class_name', 'unknown')
                    frame_class_counts[class_name] = frame_class_counts.get(
                        class_name, 0) + 1
            for class_name, count in frame_class_counts.items():
                class_counts[class_name] = max(class_counts.get(class_name,
                    0), count)
        for frame_num, annotations in self.frame_annotations.items():
            for ann in annotations:
                if hasattr(ann, 'rect'):
                    if not hasattr(ann, 'attributes'):
                        ann.attributes = {}
                    if 'track_id' in ann.attributes:
                        del ann.attributes['track_id']
        for ann in self.canvas.annotations:
            if hasattr(ann, 'rect'):
                if not hasattr(ann, 'attributes'):
                    ann.attributes = {}
                if 'track_id' in ann.attributes:
                    del ann.attributes['track_id']
        tracked_objects = {}
        updated_count = 0
        sorted_frames = sorted(self.frame_annotations.keys())
        memory_window = 30
        for frame_num in sorted_frames:
            annotations = self.frame_annotations[frame_num]
            class_annotations = {}
            for ann in annotations:
                if hasattr(ann, 'rect'):
                    if not hasattr(ann, 'attributes'):
                        ann.attributes = {}
                    class_name = getattr(ann, 'class_name', 'unknown')
                    if class_name not in class_annotations:
                        class_annotations[class_name] = []
                    class_annotations[class_name].append(ann)
                    if class_name not in tracked_objects:
                        tracked_objects[class_name] = []
                        if class_name not in class_counts:
                            class_counts[class_name] = 1
            for class_name, anns in class_annotations.items():
                matched_objs = set()
                matched_anns = set()
                for i, obj in enumerate(tracked_objects[class_name]):
                    if not obj['active']:
                        continue
                    best_iou = 0.5
                    best_ann = None
                    best_ann_idx = -1
                    for j, ann in enumerate(anns):
                        if j in matched_anns:
                            continue
                        iou_val = self.iou(ann.rect, obj['rect'])
                        if iou_val > best_iou:
                            best_iou = iou_val
                            best_ann = ann
                            best_ann_idx = j
                    if best_ann is not None:
                        matched_objs.add(i)
                        matched_anns.add(best_ann_idx)
                        best_ann.attributes['track_id'] = obj['track_id']
                        obj['rect'] = best_ann.rect
                        obj['last_seen'] = frame_num
                        obj['active'] = True
                        updated_count += 1
                for i, obj in enumerate(tracked_objects[class_name]):
                    if i not in matched_objs and obj['active']:
                        obj['active'] = False
                for j, ann in enumerate(anns):
                    if j in matched_anns:
                        continue
                    best_iou = 0.4
                    best_obj_idx = -1
                    for i, obj in enumerate(tracked_objects[class_name]):
                        if i in matched_objs or obj['active']:
                            continue
                        if frame_num - obj['last_seen'] > memory_window:
                            continue
                        iou_val = self.iou(ann.rect, obj['rect'])
                        if iou_val > best_iou:
                            best_iou = iou_val
                            best_obj_idx = i
                    if best_obj_idx != -1:
                        matched_objs.add(best_obj_idx)
                        matched_anns.add(j)
                        ann.attributes['track_id'] = tracked_objects[class_name
                            ][best_obj_idx]['track_id']
                        tracked_objects[class_name][best_obj_idx]['rect'
                            ] = ann.rect
                        tracked_objects[class_name][best_obj_idx]['last_seen'
                            ] = frame_num
                        tracked_objects[class_name][best_obj_idx]['active'
                            ] = True
                        updated_count += 1
                for j, ann in enumerate(anns):
                    if j in matched_anns:
                        continue
                    available_ids = set(range(class_counts[class_name]))
                    used_ids = {obj['track_id'] for obj in tracked_objects[
                        class_name]}
                    available_ids -= used_ids
                    if available_ids:
                        track_id = min(available_ids)
                    elif len(tracked_objects[class_name]) < class_counts[
                        class_name]:
                        track_id = len(tracked_objects[class_name])
                    else:
                        oldest_frame = float('inf')
                        oldest_idx = 0
                        for i, obj in enumerate(tracked_objects[class_name]):
                            if not obj['active'] and obj['last_seen'
                                ] < oldest_frame:
                                oldest_frame = obj['last_seen']
                                oldest_idx = i
                        if not math.isinf(oldest_frame):
                            track_id = tracked_objects[class_name][oldest_idx][
                                'track_id']
                            tracked_objects[class_name].pop(oldest_idx)
                        else:
                            track_id = 0
                    tracked_objects[class_name].append({'track_id':
                        track_id, 'rect': ann.rect, 'last_seen': frame_num,
                        'active': True})
                    ann.attributes['track_id'] = track_id
                    updated_count += 1
        if self.current_frame in self.frame_annotations:
            for ann in self.canvas.annotations:
                if hasattr(ann, 'rect'):
                    if not hasattr(ann, 'attributes'):
                        ann.attributes = {}
                    for frame_ann in self.frame_annotations[self.current_frame
                        ]:
                        if hasattr(frame_ann, 'rect') and self.iou(ann.rect,
                            frame_ann.rect) > 0.9:
                            if 'track_id' in frame_ann.attributes:
                                ann.attributes['track_id'
                                    ] = frame_ann.attributes['track_id']
                                break
        if not hasattr(self, 'iou'):
            self.iou = lambda rect1, rect2: self.calculate_iou(rect1, rect2)
        self.update_annotation_list()
        self.canvas.update()
        QMessageBox.information(self, 'Track ID',
            f"Added 'track_id' to {updated_count} bounding boxes using intelligent tracking."
            )

    def calculate_iou(self, rect1, rect2):
        """
        Calculate Intersection over Union between two QRect objects.
        """
        x_left = max(rect1.left(), rect2.left())
        y_top = max(rect1.top(), rect2.top())
        x_right = min(rect1.right(), rect2.right())
        y_bottom = min(rect1.bottom(), rect2.bottom())
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        rect1_area = rect1.width() * rect1.height()
        rect2_area = rect2.width() * rect2.height()
        union_area = rect1_area + rect2_area - intersection_area
        if union_area == 0:
            return 0.0
        return intersection_area / union_area

    def assign_track_id_to_new_bbox(self, new_bbox):
        """
        Assigns a track_id to new_bbox based on IoU with previous frame's bboxes.
        """
        if not self.tracking_mode_enabled:
            return
        prev_frame = self.current_frame - 1
        if prev_frame < 0 or prev_frame not in self.frame_annotations:
            new_id = self.get_next_track_id()
            new_bbox.attributes['track_id'] = new_id
            return
        class_name = getattr(new_bbox, 'class_name', 'unknown')
        prev_bboxes = [ann for ann in self.frame_annotations[prev_frame] if
            hasattr(ann, 'rect') and getattr(ann, 'class_name', 'unknown') ==
            class_name]
        best_iou = 0
        best_ann = None
        for ann in prev_bboxes:
            iou_val = self.iou(new_bbox.rect, ann.rect)
            if iou_val > best_iou:
                best_iou = iou_val
                best_ann = ann
        if best_iou > 0.5 and best_ann and 'track_id' in best_ann.attributes:
            new_bbox.attributes['track_id'] = best_ann.attributes['track_id']
        else:
            new_bbox.attributes['track_id'] = self.get_next_track_id()

    def get_next_track_id(self):
        max_id = -1
        for anns in self.frame_annotations.values():
            for ann in anns:
                tid = ann.attributes.get('track_id') if hasattr(ann,
                    'attributes') else None
                if tid is not None and isinstance(tid, int):
                    max_id = max(max_id, tid)
        return max_id + 1

    def iou(self, rect1, rect2):
        """
        Calculate Intersection over Union between two QRect objects.
        
        Args:
            rect1: First QRect
            rect2: Second QRect
            
        Returns:
            float: IoU value between 0 and 1
        """
        x_left = max(rect1.left(), rect2.left())
        y_top = max(rect1.top(), rect2.top())
        x_right = min(rect1.right(), rect2.right())
        y_bottom = min(rect1.bottom(), rect2.bottom())
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        rect1_area = rect1.width() * rect1.height()
        rect2_area = rect2.width() * rect2.height()
        union_area = rect1_area + rect2_area - intersection_area
        if union_area == 0:
            return 0.0
        return intersection_area / union_area
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    @log_exceptions
    def viat_convert_segmentation_to_bbox_project(self):
        """Converts segmentation polygons to bounding boxes and updates YOLO .txt files."""
        if not getattr(self, 'is_image_dataset', False):
            QMessageBox.warning(self, 'Warning', 'This feature is only for Image Datasets.')
            return

        reply = QMessageBox.question(self, 'Convert to Bounding Box Project',
                                     "This will convert all segmentations to bounding boxes and rewrite the dataset's YOLO .txt files in the same directory as the images. This cannot be undone.\n\nDo you want to proceed?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                                     
        if reply != QMessageBox.Yes:
            return

        # 1. Strip segmentations in memory
        for frame_num, annotations in self.frame_annotations.items():
            for ann in annotations:
                ann.segmentation = None
                
        # Update canvas
        if getattr(self, 'current_frame', -1) in self.frame_annotations:
            self.canvas.annotations = self.frame_annotations[self.current_frame]
            self.canvas.update()

        # 2. Rewrite .txt files
        class_list = list(self.canvas.class_colors.keys())
        class_to_id = {cls: i for i, cls in enumerate(class_list)}
        
        import cv2
        for frame_num, image_path in enumerate(self.image_files):
            base_name = os.path.splitext(image_path)[0]
            txt_filename = base_name + ".txt"
            
            if frame_num not in self.frame_annotations or not self.frame_annotations[frame_num]:
                if os.path.exists(txt_filename):
                    open(txt_filename, "w").close()
                continue
                
            try:
                img = cv2.imread(image_path)
                if img is not None:
                    image_height, image_width = img.shape[:2]
                else:
                    image_width, image_height = 640, 480
            except Exception:
                image_width, image_height = 640, 480
                
            with open(txt_filename, "w") as f:
                for annotation in self.frame_annotations[frame_num]:
                    class_id = class_to_id.get(annotation.class_name, 0)
                    rect = annotation.rect

                    x = rect.x()
                    y = rect.y()
                    w = rect.width()
                    h = rect.height()

                    x_center = (x + w / 2.0) / image_width
                    y_center = (y + h / 2.0) / image_height
                    norm_w = w / image_width
                    norm_h = h / image_height

                    # Ensure bounds
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    norm_w = max(0.0, min(1.0, norm_w))
                    norm_h = max(0.0, min(1.0, norm_h))

                    f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

        QMessageBox.information(self, "Success", "Project converted to Bounding Box and YOLO .txt files rewritten.")
        self.project_modified = True

    @log_exceptions
    def viat_import_segmentation_masks(self):
        """Imports segmentation masks and converts them to bounding boxes."""
        from PyQt5.QtWidgets import QMessageBox, QDialog
        from PyQt5.QtGui import QColor
        from PyQt5.QtCore import Qt
        if not hasattr(self, 'is_image_dataset') or not self.is_image_dataset:
            QMessageBox.warning(self, 'Error',
                'Importing segmentation masks is only supported for Image Datasets.'
                )
            return
        if not getattr(self, 'image_files', None):
            QMessageBox.warning(self, 'Error',
                'No images loaded in the dataset.')
            return
        from .widgets.import_masks_dialog import ImportMasksDialog
        from .utils.import_masks import import_segmentation_masks
        from .annotation import BoundingBox
        dialog = ImportMasksDialog(self, parent=self)
        if dialog.exec_() == QDialog.Accepted:
            color_to_class = dialog.get_mapping()
            masks_dir = dialog.get_masks_dir()
            if not color_to_class:
                QMessageBox.warning(self, 'Error',
                    'No colors mapped to classes.')
                return
            self.statusBar.showMessage('Importing segmentation masks...')
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                new_classes = set(color_to_class.values())
                for c in new_classes:
                    if c not in self.canvas.class_colors:
                        if hasattr(self, 'class_manager'):
                            self.class_manager.add_class(c, QColor(255, 0, 0))
                        else:
                            self.canvas.class_colors[c] = QColor(255, 0, 0)
                new_annotations, stats = import_segmentation_masks(rgb_files
                    =self.image_files, masks_dir=masks_dir, bbox_cls=
                    BoundingBox, color_to_class=color_to_class)
                imported_count = 0
                for frame_idx, bboxes in new_annotations.items():
                    if frame_idx not in self.frame_annotations:
                        self.frame_annotations[frame_idx] = []
                    for bbox in bboxes:
                        bbox.color = self.canvas.class_colors.get(bbox.
                            class_name, QColor(255, 0, 0))
                    self.frame_annotations[frame_idx].extend(bboxes)
                    imported_count += len(bboxes)
                if hasattr(self, 'update_annotation_list'):
                    self.update_annotation_list()
                self.canvas.update()
                self.project_modified = True
                QMessageBox.information(self, 'Import Complete',
                    f"""Processed {stats['images_processed']} images.
Extracted {stats['boxes_extracted']} bounding boxes."""
                    )
            except Exception as e:
                QMessageBox.warning(self, 'Import Error',
                    f'Failed to import masks: {e}')
            finally:
                QApplication.restoreOverrideCursor()
                self.statusBar.showMessage('Ready')


    @log_exceptions
    def compare_raya_annotations(self):
        """Show dialog to select two Raya files and compare them."""
        from viat.widgets.compare_raya_dialog import CompareRayaDialog
        dlg = CompareRayaDialog(self)
        if dlg.exec_() != QDialog.Accepted:
            return
            
        base_file = dlg.base_file
        mod_file = dlg.mod_file
        out_md = dlg.out_md
        
        from viat.utils.compare_raya import compare_annotations
        success, msg = compare_annotations(base_file, mod_file, out_md)
        if success:
            QMessageBox.information(self, 'Comparison Complete', f'Report saved to:\n{out_md}')
        else:
            QMessageBox.critical(self, 'Comparison Error', f'Failed to compare files:\n{msg}')

    @log_exceptions
    def open_evaluation_dialog(self):
        """Open the Model & Dataset Evaluation Dialog."""
        try:
            from viat.widgets.evaluation_dialog import EvaluationDialog
            dlg = EvaluationDialog(self)
            dlg.exec_()
        except Exception as e:
            import traceback
            QMessageBox.critical(
                self,
                "Evaluation Panel Error",
                f"Failed to open Evaluation Dialog:\n{str(e)}\n\nDetails:\n{traceback.format_exc()}"
            )

    @log_exceptions
    def view_labeler_analytics(self):

        """Generate labeler analytics report from a VIAT JSON project."""
        if not hasattr(self, 'project_file') or not self.project_file:
            json_file, _ = QFileDialog.getOpenFileName(self, 'Select VIAT Project File', '', 'JSON Files (*.json);;All Files (*)')
            if not json_file:
                return
        else:
            if self.project_modified:
                reply = QMessageBox.warning(self, 'Unsaved Changes', 
                                         'You must save the project before generating analytics to ensure all tracking data is included. Would you like to save now?',
                                         QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.Yes:
                    self.save_project()
                    if self.project_modified: # If they cancelled the save dialog
                        return
                else:
                    return
            json_file = self.project_file
        base_raya_file = None
        try:
            import json
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            analytics = data.setdefault('labeler_analytics', {})
            base_annotations = analytics.get('base_annotations')
            
            if not base_annotations:
                reply = QMessageBox.question(self, 'Base Pre-labels Missing', 
                                             'This project does not contain embedded base pre-labels.\nWould you like to select the original Base Annotation file (Raya/YOLO/etc) for comparison?',
                                             QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.Yes:
                    base_raya_file, _ = QFileDialog.getOpenFileName(self, 'Select Base Annotation File', '', 'All Files (*);;Text Files (*.txt);;JSON Files (*.json);;XML Files (*.xml)')
                    if base_raya_file:
                        try:
                            from viat.utils.file_operations import import_annotations as import_annotations_func
                            from viat.annotation import BoundingBox
                            
                            # Get canvas dimensions or default to 640x480
                            if self.canvas.pixmap:
                                image_width = self.canvas.pixmap.width()
                                image_height = self.canvas.pixmap.height()
                            else:
                                image_width = 640
                                image_height = 480
                                
                            res_parsed = import_annotations_func(
                                base_raya_file, BoundingBox, image_width, image_height, data.get('class_colors', {})
                            )
                            base_frames_parsed = res_parsed[2]
                            
                            loaded_base = {}
                            for f_num, objs in base_frames_parsed.items():
                                loaded_base[str(f_num)] = [obj.to_dict() for obj in objs]
                                
                            if loaded_base:
                                analytics['base_annotations'] = loaded_base
                                if hasattr(self, 'labeler_analytics') and self.project_file == json_file:
                                    self.labeler_analytics['base_annotations'] = loaded_base
                                try:
                                    with open(json_file, 'w') as f:
                                        json.dump(data, f, indent=4)
                                except Exception as save_err:
                                    print(f"Error saving imported base annotations: {save_err}")
                        except Exception as import_err:
                            QMessageBox.warning(self, 'Import Error', f'Failed to import base annotations: {import_err}')
                
                # Fallback if still not present
                if not analytics.get('base_annotations'):
                    reconstructed_base = {}
                    frame_anns = data.get('frame_annotations', {})
                    for frame_str, anns in frame_anns.items():
                        base_anns_in_frame = []
                        for ann in anns:
                            if ann.get('deleted', False):
                                continue
                            orig_src = ann.get('original_source', ann.get('source', 'manual'))
                            if orig_src in ('loaded', 'imported', 'detected'):
                                import copy
                                base_anns_in_frame.append(copy.deepcopy(ann))
                        if base_anns_in_frame:
                            reconstructed_base[frame_str] = base_anns_in_frame
                    
                    if reconstructed_base:
                        analytics['base_annotations'] = reconstructed_base
                        if hasattr(self, 'labeler_analytics') and self.project_file == json_file:
                            self.labeler_analytics['base_annotations'] = reconstructed_base
                        
                        try:
                            with open(json_file, 'w') as f:
                                json.dump(data, f, indent=4)
                        except Exception as save_err:
                            print(f"Error saving reconstructed base annotations: {save_err}")
        except Exception as e:
            print(f"Error checking for embedded base annotations: {e}")

        out_md = os.path.join(os.path.dirname(json_file), 'analytics_report.md')
        
        from viat.utils.analytics_report import generate_analytics_report
        success, msg = generate_analytics_report(json_file, out_md, base_raya_file)
        if success:
            QMessageBox.information(self, 'Analytics Complete', f'Report saved to:\n{out_md}')
        else:
            QMessageBox.critical(self, 'Analytics Error', f'Failed to generate report:\n{msg}')

class ClassMappingDialog(QDialog):

    def __init__(self, imported_classes, existing_classes, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Map Imported Classes')
        self.setMinimumWidth(400)
        self.imported_classes = imported_classes
        self.existing_classes = existing_classes
        self.mapping = {}
        self.init_ui()

    def init_ui(self):
        from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QDialogButtonBox, QScrollArea, QWidget
        layout = QVBoxLayout(self)
        info_label = QLabel(
            'The imported file contains classes. Please map them to existing classes or create new ones.'
            )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        self.combos = {}
        for imp_class in self.imported_classes:
            row_layout = QHBoxLayout()
            label = QLabel(f'Imported: {imp_class}')
            combo = QComboBox()
            combo.addItem(f'Create new class: {imp_class}')
            for ext_class in self.existing_classes:
                combo.addItem(f'Map to: {ext_class}')
            if imp_class in self.existing_classes:
                combo.setCurrentText(f'Map to: {imp_class}')
            self.combos[imp_class] = combo
            row_layout.addWidget(label)
            row_layout.addWidget(combo)
            scroll_layout.addLayout(row_layout)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.
            Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self):
        for imp_class, combo in self.combos.items():
            text = combo.currentText()
            if text.startswith('Create new class: '):
                self.mapping[imp_class] = text.replace('Create new class: ', ''
                    )
            else:
                self.mapping[imp_class] = text.replace('Map to: ', '')
        super().accept()

    def get_mapping(self):
        id_mapping = {}
        for idx, imp_class in enumerate(self.imported_classes):
            if imp_class in self.mapping:
                id_mapping[idx] = self.mapping[imp_class]
        return id_mapping

    