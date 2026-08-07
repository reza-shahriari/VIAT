from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox,
    QRadioButton, QPushButton, QButtonGroup, QMessageBox, QSpinBox, QWidget,
    QListWidget, QListWidgetItem, QCheckBox, QDoubleSpinBox, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QGroupBox, QFrame,
    QScrollArea, QSizePolicy, QToolButton
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont
import re


# ──────────────────────────────────────────────────────────────────────────────
#  Model registry  (display name → internal model key)
# ──────────────────────────────────────────────────────────────────────────────
ZERO_SHOT_MODELS = [
    ('Existing Annotations (Hand-labeled)',  'existing_annotations'),
    ('YOLO-World Small',                     'yolov8s-world.pt'),
    ('YOLO-World Medium',                    'yolov8m-world.pt'),
    ('YOLO-World Large',                     'yolov8l-world.pt'),
    ('YOLO-World v2 Small',                  'yolov8s-worldv2.pt'),
    ('YOLO-World v2 Medium',                 'yolov8m-worldv2.pt'),
    ('YOLO-World v2 Large',                  'yolov8l-worldv2.pt'),
    ('YOLO-World v2 XLarge',                 'yolov8x-worldv2.pt'),
    ('YOLOv11 World Small',                  'yolo11s-world.pt'),
    ('YOLOv11 World Medium',                 'yolo11m-world.pt'),
    ('YOLOv11 World Large',                  'yolo11l-world.pt'),
    ('YOLOv11 World XLarge',                 'yolo11x-world.pt'),
    ('YOLOE 11l (segmentation)',             'yoloe-11l-seg.pt'),
    ('YOLOE 26s (segmentation)',             'yoloe-26s-seg.pt'),
    ('YOLOE 26x (segmentation)',             'yoloe-26x-seg.pt'),
    ('Grounding DINO Tiny',                  'IDEA-Research/grounding-dino-tiny'),
    ('Grounding DINO Base',                  'IDEA-Research/grounding-dino-base'),
    ('Florence-2 Base',                      'microsoft/Florence-2-base'),
    ('Florence-2 Large',                     'microsoft/Florence-2-large'),
    ('LocateAnything-3B',                    'nvidia/LocateAnything-3B'),
    ('SAM3 Text Prompt',                     'sam3.1_l.pt'),
]

SEG_REFINER_MODELS = [
    ('None',                                 None),
    ('SAM2 Fast (sam2_s.pt)',                'sam2_s.pt'),
    ('SAM2 Huge (sam2_l.pt)',                'sam2_l.pt'),
    ('SAM3 Fast (sam3_s.pt)',                'sam3_s.pt'),
    ('SAM3 Huge (sam3_l.pt)',                'sam3_l.pt'),
]


class CollapsibleSection(QWidget):
    """A titled group box whose contents can be shown/hidden."""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        self._is_collapsed = True

        self._toggle_btn = QToolButton()
        self._toggle_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle_btn.setArrowType(Qt.RightArrow)
        self._toggle_btn.setText(f'  {title}')
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(False)
        self._toggle_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        font = self._toggle_btn.font()
        font.setBold(True)
        self._toggle_btn.setFont(font)
        self._toggle_btn.setStyleSheet('QToolButton { border: none; padding: 4px; }')
        self._toggle_btn.clicked.connect(self._on_toggle)

        self._content = QWidget()
        self._content.setVisible(False)
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(16, 4, 4, 4)
        self._content_layout.setSpacing(6)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 4, 0, 4)
        outer.setSpacing(0)
        outer.addWidget(self._toggle_btn)
        outer.addWidget(sep)
        outer.addWidget(self._content)

    def _on_toggle(self, checked):
        self._is_collapsed = not checked
        self._content.setVisible(checked)
        self._toggle_btn.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)

    def add_widget(self, widget):
        self._content_layout.addWidget(widget)

    def add_layout(self, layout):
        self._content_layout.addLayout(layout)

    def content_layout(self):
        return self._content_layout


# ──────────────────────────────────────────────────────────────────────────────
#  Main Dialog
# ──────────────────────────────────────────────────────────────────────────────
class AutoAnnotateDialog(QDialog):

    def __init__(self, current_frame=0, total_frames=1, parent=None):
        super().__init__(parent)
        self.current_frame = current_frame
        self.total_frames = total_frames
        self._test_mode = False   # True when caller wants single-frame preview
        self.setWindowTitle('Auto Annotate')
        self.setMinimumWidth(560)
        self.setMinimumHeight(400)
        self._setup_ui()
        self._populate_existing_classes()

    # ──────────────────────────── UI ────────────────────────────────────────
    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(10)

        # ── Quick Mode ──────────────────────────────────────────────────────
        quick_box = QGroupBox('Quick Setup')
        quick_layout = QVBoxLayout(quick_box)
        quick_layout.setSpacing(8)

        # Class names
        cls_row = QHBoxLayout()
        cls_row.addWidget(QLabel('Classes to detect:'))
        self.classes_edit = QLineEdit()
        self.classes_edit.setPlaceholderText('e.g.  car, person, truck')
        self.classes_edit.setToolTip(
            'Comma-separated class names. These will be searched for in every frame.')
        cls_row.addWidget(self.classes_edit)
        quick_layout.addLayout(cls_row)

        # Model picker
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel('Detection model:'))
        self.model_combo = QComboBox()
        self.model_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        for display, _ in ZERO_SHOT_MODELS:
            self.model_combo.addItem(display)
        # Default: YOLO-World v2 Medium — good balance of speed vs accuracy
        self._set_combo_by_key(self.model_combo, ZERO_SHOT_MODELS, 'yolov8m-worldv2.pt')
        model_row.addWidget(self.model_combo)
        quick_layout.addLayout(model_row)

        # Confidence
        conf_row = QHBoxLayout()
        conf_row.addWidget(QLabel('Min confidence (%):'))
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(1, 100)
        self.threshold_spin.setValue(25)
        self.threshold_spin.setFixedWidth(70)
        self.threshold_spin.setToolTip(
            'Detections below this confidence score are ignored.')
        conf_row.addWidget(self.threshold_spin)
        conf_row.addStretch()
        quick_layout.addLayout(conf_row)

        # Frame scope
        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel('Frame scope:'))
        self.rb_current = QRadioButton('Current frame only')
        self.rb_all = QRadioButton('All frames')
        self.rb_all.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.rb_current)
        grp.addButton(self.rb_all)
        scope_row.addWidget(self.rb_current)
        scope_row.addWidget(self.rb_all)
        scope_row.addStretch()
        quick_layout.addLayout(scope_row)

        root.addWidget(quick_box)

        # ── Advanced ────────────────────────────────────────────────────────
        self._adv_section = CollapsibleSection('Advanced Options')

        # Per-class table (old power-user feature)
        self._adv_section.add_widget(QLabel(
            'Per-class configuration (overrides quick classes above when filled):'))
        self.classes_table = QTableWidget()
        self.classes_table.setColumnCount(6)
        self.classes_table.setHorizontalHeaderLabels([
            'Class Name', 'Action',
            'Extract Target (Helper)', 'Ignore Target (Helper)',
            'Rename To (Opt)', 'Deduplicate Against'])
        self.classes_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.classes_table.horizontalHeader().setStretchLastSection(True)
        self.classes_table.setSelectionMode(QAbstractItemView.NoSelection)
        self.classes_table.setMinimumHeight(150)
        self.classes_table.setMaximumHeight(200)
        self._adv_section.add_widget(self.classes_table)

        add_row_btn = QPushButton('+ Add Class Row')
        add_row_btn.setFixedWidth(140)
        add_row_btn.clicked.connect(lambda: self.add_new_class_row())
        self._adv_section.add_widget(add_row_btn)

        # Extra models
        self._adv_section.add_widget(QLabel('Additional detection models (multi-select):'))
        self.det_model_list = QListWidget()
        self.det_model_list.setSelectionMode(QAbstractItemView.MultiSelection)
        for display, _ in ZERO_SHOT_MODELS:
            self.det_model_list.addItem(display)
        self.det_model_list.setMaximumHeight(70)
        self.det_model_list.setToolTip(
            'Optionally run additional models in addition to the one selected above.')
        self._adv_section.add_widget(self.det_model_list)

        # Segmentation refiner
        seg_row = QHBoxLayout()
        seg_row.addWidget(QLabel('Segmentation refiner:'))
        self.seg_model_combo = QComboBox()
        for display, _ in SEG_REFINER_MODELS:
            self.seg_model_combo.addItem(display)
        seg_row.addWidget(self.seg_model_combo)
        self._adv_section.add_layout(seg_row)

        # Dedup IoU
        dedup_row = QHBoxLayout()
        dedup_row.addWidget(QLabel('Deduplication IoU threshold:'))
        self.dedup_spin = QDoubleSpinBox()
        self.dedup_spin.setRange(0.0, 1.0)
        self.dedup_spin.setSingleStep(0.05)
        self.dedup_spin.setValue(0.7)
        self.dedup_spin.setFixedWidth(80)
        dedup_row.addWidget(self.dedup_spin)
        dedup_row.addStretch()
        self._adv_section.add_layout(dedup_row)

        # Save segmentation
        self.chk_save_seg = QCheckBox('Save segmentation masks (polygons)')
        self.chk_save_seg.setChecked(True)
        self._adv_section.add_widget(self.chk_save_seg)

        root.addWidget(self._adv_section)

        # ── Buttons ──────────────────────────────────────────────────────────
        btn_layout = QHBoxLayout()

        self.btn_test = QPushButton('🔍  Test on Current Frame')
        self.btn_test.setToolTip('Run detection on the current frame only and preview results.')
        self.btn_test.clicked.connect(self._on_test)

        self.btn_run = QPushButton('▶  Run Auto Annotate')
        self.btn_run.setDefault(True)
        self.btn_run.clicked.connect(self.accept)

        btn_cancel = QPushButton('Cancel')
        btn_cancel.clicked.connect(self.reject)

        btn_layout.addWidget(self.btn_test)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_cancel)
        btn_layout.addWidget(self.btn_run)
        root.addLayout(btn_layout)

    # ──────────────────────────── Helpers ───────────────────────────────────
    @staticmethod
    def _set_combo_by_key(combo, model_list, key):
        for i, (_, k) in enumerate(model_list):
            if k == key:
                combo.setCurrentIndex(i)
                return

    def _populate_existing_classes(self):
        """Pre-fill the advanced class table with existing project classes."""
        parent = self.parent()
        if parent and hasattr(parent, 'canvas') and hasattr(parent.canvas, 'class_colors'):
            for cls_name in parent.canvas.class_colors.keys():
                self.add_new_class_row(cls_name)

    def _get_quick_classes(self):
        """Return class names from the quick-mode text field (comma-separated)."""
        text = self.classes_edit.text().strip()
        if not text:
            return []
        return [c.strip() for c in text.split(',') if c.strip()]

    def _get_advanced_classes_config(self):
        """Return classes_config list from the advanced table (if populated)."""
        classes_config = []
        for row in range(self.classes_table.rowCount()):
            name_item = self.classes_table.item(row, 0)
            if not name_item:
                continue
            class_name = name_item.text().strip()
            if not class_name:
                continue
            action_combo = self.classes_table.cellWidget(row, 1)
            action = action_combo.currentText() if action_combo else 'Detect (Zero-Shot)'
            extract_input = self.classes_table.cellWidget(row, 2)
            ignore_input  = self.classes_table.cellWidget(row, 3)
            rename_input  = self.classes_table.cellWidget(row, 4)
            dedup_input   = self.classes_table.cellWidget(row, 5)
            extract_prompt = extract_input.text().strip() if extract_input else ''
            ignore_prompt  = ignore_input.text().strip()  if ignore_input  else ''
            rename_to      = rename_input.text().strip()  if rename_input  else ''
            dedup_against  = dedup_input.text().strip()   if dedup_input   else ''
            if action != 'Ignore':
                classes_config.append({
                    'name':           class_name,
                    'action':         action,
                    'extract_prompt': extract_prompt,
                    'ignore_prompt':  ignore_prompt,
                    'rename_to':      rename_to,
                    'dedup_against':  dedup_against,
                })
        return classes_config

    def _build_classes_config(self):
        """
        Merge quick-mode class names with the advanced table.
        Advanced table rows take precedence; quick-mode names not already in
        the table are appended as plain Detect (Zero-Shot) rows.
        """
        adv = self._get_advanced_classes_config()
        adv_names_lower = {c['name'].lower() for c in adv}

        quick = self._get_quick_classes()
        for name in quick:
            if name.lower() not in adv_names_lower:
                adv.append({
                    'name':           name,
                    'action':         'Detect (Zero-Shot)',
                    'extract_prompt': '',
                    'ignore_prompt':  '',
                    'rename_to':      '',
                    'dedup_against':  '',
                })
        return adv

    def _build_det_models(self):
        """
        Primary model = what's selected in the quick-mode combo.
        Additional models = anything selected in the advanced multi-select list.
        Deduplicated.
        """
        primary_idx = self.model_combo.currentIndex()
        _, primary_key = ZERO_SHOT_MODELS[primary_idx]
        models = [primary_key] if primary_key else []

        for i in range(self.det_model_list.count()):
            item = self.det_model_list.item(i)
            if item.isSelected():
                _, key = ZERO_SHOT_MODELS[i]
                if key and key not in models:
                    models.append(key)

        return models

    def _build_config(self, single_frame=False):
        classes_config = self._build_classes_config()
        if not classes_config:
            return None

        det_models = self._build_det_models()
        if not det_models:
            QMessageBox.warning(self, 'Auto Annotate', 'Please select at least one detection model.')
            return None

        seg_idx = self.seg_model_combo.currentIndex()
        _, seg_key = SEG_REFINER_MODELS[seg_idx]

        if single_frame or self.rb_current.isChecked():
            start_frame = self.current_frame
            end_frame   = self.current_frame
        else:
            start_frame = 0
            end_frame   = self.total_frames - 1

        threshold = self.threshold_spin.value()
        return {
            'classes_config':    classes_config,
            'det_models':        det_models,
            'seg_model':         seg_key,
            'save_segmentation': self.chk_save_seg.isChecked(),
            'strategy':          'independent',
            'start_frame':       start_frame,
            'end_frame':         end_frame,
            'threshold':         threshold,
            'min_score':         threshold / 100.0,
            'dedup_iou':         self.dedup_spin.value(),
        }

    # ──────────────────────────── Slots ─────────────────────────────────────
    def _on_test(self):
        """Trigger accept() but mark it as a single-frame test run."""
        self._test_mode = True
        self.accept()

    def is_test_mode(self):
        return self._test_mode

    # ──────────────────────────── Public API ────────────────────────────────
    def get_config(self):
        return self._build_config(single_frame=self._test_mode)

    # ──────────────────────────── Advanced table ────────────────────────────
    def add_new_class_row(self, class_name=''):
        row = self.classes_table.rowCount()
        self.classes_table.insertRow(row)

        name_item = QTableWidgetItem(class_name)
        if class_name:
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
        self.classes_table.setItem(row, 0, name_item)

        action_combo = QComboBox()
        action_combo.addItems([
            'Detect (Zero-Shot)',
            'Use as Helper (SAM3 Refine)',
            'Remove Labels',
            'Ignore'])
        action_combo.setCurrentText('Ignore' if class_name else 'Detect (Zero-Shot)')
        self.classes_table.setCellWidget(row, 1, action_combo)

        extract_input = QLineEdit()
        extract_input.setPlaceholderText('e.g. truck chassis')
        self.classes_table.setCellWidget(row, 2, extract_input)

        ignore_input = QLineEdit()
        ignore_input.setPlaceholderText('e.g. cargo')
        self.classes_table.setCellWidget(row, 3, ignore_input)

        rename_input = QLineEdit()
        rename_input.setPlaceholderText('e.g. Red Car')
        self.classes_table.setCellWidget(row, 4, rename_input)

        dedup_input = QLineEdit()
        dedup_input.setPlaceholderText('e.g. car, vehicle, *')
        self.classes_table.setCellWidget(row, 5, dedup_input)

        def _sync_fields(text, ei=extract_input, ii=ignore_input, ri=rename_input, di=dedup_input):
            if 'Helper' in text:
                ei.setEnabled(True); ii.setEnabled(True)
                ri.setEnabled(True); di.setEnabled(False); di.clear()
            elif 'Detect' in text:
                ei.setEnabled(True); ii.setEnabled(True)
                ri.setEnabled(False); ri.clear(); di.setEnabled(True)
            else:
                for w in (ei, ii, ri, di):
                    w.setEnabled(False); w.clear()

        action_combo.currentTextChanged.connect(_sync_fields)
        _sync_fields(action_combo.currentText())

    # ──────────────────────────── Legacy compat ─────────────────────────────
    def populate_classes_table(self):
        """Kept for backwards-compatibility with any code that calls this."""
        self._populate_existing_classes()
