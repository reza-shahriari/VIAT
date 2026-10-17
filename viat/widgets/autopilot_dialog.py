"""Settings dialog for Tools > Autopilot (unattended video labeling)."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QApplication, QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
                             QTableWidget, QTableWidgetItem, QPushButton, QListWidget,
                             QListWidgetItem, QSpinBox, QDoubleSpinBox, QComboBox, QCheckBox,
                             QGroupBox, QDialogButtonBox, QHeaderView, QAbstractItemView,
                             QMessageBox, QScrollArea, QSizePolicy, QFrame, QWidget)

from .auto_annotate_dialog import ZERO_SHOT_MODELS
from ..utils.autopilot import AutopilotConfig
from ..utils.autopilot.learned import LearnedJudge
from ..utils.autopilot.weights import is_available, missing_message, KNOWN_WEIGHTS

# Detectors Autopilot supports on top of the Auto Label list (own local-only loaders).
EXTRA_MODELS = [
    ('OWLv2 Base', 'google/owlv2-base-patch16-ensemble'),
    ('OWLv2 Large', 'google/owlv2-large-patch14-ensemble'),
]
JUDGE_INFO = [
    ('clip', 'CLIP crop check (cheap; class confusion, parts, shadows)'),
    ('sam_zoom', 'SAM 3 zoom-in re-detection (strong; small objects)'),
    ('vlm', 'Vision-language model (Qwen-VL; slowest, only for unsure tracks)'),
    ('learned', 'Learned judge (trained on your review decisions / labeled videos)'),
]
VLM_MODELS = [k for k in KNOWN_WEIGHTS if 'VL' in k]

# preset -> (stride, tiles, flip)
PRESETS = [
    ('Maximum quality - every frame, 2x2 tiles + flip (slowest)', (1, 2, True)),
    ('High quality - every frame, 2x2 tiles', (1, 2, False)),
    ('Balanced - every frame, full image only', (1, 0, False)),
    ('Fast - every 3rd frame, gaps interpolated', (3, 0, False)),
]

ADVANCED = [
    # attr, label, min, max, step, decimals, tooltip
    ('candidate_score', 'Candidate score', 0.01, 0.9, 0.01, 2, 'Detections kept as candidates (low on purpose: recall first).'),
    ('high_score', 'Track-match score', 0.05, 0.95, 0.05, 2, 'Detections this confident are matched to tracks first.'),
    ('birth_score', 'New-object score', 0.05, 0.99, 0.05, 2, 'A new track starts only from a detection at least this confident.'),
    ('box_slack', 'Box padding (slack)', 0.0, 0.20, 0.01, 2, 'Fraction of width/height added to edges to prevent clipping car corners/bumpers.'),
    ('min_hits', 'Confirm after (detections)', 1, 20, 1, 0, 'A tentative track becomes real after this many matched detections.'),
    ('max_lost_sec', 'Keep lost track (s)', 0.1, 10.0, 0.1, 1, 'How long a lost object may stay unseen before its track ends.'),
    ('ext_max_sec', 'Extend tracks up to (s)', 0.0, 10.0, 0.5, 1, 'Walk each track backward/forward along weak detections.'),
    ('gap_fill_sec', 'Fill gaps up to (s)', 0.0, 10.0, 0.5, 1, 'Interpolate (camera-compensated) gaps up to this length.'),
    ('link_gap_sec', 'Join fragments across (s)', 0.0, 10.0, 0.5, 1, 'Join two tracks of the same object separated by up to this gap.'),
    ('review_risk', 'Review if risk >=', 0.05, 0.95, 0.05, 2, 'Tracks at or above this risk go to the review list.'),
    ('miss_score', 'Possible-miss score', 0.05, 0.9, 0.05, 2, 'Untracked candidates this confident that persist are listed as possible misses.'),
    ('audit_size', 'Audit sample (frames)', 5, 1000, 5, 0, 'Random frames to check. 0 errors in 59 frames proves <=5%, in 299 proves <=1%.'),
]


class AutopilotDialog(QDialog):
    def __init__(self, parent, class_names, current_frame, total_frames, checkpoints_dir, last_config=None, seed_counts=None):
        super().__init__(parent)
        self.setWindowTitle('Autopilot - Unattended Video Labeling')
        self.total_frames = total_frames
        self.checkpoints_dir = checkpoints_dir
        last = AutopilotConfig.from_dict(last_config) if last_config else AutopilotConfig()
        self._adv = {}

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(10)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        content = QWidget()
        content.setMinimumWidth(0)
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        lay = QVBoxLayout(content)
        lay.setContentsMargins(2, 2, 8, 2)
        lay.setSpacing(10)
        scroll.setWidget(content)
        root.addWidget(scroll, 1)

        intro = QLabel(
            'Detects every frame, follows each object from when it enters until it leaves, repairs tracks '
            'using the whole video, and flags only the tracks that are probably wrong. '
            'Afterwards you review the flagged list and check a few random frames.\n'
            'Existing labels are never changed; results are added as unverified "autopilot" boxes.')
        intro.setWordWrap(True)
        lay.addWidget(intro)

        # ---------------------------------------------------------------- classes
        cls_box = QGroupBox('Classes and prompts (comma-separated prompts map to one class)')
        cl = QVBoxLayout(cls_box)
        cl.setContentsMargins(10, 10, 10, 10)
        cl.setSpacing(8)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(['Class', 'Prompts'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setMinimumHeight(140)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        cl.addWidget(self.table)
        prev = {c['name']: ', '.join(c.get('prompts') or []) for c in last.classes}
        for name in class_names:
            self._add_row(name, prev.get(name, name), checked=(name in prev) if prev else True)
        for name, prompts in prev.items():
            if name not in class_names:
                self._add_row(name, prompts, checked=True)
        row = QHBoxLayout()
        b_add, b_del = QPushButton('Add class'), QPushButton('Remove selected')
        b_add.clicked.connect(lambda: self._add_row('new_class', 'new class', True))
        b_del.clicked.connect(lambda: [self.table.removeRow(r) for r in
                                       sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)])
        row.addWidget(b_add)
        row.addWidget(b_del)
        row.addStretch()
        cl.addLayout(row)
        lay.addWidget(cls_box)

        # ---------------------------------------------------------- visual seeds
        seed_box = QGroupBox('Visual examples (optional)')
        seed_layout = QVBoxLayout(seed_box)
        seed_layout.setContentsMargins(10, 8, 10, 8)
        seed_layout.setSpacing(6)

        self.seed_status = QLabel()
        self.seed_status.setWordWrap(True)
        if seed_counts:
            total_seeds = sum(seed_counts.values())
            summary = ', '.join(f"{cls} ({cnt})" for cls, cnt in sorted(seed_counts.items()))
            self.seed_status.setText(f"<b>Visual seeds available:</b> {total_seeds} verified box(es) found ({summary}).")
            self.seed_status.setStyleSheet("color: #2e7d32; padding: 2px;")
        else:
            self.seed_status.setText("<i>No verified boxes found in project yet. Draw a few boxes for difficult objects on any frame to use as visual prompts.</i>")
            self.seed_status.setStyleSheet("color: #757575; padding: 2px;")
        seed_layout.addWidget(self.seed_status)

        self.use_exemplars = QCheckBox(
            'Use verified boxes already drawn in this frame range as visual examples')
        self.use_exemplars.setChecked(last.use_exemplars or bool(seed_counts))
        self.use_exemplars.setToolTip(
            'Useful for OWLv2: draw a few boxes for difficult objects first, then use their appearance '
            'as one-shot queries on every frame. This does not fine-tune the model.')
        seed_layout.addWidget(self.use_exemplars)

        seed_row = QHBoxLayout()
        seed_row.addWidget(QLabel('Examples per class:'))
        self.exemplar_limit = QSpinBox()
        self.exemplar_limit.setRange(1, 10)
        self.exemplar_limit.setValue(max(1, min(10, int(getattr(last, 'exemplar_limit', 3)))))
        self.exemplar_limit.setToolTip('At most this many manually verified boxes per class are collected.')
        seed_row.addWidget(self.exemplar_limit)

        seed_row.addSpacing(16)
        seed_row.addWidget(QLabel('Prompt mode:'))
        self.prompt_mode = QComboBox()
        self.prompt_mode.addItem('Hybrid (Visual + Text)', 'hybrid')
        self.prompt_mode.addItem('Visual Exemplars Only', 'visual_only')
        self.prompt_mode.addItem('Text Prompts Only', 'text_only')
        cur_mode = getattr(last, 'prompt_mode', 'hybrid')
        idx = self.prompt_mode.findData(cur_mode)
        if idx >= 0:
            self.prompt_mode.setCurrentIndex(idx)
        self.prompt_mode.setToolTip(
            'Hybrid: Uses visual crops for seed objects and text prompts for general detection.\n'
            'Visual Only: Detects only objects matching the visual examples.\n'
            'Text Only: Ignores visual crops and detects using text prompts.')
        seed_row.addWidget(self.prompt_mode)

        seed_row.addSpacing(16)
        seed_row.addWidget(QLabel('Visual threshold:'))
        self.visual_query_threshold = QDoubleSpinBox()
        self.visual_query_threshold.setRange(0.04, 0.60)
        self.visual_query_threshold.setSingleStep(0.02)
        self.visual_query_threshold.setDecimals(2)
        self.visual_query_threshold.setValue(float(getattr(last, 'visual_query_threshold', 0.10)))
        self.visual_query_threshold.setToolTip(
            'OWLv2 image-guided similarity threshold (default 0.10).\n'
            'Prevents background false positives (values below 0.08 often match walls, grass, or pavement).')
        seed_row.addWidget(self.visual_query_threshold)
        seed_row.addStretch()
        seed_layout.addLayout(seed_row)

        seed_hint = QLabel(
            'Draw a few boxes for difficult objects on any frame (e.g. frame 1). '
            'OWLv2 uses their appearance as visual queries across all frames; other detectors use text prompts.')
        self._configure_hint(seed_hint)
        seed_layout.addWidget(seed_hint)
        lay.addWidget(seed_box)

        # ------------------------------------------------ boundary refinement (SAM)
        sam_box = QGroupBox('Boundary refinement (SAM)')
        sl = QVBoxLayout(sam_box)
        sl.setContentsMargins(10, 8, 10, 8)
        sl.setSpacing(6)
        self.refine_with_sam = QCheckBox(
            'Refine detected boxes with SAM (ensures whole object is captured)')
        self.refine_with_sam.setChecked(getattr(last, 'refine_with_sam', True))
        self.refine_with_sam.setToolTip(
            'Passes candidate bounding boxes to SAM (Segment Anything) to segment exact object contours.\n'
            'Snaps the bounding box to the physical object envelope, capturing corners, bumpers, heads,\n'
            'and feet without clipping, and eliminates coordinate averaging noise across models.')
        sl.addWidget(self.refine_with_sam)

        sam_row = QHBoxLayout()
        sam_row.addWidget(QLabel('SAM refiner model:'))
        self.sam_refine_model = QComboBox()
        sam_candidates = [
            ('SAM 2.1 Large (sam2.1_l.pt)', 'sam2.1_l.pt'),
            ('SAM 3.1 Large (sam3.1_l.pt)', 'sam3.1_l.pt'),
            ('SAM 2 Large (sam2_l.pt)', 'sam2_l.pt'),
            ('SAM 2.1 Small (sam2.1_s.pt)', 'sam2.1_s.pt'),
            ('SAM 2 Small (sam2_s.pt)', 'sam2_s.pt'),
            ('SAM 3 Large (sam3_l.pt)', 'sam3_l.pt'),
        ]
        cur_sam = getattr(last, 'sam_refine_model', 'sam2.1_l.pt')
        for display, key in sam_candidates:
            avail = is_available(key)
            item_text = display if avail else f"{display}  (missing)"
            self.sam_refine_model.addItem(item_text, key)
            if not avail:
                m_item = self.sam_refine_model.model().item(self.sam_refine_model.count() - 1)
                if m_item:
                    m_item.setEnabled(False)
        s_idx = self.sam_refine_model.findData(cur_sam)
        if s_idx >= 0 and self.sam_refine_model.model().item(s_idx) and self.sam_refine_model.model().item(s_idx).isEnabled():
            self.sam_refine_model.setCurrentIndex(s_idx)
        else:
            for i in range(self.sam_refine_model.count()):
                item = self.sam_refine_model.model().item(i)
                if item and item.isEnabled():
                    self.sam_refine_model.setCurrentIndex(i)
                    break
        sam_row.addWidget(self.sam_refine_model, 1)
        sl.addLayout(sam_row)
        lay.addWidget(sam_box)

        # ----------------------------------------------------------------- models
        mdl_box = QGroupBox('Detectors (several = agreement-weighted fusion)')
        ml = QVBoxLayout(mdl_box)
        ml.setContentsMargins(10, 10, 10, 10)
        ml.setSpacing(8)
        self.models = QListWidget()
        chosen = set(last.models) or {self._default_model()}
        for display, key in [m for m in ZERO_SHOT_MODELS if m[1] != 'existing_annotations'] + EXTRA_MODELS:
            available = is_available(key)
            item = QListWidgetItem(display + ('' if available else '  (weights missing)'))
            item.setData(Qt.UserRole, key)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if key in chosen and available else Qt.Unchecked)
            if not available:  # never download: show what is needed instead
                item.setFlags(item.flags() & ~Qt.ItemIsEnabled)
                item.setToolTip(missing_message(key))
            self.models.addItem(item)
        self.models.setMinimumHeight(120)
        self.models.setMaximumHeight(180)
        self.models.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        ml.addWidget(self.models)
        lay.addWidget(mdl_box)

        # ----------------------------------------------------------------- judges
        jbox = QGroupBox('Judges (second opinion on every track)')
        jl = QVBoxLayout(jbox)
        jl.setContentsMargins(10, 10, 10, 10)
        jl.setSpacing(6)
        self.judge_checks = {}
        vlm_local = [k for k in VLM_MODELS if is_available(k)]
        for key, label in JUDGE_INFO:
            cb = QCheckBox(label)
            if key == 'clip':
                ok, why = is_available(last.clip_model), missing_message(last.clip_model)
            elif key == 'sam_zoom':
                ok, why = is_available(last.sam_judge_model), missing_message(last.sam_judge_model)
            elif key == 'vlm':
                ok = bool(vlm_local)
                why = 'weights needed (one of): ' + '; '.join(missing_message(k) for k in VLM_MODELS)
            else:
                ok, why = LearnedJudge.load() is not None, \
                    'not trained yet: review some tracks, then press "Retrain judge" in the review panel'
            cb.setChecked(ok and key in last.judges)
            cb.setEnabled(ok)
            if not ok:
                cb.setToolTip(why)
                cb.setText(label + '  (unavailable)')
            self.judge_checks[key] = cb
            jl.addWidget(cb)
        self.vlm_model = QComboBox()
        self.vlm_model.addItems(vlm_local or [last.vlm_model])
        if last.vlm_model in vlm_local:
            self.vlm_model.setCurrentText(last.vlm_model)
        self.vlm_model.setEnabled(bool(vlm_local))
        vrow = QHBoxLayout()
        vrow.addWidget(QLabel('VLM:'))
        vrow.addWidget(self.vlm_model, 1)
        jl.addLayout(vrow)
        rrow = QHBoxLayout()
        self.auto_reject = QCheckBox('Auto-reject tracks the judges score at or below')
        self.auto_reject.setChecked(last.auto_reject)
        self.reject_below = QDoubleSpinBox()
        self.reject_below.setRange(0.01, 0.9)
        self.reject_below.setSingleStep(0.05)
        self.reject_below.setValue(last.reject_below)
        self.reject_below.setToolTip('Rejected tracks stay visible in the review panel and can be restored. '
                                     'autopilot_benchmark.py suggests a value from your labeled videos.')
        rrow.addWidget(self.auto_reject)
        rrow.addWidget(self.reject_below)
        rrow.addStretch()
        jl.addLayout(rrow)
        lay.addWidget(jbox)

        # ------------------------------------------------------------ range/quality
        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        self.start = QSpinBox()
        self.end = QSpinBox()
        for sb in (self.start, self.end):
            sb.setRange(1, max(1, total_frames))
        self.start.setValue(1)
        self.end.setValue(max(1, total_frames))
        rng = QHBoxLayout()
        rng.addWidget(self.start)
        rng.addWidget(QLabel('to'))
        rng.addWidget(self.end)
        b_all = QPushButton('Whole video')
        b_all.clicked.connect(lambda: (self.start.setValue(1), self.end.setValue(max(1, total_frames))))
        rng.addWidget(b_all)
        form.addRow('Frames:', rng)
        self.preset = QComboBox()
        for label, _ in PRESETS:
            self.preset.addItem(label)
        self.preset.setCurrentIndex(self._preset_index(last))
        form.addRow('Quality:', self.preset)
        lay.addLayout(form)

        # ---------------------------------------------------------------- output
        self.replace_prev = QCheckBox('Replace earlier Autopilot boxes in this range (manual labels are kept)')
        self.replace_prev.setChecked(True)
        self.skip_existing = QCheckBox('Skip objects that are already labeled (overlap >= 60%)')
        self.skip_existing.setChecked(True)
        lay.addWidget(self.replace_prev)
        lay.addWidget(self.skip_existing)

        # -------------------------------------------------------------- advanced
        adv = QGroupBox('Advanced')
        adv.setCheckable(True)
        adv.setChecked(False)
        af = QFormLayout()
        af.setHorizontalSpacing(12)
        af.setVerticalSpacing(7)
        holder = QGroupBox()
        holder.setFlat(True)
        holder.setLayout(af)
        for attr, label, lo, hi, step, dec, tip in ADVANCED:
            w = QSpinBox() if dec == 0 else QDoubleSpinBox()
            if dec:
                w.setDecimals(dec)
            w.setRange(lo, hi)
            w.setSingleStep(step)
            w.setValue(getattr(last, attr))
            w.setToolTip(tip)
            self._adv[attr] = w
            af.addRow(label + ':', w)
        self.cut_detection = QCheckBox('Split at scene cuts')
        self.cut_detection.setChecked(last.cut_detection)
        af.addRow(self.cut_detection)
        self.video_calibration = QCheckBox('Adapt score calibration to each video')
        self.video_calibration.setChecked(last.video_calibration)
        af.addRow(self.video_calibration)
        self.fusion_box_mode = QComboBox()
        self.fusion_box_mode.addItem('Best model box (no coordinate blur; recommended with SAM)', 'best')
        self.fusion_box_mode.addItem('Weighted coordinate average (legacy)', 'average')
        self.fusion_box_mode.addItem('Enclosing union envelope', 'union')
        cur_fusion = getattr(last, 'fusion_box_mode', 'best')
        f_idx = self.fusion_box_mode.findData(cur_fusion)
        if f_idx >= 0:
            self.fusion_box_mode.setCurrentIndex(f_idx)
        self.fusion_box_mode.setToolTip(
            'How candidate boxes from agreeing models are combined:\n'
            'Best model: keeps the sharpest box from the most confident model.\n'
            'Weighted average: blends coordinates together (can blur edges).\n'
            'Enclosing union: bounding envelope of all agreeing models.')
        af.addRow('Fusion box mode:', self.fusion_box_mode)
        al = QVBoxLayout(adv)
        al.setContentsMargins(10, 8, 10, 8)
        al.setSpacing(6)
        al.addWidget(holder)
        holder.setVisible(False)
        adv.toggled.connect(holder.setVisible)
        lay.addWidget(adv)

        buttons = QDialogButtonBox()
        self.run_btn = buttons.addButton('Run Autopilot', QDialogButtonBox.AcceptRole)
        buttons.addButton(QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._set_responsive_size()

    # ------------------------------------------------------------------ helpers
    def _set_responsive_size(self):
        """Choose a useful initial size without exceeding the available screen."""
        # ``QWidget.screen()`` is not exposed by some PyQt5/Qt5 builds used by
        # VIAT.  Prefer it when available, but keep the desktop fallback so the
        # dialog can still be constructed on those versions and in offscreen
        # smoke tests.
        screen = None
        desktop = None
        screen_number = 0
        screen_getter = getattr(self, 'screen', None)
        if callable(screen_getter):
            screen = screen_getter()
        if screen is None and self.parentWidget() is not None:
            parent_screen = getattr(self.parentWidget(), 'screen', None)
            if callable(parent_screen):
                screen = parent_screen()
        if screen is None:
            desktop = QApplication.desktop()
            if desktop is not None:
                try:
                    screen_number = desktop.screenNumber(self)
                except TypeError:
                    screen_number = desktop.screenNumber()
                screen = desktop.screen(screen_number if screen_number >= 0 else 0)
        if screen is None:
            screen = QApplication.primaryScreen()
        if screen is None:
            self.resize(760, 720)
            return

        if hasattr(screen, 'availableGeometry'):
            available = screen.availableGeometry()
        elif desktop is not None:
            # QApplication.desktop().screen() returns a QWidget in PyQt5,
            # while its geometry is exposed by the desktop object itself.
            available = desktop.availableGeometry(screen_number if screen_number >= 0 else 0)
        else:
            primary = QApplication.primaryScreen()
            available = primary.availableGeometry() if primary is not None else None
        if available is None:
            self.resize(760, 720)
            return
        max_width = max(1, available.width() - 32)
        max_height = max(1, available.height() - 48)

        # Keep the dialog comfortably readable on a desktop, but allow it to
        # shrink below those preferences on a small or tiled display.  The
        # scroll area handles the remaining vertical content in either case.
        preferred_width = min(900, max(560, int(available.width() * 0.72)))
        preferred_height = min(900, max(420, int(available.height() * 0.84)))
        self.setMinimumSize(min(560, max_width), min(360, max_height))
        self.resize(min(max_width, preferred_width), min(max_height, preferred_height))

    def _add_row(self, name, prompts, checked):
        r = self.table.rowCount()
        self.table.insertRow(r)
        item = QTableWidgetItem(name)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self.table.setItem(r, 0, item)
        self.table.setItem(r, 1, QTableWidgetItem(prompts))

    @staticmethod
    def _configure_hint(label):
        label.setWordWrap(True)
        label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

    def _default_model(self):
        for key in ('sam3.1_l.pt', 'yolov8x-worldv2.pt', 'yoloe-26x-seg.pt', 'yolov8m-worldv2.pt'):
            if is_available(key):
                return key
        return 'yolov8m-worldv2.pt'

    @staticmethod
    def _preset_index(cfg):
        for i, (_, (stride, tiles, flip)) in enumerate(PRESETS):
            if (cfg.stride, cfg.tiles, cfg.flip_tta) == (stride, tiles, flip):
                return i
        return 2

    def _accept(self):
        cfg = self.get_config()
        if not cfg.classes:
            QMessageBox.warning(self, 'Autopilot', 'Check at least one class with a prompt.')
            return
        if not cfg.models:
            QMessageBox.warning(self, 'Autopilot', 'Check at least one detector.')
            return
        if cfg.end_frame < cfg.start_frame:
            QMessageBox.warning(self, 'Autopilot', 'The frame range is empty.')
            return
        self.accept()

    def get_config(self):
        classes = []
        for r in range(self.table.rowCount()):
            name_item, prompt_item = self.table.item(r, 0), self.table.item(r, 1)
            if not name_item or name_item.checkState() != Qt.Checked:
                continue
            name = name_item.text().strip()
            prompts = [p.strip() for p in (prompt_item.text() if prompt_item else '').split(',') if p.strip()]
            if name:
                classes.append({'name': name, 'prompts': prompts or [name]})
        models = [self.models.item(i).data(Qt.UserRole) for i in range(self.models.count())
                  if self.models.item(i).checkState() == Qt.Checked]
        stride, tiles, flip = PRESETS[self.preset.currentIndex()][1]
        cfg = AutopilotConfig(classes=classes, models=models, start_frame=self.start.value() - 1,
                              end_frame=self.end.value() - 1, stride=stride, tiles=tiles, flip_tta=flip,
                              use_exemplars=self.use_exemplars.isChecked(),
                              exemplar_limit=self.exemplar_limit.value(),
                              prompt_mode=self.prompt_mode.currentData(),
                              visual_query_threshold=self.visual_query_threshold.value(),
                              refine_with_sam=self.refine_with_sam.isChecked(),
                              sam_refine_model=self.sam_refine_model.currentData(),
                              fusion_box_mode=self.fusion_box_mode.currentData(),
                              cut_detection=self.cut_detection.isChecked(),
                              video_calibration=self.video_calibration.isChecked(),
                              judges=[k for k, cb in self.judge_checks.items() if cb.isChecked()],
                              vlm_model=self.vlm_model.currentText(), auto_reject=self.auto_reject.isChecked(),
                              reject_below=self.reject_below.value())
        for attr, w in self._adv.items():
            setattr(cfg, attr, w.value())
        cfg.low_score = min(cfg.low_score, cfg.high_score)
        return cfg

    def apply_options(self):
        return {'replace_previous': self.replace_prev.isChecked(), 'skip_existing': self.skip_existing.isChecked()}
