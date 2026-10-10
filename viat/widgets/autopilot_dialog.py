"""Settings dialog for Tools > Autopilot (unattended video labeling)."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QTableWidget,
                             QTableWidgetItem, QPushButton, QListWidget, QListWidgetItem, QSpinBox,
                             QDoubleSpinBox, QComboBox, QCheckBox, QGroupBox, QDialogButtonBox,
                             QHeaderView, QAbstractItemView, QMessageBox)

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
    def __init__(self, parent, class_names, current_frame, total_frames, checkpoints_dir, last_config=None):
        super().__init__(parent)
        self.setWindowTitle('Autopilot - Unattended Video Labeling')
        self.setMinimumWidth(640)
        self.total_frames = total_frames
        self.checkpoints_dir = checkpoints_dir
        last = AutopilotConfig.from_dict(last_config) if last_config else AutopilotConfig()
        self._adv = {}

        lay = QVBoxLayout(self)
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
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(['Class', 'Prompts'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
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

        # ----------------------------------------------------------------- models
        mdl_box = QGroupBox('Detectors (several = agreement-weighted fusion)')
        ml = QVBoxLayout(mdl_box)
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
        self.models.setMaximumHeight(150)
        ml.addWidget(self.models)
        lay.addWidget(mdl_box)

        # ----------------------------------------------------------------- judges
        jbox = QGroupBox('Judges (second opinion on every track)')
        jl = QVBoxLayout(jbox)
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
        al = QVBoxLayout(adv)
        al.addWidget(holder)
        holder.setVisible(False)
        adv.toggled.connect(holder.setVisible)
        lay.addWidget(adv)

        buttons = QDialogButtonBox()
        self.run_btn = buttons.addButton('Run Autopilot', QDialogButtonBox.AcceptRole)
        buttons.addButton(QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

    # ------------------------------------------------------------------ helpers
    def _add_row(self, name, prompts, checked):
        r = self.table.rowCount()
        self.table.insertRow(r)
        item = QTableWidgetItem(name)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
        self.table.setItem(r, 0, item)
        self.table.setItem(r, 1, QTableWidgetItem(prompts))

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
