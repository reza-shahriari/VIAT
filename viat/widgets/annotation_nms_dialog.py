from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QDialogButtonBox, QWidget, QGroupBox, QListWidget,
    QListWidgetItem
)
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal

from viat.utils.annotation_nms import plan_cleanup, plan_removed_count, KEEP_LARGER, KEEP_OLDER


class _PlanWorker(QThread):
    """Computes a cleanup plan off the UI thread so large projects don't freeze the dialog."""
    progress = pyqtSignal(int, int, int)   # generation, done, total
    done = pyqtSignal(int, object)         # generation, plan (None if stopped)

    def __init__(self, generation, frame_annotations, frames, options, parent=None):
        super().__init__(parent)
        self.generation = generation
        self.frame_annotations = frame_annotations
        self.frames = frames
        self.options = options
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        plan = plan_cleanup(self.frame_annotations, self.frames, should_stop=lambda: self._stop,
                            on_progress=lambda n, t: self.progress.emit(self.generation, n, t),
                            **self.options)
        self.done.emit(self.generation, plan)


class AnnotationNMSDialog(QDialog):
    """
    Options for removing duplicate / ignore-region annotations. The removal plan is computed in a
    background thread (debounced) and shown as a live count; get_plan() returns it on accept.
    """

    SCOPE_CURRENT, SCOPE_ALL, SCOPE_RANGE = range(3)

    def __init__(self, frame_annotations, current_frame, total_frames, class_names, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Remove Duplicate / Overlapping Annotations")
        self.frame_annotations = frame_annotations
        self.current_frame = current_frame
        self.total_frames = max(1, total_frames)
        self._generation = 0
        self._workers = []
        self._plan = None

        layout = QVBoxLayout(self)

        # Scope
        scope_form = QFormLayout()
        self.cmb_scope = QComboBox()
        self.cmb_scope.addItems(["Current Frame", "All Frames", "Custom Range"])
        self.cmb_scope.setCurrentIndex(self.SCOPE_ALL)
        scope_form.addRow("Scope:", self.cmb_scope)

        self.range_widget = QWidget()
        range_layout = QHBoxLayout(self.range_widget)
        range_layout.setContentsMargins(0, 0, 0, 0)
        self.spin_start = QSpinBox()
        self.spin_start.setRange(1, self.total_frames)
        self.spin_start.setValue(current_frame + 1)
        self.spin_end = QSpinBox()
        self.spin_end.setRange(1, self.total_frames)
        self.spin_end.setValue(self.total_frames)
        range_layout.addWidget(QLabel("Start:"))
        range_layout.addWidget(self.spin_start)
        range_layout.addWidget(QLabel("End:"))
        range_layout.addWidget(self.spin_end)
        scope_form.addRow("", self.range_widget)
        layout.addLayout(scope_form)

        # Duplicate removal (NMS)
        self.grp_nms = QGroupBox("Remove duplicate boxes (NMS)")
        self.grp_nms.setCheckable(True)
        self.grp_nms.setChecked(True)
        nms_form = QFormLayout(self.grp_nms)
        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setRange(0.05, 0.95)
        self.spin_iou.setSingleStep(0.05)
        self.spin_iou.setValue(0.5)
        self.spin_iou.setToolTip("Boxes overlapping at or above this IoU are treated as the same object.")
        nms_form.addRow("IoU threshold:", self.spin_iou)

        nested_row = QWidget()
        nested_layout = QHBoxLayout(nested_row)
        nested_layout.setContentsMargins(0, 0, 0, 0)
        self.chk_nested = QCheckBox("Also boxes nested inside each other, when")
        self.chk_nested.setChecked(True)
        self.spin_nested = QSpinBox()
        self.spin_nested.setRange(10, 100)
        self.spin_nested.setValue(80)
        self.spin_nested.setSuffix(" %")
        nested_layout.addWidget(self.chk_nested)
        nested_layout.addWidget(self.spin_nested)
        nested_layout.addWidget(QLabel("of the smaller box is inside"))
        nested_layout.addStretch()
        nested_row.setToolTip(
            "A small box inside a big one has a low IoU, so IoU alone misses it.\n"
            "This treats them as duplicates when most of the smaller box is covered.")
        nms_form.addRow("", nested_row)

        self.cmb_keep = QComboBox()
        self.cmb_keep.addItem("Larger box", KEEP_LARGER)
        self.cmb_keep.addItem("Older box", KEEP_OLDER)
        self.cmb_keep.setToolTip("Manual / verified boxes are always kept first, whichever you choose.")
        nms_form.addRow("Keep:", self.cmb_keep)

        self.chk_class_agnostic = QCheckBox("Across classes (e.g. 'car' duplicate of 'vehicle')")
        self.chk_class_agnostic.setToolTip(
            "When off, only boxes of the same class can be duplicates of each other.")
        nms_form.addRow("", self.chk_class_agnostic)
        layout.addWidget(self.grp_nms)

        # Ignore regions
        self.grp_regions = QGroupBox("Ignore regions")
        self.grp_regions.setCheckable(True)
        self.grp_regions.setChecked(False)
        region_layout = QVBoxLayout(self.grp_regions)
        region_layout.addWidget(QLabel(
            "Boxes of the checked classes (e.g. 'shadow') remove every other\n"
            "box that lies mostly inside them."))
        self.lst_region_classes = QListWidget()
        self.lst_region_classes.setMaximumHeight(120)
        for name in class_names:
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Unchecked)
            self.lst_region_classes.addItem(item)
        region_layout.addWidget(self.lst_region_classes)
        region_form = QFormLayout()
        self.spin_region_overlap = QSpinBox()
        self.spin_region_overlap.setRange(5, 100)
        self.spin_region_overlap.setValue(60)
        self.spin_region_overlap.setSuffix(" %")
        self.spin_region_overlap.setToolTip("Remove a box when at least this much of its area is inside a region box.")
        region_form.addRow("Remove boxes inside by at least:", self.spin_region_overlap)
        self.chk_remove_regions = QCheckBox("Also delete the region boxes themselves")
        region_form.addRow("", self.chk_remove_regions)
        region_layout.addLayout(region_form)
        layout.addWidget(self.grp_regions)

        self.lbl_preview = QLabel()
        layout.addWidget(self.lbl_preview)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText("Remove")
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        # Recompute the plan shortly after the user stops changing options
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)
        self._debounce.timeout.connect(self._start_plan)

        for signal in (self.cmb_scope.currentIndexChanged, self.spin_start.valueChanged,
                       self.spin_end.valueChanged, self.spin_iou.valueChanged,
                       self.chk_nested.toggled, self.spin_nested.valueChanged,
                       self.cmb_keep.currentIndexChanged, self.chk_class_agnostic.toggled,
                       self.grp_nms.toggled, self.grp_regions.toggled,
                       self.lst_region_classes.itemChanged, self.spin_region_overlap.valueChanged,
                       self.chk_remove_regions.toggled):
            signal.connect(self._options_changed)
        self._options_changed()

    def get_frames(self):
        """0-indexed frames in the selected scope."""
        scope = self.cmb_scope.currentIndex()
        if scope == self.SCOPE_CURRENT:
            return [self.current_frame]
        if scope == self.SCOPE_ALL:
            return sorted(self.frame_annotations.keys())
        start, end = sorted((self.spin_start.value() - 1, self.spin_end.value() - 1))
        return [f for f in sorted(self.frame_annotations.keys()) if start <= f <= end]

    def get_options(self):
        """Keyword arguments for annotation_nms.clean_annotations / plan_cleanup."""
        region_classes = []
        if self.grp_regions.isChecked():
            for i in range(self.lst_region_classes.count()):
                item = self.lst_region_classes.item(i)
                if item.checkState() == Qt.Checked:
                    region_classes.append(item.text())
        return dict(
            remove_duplicates=self.grp_nms.isChecked(),
            iou_threshold=self.spin_iou.value(),
            class_agnostic=self.chk_class_agnostic.isChecked(),
            nested_overlap=self.spin_nested.value() / 100.0 if self.chk_nested.isChecked() else None,
            keep=self.cmb_keep.currentData(),
            region_classes=region_classes,
            region_overlap=self.spin_region_overlap.value() / 100.0,
            remove_regions=self.chk_remove_regions.isChecked(),
        )

    def get_plan(self):
        """{frame: kept_annotations} for the options shown when the dialog was accepted."""
        return self._plan or {}

    def _options_changed(self, *_):
        self.range_widget.setVisible(self.cmb_scope.currentIndex() == self.SCOPE_RANGE)
        self.spin_nested.setEnabled(self.chk_nested.isChecked())
        self._plan = None
        self._generation += 1
        for worker in self._workers:
            worker.stop()
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(False)
        self.lbl_preview.setText("Calculating...")
        self._debounce.start()

    def _start_plan(self):
        worker = _PlanWorker(self._generation, self.frame_annotations, self.get_frames(),
                             self.get_options(), self)
        worker.progress.connect(self._on_progress)
        worker.done.connect(self._on_plan_done)
        worker.finished.connect(lambda w=worker: self._workers.remove(w) if w in self._workers else None)
        self._workers.append(worker)
        worker.start()

    def _on_progress(self, generation, done, total):
        if generation == self._generation and total:
            self.lbl_preview.setText(f"Calculating... {100 * done // total}%")

    def _on_plan_done(self, generation, plan):
        if generation != self._generation or plan is None:
            return  # stale result: options changed while it was computing
        self._plan = plan
        removed = plan_removed_count(self.frame_annotations, plan)
        self.lbl_preview.setText(f"Will remove {removed} annotation(s) in {len(plan)} frame(s).")
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(removed > 0)

    def done(self, result):
        # Don't leave threads running against the annotations once the dialog closes
        self._debounce.stop()
        for worker in list(self._workers):
            worker.stop()
            worker.wait()
        super().done(result)
