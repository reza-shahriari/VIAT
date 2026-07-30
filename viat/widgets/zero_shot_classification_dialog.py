import os
import json
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox,
    QPushButton, QFileDialog, QDoubleSpinBox, QMessageBox, QGroupBox,
    QFormLayout, QTabWidget, QWidget, QTextEdit, QTableWidget,
    QTableWidgetItem, QHeaderView, QAbstractItemView, QToolButton,
    QSplitter, QFrame, QCheckBox, QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QColor, QIcon


# ---------------------------------------------------------------------------
# Helper: small section title
# ---------------------------------------------------------------------------
def _section_label(text):
    lbl = QLabel(text)
    f = lbl.font()
    f.setBold(True)
    lbl.setFont(f)
    return lbl


class ZeroShotClassificationDialog(QDialog):
    """
    Dialog for zero-shot classification-based annotation refinement.

    Supports two independent modes:
      • Correctness Check  – make sure that each annotation still belongs to
                             one of its *expected* candidate classes (e.g. a
                             "red car" must remain "car", not get re-labelled
                             to some sub-class).  Annotations that fail are
                             marked uncertain.
      • Mislabel Check     – verify that no annotation was placed under a
                             completely wrong top-level class (e.g. a car
                             labelled as human). Candidates here come from ALL
                             top-level classes so CLIP can pick the closest one.

    Both modes work with or without a JSON preset file.
    """

    def __init__(self, parent=None, known_classes=None):
        super().__init__(parent)
        self.setWindowTitle("Zero-Shot Classification Refiner")
        self.setMinimumWidth(640)
        self.setMinimumHeight(560)
        self._known_classes = known_classes or []   # injected from parent
        self._build_ui()
        self._apply_styles()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(8)

        # ── Tabs ──────────────────────────────────────────────────────
        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self.tabs.addTab(self._build_mode_tab(), "🎯  Mode & Classes")
        self.tabs.addTab(self._build_model_tab(), "⚙️  Model Settings")
        self.tabs.addTab(self._build_json_tab(), "📄  JSON Preset")

        # ── Bottom bar ────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setFrameShadow(QFrame.Sunken)
        root.addWidget(sep)

        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("▶  Run Classification")
        self.btn_run.setDefault(True)
        self.btn_run.clicked.connect(self._on_run)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_run)
        btn_row.addWidget(self.btn_cancel)
        root.addLayout(btn_row)

    # ── Tab 1: Mode & Classes ─────────────────────────────────────────
    def _build_mode_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(10)

        # Mode selector
        mode_box = QGroupBox("What do you want to check?")
        mode_inner = QVBoxLayout(mode_box)

        self.rb_correctness = QCheckBox(
            "Correctness Check  –  each annotation must still score highest "
            "within its own allowed classes  (e.g. 'red car' must win against "
            "only car variants, not be confused with other top-level classes)"
        )
        # NOTE: do NOT connect toggled yet — the target group widgets don't exist yet.
        # setChecked is called after construction below.

        self.rb_mislabel = QCheckBox(
            "Mislabel Check  –  verify that no annotation was assigned to a "
            "completely wrong class  (e.g. a car labelled as human).  "
            "CLIP compares every annotation against ALL defined top-level classes."
        )

        mode_inner.addWidget(self.rb_correctness)
        mode_inner.addWidget(self.rb_mislabel)
        lay.addWidget(mode_box)

        # ── Correctness: class → allowed aliases ───────────────────
        self.correctness_group = QGroupBox(
            "Correctness rules  –  for each dataset class, list the CLIP labels "
            "it is allowed to match (the first one is the 'canonical' label)"
        )
        cg_lay = QVBoxLayout(self.correctness_group)

        hint = QLabel(
            "<i>Example: class 'car' → allowed labels: car, vehicle, sedan, "
            "truck.  CLIP will score the crop against only these labels.  "
            "If the top score is not one of them, the annotation is flagged uncertain.</i>"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888;")
        cg_lay.addWidget(hint)

        self.correctness_table = QTableWidget(0, 2)
        self.correctness_table.setHorizontalHeaderLabels(
            ["Dataset Class", "Allowed CLIP Labels  (comma-separated)"])
        self.correctness_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Interactive)
        self.correctness_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch)
        self.correctness_table.setSelectionBehavior(
            QAbstractItemView.SelectRows)
        self.correctness_table.setAlternatingRowColors(True)
        cg_lay.addWidget(self.correctness_table)

        ct_btns = QHBoxLayout()
        btn_add_row = QPushButton("+ Add Row")
        btn_add_row.clicked.connect(self._add_correctness_row)
        btn_del_row = QPushButton("✕ Remove Selected")
        btn_del_row.clicked.connect(self._del_correctness_row)
        ct_btns.addWidget(btn_add_row)
        ct_btns.addWidget(btn_del_row)
        ct_btns.addStretch()
        cg_lay.addLayout(ct_btns)

        # Pre-populate from known_classes if any
        if self._known_classes:
            for cls in self._known_classes:
                self._add_correctness_row(cls, cls)

        lay.addWidget(self.correctness_group)

        # ── Mislabel: top-level classes list ──────────────────────
        self.mislabel_group = QGroupBox(
            "Mislabel rules  –  list ALL top-level class names that exist in "
            "your dataset.  CLIP will score each annotation against this full list."
        )
        mg_lay = QVBoxLayout(self.mislabel_group)

        hint2 = QLabel(
            "<i>Example: car, human, bicycle, truck.  "
            "If an annotation labelled 'car' scores highest as 'human', "
            "it will be flagged as uncertain / mislabelled.</i>"
        )
        hint2.setWordWrap(True)
        hint2.setStyleSheet("color: #888;")
        mg_lay.addWidget(hint2)

        self.mislabel_classes_edit = QLineEdit()
        self.mislabel_classes_edit.setPlaceholderText(
            "car, human, bicycle, truck, dog, …")
        mg_lay.addWidget(self.mislabel_classes_edit)

        # Pre-populate from known_classes
        if self._known_classes:
            self.mislabel_classes_edit.setText(
                ", ".join(self._known_classes))

        self.mislabel_group.setVisible(False)
        lay.addWidget(self.mislabel_group)

        # Connect signals now that both groups exist, then set initial state.
        self.rb_correctness.toggled.connect(self._on_mode_changed)
        self.rb_mislabel.toggled.connect(self._on_mode_changed)
        self.rb_correctness.setChecked(True)   # fires _on_mode_changed safely

        lay.addStretch()
        return w

    # ── Tab 2: Model Settings ─────────────────────────────────────────
    def _build_model_tab(self):
        w = QWidget()
        lay = QFormLayout(w)
        lay.setSpacing(10)
        lay.setContentsMargins(12, 12, 12, 12)

        self.model_combo = QComboBox()
        self.model_combo.addItems([
            "openai/clip-vit-base-patch32",
            "openai/clip-vit-large-patch14",
            "google/siglip-base-patch16-224",
            "google/siglip-so400m-patch14-384",
        ])
        self.model_combo.setEditable(True)
        lay.addRow("Classification Model:", self.model_combo)

        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(0.3)
        self.confidence_spin.setToolTip(
            "Minimum softmax probability for the top class. "
            "If the top class scores below this the annotation is flagged uncertain."
        )
        lay.addRow("Min Confidence:", self.confidence_spin)

        self.overlap_margin_spin = QDoubleSpinBox()
        self.overlap_margin_spin.setRange(0.0, 1.0)
        self.overlap_margin_spin.setSingleStep(0.01)
        self.overlap_margin_spin.setValue(0.05)
        self.overlap_margin_spin.setToolTip(
            "If the gap between the #1 and #2 score is smaller than this value, "
            "the result is considered ambiguous and the annotation is flagged uncertain."
        )
        lay.addRow("Ambiguity Margin:", self.overlap_margin_spin)

        pad_note = QLabel(
            "<i>The crop is padded by 15 % on each side to give CLIP "
            "some background context.</i>"
        )
        pad_note.setWordWrap(True)
        pad_note.setStyleSheet("color: #888;")
        lay.addRow("", pad_note)

        return w

    # ── Tab 3: JSON Preset ────────────────────────────────────────────
    def _build_json_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setSpacing(8)
        lay.setContentsMargins(12, 12, 12, 12)

        info = QLabel(
            "A <b>JSON preset</b> lets you save and reload your class rules so "
            "you don't have to retype them every time.  It is <b>completely "
            "optional</b> — you can run the classifier without one.\n\n"
            "Use the buttons below to import an existing preset into the "
            "Mode &amp; Classes tab, or export the current settings to a file."
        )
        info.setWordWrap(True)
        lay.addWidget(info)

        # Load row
        load_row = QHBoxLayout()
        self.json_path_input = QLineEdit()
        self.json_path_input.setPlaceholderText(
            "Optional: path to an existing JSON preset file…")
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_json)
        btn_load = QPushButton("⬆  Import")
        btn_load.setToolTip("Load the selected JSON and fill in the form above.")
        btn_load.clicked.connect(self._import_json)
        load_row.addWidget(self.json_path_input)
        load_row.addWidget(btn_browse)
        load_row.addWidget(btn_load)
        lay.addLayout(load_row)

        # Save row
        save_row = QHBoxLayout()
        btn_export = QPushButton("⬇  Export current settings…")
        btn_export.clicked.connect(self._export_json)
        save_row.addWidget(btn_export)
        save_row.addStretch()
        lay.addLayout(save_row)

        # JSON format reference
        ref_box = QGroupBox("JSON format reference")
        ref_inner = QVBoxLayout(ref_box)
        example = QTextEdit()
        example.setReadOnly(True)
        example.setFont(QFont("Monospace", 9))
        example.setMaximumHeight(260)
        example.setPlainText(
            '// --- Correctness Check preset ---\n'
            '{\n'
            '  "mode": "correctness",\n'
            '  "rules": {\n'
            '    "car":   ["car", "vehicle", "sedan", "truck"],\n'
            '    "human": ["person", "human", "pedestrian"],\n'
            '    "bike":  ["bicycle", "bike", "motorcycle"]\n'
            '  },\n'
            '  "overlap_groups": [\n'
            '    ["car", "vehicle", "truck"],\n'
            '    ["person", "human", "pedestrian"]\n'
            '  ],\n'
            '  "global_fallback": []\n'
            '}\n\n'
            '// --- Mislabel Check preset ---\n'
            '{\n'
            '  "mode": "mislabel",\n'
            '  "classes": ["car", "human", "bicycle", "truck"]\n'
            '}\n\n'
            '// --- Combined preset (both modes) ---\n'
            '{\n'
            '  "mode": "both",\n'
            '  "rules": { "car": ["car", "vehicle"], "human": ["person"] },\n'
            '  "classes": ["car", "human", "bicycle"],\n'
            '  "overlap_groups": [],\n'
            '  "global_fallback": []\n'
            '}'
        )
        ref_inner.addWidget(example)
        lay.addWidget(ref_box)

        lay.addStretch()
        return w

    # ------------------------------------------------------------------
    # Styles
    # ------------------------------------------------------------------
    def _apply_styles(self):
        self.setStyleSheet("""
            QDialog {
                background: #1e1e2e;
                color: #cdd6f4;
            }
            QGroupBox {
                border: 1px solid #45475a;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 4px;
                color: #cba6f7;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QTabWidget::pane {
                border: 1px solid #45475a;
                border-radius: 4px;
            }
            QTabBar::tab {
                background: #313244;
                color: #cdd6f4;
                padding: 6px 16px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }
            QTabBar::tab:selected {
                background: #45475a;
                color: #cba6f7;
            }
            QLineEdit, QComboBox, QDoubleSpinBox, QTextEdit {
                background: #313244;
                border: 1px solid #45475a;
                border-radius: 4px;
                color: #cdd6f4;
                padding: 4px 6px;
            }
            QTableWidget {
                background: #181825;
                alternate-background-color: #1e1e2e;
                color: #cdd6f4;
                gridline-color: #313244;
                border: 1px solid #45475a;
            }
            QHeaderView::section {
                background: #313244;
                color: #cba6f7;
                padding: 4px;
                border: none;
            }
            QPushButton {
                background: #45475a;
                color: #cdd6f4;
                border: 1px solid #585b70;
                border-radius: 5px;
                padding: 5px 14px;
            }
            QPushButton:hover {
                background: #585b70;
            }
            QPushButton:pressed {
                background: #313244;
            }
            QPushButton[text="▶  Run Classification"] {
                background: #89b4fa;
                color: #1e1e2e;
                font-weight: bold;
            }
            QPushButton[text="▶  Run Classification"]:hover {
                background: #b4d0fb;
            }
            QCheckBox {
                color: #cdd6f4;
                spacing: 6px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:unchecked {
                border: 2px solid #585b70;
                border-radius: 3px;
                background: #313244;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #89b4fa;
                border-radius: 3px;
                background: #89b4fa;
            }
        """)

    # ------------------------------------------------------------------
    # Slot: mode checkbox toggling
    # ------------------------------------------------------------------
    def _on_mode_changed(self):
        correctness_on = self.rb_correctness.isChecked()
        mislabel_on = self.rb_mislabel.isChecked()
        self.correctness_group.setVisible(correctness_on)
        self.mislabel_group.setVisible(mislabel_on)

    # ------------------------------------------------------------------
    # Correctness table helpers
    # ------------------------------------------------------------------
    def _add_correctness_row(self, class_name="", labels=""):
        row = self.correctness_table.rowCount()
        self.correctness_table.insertRow(row)
        self.correctness_table.setItem(row, 0, QTableWidgetItem(class_name))
        self.correctness_table.setItem(row, 1, QTableWidgetItem(labels))

    def _del_correctness_row(self):
        rows = sorted(
            {idx.row() for idx in self.correctness_table.selectedIndexes()},
            reverse=True
        )
        for r in rows:
            self.correctness_table.removeRow(r)

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------
    def _browse_json(self):
        default_dir = os.path.join(os.getcwd(), "checkpoints", "zero_shot")
        os.makedirs(default_dir, exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(
            self, "Select JSON Preset File", default_dir, "JSON Files (*.json)"
        )
        if path:
            self.json_path_input.setText(path)

    def _import_json(self):
        path = self.json_path_input.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "File Not Found",
                                "Please select a valid JSON file first.")
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            QMessageBox.critical(self, "JSON Error",
                                 f"Could not parse JSON:\n{e}")
            return

        mode = data.get("mode", "correctness")

        # Set mode checkboxes
        self.rb_correctness.setChecked(mode in ("correctness", "both"))
        self.rb_mislabel.setChecked(mode in ("mislabel", "both"))

        # Fill correctness table
        if "rules" in data:
            self.correctness_table.setRowCount(0)
            for cls, labels in data["rules"].items():
                self._add_correctness_row(cls, ", ".join(labels))

        # Fill mislabel classes
        if "classes" in data:
            classes = data["classes"]
            if isinstance(classes, list):
                self.mislabel_classes_edit.setText(", ".join(classes))

        # Model settings if present
        if "model" in data:
            idx = self.model_combo.findText(data["model"])
            if idx >= 0:
                self.model_combo.setCurrentIndex(idx)
            else:
                self.model_combo.setCurrentText(data["model"])
        if "min_confidence" in data:
            self.confidence_spin.setValue(float(data["min_confidence"]))
        if "overlap_margin" in data:
            self.overlap_margin_spin.setValue(float(data["overlap_margin"]))

        QMessageBox.information(self, "Imported",
                                f"Settings loaded from:\n{path}")

    def _export_json(self):
        default_dir = os.path.join(os.getcwd(), "checkpoints", "zero_shot")
        os.makedirs(default_dir, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save JSON Preset", default_dir, "JSON Files (*.json)"
        )
        if not path:
            return
        if not path.endswith(".json"):
            path += ".json"

        data = self._collect_config()
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            QMessageBox.information(self, "Saved", f"Preset saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    # ------------------------------------------------------------------
    # Collect & validate
    # ------------------------------------------------------------------
    def _collect_config(self):
        correctness_on = self.rb_correctness.isChecked()
        mislabel_on = self.rb_mislabel.isChecked()

        if correctness_on and mislabel_on:
            mode = "both"
        elif mislabel_on:
            mode = "mislabel"
        else:
            mode = "correctness"

        # Correctness rules
        rules = {}
        for row in range(self.correctness_table.rowCount()):
            cls_item = self.correctness_table.item(row, 0)
            lbl_item = self.correctness_table.item(row, 1)
            cls = cls_item.text().strip() if cls_item else ""
            lbl = lbl_item.text().strip() if lbl_item else ""
            if cls:
                labels = [l.strip() for l in lbl.split(",") if l.strip()]
                if not labels:
                    labels = [cls]
                rules[cls] = labels

        # Mislabel classes
        raw = self.mislabel_classes_edit.text().strip()
        classes = [c.strip() for c in raw.split(",") if c.strip()]

        return {
            "mode": mode,
            "rules": rules,
            "classes": classes,
            "overlap_groups": [],
            "global_fallback": [],
            "model": self.model_combo.currentText(),
            "min_confidence": self.confidence_spin.value(),
            "overlap_margin": self.overlap_margin_spin.value(),
        }

    # ------------------------------------------------------------------
    # Validate & accept
    # ------------------------------------------------------------------
    def _on_run(self):
        cfg = self._collect_config()
        mode = cfg["mode"]

        if mode in ("correctness", "both") and not cfg["rules"]:
            QMessageBox.warning(
                self, "No Rules Defined",
                "You selected Correctness Check but have not defined any class "
                "rules.\n\nPlease add at least one row in the 'Mode & Classes' "
                "tab (or switch to Mislabel-only mode)."
            )
            return

        if mode in ("mislabel", "both") and not cfg["classes"]:
            QMessageBox.warning(
                self, "No Classes Defined",
                "You selected Mislabel Check but have not listed any classes.\n\n"
                "Please enter your dataset class names in the 'Mode & Classes' tab."
            )
            return

        self.accept()

    # ------------------------------------------------------------------
    # Public API consumed by main.py
    # ------------------------------------------------------------------
    def get_config(self):
        """Return the full configuration dict ready for ZeroShotClassifierManager."""
        return self._collect_config()
