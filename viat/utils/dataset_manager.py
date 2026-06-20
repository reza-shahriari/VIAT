"""
Dataset detection, scanning & loading for VIAT.

Designed for Roboflow YOLO exports but handles every common layout:

  Layout A -- single folder, images + labels mixed:
      dataset/
        img1.jpg
        img1.txt
        img2.jpg
        img2.txt
        data.yaml

  Layout B -- images/ + labels/ subfolders:
      dataset/
        images/
          img1.jpg
        labels/
          img1.txt
        data.yaml

  Layout C -- train/valid/test splits, each split is Layout A or B:
      dataset/
        data.yaml
        train/
          images/  labels/
        valid/
          images/  labels/
        test/
          images/  labels/

Class-name resolution (priority order, with conflict warnings):
  1. data.yaml (Roboflow / Ultralytics) -- ``names`` list or dict
  2. classes.txt / obj.names / labels.txt
  3. Inferred from label class indices (numeric fallback, ``class_0``...)

Label formats are pluggable -- see utils/label_formats/__init__.py.
YOLO is the default and is tried first.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    import yaml  # PyYAML -- optional but very common
except ImportError:  # pragma: no cover
    yaml = None

from .label_formats import PRIORITY, get_format, all_formats, LabelParseError

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")

SPLIT_NAMES = ("train", "valid", "val", "test", "validation")
# Aliases normalized to canonical split names
SPLIT_ALIASES = {"val": "valid", "validation": "valid"}

CLASS_FILE_NAMES = ("classes.txt", "obj.names", "labels.txt")


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #


@dataclass
class SplitInfo:
    name: str  # canonical: train / valid / test (or "root" for no splits)
    path: str
    image_dir: str
    label_dirs: List[str]
    images: List[str] = field(default_factory=list)  # absolute paths
    label_format: Optional[str] = None  # detected format name

    def __repr__(self):
        return (
            f"SplitInfo(name={self.name!r}, images={len(self.images)}, "
            f"format={self.label_format})"
        )


@dataclass
class DatasetInfo:
    root: str
    layout: str  # "single_mixed" | "images_labels" | "splits_single" | "splits_sep" | "simple"
    splits: List[SplitInfo] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    classes_source: Optional[str] = None  # "data.yaml" | "classes.txt" | "inferred"
    classes_conflict: Optional[str] = None  # warning text if sources disagree
    label_format: Optional[str] = None  # global default format

    @property
    def all_images(self) -> List[str]:
        out = []
        for s in self.splits:
            out.extend(s.images)
        return out

    @property
    def image_count(self) -> int:
        return sum(len(s.images) for s in self.splits)


# --------------------------------------------------------------------------- #
# Public API expected by main.py
# --------------------------------------------------------------------------- #


def detect_folder_type(folder_path: str) -> str:
    """Quick classifier used by main.py to decide simple-folder vs dataset.

    Returns "dataset" if the folder looks like a labeled dataset (has labels/,
    a data.yaml, classes.txt, or train/valid/test subfolders), else "simple".
    """
    if not os.path.isdir(folder_path):
        return "simple"
    try:
        entries = set(os.listdir(folder_path))
    except OSError:
        return "simple"

    # Roboflow / Ultralytics markers
    if "data.yaml" in entries or "data.yml" in entries or "dataset.yaml" in entries or "dataset.yml" in entries:
        return "dataset"
    # images/labels split
    if "images" in entries and "labels" in entries:
        return "dataset"
    # splits
    lower = {e.lower() for e in entries}
    if any(s in lower for s in ("train", "valid", "val", "test")):
        # but only treat as dataset if at least one split has images or
        # an images/labels structure
        for s in ("train", "valid", "val", "test"):
            sp = os.path.join(folder_path, s)
            if os.path.isdir(sp):
                try:
                    sub = set(os.listdir(sp))
                except OSError:
                    continue
                if sub & {"images", "labels"} or any(
                    f.lower().endswith(IMAGE_EXTENSIONS) for f in sub
                ):
                    return "dataset"
    # classes.txt / obj.names next to images
    if any(c in entries for c in CLASS_FILE_NAMES):
        return "dataset"
    # COCO-style shared annotation file
    if any(
        f.lower().endswith((".coco.json",)) or f.lower() in ("annotations.json", "_annotations.coco.json")
        for f in entries
    ):
        return "dataset"
    # any .xml (pascal voc) alongside images
    if any(f.lower().endswith(".xml") for f in entries):
        return "dataset"
    return "simple"


def scan_dataset(folder_path: str) -> DatasetInfo:
    """Fully scan a dataset folder and return a :class:`DatasetInfo`."""
    info = DatasetInfo(root=folder_path, layout="simple", splits=[])

    # 1. Class names (resolve early so we can warn about conflicts)
    _resolve_classes(info)

    # 2. Detect layout & splits
    _detect_layout_and_splits(info)

    # 3. For each split, collect images + detect label format
    for split in info.splits:
        split.images = _list_images(split.image_dir)
        split.label_format, _ = _detect_label_format_for_split(split, info)

    # 4. Global default format (from splits, majority vote)
    fmt_votes: Dict[str, int] = {}
    for s in info.splits:
        if s.label_format:
            fmt_votes[s.label_format] = fmt_votes.get(s.label_format, 0) + len(s.images)
    if fmt_votes:
        info.label_format = max(fmt_votes, key=fmt_votes.get)

    return info


def load_dataset_into_app(
    app,
    info: DatasetInfo,
    bbox_cls,
    *,
    splits_to_load: Optional[List[str]] = None,
):
    """Load *info* into the VIAT main window.

    Args:
        app: the VideoAnnotationTool main window.
        info: scanned dataset.
        bbox_cls: the BoundingBox class (used to build annotation objects).
        splits_to_load: which splits to load (names). None = all (show all by
            default, per user requirement). "root" loads the no-split case.

    Returns:
        dict with keys: image_files (list[str]), frame_to_split (list[str]),
        per_split_counts (dict), classes (list[str]), warnings (list[str]).
    """
    from PyQt5.QtCore import QRect
    from PyQt5.QtGui import QColor
    import random
    import cv2

    warnings = []
    image_files: List[str] = []
    frame_to_split: List[str] = []
    per_split: Dict[str, int] = {}

    # Keep only class colors/attributes that are present in the new dataset
    old_colors = dict(getattr(app.canvas, "class_colors", {}) or {})
    old_attributes = dict(getattr(app.canvas, "class_attributes", {}) or {})
    
    new_colors = {}
    new_attributes = {}
    
    for cls in info.classes:
        if cls in old_colors:
            new_colors[cls] = old_colors[cls]
        else:
            new_colors[cls] = QColor(
                random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            )
            
        if cls in old_attributes:
            new_attributes[cls] = old_attributes[cls]
        else:
            new_attributes[cls] = {
                "Size": {"type": "int", "default": -1, "min": 0, "max": 100},
                "Quality": {"type": "int", "default": -1, "min": 0, "max": 100},
            }
            
    app.canvas.class_colors = new_colors
    app.canvas.class_attributes = new_attributes
    app.class_attributes = app.canvas.class_attributes  # legacy alias

    # Reset frame annotations (caller usually already did, but be safe)
    app.frame_annotations = {}

    target_splits = {s.name for s in info.splits}
    if splits_to_load is not None:
        target_splits = set(splits_to_load)

    for split in info.splits:
        if split.name not in target_splits:
            continue
        fmt = get_format(split.label_format or info.label_format or "yolo")
        if fmt is None:
            warnings.append(f"Split {split.name}: unknown format, skipped.")
            continue

        for img_path in split.images:
            frame_idx = len(image_files)
            image_files.append(img_path)
            frame_to_split.append(split.name)
            per_split[split.name] = per_split.get(split.name, 0) + 1

            # Get image size (needed by YOLO de-normalization)
            img_size = _image_size(img_path)
            if img_size is None:
                warnings.append(f"Could not read size: {os.path.basename(img_path)}")
                continue

            try:
                label_path = fmt.find_label_file(img_path, split.label_dirs)
            except Exception as e:
                warnings.append(f"{os.path.basename(img_path)}: {e}")
                label_path = None

            if not label_path:
                continue

            try:
                boxes = fmt.load(label_path, img_size, info.classes)
            except LabelParseError as e:
                warnings.append(str(e))
                continue

            if not boxes:
                continue

            anns = []
            for b in boxes:
                cls_name = b["class_name"]
                if cls_name not in app.canvas.class_colors:
                    app.canvas.class_colors[cls_name] = QColor(
                        random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                    )
                    app.canvas.class_attributes[cls_name] = {
                        "Size": {"type": "int", "default": -1, "min": 0, "max": 100},
                        "Quality": {"type": "int", "default": -1, "min": 0, "max": 100},
                    }
                    if cls_name not in info.classes:
                        info.classes.append(cls_name)

                rect = QRect(b["x"], b["y"], max(1, b["w"]), max(1, b["h"]))
                color = app.canvas.class_colors[cls_name]
                ann = bbox_cls(
                    rect=rect,
                    class_name=cls_name,
                    attributes=b.get("attributes", {}),
                    color=color,
                    source=b.get("source", "manual"),
                    score=b.get("score", 1.0),
                    segmentation=b.get("segmentation"),
                )
                # Set verified flag (for viat_json, accepted=True -> verified)
                if "verified" in b:
                    ann.verified = bool(b["verified"])
                anns.append(ann)
            app.frame_annotations[frame_idx] = anns

    return {
        "image_files": image_files,
        "frame_to_split": frame_to_split,
        "per_split_counts": per_split,
        "classes": info.classes,
        "warnings": warnings,
    }


def load_viat_json_for_video(app, json_path, bbox_cls, *, frame_offset=0):
    """Load a VIAT custom JSON annotation file into a VIDEO project.

    Unlike load_dataset_into_app (which is for image datasets), this loads
    annotations for an already-open video, keyed by frame number.

    Args:
        app: the VideoAnnotationTool main window (must have a video loaded).
        json_path: path to the VIAT JSON file.
        bbox_cls: the BoundingBox class.
        frame_offset: if the JSON frame keys don't start at 0 (e.g. the video
            was trimmed), add this offset to each frame key.

    Returns:
        dict: {frames_loaded, actors_loaded, classes_found, warnings}
    """
    from PyQt5.QtCore import QRect
    from PyQt5.QtGui import QColor
    from .label_formats.viat_json import ViatJsonLabelFormat
    import random

    warnings = []
    fmt = ViatJsonLabelFormat()
    fmt._parse(json_path)
    all_frames = fmt.load_all_frames()

    if not all_frames:
        return {"frames_loaded": 0, "actors_loaded": 0, "classes_found": [], "warnings": ["No frames found in JSON"]}

    # Collect all classes and actors
    classes_found = set()
    actor_ids = set()
    total_actors = 0

    # Ensure class_colors has all classes
    existing_colors = dict(getattr(app.canvas, "class_colors", {}) or {})
    if not hasattr(app.canvas, "class_attributes") or app.canvas.class_attributes is None:
        app.canvas.class_attributes = {}

    for frame_num, boxes in all_frames.items():
        for b in boxes:
            classes_found.add(b["class_name"])
            if b.get("actor_id"):
                actor_ids.add(b["actor_id"])
            total_actors += 1

    for cls in classes_found:
        if cls not in existing_colors:
            existing_colors[cls] = QColor(
                random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            )
        if cls not in app.canvas.class_attributes:
            app.canvas.class_attributes[cls] = {
                "Size": {"type": "int", "default": -1, "min": 0, "max": 100},
                "Quality": {"type": "int", "default": -1, "min": 0, "max": 100},
            }
    app.canvas.class_colors = existing_colors
    app.class_attributes = app.canvas.class_attributes

    # Load into frame_annotations
    frames_loaded = 0
    for frame_num, boxes in all_frames.items():
        actual_frame = frame_num + frame_offset
        if actual_frame < 0 or actual_frame >= getattr(app, "total_frames", 10**9):
            continue

        anns = []
        for b in boxes:
            rect = QRect(b["x"], b["y"], max(1, b["w"]), max(1, b["h"]))
            color = app.canvas.class_colors.get(b["class_name"], QColor(255, 0, 0))
            ann = bbox_cls(
                rect=rect,
                class_name=b["class_name"],
                attributes=b.get("attributes", {}),
                color=color,
                source=b.get("source", "manual"),
                score=b.get("score", 1.0),
                segmentation=b.get("segmentation"),
            )
            if "verified" in b:
                ann.verified = bool(b["verified"])
            anns.append(ann)
        app.frame_annotations[actual_frame] = anns
        frames_loaded += 1

    return {
        "frames_loaded": frames_loaded,
        "actors_loaded": total_actors,
        "classes_found": sorted(classes_found),
        "actor_ids": sorted(actor_ids),
        "warnings": warnings,
    }


# --------------------------------------------------------------------------- #
# Class-name resolution
# --------------------------------------------------------------------------- #


def _resolve_classes(info: DatasetInfo) -> None:
    yaml_classes = _parse_data_yaml_classes(info.root)
    txt_classes = _parse_classes_txt(info.root)

    # If splits exist, also look inside them for class files
    if not txt_classes:
        for s in SPLIT_NAMES:
            sp = os.path.join(info.root, s)
            if os.path.isdir(sp):
                txt_classes = _parse_classes_txt(sp)
                if txt_classes:
                    break

    if yaml_classes and txt_classes:
        if list(yaml_classes) == list(txt_classes):
            info.classes = list(yaml_classes)
            info.classes_source = "data.yaml + classes.txt (agree)"
        else:
            info.classes = list(yaml_classes)  # yaml wins
            info.classes_source = "data.yaml"
            info.classes_conflict = (
                f"data.yaml names {yaml_classes!r} differ from "
                f"classes.txt {txt_classes!r}. Using data.yaml. "
                f"Delete one source if this is wrong."
            )
    elif yaml_classes:
        info.classes = list(yaml_classes)
        info.classes_source = "data.yaml"
    elif txt_classes:
        info.classes = list(txt_classes)
        info.classes_source = "classes.txt"
    else:
        info.classes = []
        info.classes_source = None  # will be inferred later


def _parse_data_yaml_classes(root: str) -> Optional[List[str]]:
    if yaml is None:
        # Minimal fallback parser (handles the common ``names: [a, b, c]``
        # and ``names:\n  0: a`` forms) so VIAT works without PyYAML.
        return _parse_data_yaml_fallback(root)
    for name in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml"):
        p = os.path.join(root, name)
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
            except Exception:
                continue
            if isinstance(data, dict) and "names" in data:
                names = data["names"]
                if isinstance(names, list):
                    return [str(n) for n in names]
                if isinstance(names, dict):
                    return [str(names[k]) for k in sorted(names.keys(), key=_sort_key)]
    return None


def _sort_key(x):
    return int(x) if str(x).isdigit() else str(x)


def _parse_data_yaml_fallback(root: str) -> Optional[List[str]]:
    for name in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml"):
        p = os.path.join(root, name)
        if not os.path.isfile(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            continue
        in_names = False
        names: List[str] = []
        for line in lines:
            s = line.rstrip()
            if not s.strip() or s.strip().startswith("#"):
                if in_names and names:
                    break
                continue
            if s.strip().startswith("names:"):
                in_names = True
                rest = s.split(":", 1)[1].strip()
                if rest.startswith("["):
                    inner = rest.strip("[]")
                    names = [
                        p.strip().strip("'\"")
                        for p in inner.split(",")
                        if p.strip()
                    ]
                    return names
                continue
            if in_names:
                st = s.strip()
                if st.startswith("- "):
                    names.append(st[2:].strip().strip("'\""))
                elif re.match(r"^\d+\s*:", st):
                    names.append(st.split(":", 1)[1].strip().strip("'\""))
                else:
                    break
        if names:
            return names
    return None


def _parse_classes_txt(root: str) -> Optional[List[str]]:
    for name in CLASS_FILE_NAMES:
        p = os.path.join(root, name)
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    names = [
                        l.strip().strip("'\"")
                        for l in f
                        if l.strip() and not l.strip().startswith("#")
                    ]
                if names:
                    return names
            except OSError:
                continue
    return None


# --------------------------------------------------------------------------- #
# Layout / split detection
# --------------------------------------------------------------------------- #


def _detect_layout_and_splits(info: DatasetInfo) -> None:
    root = info.root
    try:
        entries = set(os.listdir(root))
    except OSError:
        entries = set()
    lower = {e.lower() for e in entries}

    # Case 1: explicit train/valid/test subfolders
    split_dirs = []
    for s in ("train", "valid", "val", "test", "validation"):
        if s in lower:
            sp = os.path.join(root, s)
            if os.path.isdir(sp):
                split_dirs.append(sp)

    if split_dirs:
        # Determine per-split layout (mixed or images/labels)
        any_sep = False
        any_mixed = False
        for sp in split_dirs:
            sub = set(os.listdir(sp)) if os.path.isdir(sp) else set()
            sub_lower = {e.lower() for e in sub}
            if "images" in sub_lower and "labels" in sub_lower:
                any_sep = True
            else:
                any_mixed = True
        info.layout = "splits_sep" if any_sep and not any_mixed else (
            "splits_single" if any_mixed and not any_sep else "splits_mixed"
        )
        for sp in split_dirs:
            name = os.path.basename(sp).lower()
            name = SPLIT_ALIASES.get(name, name)
            split = _build_split(name, sp)
            info.splits.append(split)
        return

    # Case 2: images/ + labels/ at root
    if "images" in lower and "labels" in lower:
        info.layout = "images_labels"
        info.splits.append(
            SplitInfo(
                name="root",
                path=root,
                image_dir=os.path.join(root, "images"),
                label_dirs=[os.path.join(root, "labels")],
            )
        )
        return

    # Case 3: single folder, images + labels mixed
    info.layout = "single_mixed"
    info.splits.append(
        SplitInfo(
            name="root",
            path=root,
            image_dir=root,
            label_dirs=[root],
        )
    )


def _build_split(name: str, sp: str) -> SplitInfo:
    sub = set(os.listdir(sp)) if os.path.isdir(sp) else set()
    sub_lower = {e.lower() for e in sub}
    if "images" in sub_lower and "labels" in sub_lower:
        return SplitInfo(
            name=name,
            path=sp,
            image_dir=os.path.join(sp, "images"),
            label_dirs=[os.path.join(sp, "labels")],
        )
    # mixed: images and labels together
    return SplitInfo(
        name=name,
        path=sp,
        image_dir=sp,
        label_dirs=[sp],
    )


def _list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    try:
        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        ]
    except OSError:
        return []
    files.sort()
    return files


# --------------------------------------------------------------------------- #
# Label-format detection per split
# --------------------------------------------------------------------------- #


def _detect_label_format_for_split(split: SplitInfo, info: DatasetInfo) -> Tuple[Optional[str], Optional[str]]:
    """Detect the label format for a split by probing the first image.

    Returns (format_name, label_path_for_first_image).
    """
    if not split.images:
        return None, None

    # If classes are known, try every format in priority order against the
    # first image; first one that yields >=1 box wins.
    sample = split.images[0]
    img_size = _image_size(sample)
    for name, fmt in all_formats():
        try:
            lp = fmt.find_label_file(sample, split.label_dirs)
        except Exception:
            lp = None
        if not lp:
            continue
        try:
            boxes = fmt.load(lp, img_size, info.classes)
        except LabelParseError:
            continue
        if boxes:
            return name, lp

    # Fallback: no boxes found, but a label file exists. Pick the first format
    # that finds ANY file, so we can at least report the format.
    for name, fmt in all_formats():
        try:
            lp = fmt.find_label_file(sample, split.label_dirs)
        except Exception:
            lp = None
        if lp:
            return name, lp

    return None, None


def _image_size(path: str) -> Optional[Tuple[int, int]]:
    """Return (width, height) without loading the full image when possible."""
    try:
        import cv2

        # imread with reduced load is faster but still decodes; fall back to
        # full read if needed.
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        h, w = img.shape[:2]
        return (w, h)
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Backward-compat shims so the old main.py imports keep working
# --------------------------------------------------------------------------- #


def import_dataset_dialog(parent, folder_path):
    """Minimal stand-in. Returns a config dict consumed by load_dataset()."""
    info = scan_dataset(folder_path)
    return {
        "folder_path": folder_path,
        "info": info,
        "splits": [s.name for s in info.splits],
        "format": info.label_format,
    }


def load_dataset(parent, config, frame_annotations, class_colors, bbox_cls):
    """Backward-compat wrapper matching the old (image_files, message) return."""
    info: DatasetInfo = config["info"]
    result = load_dataset_into_app(parent, info, bbox_cls)
    msg = (
        f"Loaded {len(result['image_files'])} images "
        f"({', '.join(f'{k}={v}' for k, v in result['per_split_counts'].items())}); "
        f"{len(result['classes'])} classes from {info.classes_source}."
    )
    if info.classes_conflict:
        msg += f"  WARNING: {info.classes_conflict}"
    if result["warnings"]:
        msg += f"  ({len(result['warnings'])} warnings)"
    # Attach split info to app for later use (filtering, ops)
    parent._viat_dataset_info = info
    parent._viat_frame_to_split = result["frame_to_split"]
    return result["image_files"], msg


# --------------------------------------------------------------------------- #
# Backward-compat: export / create dataset
# --------------------------------------------------------------------------- #
# main.py calls these four functions. The ORIGINAL utils/dataset_manager.py
# (which this module replaced) implemented them; we provide working versions
# here that use the label_format plugins so your existing Export/Create
# Dataset menu items keep working without any main.py changes.
# --------------------------------------------------------------------------- #


def export_dataset_dialog(parent, image_files, frame_annotations):
    """Dialog for exporting the current image dataset to a chosen format.

    Returns a config dict, or None if the user cancelled.
    """
    from PyQt5.QtWidgets import (
        QDialog, QVBoxLayout, QFormLayout, QComboBox, QLineEdit,
        QCheckBox, QDialogButtonBox, QFileDialog, QLabel, QSpinBox,
    )

    dialog = QDialog(parent)
    dialog.setWindowTitle("Export Image Dataset")
    dialog.setMinimumWidth(420)
    layout = QVBoxLayout(dialog)

    info_label = QLabel(
        f"Exporting {len(image_files)} images, "
        f"{sum(len(v) for v in frame_annotations.values())} annotations."
    )
    layout.addWidget(info_label)

    form = QFormLayout()

    format_combo = QComboBox()
    format_combo.addItems(["YOLO", "COCO JSON", "Pascal VOC XML"])
    form.addRow("Format:", format_combo)

    output_edit = QLineEdit()
    output_edit.setPlaceholderText("Output folder...")
    from PyQt5.QtWidgets import QPushButton, QHBoxLayout
    out_row = QHBoxLayout()
    out_row.addWidget(output_edit)
    browse_btn = QPushButton("Browse...")
    def _browse():
        d = QFileDialog.getExistingDirectory(dialog, "Select Output Folder")
        if d:
            output_edit.setText(d)
    browse_btn.clicked.connect(_browse)
    out_row.addWidget(browse_btn)
    form.addRow("Output dir:", out_row)

    split_check = QCheckBox("Create train/valid/test split (90/10)")
    split_check.setChecked(False)
    form.addRow("", split_check)

    split_spin = QSpinBox()
    split_spin.setRange(1, 50)
    split_spin.setValue(10)  # % for validation
    split_spin.setEnabled(False)
    split_check.toggled.connect(split_spin.setEnabled)
    form.addRow("Validation %:", split_spin)

    layout.addLayout(form)

    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addWidget(buttons)

    if dialog.exec_() != QDialog.Accepted:
        return None

    out_dir = output_edit.text().strip()
    if not out_dir:
        return None

    fmt_map = {"YOLO": "yolo", "COCO JSON": "coco", "Pascal VOC XML": "pascal_voc"}
    return {
        "output_dir": out_dir,
        "format": fmt_map[format_combo.currentText()],
        "make_splits": split_check.isChecked(),
        "valid_pct": split_spin.value(),
    }


def export_dataset(parent, config, image_files, frame_annotations, class_colors):
    """Write the current image dataset to disk in the chosen format.

    Uses the label_format plugins. Returns a status message string.
    """
    out_dir = config["output_dir"]
    fmt_name = config.get("format", "yolo")
    make_splits = config.get("make_splits", False)
    valid_pct = config.get("valid_pct", 10)

    fmt = get_format(fmt_name)
    if fmt is None:
        return f"Unknown format: {fmt_name}"

    os.makedirs(out_dir, exist_ok=True)

    # Build the class list (preserving insertion order from class_colors)
    classes = list(class_colors.keys())

    # Determine split assignment for each image
    if make_splits:
        n_valid = max(1, int(len(image_files) * valid_pct / 100))
        # even split: every Nth image goes to valid
        split_of = {}
        stride = max(1, len(image_files) // n_valid) if n_valid else len(image_files)
        for i in range(len(image_files)):
            split_of[i] = "valid" if (i % stride == 0 and i < stride * n_valid) else "train"
        # any leftover -> train
        subdirs = ["train", "valid"]
    else:
        split_of = {i: "root" for i in range(len(image_files))}
        subdirs = ["root"]

    # For per-image formats: one file per image.
    # For dataset-wide formats (COCO): one file per split.
    import shutil
    from .label_formats.coco import CocoLabelFormat

    written = 0
    for i, img_path in enumerate(image_files):
        split = split_of[i]
        if make_splits:
            img_dest_dir = os.path.join(out_dir, split, "images")
            lbl_dest_dir = os.path.join(out_dir, split, "labels")
        else:
            img_dest_dir = out_dir
            lbl_dest_dir = out_dir
        os.makedirs(img_dest_dir, exist_ok=True)
        os.makedirs(lbl_dest_dir, exist_ok=True)

        # Copy the image
        img_name = os.path.basename(img_path)
        try:
            shutil.copy2(img_path, os.path.join(img_dest_dir, img_name))
        except OSError:
            pass

        # Get the boxes for this frame
        anns = frame_annotations.get(i, [])
        if not anns:
            if not isinstance(fmt, CocoLabelFormat):
                # write an empty label file for YOLO (some trainers expect it)
                continue
            boxes = []
        else:
            boxes = []
            for ann in anns:
                boxes.append({
                    "class_name": getattr(ann, "class_name", "unknown"),
                    "class_index": classes.index(ann.class_name) if ann.class_name in classes else 0,
                    "x": ann.rect.x(),
                    "y": ann.rect.y(),
                    "w": ann.rect.width(),
                    "h": ann.rect.height(),
                    "score": getattr(ann, "score", 1.0),
                })

        if isinstance(fmt, CocoLabelFormat):
            continue  # handled in the COCO batch pass below

        # Write the label file
        try:
            img_size = _image_size(img_path)
            if img_size is None:
                continue
            content = fmt.dump(boxes, img_size, classes)
            stem = os.path.splitext(img_name)[0]
            lbl_path = os.path.join(lbl_dest_dir, stem + fmt.extensions[0])
            with open(lbl_path, "w", encoding="utf-8") as f:
                f.write(content)
            written += 1
        except Exception:
            continue

    # COCO dataset-wide pass (one json per split)
    if isinstance(fmt, CocoLabelFormat):
        import json
        for split in subdirs:
            cat_id = {c: i for i, c in enumerate(classes, 1)}
            images_json = []
            anns_json = []
            ann_id = 1
            for i, img_path in enumerate(image_files):
                if split_of[i] != split:
                    continue
                img_size = _image_size(img_path) or (0, 0)
                img_name = os.path.basename(img_path)
                images_json.append({
                    "id": i, "file_name": img_name,
                    "width": img_size[0], "height": img_size[1],
                })
                for b in (frame_annotations.get(i, [])):
                    cat = cat_id.get(b.class_name, 0)
                    anns_json.append({
                        "id": ann_id, "image_id": i, "category_id": cat,
                        "bbox": [b.rect.x(), b.rect.y(), b.rect.width(), b.rect.height()],
                        "area": b.rect.width() * b.rect.height(),
                        "iscrowd": 0,
                    })
                    ann_id += 1
            coco = {
                "images": images_json,
                "annotations": anns_json,
                "categories": [{"id": i, "name": c} for i, c in enumerate(classes, 1)],
            }
            out_json = os.path.join(
                out_dir, split if make_splits else "",
                "_annotations.coco.json" if make_splits else "annotations.json"
            ).strip()
            os.makedirs(os.path.dirname(out_json), exist_ok=True)
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(coco, f, indent=2)
            written += 1

    # Also write classes.txt / data.yaml so the export is a valid dataset
    _write_class_files(out_dir, classes)

    return f"Exported {len(image_files)} images ({written} label files) to {out_dir} as {fmt_name}."


def _write_class_files(out_dir, classes):
    """Write classes.txt and data.yaml for an exported dataset."""
    try:
        with open(os.path.join(out_dir, "classes.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(classes) + "\n")
    except OSError:
        pass
    try:
        with open(os.path.join(out_dir, "data.yaml"), "w", encoding="utf-8") as f:
            f.write(f"path: .\nnc: {len(classes)}\nnames: [{', '.join(classes)}]\n")
    except OSError:
        pass


def create_dataset_dialog(parent, image_files, frame_annotations, class_colors):
    """Dialog for creating a NEW labeled dataset from current annotations.

    Returns a config dict or None.
    """
    return export_dataset_dialog(parent, image_files, frame_annotations)


def create_dataset(parent, config, image_files, frame_annotations, class_colors):
    """Create a dataset on disk. Delegates to export_dataset()."""
    msg = export_dataset(parent, config, image_files, frame_annotations, class_colors)
    return bool(msg and not msg.startswith("Unknown"))
