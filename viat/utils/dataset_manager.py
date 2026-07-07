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
    for e in entries:
        if e.endswith((".yaml", ".yml")):
            p = os.path.join(folder_path, e)
            if os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        if "names:" in f.read(1024):
                            return "dataset"
                except Exception:
                    pass
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


from PyQt5.QtCore import QThread, pyqtSignal

class DatasetLoaderThread(QThread):
    batchLoaded = pyqtSignal(list, list, dict)  # image_files_batch, split_batch, boxes_batch
    finishedLoading = pyqtSignal(dict)  # final_stats

    def __init__(self, info: DatasetInfo, target_splits: set):
        super().__init__()
        self.info = info
        self.target_splits = target_splits
        self.is_cancelled = False

    def run(self):
        import os
        from .label_formats import get_format, LabelParseError

        warnings = []
        per_split = {}
        
        # Batching setup
        batch_size = 100
        current_img_batch = []
        current_split_batch = []
        current_boxes_batch = {}
        
        total_index = 0

        # Collect all images across target splits and sort by filename
        all_images_with_info = []
        split_formats = {}
        
        for split in self.info.splits:
            if split.name not in self.target_splits:
                continue
                
            fmt = get_format(split.label_format or self.info.label_format or "yolo")
            if fmt is None:
                warnings.append(f"Split {split.name}: unknown format, skipped.")
                continue
            split_formats[split.name] = fmt
            
            for img_path in split.images:
                all_images_with_info.append((img_path, split))

        # Sort by basename
        all_images_with_info.sort(key=lambda x: os.path.basename(x[0]))

        for img_path, split in all_images_with_info:
            if self.is_cancelled:
                break
                
            fmt = split_formats[split.name]
            current_img_batch.append(img_path)
            current_split_batch.append(split.name)
            per_split[split.name] = per_split.get(split.name, 0) + 1

            img_size = _image_size(img_path)
            if img_size is None:
                warnings.append(f"Could not read size: {os.path.basename(img_path)}")
                current_boxes_batch[total_index] = []
                total_index += 1
                continue

            try:
                label_path = fmt.find_label_file(img_path, split.label_dirs)
            except Exception as e:
                warnings.append(f"{os.path.basename(img_path)}: {e}")
                label_path = None

            boxes = []
            if label_path:
                try:
                    boxes = fmt.load(label_path, img_size, self.info.classes)
                except LabelParseError as e:
                    warnings.append(str(e))
            
            current_boxes_batch[total_index] = boxes or []
            total_index += 1

            # Emit batch
            if len(current_img_batch) >= batch_size:
                self.batchLoaded.emit(list(current_img_batch), list(current_split_batch), dict(current_boxes_batch))
                current_img_batch.clear()
                current_split_batch.clear()
                current_boxes_batch.clear()

        # Emit remaining batch
        if current_img_batch and not self.is_cancelled:
            self.batchLoaded.emit(current_img_batch, current_split_batch, current_boxes_batch)

        if not self.is_cancelled:
            self.finishedLoading.emit({
                "per_split_counts": per_split,
                "classes": self.info.classes,
                "warnings": warnings,
            })

    def cancel(self):
        self.is_cancelled = True

def load_dataset_into_app(
    app,
    info: DatasetInfo,
    bbox_cls,
    *,
    splits_to_load: Optional[List[str]] = None,
):
    """Load *info* into the VIAT main window using a background thread."""
    from PyQt5.QtCore import QRect
    from PyQt5.QtGui import QColor
    import random

    target_splits = {s.name for s in info.splits}
    if splits_to_load is not None:
        target_splits = set(splits_to_load)

    # Initialize app states
    app.frame_annotations = {}
    app.image_files = getattr(app, "image_files", [])
    app._viat_frame_to_split = getattr(app, "_viat_frame_to_split", [])
    
    app.image_files.clear()
    app._viat_frame_to_split.clear()
    app.total_frames = 0
    app.current_frame = 0

    old_colors = dict(getattr(app.canvas, "class_colors", {}) or {})
    old_attributes = dict(getattr(app.canvas, "class_attributes", {}) or {})
    new_colors, new_attributes = {}, {}
    for cls in info.classes:
        new_colors[cls] = old_colors.get(cls, QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        new_attributes[cls] = old_attributes.get(cls, {
            "Size": {"type": "int", "default": -1, "min": 0, "max": 100},
            "Quality": {"type": "int", "default": -1, "min": 0, "max": 100},
        })
    app.canvas.class_colors = new_colors
    app.canvas.class_attributes = new_attributes
    app.class_attributes = app.canvas.class_attributes

    loader = DatasetLoaderThread(info, target_splits)

    def on_batch_loaded(imgs, splits, boxes_batch):
        if getattr(app, "_dataset_loader", None) is not loader:
            return
        start_idx = len(app.image_files)
        app.image_files.extend(imgs)
        app._viat_frame_to_split.extend(splits)
        app.total_frames = len(app.image_files)

        for abs_idx, boxes in boxes_batch.items():
            anns = []
            for b in boxes:
                cls_name = b["class_name"]
                if cls_name not in app.canvas.class_colors:
                    app.canvas.class_colors[cls_name] = QColor(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                    app.canvas.class_attributes[cls_name] = {
                        "Size": {"type": "int", "default": -1, "min": 0, "max": 100},
                        "Quality": {"type": "int", "default": -1, "min": 0, "max": 100},
                    }
                    if cls_name not in info.classes:
                        info.classes.append(cls_name)

                rect = QRect(b["x"], b["y"], max(1, b["w"]), max(1, b["h"]))
                ann = bbox_cls(
                    rect=rect,
                    class_name=cls_name,
                    attributes=b.get("attributes", {}),
                    color=app.canvas.class_colors[cls_name],
                    source=b.get("source", "manual"),
                    score=b.get("score", 1.0),
                    segmentation=b.get("segmentation"),
                )
                if "verified" in b:
                    ann.verified = bool(b["verified"])
                anns.append(ann)
            app.frame_annotations[abs_idx] = anns
        
        app.frame_slider.setMaximum(max(0, app.total_frames - 1))
        app.update_frame_info()
        app.update_annotation_list()
        app.refresh_class_ui()
        if start_idx == 0:
            app.load_current_image()

    def on_finished(stats):
        if getattr(app, "_dataset_loader", None) is loader:
            app._dataset_loader = None
        if hasattr(app, "cancel_loading_action"):
            app.cancel_loading_action.setVisible(False)
        
        msg = f"Loaded {app.total_frames} images; {len(stats['classes'])} classes."
        if stats['warnings']:
            msg += f" ({len(stats['warnings'])} warnings)"
        app.statusBar.showMessage(msg, 8000)

    loader.batchLoaded.connect(on_batch_loaded)
    loader.finishedLoading.connect(on_finished)
    
    app._dataset_loader = loader
    if hasattr(app, "cancel_loading_action"):
        app.cancel_loading_action.setVisible(True)
    loader.start()

    return {
        "image_files": [], # Return empty immediately as loading is background
        "frame_to_split": [],
        "per_split_counts": {},
        "classes": info.classes,
        "warnings": [],
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
    candidates = ["data.yaml", "data.yml", "dataset.yaml", "dataset.yml"]
    try:
        for f in os.listdir(root):
            if f.endswith((".yaml", ".yml")) and f not in candidates:
                candidates.append(f)
    except OSError:
        pass
    for name in candidates:
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
    candidates = ["data.yaml", "data.yml", "dataset.yaml", "dataset.yml"]
    try:
        for f in os.listdir(root):
            if f.endswith((".yaml", ".yml")) and f not in candidates:
                candidates.append(f)
    except OSError:
        pass
    for name in candidates:
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
    format_combo.addItems(["YOLO", "COCO JSON", "Pascal VOC XML", "Raya Video"])
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
    
    # Raya Video specific options
    from PyQt5.QtWidgets import QWidget, QRadioButton, QButtonGroup
    raya_widget = QWidget()
    raya_layout = QFormLayout(raya_widget)
    raya_layout.setContentsMargins(0, 0, 0, 0)
    
    width_edit = QLineEdit()
    width_edit.setPlaceholderText("Auto (Max)")
    height_edit = QLineEdit()
    height_edit.setPlaceholderText("Auto (Max)")
    
    size_layout = QHBoxLayout()
    size_layout.addWidget(QLabel("W:"))
    size_layout.addWidget(width_edit)
    size_layout.addWidget(QLabel("H:"))
    size_layout.addWidget(height_edit)
    raya_layout.addRow("Video Size:", size_layout)
    
    resize_group = QButtonGroup(raya_widget)
    radio_pad = QRadioButton("Pad to Top-Left")
    radio_pad.setChecked(True)
    radio_yolo = QRadioButton("YOLO-style Resize")
    resize_group.addButton(radio_pad)
    resize_group.addButton(radio_yolo)
    
    resize_layout = QVBoxLayout()
    resize_layout.addWidget(radio_pad)
    resize_layout.addWidget(radio_yolo)
    raya_layout.addRow("Resize Mode:", resize_layout)
    
    class_check = QCheckBox("Include frame category (Raya with classes)")
    class_check.setChecked(True)
    raya_layout.addRow("", class_check)
    
    raya_widget.setVisible(False)
    form.addRow(raya_widget)
    
    def on_format_changed(text):
        is_raya = (text == "Raya Video")
        raya_widget.setVisible(is_raya)
        split_check.setVisible(not is_raya)
        if split_check.isVisible():
            split_spin.setVisible(split_check.isChecked())
            form.labelForField(split_spin).setVisible(True)
        else:
            split_spin.setVisible(False)
            form.labelForField(split_spin).setVisible(False)
            
    format_combo.currentTextChanged.connect(on_format_changed)

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

    fmt_map = {"YOLO": "yolo", "COCO JSON": "coco", "Pascal VOC XML": "pascal_voc", "Raya Video": "raya_video"}
    
    video_width = None
    video_height = None
    if width_edit.text().strip().isdigit() and height_edit.text().strip().isdigit():
        video_width = int(width_edit.text().strip())
        video_height = int(height_edit.text().strip())
        
    return {
        "output_dir": out_dir,
        "format": fmt_map[format_combo.currentText()],
        "make_splits": split_check.isChecked(),
        "valid_pct": split_spin.value(),
        "video_width": video_width,
        "video_height": video_height,
        "resize_mode": "pad" if radio_pad.isChecked() else "yolo",
        "include_classes": class_check.isChecked()
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
    if fmt_name == "raya_video":
        # Handle Raya Video export separately
        yield from export_raya_video_dataset(config, image_files, frame_annotations, class_colors)
        return
    elif fmt is None:
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
    import cv2
    from .label_formats.coco import CocoLabelFormat

    is_video = hasattr(parent, 'video_filename') and parent.video_filename and not getattr(parent, 'is_image_dataset', False)
    cap = None
    video_metadata = None
    frame_sizes = {}
    if is_video:
        cap = cv2.VideoCapture(parent.video_filename)
        video_metadata = getattr(parent, 'video_metadata', None)

    written = 0
    total = len(image_files)
    for i, img_path in enumerate(image_files):
        if i % 5 == 0 and total > 0:
            yield int((i / total) * 90), f"Exporting image {i}/{total}..."
            
        split = split_of[i]
        if make_splits:
            img_dest_dir = os.path.join(out_dir, split, "images")
            lbl_dest_dir = os.path.join(out_dir, split, "labels")
        else:
            img_dest_dir = out_dir
            lbl_dest_dir = out_dir
        os.makedirs(img_dest_dir, exist_ok=True)
        os.makedirs(lbl_dest_dir, exist_ok=True)

        # Copy the image (or extract from video)
        img_name = os.path.basename(img_path)
        img_size = None
        if is_video:
            ret = False
            if cap and cap.isOpened():
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
            if ret and frame is not None:
                # Crop if video metadata has resize_mode == "pad"
                if video_metadata and video_metadata.get("resize_mode") == "pad":
                    sizes = video_metadata.get("original_sizes", {})
                    orig_size = sizes.get(str(i))
                    if orig_size and len(orig_size) == 2:
                        orig_w, orig_h = orig_size
                        frame = frame[:orig_h, :orig_w]
                
                # Normalize resolution if config has video_width/video_height specified
                video_width = config.get("video_width")
                video_height = config.get("video_height")
                if video_width and video_height:
                    frame = cv2.resize(frame, (video_width, video_height), interpolation=cv2.INTER_AREA)

                img_size = (frame.shape[1], frame.shape[0])
                cv2.imwrite(os.path.join(img_dest_dir, img_name), frame)
        else:
            try:
                shutil.copy2(img_path, os.path.join(img_dest_dir, img_name))
            except OSError:
                pass
            img_size = _image_size(img_path)

        if img_size is not None:
            frame_sizes[i] = img_size

        # Get the boxes for this frame
        anns = frame_annotations.get(i, [])
        if not anns:
            if not isinstance(fmt, CocoLabelFormat):
                # write an empty label file for YOLO (some trainers expect it)
                continue
            boxes = []
        else:
            boxes = []
            thresholds = getattr(parent, "class_thresholds", {}) or {}
            for ann in anns:
                if hasattr(ann, 'score') and ann.score is not None:
                    thresh = thresholds.get(ann.class_name, 0.0)
                    if ann.score < thresh:
                        continue
                        
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

    yield 90, "Finalizing export files..."
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
                img_size = frame_sizes.get(i) or _image_size(img_path) or (0, 0)
                img_name = os.path.basename(img_path)
                images_json.append({
                    "id": i, "file_name": img_name,
                    "width": img_size[0], "height": img_size[1],
                })
                thresholds = getattr(parent, "class_thresholds", {}) or {}
                for b in (frame_annotations.get(i, [])):
                    if hasattr(b, 'score') and b.score is not None:
                        thresh = thresholds.get(b.class_name, 0.0)
                        if b.score < thresh:
                            continue
                            
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

    if cap is not None:
        cap.release()

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


def export_raya_video_dataset(config, image_files, frame_annotations, class_colors):
    import cv2
    import json
    import os
    import numpy as np
    from viat.utils.file_operations import export_raya_annotations, export_raya_with_classes_annotations
    from annotation import BoundingBox
    from PyQt5.QtCore import QRectF

    out_dir = config["output_dir"]
    video_width = config.get("video_width")
    video_height = config.get("video_height")
    resize_mode = config.get("resize_mode", "pad")
    include_classes = config.get("include_classes", True)

    os.makedirs(out_dir, exist_ok=True)
    video_path = os.path.join(out_dir, "dataset_video.mp4")
    
    # 1. Determine video resolution if not provided
    if video_width is None or video_height is None:
        max_w = 0
        max_h = 0
        for img_path in image_files:
            size = _image_size(img_path)
            if size:
                max_w = max(max_w, size[0])
                max_h = max(max_h, size[1])
        video_width = max_w if video_width is None else video_width
        video_height = max_h if video_height is None else video_height

    if video_width == 0 or video_height == 0:
        return "Error: Could not determine video resolution."

    # 2. Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(video_path, fourcc, 30.0, (video_width, video_height))
    
    original_sizes = {}
    total = len(image_files)
    all_annotations = []
    
    for i, img_path in enumerate(image_files):
        if i % 5 == 0 and total > 0:
            yield int((i / total) * 80), f"Writing video frame {i}/{total}..."
            
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        orig_h, orig_w = img.shape[:2]
        original_sizes[str(i)] = [orig_w, orig_h]
        
        dx, dy = 0, 0
        scale = 1.0
        
        if resize_mode == "pad":
            frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
            copy_w = min(orig_w, video_width)
            copy_h = min(orig_h, video_height)
            frame[0:copy_h, 0:copy_w] = img[0:copy_h, 0:copy_w]
        else:
            scale = min(video_width / orig_w, video_height / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            frame = np.zeros((video_height, video_width, 3), dtype=np.uint8)
            dx = (video_width - new_w) // 2
            dy = (video_height - new_h) // 2
            frame[dy:dy+new_h, dx:dx+new_w] = resized_img
            
        out_video.write(frame)
        
        anns = frame_annotations.get(i, [])
        for ann in anns:
            new_rect = QRectF(
                ann.rect.x() * scale + dx,
                ann.rect.y() * scale + dy,
                ann.rect.width() * scale,
                ann.rect.height() * scale
            )
            
            ann_copy = BoundingBox(new_rect, getattr(ann, 'class_name', ''), getattr(ann, 'attributes', {}), source=getattr(ann, 'source', 'manual'))
            ann_copy.frame = i
            all_annotations.append(ann_copy)

    out_video.release()
    
    yield 85, "Exporting metadata and annotations..."
    
    metadata_path = os.path.join(out_dir, "dataset_video_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "original_sizes": original_sizes,
            "resize_mode": resize_mode,
            "video_width": video_width,
            "video_height": video_height
        }, f, indent=2)
        
    txt_path = os.path.join(out_dir, "dataset_video.txt")
    if include_classes:
        classes = list(class_colors.keys())
        export_raya_with_classes_annotations(txt_path, all_annotations, classes)
    else:
        export_raya_annotations(txt_path, all_annotations)
        
    yield 100, "Done"
    return f"Exported video and Raya annotations to {out_dir}"


def create_dataset(parent, config, image_files, frame_annotations, class_colors):
    """Create a dataset on disk. Delegates to export_dataset()."""
    msg = export_dataset(parent, config, image_files, frame_annotations, class_colors)
    return bool(msg and not msg.startswith("Unknown"))

def update_dataset_labels(info, frame_annotations, image_files, current_classes=None):
    """
    Update original dataset label files with current annotations.
    """
    import os
    from .label_formats import get_format
    
    if current_classes is not None:
        info.classes = list(current_classes)
        # Rewrite class definition files
        for name in ("classes.txt", "obj.names", "labels.txt"):
            p = os.path.join(info.root, name)
            if os.path.isfile(p) or name == "classes.txt":
                try:
                    with open(p, "w", encoding="utf-8") as f:
                        f.write("\n".join(info.classes) + "\n")
                except OSError:
                    pass
        try:
            from .dataset_merger import _update_target_yaml
            _update_target_yaml(info.root, info.classes, replace=True)
        except Exception:
            pass
    
    # Pre-compute image path to split mapping for speed
    image_to_split = {}
    for split in info.splits:
        for img_path in split.images:
            image_to_split[img_path] = split
            
    updated_count = 0
    errors = []

    for frame_idx_str, anns in frame_annotations.items():
        frame_idx = int(frame_idx_str)
        if frame_idx >= len(image_files):
            continue
            
        img_path = image_files[frame_idx]
        split_info = image_to_split.get(img_path)
        if not split_info:
            continue
            
        fmt = get_format(split_info.label_format or info.label_format or "yolo")
        if not fmt:
            errors.append(f"Unknown format for {img_path}")
            continue
            
        img_size = _image_size(img_path)
        if not img_size:
            errors.append(f"Could not read size for {img_path}")
            continue
            
        try:
            # Prepare annotations to match expected format (dict)
            boxes = []
            for ann in anns:
                if hasattr(ann, 'to_dict'):
                    d = ann.to_dict()
                    boxes.append({
                        "x": d["rect"]["x"],
                        "y": d["rect"]["y"],
                        "w": d["rect"]["width"],
                        "h": d["rect"]["height"],
                        "class_name": d["class_name"],
                        "score": d.get("score", 1.0),
                        "segmentation": d.get("segmentation")
                    })
                else:
                    boxes.append(ann)
            text_content = fmt.dump(boxes, img_size, info.classes)
            
            label_path = fmt.find_label_file(img_path, split_info.label_dirs)
            if not label_path:
                # Create a new label file path
                stem = os.path.splitext(os.path.basename(img_path))[0]
                if split_info.label_dirs:
                    label_path = os.path.join(split_info.label_dirs[0], stem + fmt.extensions[0])
                else:
                    label_path = os.path.join(os.path.dirname(img_path), stem + fmt.extensions[0])
                    
            with open(label_path, "w", encoding="utf-8") as f:
                f.write(text_content)
            updated_count += 1
        except Exception as e:
            errors.append(f"Error saving {img_path}: {e}")
            
    return updated_count, errors

