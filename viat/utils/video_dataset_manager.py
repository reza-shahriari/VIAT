"""
Video Dataset detection, scanning & management for VIAT.

Handles video datasets with various structures:
  Layout A -- flat folder:
      dataset/
        video1.mp4
        video1.txt  (or .json)
        video2.mp4
        classes.txt (or data.yaml)

  Layout B -- videos/ + labels/ subfolders:
      dataset/
        videos/
          video1.mp4
          video2.mp4
        labels/
          video1.txt
          video2.txt
        data.yaml

  Layout C -- train/valid/test splits:
      dataset/
        train/
          video1.mp4
          video1.txt
        valid/
          video2.mp4
          video2.txt
        test/
          video3.mp4
        data.yaml

  Layout D -- nested subfolders (e.g. sequence/camera folders):
      dataset/
        cam1/
          clip1.mp4
          clip1.txt
        cam2/
          clip2.mp4
"""

import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from natsort import natsorted

try:
    import yaml
except ImportError:
    yaml = None

VIDEO_EXTENSIONS = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".flv",
    ".m4v",
    ".wmv",
)

ANNOTATION_EXTENSIONS = (".txt", ".json", ".xml", ".csv")

SPLIT_NAMES = ("train", "valid", "val", "test", "validation")
SPLIT_ALIASES = {"val": "valid", "validation": "valid"}

CLASS_FILE_NAMES = ("classes.txt", "obj.names", "labels.txt")
EXCLUDE_OUTPUT_VIDEOS = {"outvideo.mp4", "out_video.mp4"}


@dataclass
class VideoInfo:
    """Represents a single video inside a video dataset."""

    path: str  # Absolute path to video file
    filename: str  # Base filename (e.g. "seq1.mp4")
    relative_path: str  # Relative path from dataset root
    split: str = "root"  # Canonical split: "train", "valid", "test", "root", or subfolder name
    annotation_file: Optional[str] = None  # Path to matching annotation file if found
    annotation_format: Optional[str] = None  # "raya_txt", "yolo_txt", "coco_json", "voc_xml", "unknown"
    has_annotations: bool = False
    status: str = "unannotated"  # "annotated" | "unannotated"
    total_frames: Optional[int] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None

    @property
    def base_name(self) -> str:
        return os.path.splitext(self.filename)[0]

    def update_status(self):
        """Update status based on whether annotation_file exists and has content."""
        if self.annotation_file and os.path.isfile(self.annotation_file):
            try:
                if os.path.getsize(self.annotation_file) > 0:
                    self.has_annotations = True
                    self.status = "annotated"
                    return
            except OSError:
                pass
        self.has_annotations = False
        self.status = "unannotated"


@dataclass
class VideoSplitInfo:
    """Represents a split or partition of a video dataset."""

    name: str  # canonical: train / valid / test / root
    path: str  # Directory path for this split
    videos: List[VideoInfo] = field(default_factory=list)

    @property
    def video_count(self) -> int:
        return len(self.videos)

    @property
    def annotated_count(self) -> int:
        return sum(1 for v in self.videos if v.has_annotations)


@dataclass
class VideoDatasetInfo:
    """Represents a complete scanned video dataset."""

    root: str
    layout: str  # "flat" | "splits" | "videos_labels" | "nested"
    splits: List[VideoSplitInfo] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    classes_source: Optional[str] = None  # "data.yaml" | "classes.txt" | "raya_header" | "inferred"
    classes_conflict: Optional[str] = None

    @property
    def all_videos(self) -> List[VideoInfo]:
        out = []
        for s in self.splits:
            out.extend(s.videos)
        return out

    @property
    def video_count(self) -> int:
        return sum(len(s.videos) for s in self.splits)

    @property
    def annotated_count(self) -> int:
        return sum(1 for v in self.all_videos if v.has_annotations)

    @property
    def unannotated_count(self) -> int:
        return sum(1 for v in self.all_videos if not v.has_annotations)

    def find_video_by_path(self, path: str) -> Optional[VideoInfo]:
        norm_path = os.path.abspath(path)
        for v in self.all_videos:
            if os.path.abspath(v.path) == norm_path:
                return v
        return None


# --------------------------------------------------------------------------- #
# Class file parsing
# --------------------------------------------------------------------------- #


def _parse_yaml_classes(yaml_path: str) -> Tuple[List[str], Optional[str]]:
    """Parse class names from data.yaml / dataset.yaml."""
    if not os.path.isfile(yaml_path):
        return [], None
    if yaml is not None:
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                names = data.get("names")
                if isinstance(names, list):
                    return [str(x).strip() for x in names if str(x).strip()], "data.yaml"
                elif isinstance(names, dict):
                    sorted_keys = sorted(names.keys(), key=lambda k: int(k) if str(k).isdigit() else str(k))
                    return [str(names[k]).strip() for k in sorted_keys if str(names[k]).strip()], "data.yaml"
        except Exception:
            pass

    # Regex fallback for yaml
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            content = f.read()
        match = re.search(r"names:\s*\[(.*?)\]", content, re.DOTALL)
        if match:
            raw = match.group(1)
            names = [
                s.strip().strip("'\"")
                for s in raw.split(",")
                if s.strip().strip("'\"")
            ]
            if names:
                return names, "data.yaml"
    except Exception:
        pass
    return [], None


def _parse_classes_txt(file_path: str) -> List[str]:
    """Parse class names from classes.txt / labels.txt / obj.names."""
    if not os.path.isfile(file_path):
        return []
    classes = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                c = line.strip()
                if c:
                    classes.append(c)
    except Exception:
        pass
    return classes


def _parse_raya_header_classes(txt_path: str) -> List[str]:
    """Parse class names from a Raya TXT annotation file header."""
    if not os.path.isfile(txt_path):
        return []
    classes = []
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            in_header = False
            for line in f:
                stripped = line.strip()
                if stripped == "###":
                    if not in_header:
                        in_header = True
                    else:
                        break
                    continue
                if in_header:
                    if stripped.startswith("-") and not re.match(r"^-\s*nc\s*:", stripped, re.IGNORECASE):
                        name = stripped.lstrip("-").strip()
                        if name:
                            classes.append(name)
    except Exception:
        pass
    return classes


# --------------------------------------------------------------------------- #
# Annotation format detection
# --------------------------------------------------------------------------- #


def _detect_annotation_format(file_path: str) -> str:
    """Detect format of an annotation file."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".txt":
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                head = f.read(512)
                if "###" in head:
                    return "raya_txt"
                # Check for YOLO bounding boxes: "class x_center y_center width height"
                lines = head.strip().split("\n")
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5 and parts[0].isdigit():
                        return "yolo_txt"
            return "raya_txt"
        except Exception:
            return "raya_txt"
    elif ext == ".json":
        return "coco_json"
    elif ext == ".xml":
        return "voc_xml"
    elif ext == ".csv":
        return "csv"
    return "unknown"


def _find_annotation_file(
    video_path: str,
    root_dir: str,
    split_dir: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Search for an annotation file matching a video file.
    Checks:
      1. Alongside video: <video_dir>/<base_name>.<ext>
      2. In split labels folder: <split_dir>/labels/<base_name>.<ext>
      3. In dataset labels folder: <root_dir>/labels/<base_name>.<ext>
    """
    video_dir = os.path.dirname(video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    candidate_dirs = [video_dir]
    if split_dir:
        candidate_dirs.extend([
            os.path.join(split_dir, "labels"),
            os.path.join(split_dir, "annotations"),
        ])
    candidate_dirs.extend([
        os.path.join(root_dir, "labels"),
        os.path.join(root_dir, "annotations"),
    ])

    clean_base = re.sub(r"_(blurred|blured)$", "", base_name, flags=re.IGNORECASE)

    for c_dir in candidate_dirs:
        if not os.path.isdir(c_dir):
            continue
        # Exact base name matches first
        for ext in ANNOTATION_EXTENSIONS:
            p = os.path.join(c_dir, base_name + ext)
            if os.path.isfile(p):
                return p, _detect_annotation_format(p)
        # Clean base name matches
        if clean_base != base_name:
            for ext in ANNOTATION_EXTENSIONS:
                p = os.path.join(c_dir, clean_base + ext)
                if os.path.isfile(p):
                    return p, _detect_annotation_format(p)

    return None, None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def is_video_file(path: str) -> bool:
    """Return True if path is a recognized video file."""
    ext = os.path.splitext(path)[1].lower()
    return ext in VIDEO_EXTENSIONS


def detect_video_folder(folder_path: str) -> bool:
    """
    Check if a folder contains videos (directly or within splits/subfolders).
    Returns True if at least one valid video file is found.
    """
    if not os.path.isdir(folder_path):
        return False
    try:
        for root, dirs, files in os.walk(folder_path):
            if "removed" in root or ".git" in root:
                continue
            for f in files:
                if is_video_file(f) and f.lower() not in EXCLUDE_OUTPUT_VIDEOS:
                    return True
    except OSError:
        return False
    return False


def scan_video_dataset(folder_path: str) -> VideoDatasetInfo:
    """
    Scans a directory for video files and builds a VideoDatasetInfo object.
    
    Supports:
    - Flat directories
    - train / valid / test split directories
    - videos/ and labels/ structures
    - Nested subdirectories
    """
    folder_path = os.path.abspath(folder_path)
    if not os.path.isdir(folder_path):
        return VideoDatasetInfo(root=folder_path, layout="flat")

    entries = os.listdir(folder_path)
    
    # 1. Discover classes from root data.yaml / classes.txt
    classes: List[str] = []
    classes_source: Optional[str] = None
    classes_conflict: Optional[str] = None

    for yaml_candidate in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml"):
        yp = os.path.join(folder_path, yaml_candidate)
        if os.path.isfile(yp):
            c, src = _parse_yaml_classes(yp)
            if c:
                classes = c
                classes_source = src
                break

    if not classes:
        for cf_candidate in CLASS_FILE_NAMES:
            cfp = os.path.join(folder_path, cf_candidate)
            if os.path.isfile(cfp):
                c = _parse_classes_txt(cfp)
                if c:
                    classes = c
                    classes_source = cf_candidate
                    break

    # 2. Check for train/val/test splits
    split_dirs: Dict[str, str] = {}
    for entry in entries:
        full_entry = os.path.join(folder_path, entry)
        if os.path.isdir(full_entry):
            entry_lower = entry.lower()
            if entry_lower in SPLIT_ALIASES:
                canonical = SPLIT_ALIASES[entry_lower]
                split_dirs[canonical] = full_entry
            elif entry_lower in SPLIT_NAMES:
                split_dirs[entry_lower] = full_entry

    splits: List[VideoSplitInfo] = []
    layout = "flat"

    if split_dirs:
        # Layout C: splits
        layout = "splits"
        for split_name in ("train", "valid", "test"):
            if split_name in split_dirs:
                s_dir = split_dirs[split_name]
                split_info = _scan_split_directory(s_dir, split_name, folder_path)
                splits.append(split_info)
        # Any custom named splits
        for s_name, s_dir in split_dirs.items():
            if s_name not in ("train", "valid", "test"):
                splits.append(_scan_split_directory(s_dir, s_name, folder_path))
    elif "videos" in [e.lower() for e in entries] and os.path.isdir(os.path.join(folder_path, "videos")):
        # Layout B: videos/ + labels/
        layout = "videos_labels"
        v_dir = os.path.join(folder_path, "videos")
        split_info = _scan_split_directory(v_dir, "root", folder_path)
        splits.append(split_info)
    else:
        # Layout A or D: Flat or Nested
        root_videos = [
            f for f in entries
            if is_video_file(f) and f.lower() not in EXCLUDE_OUTPUT_VIDEOS and os.path.isfile(os.path.join(folder_path, f))
        ]
        
        if root_videos:
            layout = "flat"
            split_info = _scan_split_directory(folder_path, "root", folder_path, recursive=False)
            splits.append(split_info)
        else:
            # Check for nested subfolders containing videos
            layout = "nested"
            subdirs = [
                d for d in entries
                if os.path.isdir(os.path.join(folder_path, d)) and d not in ("removed", ".git", "autosaves", "auto_save")
            ]
            if subdirs:
                for sub in natsorted(subdirs):
                    s_dir = os.path.join(folder_path, sub)
                    split_info = _scan_split_directory(s_dir, sub, folder_path, recursive=True)
                    if split_info.videos:
                        splits.append(split_info)
            if not splits:
                # Fallback recursive scan from root
                split_info = _scan_split_directory(folder_path, "root", folder_path, recursive=True)
                splits.append(split_info)

    # 3. If no classes found yet from yaml/classes.txt, try inferring from Raya TXT headers
    if not classes:
        inferred_classes: List[str] = []
        seen_classes: Set[str] = set()
        for split in splits:
            for video in split.videos:
                if video.annotation_file and video.annotation_format == "raya_txt":
                    hdr_classes = _parse_raya_header_classes(video.annotation_file)
                    for c in hdr_classes:
                        c_clean = c.strip()
                        if c_clean and c_clean.lower() not in seen_classes:
                            seen_classes.add(c_clean.lower())
                            inferred_classes.append(c_clean)
        if inferred_classes:
            classes = inferred_classes
            classes_source = "raya_header"

    # Fallback to default class if still empty
    if not classes:
        classes = ["object"]
        classes_source = "default"

    dataset_info = VideoDatasetInfo(
        root=folder_path,
        layout=layout,
        splits=splits,
        classes=classes,
        classes_source=classes_source,
        classes_conflict=classes_conflict,
    )

    return dataset_info


def _scan_split_directory(
    split_dir: str,
    split_name: str,
    dataset_root: str,
    recursive: bool = True,
) -> VideoSplitInfo:
    """Scans a directory for video files and matching annotations."""
    videos: List[VideoInfo] = []

    if recursive:
        all_video_paths = []
        for root, dirs, files in os.walk(split_dir):
            if "removed" in root or ".git" in root or "autosaves" in root:
                continue
            for f in files:
                if is_video_file(f) and f.lower() not in EXCLUDE_OUTPUT_VIDEOS:
                    all_video_paths.append(os.path.join(root, f))
    else:
        all_video_paths = [
            os.path.join(split_dir, f)
            for f in os.listdir(split_dir)
            if is_video_file(f) and f.lower() not in EXCLUDE_OUTPUT_VIDEOS and os.path.isfile(os.path.join(split_dir, f))
        ]

    # Natural sort
    all_video_paths = natsorted(all_video_paths)

    for vp in all_video_paths:
        fn = os.path.basename(vp)
        rel_path = os.path.relpath(vp, dataset_root)
        ann_file, ann_fmt = _find_annotation_file(vp, dataset_root, split_dir)

        v_info = VideoInfo(
            path=vp,
            filename=fn,
            relative_path=rel_path,
            split=split_name,
            annotation_file=ann_file,
            annotation_format=ann_fmt,
        )
        v_info.update_status()
        videos.append(v_info)

    return VideoSplitInfo(name=split_name, path=split_dir, videos=videos)
