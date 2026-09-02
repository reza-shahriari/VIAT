#!/usr/bin/env python3
"""
Merge a folder of (video, annotation.txt) dataset pairs into a single
video + single label file.

Rules implemented:
- Matches any video files (.mp4, .avi, .mov, .mkv, .webm, .flv, .m4v, .wmv)
  regardless of naming pattern.
- If both plain and blurred versions (ending in _blurred or _blured) exist
  for a video, the blurred version is used and the plain one is ignored.
- Each .txt annotation file must have a valid header block:
    ###
    clasess:
    names:
    - Human
    - AirPlane
    - Car
    - buildings
    -nc:4
    ###
  followed by one label line per frame, in frame order. A line is either:
    []                                  -> no objects, keep frame
    [c,x1,y1,x2,y2,q,s];[c,x1,y1,...];  -> one or more objects, keep frame
    DELETE;                             -> drop this frame entirely (do not
                                            write to output video, do not
                                            write a label line)
- Class lists can differ per video (different order / different sets).
  Names are matched case-insensitively and trimmed of whitespace. A single
  global class list is built (first-appearance order across videos, in the
  order videos are processed), and every label's class index is remapped
  from its source video's local index to the global index.
- Output: one H.264 .mp4 (same resolution/fps as source, assumed identical
  across all videos) and one merged .txt with a single combined header
  followed by all kept, remapped label lines in order.

Usage:
    python3 many2single.py --folder /path/to/main_folder --out-video outvideo.mp4 --out-labels outvideo.txt
"""

import argparse
import re
import sys
from pathlib import Path

import cv2
from natsort import natsorted


HEADER_RE = re.compile(r"^###\s*$")
LABEL_ITEM_RE = re.compile(r"\[([^\]]*)\]")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".m4v", ".wmv"}


def parse_header(lines):
    """
    Parse a header block of the form:
        ###
        clasess:
        names:
        - Human
        - AirPlane
        - Car
        - buildings
        -nc:4
        ###
    Returns (names_list, nc, index_after_header) where index_after_header
    is the line index in `lines` right after the closing '###'.
    """
    idx = 0
    n = len(lines)

    # skip blank lines
    while idx < n and lines[idx].strip() == "":
        idx += 1

    if idx >= n or not HEADER_RE.match(lines[idx].strip()):
        raise ValueError("Expected '###' to start header block")
    idx += 1

    names = []
    nc = None
    while idx < n:
        line = lines[idx]
        stripped = line.strip()
        if HEADER_RE.match(stripped):
            idx += 1
            break
        # Match the nc line strictly: "-nc:4" or "- nc: 4" etc.
        nc_match = re.match(r"^-\s*nc\s*:\s*(\d+)\s*$", stripped, re.IGNORECASE)
        if nc_match:
            nc = int(nc_match.group(1))
        elif stripped.startswith("-"):
            name = stripped.lstrip("-").strip()
            if name:
                names.append(name)
        idx += 1

    return names, nc, idx


def is_valid_label_file(label_path: Path) -> bool:
    """Check if the path exists, is a file, and has a parseable Raya annotation header."""
    if not label_path.is_file():
        return False
    try:
        with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = [f.readline() for _ in range(100)]
        names, nc, body_start = parse_header(lines)
        return body_start > 0
    except Exception:
        return False


def parse_label_line(line):
    """
    Parse one label line (post-header) into one of:
        ("DELETE", None)
        ("EMPTY", [])
        ("BOXES", [ [cls, x1, y1, x2, y2, q, s], ... ])   cls as int, rest as str/num kept as strings
    Raises ValueError on unrecognized format.
    """
    stripped = line.strip()
    if stripped == "":
        return ("EMPTY", [])  # treat stray blank lines as empty frames

    if stripped.upper().startswith("DELETE"):
        return ("DELETE", None)

    if stripped == "[]":
        return ("EMPTY", [])

    items = LABEL_ITEM_RE.findall(stripped)
    if not items:
        raise ValueError(f"Unrecognized label line: {line!r}")

    boxes = []
    for item in items:
        parts = [p.strip() for p in item.split(",")]
        if len(parts) < 5:
            raise ValueError(f"Malformed label item {item!r} in line {line!r}")
        cls = int(float(parts[0]))
        rest = parts[1:]
        boxes.append([cls] + rest)
    return ("BOXES", boxes)


def discover_pairs(folder: Path, out_video: Path = None, out_labels: Path = None):
    """
    Find video + label file pairs in `folder`.
    Supports arbitrary video names and extensions (.mp4, .avi, .mov, etc.).
    If both plain and blurred versions (ending in _blurred or _blured) exist for a video,
    the blurred version is preferred.
    Matches with a .txt file that has a valid Raya annotation format.
    Returns list of dicts: {"name": base_name, "video_path": video_path, "label_path": label_path}
    """
    all_files = list(folder.iterdir())

    # Exclude output video/label filenames if specified, or default outvideo.*
    exclude_names = {"outvideo.mp4", "outvideo.txt", "out_video.mp4", "out_video.txt"}
    if out_video:
        exclude_names.add(out_video.name.lower())
    if out_labels:
        exclude_names.add(out_labels.name.lower())

    # Group video files by base stem (without _blurred / _blured suffix)
    by_base = {}  # base_stem -> {"plain": path, "blurred": path}

    for f in all_files:
        if not f.is_file():
            continue
        if f.name.lower() in exclude_names:
            continue
        if f.suffix.lower() not in VIDEO_EXTENSIONS:
            continue

        stem = f.stem
        stem_lower = stem.lower()
        if stem_lower.endswith("_blurred"):
            base_stem = stem[:-8]
            is_blurred = True
        elif stem_lower.endswith("_blured"):
            base_stem = stem[:-7]
            is_blurred = True
        else:
            base_stem = stem
            is_blurred = False

        by_base.setdefault(base_stem, {"plain": None, "blurred": None})
        if is_blurred:
            by_base[base_stem]["blurred"] = f
        else:
            by_base[base_stem]["plain"] = f

    pairs = []
    for base_stem in natsorted(by_base.keys()):
        entry = by_base[base_stem]
        use_blurred = entry["blurred"] is not None
        video_path = entry["blurred"] if use_blurred else entry["plain"]
        if video_path is None:
            continue

        # Candidate label files in order of preference
        if use_blurred:
            candidate_label_paths = [
                folder / f"{base_stem}_blurred.txt",
                folder / f"{base_stem}_blured.txt",
                folder / f"{video_path.stem}.txt",
                folder / f"{base_stem}.txt",
            ]
        else:
            candidate_label_paths = [
                folder / f"{video_path.stem}.txt",
                folder / f"{base_stem}.txt",
            ]

        # Find the first candidate that exists and has a valid format
        label_path = None
        for cand in candidate_label_paths:
            if is_valid_label_file(cand):
                label_path = cand
                break

        if label_path is None:
            existing_candidates = [p.name for p in candidate_label_paths if p.is_file()]
            if existing_candidates:
                print(f"WARNING: Label file for '{video_path.name}' found [{', '.join(existing_candidates)}] but format is invalid (missing '###' header), skipping this video", file=sys.stderr)
            else:
                tried = ", ".join(p.name for p in candidate_label_paths)
                print(f"WARNING: No label file found for '{video_path.name}' (tried [{tried}]), skipping this video", file=sys.stderr)
            continue

        pairs.append({
            "name": base_stem,
            "video_path": video_path,
            "label_path": label_path,
        })

    return pairs


def normalize_name(name: str) -> str:
    return name.strip().lower()


def merge_dataset_programmatic(folder: Path, out_video: Path, out_labels: Path, fourcc: str = "avc1"):
    if not folder.is_dir():
        raise ValueError(f"'{folder}' is not a directory")

    pairs = discover_pairs(folder, out_video=out_video, out_labels=out_labels)
    if not pairs:
        raise ValueError(f"No valid video/label pairs found in '{folder}'")

    print(f"Found {len(pairs)} video/label pairs (in order):")
    for p in pairs:
        print(f"  [{p['name']}]: {p['video_path'].name}  |  {p['label_path'].name}")

    # ---- Pass 1: parse all headers, build global class list, and read all label lines ----
    global_names = []          # display names, first-appearance order
    global_name_to_idx = {}    # normalized name -> global idx

    per_video_data = []  # list of dicts: {name, video_path, local_idx_to_global_idx, label_lines}

    for p in pairs:
        with open(p["label_path"], "r", encoding="utf-8") as f:
            raw_lines = f.readlines()

        try:
            local_names, nc, body_start = parse_header(raw_lines)
        except ValueError as e:
            raise ValueError(f"ERROR parsing header of {p['label_path']}: {e}")

        if nc is not None and len(local_names) != nc:
            print(f"WARNING: {p['label_path'].name} declares nc={nc} but has {len(local_names)} names", file=sys.stderr)

        local_idx_to_global_idx = {}
        for local_idx, name in enumerate(local_names):
            key = normalize_name(name)
            if key not in global_name_to_idx:
                global_name_to_idx[key] = len(global_names)
                global_names.append(name.strip())
            local_idx_to_global_idx[local_idx] = global_name_to_idx[key]

        body_lines = [ln.rstrip("\n") for ln in raw_lines[body_start:]]
        # drop trailing fully-blank lines at EOF
        while body_lines and body_lines[-1].strip() == "":
            body_lines.pop()

        per_video_data.append({
            "name": p["name"],
            "video_path": p["video_path"],
            "label_path": p["label_path"],
            "local_idx_to_global_idx": local_idx_to_global_idx,
            "body_lines": body_lines,
        })

    print(f"\nGlobal class list ({len(global_names)} classes):")
    for i, name in enumerate(global_names):
        print(f"  {i}: {name}")

    # ---- Pass 2: open first video to get fps/resolution for the writer ----
    first_cap = cv2.VideoCapture(str(per_video_data[0]["video_path"]))
    if not first_cap.isOpened():
        raise ValueError(f"ERROR: could not open {per_video_data[0]['video_path']}")
    fps = first_cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_cap.release()

    print(f"\nOutput video params: {width}x{height} @ {fps:.3f}fps")

    fourcc_val = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(str(out_video), fourcc_val, fps, (width, height))
    if not writer.isOpened():
        print(f"WARNING: VideoWriter failed to open with fourcc={fourcc_val}, retrying with 'mp4v'", file=sys.stderr)
        fourcc_val = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_video), fourcc_val, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("ERROR: could not open VideoWriter with any fourcc")

    out_label_lines = []
    total_frames_in = 0
    total_frames_kept = 0
    total_frames_deleted = 0

    for vd in per_video_data:
        cap = cv2.VideoCapture(str(vd["video_path"]))
        if not cap.isOpened():
            raise ValueError(f"ERROR: could not open {vd['video_path']}")

        n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_label_lines = len(vd["body_lines"])
        if n_video_frames != n_label_lines:
            print(f"WARNING: {vd['video_path'].name} has {n_video_frames} frames but "
                  f"{vd['label_path'].name} has {n_label_lines} label lines. "
                  f"Will process min({n_video_frames},{n_label_lines}).", file=sys.stderr)

        local_idx_to_global_idx = vd["local_idx_to_global_idx"]

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx >= n_label_lines:
                print(f"WARNING: {vd['video_path'].name} has more frames than label lines; "
                      f"stopping at frame {frame_idx} for this video", file=sys.stderr)
                break

            total_frames_in += 1
            line = vd["body_lines"][frame_idx]

            try:
                kind, payload = parse_label_line(line)
            except ValueError as e:
                raise ValueError(f"ERROR parsing label line {frame_idx} in {vd['label_path']}: {e}")

            if kind == "DELETE":
                total_frames_deleted += 1
                frame_idx += 1
                continue

            if kind == "EMPTY":
                out_label_lines.append("[]")
            else:  # BOXES
                remapped_items = []
                for box in payload:
                    cls_local = box[0]
                    if cls_local not in local_idx_to_global_idx:
                        raise ValueError(f"ERROR: class index {cls_local} in {vd['label_path']} frame {frame_idx} "
                              f"has no entry in that video's header names list")
                    cls_global = local_idx_to_global_idx[cls_local]
                    rest = box[1:]
                    remapped_items.append(f"[{cls_global}," + ",".join(rest) + "]")
                out_label_lines.append(";".join(remapped_items) + ";")

            writer.write(frame)
            total_frames_kept += 1
            frame_idx += 1

        cap.release()
        print(f"Processed {vd['video_path'].name}: {frame_idx} frames read, "
              f"{n_video_frames - frame_idx if n_video_frames > frame_idx else 0} unread remainder")

    writer.release()

    # ---- Write merged label file ----
    header_lines = ["###", "clasess:", "names:"]
    header_lines += [f"- {name}" for name in global_names]
    header_lines.append(f"-nc:{len(global_names)}")
    header_lines.append("###")

    with open(out_labels, "w", encoding="utf-8") as f:
        f.write("\n".join(header_lines) + "\n")
        f.write("\n".join(out_label_lines) + "\n")

    print(f"\nDone.")
    print(f"  Frames read total:    {total_frames_in}")
    print(f"  Frames deleted:       {total_frames_deleted}")
    print(f"  Frames kept/written:  {total_frames_kept}")
    print(f"  Output video:  {out_video}")
    print(f"  Output labels: {out_labels}")
    return True, f"Merged successfully. Frames kept: {total_frames_kept}"


def main():
    ap = argparse.ArgumentParser(description="Merge dataset videos + labels into one video/txt")
    ap.add_argument("--fourcc", type=str, default="avc1", help="FourCC for H.264 (try 'avc1' or 'h264' or 'mp4v' as fallback)")
    ap.add_argument("--folder", type=str, required=True, help="Folder containing videos and label files")
    ap.add_argument("--out-video", type=str, default="outvideo.mp4", help="Output video filename/path (default: outvideo.mp4)")
    ap.add_argument("--out-labels", type=str, default="outvideo.txt", help="Output label filename/path (default: outvideo.txt)")
    args = ap.parse_args()

    folder = Path(args.folder)
    out_video = Path(args.out_video) if Path(args.out_video).is_absolute() else folder / args.out_video
    out_labels = Path(args.out_labels) if Path(args.out_labels).is_absolute() else folder / args.out_labels

    try:
        merge_dataset_programmatic(folder, out_video, out_labels, args.fourcc)
    except Exception as e:
        print(e, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
