#!/usr/bin/env python3
"""
Merge a folder of (video_N.mp4, video_N.txt) dataset pairs into a single
video + single label file.

Rules implemented (per spec discussed with user):
- Videos are natsorted by their N index (video_0, video_1, video_2, ...).
- If both video_N.mp4 and video_N_blurred.mp4 exist, the blurred version is
  used and the plain one is ignored.
- Each .txt has a header block:
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
  across all videos per user confirmation) and one merged .txt with a
  single combined header followed by all kept, remapped label lines in
  order.

Usage:
    python3 merge_dataset.py /path/to/main_folder --out-video out.mp4 --out-labels out.txt
"""

import argparse
import re
import sys
from pathlib import Path

import cv2
from natsort import natsorted


HEADER_RE = re.compile(r"^###\s*$")
LABEL_ITEM_RE = re.compile(r"\[([^\]]*)\]")


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
        # Match the nc line strictly: "-nc:4" or "- nc: 4" etc, where the
        # token right after the leading dash(es)/spaces is literally "nc".
        # This must NOT match class names that merely contain "nc" as a
        # substring (e.g. "Motorcycle", "Ambulance", "Fence").
        nc_match = re.match(r"^-\s*nc\s*:\s*(\d+)\s*$", stripped, re.IGNORECASE)
        if nc_match:
            nc = int(nc_match.group(1))
        elif stripped.startswith("-"):
            # e.g. "- Human"
            name = stripped.lstrip("-").strip()
            if name:
                names.append(name)
        # lines like "clasess:" / "names:" are ignored structurally
        idx += 1

    return names, nc, idx


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
        return ("EMPTY", [])  # treat stray blank lines as empty frames (defensive)

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


def discover_pairs(folder: Path):
    """
    Find video_N(.mp4 / _blurred.mp4) + video_N.txt triples, natsorted by N.
    Returns list of dicts: {n, video_path, label_path}
    """
    all_files = list(folder.iterdir())
    video_re = re.compile(r"^video_(\d+)(_blurred)?\.mp4$", re.IGNORECASE)

    by_n = {}  # n -> {"plain": path or None, "blurred": path or None}
    for f in all_files:
        m = video_re.match(f.name)
        if not m:
            continue
        n = int(m.group(1))
        is_blurred = m.group(2) is not None
        by_n.setdefault(n, {"plain": None, "blurred": None})
        if is_blurred:
            by_n[n]["blurred"] = f
        else:
            by_n[n]["plain"] = f

    pairs = []
    for n in natsorted(by_n.keys()):
        entry = by_n[n]
        use_blurred = entry["blurred"] is not None
        video_path = entry["blurred"] if use_blurred else entry["plain"]
        if video_path is None:
            continue

        # Label file naming follows the chosen video's naming: if we're
        # using the blurred video, prefer video_N_blurred.txt (falling back
        # to video_N.txt if the blurred-specific label file doesn't exist).
        if use_blurred:
            candidate_label_paths = [folder / f"video_{n}_blurred.txt", folder / f"video_{n}.txt"]
        else:
            candidate_label_paths = [folder / f"video_{n}.txt"]

        label_path = next((p for p in candidate_label_paths if p.exists()), None)
        if label_path is None:
            tried = ", ".join(p.name for p in candidate_label_paths)
            print(f"WARNING: no label file for video_{n} ({video_path.name}), tried [{tried}], skipping this video", file=sys.stderr)
            continue
        pairs.append({"n": n, "video_path": video_path, "label_path": label_path})

    return pairs


def normalize_name(name: str) -> str:
    return name.strip().lower()


def main():
    ap = argparse.ArgumentParser(description="Merge dataset videos + labels into one video/txt")

    ap.add_argument("--fourcc", type=str, default="avc1", help="FourCC for H.264 (try 'avc1' or 'h264' or 'mp4v' as fallback)")
    args = ap.parse_args()
    folder = Path(
        "/media/reza/New Volume/Reza/AiCLab/MLM/Lableing/videos/IDF_D9_armored_bulldozer/"
    )
    out_video = folder / 'outvideo.mp4'
    out_labels = folder / 'outvideo.txt'
    folder = Path(folder)
    if not folder.is_dir():
        print(f"ERROR: {folder} is not a directory", file=sys.stderr)
        sys.exit(1)

    pairs = discover_pairs(folder)
    if not pairs:
        print("ERROR: no video/label pairs found", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pairs)} video/label pairs (in order):")
    for p in pairs:
        print(f"  video_{p['n']}: {p['video_path'].name}  |  {p['label_path'].name}")

    # ---- Pass 1: parse all headers, build global class list, and read all label lines ----
    global_names = []          # display names, first-appearance order
    global_name_to_idx = {}    # normalized name -> global idx

    per_video_data = []  # list of dicts: {n, video_path, local_idx_to_global_idx, label_lines}

    for p in pairs:
        with open(p["label_path"], "r", encoding="utf-8") as f:
            raw_lines = f.readlines()

        try:
            local_names, nc, body_start = parse_header(raw_lines)
        except ValueError as e:
            print(f"ERROR parsing header of {p['label_path']}: {e}", file=sys.stderr)
            sys.exit(1)

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
            "n": p["n"],
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
        print(f"ERROR: could not open {per_video_data[0]['video_path']}", file=sys.stderr)
        sys.exit(1)
    fps = first_cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(first_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(first_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    first_cap.release()

    print(f"\nOutput video params: {width}x{height} @ {fps:.3f}fps")

    fourcc = cv2.VideoWriter_fourcc(*args.fourcc)
    writer = cv2.VideoWriter(out_video, fourcc, fps, (width, height))
    if not writer.isOpened():
        print(f"WARNING: VideoWriter failed to open with fourcc={fourcc}, retrying with 'mp4v'", file=sys.stderr)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_video, fourcc, fps, (width, height))
        if not writer.isOpened():
            print("ERROR: could not open VideoWriter with any fourcc", file=sys.stderr)
            sys.exit(1)

    out_label_lines = []
    total_frames_in = 0
    total_frames_kept = 0
    total_frames_deleted = 0

    for vd in per_video_data:
        cap = cv2.VideoCapture(str(vd["video_path"]))
        if not cap.isOpened():
            print(f"ERROR: could not open {vd['video_path']}", file=sys.stderr)
            sys.exit(1)

        n_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        n_label_lines = len(vd["body_lines"])
        if n_video_frames != n_label_lines:
            print(f"WARNING: {vd['video_path'].name} has {n_video_frames} frames but "
                  f"{vd['label_path'].name} has {n_label_lines} label lines. "
                  f"Will process min({n_video_frames},{n_label_lines}).", file=sys.stderr)

        n_to_process = min(n_video_frames, n_label_lines) if n_video_frames and n_label_lines else max(n_video_frames, n_label_lines)

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
                print(f"ERROR parsing label line {frame_idx} in {vd['label_path']}: {e}", file=sys.stderr)
                sys.exit(1)

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
                        print(f"ERROR: class index {cls_local} in {vd['label_path']} frame {frame_idx} "
                              f"has no entry in that video's header names list", file=sys.stderr)
                        sys.exit(1)
                    cls_global = local_idx_to_global_idx[cls_local]
                    rest = box[1:]
                    remapped_items.append(f"[{cls_global}," + ",".join(rest) + "]")
                out_label_lines.append(";".join(remapped_items) + ";")

            writer.write(frame)
            total_frames_kept += 1
            frame_idx += 1

        cap.release()
        print(f"Processed video_{vd['n']}: {frame_idx} frames read, "
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


if __name__ == "__main__":
    main()
