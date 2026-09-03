#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust MOT format converter for VIAT Evaluation Engine.
Converts various Ground Truth and Detection/Prediction formats (.txt, .json, MOT, Raya, YOLO)
into standard MOT16 benchmark format for TrackEval tracking evaluation.
"""

import os
import glob
import json
import re
import cv2
import pathlib

try:
    import tqdm
except ImportError:
    class _DummyPbar:
        def __init__(self, iterable=None, total=None, desc=None, *args, **kwargs):
            self.iterable = iterable
            self.total = total
            self.n = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            if self.iterable is not None:
                return iter(self.iterable)
            return iter([])

        def update(self, n=1):
            self.n += n

        def set_description(self, *args, **kwargs):
            pass

        def set_postfix(self, *args, **kwargs):
            pass

    class _DummyTqdmModule:
        def __call__(self, iterable=None, *args, **kwargs):
            return _DummyPbar(iterable, *args, **kwargs)

        def tqdm(self, iterable=None, *args, **kwargs):
            return _DummyPbar(iterable, *args, **kwargs)

    tqdm = _DummyTqdmModule()

try:
    from viat.evaluation.conf.configs import config
except ImportError:
    try:
        from ..conf.configs import config
    except ImportError:
        try:
            from conf.configs import config
        except ImportError:
            config = None


def strip_header_lines(lines):
    """
    Strips Raya header blocks (###), comments (#), and YOLO format headers (names: / nc:).
    Returns (cleaned_lines, class_names).
    """
    data_lines = []
    class_names = []
    in_raya_header = False
    in_names_block = False
    header_end_idx = 0
    has_yolo_header = False

    for i, raw in enumerate(lines):
        sline = raw.strip()
        if not in_names_block:
            if sline.lower().startswith("names:"):
                in_names_block = True
                has_yolo_header = True
            elif sline == "###":
                in_raya_header = not in_raya_header
            elif in_raya_header or sline.startswith("#"):
                pass
        else:
            if re.match(r"^-?\s*nc\s*:\s*\d+", sline, re.IGNORECASE):
                header_end_idx = i + 1
                break
            name = re.sub(r"^[\-\*\u2022]\s*", "", sline).strip()
            if name:
                class_names.append(name)

    if has_yolo_header:
        while header_end_idx < len(lines):
            candidate = lines[header_end_idx].strip()
            if candidate == "" or candidate.upper() == "DELETED;" or "[" in candidate:
                break
            header_end_idx += 1
        data_lines = lines[header_end_idx:]
    else:
        in_header = False
        for line in lines:
            sline = line.strip()
            if sline == "###":
                in_header = not in_header
                continue
            if in_header or sline.startswith("#"):
                continue
            data_lines.append(line)

    return data_lines, class_names


def parse_ground_truth_file(file_path, size_thr=0, quality_thr=0):
    """
    Parses a Ground Truth file (.txt or .json) into a list of MOT records:
    [ (frame_idx (1-indexed), track_id (int), x, y, w, h, conf (1.0), class_id (1), visibility (1.0)), ... ]
    and returns (records, max_frame).
    """
    records = []
    max_frame = 0

    if not file_path or not os.path.isfile(file_path):
        return records, max_frame

    # 1. COCO JSON format
    if file_path.endswith('.json'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            images = data.get('images', [])
            annotations = data.get('annotations', [])

            img_id_to_frame = {}
            for idx, img in enumerate(sorted(images, key=lambda x: x.get('id', 0))):
                img_id = img.get('id', idx + 1)
                fn = img.get('file_name', '')
                num_match = re.search(r'(\d+)(?:\.\w+)?$', fn)
                if num_match:
                    try:
                        frame_num = int(num_match.group(1))
                    except ValueError:
                        frame_num = idx + 1
                else:
                    frame_num = idx + 1
                img_id_to_frame[img_id] = frame_num

            for ann in annotations:
                if ann.get('ignore', 0) or ann.get('iscrowd', 0):
                    continue
                bbox = ann.get('bbox', [0, 0, 0, 0])
                if len(bbox) >= 4:
                    x, y, w, h = [float(v) for v in bbox[:4]]
                    if w <= 0 or h <= 0:
                        continue
                    frame_idx = img_id_to_frame.get(ann.get('image_id'), ann.get('image_id', 1))
                    track_id = int(ann.get('track_id', ann.get('id', 1)))
                    max_frame = max(max_frame, frame_idx)
                    records.append((frame_idx, track_id, x, y, w, h, 1.0, 1, 1.0))
            return records, max_frame
        except Exception:
            pass

    # 2. Text format (.txt)
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        raw_lines = f.readlines()

    lines, _ = strip_header_lines(raw_lines)

    # Check if lines are in standard MOT format (e.g., "1, 2, 100, 200, 50, 60, 1, 1, 1")
    is_mot_csv = False
    if lines:
        sample = lines[0].strip().replace(' ', ',').split(',')
        if len(sample) >= 7 and all(re.match(r'^[-+]?(?:\d*\.\d+|\d+)$', s.strip()) for s in sample[:6] if s.strip()):
            is_mot_csv = True

    if is_mot_csv:
        for line in lines:
            parts = [p.strip() for p in line.strip().replace(' ', ',').split(',') if p.strip()]
            if len(parts) >= 6:
                try:
                    f_idx = int(float(parts[0]))
                    t_id = int(float(parts[1]))
                    x = float(parts[2])
                    y = float(parts[3])
                    w = float(parts[4])
                    h = float(parts[5])
                    conf = float(parts[6]) if len(parts) > 6 else 1.0
                    cls_id = int(float(parts[7])) if len(parts) > 7 else 1
                    vis = float(parts[8]) if len(parts) > 8 else 1.0
                    if w > 0 and h > 0:
                        max_frame = max(max_frame, f_idx)
                        records.append((f_idx, t_id, x, y, w, h, conf, cls_id, vis))
                except (ValueError, IndexError):
                    continue
        return records, max_frame

    # VIAT / Raya frame-by-frame text format (line i = frame i + 1)
    for i, line in enumerate(lines):
        frame_idx = i + 1
        max_frame = max(max_frame, frame_idx)
        sline = line.strip()
        if not sline or "DELETED" in sline.upper() or len(sline) < 3:
            continue

        boxes = [b.strip() for b in sline.split(';') if b.strip()]
        for b_idx, b_str in enumerate(boxes):
            parsed = None
            try:
                parsed = eval(b_str)
            except Exception:
                pass

            if isinstance(parsed, (list, tuple)) and len(parsed) >= 4:
                # Raya: [cat, x, y, w, h, size, quality, difficult, track_id?]
                if len(parsed) >= 8 and parsed[7] == 1:
                    continue
                if len(parsed) >= 7:
                    sz = parsed[5]
                    q = parsed[6]
                    if (size_thr > 0 and sz > 0 and sz < size_thr) or (quality_thr > 0 and q > 0 and q < quality_thr):
                        continue
                track_id = int(parsed[8]) if len(parsed) >= 9 else (b_idx + 1)
                x, y, w, h = [float(v) for v in parsed[1:5]]
                if w > 0 and h > 0:
                    records.append((frame_idx, track_id, x, y, w, h, 1.0, 1, 1.0))
            else:
                nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", b_str)
                if len(nums) >= 4:
                    vals = [float(v) for v in nums]
                    if len(vals) >= 5:
                        x, y, w, h = vals[1:5]
                    else:
                        x, y, w, h = vals[0:4]
                    if w > 0 and h > 0:
                        records.append((frame_idx, b_idx + 1, x, y, w, h, 1.0, 1, 1.0))

    # Ensure unique track_id per frame for TrackEval MOT compliance
    cleaned_records = []
    used_frame_ids = {}
    for rec in records:
        f_idx, t_id, x, y, w, h, conf, cls_id, vis = rec
        if f_idx not in used_frame_ids:
            used_frame_ids[f_idx] = set()
        while t_id in used_frame_ids[f_idx] or t_id <= 0:
            t_id += 1
        used_frame_ids[f_idx].add(t_id)
        cleaned_records.append((f_idx, t_id, x, y, w, h, conf, cls_id, vis))

    return cleaned_records, max_frame


def parse_detection_file(file_path):
    """
    Parses a Prediction/Detection file (.txt or .json) into a list of MOT records:
    [ (frame_idx (1-indexed), track_id (int), x, y, w, h, score (float), -1, -1, -1), ... ]
    and returns (records, max_frame).
    """
    records = []
    max_frame = 0

    if not file_path or not os.path.isfile(file_path):
        return records, max_frame

    # 1. COCO JSON format
    if file_path.endswith('.json'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            images = data.get('images', [])
            annotations = data.get('annotations', [])

            img_id_to_frame = {}
            for idx, img in enumerate(sorted(images, key=lambda x: x.get('id', 0))):
                img_id = img.get('id', idx + 1)
                fn = img.get('file_name', '')
                num_match = re.search(r'(\d+)(?:\.\w+)?$', fn)
                if num_match:
                    try:
                        frame_num = int(num_match.group(1))
                    except ValueError:
                        frame_num = idx + 1
                else:
                    frame_num = idx + 1
                img_id_to_frame[img_id] = frame_num

            for ann in annotations:
                bbox = ann.get('bbox', [0, 0, 0, 0])
                if len(bbox) >= 4:
                    x, y, w, h = [float(v) for v in bbox[:4]]
                    if w <= 0 or h <= 0:
                        continue
                    frame_idx = img_id_to_frame.get(ann.get('image_id'), ann.get('image_id', 1))
                    track_id = int(ann.get('track_id', ann.get('id', 1)))
                    score = float(ann.get('score', 1.0))
                    max_frame = max(max_frame, frame_idx)
                    records.append((frame_idx, track_id, x, y, w, h, score, -1, -1, -1))
        except Exception:
            pass
    else:
        # 2. Text format (.txt)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            raw_lines = f.readlines()

        lines, _ = strip_header_lines(raw_lines)

        # Check if lines are in standard MOT format
        is_mot_csv = False
        if lines:
            sample = lines[0].strip().replace(' ', ',').split(',')
            if len(sample) >= 6 and all(re.match(r'^[-+]?(?:\d*\.\d+|\d+)$', s.strip()) for s in sample[:6] if s.strip()):
                is_mot_csv = True

        if is_mot_csv:
            for line in lines:
                parts = [p.strip() for p in line.strip().replace(' ', ',').split(',') if p.strip()]
                if len(parts) >= 6:
                    try:
                        f_idx = int(float(parts[0]))
                        t_id = int(float(parts[1]))
                        x = float(parts[2])
                        y = float(parts[3])
                        w = float(parts[4])
                        h = float(parts[5])
                        score = float(parts[6]) if len(parts) > 6 else 1.0
                        if w > 0 and h > 0:
                            max_frame = max(max_frame, f_idx)
                            records.append((f_idx, t_id, x, y, w, h, score, -1, -1, -1))
                    except (ValueError, IndexError):
                        continue
        else:
            # VIAT / Model prediction line format (line i = frame i + 1)
            for i, line in enumerate(lines):
                frame_idx = i + 1
                max_frame = max(max_frame, frame_idx)
                sline = line.strip().rstrip(';').strip()
                if not sline or "DELETED" in sline.upper() or len(sline) < 3:
                    continue

                det_items = None
                try:
                    det_items = eval(sline)
                except Exception:
                    pass

                if isinstance(det_items, (list, tuple)):
                    if len(det_items) > 0 and isinstance(det_items[0], (int, float)):
                        det_items = [det_items]
                else:
                    matches = re.findall(r'\[([^\[\]]+)\]', sline)
                    det_items = []
                    for m in matches:
                        nums = [float(v) for v in re.findall(r"[-+]?(?:\d*\.\d+|\d+)", m)]
                        if len(nums) >= 4:
                            det_items.append(nums)

                for d_idx, det in enumerate(det_items):
                    if not isinstance(det, (list, tuple)) or len(det) < 4:
                        continue

                    vals = [float(v) for v in det]
                    track_id = d_idx + 1
                    score = 1.0

                    if len(vals) == 5:
                        # [x1, y1, x2, y2, score] or [x, y, w, h, score]
                        score = vals[4]
                        if vals[2] > vals[0] and vals[3] > vals[1]:
                            x, y, w, h = vals[0], vals[1], vals[2] - vals[0], vals[3] - vals[1]
                        else:
                            x, y, w, h = vals[0], vals[1], vals[2], vals[3]
                    elif len(vals) >= 6:
                        # [class_id, x1, y1, x2, y2, score] or [track_id, x1, y1, x2, y2, score]
                        score = vals[5]
                        if vals[3] > vals[1] and vals[4] > vals[2]:
                            x, y, w, h = vals[1], vals[2], vals[3] - vals[1], vals[4] - vals[2]
                        else:
                            x, y, w, h = vals[1], vals[2], vals[3], vals[4]
                    else:
                        x, y, w, h = vals[0], vals[1], vals[2], vals[3]

                    if w > 0 and h > 0:
                        records.append((frame_idx, track_id, x, y, w, h, score, -1, -1, -1))

    # Ensure unique track_id per frame for TrackEval MOT compliance
    cleaned_records = []
    used_frame_ids = {}
    for rec in records:
        f_idx, t_id, x, y, w, h, score, x_val, y_val, z_val = rec
        if f_idx not in used_frame_ids:
            used_frame_ids[f_idx] = set()
        while t_id in used_frame_ids[f_idx] or t_id <= 0:
            t_id += 1
        used_frame_ids[f_idx].add(t_id)
        cleaned_records.append((f_idx, t_id, x, y, w, h, score, x_val, y_val, z_val))

    return cleaned_records, max_frame


def find_video_properties(seq_name, search_dirs, fallback_length=1):
    """
    Locates video file for sequence and extracts fps, width, height, seq_length.
    Falls back gracefully to standard defaults if video file is not found.
    """
    exts = ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.mpg', '.MOV', '.m4v', '.flv']
    video_path = None

    for sdir in search_dirs:
        if not sdir or not os.path.exists(sdir):
            continue
        for ext in exts:
            c1 = os.path.join(sdir, seq_name + ext)
            if os.path.isfile(c1):
                video_path = c1
                break
        if video_path:
            break
        for sub in ['videos', 'images', os.path.basename(seq_name)]:
            subdir = os.path.join(sdir, sub)
            if os.path.isdir(subdir):
                for ext in exts:
                    c2 = os.path.join(subdir, seq_name + ext)
                    if os.path.isfile(c2):
                        video_path = c2
                        break
            if video_path:
                break
        if video_path:
            break

    if video_path and os.path.isfile(video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        seq_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if seq_len <= 0:
            seq_len = max(fallback_length, 1)
        return int(fps), width, height, seq_len

    return 30, 1920, 1080, max(fallback_length, 1)


def convert_to_mot(input_gt_path, input_dt_path, Config: config, size_thr=0, quality_thr=0):
    """
    Converts ground truth and prediction annotations into MOT16 benchmark structure.
    Does NOT assert line counts and supports any format (.txt, .json, Raya, YOLO, MOT).
    """
    gt_paths = [input_gt_path] if isinstance(input_gt_path, str) else list(input_gt_path)
    dt_paths = [input_dt_path] if isinstance(input_dt_path, str) else list(input_dt_path)

    # Collect all available sequences in GT and DT
    gt_seq_files = {}
    for p in gt_paths:
        if not os.path.isdir(p):
            continue
        for ext in ['*.txt', '*.json']:
            for f in glob.glob(os.path.join(p, ext)):
                bname = os.path.splitext(os.path.basename(f))[0]
                if bname in ('all_video', 'diagnostics', 'classes', 'obj.names', 'seqinfo'):
                    continue
                gt_seq_files[bname] = f

    dt_seq_files = {}
    for p in dt_paths:
        if not os.path.isdir(p):
            continue
        for ext in ['*.txt', '*.json']:
            for f in glob.glob(os.path.join(p, ext)):
                bname = os.path.splitext(os.path.basename(f))[0]
                if bname in ('all_video', 'diagnostics', 'classes', 'obj.names', 'seqinfo'):
                    continue
                dt_seq_files[bname] = f

    common_seqs = sorted(list(set(gt_seq_files.keys()).intersection(set(dt_seq_files.keys()))))
    if not common_seqs:
        common_seqs = sorted(list(gt_seq_files.keys()))

    if not common_seqs:
        return

    gt_out_path = os.path.join(Config.gt_path, 'MOT16-train')
    dt_out_path = os.path.join(Config.tracker_path, 'MOT16-train', 'botsort', 'data')
    pathlib.Path(gt_out_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(dt_out_path).mkdir(parents=True, exist_ok=True)

    evaluated_seq_names = []

    for seq_name in common_seqs:
        gt_file = gt_seq_files.get(seq_name)
        dt_file = dt_seq_files.get(seq_name)

        gt_records, max_gt_f = parse_ground_truth_file(gt_file, size_thr=size_thr, quality_thr=quality_thr)
        dt_records, max_dt_f = parse_detection_file(dt_file)

        fallback_len = max(max_gt_f, max_dt_f, 1)
        fps, width, height, seq_len = find_video_properties(
            seq_name,
            search_dirs=gt_paths + dt_paths + [os.path.dirname(p) for p in gt_paths],
            fallback_length=fallback_len
        )

        # 1. Write GT file: <gt_out_path>/<seq_name>/gt/gt.txt
        seq_gt_dir = os.path.join(gt_out_path, seq_name, 'gt')
        pathlib.Path(seq_gt_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(seq_gt_dir, 'gt.txt'), 'w', encoding='utf-8') as f:
            for rec in gt_records:
                # frame, id, bb_left, bb_top, bb_width, bb_height, conf, class_id, visibility
                f.write(f"{rec[0]},{rec[1]},{rec[2]:.2f},{rec[3]:.2f},{rec[4]:.2f},{rec[5]:.2f},{rec[6]:.2f},{rec[7]},{rec[8]:.2f}\n")

        # 2. Write Tracker detection file: <dt_out_path>/<seq_name>.txt
        with open(os.path.join(dt_out_path, f"{seq_name}.txt"), 'w', encoding='utf-8') as f:
            for rec in dt_records:
                # frame, id, bb_left, bb_top, bb_width, bb_height, conf, x, y, z
                f.write(f"{rec[0]},{rec[1]},{rec[2]:.2f},{rec[3]:.2f},{rec[4]:.2f},{rec[5]:.2f},{rec[6]:.4f},-1,-1,-1\n")

        # 3. Write GT det file: <gt_out_path>/<seq_name>/det/det.txt
        seq_det_dir = os.path.join(gt_out_path, seq_name, 'det')
        pathlib.Path(seq_det_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(seq_det_dir, 'det.txt'), 'w', encoding='utf-8') as f:
            for rec in dt_records:
                f.write(f"{rec[0]},{rec[1]},{rec[2]:.2f},{rec[3]:.2f},{rec[4]:.2f},{rec[5]:.2f},{rec[6]:.4f},-1,-1,-1\n")

        # 4. Write sequence info: <gt_out_path>/<seq_name>/seqinfo.ini
        seq_info_file = os.path.join(gt_out_path, seq_name, 'seqinfo.ini')
        with open(seq_info_file, 'w', encoding='utf-8') as f:
            f.write('[Sequence]\n')
            f.write(f'name={seq_name}\n')
            f.write('imDir=img1\n')
            f.write(f'frameRate={fps}\n')
            f.write(f'seqLength={seq_len}\n')
            f.write(f'imWidth={width}\n')
            f.write(f'imHeight={height}\n')
            f.write('imExt=.jpg\n')

        evaluated_seq_names.append(seq_name)

    # 5. Write seqmap file
    seqmap_dirs = [
        os.path.join(Config.gt_path, 'seqmaps'),
        os.path.join(Config.gt_path, 'MOT16-train', 'seqmaps'),
        os.path.join(Config.gt_path, 'Track', 'gt', 'seqmaps')
    ]
    for sm_dir in seqmap_dirs:
        pathlib.Path(sm_dir).mkdir(parents=True, exist_ok=True)
        for sm_name in ['MOT16-train.txt', 'MOT16.txt']:
            with open(os.path.join(sm_dir, sm_name), 'w', encoding='utf-8') as f:
                f.write("MOT16\n")
                for name in evaluated_seq_names:
                    f.write(f"{name}\n")

    



