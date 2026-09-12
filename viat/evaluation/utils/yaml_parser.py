import os
import re

def parse_yolo_yaml(yaml_path):
    """
    Parse a YOLO data.yaml file to extract class names dictionary: {id: name}.
    Supports standard pyyaml format or simple regex parsing.
    """
    if not yaml_path or not os.path.exists(yaml_path):
        return {}

    names_dict = {}
    try:
        import yaml
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            if isinstance(data, dict):
                names = data.get('names', {})
                if isinstance(names, dict):
                    names_dict = {int(k): str(v) for k, v in names.items()}
                elif isinstance(names, list):
                    names_dict = {i: str(v) for i, v in enumerate(names)}
    except Exception:
        # Fallback regex parser if PyYAML is unavailable or malformed file
        with open(yaml_path, 'r', encoding='utf-8') as f:
            content = f.read()
            # Match names: ['cat', 'dog'] or names:\n 0: cat\n 1: dog
            list_match = re.search(r'names:\s*\[(.*?)\]', content, re.DOTALL)
            if list_match:
                raw_items = list_match.group(1).split(',')
                names_dict = {i: item.strip().strip("'\"") for i, item in enumerate(raw_items) if item.strip()}
            else:
                dict_matches = re.findall(r'(\d+)\s*:\s*[\'"]?([^\'"\n#]+)[\'"]?', content)
                for k, v in dict_matches:
                    names_dict[int(k)] = v.strip()

    return names_dict


NC_LINE_RE = re.compile(r"^-?\s*nc\s*(:\s*\d*)?$", re.IGNORECASE)
BULLET_RE = re.compile(r"^[\-\*\u2022]\s*")
BOX_GROUP_RE = re.compile(r"\[([^\[\]]*)\]")
HASH_HEADER_RE = re.compile(r"^#{3,}")


def parse_annotation_file(path):
    """
    Parse a Raya format per-frame annotation file.
    Returns (local_class_names: list[str], frame_lines: list[str]).
    """
    from pathlib import Path
    if isinstance(path, (str, bytes)):
        path = Path(path)
    raw_lines = path.read_text(encoding="utf-8", errors="replace").splitlines()

    names = []
    header_end = None
    in_names_block = False
    in_hash_header = False

    for i, raw in enumerate(raw_lines):
        line = raw.strip()
        if HASH_HEADER_RE.match(line):
            in_hash_header = not in_hash_header
            if not in_hash_header:
                header_end = i + 1
                break
            continue

        if in_hash_header:
            if line.lower().startswith(("names:", "clasess:", "classes:")):
                in_names_block = True
                continue
            if NC_LINE_RE.match(line):
                continue
            if in_names_block and (line.startswith("-") or line.startswith("*") or line.startswith("•")):
                name = BULLET_RE.sub("", line).strip()
                if name and not HASH_HEADER_RE.match(name) and not NC_LINE_RE.match(name):
                    names.append(name)
            continue

        if not in_names_block:
            if line.lower().startswith(("names:", "clasess:", "classes:")):
                in_names_block = True
            continue
        if NC_LINE_RE.match(line) or HASH_HEADER_RE.match(line):
            header_end = i + 1
            break
        if line == "":
            continue
        name = BULLET_RE.sub("", line).strip()
        if name and not HASH_HEADER_RE.match(name) and not NC_LINE_RE.match(name):
            names.append(name)

    if header_end is None:
        header_end = 0

    while header_end < len(raw_lines):
        candidate = raw_lines[header_end].strip()
        if (
            candidate == ""
            or candidate.upper() in ("DELETED;", "DELETE;", "DELETED", "DELETE")
            or BOX_GROUP_RE.search(candidate)
            or candidate == "[]"
        ):
            break
        header_end += 1

    frame_lines = raw_lines[header_end:]
    return names, frame_lines


def scan_dataset_classes(gt_path):
    """
    Scan a directory or file for annotated class IDs or names.
    Supports YOLO txt, Raya txt, COCO json, and Pascal VOC XML.
    """
    if not gt_path or not os.path.exists(gt_path):
        return []

    classes_found = []

    # 1. Check txt files (support both file and directory paths)
    txt_files = []
    json_files = []
    if os.path.isfile(gt_path):
        if gt_path.endswith('.txt'):
            txt_files.append(gt_path)
        elif gt_path.endswith('.json'):
            json_files.append(gt_path)
    else:
        for root, _, files in os.walk(gt_path):
            for f in files:
                if f.endswith('.txt'):
                    txt_files.append(os.path.join(root, f))
                elif f.endswith('.json'):
                    json_files.append(os.path.join(root, f))

    # 2. Check for classes defined in annotation headers or classes.txt
    header_classes = []
    for file_p in txt_files:
        base_name = os.path.splitext(os.path.basename(file_p))[0].lower()
        if base_name in {'per_class_metrics', 'diagnostics', 'combined', 'notes', 'readme'}:
            continue
        if base_name == 'classes':
            try:
                with open(file_p, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if line and not HASH_HEADER_RE.match(line) and not NC_LINE_RE.match(line):
                            if line not in header_classes:
                                header_classes.append(line)
            except Exception:
                pass
            continue

        try:
            local_names, _ = parse_annotation_file(file_p)
            for name in local_names:
                if name not in header_classes:
                    header_classes.append(name)
        except Exception:
            pass

    # 3. If header-defined class names exist, use them exclusively.
    # Otherwise fallback to scanning bounding boxes for numeric class IDs.
    if not header_classes:
        for file_p in txt_files:
            base_name = os.path.splitext(os.path.basename(file_p))[0].lower()
            if base_name in {'data', 'dataset', 'labels', 'per_class_metrics', 'diagnostics', 'combined', 'notes', 'readme'}:
                continue
            try:
                with open(file_p, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if not line or HASH_HEADER_RE.match(line) or NC_LINE_RE.match(line):
                            continue
                        if line.lower().startswith(('names:', 'clasess:', 'classes:', 'deleted;')):
                            continue
                        if line.startswith('['):
                            # Raya format: iterate through all boxes in the line separated by ';'
                            for raw in line.split(';'):
                                raw = raw.strip()
                                if not raw.startswith('['):
                                    continue
                                try:
                                    sline = eval(raw)
                                    if isinstance(sline, list) and len(sline) > 0 and isinstance(sline[0], list):
                                        sline = sline[0]
                                    if isinstance(sline, list) and len(sline) > 0:
                                        val = sline[0]
                                        if isinstance(val, float) and val.is_integer():
                                            val = int(val)
                                        str_val = str(val)
                                        if str_val not in classes_found:
                                            classes_found.append(str_val)
                                except Exception:
                                    pass
                        else:
                            parts = line.split()
                            if len(parts) >= 5:
                                try:
                                    val = float(parts[0])
                                    if val.is_integer():
                                        val = int(val)
                                    str_val = str(val)
                                    if str_val not in classes_found:
                                        classes_found.append(str_val)
                                except ValueError:
                                    pass
            except Exception:
                pass
    else:
        classes_found = list(header_classes)

    # 2. Check JSON files (COCO categories or Raya json)
    for j_file in json_files:
        try:
            import json as _json
            with open(j_file, 'r', encoding='utf-8', errors='ignore') as f:
                jdata = _json.load(f)
                if isinstance(jdata, dict) and 'categories' in jdata:
                    for cat in jdata['categories']:
                        c_name = cat.get('name') if cat.get('name') is not None else cat.get('id')
                        if c_name is not None and str(c_name) not in classes_found:
                            classes_found.append(str(c_name))
        except Exception:
            pass

    def _sort_key(c):
        try:
            return (0, int(c))
        except ValueError:
            return (1, str(c).lower())

    return sorted(classes_found, key=_sort_key)


def scan_dataset_videos(gt_path):
    """
    Scan a directory for video sequence names (e.g. from txt, json, or video files).
    Returns a sorted list of unique video/sequence base names without extensions.
    """
    if not gt_path or not os.path.exists(gt_path):
        return []
    videos = []
    excluded_names = {
        'data', 'dataset', 'labels', 'classes', 'per_class_metrics', 
        'diagnostics', 'combined', 'notes', 'readme'
    }
    for root, _, files in os.walk(gt_path):
        for f in files:
            if f.endswith(('.txt', '.json', '.mp4', '.avi', '.mkv', '.mov', '.webm', '.MOV', '.MP4')):
                base = os.path.splitext(f)[0]
                if base.lower() not in excluded_names and not base.startswith(('per_class', 'diagnostics', 'combined')):
                    if base not in videos:
                        videos.append(base)
    return sorted(videos)
