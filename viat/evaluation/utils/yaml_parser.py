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


def scan_dataset_classes(gt_path):
    """
    Scan a directory for annotated class IDs or names.
    Supports YOLO txt, Raya txt, COCO json, and Pascal VOC XML.
    """
    if not gt_path or not os.path.exists(gt_path):
        return []

    classes_found = []

    # 1. Check txt files
    txt_files = []
    json_files = []
    for root, _, files in os.walk(gt_path):
        for f in files:
            if f.endswith('.txt'):
                txt_files.append(os.path.join(root, f))
            elif f.endswith('.json'):
                json_files.append(os.path.join(root, f))

    for file_p in txt_files:
        try:
            with open(file_p, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
                # First check for inline header block like video_to_yolo.py uses
                in_names_block = False
                header_names = []
                has_header = False
                for line in lines:
                    sline = line.strip()
                    if not sline: continue
                    if not in_names_block:
                        if sline.lower().startswith("names:"):
                            in_names_block = True
                            has_header = True
                        continue
                    if re.match(r"^-?\s*nc\s*:\s*\d+", sline, re.IGNORECASE):
                        break
                    # Parse bullet points
                    name = re.sub(r"^[\-\*\u2022]\s*", "", sline).strip()
                    if name:
                        header_names.append(name)
                        
                if has_header and header_names:
                    for name in header_names:
                        if name not in classes_found:
                            classes_found.append(name)
                    continue
                    
                # Fallback to scanning bounding boxes
                for line in lines:
                    line = line.strip()
                    if not line:
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
                        if parts:
                            try:
                                val = float(parts[0])
                                if val.is_integer():
                                    val = int(val)
                                str_val = str(val)
                                if str_val not in classes_found:
                                    classes_found.append(str_val)
                            except ValueError:
                                str_val = parts[0]
                                if str_val not in classes_found:
                                    classes_found.append(str_val)
        except Exception:
            pass

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
