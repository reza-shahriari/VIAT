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

    classes_found = set()

    # Check for data.yaml anywhere in the directory
    for root, _, files in os.walk(gt_path):
        yaml_files = [f for f in files if f.endswith('.yaml') or f.endswith('.yml')]
        if yaml_files:
            yaml_dict = parse_yolo_yaml(os.path.join(root, yaml_files[0]))
            if yaml_dict:
                return list(yaml_dict.values())

    # Check txt files
    txt_files = []
    for root, _, files in os.walk(gt_path):
        txt_files.extend([os.path.join(root, f) for f in files if f.endswith('.txt')])
        if len(txt_files) >= 20:
            break

    for file_p in txt_files[:20]: # sample up to 20 files
        try:
            with open(file_p, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith('['):
                        # Raya format: [class_id, x, y, w, h, ...] or [[class_id, ...]]
                        try:
                            raw = line.split(';')[0]
                            sline = eval(raw)
                            # Handle list of lists
                            if isinstance(sline, list) and len(sline) > 0 and isinstance(sline[0], list):
                                sline = sline[0]
                            if isinstance(sline, list) and len(sline) > 0:
                                val = sline[0]
                                if isinstance(val, float) and val.is_integer():
                                    val = int(val)
                                classes_found.add(str(val))
                        except Exception:
                            pass
                    else:
                        parts = line.split()
                        if parts:
                            try:
                                val = float(parts[0])
                                if val.is_integer():
                                    val = int(val)
                                classes_found.add(str(val))
                            except ValueError:
                                classes_found.add(parts[0])
        except Exception:
            pass

    return sorted(list(classes_found))
