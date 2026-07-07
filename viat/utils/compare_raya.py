import os

def parse_raya_text(content):
    classes = []
    frames = {}
    deleted_frames = set()

    try:
        lines = content.splitlines()
        
        in_header = False
        in_names = False
        data_lines = []
        
        for line in lines:
            line = line.strip()
            if line == "###":
                in_header = not in_header
                continue
            if in_header:
                if line.startswith("names:"):
                    in_names = True
                    continue
                elif line.startswith("-nc:"):
                    in_names = False
                    continue
                elif line.startswith("- ") and in_names:
                    classes.append(line[2:].strip())
                continue
            data_lines.append(line)
            
        for frame_num, line in enumerate(data_lines):
            if line == "DELETE;":
                deleted_frames.add(frame_num)
                continue
            
            if not line or line == "[]":
                frames[frame_num] = []
                continue
                
            if not ("[" in line and "]" in line):
                continue
                
            objs = []
            annotation_strs = line.split(";")
            for ann_str in annotation_strs:
                ann_str = ann_str.strip()
                if not ann_str or ann_str == "[]":
                    continue
                    
                ann_str = ann_str.replace("[", "").replace("]", "")
                parts = ann_str.split(",")
                if len(parts) >= 5:
                    class_id = int(float(parts[0]))
                    x = float(parts[1])
                    y = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])
                    
                    class_name = classes[class_id] if class_id < len(classes) else f"Class {class_id}"
                    
                    attributes = []
                    for p in parts[5:]:
                        attributes.append(float(p))
                        
                    objs.append({
                        "class_id": class_id,
                        "class_name": class_name,
                        "box": [x, y, w, h],
                        "attributes": attributes
                    })
            frames[frame_num] = objs
            
    except Exception as e:
        print(f"Error parsing text: {e}")
        
    return classes, deleted_frames, frames

def parse_raya_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    return parse_raya_text(content)


def bb_intersection_over_union(boxA, boxB):
    # box format: [x, y, w, h]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    iou = interArea / float(boxAArea + boxBArea - interArea) if (boxAArea + boxBArea - interArea) > 0 else 0
    return iou



def generate_comparison_report_string(base_classes, base_deleted, base_frames, mod_classes, mod_deleted, mod_frames, base_filename="Base", mod_filename="Modified"):
    # 1. Removed frames and removed labels
    removed_frames_count = len(mod_deleted - base_deleted)
    
    # Pre-compute robust class mapping using IoU
    # mapping_votes[base_class_name][mod_class_name] = count
    mapping_votes = {}
    for frame_num in set(base_frames.keys()) & set(mod_frames.keys()):
        if frame_num in mod_deleted:
            continue
        base_objs = base_frames[frame_num]
        mod_objs = mod_frames.get(frame_num, [])
        for b_obj in base_objs:
            best_iou = 0
            best_m = None
            for m_obj in mod_objs:
                iou = bb_intersection_over_union(b_obj["box"], m_obj["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_m = m_obj
            if best_iou > 0.5 and best_m:
                bc = b_obj["class_name"]
                mc = best_m["class_name"]
                if bc not in mapping_votes:
                    mapping_votes[bc] = {}
                mapping_votes[bc][mc] = mapping_votes[bc].get(mc, 0) + 1

    # Decide class mapping: if > 80% of matches for a base class go to a specific mod class, map it.
    class_name_map = {}
    for bc, votes in mapping_votes.items():
        total_votes = sum(votes.values())
        for mc, count in votes.items():
            if count / total_votes > 0.8:
                class_name_map[bc] = mc
                break

    # Stats counters
    removed_labels_count = 0
    added_labels_count = 0
    bbox_changes_count = 0
    class_changes_count = 0
    
    # Attribute changes by index
    # Format: { 0: 5, 1: 10 }
    attr_changes = {}
    added_labels_with_attrs = 0
    added_labels_default = 0
    
    # Match objects per frame
    for frame_num in set(base_frames.keys()) | set(mod_frames.keys()):
        if frame_num in mod_deleted:
            continue # If whole frame deleted, don't count individual removed labels
            
        base_objs = base_frames.get(frame_num, [])
        mod_objs = mod_frames.get(frame_num, [])
        
        # To avoid matching one mod object to multiple base objects
        matched_mod_indices = set()
        
        for b_obj in base_objs:
            best_iou = 0
            best_m_idx = -1
            
            for idx, m_obj in enumerate(mod_objs):
                if idx in matched_mod_indices:
                    continue
                iou = bb_intersection_over_union(b_obj["box"], m_obj["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_m_idx = idx
                    
            if best_iou > 0.3: # Threshold to be considered the same object
                matched_mod_indices.add(best_m_idx)
                m_obj = mod_objs[best_m_idx]
                
                # Check BBox change > 15% (IoU < 0.85)
                if best_iou < 0.85:
                    bbox_changes_count += 1
                    
                # Check class change (accounting for global renames)
                mapped_bc = class_name_map.get(b_obj["class_name"], b_obj["class_name"])
                if mapped_bc != m_obj["class_name"]:
                    class_changes_count += 1
                    
                # Check attribute changes
                b_attrs = b_obj["attributes"]
                m_attrs = m_obj["attributes"]
                
                attr_names = ["Size", "Quality", "Difficult"]
                if isinstance(b_attrs, list):
                    b_attrs = {
                        attr_names[i] if i < len(attr_names) else f"Attr {i+1}": val
                        for i, val in enumerate(b_attrs)
                    }
                if isinstance(m_attrs, list):
                    m_attrs = {
                        attr_names[i] if i < len(attr_names) else f"Attr {i+1}": val
                        for i, val in enumerate(m_attrs)
                    }
                    
                all_keys = set(b_attrs.keys()) | set(m_attrs.keys())
                for k in all_keys:
                    b_val = b_attrs.get(k, -1)
                    m_val = m_attrs.get(k, -1)
                    if b_val != m_val:
                        attr_changes[k] = attr_changes.get(k, 0) + 1
            else:
                # No match found for this base object -> removed
                removed_labels_count += 1
                
        # Unmatched modified objects -> added labels
        for idx, m_obj in enumerate(mod_objs):
            if idx not in matched_mod_indices:
                added_labels_count += 1
                # Check if attributes are defaults (-1 or 0)
                m_attrs = m_obj["attributes"]
                if isinstance(m_attrs, list):
                    vals = m_attrs
                else:
                    vals = m_attrs.values()
                    
                has_custom = False
                for val in vals:
                    if val not in [-1, 0, -1.0, 0.0]:
                        has_custom = True
                        break
                if has_custom:
                    added_labels_with_attrs += 1
                else:
                    added_labels_default += 1

    # Output to markdown
    report = f"## Summary of Changes (Compared to Base)\n\n"
    report += "| Metric | Count |\n"
    report += "|--------|-------|\n"
    report += f"| Removed Frames (`DELETE;`) | {removed_frames_count} |\n"
    report += f"| Added Labels (from scratch) | {added_labels_count} |\n"
    report += f"| &nbsp;&nbsp;&nbsp;*With custom attributes* | {added_labels_with_attrs} |\n"
    report += f"| &nbsp;&nbsp;&nbsp;*With default attributes (-1 or 0)* | {added_labels_default} |\n"
    report += f"| Removed Labels | {removed_labels_count} |\n"
    report += f"| Bounding Box Changes (>15% / IoU<0.85) | {bbox_changes_count} |\n"
    report += f"| Individual Class Changes | {class_changes_count} |\n\n"
    
    report += "## Attribute Changes\n\n"
    if attr_changes:
        report += "| Attribute | Number of Changes |\n"
        report += "|-----------|-------------------|\n"
        for k in sorted(attr_changes.keys()):
            report += f"| {k} | {attr_changes[k]} |\n"
    else:
        report += "No attribute changes detected.\n\n"
        
    report += "\n## Global Class Modifications\n\n"
    report += "| Base Class | Modified Class (Inferred Rename) |\n"
    report += "|------------|----------------------------------|\n"
    rename_found = False
    for bc, mc in class_name_map.items():
        if bc != mc:
            report += f"| {bc} | {mc} |\n"
            rename_found = True
    if not rename_found:
        report += "| (None) | (None) |\n"
        
    report += "\n### New Classes Created in Modified File\n"
    new_classes = set(mod_classes) - set(base_classes) - set(class_name_map.values())
    if new_classes:
        for nc in new_classes:
            report += f"- {nc}\n"
    else:
        report += "- None\n"
        
    return report

def compare_annotations(base_file, mod_file, output_md):
    base_classes, base_deleted, base_frames = parse_raya_file(base_file)
    mod_classes, mod_deleted, mod_frames = parse_raya_file(mod_file)
    
    report_body = generate_comparison_report_string(
        base_classes, base_deleted, base_frames, 
        mod_classes, mod_deleted, mod_frames,
        os.path.basename(base_file), os.path.basename(mod_file)
    )
    
    report = f"# Annotation Comparison Report\n\n"
    report += f"**Base File:** `{os.path.basename(base_file)}`\n"
    report += f"**Modified File:** `{os.path.basename(mod_file)}`\n\n"
    report += report_body

    try:
        with open(output_md, "w") as f:
            f.write(report)
        return True, "Comparison completed successfully."
    except Exception as e:
        return False, str(e)
