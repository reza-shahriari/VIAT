import os
import glob
import json
import xml.etree.ElementTree as ET

class FormatAdapter:
    """
    Adapter to convert various dataset/annotation formats (YOLO, COCO JSON, Pascal VOC)
    into the unified evaluation text format expected by evaluate_everything.
    """

    @staticmethod
    def detect_format(directory):
        """
        Detect format of annotations in a directory.
        Returns: 'custom', 'yolo', 'coco', 'xml', or 'unknown'
        """
        if not directory or not os.path.exists(directory):
            return 'unknown'

        files = os.listdir(directory)
        json_files = [f for f in files if f.endswith('.json') and not f.endswith('coco_ann.json') and not f.endswith('all_video.json')]
        xml_files = [f for f in files if f.endswith('.xml')]
        txt_files = [f for f in files if f.endswith('.txt')]

        if json_files:
            return 'coco'
        if xml_files:
            return 'xml'
        
        if txt_files:
            # Check content of first txt file
            sample_file = os.path.join(directory, txt_files[0])
            try:
                with open(sample_file, 'r') as f:
                    lines = [l.strip() for l in f if l.strip()]
                    if lines:
                        first = lines[0]
                        if first.startswith('[') or ';' in first or ('[' in first and ']' in first):
                            return 'custom'
                        parts = first.split()
                        if len(parts) in [5, 6]:
                            try:
                                floats = [float(p) for p in parts[1:5]]
                                if all(0.0 <= f <= 1.0 for f in floats):
                                    return 'yolo'
                            except ValueError:
                                pass
            except Exception:
                pass
            return 'custom'
        return 'unknown'

    @staticmethod
    def normalize_yolo_to_custom(yolo_file, img_w=1920, img_h=1080, is_gt=True):
        """Convert a YOLO format file to custom format list of strings."""
        result_lines = []
        with open(yolo_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(parts[0])
                xc, yc, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                
                # Convert normalized YOLO (cx, cy, w, h) to absolute (x1, y1, w, h) or (x1, y1, x2, y2)
                abs_w = int(w * img_w)
                abs_h = int(h * img_h)
                abs_x1 = int((xc - w / 2) * img_w)
                abs_y1 = int((yc - h / 2) * img_h)

                if is_gt:
                    # sline: [class_id, x1, y1, w, h, size_thr, quality_thr, difficult]
                    sline = f"[{cls_id}, {abs_x1}, {abs_y1}, {abs_w}, {abs_h}, 100, 100, 0]"
                else:
                    conf = float(parts[5]) if len(parts) >= 6 else 1.0
                    abs_x2 = abs_x1 + abs_w
                    abs_y2 = abs_y1 + abs_h
                    # sline: [0, x1, y1, x2, y2, conf]
                    sline = f"[{cls_id}, {abs_x1}, {abs_y1}, {abs_x2}, {abs_y2}, {conf}]"
                result_lines.append(sline)
        return result_lines
