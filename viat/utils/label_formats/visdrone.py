"""
VisDrone label-format plugin for VIAT.

VisDrone annotation format:
    <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>

Fields:
    - bbox_left, bbox_top, bbox_width, bbox_height: pixel coordinates (top-left, width, height)
    - score: confidence score (1 for ground truth, [0, 1] for predictions)
    - object_category:
        0: ignored regions
        1: pedestrian
        2: person
        3: bicycle
        4: car
        5: van
        6: truck
        7: tricycle
        8: awning-tricycle
        9: bus
        10: motor
        11: others
    - truncation: 0 (no truncation), 1 (partial 1%~50%), 2 (heavy >50%)
    - occlusion: 0 (no occlusion), 1 (partial 1%~50%), 2 (heavy >50%)
"""

import os
from .base import LabelFormat, LabelParseError

VISDRONE_CLASSES = [
    "ignored_region",   # 0
    "pedestrian",       # 1
    "person",           # 2
    "bicycle",          # 3
    "car",              # 4
    "van",              # 5
    "truck",            # 6
    "tricycle",         # 7
    "awning-tricycle",  # 8
    "bus",              # 9
    "motor",            # 10
    "others",           # 11
]

# Standard 10 target classes (excluding ignored_region and others)
VISDRONE_TARGET_CLASSES = [
    "pedestrian",
    "person",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


class VisDroneLabelFormat(LabelFormat):
    name = "visdrone"
    extensions = (".txt",)
    per_image = True

    def find_label_file(self, image_path, label_dirs):
        """Return the label file path for image_path or None.

        Tries matching by stem in:
          - provided label_dirs (e.g. annotations/ or labels/)
          - sibling 'annotations' folder next to 'images'
          - same folder as image
        """
        stem = os.path.splitext(os.path.basename(image_path))[0]
        for d in label_dirs:
            for ext in self.extensions:
                cand = os.path.join(d, stem + ext)
                if os.path.isfile(cand):
                    return cand

        # Check sibling 'annotations' directory if image is in 'images'
        img_dir = os.path.dirname(image_path)
        parent_dir = os.path.dirname(img_dir)
        ann_sibling = os.path.join(parent_dir, "annotations")
        if os.path.isdir(ann_sibling):
            cand = os.path.join(ann_sibling, stem + ".txt")
            if os.path.isfile(cand):
                return cand

        # Same folder as image
        cand = os.path.join(img_dir, stem + ".txt")
        if os.path.isfile(cand):
            return cand

        return None

    def load(self, label_path, image_size, classes=None):
        """Parse a VisDrone .txt annotation file.

        Args:
            label_path: path to the .txt file
            image_size: (width, height) of the image
            classes: list of class names (optional)

        Returns:
            list of dicts, each with keys:
                class_name, class_index, x, y, w, h (pixels),
                source, score, attributes, segmentation
        """
        boxes = []
        if not os.path.isfile(label_path):
            return boxes

        class_map = classes if classes else VISDRONE_CLASSES

        with open(label_path, "r", encoding="utf-8", errors="replace") as f:
            for lineno, raw in enumerate(f, 1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue

                # VisDrone annotations are comma-separated, occasionally space-separated
                if "," in line:
                    parts = [p.strip() for p in line.split(",")]
                else:
                    parts = line.split()

                if len(parts) < 8:
                    continue

                try:
                    x = int(round(float(parts[0])))
                    y = int(round(float(parts[1])))
                    w = int(round(float(parts[2])))
                    h = int(round(float(parts[3])))
                    score = float(parts[4])
                    cat_id = int(float(parts[5]))
                    truncation = int(float(parts[6]))
                    occlusion = int(float(parts[7]))
                except (ValueError, IndexError):
                    raise LabelParseError(
                        f"{label_path}:{lineno}: invalid VisDrone format in line {raw!r}"
                    )

                # Clamp/validate bbox
                if w <= 0 or h <= 0:
                    continue

                # Resolve class name
                if 0 <= cat_id < len(class_map):
                    class_name = class_map[cat_id]
                elif 0 <= cat_id < len(VISDRONE_CLASSES):
                    class_name = VISDRONE_CLASSES[cat_id]
                else:
                    class_name = f"category_{cat_id}"

                attributes = {
                    "truncation": truncation,
                    "occlusion": occlusion,
                    "score": score,
                    "category_id": cat_id,
                }

                boxes.append({
                    "class_name": class_name,
                    "class_index": cat_id,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "source": "manual",
                    "score": score,
                    "attributes": attributes,
                    "segmentation": None,
                })

        return boxes

    def dump(self, boxes, image_size, classes=None):
        """Serialize boxes back to VisDrone format."""
        class_map = classes if classes else VISDRONE_CLASSES
        lines = []
        for b in boxes:
            x = int(round(b.get("x", 0)))
            y = int(round(b.get("y", 0)))
            w = int(round(b.get("w", 0)))
            h = int(round(b.get("h", 0)))
            score = float(b.get("score", 1.0) or 1.0)
            score_val = int(score) if score == int(score) else round(score, 3)

            attrs = b.get("attributes", {}) or {}
            truncation = int(attrs.get("truncation", 0))
            occlusion = int(attrs.get("occlusion", 0))

            cls_name = b.get("class_name", "")
            if "category_id" in attrs:
                cat_id = int(attrs["category_id"])
            elif cls_name in class_map:
                cat_id = class_map.index(cls_name)
            elif cls_name in VISDRONE_CLASSES:
                cat_id = VISDRONE_CLASSES.index(cls_name)
            else:
                cat_id = int(b.get("class_index", 0))

            lines.append(f"{x},{y},{w},{h},{score_val},{cat_id},{truncation},{occlusion}\n")

        return "".join(lines)
