"""Pascal VOC XML label-format plugin.

One ``.xml`` file per image, with ``<object><name>`` and ``<bndbox>``.
"""

import os
import xml.etree.ElementTree as ET

from .base import LabelFormat, LabelParseError


class PascalVocLabelFormat(LabelFormat):
    name = "pascal_voc"
    extensions = (".xml",)
    per_image = True

    def find_label_file(self, image_path, label_dirs):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        for d in label_dirs:
            cand = os.path.join(d, stem + ".xml")
            if os.path.isfile(cand):
                return cand
        cand = os.path.join(os.path.dirname(image_path), stem + ".xml")
        if os.path.isfile(cand):
            return cand
        return None

    def load(self, label_path, image_size, classes):
        boxes = []
        if not os.path.isfile(label_path):
            return boxes
        try:
            tree = ET.parse(label_path)
        except ET.ParseError as e:
            raise LabelParseError(f"{label_path}: {e}")
        root = tree.getroot()
        # Build a name->index map for class_index lookup
        name_to_idx = {n: i for i, n in enumerate(classes)}
        for obj in root.findall("object"):
            name_el = obj.find("name")
            cls_name = name_el.text.strip() if name_el is not None and name_el.text else "unknown"
            bnd = obj.find("bndbox")
            if bnd is None:
                continue
            try:
                xmin = float(bnd.findtext("xmin", "0"))
                ymin = float(bnd.findtext("ymin", "0"))
                xmax = float(bnd.findtext("xmax", "0"))
                ymax = float(bnd.findtext("ymax", "0"))
            except (TypeError, ValueError):
                continue
            boxes.append(
                {
                    "class_name": cls_name,
                    "class_index": name_to_idx.get(cls_name),
                    "x": int(round(xmin)),
                    "y": int(round(ymin)),
                    "w": int(round(xmax - xmin)),
                    "h": int(round(ymax - ymin)),
                    "source": "manual",
                    "score": 1.0,
                }
            )
        return boxes
