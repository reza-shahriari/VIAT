"""
YOLO label-format plugin.

Supports two line formats:

  * **Bounding box** (default): ``class_index cx cy w h`` (normalized)
  * **Segmentation polygon**: ``class_index x1 y1 x2 y2 x3 y3 ...``
    (normalized polygon points; any line with >6 values and an even count
    is treated as a polygon)

For polygon lines, the parser computes the axis-aligned bounding box from
the polygon extent and stores the polygon points in the ``segmentation``
field of the returned box dict. The canvas can optionally draw the polygon
outline when ``show_segmentation`` is enabled.

This is the default format for Roboflow YOLO exports and Ultralytics.
"""

import os

from .base import LabelFormat, LabelParseError


class YoloLabelFormat(LabelFormat):
    name = "yolo"
    extensions = (".txt",)
    # Per-image file (one label file per image). The loader matches by stem.
    per_image = True

    def find_label_file(self, image_path, label_dirs):
        """Return the label file path for *image_path* or None.

        Tries, in order:
          * <stem>.txt in each label_dir
          * <stem>.txt next to the image (single-folder layout)
        """
        stem = os.path.splitext(os.path.basename(image_path))[0]
        for d in label_dirs:
            cand = os.path.join(d, stem + ".txt")
            if os.path.isfile(cand):
                return cand
        # Same folder as image
        cand = os.path.join(os.path.dirname(image_path), stem + ".txt")
        if os.path.isfile(cand):
            return cand
        # Roboflow sometimes nests: images/train/x.jpg -> labels/train/x.txt
        return None

    def load(self, label_path, image_size, classes):
        """Parse a YOLO .txt file.

        Args:
            label_path: path to the .txt file
            image_size: (width, height) of the corresponding image
            classes: list of class names (index -> name)

        Returns:
            list of dicts, each with keys:
                class_name, class_index, x, y, w, h (pixels),
                source, score, segmentation (optional: list of (x,y) pixel
                tuples when the label was a polygon).
        """
        if image_size is None:
            raise LabelParseError("YOLO format requires image dimensions")
        img_w, img_h = image_size
        boxes = []
        if not os.path.isfile(label_path):
            return boxes

        with open(label_path, "r", encoding="utf-8", errors="replace") as f:
            for lineno, raw in enumerate(f, 1):
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue

                try:
                    cls_idx = int(float(parts[0]))
                except ValueError:
                    raise LabelParseError(
                        f"{label_path}:{lineno}: cannot parse class index {raw!r}"
                    )

                # --- Detect format: bbox (5-6 values) vs polygon (>6, even) ---
                n_coords = len(parts) - 1  # everything after class index
                is_polygon = n_coords >= 6 and n_coords % 2 == 0 and n_coords != 6
                # Note: exactly 6 values = cls + cx cy w h + score (bbox with conf)
                #       >=8 and even = polygon (cls + pairs of x,y)

                if is_polygon:
                    box_dict = self._parse_polygon(
                        parts[1:], cls_idx, img_w, img_h, classes
                    )
                else:
                    box_dict = self._parse_bbox(
                        parts, cls_idx, img_w, img_h, classes
                    )
                boxes.append(box_dict)
        return boxes

    def _parse_bbox(self, parts, cls_idx, img_w, img_h, classes):
        """Parse a bounding-box line: cls cx cy w h [score]."""
        try:
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            score = float(parts[5]) if len(parts) >= 6 else 1.0
        except (ValueError, IndexError):
            raise LabelParseError(f"cannot parse bbox line {parts!r}")

        abs_w = w * img_w
        abs_h = h * img_h
        abs_cx = cx * img_w
        abs_cy = cy * img_h
        x = abs_cx - abs_w / 2.0
        y = abs_cy - abs_h / 2.0

        class_name = (
            classes[cls_idx]
            if 0 <= cls_idx < len(classes)
            else f"class_{cls_idx}"
        )
        return {
            "class_name": class_name,
            "class_index": cls_idx,
            "x": int(round(x)),
            "y": int(round(y)),
            "w": int(round(abs_w)),
            "h": int(round(abs_h)),
            "source": "manual",
            "score": score,
            "segmentation": None,
        }

    def _parse_polygon(self, coord_parts, cls_idx, img_w, img_h, classes):
        """Parse a segmentation polygon: cls x1 y1 x2 y2 ...

        Computes the bounding box from the polygon extent.
        Stores the de-normalized polygon points in ``segmentation``.
        """
        try:
            coords = [float(v) for v in coord_parts]
        except ValueError:
            raise LabelParseError(f"cannot parse polygon coords {coord_parts!r}")

        # coords = [x1, y1, x2, y2, ...]
        xs = coords[0::2]
        ys = coords[1::2]
        if not xs or not ys:
            raise LabelParseError("empty polygon")

        # De-normalize
        px = [x * img_w for x in xs]
        py = [y * img_h for y in ys]

        min_x, max_x = min(px), max(px)
        min_y, max_y = min(py), max(py)
        w = max_x - min_x
        h = max_y - min_y

        class_name = (
            classes[cls_idx]
            if 0 <= cls_idx < len(classes)
            else f"class_{cls_idx}"
        )
        return {
            "class_name": class_name,
            "class_index": cls_idx,
            "x": int(round(min_x)),
            "y": int(round(min_y)),
            "w": int(round(w)),
            "h": int(round(h)),
            "source": "manual",
            "score": 1.0,
            "segmentation": list(zip(px, py)),  # list of (x,y) pixel tuples
        }

    def dump(self, boxes, image_size, classes):
        """Serialize boxes to YOLO text lines.

        If a box has ``segmentation`` (a list of (x,y) pixel tuples), it is
        dumped as a polygon line; otherwise as a bbox line.

        classes: list whose index matches class_index.
        """
        img_w, img_h = image_size
        name_to_idx = {n: i for i, n in enumerate(classes)}
        lines = []
        for b in boxes:
            cls_idx = b.get("class_index")
            if cls_idx is None:
                cls_idx = name_to_idx.get(b["class_name"])
            if cls_idx is None:
                continue  # skip unknown class

            seg = b.get("segmentation")
            if seg and len(seg) >= 3:
                # Polygon
                coord_str = " ".join(
                    f"{x / img_w:.6f} {y / img_h:.6f}" for (x, y) in seg
                )
                lines.append(f"{cls_idx} {coord_str}")
            else:
                # Bbox
                cx = (b["x"] + b["w"] / 2.0) / img_w
                cy = (b["y"] + b["h"] / 2.0) / img_h
                w = b["w"] / img_w
                h = b["h"] / img_h
                line = f"{cls_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                if b.get("score") is not None and b.get("score") != 1.0:
                    line += f" {b['score']:.6f}"
                lines.append(line)
        return "\n".join(lines) + ("\n" if lines else "")
