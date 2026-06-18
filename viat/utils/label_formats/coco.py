"""COCO label-format plugin.

A single JSON file per split (often named ``_annotations.coco.json`` or
``annotations.json``). Boxes are ``[x, y, w, h]`` in absolute pixels, with a
top-level ``categories`` list and per-image ``annotations`` entries.
"""

import json
import os

from .base import LabelFormat, LabelParseError


class CocoLabelFormat(LabelFormat):
    name = "coco"
    extensions = (".json",)
    per_image = False  # one file per split

    # Shared-state: the loader pre-parses the COCO file and stores
    # image_filename -> boxes here. find_label_file returns the json path;
    # load() looks up the pre-indexed boxes.
    _index = None  # {filename: [box_dict, ...]}
    _file = None

    def reset(self):
        self._index = None
        self._file = None

    def discover(self, label_dirs):
        """Find and pre-parse the COCO json in any label dir."""
        self.reset()
        for d in label_dirs:
            for name in (
                "_annotations.coco.json",
                "annotations.json",
                "coco.json",
                "instances.json",
            ):
                cand = os.path.join(d, name)
                if os.path.isfile(cand):
                    self._parse(cand)
                    return cand
            # any *.coco.json or instances_*.json
            try:
                for f in os.listdir(d):
                    if f.endswith(".coco.json") or f.startswith("instances_"):
                        self._parse(os.path.join(d, f))
                        return os.path.join(d, f)
            except OSError:
                pass
        return None

    def _parse(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise LabelParseError(f"Cannot read COCO file {path}: {e}")

        cats = {c["id"]: c["name"] for c in data.get("categories", [])}
        idx = {}
        for img in data.get("images", []):
            idx[os.path.basename(img["file_name"])] = {
                "boxes": [],
                "width": img.get("width"),
                "height": img.get("height"),
            }
        for ann in data.get("annotations", []):
            img_id = ann["image_id"]
            # find the image filename
            fname = None
            for img in data.get("images", []):
                if img["id"] == img_id:
                    fname = os.path.basename(img["file_name"])
                    break
            if fname is None or fname not in idx:
                continue
            x, y, w, h = ann["bbox"]
            cat_id = ann.get("category_id")
            idx[fname]["boxes"].append(
                {
                    "class_name": cats.get(cat_id, f"class_{cat_id}"),
                    "class_index": cat_id,
                    "x": int(round(x)),
                    "y": int(round(y)),
                    "w": int(round(w)),
                    "h": int(round(h)),
                    "source": "manual",
                    "score": float(ann.get("score", 1.0)),
                }
            )
        self._index = idx
        self._file = path

    def find_label_file(self, image_path, label_dirs):
        # The shared file is discovered once; return it if known.
        if self._file:
            return self._file
        return self.discover(label_dirs)

    def load(self, label_path, image_size, classes):
        if self._index is None:
            self._parse(label_path)
        fname = os.path.basename(image_path)
        entry = self._index.get(fname)
        if entry is None:
            return []
        # COCO boxes are already in pixels; no de-normalization needed.
        return [dict(b) for b in entry["boxes"]]
