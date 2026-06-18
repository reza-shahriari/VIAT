"""CreateML JSON label-format plugin.

One ``.json`` file per image (or one shared file), each entry:
    [{"image": "x.jpg", "annotations": [
        {"label": "car", "coordinates": {"x":..,"y":..,"width":..,"height":..}}
    ]}]
Coordinates are center-based, in pixels.
"""

import json
import os

from .base import LabelFormat, LabelParseError


class CreateMlLabelFormat(LabelFormat):
    name = "createml"
    extensions = (".json",)
    per_image = True

    _shared_index = None

    def reset(self):
        self._shared_index = None

    def _maybe_load_shared(self, label_dirs):
        # Some exports use one shared _annotations.json
        if self._shared_index is not None:
            return
        for d in label_dirs:
            for name in ("_annotations.json", "annotations.json", "createml.json"):
                cand = os.path.join(d, name)
                if os.path.isfile(cand):
                    try:
                        with open(cand, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        idx = {}
                        if isinstance(data, list):
                            for entry in data:
                                fname = entry.get("image")
                                if fname:
                                    idx[os.path.basename(fname)] = entry.get("annotations", [])
                        self._shared_index = idx
                        return
                    except (OSError, json.JSONDecodeError):
                        pass

    def find_label_file(self, image_path, label_dirs):
        stem = os.path.splitext(os.path.basename(image_path))[0]
        for d in label_dirs:
            cand = os.path.join(d, stem + ".json")
            if os.path.isfile(cand):
                return cand
        # fall back to shared file
        self._maybe_load_shared(label_dirs)
        if self._shared_index is not None and os.path.basename(image_path) in self._shared_index:
            return "shared"
        return None

    def load(self, label_path, image_size, classes):
        boxes = []
        name_to_idx = {n: i for i, n in enumerate(classes)}

        def parse_anns(anns):
            out = []
            for a in anns:
                lbl = a.get("label", "unknown")
                c = a.get("coordinates", {})
                cx = float(c.get("x", 0))
                cy = float(c.get("y", 0))
                w = float(c.get("width", 0))
                h = float(c.get("height", 0))
                out.append(
                    {
                        "class_name": lbl,
                        "class_index": name_to_idx.get(lbl),
                        "x": int(round(cx - w / 2.0)),
                        "y": int(round(cy - h / 2.0)),
                        "w": int(round(w)),
                        "h": int(round(h)),
                        "source": "manual",
                        "score": float(a.get("score", 1.0)),
                    }
                )
            return out

        if label_path == "shared":
            return parse_anns(self._shared_index.get("", []))

        if not os.path.isfile(label_path):
            return boxes
        try:
            with open(label_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            raise LabelParseError(f"{label_path}: {e}")
        if isinstance(data, list):
            # could be the list for THIS image, or the shared list
            if data and isinstance(data[0], dict) and "image" in data[0]:
                # shared-style; find our image
                fname = None
                for entry in data:
                    # we can't know filename here reliably; return empty
                    pass
                return []
            return parse_anns(data)
        if isinstance(data, dict) and "annotations" in data:
            return parse_anns(data["annotations"])
        return boxes
