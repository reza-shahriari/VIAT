"""
VIAT custom JSON label-format plugin.

A single JSON file per video, keyed by zero-padded frame number string:

    {
      "0000": {
        "actors": {
          "SK_HelicopterAdjustedRigged": {
            "class": "HeliCopostpter",
            "pixels": 48477,
            "occupancy": 0.023,
            "projection_visibility": 0.893,
            "accepted": true,
            "truncated": false,
            "occlusion_ratio": 0.892,
            "bbox": [605, 503, 524, 508],
            "segmentation": [[1095, 503, 992, 552, ...]]
          }
        }
      },
      "0001": { ... }
    }

Mapping to VIAT BoundingBox:
  * ``class``            -> class_name
  * ``bbox`` [x,y,w,h]   -> rect (absolute pixels)
  * ``segmentation``     -> segmentation (list of (x,y) pixel tuples; first
                            polygon only -- multi-polygon could be added later)
  * ``accepted``         -> verified; if True: Size=100, Quality=100;
                           if False: verified=False, Size=-1, Quality=-1
  * actor ID (key)       -> attributes["actor_id"]
  * ``truncated``        -> attributes["truncated"]
  * ``occlusion_ratio``  -> attributes["occlusion_ratio"]
  * ``pixels``           -> attributes["pixels"]
  * ``occupancy``        -> attributes["occupancy"]
  * ``projection_visibility`` -> attributes["projection_visibility"]
"""

import json
import os

from .base import LabelFormat, LabelParseError


class ViatJsonLabelFormat(LabelFormat):
    name = "viat_json"
    extensions = (".json",)
    # One file for the whole video (not per-image).
    per_image = False

    # Parsed index: {frame_int: [box_dict, ...]}
    _index = None
    _file = None

    def reset(self):
        self._index = None
        self._file = None

    def discover(self, label_dirs):
        """Find and pre-parse the VIAT JSON file in any label dir.

        Returns the file path, or None.
        """
        self.reset()
        candidates = []
        for d in label_dirs:
            if not os.path.isdir(d):
                continue
            for f in os.listdir(d):
                if not f.lower().endswith(".json"):
                    continue
                p = os.path.join(d, f)
                # Heuristic: it's a VIAT JSON if the top-level keys look like
                # frame numbers (zero-padded ints) and values have "actors".
                try:
                    with open(p, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    if isinstance(data, dict):
                        # check first key
                        if data:
                            first_key = next(iter(data))
                            first_val = data[first_key]
                            if (
                                isinstance(first_val, dict)
                                and "actors" in first_val
                            ):
                                candidates.append((p, data))
                except (OSError, json.JSONDecodeError):
                    continue

        if not candidates:
            return None

        # Pick the largest file (most likely the full annotation set)
        candidates.sort(key=lambda x: os.path.getsize(x[0]), reverse=True)
        path, data = candidates[0]
        self._parse(path, data)
        return path

    def _parse(self, path, data=None):
        if data is None:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError) as e:
                raise LabelParseError(f"Cannot read VIAT JSON {path}: {e}")

        index = {}
        for frame_key, frame_data in data.items():
            try:
                frame_num = int(frame_key)
            except (ValueError, TypeError):
                # Non-numeric key -- skip or treat as 0-indexed position
                continue

            actors = frame_data.get("actors", {}) if isinstance(frame_data, dict) else {}
            boxes = []
            for actor_id, actor_data in actors.items():
                if not isinstance(actor_data, dict):
                    continue
                class_name = actor_data.get("class", "unknown")
                bbox = actor_data.get("bbox", [0, 0, 0, 0])
                if len(bbox) != 4:
                    continue
                x, y, w, h = bbox

                accepted = bool(actor_data.get("accepted", False))

                # Segmentation: list of polygons, each a flat [x1,y1,x2,y2,...]
                seg = None
                segmentation = actor_data.get("segmentation")
                if segmentation and isinstance(segmentation, list):
                    # Take the first polygon
                    poly = segmentation[0] if isinstance(segmentation[0], list) else segmentation
                    if isinstance(poly, list) and len(poly) >= 6:
                        pairs = [(float(poly[i]), float(poly[i + 1])) for i in range(0, len(poly) - 1, 2)]
                        if len(pairs) >= 3:
                            seg = pairs

                # Minimal attributes: only actor_id (needed for object visibility mode).
                # The 'accepted' field maps directly to 'verified' -- no Size/Quality
                # or other metadata is stored, per user request.
                attributes = {
                    "actor_id": actor_id,
                }

                boxes.append({
                    "class_name": class_name,
                    "class_index": None,
                    "x": int(round(x)),
                    "y": int(round(y)),
                    "w": int(round(max(1, w))),
                    "h": int(round(max(1, h))),
                    "source": "manual",
                    "score": 1.0,
                    "segmentation": seg,
                    "attributes": attributes,
                    "verified": accepted,
                    "actor_id": actor_id,
                })
            index[frame_num] = boxes

        self._index = index
        self._file = path

    def find_label_file(self, image_path, label_dirs):
        if self._file:
            return self._file
        return self.discover(label_dirs)

    def load(self, label_path, image_size, classes):
        """Load boxes for a single frame.

        For VIAT JSON, *image_size* is not used (bbox is already in pixels),
        and *label_path* should be the JSON file path. However, since the
        main loader calls load() per-frame, we return the boxes for the
        frame number encoded in image_path's stem (if numeric) or frame 0.

        Actually, the VIAT JSON is a whole-video file. The caller
        (load_dataset_into_app) should use load_all_frames() instead.
        This per-frame load() returns boxes for the frame number parsed
        from the image filename, or empty if not found.
        """
        if self._index is None:
            self._parse(label_path)

        # Try to extract a frame number from the image path
        stem = os.path.splitext(os.path.basename(image_path))[0]
        frame_num = None
        # zero-padded number (e.g. "0000", "0001")
        if stem.isdigit():
            frame_num = int(stem)
        else:
            # try trailing number
            import re
            m = re.search(r"(\d+)$", stem)
            if m:
                frame_num = int(m.group(1))

        if frame_num is None:
            return []

        boxes = self._index.get(frame_num, [])
        # Return copies with attributes
        return [dict(b) for b in boxes]

    def load_all_frames(self):
        """Return the full {frame_num: [box_dict, ...]} index.

        Use this for video annotations (not image datasets).
        """
        if self._index is None:
            return {}
        return self._index

    def dump(self, boxes_by_frame, image_size, classes):
        """Serialize {frame_num: [box_dict, ...]} back to VIAT JSON.

        boxes_by_frame: dict {frame_int: [box_dict, ...]}

        Output format is minimal: class, bbox, accepted, segmentation.
        No extra attributes (truncated, occlusion_ratio, etc.) are written,
        per user request.
        """
        out = {}
        for frame_num in sorted(boxes_by_frame.keys()):
            frame_key = str(frame_num).zfill(4)
            actors = {}
            for b in boxes_by_frame[frame_num]:
                actor_id = b.get("actor_id") or b.get("attributes", {}).get("actor_id") or f"actor_{frame_num}"
                accepted = b.get("verified", False)
                actor = {
                    "class": b["class_name"],
                    "bbox": [b["x"], b["y"], b["w"], b["h"]],
                    "accepted": bool(accepted),
                }
                if b.get("segmentation"):
                    actor["segmentation"] = [[int(x) for x in sum(([p[0], p[1]] for p in b["segmentation"]), [])]]
                actors[actor_id] = actor
            out[frame_key] = {"actors": actors}
        return json.dumps(out, indent=2)
