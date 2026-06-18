"""Base class for label-format plugins.

A *LabelFormat* knows how to:
  * locate the label file for a given image (``find_label_file``)
  * parse a label file into box dicts (``load``)
  * serialize box dicts back to file content (``dump``)

Adding a new format = subclass :class:`LabelFormat` and register it in
``__init__.py``. Nothing else in VIAT needs to change.
"""

import os


class LabelParseError(Exception):
    """Raised when a label file cannot be parsed."""


class LabelFormat:
    # Human-readable id ("yolo", "coco", ...)
    name = ""
    # File extensions this format uses (".txt", ".json", ".xml")
    extensions = ()
    # True = one label file per image (YOLO, Pascal VOC).
    # False = one big file for the whole dataset/split (COCO, CreateML).
    per_image = True

    def find_label_file(self, image_path, label_dirs):
        """Return the label file path for *image_path* or None.

        For per_image formats, default impl matches by stem in label_dirs.
        Dataset-wide formats override this to return the shared file.
        """
        if self.per_image:
            stem = os.path.splitext(os.path.basename(image_path))[0]
            for d in label_dirs:
                for ext in self.extensions:
                    cand = os.path.join(d, stem + ext)
                    if os.path.isfile(cand):
                        return cand
        return None

    def load(self, label_path, image_size, classes):
        """Parse *label_path* and return a list of box dicts.

        Each box dict has keys:
            class_name, class_index, x, y, w, h (pixels), source, score
        """
        raise NotImplementedError

    def dump(self, boxes, image_size, classes):
        """Serialize boxes to file content (str)."""
        raise NotImplementedError
