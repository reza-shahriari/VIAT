"""Label-format plugin registry for VIAT.

To add a new format:
  1. Create ``your_format.py`` with a subclass of :class:`LabelFormat`.
  2. Register it in :data:`FORMATS` below.
  3. (optional) add it to :data:`PRIORITY` to control detection order.

Nothing else in VIAT needs to change -- :mod:`utils.dataset_manager` iterates
registered formats to detect + parse labels.
"""

from .base import LabelFormat, LabelParseError
from .yolo import YoloLabelFormat
from .coco import CocoLabelFormat
from .pascal_voc import PascalVocLabelFormat
from .createml import CreateMlLabelFormat
from .viat_json import ViatJsonLabelFormat

# Registry: name -> class. Order here is the default detection priority.
FORMATS = {
    YoloLabelFormat.name: YoloLabelFormat,
    PascalVocLabelFormat.name: PascalVocLabelFormat,
    CocoLabelFormat.name: CocoLabelFormat,
    CreateMlLabelFormat.name: CreateMlLabelFormat,
    ViatJsonLabelFormat.name: ViatJsonLabelFormat,
}

# Detection priority: when scanning a folder for the format, try these first.
# YOLO is the Roboflow default, so it gets top priority.
# viat_json is last (it's for video, not image datasets).
PRIORITY = ["yolo", "pascal_voc", "coco", "createml", "viat_json"]


def get_format(name):
    """Return a fresh instance of the named format, or None."""
    cls = FORMATS.get(name)
    return cls() if cls else None


def all_formats():
    """Yield (name, instance) for every registered format."""
    for name in PRIORITY:
        cls = FORMATS.get(name)
        if cls:
            yield name, cls()


__all__ = [
    "LabelFormat",
    "LabelParseError",
    "FORMATS",
    "PRIORITY",
    "get_format",
    "all_formats",
]
