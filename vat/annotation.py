from PyQt5.QtCore import QRect, QPoint
from PyQt5.QtGui import QColor
import uuid

from PyQt5.QtCore import QRect
from PyQt5.QtGui import QColor


class BoundingBox:
    def __init__(self, rect, class_name="default", attributes=None, color=None):
        self.rect = rect
        self.class_name = class_name
        self.attributes = attributes or {"Size": -1, "Quality": -1}
        self.color = color or QColor(255, 0, 0)
        self.id = id(self)  # Unique identifier
        self.frame_id = None  # Frame this annotation belongs to

    def contains_point(self, point):
        """Check if the point is inside the bounding box"""
        return self.rect.contains(point)

    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "x": self.rect.x(),
            "y": self.rect.y(),
            "width": self.rect.width(),
            "height": self.rect.height(),
            "class_name": self.class_name,
            "attributes": self.attributes,
            "color": [
                self.color.red(),
                self.color.green(),
                self.color.blue(),
                self.color.alpha(),
            ],
            "frame_id": self.frame_id,
        }

    @classmethod
    def from_dict(cls, data):
        """Create a BoundingBox from a dictionary"""
        rect = QRect(data["x"], data["y"], data["width"], data["height"])
        bbox = cls(rect, data["class_name"], data["attributes"])
        bbox.id = data["id"]
        bbox.color = QColor(*data["color"])
        bbox.frame_id = data.get("frame_id")
        return bbox

    def __str__(self):
        """String representation for display"""
        attr_str = ", ".join(f"{k}: {v}" for k, v in self.attributes.items())
        if attr_str:
            return f"{self.class_name} [{attr_str}]"
        return self.class_name
