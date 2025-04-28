from PyQt5.QtCore import QRect, QPoint
from PyQt5.QtGui import QColor
import uuid

from PyQt5.QtCore import QRect
from PyQt5.QtGui import QColor


class BoundingBox:
    """
    Represents a bounding box annotation with class and attributes.
    """

    def __init__(self, rect, class_name, attributes=None, color=None):
        """
        Initialize a bounding box annotation.

        Args:
            rect (QRect): Rectangle coordinates
            class_name (str): Class name
            attributes (dict, optional): Dictionary of attributes
            color (QColor, optional): Color for display
        """
        self.rect = rect
        self.class_name = class_name
        self.attributes = attributes or {}
        self.color = color

    def to_dict(self):
        """Convert to a dictionary for serialization"""

        return {
            "rect": {
                "x": self.rect.x(),
                "y": self.rect.y(),
                "width": self.rect.width(),
                "height": self.rect.height(),
            },
            "class_name": self.class_name,
            "attributes": self.attributes,
            "color": (
                {
                    "r": self.color.red(),
                    "g": self.color.green(),
                    "b": self.color.blue(),
                    "a": self.color.alpha(),
                }
                if self.color
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data):
        """Create a BoundingBox from a dictionary"""

        rect_data = data.get("rect", {})
        rect = QRect(
            rect_data.get("x", 0),
            rect_data.get("y", 0),
            rect_data.get("width", 0),
            rect_data.get("height", 0),
        )

        class_name = data.get("class_name", "")
        attributes = data.get("attributes", {})

        color_data = data.get("color")
        if color_data:
            color = QColor(
                color_data.get("r", 0),
                color_data.get("g", 0),
                color_data.get("b", 0),
                color_data.get("a", 255),
            )
        else:
            color = QColor(255, 0, 0)  # Default red

        return cls(rect, class_name, attributes, color)
