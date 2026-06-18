from PyQt5.QtCore import Qt, QAbstractListModel, QModelIndex
from PyQt5.QtGui import QColor, QFont

class AnnotationListModel(QAbstractListModel):
    def __init__(self, annotations=None, parent=None):
        super().__init__(parent)
        self.annotations = annotations or []

    def data(self, index, role):
        if not index.isValid():
            return None

        annotation = self.annotations[index.row()]

        if role == Qt.DisplayRole:
            # Main text to display
            text = f"Class: {annotation.class_name}"
            # Add attributes info
            if annotation.attributes:
                attrs = ", ".join(f"{k}={v}" for k, v in annotation.attributes.items())
                text += f" | {attrs}"
            return text

        elif role == Qt.UserRole:
            return annotation

        return None

    def rowCount(self, parent=QModelIndex()):
        return len(self.annotations)

    def set_annotations(self, annotations):
        self.beginResetModel()
        self.annotations = annotations
        self.endResetModel()

    def get_annotation(self, index):
        if index.isValid() and 0 <= index.row() < len(self.annotations):
            return self.annotations[index.row()]
        return None
