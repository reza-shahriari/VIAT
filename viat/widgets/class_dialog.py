from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QDialogButtonBox,
    QColorDialog,
    QGroupBox,
    QGridLayout,
    QSpinBox,
)
from PyQt5.QtGui import QColor


class ClassDialog(QDialog):
    def __init__(self, parent=None, class_name="", color=None, attributes=None):
        super().__init__(parent)
        self.setWindowTitle("Add/Edit Class")
        self.class_name = class_name
        self.color = color or QColor(255, 0, 0)
        self.attributes = attributes or {"size": -1, "quality": -1}
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Class name
        name_layout = QHBoxLayout()
        name_label = QLabel("Class Name:")
        self.name_edit = QLineEdit(self.class_name)
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.name_edit)
        layout.addLayout(name_layout)

        # Color picker
        color_layout = QHBoxLayout()
        color_label = QLabel("Class Color:")
        self.color_button = QPushButton()
        self.color_button.setStyleSheet(f"background-color: {self.color.name()}")
        self.color_button.clicked.connect(self.pick_color)
        color_layout.addWidget(color_label)
        color_layout.addWidget(self.color_button)
        layout.addLayout(color_layout)

        # Attributes section
        attributes_group = QGroupBox("Default Attributes")
        self.attributes_layout = QGridLayout()
        attributes_group.setLayout(self.attributes_layout)

        # Add size attribute
        size_label = QLabel("Size:")
        self.size_spinner = QSpinBox()
        self.size_spinner.setRange(-1, 100)
        self.size_spinner.setValue(self.attributes.get("size", -1))
        self.size_spinner.setSpecialValueText("Default (-1)")

        self.attributes_layout.addWidget(size_label, 0, 0)
        self.attributes_layout.addWidget(self.size_spinner, 0, 1)

        # Add quality attribute
        quality_label = QLabel("Quality:")
        self.quality_spinner = QSpinBox()
        self.quality_spinner.setRange(-1, 100)
        self.quality_spinner.setValue(self.attributes.get("quality", -1))
        self.quality_spinner.setSpecialValueText("Default (-1)")

        self.attributes_layout.addWidget(quality_label, 1, 0)
        self.attributes_layout.addWidget(self.quality_spinner, 1, 1)

        # Add custom attributes
        row = 2
        self.custom_attributes = {}
        for attr_name, attr_value in self.attributes.items():
            if attr_name not in ["size", "quality"]:
                self.add_custom_attribute_row(row, attr_name, attr_value)
                row += 1

        layout.addWidget(attributes_group)

        # Add/Remove custom attribute buttons
        attr_buttons_layout = QHBoxLayout()
        add_attr_btn = QPushButton("Add Custom Attribute")
        add_attr_btn.clicked.connect(self.add_custom_attribute)
        remove_attr_btn = QPushButton("Remove Custom Attribute")
        remove_attr_btn.clicked.connect(self.remove_custom_attribute)
        attr_buttons_layout.addWidget(add_attr_btn)
        attr_buttons_layout.addWidget(remove_attr_btn)
        layout.addLayout(attr_buttons_layout)

        # Dialog buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def add_custom_attribute_row(self, row, name="", value=-1):
        name_edit = QLineEdit(name)
        value_spinner = QSpinBox()
        value_spinner.setRange(-1, 100)
        value_spinner.setValue(int(value) if isinstance(value, (int, float)) else -1)
        value_spinner.setSpecialValueText("Default (-1)")

        self.attributes_layout.addWidget(name_edit, row, 0)
        self.attributes_layout.addWidget(value_spinner, row, 1)

        self.custom_attributes[row] = (name_edit, value_spinner)

    def add_custom_attribute(self):
        row = 2 + len(self.custom_attributes)
        self.add_custom_attribute_row(row)

    def remove_custom_attribute(self):
        if not self.custom_attributes:
            return

        row = max(self.custom_attributes.keys())
        name_edit, value_spinner = self.custom_attributes.pop(row)
        name_edit.deleteLater()
        value_spinner.deleteLater()

    def pick_color(self):
        color = QColorDialog.getColor(self.color, self)
        if color.isValid():
            self.color = color
            self.color_button.setStyleSheet(f"background-color: {color.name()}")

    def get_attributes(self):
        attributes = {
            "size": self.size_spinner.value(),
            "quality": self.quality_spinner.value(),
        }

        # Add custom attributes
        for name_edit, value_spinner in self.custom_attributes.values():
            name = name_edit.text().strip()
            if name:
                attributes[name] = value_spinner.value()

        return attributes
