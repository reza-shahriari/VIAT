from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QLineEdit,
    QDialogButtonBox,
    QMessageBox,
    QLabel,
    QGroupBox,
    QHBoxLayout,
    QPushButton,
    QColorDialog,
    QWidget,
    QCheckBox,
)
from PyQt5.QtCore import QTimer, QRect
import random


class BoundingBox:
    """
    Represents a bounding box annotation with class and attributes.
    """

    def __init__(self, rect, class_name, attributes=None, color=None, source="manual", score=None):
        """
        Initialize a bounding box annotation.

        Args:
            rect (QRect): Rectangle coordinates
            class_name (str): Class name
            attributes (dict, optional): Dictionary of attributes
            color (QColor, optional): Color for display
            source (str, optional): Source of the annotation (manual, interpolated, tracked, detected)
            score (float, optional): Confidence score for detected annotations
        """
        self.rect = rect
        self.class_name = class_name
        self.attributes = attributes or {}
        self.color = color
        self.source = source
        self.original_source = source  # Unchangeable record of how it was first created
        self.verified = source == "manual"  # Auto-verify manual annotations
        self.score = score  # Store confidence score for detections

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
            "source": self.source,
            "original_source": self.original_source,
            "verified": self.verified,
            "score": self.score
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

        source = data.get("source", "manual")
        score = data.get("score", None)
        
        bbox = cls(rect, class_name, attributes, color, source, score)
        bbox.verified = data.get("verified", source == "manual")
        bbox.original_source = data.get("original_source", source)
        return bbox

    def copy(self):
        """Create a deep copy of this bounding box."""

        # Create a new rect with the same dimensions
        new_rect = QRect(
            self.rect.x(), self.rect.y(), self.rect.width(), self.rect.height()
        )

        # Create a new color if one exists
        new_color = None
        if self.color:
            new_color = QColor(
                self.color.red(),
                self.color.green(),
                self.color.blue(),
                self.color.alpha(),
            )

        # Create a copy of attributes
        new_attributes = self.attributes.copy() if self.attributes else {}

        # Return a new BoundingBox with the copied properties
        bbox = BoundingBox(new_rect, self.class_name, new_attributes, new_color, self.source, self.score)
        bbox.verified = self.verified
        bbox.original_source = self.original_source
        return bbox

    def verify(self):
        """Mark the annotation as verified."""
        self.verified = True
        self.source = "manual"  
        self.score = None

class AnnotationManager:
    """
    Manages annotation operations including creation, editing, and deletion.
    Centralizes annotation-related functionality to improve code organization.
    """

    def __init__(self, main_window, canvas):
        """
        Initialize the annotation manager.

        Args:
            main_window: Reference to the main application window
            canvas: Reference to the video canvas where annotations are displayed
        """
        self.main_window = main_window
        self.canvas = canvas

    def edit_annotation(self, annotation, focus_first_field=False):
        """
        Edit the properties of an annotation.

        Args:
            annotation: The annotation to edit
            focus_first_field: Whether to focus on the first attribute field
        """
        if not annotation:
            return

        dialog = QDialog(self.main_window)
        dialog.setWindowTitle("Edit Annotation")
        dialog.setMinimumWidth(350)

        layout = QVBoxLayout()
        form_layout = QFormLayout()

        # Class selector
        class_combo = QComboBox()
        class_combo.addItems(self.canvas.class_colors.keys())
        class_combo.setCurrentText(annotation.class_name)
        form_layout.addRow("Class:", class_combo)

        # Get class attribute configuration if available
        class_attributes = {}
        if hasattr(self.canvas, "class_attributes"):
            class_attributes = self.canvas.class_attributes.get(
                annotation.class_name, {}
            )

        # Create input widgets for all attributes
        attribute_widgets = {}
        first_widget = None

        for attr_name, attr_value in sorted(annotation.attributes.items()):
            # Get attribute type from class configuration or infer from value
            attr_type = "string"
            attr_min = None
            attr_max = None

            if attr_name in class_attributes:
                attr_config = class_attributes[attr_name]
                attr_type = attr_config.get("type", "string")
                attr_min = attr_config.get("min", None)
                attr_max = attr_config.get("max", None)
            elif isinstance(attr_value, int):
                attr_type = "int"
            elif isinstance(attr_value, float):
                attr_type = "float"
            elif isinstance(attr_value, bool):
                attr_type = "boolean"

            # Create appropriate input widget based on type
            if attr_type == "boolean":
                input_widget = QComboBox()
                input_widget.addItems(["False", "True"])
                input_widget.setCurrentText(str(bool(attr_value)))
            elif attr_type == "int":
                input_widget = QSpinBox()
                if attr_min is not None:
                    try:
                        input_widget.setMinimum(int(attr_min))
                    except (ValueError, TypeError):
                        input_widget.setMinimum(-999999)
                else:
                    input_widget.setMinimum(-999999)
                if attr_max is not None:
                    try:
                        input_widget.setMaximum(int(attr_max))
                    except (ValueError, TypeError):
                        input_widget.setMaximum(999999)
                else:
                    input_widget.setMaximum(999999)
                input_widget.setValue(int(attr_value))
            elif attr_type == "float":
                input_widget = QDoubleSpinBox()
                if attr_min is not None:
                    try:
                        input_widget.setMinimum(float(attr_min))
                    except (ValueError, TypeError):
                        input_widget.setMinimum(-999999.0)
                else:
                    input_widget.setMinimum(-999999.0)
                if attr_max is not None:
                    try:
                        input_widget.setMaximum(float(attr_max))
                    except (ValueError, TypeError):
                        input_widget.setMaximum(999999.0)
                else:
                    input_widget.setMaximum(999999.0)
                input_widget.setValue(float(attr_value))
                input_widget.setDecimals(2)
            else:  # string or default
                input_widget = QLineEdit()
                input_widget.setText(str(attr_value))

            # Store the first widget for focus
            if first_widget is None and attr_name in ["Size", "Quality"]:
                first_widget = input_widget

            form_layout.addRow(f"{attr_name}:", input_widget)
            attribute_widgets[attr_name] = input_widget

        layout.addLayout(form_layout)

        # Add verification checkbox for machine-generated annotations
        verification_checkbox = None
        if hasattr(annotation, 'source') and annotation.source != "manual" and not annotation.verified:
            verification_checkbox = QCheckBox("Verify this annotation (mark as manually confirmed)")
            verification_checkbox.setChecked(True)  # Default to verified when editing
            layout.addWidget(verification_checkbox)
            
            # Add source information
            source_label = QLabel(f"Source: {annotation.source} (originally {annotation.original_source})")
            source_label.setStyleSheet("color: #888888;")
            layout.addWidget(source_label)

        # Add OK and Cancel buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        # Get direct references to the OK and Cancel buttons
        ok_button = button_box.button(QDialogButtonBox.Ok)
        cancel_button = button_box.button(QDialogButtonBox.Cancel)

        dialog.setLayout(layout)

        # Set focus on the first attribute field if requested
        if focus_first_field and first_widget:
            # Use singleShot timer to ensure focus happens after dialog is shown
            QTimer.singleShot(0, lambda: first_widget.setFocus())
            # For QSpinBox and QDoubleSpinBox, select all text
            if isinstance(first_widget, (QSpinBox, QDoubleSpinBox)):
                QTimer.singleShot(0, lambda: first_widget.selectAll())
            # For QLineEdit, select all text
            elif isinstance(first_widget, QLineEdit):
                QTimer.singleShot(0, lambda: first_widget.selectAll())

        # Set proper tab order
        previous_widget = class_combo
        for attr_name, widget in attribute_widgets.items():
            dialog.setTabOrder(previous_widget, widget)
            previous_widget = widget

        # Make sure the last tab goes to the OK button first, then Cancel
        dialog.setTabOrder(previous_widget, ok_button)
        dialog.setTabOrder(ok_button, cancel_button)

        # If dialog is accepted, update the annotation
        if dialog.exec_() == QDialog.Accepted:
            old_class = annotation.class_name
            new_class = class_combo.currentText()

            # Update class and color
            annotation.class_name = new_class
            annotation.color = self.canvas.class_colors[new_class]

            # If class changed, update attributes based on new class configuration
            if old_class != new_class and hasattr(self.canvas, "class_attributes"):
                new_class_attributes = self.canvas.class_attributes.get(new_class, {})
                self.update_annotation_attributes(annotation, new_class_attributes)

            # Update attribute values from the dialog
            for attr_name, widget in attribute_widgets.items():
                if attr_name in annotation.attributes:
                    if isinstance(widget, QComboBox):  # Boolean
                        annotation.attributes[attr_name] = (
                            widget.currentText() == "True"
                        )
                    elif isinstance(widget, QSpinBox):  # Int
                        annotation.attributes[attr_name] = widget.value()
                    elif isinstance(widget, QDoubleSpinBox):  # Float
                        annotation.attributes[attr_name] = widget.value()
                    else:  # String or other
                        annotation.attributes[attr_name] = widget.text()

            # Handle verification if applicable
            if verification_checkbox and verification_checkbox.isChecked():
                annotation.verify()
                # Show confirmation in status bar
                if hasattr(self.main_window, "statusBar"):
                    self.main_window.statusBar.showMessage(
                        f"Annotation verified and marked as manual (originally {annotation.original_source})", 
                        3000
                    )

            # Update canvas and mark project as modified
            self.canvas.update()
            self.main_window.project_modified = True

            # Update annotation list
            self.main_window.update_annotation_list()

            # Save to frame annotations
            self.main_window.frame_annotations[self.main_window.current_frame] = (
                self.canvas.annotations
            )

    def update_annotation_attributes(self, annotation, class_attributes):
        """
        Update annotation attributes based on class configuration.

        Args:
            annotation: The annotation to update
            class_attributes: The class attribute configuration
        """
        # Keep existing attributes that are still valid for the new class
        current_attrs = annotation.attributes.copy()
        annotation.attributes = {}

        # Add default attributes from class configuration
        for attr_name, attr_config in class_attributes.items():
            # If attribute already exists, keep its value
            if attr_name in current_attrs:
                annotation.attributes[attr_name] = current_attrs[attr_name]
            # Otherwise, set default value based on type
            else:
                attr_type = attr_config.get("type", "string")
                if attr_type == "boolean":
                    annotation.attributes[attr_name] = False
                elif attr_type == "int":
                    annotation.attributes[attr_name] = 0
                elif attr_type == "float":
                    annotation.attributes[attr_name] = 0.0
                else:  # string or default
                    annotation.attributes[attr_name] = ""

    def add_empty_annotation(self):
        """Add a new empty annotation with default values."""
        from .annotation import BoundingBox

        # Get current frame dimensions
        frame_width = self.canvas.pixmap.width() if self.canvas.pixmap else 640
        frame_height = self.canvas.pixmap.height() if self.canvas.pixmap else 480

        # Create a default bounding box in the center of the frame
        center_x = frame_width // 2
        center_y = frame_height // 2
        width = frame_width // 4
        height = frame_height // 4

        # Create the bounding box with default class
        current_class = self.canvas.current_class
        bbox = BoundingBox(
            center_x - width // 2,
            center_y - height // 2,
            center_x + width // 2,
            center_y + height // 2,
            current_class,
            self.canvas.class_colors[current_class],
        )

        # Add default attributes if class has attribute configuration
        if hasattr(self.canvas, "class_attributes"):
            class_attributes = self.canvas.class_attributes.get(current_class, {})
            self.update_annotation_attributes(bbox, class_attributes)

        # Add the annotation to the canvas
        self.canvas.annotations.append(bbox)
        self.canvas.selected_annotation = bbox

        # Update the canvas and annotation list
        self.canvas.update()
        self.update_annotation_list()

        # Mark project as modified
        self.main_window.project_modified = True

        # Save to frame annotations
        self.main_window.frame_annotations[self.main_window.current_frame] = (
            self.canvas.annotations
        )

        # Edit the new annotation
        self.edit_annotation(bbox, focus_first_field=True)

    def delete_annotation(self, annotation):
        """
        Delete an annotation.

        Args:
            annotation: The annotation to delete
        """
        if annotation in self.canvas.annotations:
            # Remove from canvas annotations
            self.canvas.annotations.remove(annotation)

            # If this was the selected annotation, clear selection
            if self.canvas.selected_annotation == annotation:
                self.canvas.selected_annotation = None

            # Update the canvas and annotation list
            self.canvas.update()
            self.update_annotation_list()

            # Mark project as modified
            self.main_window.project_modified = True

            # Save to frame annotations
            self.main_window.frame_annotations[self.main_window.current_frame] = (
                self.canvas.annotations
            )

    def update_annotation_list(self):
        """Update the annotation list in the UI."""

        self.main_window.annotation_dock.update_annotation_list()

        # If duplicate frame detection is enabled, propagate annotations to duplicates
        if (
            self.main_window.duplicate_frames_enabled
            and self.main_window.current_frame in self.main_window.frame_hashes
        ):
            current_hash = self.main_window.frame_hashes[self.main_window.current_frame]
            if (
                current_hash in self.main_window.duplicate_frames_cache
                and len(self.main_window.duplicate_frames_cache[current_hash]) > 1
            ):
                # Propagate current frame annotations to all duplicates
                self.main_window.propagate_to_duplicate_frames(current_hash)

        self.main_window.perform_autosave()

    def clear_annotations(self):
        """Clear all annotations after confirmation."""

        if not self.canvas.annotations:
            return

        reply = QMessageBox.question(
            self.main_window,
            "Clear Annotations",
            "Are you sure you want to clear all annotations?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.canvas.annotations = []
            self.canvas.selected_annotation = None
            self.update_annotation_list()
            self.canvas.update()
            self.main_window.statusBar.showMessage("All annotations cleared")

    def add_annotation(self):
        """Add annotation manually through a dialog."""
        from .annotation import BoundingBox

        if not self.main_window.cap:
            QMessageBox.warning(
                self.main_window, "Add Annotation", "Please open a video first!"
            )
            return

        # Create dialog
        dialog = self.create_annotation_dialog()

        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            # Get form widgets
            class_combo = dialog.findChild(QComboBox)
            x_spin = dialog.findChildren(QSpinBox)[0]
            y_spin = dialog.findChildren(QSpinBox)[1]
            width_spin = dialog.findChildren(QSpinBox)[2]
            height_spin = dialog.findChildren(QSpinBox)[3]
            size_spin = dialog.findChildren(QSpinBox)[4]
            quality_spin = dialog.findChildren(QSpinBox)[5]

            # Create attributes dictionary
            attributes = {"Size": size_spin.value(), "Quality": quality_spin.value()}

            # Create rectangle
            rect = QRect(
                x_spin.value(), y_spin.value(), width_spin.value(), height_spin.value()
            )

            # Get class and color
            class_name = class_combo.currentText()
            color = self.canvas.class_colors.get(class_name, QColor(255, 0, 0))

            # Create bounding box
            bbox = BoundingBox(rect, class_name, attributes, color,source='manual')

            # Add to annotations
            self.canvas.annotations.append(bbox)

            # Save to frame annotations
            self.main_window.frame_annotations[self.main_window.current_frame] = (
                self.canvas.annotations
            )

            self.update_annotation_list()
            self.canvas.update()

    def create_annotation_dialog(self):
        """Create a dialog for adding or editing annotations."""

        dialog = QDialog(self.main_window)
        dialog.setWindowTitle("Add Annotation")
        dialog.setMinimumWidth(300)

        layout = QVBoxLayout(dialog)

        # Class selection
        class_label = QLabel("Class:")
        class_combo = QComboBox()
        class_combo.addItems(list(self.canvas.class_colors.keys()))

        # Coordinates
        coords_layout = QFormLayout()
        x_spin = QSpinBox()
        x_spin.setRange(0, self.canvas.pixmap.width() if self.canvas.pixmap else 1000)
        y_spin = QSpinBox()
        y_spin.setRange(0, self.canvas.pixmap.height() if self.canvas.pixmap else 1000)
        width_spin = QSpinBox()
        width_spin.setRange(
            5, self.canvas.pixmap.width() if self.canvas.pixmap else 1000
        )
        height_spin = QSpinBox()
        height_spin.setRange(
            5, self.canvas.pixmap.height() if self.canvas.pixmap else 1000
        )

        coords_layout.addRow("X:", x_spin)
        coords_layout.addRow("Y:", y_spin)
        coords_layout.addRow("Width:", width_spin)
        coords_layout.addRow("Height:", height_spin)

        # Attributes
        attributes_layout = QFormLayout()

        # Get default values from previous annotations if enabled
        default_size = -1
        default_quality = -1

        if (
            hasattr(self.main_window, "use_previous_attributes")
            and self.main_window.use_previous_attributes
        ):
            # Get the current selected class
            current_class = class_combo.currentText()
            prev_attributes = self.get_previous_annotation_attributes(current_class)

            if prev_attributes:
                default_size = prev_attributes.get("Size", -1)
                default_quality = prev_attributes.get("Quality", -1)

        # Create attribute spinboxes with default values
        size_spin = QSpinBox()
        size_spin.setRange(0, 100)
        size_spin.setValue(default_size)  # Use default or previous value

        quality_spin = QSpinBox()
        quality_spin.setRange(0, 100)
        quality_spin.setValue(default_quality)  # Use default or previous value

        attributes_layout.addRow("Size (0-100):", size_spin)
        attributes_layout.addRow("Quality (0-100):", quality_spin)

        # Update attributes when class changes
        def update_attributes_for_class(class_name):
            if (
                hasattr(self.main_window, "use_previous_attributes")
                and self.main_window.use_previous_attributes
            ):
                prev_attributes = self.get_previous_annotation_attributes(class_name)
                if prev_attributes:
                    size_spin.setValue(prev_attributes.get("Size", -1))
                    quality_spin.setValue(prev_attributes.get("Quality", -1))

        class_combo.currentTextChanged.connect(update_attributes_for_class)

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        # Add widgets to layout
        layout.addWidget(class_label)
        layout.addWidget(class_combo)
        layout.addLayout(coords_layout)
        layout.addLayout(attributes_layout)
        layout.addWidget(buttons)

        # Set tab order for easy navigation
        dialog.setTabOrder(class_combo, x_spin)
        dialog.setTabOrder(x_spin, y_spin)
        dialog.setTabOrder(y_spin, width_spin)
        dialog.setTabOrder(width_spin, height_spin)
        dialog.setTabOrder(height_spin, size_spin)
        dialog.setTabOrder(size_spin, quality_spin)

        # Focus on the first field
        class_combo.setFocus()

        return dialog

    def get_previous_annotation_attributes(self, class_name):
        """
        Get attributes from previous annotations of the same class.

        Args:
            class_name: The class name to look for

        Returns:
            Dictionary of attributes or None if no previous annotations found
        """
        # Look through all frames for annotations of this class
        for frame_num, annotations in self.main_window.frame_annotations.items():
            for annotation in annotations:
                if annotation.class_name == class_name:
                    return annotation.attributes

        return None

    def parse_attributes(self, text):
        """
        Parse attributes from text input.

        Args:
            text: Text containing attribute definitions

        Returns:
            Dictionary of parsed attributes
        """
        attributes = {}
        for line in text.strip().split("\n"):
            if "=" in line:
                key, value = line.split("=", 1)
                attributes[key.strip()] = value.strip()
        return attributes

class ClassManager:
    """
    Manages class-related operations including adding, editing, and deleting classes.
    Centralizes class-related functionality to improve code organization.
    """

    def __init__(self, main_window):
        """
        Initialize the class manager.

        Args:
            main_window: Reference to the main application window
        """
        self.main_window = main_window

    def add_class(self):
        """Add a new class with custom attributes."""
        # Create dialog
        dialog = self.create_class_dialog()

        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            class_name = dialog.name_edit.text().strip()

            if not class_name:
                QMessageBox.warning(
                    self.main_window, "Add Class", "Class name cannot be empty!"
                )
                return

            if class_name in self.main_window.canvas.class_colors:
                QMessageBox.warning(
                    self.main_window,
                    "Add Class",
                    f"Class '{class_name}' already exists!",
                )
                return

            # Get color from dialog
            color = dialog.color

            # Process attributes
            attributes_config = {}
            for (
                _,
                name_edit,
                type_combo,
                default_edit,
                min_edit,
                max_edit,
            ) in dialog.attribute_widgets:
                attr_name = name_edit.text().strip()
                if attr_name:
                    attr_type = type_combo.currentText()

                    # Parse default value based on type
                    default_value = default_edit.text()
                    if attr_type == "int":
                        try:
                            default_value = int(default_value)
                        except ValueError:
                            default_value = 0
                    elif attr_type == "float":
                        try:
                            default_value = float(default_value)
                        except ValueError:
                            default_value = 0.0
                    elif attr_type == "boolean":
                        default_value = default_value.lower() in ["true", "1", "yes"]

                    # Parse min/max for numeric types
                    attr_config = {"type": attr_type, "default": default_value}

                    if attr_type in ["int", "float"]:
                        try:
                            attr_config["min"] = (
                                int(min_edit.text())
                                if attr_type == "int"
                                else float(min_edit.text())
                            )
                        except ValueError:
                            attr_config["min"] = 0

                        try:
                            attr_config["max"] = (
                                int(max_edit.text())
                                if attr_type == "int"
                                else float(max_edit.text())
                            )
                        except ValueError:
                            attr_config["max"] = 100

                    attributes_config[attr_name] = attr_config

            # Add class to canvas
            self.main_window.canvas.class_colors[class_name] = color

            # Store attributes configuration
            if not hasattr(self.main_window, "class_attributes"):
                self.main_window.class_attributes = {}
            self.main_window.class_attributes[class_name] = attributes_config
            if hasattr(self, "annotation_dock"):
                self.main_window.annotation_dock.update_class_selector()
            # Update UI
            self.main_window.toolbar.update_class_selector()
            self.main_window.class_dock.update_class_list()
            self.main_window.refresh_class_lists()

            # Set as current class
            self.main_window.canvas.set_current_class(class_name)

            # Update the class selector in the toolbar directly
            if hasattr(self, "class_selector"):
                self.main_window.class_selector.blockSignals(True)
                self.main_window.class_selector.setCurrentText(class_name)
                self.main_window.class_selector.blockSignals(False)

    def create_class_dialog(self, class_name=None, color=None, attributes=None):
        """Create a dialog for adding or editing classes with custom attributes."""
        dialog = QDialog(self.main_window)
        dialog.setWindowTitle("Add Class" if class_name is None else "Edit Class")
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout(dialog)

        # Class name
        name_layout = QHBoxLayout()
        name_label = QLabel("Class Name:")
        name_edit = QLineEdit(class_name if class_name else "")
        name_layout.addWidget(name_label)
        name_layout.addWidget(name_edit)

        # Color selection
        color_layout = QHBoxLayout()
        color_label = QLabel("Color:")
        color_button = QPushButton()
        color_button.setAutoFillBackground(True)

        # Set initial color
        if color is None:
            color = QColor(
                random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
            )
        color_button.setStyleSheet(f"background-color: {color.name()}")

        def choose_color():
            nonlocal color
            new_color = QColorDialog.getColor(color, dialog, "Select Color")
            if new_color.isValid():
                color = new_color
                dialog.color = new_color  # Make sure this is set
                color_button.setStyleSheet(f"background-color: {color.name()}")

        color_button.clicked.connect(choose_color)
        color_layout.addWidget(color_label)
        color_layout.addWidget(color_button)

        # Attributes section
        attributes_group = QGroupBox("Attributes")
        attributes_layout = QVBoxLayout(attributes_group)

        # List to store attribute widgets
        attribute_widgets = []

        # Function to add a new attribute row
        def add_attribute_row(
            name="", attr_type="int", default_value="0", min_value="0", max_value="100"
        ):
            row_widget = QWidget()
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(0, 0, 0, 0)

            name_edit = QLineEdit(name)
            name_edit.setPlaceholderText("Attribute Name")

            type_combo = QComboBox()
            type_combo.addItems(["int", "float", "string", "boolean"])
            type_combo.setCurrentText(attr_type)

            default_edit = QLineEdit(default_value)
            default_edit.setPlaceholderText("Default")

            min_edit = QLineEdit(min_value)
            min_edit.setPlaceholderText("Min")

            max_edit = QLineEdit(max_value)
            max_edit.setPlaceholderText("Max")

            delete_btn = QPushButton("X")
            delete_btn.setMaximumWidth(30)
            delete_btn.clicked.connect(lambda: remove_attribute_row(row_widget))

            row_layout.addWidget(name_edit)
            row_layout.addWidget(type_combo)
            row_layout.addWidget(default_edit)
            row_layout.addWidget(min_edit)
            row_layout.addWidget(max_edit)
            row_layout.addWidget(delete_btn)

            attributes_layout.addWidget(row_widget)
            attribute_widgets.append(
                (row_widget, name_edit, type_combo, default_edit, min_edit, max_edit)
            )

            # Update type-dependent visibility
            def update_type_visibility():
                is_numeric = type_combo.currentText() in ["int", "float"]
                min_edit.setVisible(is_numeric)
                max_edit.setVisible(is_numeric)

            type_combo.currentTextChanged.connect(update_type_visibility)
            update_type_visibility()

            return row_widget

        def remove_attribute_row(row_widget):
            for widget, *_ in attribute_widgets[:]:
                if widget == row_widget:
                    attributes_layout.removeWidget(widget)
                    widget.deleteLater()
                    attribute_widgets.remove((widget, *_))
                    break

        # Add button for attributes
        add_attr_btn = QPushButton("Add Attribute")
        add_attr_btn.clicked.connect(lambda: add_attribute_row())
        attributes_layout.addWidget(add_attr_btn)

        # Add default attributes if editing an existing class
        if attributes:
            for attr_name, attr_info in attributes.items():
                add_attribute_row(
                    name=attr_name,
                    attr_type=attr_info.get("type", "int"),
                    default_value=str(attr_info.get("default", "0")),
                    min_value=str(attr_info.get("min", "0")),
                    max_value=str(attr_info.get("max", "100")),
                )
        else:
            # Add default Size and Quality attributes for new classes
            add_attribute_row("Size", "int", "-1", "0", "100")
            add_attribute_row("Quality", "int", "-1", "0", "100")

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        # Add widgets to layout
        layout.addLayout(name_layout)
        layout.addLayout(color_layout)
        layout.addWidget(attributes_group)
        layout.addWidget(buttons)

        # Store attribute widgets for access when dialog is accepted
        dialog.attribute_widgets = attribute_widgets
        dialog.name_edit = name_edit
        dialog.color = color

        return dialog

    def convert_class_with_attributes(
        self, source_class, target_class, keep_original=True
    ):
        """
        Convert annotations from one class to another with attribute handling.

        Args:
            source_class (str): The source class name
            target_class (str): The target class name
            keep_original (bool): Whether to keep original attributes or use target class defaults
        """
        # Get target class attribute configuration
        target_attributes = {}
        if hasattr(self.main_window.canvas, "class_attributes"):
            target_attributes = self.main_window.canvas.class_attributes.get(
                target_class, {}
            )

        # Convert all annotations of the source class to the target class
        for frame_num, annotations in self.main_window.frame_annotations.items():
            for annotation in annotations:
                if annotation.class_name == source_class:
                    # Change class name
                    annotation.class_name = target_class

                    # Update color
                    annotation.color = self.main_window.canvas.class_colors.get(
                        target_class, annotation.color
                    )

                    # Handle attributes based on the keep_original flag
                    if not keep_original:
                        # Use target class attribute defaults
                        new_attributes = {}
                        for attr_name, attr_config in target_attributes.items():
                            attr_type = attr_config.get("type", "string")
                            default_value = attr_config.get("default", "")

                            if attr_type == "int":
                                new_attributes[attr_name] = (
                                    int(default_value) if default_value else 0
                                )
                            elif attr_type == "float":
                                new_attributes[attr_name] = (
                                    float(default_value) if default_value else 0.0
                                )
                            elif attr_type == "boolean":
                                new_attributes[attr_name] = default_value in [
                                    True,
                                    "True",
                                    "true",
                                    "1",
                                ]
                            else:  # string or default
                                new_attributes[attr_name] = str(default_value)

                        annotation.attributes = new_attributes

    def convert_class_with_attribute_mapping(self, source_class, target_class):
        """
        Convert annotations with user-defined attribute mapping.

        Args:
            source_class (str): The source class name
            target_class (str): The target class name
        """
        # Get source and target attribute configurations
        source_attributes = {}
        target_attributes = {}

        if hasattr(self.main_window.canvas, "class_attributes"):
            source_attributes = self.main_window.canvas.class_attributes.get(
                source_class, {}
            )
            target_attributes = self.main_window.canvas.class_attributes.get(
                target_class, {}
            )

        # Create attribute mapping dialog
        dialog = QDialog(self.main_window)
        dialog.setWindowTitle("Map Attributes")
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout(dialog)
        layout.addWidget(
            QLabel(f"Map attributes from '{source_class}' to '{target_class}':")
        )

        # Create mapping widgets
        mapping_widgets = {}

        for source_attr in sorted(source_attributes.keys()):
            row_layout = QHBoxLayout()
            row_layout.addWidget(QLabel(f"{source_attr}:"))

            target_combo = QComboBox()
            target_combo.addItem("(Ignore)")
            target_combo.addItems(sorted(target_attributes.keys()))

            # Try to find a matching target attribute by name
            if source_attr in target_attributes:
                target_combo.setCurrentText(source_attr)

            row_layout.addWidget(target_combo)
            layout.addLayout(row_layout)
            mapping_widgets[source_attr] = target_combo

        # Add buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            # Create attribute mapping
            attr_mapping = {}
            for source_attr, combo in mapping_widgets.items():
                target_attr = combo.currentText()
                if target_attr != "(Ignore)":
                    attr_mapping[source_attr] = target_attr

            # Convert all annotations with the mapping
            for frame_num, annotations in self.main_window.frame_annotations.items():
                for annotation in annotations:
                    if annotation.class_name == source_class:
                        # Change class name
                        annotation.class_name = target_class

                        # Update color
                        annotation.color = self.main_window.canvas.class_colors.get(
                            target_class, annotation.color
                        )

                        # Map attributes
                        new_attributes = {}

                        # First, set default values for all target attributes
                        for attr_name, attr_config in target_attributes.items():
                            attr_type = attr_config.get("type", "string")
                            default_value = attr_config.get("default", "")

                            if attr_type == "int":
                                new_attributes[attr_name] = (
                                    int(default_value) if default_value else 0
                                )
                            elif attr_type == "float":
                                new_attributes[attr_name] = (
                                    float(default_value) if default_value else 0.0
                                )
                            elif attr_type == "boolean":
                                new_attributes[attr_name] = default_value in [
                                    True,
                                    "True",
                                    "true",
                                    "1",
                                ]
                            else:  # string or default
                                new_attributes[attr_name] = str(default_value)

                        # Then apply the mapping
                        for source_attr, target_attr in attr_mapping.items():
                            if source_attr in annotation.attributes:
                                new_attributes[target_attr] = annotation.attributes[
                                    source_attr
                                ]

                        annotation.attributes = new_attributes

    def edit_selected_class(self):
        """Edit the selected class with option to convert to another class."""
        # Get the selected class
        selected_class = None
        if (
            hasattr(self.main_window, "class_dock")
            and self.main_window.class_dock.classes_list.currentItem()
        ):
            selected_class = (
                self.main_window.class_dock.classes_list.currentItem().text()
            )
        elif hasattr(self.main_window.canvas, "current_class"):
            selected_class = self.main_window.canvas.current_class

        if not selected_class:
            QMessageBox.warning(
                self.main_window, "No Class Selected", "Please select a class to edit."
            )
            return

        # Get current color and attributes
        current_color = self.main_window.canvas.class_colors.get(
            selected_class, QColor(255, 0, 0)
        )
        current_attributes = self.main_window.canvas.class_attributes.get(
            selected_class, {}
        )

        # Create dialog
        dialog = self.create_class_dialog(
            selected_class, current_color, current_attributes
        )

        # Add conversion option
        conversion_group = QGroupBox("Convert Class")
        conversion_layout = QVBoxLayout(conversion_group)

        convert_check = QCheckBox("Convert to another existing class")
        convert_check.setChecked(False)

        target_class_combo = QComboBox()
        target_class_combo.addItems(
            [
                c
                for c in self.main_window.canvas.class_colors.keys()
                if c != selected_class
            ]
        )
        target_class_combo.setEnabled(False)

        attribute_handling_combo = QComboBox()
        attribute_handling_combo.addItems(
            [
                "Keep original attributes",
                "Use target class attributes",
                "Merge attributes",
            ]
        )
        attribute_handling_combo.setEnabled(False)

        # Connect checkbox to enable/disable conversion options
        def toggle_conversion_options(checked):
            target_class_combo.setEnabled(checked)
            attribute_handling_combo.setEnabled(checked)

        convert_check.toggled.connect(toggle_conversion_options)

        conversion_layout.addWidget(convert_check)
        conversion_layout.addWidget(QLabel("Target class:"))
        conversion_layout.addWidget(target_class_combo)
        conversion_layout.addWidget(QLabel("Attribute handling:"))
        conversion_layout.addWidget(attribute_handling_combo)

        # Add conversion group to dialog layout
        dialog.layout().insertWidget(dialog.layout().count() - 1, conversion_group)

        if dialog.exec_() == QDialog.Accepted:
            new_class_name = dialog.name_edit.text().strip()
            new_color = dialog.color

            # Get attributes from dialog
            new_attributes = {}
            for (
                _,
                name_edit,
                type_combo,
                default_edit,
                min_edit,
                max_edit,
            ) in dialog.attribute_widgets:
                attr_name = name_edit.text().strip()
                if not attr_name:
                    continue

                attr_type = type_combo.currentText()
                attr_default = default_edit.text()

                # Create attribute config
                attr_config = {"type": attr_type, "default": attr_default}

                # Add min/max for numeric types
                if attr_type in ["int", "float"]:
                    attr_config["min"] = min_edit.text()
                    attr_config["max"] = max_edit.text()

                new_attributes[attr_name] = attr_config

            # Check if we're converting to another class
            if convert_check.isChecked() and target_class_combo.currentText():
                target_class = target_class_combo.currentText()
                attribute_handling = attribute_handling_combo.currentText()

                # Confirm conversion
                reply = QMessageBox.question(
                    self.main_window,
                    "Convert Class",
                    f"Are you sure you want to convert all '{selected_class}' annotations to '{target_class}'?\n\n"
                    f"The '{selected_class}' class will be deleted after conversion.\n"
                    f"This action cannot be undone.",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )

                if reply == QMessageBox.Yes:
                    # Handle attributes based on user selection
                    if attribute_handling == "Keep original attributes":
                        # Keep original attributes for each annotation
                        self.convert_class_with_attributes(
                            selected_class, target_class, keep_original=True
                        )
                    elif attribute_handling == "Use target class attributes":
                        # Use target class attribute defaults
                        self.convert_class_with_attributes(
                            selected_class, target_class, keep_original=False
                        )
                    else:  # "Merge attributes"
                        # Show attribute mapping dialog
                        self.convert_class_with_attribute_mapping(
                            selected_class, target_class
                        )

                    # Delete the original class
                    if selected_class in self.main_window.canvas.class_colors:
                        del self.main_window.canvas.class_colors[selected_class]
                    if (
                        hasattr(self.main_window.canvas, "class_attributes")
                        and selected_class in self.main_window.canvas.class_attributes
                    ):
                        del self.main_window.canvas.class_attributes[selected_class]

                    # If the current class was the one deleted, set it to the target class
                    if self.main_window.canvas.current_class == selected_class:
                        self.main_window.canvas.current_class = target_class

                    # Update UI
                    self.main_window.refresh_class_ui()

                    # Mark project as modified
                    self.main_window.project_modified = True

                    # Show success message
                    self.main_window.statusBar.showMessage(
                        f"Converted all '{selected_class}' annotations to '{target_class}' and deleted '{selected_class}' class",
                        5000,
                    )
                    return

            # Handle regular class editing (no conversion)
            # Skip if no changes
            if (
                selected_class == new_class_name
                and current_color.name() == new_color.name()
                and current_attributes == new_attributes
            ):
                return

            # Handle class name change
            if selected_class != new_class_name:
                # Update class name in all annotations
                for (
                    frame_num,
                    annotations,
                ) in self.main_window.frame_annotations.items():
                    for annotation in annotations:
                        if annotation.class_name == selected_class:
                            annotation.class_name = new_class_name

                # Update class colors dictionary
                if selected_class in self.main_window.canvas.class_colors:
                    self.main_window.canvas.class_colors[new_class_name] = (
                        self.main_window.canvas.class_colors.pop(selected_class)
                    )

                # Update class attributes dictionary
                if (
                    hasattr(self.main_window.canvas, "class_attributes")
                    and selected_class in self.main_window.canvas.class_attributes
                ):
                    self.main_window.canvas.class_attributes[new_class_name] = (
                        self.main_window.canvas.class_attributes.pop(selected_class)
                    )

                # Update current class if needed
                if self.main_window.canvas.current_class == selected_class:
                    self.main_window.canvas.current_class = new_class_name

            # Update color
            self.main_window.canvas.class_colors[new_class_name] = new_color

            # Update all annotations with this class to use the new color
            for frame_num, annotations in self.main_window.frame_annotations.items():
                for annotation in annotations:
                    if annotation.class_name == new_class_name:
                        annotation.color = new_color

            # Update class attributes
            if hasattr(self.main_window.canvas, "class_attributes"):
                self.main_window.canvas.class_attributes[new_class_name] = (
                    new_attributes
                )

            # Update UI
            self.main_window.refresh_class_ui()

            # Mark project as modified
            self.main_window.project_modified = True

    def convert_class(self, old_class, new_class):
        """Convert all annotations from one class to another."""
        # Update annotations in current frame
        for annotation in self.main_window.canvas.annotations:
            if annotation.class_name == old_class:
                annotation.class_name = new_class
                annotation.color = self.main_window.canvas.class_colors[new_class]

        # Update annotations in all frames
        for frame_num, annotations in self.main_window.frame_annotations.items():
            for annotation in annotations:
                if annotation.class_name == old_class:
                    annotation.class_name = new_class
                    annotation.color = self.main_window.canvas.class_colors[new_class]

        self.main_window.statusBar.showMessage(
            f"Converted all '{old_class}' annotations to '{new_class}'"
        )

    def update_class(self, old_name, new_name, color):
        """Update a class with new name and color."""
        # Update class name in annotations
        if old_name != new_name:
            for annotation in self.main_window.canvas.annotations:
                if annotation.class_name == old_name:
                    annotation.class_name = new_name

            # Update class colors dictionary
            self.main_window.canvas.class_colors[new_name] = color
            del self.main_window.canvas.class_colors[old_name]
        else:
            # Just update the color
            self.main_window.canvas.class_colors[old_name] = color

        # Update annotations with new color
        for annotation in self.main_window.canvas.annotations:
            if annotation.class_name == new_name:
                annotation.color = color

        # Update UI
        self.main_window.toolbar.update_class_selector()
        self.main_window.class_dock.update_class_list()
        self.main_window.update_annotation_list()

        # Update canvas
        self.main_window.canvas.update()

    def delete_selected_class(self):
        """Delete the selected class."""
        item = self.main_window.class_dock.classes_list.currentItem()
        if not item:
            return

        class_name = item.text()

        # Check if class is in use
        in_use = any(
            annotation.class_name == class_name
            for annotation in self.main_window.canvas.annotations
        )

        # Also check if class is used in any frame
        for frame_num, annotations in self.main_window.frame_annotations.items():
            if any(annotation.class_name == class_name for annotation in annotations):
                in_use = True
                break

        message = f"Are you sure you want to delete the class '{class_name}'?"
        if in_use:
            message += "\n\nThis class is currently in use by annotations. Deleting it will remove all annotations of this class."

        reply = QMessageBox.question(
            self.main_window,
            "Delete Class",
            message,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            # Remove annotations of this class from current frame
            self.main_window.canvas.annotations = [
                a for a in self.main_window.canvas.annotations if a.class_name != class_name
            ]

            # Remove annotations of this class from all frames
            for frame_num in self.main_window.frame_annotations:
                self.main_window.frame_annotations[frame_num] = [
                    a for a in self.main_window.frame_annotations[frame_num] if a.class_name != class_name
                ]

            # Remove class from colors dictionary
            if class_name in self.main_window.canvas.class_colors:
                del self.main_window.canvas.class_colors[class_name]

            # Remove class from class_attributes if it exists
            if hasattr(self.main_window.canvas, "class_attributes") and class_name in self.main_window.canvas.class_attributes:
                del self.main_window.canvas.class_attributes[class_name]

            # Update UI
            self.main_window.toolbar.update_class_selector()
            self.main_window.class_dock.update_class_list()
            self.main_window.update_annotation_list()

            # Update canvas
            if hasattr(self.main_window, "class_selector") and self.main_window.class_selector.count() > 0:
                self.main_window.canvas.set_current_class(self.main_window.class_selector.currentText())
            self.main_window.canvas.update()
            
            # Mark project as modified
            self.main_window.project_modified = True
            
            # Show success message
            self.main_window.statusBar.showMessage(f"Deleted class '{class_name}' and all its annotations", 5000)
