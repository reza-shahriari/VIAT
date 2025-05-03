from PyQt5.QtCore import QRect, QPoint
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
    
    def copy(self):
        """Create a deep copy of this bounding box."""
        from PyQt5.QtCore import QRect
        from PyQt5.QtGui import QColor
        
        # Create a new rect with the same dimensions
        new_rect = QRect(
            self.rect.x(),
            self.rect.y(),
            self.rect.width(),
            self.rect.height()
        )
        
        # Create a new color if one exists
        new_color = None
        if self.color:
            new_color = QColor(
                self.color.red(),
                self.color.green(),
                self.color.blue(),
                self.color.alpha()
            )
        
        # Create a copy of attributes
        new_attributes = self.attributes.copy() if self.attributes else {}
        
        # Return a new BoundingBox with the copied properties
        return BoundingBox(new_rect, self.class_name, new_attributes, new_color)


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

        from PyQt5.QtWidgets import (
            QDialog, QVBoxLayout, QFormLayout, QComboBox, 
            QSpinBox, QDoubleSpinBox, QLineEdit, QDialogButtonBox
        )
        from PyQt5.QtCore import QTimer

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

            # Update canvas and mark project as modified
            self.canvas.update()
            self.main_window.project_modified = True

            # Update annotation list
            self.main_window.update_annotation_list()

            # Save to frame annotations
            self.main_window.frame_annotations[self.main_window.current_frame] = self.canvas.annotations

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
            self.canvas.class_colors[current_class]
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
        self.main_window.frame_annotations[self.main_window.current_frame] = self.canvas.annotations
        
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
            self.main_window.frame_annotations[self.main_window.current_frame] = self.canvas.annotations

    def update_annotation_list(self):
        """Update the annotation list in the UI."""
        
        self.main_window.annotation_dock.update_annotation_list()
        

        # If duplicate frame detection is enabled, propagate annotations to duplicates
        if self.main_window.duplicate_frames_enabled and self.main_window.current_frame in self.main_window.frame_hashes:
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
        from PyQt5.QtWidgets import QMessageBox
        
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
        from PyQt5.QtWidgets import QMessageBox, QDialog, QComboBox, QSpinBox
        from PyQt5.QtCore import QRect
        from .annotation import BoundingBox
        
        if not self.main_window.cap:
            QMessageBox.warning(self.main_window, "Add Annotation", "Please open a video first!")
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
            bbox = BoundingBox(rect, class_name, attributes, color)

            # Add to annotations
            self.canvas.annotations.append(bbox)

            # Save to frame annotations
            self.main_window.frame_annotations[self.main_window.current_frame] = self.canvas.annotations

            self.update_annotation_list()
            self.canvas.update()

    def create_annotation_dialog(self):
        """Create a dialog for adding or editing annotations."""
        from PyQt5.QtWidgets import (
            QDialog, QVBoxLayout, QFormLayout, QLabel, QComboBox, 
            QSpinBox, QDialogButtonBox
        )
        
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

        if hasattr(self.main_window, "use_previous_attributes") and self.main_window.use_previous_attributes:
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
