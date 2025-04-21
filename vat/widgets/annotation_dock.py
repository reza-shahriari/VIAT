from PyQt5.QtWidgets import (
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QListWidget,
    QPushButton,
    QMenu,
    QDialog,
    QFormLayout,
    QSpinBox,
    QTextEdit,
    QDialogButtonBox,
    QListWidgetItem,
    QGroupBox,
    QGridLayout,
    QLineEdit,
    QFrame,
    QVBoxLayout,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
)
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QColor,QIntValidator,QDoubleValidator


from PyQt5.QtCore import Qt, QRect



class SelectAllLineEdit(QLineEdit):
    """A QLineEdit that automatically selects all text when clicked"""

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        self.selectAll()

class AnnotationItemWidget(QWidget):
    def __init__(self, annotation, parent=None):
        super().__init__(parent)
        self.annotation = annotation
        self.parent_dock = parent
        self.attribute_inputs = {}  # Store references to attribute input widgets
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Annotation class name and label
        header_layout = QHBoxLayout()
        class_label = QLabel(f"{self.annotation.class_name}")
        class_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(class_label)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Attributes grid
        attributes_layout = QGridLayout()
        attributes_layout.setContentsMargins(0, 0, 0, 0)

        # Get class attribute configuration if available
        class_attributes = {}
        if hasattr(self.parent_dock, "main_window") and hasattr(self.parent_dock.main_window.canvas, "class_attributes"):
            class_attributes = self.parent_dock.main_window.canvas.class_attributes.get(self.annotation.class_name, {})

        # First, collect all attribute names from both class definition and annotation
        all_attributes = set(self.annotation.attributes.keys())
        
        # Add all attributes defined for this class
        if class_attributes:
            all_attributes.update(class_attributes.keys())
        
        # Add all attributes from the annotation and class definition
        row = 0
        for attr_name in sorted(all_attributes):
            # Get current attribute value (default to None if not present)
            attr_value = self.annotation.attributes.get(attr_name, None)
            
            # If attribute doesn't exist in annotation but exists in class definition,
            # initialize it with the default value from class definition
            if attr_value is None and attr_name in class_attributes:
                attr_config = class_attributes[attr_name]
                attr_value = attr_config.get("default", None)
                # Add the attribute to the annotation
                self.annotation.attributes[attr_name] = attr_value
            
            # Skip if we still don't have a value
            if attr_value is None:
                continue
                
            attr_label = QLabel(f"{attr_name}:")
            
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
                input_widget.currentTextChanged.connect(
                    lambda value, name=attr_name: self.update_boolean_attribute(name, value)
                )
            elif attr_type in ["int", "float"]:
                input_widget = SelectAllLineEdit()
                input_widget.setText(str(attr_value))
                input_widget.setPlaceholderText("0")
                
                # Set validator based on min/max if available
                if attr_min is not None and attr_max is not None:
                    if attr_type == "int":
                        input_widget.setValidator(QIntValidator(attr_min, attr_max))
                    else:
                        input_widget.setValidator(QDoubleValidator(attr_min, attr_max, 2))
                
                input_widget.textChanged.connect(
                    lambda text, name=attr_name, type=attr_type: self.update_numeric_attribute(name, text, type)
                )
            else:  # string or default
                input_widget = SelectAllLineEdit()
                input_widget.setText(str(attr_value))
                input_widget.textChanged.connect(
                    lambda text, name=attr_name: self.update_string_attribute(name, text)
                )
            
            attributes_layout.addWidget(attr_label, row, 0)
            attributes_layout.addWidget(input_widget, row, 1)
            
            # Store reference to input widget
            self.attribute_inputs[attr_name] = input_widget
            
            row += 1

        layout.addLayout(attributes_layout)

        # Add a separator line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

class AnnotationDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Annotations", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self.main_window = parent
        self.init_ui()

    def init_ui(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Class selector
        class_label = QLabel("Class:")
        self.class_selector = QComboBox()
        self.update_class_selector()
        self.class_selector.currentTextChanged.connect(self.on_class_selected)

        # Annotation list
        annotations_label = QLabel("Annotations:")
        self.annotations_list = QListWidget()
        self.annotations_list.setSelectionMode(QListWidget.SingleSelection)
        self.annotations_list.itemClicked.connect(self.on_annotation_selected)
        self.annotations_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.annotations_list.customContextMenuRequested.connect(self.show_context_menu)

        # Set a reasonable height for list items to accommodate the custom widgets
        self.annotations_list.setStyleSheet("QListWidget::item { min-height: 80px; }")

        # Annotation controls
        controls_layout = QHBoxLayout()
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self.add_annotation)
        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(self.delete_selected_annotation)

        # Add batch edit button
        batch_edit_btn = QPushButton("Batch Edit")
        batch_edit_btn.clicked.connect(self.batch_edit_annotations)
        controls_layout.addWidget(batch_edit_btn)

        controls_layout.addWidget(add_btn)
        controls_layout.addWidget(delete_btn)

        # Add widgets to layout
        layout.addWidget(class_label)
        layout.addWidget(self.class_selector)
        layout.addWidget(annotations_label)
        layout.addWidget(self.annotations_list)
        layout.addLayout(controls_layout)

        # Set the widget as the dock's widget
        self.setWidget(widget)

    def update_annotation_list(self):
        """Update the annotation list with current frame's annotations."""
        self.annotations_list.clear()

        # Get current frame
        current_frame = self.main_window.current_frame

        # Check if frame_annotations exists and has entries for the current frame
        if (
            hasattr(self.main_window, "frame_annotations")
            and current_frame in self.main_window.frame_annotations
        ):
            # Add annotations for the current frame to the list
            for annotation in self.main_window.frame_annotations[current_frame]:
                item = QListWidgetItem()
                annotation_widget = AnnotationItemWidget(annotation, self)
                item.setSizeHint(annotation_widget.sizeHint())
                self.annotations_list.addItem(item)
                self.annotations_list.setItemWidget(item, annotation_widget)

    def on_annotation_selected(self, item):
        """Handle selection of an annotation in the list"""
        widget = self.annotations_list.itemWidget(item)
        if widget and hasattr(widget, "annotation"):
            annotation = widget.annotation
            # Select this annotation on the canvas
            self.main_window.canvas.select_annotation(annotation)

    def update_class_selector(self):
        """Update the class selector with available classes"""
        self.class_selector.clear()
        if hasattr(self.main_window, "canvas"):
            self.class_selector.addItems(self.main_window.canvas.class_colors.keys())

    def on_class_selected(self, class_name):
        """Handle selection of a class"""
        if class_name and hasattr(self.main_window, "canvas"):
            self.main_window.canvas.set_current_class(class_name)

    def add_annotation(self):
        """Add a new annotation with the current class"""
        if hasattr(self.main_window, "add_empty_annotation"):
            self.main_window.add_empty_annotation()

    def delete_selected_annotation(self):
        """Delete the selected annotation"""
        selected_items = self.annotations_list.selectedItems()
        if selected_items:
            item = selected_items[0]
            widget = self.annotations_list.itemWidget(item)
            if widget and hasattr(widget, "annotation"):
                annotation = widget.annotation
                if hasattr(self.main_window, "delete_annotation"):
                    self.main_window.delete_annotation(annotation)

    def show_context_menu(self, position):
        """Show context menu for the selected annotation"""
        selected_items = self.annotations_list.selectedItems()
        if not selected_items:
            return

        item = selected_items[0]
        widget = self.annotations_list.itemWidget(item)
        if widget and hasattr(widget, "annotation"):
            annotation = widget.annotation

            menu = QMenu()
            delete_action = menu.addAction("Delete")

            action = menu.exec_(self.annotations_list.mapToGlobal(position))

            if action == delete_action:
                self.delete_selected_annotation()

    def batch_edit_annotations(self):
        """Open a dialog to edit all annotation attributes at once"""
        # Check if there are any annotations in any frame, not just the current frame
        if (
            not hasattr(self.main_window, "frame_annotations")
            or not self.main_window.frame_annotations
        ):
            from PyQt5.QtWidgets import QMessageBox

            QMessageBox.information(
                self, "Batch Edit", "No annotations found in any frame."
            )
            return

        # Create and show the batch edit dialog
        dialog = self.create_batch_edit_dialog()
        if dialog.exec_() == QDialog.Accepted:
            # Get the values from the dialog
            size_value = dialog.size_spin.value()
            quality_value = dialog.quality_spin.value()
            apply_to_all_frames = dialog.apply_all_frames_checkbox.isChecked()

            # Apply changes based on user selection
            if apply_to_all_frames:
                self.apply_attributes_to_all_frames(size_value, quality_value)
            else:
                self.apply_attributes_to_current_frame(size_value, quality_value)

            # Update the UI
            self.update_annotation_list()
            self.main_window.canvas.update()

    def create_batch_edit_dialog(self):
        """Create a dialog for batch editing annotation attributes"""

        dialog = QDialog(self)
        dialog.setWindowTitle("Batch Edit Annotations")
        dialog.setMinimumWidth(350)

        layout = QVBoxLayout(dialog)
        form_layout = QFormLayout()

        # Collect all unique attributes across all annotations in all frames
        all_attributes = set()
        attribute_types = {}  # Store attribute types

        for frame_annotations in self.main_window.frame_annotations.values():
            for annotation in frame_annotations:
                for attr_name in annotation.attributes.keys():
                    all_attributes.add(attr_name)

                    # Try to determine attribute type if not already known
                    if attr_name not in attribute_types:
                        attr_value = annotation.attributes[attr_name]
                        if isinstance(attr_value, bool):
                            attribute_types[attr_name] = "boolean"
                        elif isinstance(attr_value, int):
                            attribute_types[attr_name] = "int"
                        elif isinstance(attr_value, float):
                            attribute_types[attr_name] = "float"
                        else:
                            attribute_types[attr_name] = "string"

        # Also check class attribute configurations
        if hasattr(self.main_window.canvas, "class_attributes"):
            for class_config in self.main_window.canvas.class_attributes.values():
                for attr_name, attr_config in class_config.items():
                    all_attributes.add(attr_name)
                    attribute_types[attr_name] = attr_config.get("type", "string")

        # Create input widgets for all attributes
        dialog.attribute_widgets = {}

        for attr_name in sorted(all_attributes):
            attr_type = attribute_types.get(attr_name, "string")

            if attr_type == "boolean":
                input_widget = QComboBox()
                input_widget.addItems(["No Change", "False", "True"])
                input_widget.setCurrentText("No Change")
            elif attr_type == "int":
                input_widget = QSpinBox()
                input_widget.setRange(-999999, 999999)
                input_widget.setValue(0)
                input_widget.setSpecialValueText("No Change")  # 0 means no change
                input_widget.setProperty("no_change_value", 0)
            elif attr_type == "float":
                input_widget = QDoubleSpinBox()
                input_widget.setRange(-999999.0, 999999.0)
                input_widget.setValue(0.0)
                input_widget.setSpecialValueText("No Change")  # 0.0 means no change
                input_widget.setProperty("no_change_value", 0.0)
                input_widget.setDecimals(2)
            else:  # string
                input_widget = QLineEdit()
                input_widget.setPlaceholderText("No Change (leave empty)")

            form_layout.addRow(f"{attr_name}:", input_widget)
            dialog.attribute_widgets[attr_name] = (input_widget, attr_type)

        # Add form layout to main layout
        layout.addLayout(form_layout)

        # Option to apply to all frames - checked by default
        dialog.apply_all_frames_checkbox = QCheckBox("Apply to all frames")
        dialog.apply_all_frames_checkbox.setChecked(True)
        layout.addWidget(dialog.apply_all_frames_checkbox)

        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        return dialog

    def apply_batch_attributes_to_current_frame(self, attribute_values):
        """Apply attribute changes to all annotations in the current frame"""
        current_frame = self.main_window.current_frame

        # If there are no annotations for the current frame, create an empty list
        if current_frame not in self.main_window.frame_annotations:
            self.main_window.frame_annotations[current_frame] = []

        annotations = self.main_window.frame_annotations[current_frame]

        for annotation in annotations:
            # Update each attribute if it should be changed
            for attr_name, attr_value in attribute_values.items():
                annotation.attributes[attr_name] = attr_value

        # Update the canvas annotations if they're from the current frame
        self.main_window.canvas.annotations = annotations.copy()

    def apply_batch_attributes_to_all_frames(self, attribute_values):
        """Apply attribute changes to all annotations in all frames"""
        for frame_num, annotations in self.main_window.frame_annotations.items():
            for annotation in annotations:
                # Update each attribute if it should be changed
                for attr_name, attr_value in attribute_values.items():
                    annotation.attributes[attr_name] = attr_value

        # Update the current frame's annotations on the canvas
        current_frame = self.main_window.current_frame
        if current_frame in self.main_window.frame_annotations:
            self.main_window.canvas.annotations = self.main_window.frame_annotations[
                current_frame
            ].copy()
