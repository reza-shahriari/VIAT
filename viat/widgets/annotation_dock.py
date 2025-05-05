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
    QRadioButton,
    QProgressBar,
    QApplication,
)
from PyQt5.QtCore import Qt, QRect,QTimer
from PyQt5.QtGui import QColor, QIntValidator, QDoubleValidator,QPainter





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
        self.selected = False

    def paintEvent(self, event):
        """Override paint event to draw selection highlight"""
        super().paintEvent(event)
        if self.selected:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 0, 255, 80))  # Semi-transparent blue
            painter.drawRoundedRect(self.rect(), 5, 5)

    def init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # Annotation class name and label
        header_layout = QHBoxLayout()
        self.class_label = QLabel(f"{self.annotation.class_name}")
        self.class_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(self.class_label)
        header_layout.addStretch()
        layout.addLayout(header_layout)

        # Attributes grid
        attributes_layout = QGridLayout()
        attributes_layout.setContentsMargins(0, 0, 0, 0)

        # Get class attribute configuration if available
        class_attributes = {}
        if hasattr(self.parent_dock, "main_window") and hasattr(
            self.parent_dock.main_window.canvas, "class_attributes"
        ):
            class_attributes = self.parent_dock.main_window.canvas.class_attributes.get(
                self.annotation.class_name, {}
            )

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
                    lambda value, name=attr_name: self.update_boolean_attribute(
                        name, value
                    )
                )
            elif attr_type in ["int", "float"]:
                input_widget = SelectAllLineEdit()
                input_widget.setText(str(attr_value))
                input_widget.setPlaceholderText("0")

                # Set validator based on min/max if available
                if attr_min is not None and attr_max is not None:
                    if attr_type == "int":
                        if attr_min is not None and attr_max is not None:
                            try:
                                min_val = int(attr_min)
                                max_val = int(attr_max)
                                input_widget.setValidator(QIntValidator(min_val, max_val))
                            except (ValueError, TypeError):
                                # Handle case where conversion fails
                                print(f"Warning: Could not convert min/max values to integers: {attr_min}, {attr_max}")
                    else:
                        input_widget.setValidator(
                            QDoubleValidator(attr_min, attr_max, 2)
                        )

                input_widget.textChanged.connect(
                    lambda text, name=attr_name, type=attr_type: self.update_numeric_attribute(
                        name, text, type
                    )
                )
            else:  # string or default
                input_widget = SelectAllLineEdit()
                input_widget.setText(str(attr_value))
                input_widget.textChanged.connect(
                    lambda text, name=attr_name: self.update_string_attribute(
                        name, text
                    )
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
    def set_selected(self, selected):
        """Set the selection state of this widget"""
        self.is_selected = selected
        if selected:
            self.setStyleSheet("background-color: #3399ff; color: white; border-radius: 3px;")
            self.class_label.setStyleSheet("font-weight: bold; color: white;")
        else:
            self.setStyleSheet("")
            self.class_label.setStyleSheet("font-weight: bold;")
        self.update()

    def update_numeric_attribute(self, name, text, attr_type):
        """Update a numeric attribute value"""
        if not text:
            # Handle empty text
            self.annotation.attributes[name] = 0
            return

        try:
            if attr_type == "int":
                value = int(text)
            else:  # float
                value = float(text)

            self.annotation.attributes[name] = value

            # Update canvas if needed
            if hasattr(self.parent_dock, "main_window") and hasattr(
                self.parent_dock.main_window, "canvas"
            ):
                self.parent_dock.main_window.canvas.update()
        except ValueError:
            # Invalid numeric input, revert to previous value
            if name in self.attribute_inputs:
                current_value = self.annotation.attributes.get(name, 0)
                self.attribute_inputs[name].setText(str(current_value))

    def update_boolean_attribute(self, name, value):
        """Update a boolean attribute value"""
        bool_value = value == "True"
        self.annotation.attributes[name] = bool_value

        # Update canvas if needed
        if hasattr(self.parent_dock, "main_window") and hasattr(
            self.parent_dock.main_window, "canvas"
        ):
            self.parent_dock.main_window.canvas.update()

    def update_string_attribute(self, name, text):
        """Update a string attribute value"""
        self.annotation.attributes[name] = text

        # Update canvas if needed
        if hasattr(self.parent_dock, "main_window") and hasattr(
            self.parent_dock.main_window, "canvas"
        ):
            self.parent_dock.main_window.canvas.update()


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
        # Defer the update to ensure canvas is ready
        QTimer.singleShot(100, self.update_class_selector)
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

    def update_class_selector(self):
        """Update the class selector with available classes"""
        # Store the current selection before clearing
        current_text = self.class_selector.currentText()
        
        # Clear the selector
        self.class_selector.clear()
        
        
        if hasattr(self.main_window, 'canvas'):
            if hasattr(self.main_window.canvas, 'class_colors') and self.main_window.canvas.class_colors:
                class_colors = self.main_window.canvas.class_colors
                
                # Use a list to maintain order and prevent duplicates
                class_list = list(dict.fromkeys(class_colors.keys()))
                self.class_selector.addItems(class_list)
                
                # Restore previous selection if it exists
                if current_text and self.class_selector.findText(current_text) >= 0:
                    self.class_selector.setCurrentText(current_text)
                # Otherwise select the current class if it exists
                elif hasattr(self.main_window.canvas, 'current_class'):
                    current_class = self.main_window.canvas.current_class
                    if current_class:
                        index = self.class_selector.findText(current_class)
                        if index >= 0:
                            self.class_selector.setCurrentIndex(index)


    def select_all_in_list(self):
        """Select all items in the annotation list."""
        if hasattr(self, "annotations_list"):
            for i in range(self.annotations_list.count()):
                item = self.annotations_list.item(i)
                item.setSelected(True)


    def on_class_selected(self, class_name):
        """Handle selection of a class"""
        if class_name and hasattr(self.main_window, "canvas"):
            # Block signals to prevent recursive calls
            self.main_window.canvas.blockSignals(True)
            self.main_window.canvas.set_current_class(class_name)
            self.main_window.canvas.blockSignals(False)
            
            # Update the canvas
            self.main_window.canvas.update()
            
            # Update class selector in class dock if it exists
            if hasattr(self.main_window, "class_dock") and hasattr(self.main_window.class_dock, "classes_list"):
                items = self.main_window.class_dock.classes_list.findItems(class_name, Qt.MatchExactly)
                if items:
                    self.main_window.class_dock.classes_list.setCurrentItem(items[0])
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
        self.update_annotation_list()

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
        # Clear all selections first
        for i in range(self.annotations_list.count()):
            current_item = self.annotations_list.item(i)
            widget = self.annotations_list.itemWidget(current_item)
            if hasattr(widget, "set_selected"):
                widget.set_selected(False)
        
        # Set the selected item
        widget = self.annotations_list.itemWidget(item)
        if widget and hasattr(widget, "annotation"):
            if hasattr(widget, "set_selected"):
                widget.set_selected(True)
            
            # Select this annotation on the canvas
            if hasattr(self.main_window, "canvas") and self.main_window.canvas:
                # Block signals to prevent recursive selection
                old_block_state = self.main_window.canvas.blockSignals(True)
                self.main_window.canvas.selected_annotation = widget.annotation
                self.main_window.canvas.update()
                self.main_window.canvas.blockSignals(old_block_state)


   
    def select_annotation_in_list(self, target_annotation):
        """
        Select the specified annotation in the list and scroll to it.
        
        Args:
            target_annotation: The annotation to select
        """
        if not target_annotation:
            # Clear all selections if no annotation is provided
            for i in range(self.annotations_list.count()):
                item = self.annotations_list.item(i)
                widget = self.annotations_list.itemWidget(item)
                if hasattr(widget, "set_selected"):
                    widget.set_selected(False)
            return
        
        # Find and select the matching annotation widget
        found = False
        for i in range(self.annotations_list.count()):
            item = self.annotations_list.item(i)
            widget = self.annotations_list.itemWidget(item)
            
            if not widget or not hasattr(widget, "annotation"):
                continue
                
            # Check if this is the annotation we're looking for
            is_match = False
            if widget.annotation is target_annotation:
                is_match = True
            elif (hasattr(widget.annotation, 'rect') and hasattr(target_annotation, 'rect') and
                widget.annotation.rect == target_annotation.rect and 
                widget.annotation.class_name == target_annotation.class_name):
                is_match = True
                
            # Set selection state
            if hasattr(widget, "set_selected"):
                widget.set_selected(is_match)
                
            if is_match:
                found = True
                # Scroll to make this item visible
                self.annotations_list.scrollToItem(item)
        
        return found


    def update_annotation_list(self):
        """Update the annotation list with current frame's annotations."""
        # Remember the currently selected annotation
        selected_annotation = None
        if hasattr(self.main_window, "canvas") and self.main_window.canvas:
            selected_annotation = self.main_window.canvas.selected_annotation
        
        # Clear the list
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
                
                # Set selection state
                if selected_annotation and (
                    annotation is selected_annotation or
                    (hasattr(annotation, 'rect') and hasattr(selected_annotation, 'rect') and
                    annotation.rect == selected_annotation.rect and 
                    annotation.class_name == selected_annotation.class_name)
                ):
                    annotation_widget.set_selected(True)

                      
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
        """Show dialog for batch editing annotations across frames."""
        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Batch Operations")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout(dialog)

        # Create operation type group
        operation_group = QGroupBox("Operation Type")
        operation_layout = QVBoxLayout(operation_group)
        
        edit_radio = QRadioButton("Edit Annotation Attributes")
        delete_radio = QRadioButton("Delete Annotations")
        edit_radio.setChecked(True)
        
        operation_layout.addWidget(edit_radio)
        operation_layout.addWidget(delete_radio)
        layout.addWidget(operation_group)

        # Add frame range selection
        range_group = QGroupBox("Frame Range")
        range_layout = QFormLayout(range_group)

        start_spin = QSpinBox()
        start_spin.setRange(0, self.main_window.total_frames - 1)
        start_spin.setValue(self.main_window.current_frame)

        end_spin = QSpinBox()
        end_spin.setRange(0, self.main_window.total_frames - 1)
        end_spin.setValue(
            min(self.main_window.current_frame + 10, self.main_window.total_frames - 1)
        )

        range_layout.addRow("Start Frame:", start_spin)
        range_layout.addRow("End Frame:", end_spin)
        layout.addWidget(range_group)

        # Add propagation options
        prop_group = QGroupBox("Apply To")
        prop_layout = QVBoxLayout(prop_group)

        all_frames_radio = QRadioButton("All frames in range")
        all_frames_radio.setChecked(True)

        duplicate_frames_radio = QRadioButton("Only duplicate frames")
        duplicate_frames_radio.setEnabled(self.main_window.duplicate_frames_enabled)

        similar_frames_radio = QRadioButton("Similar frames (experimental)")
        similar_frames_radio.setEnabled(False)  # Not implemented yet

        prop_layout.addWidget(all_frames_radio)
        prop_layout.addWidget(duplicate_frames_radio)
        prop_layout.addWidget(similar_frames_radio)
        layout.addWidget(prop_group)

        # Class filter for delete operation
        class_filter_group = QGroupBox("Filter by Class")
        class_filter_layout = QVBoxLayout(class_filter_group)
        
        class_filter = QComboBox()
        class_filter.addItem("All Classes")
        
        # Add all available classes
        if hasattr(self.main_window.canvas, 'class_colors'):
            for class_name in self.main_window.canvas.class_colors.keys():
                class_filter.addItem(class_name)
        
        class_filter_layout.addWidget(class_filter)
        layout.addWidget(class_filter_group)
        
        # Only show class filter for delete operation
        class_filter_group.setVisible(False)
        delete_radio.toggled.connect(class_filter_group.setVisible)

        # Create attribute editing section (only for edit operation)
        attributes_group = QGroupBox("Edit Attributes")
        attributes_layout = QFormLayout(attributes_group)
        
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

            attributes_layout.addRow(f"{attr_name}:", input_widget)
            dialog.attribute_widgets[attr_name] = input_widget

        layout.addWidget(attributes_group)
        
        # Only show attributes for edit operation
        attributes_group.setVisible(True)
        edit_radio.toggled.connect(attributes_group.setVisible)
        delete_radio.toggled.connect(lambda checked: attributes_group.setVisible(not checked))

        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            # Get frame range
            start_frame = start_spin.value()
            end_frame = end_spin.value()

            # Get propagation mode
            if duplicate_frames_radio.isChecked():
                prop_mode = "duplicate"
            elif similar_frames_radio.isChecked():
                prop_mode = "similar"
            else:
                prop_mode = "all"

            # Determine operation type
            if delete_radio.isChecked():
                # Get class filter
                class_name_filter = None
                if class_filter.currentText() != "All Classes":
                    class_name_filter = class_filter.currentText()
                
                # Apply batch delete
                self.apply_batch_delete(start_frame, end_frame, class_name_filter, prop_mode)
            else:
                # Get attribute values from dialog
                attribute_values = {}
                for attr_name, widget in dialog.attribute_widgets.items():
                    if isinstance(widget, QComboBox):
                        if widget.currentText() != "No Change":
                            attribute_values[attr_name] = widget.currentText() == "True"
                    elif isinstance(widget, QSpinBox):
                        if widget.text() != "No Change":
                            attribute_values[attr_name] = widget.value()
                    elif isinstance(widget, QDoubleSpinBox):
                        if widget.text() != "No Change":
                            attribute_values[attr_name] = widget.value()
                    elif isinstance(widget, QLineEdit) and widget.text():
                        attribute_values[attr_name] = widget.text()

                # Apply batch edit
                self.apply_batch_edit(start_frame, end_frame, attribute_values, prop_mode)

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

    def apply_batch_edit(
        self, start_frame, end_frame, attribute_values, prop_mode="all"
    ):
        """
        Apply batch edit to annotations across frames.

        Args:
            start_frame (int): Start frame number
            end_frame (int): End frame number
            attribute_values (dict): Dictionary of attribute name to value
            prop_mode (str): Propagation mode - "all", "duplicate", or "similar"
        """
        if start_frame > end_frame:
            start_frame, end_frame = end_frame, start_frame

        # Create progress dialog
        progress = QDialog(self)
        progress.setWindowTitle("Applying Batch Edit")
        progress.setFixedSize(300, 100)
        progress_layout = QVBoxLayout(progress)

        label = QLabel(f"Updating annotations in frames {start_frame}-{end_frame}...")
        progress_layout.addWidget(label)

        progress_bar = QProgressBar()
        progress_bar.setRange(start_frame, end_frame)
        progress_layout.addWidget(progress_bar)

        # Non-blocking progress dialog
        progress.setModal(False)
        progress.show()
        QApplication.processEvents()

        # Track processed frames for duplicate mode
        processed_hashes = set()
        update_count = 0

        # Apply edits to each frame
        for frame_num in range(start_frame, end_frame + 1):
            # Update progress
            progress_bar.setValue(frame_num)
            if frame_num % 5 == 0:  # Update UI every 5 frames
                QApplication.processEvents()

            # Skip frames without annotations
            if frame_num not in self.main_window.frame_annotations:
                continue

            # For duplicate mode, check if this is a duplicate frame
            if prop_mode == "duplicate":
                if frame_num not in self.main_window.frame_hashes:
                    continue

                frame_hash = self.main_window.frame_hashes[frame_num]

                # Skip if we've already processed a frame with this hash
                if frame_hash in processed_hashes:
                    continue

                # Skip if this isn't a duplicate frame
                if (
                    frame_hash not in self.main_window.duplicate_frames_cache
                    or len(self.main_window.duplicate_frames_cache[frame_hash]) <= 1
                ):
                    continue

                processed_hashes.add(frame_hash)

            # Update annotations in this frame
            for annotation in self.main_window.frame_annotations[frame_num]:
                # Update attributes
                for attr_name, attr_value in attribute_values.items():
                    annotation.attributes[attr_name] = attr_value

                update_count += 1

            # If this is the current frame, update the canvas
            if frame_num == self.main_window.current_frame:
                self.main_window.canvas.update()
                self.update_annotation_list()

        # Close progress dialog
        progress.close()

        self.main_window.statusBar.showMessage(
            f"Updated {update_count} annotations across {end_frame - start_frame + 1} frames",
            5000,
        )

        # Mark project as modified
        self.main_window.project_modified = True

    def apply_batch_delete(self, start_frame, end_frame, class_name_filter=None, prop_mode="all"):
        """
        Apply batch delete to annotations across frames.

        Args:
            start_frame (int): Start frame number
            end_frame (int): End frame number
            class_name_filter (str): Only delete annotations of this class (None for all)
            prop_mode (str): Propagation mode - "all", "duplicate", or "similar"
        """
        if start_frame > end_frame:
            start_frame, end_frame = end_frame, start_frame

        # Create progress dialog
        progress = QDialog(self)
        progress.setWindowTitle("Deleting Annotations")
        progress.setFixedSize(300, 100)
        progress_layout = QVBoxLayout(progress)

        label = QLabel(f"Deleting annotations in frames {start_frame}-{end_frame}...")
        progress_layout.addWidget(label)

        progress_bar = QProgressBar()
        progress_bar.setRange(start_frame, end_frame)
        progress_layout.addWidget(progress_bar)

        # Non-blocking progress dialog
        progress.setModal(False)
        progress.show()
        QApplication.processEvents()

        # Track processed frames for duplicate mode
        processed_hashes = set()
        delete_count = 0

        # Apply deletes to each frame
        for frame_num in range(start_frame, end_frame + 1):
            # Update progress
            progress_bar.setValue(frame_num)
            if frame_num % 5 == 0:  # Update UI every 5 frames
                QApplication.processEvents()

            # Skip frames without annotations
            if frame_num not in self.main_window.frame_annotations:
                continue

            # For duplicate mode, check if this is a duplicate frame
            if prop_mode == "duplicate":
                if frame_num not in self.main_window.frame_hashes:
                    continue

                frame_hash = self.main_window.frame_hashes[frame_num]

                # Skip if we've already processed a frame with this hash
                if frame_hash in processed_hashes:
                    continue

                # Skip if this isn't a duplicate frame
                if (
                    frame_hash not in self.main_window.duplicate_frames_cache
                    or len(self.main_window.duplicate_frames_cache[frame_hash]) <= 1
                ):
                    continue

                processed_hashes.add(frame_hash)

            # Delete matching annotations in this frame
            annotations_to_keep = []
            for annotation in self.main_window.frame_annotations[frame_num]:
                if class_name_filter is None or annotation.class_name == class_name_filter:
                    delete_count += 1
                else:
                    annotations_to_keep.append(annotation)
        
            # Update the frame annotations
            self.main_window.frame_annotations[frame_num] = annotations_to_keep

            # If this is the current frame, update the canvas
            if frame_num == self.main_window.current_frame:
                self.main_window.canvas.annotations = annotations_to_keep.copy()
                self.main_window.canvas.update()
                self.update_annotation_list()

        # Close progress dialog
        progress.close()

        self.main_window.statusBar.showMessage(
            f"Deleted {delete_count} annotations across {end_frame - start_frame + 1} frames",
            5000,
        )

        # Mark project as modified
        self.main_window.project_modified = True
