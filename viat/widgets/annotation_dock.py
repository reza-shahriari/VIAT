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
    QMessageBox,
    QButtonGroup,
)
from PyQt5.QtCore import Qt, QRect, QTimer
from PyQt5.QtGui import QColor, QIntValidator, QDoubleValidator, QPainter


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
                                input_widget.setValidator(
                                    QIntValidator(min_val, max_val)
                                )
                            except (ValueError, TypeError):
                                # Handle case where conversion fails
                                print(
                                    f"Warning: Could not convert min/max values to integers: {attr_min}, {attr_max}"
                                )
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
            self.setStyleSheet(
                "background-color: #3399ff; color: white; border-radius: 3px;"
            )
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

        if hasattr(self.main_window, "canvas"):
            if (
                hasattr(self.main_window.canvas, "class_colors")
                and self.main_window.canvas.class_colors
            ):
                class_colors = self.main_window.canvas.class_colors

                # Use a list to maintain order and prevent duplicates
                class_list = list(dict.fromkeys(class_colors.keys()))
                self.class_selector.addItems(class_list)

                # Restore previous selection if it exists
                if current_text and self.class_selector.findText(current_text) >= 0:
                    self.class_selector.setCurrentText(current_text)
                # Otherwise select the current class if it exists
                elif hasattr(self.main_window.canvas, "current_class"):
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
            if hasattr(self.main_window, "class_dock") and hasattr(
                self.main_window.class_dock, "classes_list"
            ):
                items = self.main_window.class_dock.classes_list.findItems(
                    class_name, Qt.MatchExactly
                )
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
            elif (
                hasattr(widget.annotation, "rect")
                and hasattr(target_annotation, "rect")
                and widget.annotation.rect == target_annotation.rect
                and widget.annotation.class_name == target_annotation.class_name
            ):
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
                    annotation is selected_annotation
                    or (
                        hasattr(annotation, "rect")
                        and hasattr(selected_annotation, "rect")
                        and annotation.rect == selected_annotation.rect
                        and annotation.class_name == selected_annotation.class_name
                    )
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
        """Open dialog to batch edit annotations across multiple frames."""
        # Create dialog
        dialog = QDialog(self.main_window)
        dialog.setWindowTitle("Batch Edit Annotations")
        dialog.setMinimumWidth(400)

        layout = QVBoxLayout(dialog)

        # Frame range selection
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

        # Action selection
        action_group = QGroupBox("Action")
        action_layout = QVBoxLayout(action_group)

        # Radio buttons for different actions
        action_radios = QButtonGroup(dialog)

        # Change class radio
        change_class_radio = QRadioButton("Change class")
        action_radios.addButton(change_class_radio)
        action_layout.addWidget(change_class_radio)

        # Delete class radio
        delete_class_radio = QRadioButton("Delete annotations of class")
        action_radios.addButton(delete_class_radio)
        action_layout.addWidget(delete_class_radio)

        # Modify attributes radio
        modify_attr_radio = QRadioButton("Modify attributes of a class")
        action_radios.addButton(modify_attr_radio)
        action_layout.addWidget(modify_attr_radio)

        # Use current frame annotations radio
        use_current_radio = QRadioButton("Use current frame annotations as template")
        action_radios.addButton(use_current_radio)
        action_layout.addWidget(use_current_radio)

        # Set default selection
        change_class_radio.setChecked(True)

        # Class selection for relevant actions
        class_group = QGroupBox("Class")
        class_layout = QVBoxLayout(class_group)

        # From class (for change class)
        from_class_combo = QComboBox()
        class_list = sorted(self.main_window.canvas.class_colors.keys())
        from_class_combo.addItems(class_list)
        from_class_label = QLabel("From class:")
        class_layout.addWidget(from_class_label)
        class_layout.addWidget(from_class_combo)

        # To class (for change class)
        to_class_combo = QComboBox()
        to_class_combo.addItems(class_list)
        to_class_label = QLabel("To class:")
        class_layout.addWidget(to_class_label)
        class_layout.addWidget(to_class_combo)

        # Class for delete and modify attributes
        target_class_combo = QComboBox()
        # Add "ALL" option for delete
        target_class_list = ["ALL"] + class_list
        target_class_combo.addItems(target_class_list)
        target_class_label = QLabel("Target class:")
        class_layout.addWidget(target_class_label)
        class_layout.addWidget(target_class_combo)

        # Attribute editing (for modify attributes)
        attr_group = QGroupBox("Attributes")
        attr_layout = QVBoxLayout(attr_group)

        # Create a form layout for attributes
        attr_form = QFormLayout()
        attr_widgets = {}  # Store attribute widgets for access later

        # Function to update attribute form based on selected class
        def update_attribute_form():
            # Clear existing widgets
            while attr_form.rowCount() > 0:
                attr_form.removeRow(0)

            attr_widgets.clear()

            selected_class = target_class_combo.currentText()
            if selected_class == "ALL":
                # Show message that ALL is not valid for attribute modification
                attr_form.addRow(
                    QLabel("Cannot modify attributes for ALL classes at once.")
                )
                attr_form.addRow(QLabel("Please select a specific class."))
                return

            # Get all attributes used by this class across all frames
            all_attributes = set()
            for frame_anns in self.main_window.frame_annotations.values():
                for ann in frame_anns:
                    if ann.class_name == selected_class and hasattr(ann, "attributes"):
                        all_attributes.update(ann.attributes.keys())

            # If no attributes found
            if not all_attributes:
                attr_form.addRow(
                    QLabel(f"No attributes found for class '{selected_class}'.")
                )
                return

            # Add widgets for each attribute
            for attr_name in sorted(all_attributes):
                attr_value_edit = QLineEdit()
                attr_widgets[attr_name] = attr_value_edit
                attr_form.addRow(f"{attr_name}:", attr_value_edit)

        # Connect class selection to attribute form update
        target_class_combo.currentTextChanged.connect(update_attribute_form)

        # Add the form to the layout
        attr_layout.addLayout(attr_form)

        # Options for using current frame annotations
        current_options_group = QGroupBox("Current Frame Options")
        current_options_layout = QVBoxLayout(current_options_group)

        overwrite_check = QCheckBox("Overwrite existing annotations")
        overwrite_check.setChecked(False)
        current_options_layout.addWidget(overwrite_check)

        keep_existing_check = QCheckBox("Keep existing annotations (add new ones)")
        keep_existing_check.setChecked(True)
        current_options_layout.addWidget(keep_existing_check)

        current_options_group.setVisible(False)  # Initially hidden

        # Connect radio buttons to show/hide relevant groups
        def update_visible_groups():
            # Class group visibility - only show for class-specific operations
            class_group.setVisible(change_class_radio.isChecked() or 
                                delete_class_radio.isChecked() or 
                                modify_attr_radio.isChecked())
            
            # Show/hide specific class selection components
            from_class_label.setVisible(change_class_radio.isChecked())
            from_class_combo.setVisible(change_class_radio.isChecked())
            to_class_label.setVisible(change_class_radio.isChecked())
            to_class_combo.setVisible(change_class_radio.isChecked())
            
            target_class_label.setVisible(delete_class_radio.isChecked() or modify_attr_radio.isChecked())
            target_class_combo.setVisible(delete_class_radio.isChecked() or modify_attr_radio.isChecked())
            
            # Attribute group visibility
            attr_group.setVisible(modify_attr_radio.isChecked())
            
            # Current frame options visibility
            current_options_group.setVisible(use_current_radio.isChecked())
            
            # Update attribute form if needed
            if modify_attr_radio.isChecked():
                update_attribute_form()


        change_class_radio.toggled.connect(update_visible_groups)
        delete_class_radio.toggled.connect(update_visible_groups)
        modify_attr_radio.toggled.connect(update_visible_groups)
        use_current_radio.toggled.connect(update_visible_groups)

        # Initial update
        update_visible_groups()

        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        # Add widgets to layout
        layout.addWidget(range_group)
        layout.addWidget(action_group)
        layout.addWidget(class_group)
        layout.addWidget(attr_group)
        layout.addWidget(current_options_group)
        layout.addWidget(buttons)

        # Show dialog
        if dialog.exec_() == QDialog.Accepted:
            # Save undo state before batch edit
            self.main_window.save_undo_state()

            # Get frame range
            start_frame = start_spin.value()
            end_frame = end_spin.value()

            # Validate range
            if start_frame > end_frame:
                start_frame, end_frame = end_frame, start_frame

            # Process based on selected action
            if change_class_radio.isChecked():
                # Change class
                from_class = from_class_combo.currentText()
                to_class = to_class_combo.currentText()
                self.batch_change_class(start_frame, end_frame, from_class, to_class)

            elif delete_class_radio.isChecked():
                # Delete annotations of class
                class_to_delete = target_class_combo.currentText()
                self.batch_delete_class(start_frame, end_frame, class_to_delete)

            elif modify_attr_radio.isChecked():
                # Modify attributes of a class
                target_class = target_class_combo.currentText()
                if target_class == "ALL":
                    QMessageBox.warning(
                        self.main_window,
                        "Invalid Selection",
                        "Cannot modify attributes for ALL classes at once. Please select a specific class.",
                    )
                    return

                # Collect attribute values
                attributes = {}
                for attr_name, widget in attr_widgets.items():
                    value = widget.text()
                    if value:  # Only include non-empty values
                        # Try to convert to int or float if possible
                        if value.isdigit():
                            value = int(value)
                        else:
                            try:
                                value = float(value)
                            except ValueError:
                                # Keep as string if not a number
                                pass
                        attributes[attr_name] = value

                if attributes:
                    self.batch_modify_class_attributes(
                        start_frame, end_frame, target_class, attributes
                    )
                else:
                    QMessageBox.warning(
                        self.main_window,
                        "No Attributes",
                        "No attribute values were provided for modification.",
                    )

            elif use_current_radio.isChecked():
                # Use current frame annotations as template
                overwrite = overwrite_check.isChecked()
                keep_existing = keep_existing_check.isChecked()
                self.batch_use_current_frame(
                    start_frame, end_frame, overwrite, keep_existing
                )

            # Update UI if we're on one of the affected frames
            if start_frame <= self.main_window.current_frame <= end_frame:
                self.main_window.load_current_frame_annotations()

            self.main_window.statusBar.showMessage(
                f"Batch edit completed for frames {start_frame}-{end_frame}", 5000
            )

    def batch_delete_class(self, start_frame, end_frame, class_name):
        """
        Delete annotations of a specific class across a range of frames.

        Args:
            start_frame (int): Start frame number
            end_frame (int): End frame number
            class_name (str): Class name to delete, or "ALL" to delete all annotations
        """
        # Create progress dialog
        progress = QDialog(self.main_window)
        progress.setWindowTitle("Deleting Annotations")
        progress.setFixedSize(300, 100)
        progress_layout = QVBoxLayout(progress)

        label = QLabel(
            f"Deleting {'all annotations' if class_name == 'ALL' else f'annotations of class {class_name}'} "
            f"from frames {start_frame}-{end_frame}..."
        )
        progress_layout.addWidget(label)

        progress_bar = QProgressBar()
        progress_bar.setRange(start_frame, end_frame)
        progress_layout.addWidget(progress_bar)

        # Non-blocking progress dialog
        progress.setModal(False)
        progress.show()
        QApplication.processEvents()

        # Delete annotations
        deleted_count = 0
        for frame_num in range(start_frame, end_frame + 1):
            # Update progress
            progress_bar.setValue(frame_num)
            if frame_num % 5 == 0:  # Update UI every 5 frames
                QApplication.processEvents()

            if frame_num in self.main_window.frame_annotations:
                if class_name == "ALL":
                    # Delete all annotations
                    deleted_count += len(self.main_window.frame_annotations[frame_num])
                    self.main_window.frame_annotations[frame_num] = []
                else:
                    # Delete only annotations of the specified class
                    original_count = len(self.main_window.frame_annotations[frame_num])
                    self.main_window.frame_annotations[frame_num] = [
                        ann
                        for ann in self.main_window.frame_annotations[frame_num]
                        if ann.class_name != class_name
                    ]
                    deleted_count += original_count - len(
                        self.main_window.frame_annotations[frame_num]
                    )

        # Close progress dialog
        progress.close()

        # Show result
        self.main_window.statusBar.showMessage(
            f"Deleted {deleted_count} annotations from frames {start_frame}-{end_frame}",
            5000,
        )

    def batch_modify_class_attributes(
        self, start_frame, end_frame, class_name, attributes
    ):
        """
        Modify attributes of a specific class across a range of frames.

        Args:
            start_frame (int): Start frame number
            end_frame (int): End frame number
            class_name (str): Class name to modify
            attributes (dict): Attributes to set/modify
        """
        # Create progress dialog
        progress = QDialog(self.main_window)
        progress.setWindowTitle("Modifying Attributes")
        progress.setFixedSize(300, 100)
        progress_layout = QVBoxLayout(progress)

        label = QLabel(
            f"Modifying attributes for class {class_name} in frames {start_frame}-{end_frame}..."
        )
        progress_layout.addWidget(label)

        progress_bar = QProgressBar()
        progress_bar.setRange(start_frame, end_frame)
        progress_layout.addWidget(progress_bar)

        # Non-blocking progress dialog
        progress.setModal(False)
        progress.show()
        QApplication.processEvents()

        # Modify attributes
        modified_count = 0
        for frame_num in range(start_frame, end_frame + 1):
            # Update progress
            progress_bar.setValue(frame_num)
            if frame_num % 5 == 0:  # Update UI every 5 frames
                QApplication.processEvents()

            if frame_num in self.main_window.frame_annotations:
                for ann in self.main_window.frame_annotations[frame_num]:
                    if ann.class_name == class_name:
                        # Initialize attributes if not present
                        if not hasattr(ann, "attributes"):
                            ann.attributes = {}

    def batch_modify_class_attributes(
        self, start_frame, end_frame, class_name, attributes
    ):
        """
        Modify attributes of a specific class across a range of frames.

        Args:
            start_frame (int): Start frame number
            end_frame (int): End frame number
            class_name (str): Class name to modify
            attributes (dict): Attributes to set/modify
        """
        # Create progress dialog
        progress = QDialog(self.main_window)
        progress.setWindowTitle("Modifying Attributes")
        progress.setFixedSize(300, 100)
        progress_layout = QVBoxLayout(progress)

        label = QLabel(
            f"Modifying attributes for class {class_name} in frames {start_frame}-{end_frame}..."
        )
        progress_layout.addWidget(label)

        progress_bar = QProgressBar()
        progress_bar.setRange(start_frame, end_frame)
        progress_layout.addWidget(progress_bar)

        # Non-blocking progress dialog
        progress.setModal(False)
        progress.show()
        QApplication.processEvents()

        # Modify attributes
        modified_count = 0
        for frame_num in range(start_frame, end_frame + 1):
            # Update progress
            progress_bar.setValue(frame_num)
            if frame_num % 5 == 0:  # Update UI every 5 frames
                QApplication.processEvents()

            if frame_num in self.main_window.frame_annotations:
                for ann in self.main_window.frame_annotations[frame_num]:
                    if ann.class_name == class_name:
                        # Initialize attributes if not present
                        if not hasattr(ann, "attributes"):
                            ann.attributes = {}

                        # Update attributes
                        for attr_name, attr_value in attributes.items():
                            ann.attributes[attr_name] = attr_value

                        modified_count += 1

        # Close progress dialog
        progress.close()

        # Show result
        self.main_window.statusBar.showMessage(
            f"Modified attributes for {modified_count} annotations of class {class_name}",
            5000,
        )

    def batch_use_current_frame(self, start_frame, end_frame, overwrite, keep_existing):
        """
        Apply current frame's annotations to a range of frames.

        Args:
            start_frame (int): Start frame number
            end_frame (int): End frame number
            overwrite (bool): Whether to overwrite existing annotations
            keep_existing (bool): Whether to keep existing annotations
        """
        # Get current frame annotations
        current_frame = self.main_window.current_frame
        if (
            current_frame not in self.main_window.frame_annotations
            or not self.main_window.frame_annotations[current_frame]
        ):
            QMessageBox.warning(
                self.main_window,
                "No Annotations",
                "Current frame has no annotations to use as template.",
            )
            return

        current_annotations = [
            self.main_window.clone_annotation(ann)
            for ann in self.main_window.frame_annotations[current_frame]
        ]

        # Create progress dialog
        progress = QDialog(self.main_window)
        progress.setWindowTitle("Applying Annotations")
        progress.setFixedSize(300, 100)
        progress_layout = QVBoxLayout(progress)

        label = QLabel(f"Applying annotations to frames {start_frame}-{end_frame}...")
        progress_layout.addWidget(label)

        progress_bar = QProgressBar()
        progress_bar.setRange(start_frame, end_frame)
        progress_layout.addWidget(progress_bar)

        # Non-blocking progress dialog
        progress.setModal(False)
        progress.show()
        QApplication.processEvents()

        # Apply to each frame in range
        frames_modified = 0
        for frame_num in range(start_frame, end_frame + 1):
            # Skip current frame
            if frame_num == current_frame:
                continue

            # Update progress
            progress_bar.setValue(frame_num)
            if frame_num % 5 == 0:  # Update UI every 5 frames
                QApplication.processEvents()

            # Handle existing annotations based on options
            if (
                frame_num in self.main_window.frame_annotations
                and self.main_window.frame_annotations[frame_num]
            ):
                if overwrite:
                    # Replace all annotations
                    self.main_window.frame_annotations[frame_num] = [
                        self.main_window.clone_annotation(ann)
                        for ann in current_annotations
                    ]
                    frames_modified += 1
                elif keep_existing:
                    # Add new annotations without removing existing ones
                    existing_annotations = self.main_window.frame_annotations[frame_num]
                    self.main_window.frame_annotations[frame_num] = (
                        existing_annotations
                        + [
                            self.main_window.clone_annotation(ann)
                            for ann in current_annotations
                        ]
                    )
                    frames_modified += 1
            else:
                # No existing annotations, just add the current ones
                self.main_window.frame_annotations[frame_num] = [
                    self.main_window.clone_annotation(ann)
                    for ann in current_annotations
                ]
                frames_modified += 1

        # Close progress dialog
        progress.close()

        # Show result
        annotation_count = len(current_annotations)
        self.main_window.statusBar.showMessage(
            f"Applied {annotation_count} annotations from current frame to {frames_modified} frames",
            5000,
        )

    def batch_change_class(self, start_frame, end_frame, from_class, to_class):
        """
        Change class of annotations across a range of frames.

        Args:
            start_frame (int): Start frame number
            end_frame (int): End frame number
            from_class (str): Original class name
            to_class (str): New class name
        """
        # Create progress dialog
        progress = QDialog(self.main_window)
        progress.setWindowTitle("Changing Class")
        progress.setFixedSize(300, 100)
        progress_layout = QVBoxLayout(progress)

        label = QLabel(
            f"Changing class from {from_class} to {to_class} in frames {start_frame}-{end_frame}..."
        )
        progress_layout.addWidget(label)

        progress_bar = QProgressBar()
        progress_bar.setRange(start_frame, end_frame)
        progress_layout.addWidget(progress_bar)

        # Non-blocking progress dialog
        progress.setModal(False)
        progress.show()
        QApplication.processEvents()

        # Change class
        changed_count = 0
        for frame_num in range(start_frame, end_frame + 1):
            # Update progress
            progress_bar.setValue(frame_num)
            if frame_num % 5 == 0:  # Update UI every 5 frames
                QApplication.processEvents()

            if frame_num in self.main_window.frame_annotations:
                for ann in self.main_window.frame_annotations[frame_num]:
                    if ann.class_name == from_class:
                        ann.class_name = to_class
                        changed_count += 1

        # Close progress dialog
        progress.close()

        # Show result
        self.main_window.statusBar.showMessage(
            f"Changed {changed_count} annotations from class {from_class} to {to_class}",
            5000,
        )

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

    def apply_batch_delete(
        self, start_frame, end_frame, class_name_filter=None, prop_mode="all"
    ):
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
                if (
                    class_name_filter is None
                    or annotation.class_name == class_name_filter
                ):
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
