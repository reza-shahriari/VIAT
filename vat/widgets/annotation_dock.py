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
)
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QColor

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
)
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QColor


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

        # Add size attribute
        size_label = QLabel("Size:")
        self.size_input = SelectAllLineEdit()
        self.size_input.setText(str(self.annotation.attributes.get("Size", -1)))
        self.size_input.setPlaceholderText("-1")
        self.size_input.textChanged.connect(self.update_size_attribute)

        attributes_layout.addWidget(size_label, 0, 0)
        attributes_layout.addWidget(self.size_input, 0, 1)

        # Add quality attribute
        quality_label = QLabel("Quality:")
        self.quality_input = SelectAllLineEdit()
        self.quality_input.setText(str(self.annotation.attributes.get("Quality", -1)))
        self.quality_input.setPlaceholderText("-1")
        self.quality_input.textChanged.connect(self.update_quality_attribute)

        attributes_layout.addWidget(quality_label, 1, 0)
        attributes_layout.addWidget(self.quality_input, 1, 1)

        layout.addLayout(attributes_layout)

        # Add a separator line
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

    def update_size_attribute(self, text):
        try:
            value = int(text) if text else -1
            self.annotation.attributes["Size"] = value
            # Update canvas to reflect changes
            if hasattr(self.parent_dock, "main_window"):
                self.parent_dock.main_window.canvas.update()
        except ValueError:
            # Reset to previous value if not a valid integer
            self.size_input.setText(str(self.annotation.attributes.get("Size", -1)))

    def update_quality_attribute(self, text):
        try:
            value = int(text) if text else -1
            self.annotation.attributes["Quality"] = value
            # Update canvas to reflect changes
            if hasattr(self.parent_dock, "main_window"):
                self.parent_dock.main_window.canvas.update()
        except ValueError:
            # Reset to previous value if not a valid integer
            self.quality_input.setText(
                str(self.annotation.attributes.get("Quality", -1))
            )


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
        if (not hasattr(self.main_window, "frame_annotations") or
                not self.main_window.frame_annotations):
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.information(self, "Batch Edit", "No annotations found in any frame.")
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
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QSpinBox, QCheckBox, QDialogButtonBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Batch Edit Annotations")
        dialog.setMinimumWidth(300)
        
        layout = QVBoxLayout(dialog)
        form_layout = QFormLayout()
        
        # Size attribute (numeric 0-100)
        dialog.size_spin = QSpinBox()
        dialog.size_spin.setRange(-1, 100)
        dialog.size_spin.setValue(-1)
        dialog.size_spin.setSpecialValueText("No Change")  # -1 means no change
        form_layout.addRow("Size (0-100):", dialog.size_spin)
        
        # Quality attribute (numeric 0-100)
        dialog.quality_spin = QSpinBox()
        dialog.quality_spin.setRange(-1, 100)
        dialog.quality_spin.setValue(-1)
        dialog.quality_spin.setSpecialValueText("No Change")  # -1 means no change
        form_layout.addRow("Quality (0-100):", dialog.quality_spin)
        
        # Option to apply to all frames - checked by default
        dialog.apply_all_frames_checkbox = QCheckBox("Apply to all frames")
        dialog.apply_all_frames_checkbox.setChecked(True)  # Set to checked by default
        
        # Add form layout to main layout
        layout.addLayout(form_layout)
        layout.addWidget(dialog.apply_all_frames_checkbox)
        
        # Add buttons
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        
        return dialog

    def apply_attributes_to_current_frame(self, size_value, quality_value):
        """Apply attribute changes to all annotations in the current frame"""
        current_frame = self.main_window.current_frame
        
        # If there are no annotations for the current frame, create an empty list
        if current_frame not in self.main_window.frame_annotations:
            self.main_window.frame_annotations[current_frame] = []
        
        annotations = self.main_window.frame_annotations[current_frame]
        
        for annotation in annotations:
            # Only update if the value is not -1 (no change)
            if size_value != -1:
                annotation.attributes["Size"] = size_value
            if quality_value != -1:
                annotation.attributes["Quality"] = quality_value
        
        # Update the canvas annotations if they're from the current frame
        self.main_window.canvas.annotations = annotations.copy()

    def apply_attributes_to_all_frames(self, size_value, quality_value):
        """Apply attribute changes to all annotations in all frames"""
        for frame_num, annotations in self.main_window.frame_annotations.items():
            for annotation in annotations:
                # Only update if the value is not -1 (no change)
                if size_value != -1:
                    annotation.attributes["Size"] = size_value
                if quality_value != -1:
                    annotation.attributes["Quality"] = quality_value
        
        # Update the current frame's annotations on the canvas
        current_frame = self.main_window.current_frame
        if current_frame in self.main_window.frame_annotations:
            self.main_window.canvas.annotations = self.main_window.frame_annotations[current_frame].copy()
