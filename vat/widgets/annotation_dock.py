from PyQt5.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QComboBox, QListWidget, QPushButton, QMenu, QDialog,
    QFormLayout, QSpinBox, QTextEdit, QDialogButtonBox, QListWidgetItem,
    QGroupBox, QGridLayout, QLineEdit,QFrame
)
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui import QColor

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
        self.size_input = QLineEdit()
        self.size_input.setText(str(self.annotation.attributes.get('size', -1)))
        self.size_input.setPlaceholderText("-1")
        self.size_input.textChanged.connect(self.update_size_attribute)
        
        attributes_layout.addWidget(size_label, 0, 0)
        attributes_layout.addWidget(self.size_input, 0, 1)
        
        # Add quality attribute
        quality_label = QLabel("Quality:")
        self.quality_input = QLineEdit()
        self.quality_input.setText(str(self.annotation.attributes.get('quality', -1)))
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
            self.annotation.attributes['size'] = value
            # Update canvas to reflect changes
            if hasattr(self.parent_dock, 'main_window'):
                self.parent_dock.main_window.canvas.update()
        except ValueError:
            # Reset to previous value if not a valid integer
            self.size_input.setText(str(self.annotation.attributes.get('size', -1)))
    
    def update_quality_attribute(self, text):
        try:
            value = int(text) if text else -1
            self.annotation.attributes['quality'] = value
            # Update canvas to reflect changes
            if hasattr(self.parent_dock, 'main_window'):
                self.parent_dock.main_window.canvas.update()
        except ValueError:
            # Reset to previous value if not a valid integer
            self.quality_input.setText(str(self.annotation.attributes.get('quality', -1)))

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
        if hasattr(self.main_window, 'frame_annotations') and current_frame in self.main_window.frame_annotations:
            # Add annotations for the current frame to the list
            for annotation in self.main_window.frame_annotations[current_frame]:
                item = QListWidgetItem(f"{annotation.class_name} - {annotation.id if hasattr(annotation, 'id') else ''}")
                item.setData(Qt.UserRole, annotation)
                self.annotations_list.addItem(item)
    
    def on_annotation_selected(self, item):
        """Handle selection of an annotation in the list"""
        annotation = item.data(Qt.UserRole)
        if annotation:
            # Select this annotation on the canvas
            self.main_window.canvas.select_annotation(annotation)
    
    def update_class_selector(self):
        """Update the class selector with available classes"""
        self.class_selector.clear()
        if hasattr(self.main_window, 'canvas'):
            self.class_selector.addItems(self.main_window.canvas.class_colors.keys())
    
    def on_class_selected(self, class_name):
        """Handle selection of a class"""
        if class_name and hasattr(self.main_window, 'canvas'):
            self.main_window.canvas.set_current_class(class_name)
    
    def add_annotation(self):
        """Add a new annotation with the current class"""
        if hasattr(self.main_window, 'add_empty_annotation'):
            self.main_window.add_empty_annotation()
    
    def delete_selected_annotation(self):
        """Delete the selected annotation"""
        selected_items = self.annotations_list.selectedItems()
        if selected_items:
            item = selected_items[0]
            annotation = item.data(Qt.UserRole)
            if annotation and hasattr(self.main_window, 'delete_annotation'):
                self.main_window.delete_annotation(annotation)
    
    def show_context_menu(self, position):
        """Show context menu for the selected annotation"""
        selected_items = self.annotations_list.selectedItems()
        if not selected_items:
            return
        
        item = selected_items[0]
        annotation = item.data(Qt.UserRole)
        
        menu = QMenu()
        delete_action = menu.addAction("Delete")
        
        action = menu.exec_(self.annotations_list.mapToGlobal(position))
        
        if action == delete_action:
            self.delete_selected_annotation()