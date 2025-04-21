from PyQt5.QtWidgets import QToolBar, QLabel, QComboBox, QPushButton, QSlider, QAction
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

class AnnotationToolbar(QToolBar):
    def __init__(self, main_window):
        super().__init__("Annotation Toolbar")
        self.main_window = main_window
        # Default annotation methods if not defined in main window
        self.default_annotation_methods = {
            "Rectangle": "Draw rectangular bounding boxes",
            "Polygon": "Draw polygon shapes",
            "Point": "Mark specific points"
        }
        self.init_ui()
    
    def init_ui(self):
        # Add class selector
        class_label = QLabel("Class: ")
        self.addWidget(class_label)
        
        self.class_selector = QComboBox()
        self.update_class_selector()
        self.class_selector.currentIndexChanged.connect(self.on_class_selected)
        self.addWidget(self.class_selector)
        
        # Add annotation method selector
        self.addSeparator()
        method_label = QLabel("Annotation Method: ")
        self.addWidget(method_label)
        
        self.method_selector = QComboBox()
        # Use annotation_methods if available, otherwise use defaults
        annotation_methods = getattr(self.main_window, 'annotation_methods', self.default_annotation_methods)
        self.method_selector.addItems(list(annotation_methods.keys()))
        
        # Set current method if available, otherwise use first method
        current_method = getattr(self.main_window, 'current_annotation_method', None)
        if current_method and current_method in annotation_methods:
            self.method_selector.setCurrentText(current_method)
        elif annotation_methods:
            # Set first method as default
            first_method = list(annotation_methods.keys())[0]
            self.method_selector.setCurrentText(first_method)
            # Update main window's current method if it exists
            if hasattr(self.main_window, 'change_annotation_method'):
                self.main_window.change_annotation_method(first_method)
        
        self.method_selector.currentTextChanged.connect(self.on_method_selected)
        self.addWidget(self.method_selector)
        
        # Add tools
        self.addSeparator()
        
        # Add button
        add_action = QAction(QIcon.fromTheme("list-add", QIcon()), "Add Annotation", self)
        add_action.triggered.connect(self.add_class)
        self.addAction(add_action)
        
        # Edit button
        edit_action = QAction(QIcon.fromTheme("document-edit", QIcon()), "Edit Selected", self)
        edit_action.triggered.connect(self.edit_selected)
        self.addAction(edit_action)
        
        # Delete button
        delete_action = QAction(QIcon.fromTheme("edit-delete", QIcon()), "Delete Selected", self)
        delete_action.triggered.connect(self.delete_selected)
        self.addAction(delete_action)
        
        # Add zoom controls
        self.addSeparator()
        
        # Zoom in button
        zoom_in_action = QAction(QIcon.fromTheme("zoom-in", QIcon()), "Zoom In", self)
        zoom_in_action.triggered.connect(self.zoom_in)
        self.addAction(zoom_in_action)
        
        # Zoom out button
        zoom_out_action = QAction(QIcon.fromTheme("zoom-out", QIcon()), "Zoom Out", self)
        zoom_out_action.triggered.connect(self.zoom_out)
        self.addAction(zoom_out_action)
        
        # Reset zoom button
        reset_zoom_action = QAction(QIcon.fromTheme("zoom-original", QIcon()), "Reset Zoom", self)
        reset_zoom_action.triggered.connect(self.reset_zoom)
        self.addAction(reset_zoom_action)
    
    def update_class_selector(self):
        """Update the class selector with available classes"""
        self.class_selector.clear()
        if hasattr(self.main_window, 'canvas') and hasattr(self.main_window.canvas, 'class_colors'):
            self.class_selector.addItems(self.main_window.canvas.class_colors.keys())
            self.class_selector.addItem("Add New...")
    
    def on_class_selected(self, index):
        """Handle selection of a class in the dropdown"""
        class_name = self.class_selector.currentText()
        if class_name == "Add New...":
            # Reset to previous selection
            if hasattr(self.main_window, 'canvas'):
                canvas = self.main_window.canvas
                if hasattr(canvas, 'annotations') and canvas.annotations:
                    self.class_selector.setCurrentText(canvas.annotations[-1].class_name)
                else:
                    self.class_selector.setCurrentIndex(0)
            
            # Show dialog to add new class
            self.add_class()
        else:
            if hasattr(self.main_window, 'canvas') and hasattr(self.main_window.canvas, 'set_current_class'):
                self.main_window.canvas.set_current_class(class_name)
    
    def on_method_selected(self, method_name):
        """Handle selection of an annotation method"""
        if hasattr(self.main_window, 'change_annotation_method'):
            self.main_window.change_annotation_method(method_name)
    
    def add_class(self):
        """Add a new class"""
        if hasattr(self.main_window, 'add_class'):
            self.main_window.add_class()
    
    def edit_selected(self):
        """Edit the selected annotation"""
        if (hasattr(self.main_window, 'canvas') and 
            hasattr(self.main_window.canvas, 'selected_annotation') and 
            self.main_window.canvas.selected_annotation and
            hasattr(self.main_window, 'edit_annotation')):
            self.main_window.edit_annotation(self.main_window.canvas.selected_annotation)
    
    def delete_selected(self):
        """Delete the selected annotation"""
        if (hasattr(self.main_window, 'canvas') and 
            hasattr(self.main_window.canvas, 'selected_annotation') and 
            self.main_window.canvas.selected_annotation and
            hasattr(self.main_window, 'delete_selected_annotation')):
            self.main_window.delete_selected_annotation()
    
    def zoom_in(self):
        """Zoom in on the canvas"""
        if hasattr(self.main_window, 'zoom_in'):
            self.main_window.zoom_in()
    
    def zoom_out(self):
        """Zoom out on the canvas"""
        if hasattr(self.main_window, 'zoom_out'):
            self.main_window.zoom_out()
    
    def reset_zoom(self):
        """Reset zoom to default"""
        if hasattr(self.main_window, 'reset_zoom'):
            self.main_window.reset_zoom()
