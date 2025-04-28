from PyQt5.QtWidgets import (
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QPushButton,
    QMenu,
    QLabel,
    QTextEdit,
)
from PyQt5.QtCore import Qt,QTimer


class ClassDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Classes", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self.old_selected_class = None
        self.main_window = parent
        self.init_ui()

    def init_ui(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Class list
        self.classes_list = QListWidget()
        self.classes_list.setSelectionMode(QListWidget.SingleSelection)
        self.classes_list.itemClicked.connect(self.on_class_selected)
        self.classes_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.classes_list.customContextMenuRequested.connect(self.show_context_menu)
        
        # Make sure the list widget has a minimum height to be visible
        self.classes_list.setMinimumHeight(100)

        # Class controls
        controls_layout = QHBoxLayout()
        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self.add_class)
        edit_btn = QPushButton("Edit")
        edit_btn.clicked.connect(self.edit_class)
        delete_btn = QPushButton("Delete")
        delete_btn.clicked.connect(self.delete_class)

        controls_layout.addWidget(add_btn)
        controls_layout.addWidget(edit_btn)
        controls_layout.addWidget(delete_btn)

        # Attribute info section
        self.attribute_info = QTextEdit()
        self.attribute_info.setReadOnly(True)
        self.attribute_info.setMaximumHeight(150)
        self.attribute_info.setPlaceholderText("Select a class to view its attributes")

        # Add widgets to layout
        layout.addWidget(QLabel("Classes:"))
        layout.addWidget(self.classes_list)
        layout.addLayout(controls_layout)
        layout.addWidget(QLabel("Class Attributes:"))
        layout.addWidget(self.attribute_info)

        # Set the widget as the dock's widget
        self.setWidget(widget)

        # Update the class list - defer this to ensure canvas is ready
        QTimer.singleShot(100, self.update_class_list)


    def update_class_list(self):
        """Update the class list with available classes"""
        # Store the currently selected class name before clearing
        current_selected = None
        if self.classes_list.currentItem():
            current_selected = self.classes_list.currentItem().text()
        
        # Clear the list
        self.classes_list.clear()
        
        
        if hasattr(self.main_window, 'canvas'):
            if hasattr(self.main_window.canvas, 'class_colors'):
                class_colors = self.main_window.canvas.class_colors
                
                # Use a set to track added classes to prevent duplicates
                added_classes = set()
                
                for class_name in class_colors.keys():
                    if class_name not in added_classes:
                        self.classes_list.addItem(class_name)
                        added_classes.add(class_name)
                
                # Restore selection or select current class
                if current_selected and self.classes_list.findItems(current_selected, Qt.MatchExactly):
                    items = self.classes_list.findItems(current_selected, Qt.MatchExactly)
                    self.classes_list.setCurrentItem(items[0])
                elif hasattr(self.main_window.canvas, 'current_class'):
                    current_class = self.main_window.canvas.current_class
                    if current_class:
                        items = self.classes_list.findItems(current_class, Qt.MatchExactly)
                        if items:
                            self.classes_list.setCurrentItem(items[0])




    def on_class_selected(self, item):
        """Handle selection of a class"""
        class_name = item.text()
        



        # Update attribute info
        self.update_attribute_info(class_name)

        # Set as current class in canvas
        if hasattr(self.main_window, "canvas"):
            self.main_window.canvas.set_current_class(class_name)
            # Force update of the canvas
            self.main_window.canvas.update()

        # Update class selector in toolbar if it exists
        if hasattr(self.main_window, "class_selector"):
            # Block signals temporarily to prevent recursive calls
            self.main_window.class_selector.blockSignals(True)
            self.main_window.class_selector.setCurrentText(class_name)
            self.main_window.class_selector.blockSignals(False)
            
        # Mark the project as modified
        if hasattr(self.main_window, "project_modified"):
            self.main_window.project_modified = True



    def update_attribute_info(self, class_name):
        """Update the attribute info text edit with class attribute details"""
        if not hasattr(self.main_window.canvas, "class_attributes"):
            self.attribute_info.setText("No attribute information available")
            return

        attributes = self.main_window.canvas.class_attributes.get(class_name, {})

        if not attributes:
            self.attribute_info.setText("No attributes defined for this class")
            return

        info_text = ""
        for attr_name, attr_config in attributes.items():
            attr_type = attr_config.get("type", "string")
            default_val = attr_config.get("default", "")

            info_text += f"<b>{attr_name}</b> ({attr_type})<br>"
            info_text += f"Default: {default_val}<br>"

            if attr_type in ["int", "float"]:
                min_val = attr_config.get("min", "")
                max_val = attr_config.get("max", "")
                info_text += f"Range: {min_val} to {max_val}<br>"

            info_text += "<br>"

        self.attribute_info.setHtml(info_text)

    def add_class(self):
        """Add a new class"""
        if hasattr(self.main_window, "add_class"):
            self.main_window.add_class()

    def edit_class(self):
        """Edit the selected class"""
        if hasattr(self.main_window, "edit_selected_class"):
            self.main_window.edit_selected_class()

    def delete_class(self):
        """Delete the selected class"""
        if hasattr(self.main_window, "delete_selected_class"):
            self.main_window.delete_selected_class()

    def show_context_menu(self, position):
        """Show context menu for the selected class"""
        selected_items = self.classes_list.selectedItems()
        if not selected_items:
            return

        menu = QMenu()
        edit_action = menu.addAction("Edit")
        delete_action = menu.addAction("Delete")

        action = menu.exec_(self.classes_list.mapToGlobal(position))

        if action == edit_action:
            self.edit_class()
        elif action == delete_action:
            self.delete_class()
