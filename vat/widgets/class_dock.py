from PyQt5.QtWidgets import (
    QDockWidget,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QListWidget,
    QPushButton,
    QMenu,
    QListWidgetItem,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor


class ClassDock(QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Classes", parent)
        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        self.main_window = parent
        self.init_ui()

    def init_ui(self):
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Class list
        self.classes_list = QListWidget()
        self.classes_list.itemClicked.connect(self.on_class_selected)
        self.classes_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.classes_list.customContextMenuRequested.connect(self.show_context_menu)

        # Update class list
        self.update_class_list()

        # Class controls
        controls_layout = QHBoxLayout()
        add_btn = QPushButton("Add Class")
        add_btn.clicked.connect(self.add_class)
        edit_btn = QPushButton("Edit Class")
        edit_btn.clicked.connect(self.edit_selected_class)
        delete_btn = QPushButton("Delete Class")
        delete_btn.clicked.connect(self.delete_selected_class)

        controls_layout.addWidget(add_btn)
        controls_layout.addWidget(edit_btn)
        controls_layout.addWidget(delete_btn)

        # Add widgets to layout
        layout.addWidget(self.classes_list)
        layout.addLayout(controls_layout)

        self.setWidget(widget)

    def update_class_list(self):
        """Update the class list widget"""
        self.classes_list.clear()
        canvas = self.main_window.canvas

        for class_name, color in canvas.class_colors.items():
            item = QListWidgetItem(class_name)
            item.setForeground(color)
            self.classes_list.addItem(item)

    def on_class_selected(self, item):
        """Handle selection of a class in the list"""
        class_name = item.text()
        self.main_window.class_selector.setCurrentText(class_name)
        self.main_window.canvas.set_current_class(class_name)

    def show_context_menu(self, position):
        """Show context menu for class list"""
        item = self.classes_list.itemAt(position)
        if not item:
            return

        class_name = item.text()

        menu = QMenu()
        edit_action = menu.addAction("Edit")
        delete_action = menu.addAction("Delete")

        # Show menu and get selected action
        action = menu.exec_(self.classes_list.mapToGlobal(position))

        if action == edit_action:
            self.edit_selected_class()

        elif action == delete_action:
            self.delete_selected_class()

    def add_class(self):
        """Add a new class"""
        self.main_window.add_class()

    def edit_selected_class(self):
        """Edit the selected class"""
        self.main_window.edit_selected_class()

    def delete_selected_class(self):
        """Delete the selected class"""
        self.main_window.delete_selected_class()
