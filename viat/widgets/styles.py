"""Style management for the Video Annotation Tool."""

from PyQt5.QtWidgets import QApplication, QStyleFactory
from PyQt5.QtGui import QPalette, QColor
import os


class StyleManager:
    """Manages application styles and themes using both palette and stylesheets."""


    @staticmethod
    def set_fusion_style():
        """Set the Fusion style."""
        QApplication.setStyle(QStyleFactory.create("Fusion"))
        QApplication.setPalette(QApplication.style().standardPalette())
        QApplication.instance().setStyleSheet("")  # Clear any stylesheet
        return True

    @staticmethod
    def set_windows_style():
        """Set the Windows style."""
        try:
            QApplication.setStyle(QStyleFactory.create("Windows"))
            QApplication.setPalette(QApplication.style().standardPalette())
            QApplication.instance().setStyleSheet("")  # Clear any stylesheet
            return True
        except Exception:
            # Fallback to Fusion if Windows style is not available
            StyleManager.set_fusion_style()
            return False

    @staticmethod
    def set_dark_style():
        """Set a dark theme."""
        QApplication.setStyle(QStyleFactory.create("Fusion"))

        # Dark palette
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Light, QColor(80, 80, 80))
        dark_palette.setColor(QPalette.Midlight, QColor(70, 70, 70))
        dark_palette.setColor(QPalette.Mid, QColor(60, 60, 60))
        dark_palette.setColor(QPalette.Dark, QColor(30, 30, 30))
        dark_palette.setColor(QPalette.Shadow, QColor(20, 20, 20))
        
        # Set disabled colors explicitly
        dark_palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(127, 127, 127))
        dark_palette.setColor(QPalette.Disabled, QPalette.Text, QColor(127, 127, 127))
        dark_palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(127, 127, 127))
        dark_palette.setColor(QPalette.Disabled, QPalette.Highlight, QColor(80, 80, 80))
        dark_palette.setColor(QPalette.Disabled, QPalette.HighlightedText, QColor(127, 127, 127))

        QApplication.setPalette(dark_palette)

        # Apply comprehensive stylesheet for dark theme
        stylesheet = """
        /* Global styles */
        QWidget {
            background-color: #353535;
            color: #ffffff;
        }
        
        /* Main window */
        QMainWindow {
            background-color: #353535;
        }
        
        /* Menus and status bar */
        QMenuBar, QMenu, QStatusBar {
            background-color: #2d2d2d;
            color: #ffffff;
        }
        
        QMenuBar::item:selected, QMenu::item:selected {
            background-color: #2a82da;
        }
        
        /* Buttons */
        QPushButton {
            background-color: #444444;
            color: #ffffff;
            border: 1px solid #555555;
            border-radius: 3px;
            padding: 5px;
        }
        
        QPushButton:hover {
            background-color: #505050;
        }
        
        QPushButton:pressed {
            background-color: #2a82da;
        }
        
        QPushButton:disabled {
            background-color: #353535;
            color: #7f7f7f;
            border: 1px solid #404040;
        }
        
        /* Input widgets */
        QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background-color: #232323;
            color: #ffffff;
            border: 1px solid #555555;
            border-radius: 3px;
            padding: 2px;
        }
        
        QComboBox QAbstractItemView {
            background-color: #232323;
            color: #ffffff;
            selection-background-color: #2a82da;
        }
        
        /* Lists and trees */
        QListWidget, QTreeWidget, QTableWidget {
            background-color: #232323;
            color: #ffffff;
            alternate-background-color: #353535;
        }
        
        QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected {
            background-color: #2a82da;
        }
        
        /* Sliders */
        QSlider::groove:horizontal {
            background-color: #232323;
            height: 4px;
        }
        
        QSlider::handle:horizontal {
            background-color: #2a82da;
            width: 10px;
            margin: -4px 0;
            border-radius: 5px;
        }
        
        /* Scroll bars */
        QScrollBar:vertical, QScrollBar:horizontal {
            background-color: #232323;
            border: none;
        }
        
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
            background-color: #505050;
            border-radius: 4px;
        }
        
        QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
            background-color: #2a82da;
        }
        
        /* Dock widgets */
        QDockWidget {
            titlebar-close-icon: url(close.png);
            titlebar-normal-icon: url(undock.png);
        }
        
        QDockWidget::title {
            background-color: #2d2d2d;
            padding-left: 5px;
            padding-top: 2px;
        }
        
        /* Tabs */
        QTabWidget::pane {
            border: 1px solid #555555;
        }
        
        QTabBar::tab {
            background-color: #353535;
            color: #ffffff;
            border: 1px solid #555555;
            padding: 5px 10px;
        }
        
        QTabBar::tab:selected {
            background-color: #2a82da;
        }
        
        /* Tool tips */
        QToolTip {
            background-color: #232323;
            color: #ffffff;
            border: 1px solid #555555;
        }
        
        /* Canvas specific styles */
        VideoCanvas {
            background-color: #232323;
            border: 1px solid #555555;
        }
        
        /* Annotation dock specific styles */
        QDockWidget[objectName="annotationDock"] QWidget {
            background-color: #353535;
            color: #ffffff;
        }
        
        QDockWidget[objectName="annotationDock"] QTextEdit, 
        QDockWidget[objectName="annotationDock"] QPlainTextEdit,
        QDockWidget[objectName="annotationDock"] QLineEdit {
            background-color: #232323;
            color: #ffffff;
            border: 1px solid #555555;
        }
        """
        
        QApplication.instance().setStyleSheet(stylesheet)
        return True

    @staticmethod
    def set_light_style():
        """Set a light theme."""
        QApplication.setStyle(QStyleFactory.create("Fusion"))

        # Light palette
        light_palette = QPalette()
        light_palette.setColor(QPalette.Window, QColor(240, 240, 240))
        light_palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
        light_palette.setColor(QPalette.Base, QColor(255, 255, 255))
        light_palette.setColor(QPalette.AlternateBase, QColor(233, 233, 233))
        light_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        light_palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
        light_palette.setColor(QPalette.Text, QColor(0, 0, 0))
        light_palette.setColor(QPalette.Button, QColor(240, 240, 240))
        light_palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
        light_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        light_palette.setColor(QPalette.Link, QColor(0, 100, 200))
        light_palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
        light_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        
        # Set disabled colors explicitly
        light_palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(120, 120, 120))
        light_palette.setColor(QPalette.Disabled, QPalette.Text, QColor(120, 120, 120))
        light_palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(120, 120, 120))
        light_palette.setColor(QPalette.Disabled, QPalette.Highlight, QColor(200, 200, 200))
        light_palette.setColor(QPalette.Disabled, QPalette.HighlightedText, QColor(120, 120, 120))

        QApplication.setPalette(light_palette)

        # Apply comprehensive stylesheet for light theme
        stylesheet = """
        /* Global styles */
        QWidget {
            background-color: #f0f0f0;
            color: #000000;
        }
        
        /* Main window */
        QMainWindow {
            background-color: #f0f0f0;
        }
        
        /* Menus and status bar */
        QMenuBar, QMenu, QStatusBar {
            background-color: #f5f5f5;
            color: #000000;
        }
        
        QMenuBar::item:selected, QMenu::item:selected {
            background-color: #0078d7;
            color: #ffffff;
        }
        
        /* Buttons */
        QPushButton {
            background-color: #e0e0e0;
            color: #000000;
            border: 1px solid #c0c0c0;
            border-radius: 3px;
            padding: 5px;
        }
        
        QPushButton:hover {
            background-color: #d0d0d0;
        }
        
        QPushButton:pressed {
            background-color: #0078d7;
            color: #ffffff;
        }
        
        QPushButton:disabled {
            background-color: #f0f0f0;
            color: #a0a0a0;
            border: 1px solid #d0d0d0;
        }
        
        /* Input widgets */
        QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background-color: #ffffff;
            color: #000000;
            border: 1px solid #c0c0c0;
            border-radius: 3px;
            padding: 2px;
        }
        
        QComboBox QAbstractItemView {
            background-color: #ffffff;
            color: #000000;
            selection-background-color: #0078d7;
            selection-color: #ffffff;
        }
        
        /* Lists and trees */
        QListWidget, QTreeWidget, QTableWidget {
            background-color: #ffffff;
            color: #000000;
            alternate-background-color: #f5f5f5;
        }
        
        QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected {
            background-color: #0078d7;
            color: #ffffff;
        }
        
        /* Sliders */
        QSlider::groove:horizontal {
            background-color: #d0d0d0;
            height: 4px;
        }
        
        QSlider::handle:horizontal {
            background-color: #0078d7;
            width: 10px;
            margin: -4px 0;
            border-radius: 5px;
        }
        
        /* Scroll bars */
        QScrollBar:vertical, QScrollBar:horizontal {
            background-color: #f0f0f0;
            border: none;
        }
        
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
            background-color: #c0c0c0;
            border-radius: 4px;
        }
        
        QScrollBar::handle:vertical
        QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
            background-color: #0078d7;
        }
        
        /* Dock widgets */
        QDockWidget {
            titlebar-close-icon: url(close.png);
            titlebar-normal-icon: url(undock.png);
        }
        
        QDockWidget::title {
            background-color: #e0e0e0;
            padding-left: 5px;
            padding-top: 2px;
        }
        
        /* Tabs */
        QTabWidget::pane {
            border: 1px solid #c0c0c0;
        }
        
        QTabBar::tab {
            background-color: #e0e0e0;
            color: #000000;
            border: 1px solid #c0c0c0;
            padding: 5px 10px;
        }
        
        QTabBar::tab:selected {
            background-color: #0078d7;
            color: #ffffff;
        }
        
        /* Tool tips */
        QToolTip {
            background-color: #ffffff;
            color: #000000;
            border: 1px solid #c0c0c0;
        }
        
        /* Canvas specific styles */
        VideoCanvas {
            background-color: #ffffff;
            border: 1px solid #c0c0c0;
        }
        
        /* Annotation dock specific styles */
        QDockWidget[objectName="annotationDock"] QWidget {
            background-color: #f0f0f0;
            color: #000000;
        }
        
        QDockWidget[objectName="annotationDock"] QTextEdit, 
        QDockWidget[objectName="annotationDock"] QPlainTextEdit,
        QDockWidget[objectName="annotationDock"] QLineEdit {
            background-color: #ffffff;
            color: #000000;
            border: 1px solid #c0c0c0;
        }
        """
        
        QApplication.instance().setStyleSheet(stylesheet)
        return True

    @staticmethod
    def set_blue_style():
        """Set a blue theme."""
        QApplication.setStyle(QStyleFactory.create("Fusion"))

        # Blue palette
        blue_palette = QPalette()
        blue_palette.setColor(QPalette.Window, QColor(213, 228, 242))
        blue_palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
        blue_palette.setColor(QPalette.Base, QColor(255, 255, 255))
        blue_palette.setColor(QPalette.AlternateBase, QColor(213, 228, 242))
        blue_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        blue_palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
        blue_palette.setColor(QPalette.Text, QColor(0, 0, 0))
        blue_palette.setColor(QPalette.Button, QColor(213, 228, 242))
        blue_palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
        blue_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        blue_palette.setColor(QPalette.Link, QColor(0, 0, 255))
        blue_palette.setColor(QPalette.Highlight, QColor(0, 120, 215))
        blue_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))

        QApplication.setPalette(blue_palette)
        
        # Apply comprehensive stylesheet for blue theme
        stylesheet = """
        /* Global styles */
        QWidget {
            background-color: #d5e4f2;
            color: #000000;
        }
        
        /* Main window */
        QMainWindow {
            background-color: #d5e4f2;
        }
        
        /* Menus and status bar */
        QMenuBar, QMenu, QStatusBar {
            background-color: #c5d9ed;
            color: #000000;
        }
        
        QMenuBar::item:selected, QMenu::item:selected {
            background-color: #0078d7;
            color: #ffffff;
        }
        
        /* Buttons */
        QPushButton {
            background-color: #b8cfe5;
            color: #000000;
            border: 1px solid #a0c0e0;
            border-radius: 3px;
            padding: 5px;
        }
        
        QPushButton:hover {
            background-color: #a0c0e0;
        }
        
        QPushButton:pressed {
            background-color: #0078d7;
            color: #ffffff;
        }
        
        QPushButton:disabled {
            background-color: #d5e4f2;
            color: #a0a0a0;
            border: 1px solid #c0d0e0;
        }
        
        /* Input widgets */
        QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background-color: #ffffff;
            color: #000000;
            border: 1px solid #a0c0e0;
            border-radius: 3px;
            padding: 2px;
        }
        
        QComboBox QAbstractItemView {
            background-color: #ffffff;
            color: #000000;
            selection-background-color: #0078d7;
            selection-color: #ffffff;
        }
        
        /* Lists and trees */
        QListWidget, QTreeWidget, QTableWidget {
            background-color: #ffffff;
            color: #000000;
            alternate-background-color: #e5f0fa;
        }
        
        QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected {
            background-color: #0078d7;
            color: #ffffff;
        }
        
        /* Sliders */
        QSlider::groove:horizontal {
            background-color: #b8cfe5;
            height: 4px;
        }
        
        QSlider::handle:horizontal {
            background-color: #0078d7;
            width: 10px;
            margin: -4px 0;
            border-radius: 5px;
        }
        
        /* Scroll bars */
        QScrollBar:vertical, QScrollBar:horizontal {
            background-color: #d5e4f2;
            border: none;
        }
        
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
            background-color: #b8cfe5;
            border-radius: 4px;
        }
        
        QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
            background-color: #0078d7;
        }
        
        /* Canvas specific styles */
        VideoCanvas {
            background-color: #ffffff;
            border: 1px solid #a0c0e0;
        }
        
        /* Annotation dock specific styles */
        QDockWidget[objectName="annotationDock"] QWidget {
            background-color: #d5e4f2;
            color: #000000;
        }
        
        QDockWidget[objectName="annotationDock"] QTextEdit, 
        QDockWidget[objectName="annotationDock"] QPlainTextEdit,
        QDockWidget[objectName="annotationDock"] QLineEdit {
            background-color: #ffffff;
            color: #000000;
            border: 1px solid #a0c0e0;
        }
        """
        
        QApplication.instance().setStyleSheet(stylesheet)
        return True

    @staticmethod
    def set_green_style():
        """Set a green theme."""
        QApplication.setStyle(QStyleFactory.create("Fusion"))

        # Green palette
        green_palette = QPalette()
        green_palette.setColor(QPalette.Window, QColor(233, 247, 239))
        green_palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
        green_palette.setColor(QPalette.Base, QColor(255, 255, 255))
        green_palette.setColor(QPalette.AlternateBase, QColor(233, 247, 239))
        green_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        green_palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
        green_palette.setColor(QPalette.Text, QColor(0, 0, 0))
        green_palette.setColor(QPalette.Button, QColor(233, 247, 239))
        green_palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
        green_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        green_palette.setColor(QPalette.Link, QColor(0, 128, 0))
        green_palette.setColor(QPalette.Highlight, QColor(46, 204, 113))
        green_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))

        QApplication.setPalette(green_palette)
        
        # Apply comprehensive stylesheet for green theme
        stylesheet = """
        /* Global styles */
        QWidget {
            background-color: #e9f7ef;
            color: #000000;
        }
        
        /* Main window */
        QMainWindow {
            background-color: #e9f7ef;
        }
        
        /* Menus and status bar */
        QMenuBar, QMenu, QStatusBar {
            background-color: #d5f0e0;
            color: #000000;
        }
        
        QMenuBar::item:selected, QMenu::item:selected {
            background-color: #2ecc71;
            color: #ffffff;
        }
        
        /* Buttons */
        QPushButton {
            background-color: #c8e6d7;
            color: #000000;
            border: 1px solid #a0d8b8;
            border-radius: 3px;
            padding: 5px;
        }
        
        QPushButton:hover {
            background-color: #a0d8b8;
        }
        
        QPushButton:pressed {
            background-color: #2ecc71;
            color: #ffffff;
        }
        
        QPushButton:disabled {
            background-color: #e9f7ef;
            color: #a0a0a0;
            border: 1px solid #c0e0d0;
        }
        
        /* Input widgets */
        QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background-color: #ffffff;
            color: #000000;
            border: 1px solid #a0d8b8;
            border-radius: 3px;
            padding: 2px;
        }
        
        QComboBox QAbstractItemView {
            background-color: #ffffff;
            color: #000000;
            selection-background-color: #2ecc71;
            selection-color: #ffffff;
        }
        
        /* Lists and trees */
        QListWidget, QTreeWidget, QTableWidget {
            background-color: #ffffff;
            color: #000000;
            alternate-background-color: #e5f5ec;
        }
        
        QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected {
            background-color: #2ecc71;
            color: #ffffff;
        }
        
        /* Canvas specific styles */
        VideoCanvas {
            background-color: #ffffff;
            border: 1px solid #a0d8b8;
        }
        
        /* Annotation dock specific styles */
        QDockWidget[objectName="annotationDock"] QWidget {
            background-color: #e9f7ef;
            color: #000000;
        }
        
        QDockWidget[objectName="annotationDock"] QTextEdit, 
        QDockWidget[objectName="annotationDock"] QPlainTextEdit,
        QDockWidget[objectName="annotationDock"] QLineEdit {
            background-color: #ffffff;
            color: #000000;
            border: 1px solid #a0d8b8;
        }
        """
        
        QApplication.instance().setStyleSheet(stylesheet)
        return True

    @staticmethod
    def set_sunset_style():
        """Set a warm sunset theme with orange and purple accents."""
        QApplication.setStyle(QStyleFactory.create("Fusion"))

        # Sunset palette with warm oranges and purples
        sunset_palette = QPalette()
        sunset_palette.setColor(QPalette.Window, QColor(45, 20, 44))  # Deep purple background
        sunset_palette.setColor(QPalette.WindowText, QColor(255, 222, 173))  # Warm text color
        sunset_palette.setColor(QPalette.Base, QColor(66, 30, 66))  # Slightly lighter purple
        sunset_palette.setColor(QPalette.AlternateBase, QColor(55, 25, 55))
        sunset_palette.setColor(QPalette.ToolTipBase, QColor(45, 20, 44))
        sunset_palette.setColor(QPalette.ToolTipText, QColor(255, 222, 173))
        sunset_palette.setColor(QPalette.Text, QColor(255, 222, 173))  # Warm text
        sunset_palette.setColor(QPalette.Button, QColor(66, 30, 66))
        sunset_palette.setColor(QPalette.ButtonText, QColor(255, 222, 173))
        sunset_palette.setColor(QPalette.BrightText, QColor(255, 132, 0))  # Bright orange
        sunset_palette.setColor(QPalette.Link, QColor(255, 165, 0))  # Orange
        sunset_palette.setColor(QPalette.Highlight, QColor(255, 140, 0))  # Dark orange highlight
        sunset_palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        sunset_palette.setColor(QPalette.Light, QColor(90, 45, 90))
        sunset_palette.setColor(QPalette.Midlight, QColor(80, 40, 80))
        sunset_palette.setColor(QPalette.Mid, QColor(70, 35, 70))
        sunset_palette.setColor(QPalette.Dark, QColor(40, 18, 40))
        sunset_palette.setColor(QPalette.Shadow, QColor(30, 15, 30))

        QApplication.setPalette(sunset_palette)
        
        # Apply comprehensive stylesheet for sunset theme
        stylesheet = """
        /* Global styles */
        QWidget {
            background-color: #2d142c;
            color: #ffdeaa;
        }
        
        /* Main window */
        QMainWindow {
            background-color: #2d142c;
        }
        /* Menus and status bar */
        QMenuBar, QMenu, QStatusBar {
            background-color: #3a1a39;
            color: #ffdeaa;
        }
        
        QMenuBar::item:selected, QMenu::item:selected {
            background-color: #ff8c00;
            color: #ffffff;
        }
        
        /* Buttons */
        QPushButton {
            background-color: #421e42;
            color: #ffdeaa;
            border: 1px solid #5a2d5a;
            border-radius: 3px;
            padding: 5px;
        }
        
        QPushButton:hover {
            background-color: #5a2d5a;
        }
        
        QPushButton:pressed {
            background-color: #ff8c00;
            color: #ffffff;
        }
        
        QPushButton:disabled {
            background-color: #2d142c;
            color: #7f6e55;
            border: 1px solid #3a1a39;
        }
        
        /* Input widgets */
        QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background-color: #421e42;
            color: #ffdeaa;
            border: 1px solid #5a2d5a;
            border-radius: 3px;
            padding: 2px;
        }
        
        QComboBox QAbstractItemView {
            background-color: #421e42;
            color: #ffdeaa;
            selection-background-color: #ff8c00;
            selection-color: #ffffff;
        }
        
        /* Lists and trees */
        QListWidget, QTreeWidget, QTableWidget {
            background-color: #421e42;
            color: #ffdeaa;
            alternate-background-color: #371937;
        }
        
        QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected {
            background-color: #ff8c00;
            color: #ffffff;
        }
        
        /* Sliders */
        QSlider::groove:horizontal {
            background-color: #371937;
            height: 4px;
        }
        
        QSlider::handle:horizontal {
            background-color: #ff8c00;
            width: 10px;
            margin: -4px 0;
            border-radius: 5px;
        }
        
        /* Scroll bars */
        QScrollBar:vertical, QScrollBar:horizontal {
            background-color: #2d142c;
            border: none;
        }
        
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
            background-color: #5a2d5a;
            border-radius: 4px;
        }
        
        QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
            background-color: #ff8c00;
        }
        
        /* Canvas specific styles */
        VideoCanvas {
            background-color: #371937;
            border: 1px solid #5a2d5a;
        }
        
        /* Annotation dock specific styles */
        QDockWidget[objectName="annotationDock"] QWidget {
            background-color: #2d142c;
            color: #ffdeaa;
        }
        
        QDockWidget[objectName="annotationDock"] QTextEdit, 
        QDockWidget[objectName="annotationDock"] QPlainTextEdit,
        QDockWidget[objectName="annotationDock"] QLineEdit {
            background-color: #421e42;
            color: #ffdeaa;
            border: 1px solid #5a2d5a;
        }
        """
        
        QApplication.instance().setStyleSheet(stylesheet)
        return True
    
    @staticmethod
    def set_darkmodern_style():
        """Set a refined modern dark theme with subtle accents and improved readability."""
        QApplication.setStyle(QStyleFactory.create("Fusion"))

        # Modern dark palette with carefully selected colors
        dark_modern = QPalette()
        
        # Main colors - using a slightly blue-tinted dark gray for sophistication
        dark_modern.setColor(QPalette.Window, QColor(22, 25, 29))  # Slightly blue-tinted dark background
        dark_modern.setColor(QPalette.WindowText, QColor(237, 240, 242))  # Off-white text for reduced eye strain
        dark_modern.setColor(QPalette.Base, QColor(15, 17, 20))  # Darker area for content
        dark_modern.setColor(QPalette.AlternateBase, QColor(30, 33, 39))  # Slightly lighter for alternating rows
        
        # Interactive elements
        dark_modern.setColor(QPalette.Button, QColor(44, 49, 58))  # Slightly blue-tinted buttons
        dark_modern.setColor(QPalette.ButtonText, QColor(237, 240, 242))  # Off-white button text
        dark_modern.setColor(QPalette.BrightText, QColor(255, 128, 128))  # Soft red for attention
        
        # Accent colors - using a teal/cyan accent for modern feel
        dark_modern.setColor(QPalette.Link, QColor(102, 195, 204))  # Teal links
        dark_modern.setColor(QPalette.Highlight, QColor(61, 174, 183))  # Teal highlight
        dark_modern.setColor(QPalette.HighlightedText, QColor(255, 255, 255))  # Pure white on highlight
        
        # Tooltip
        dark_modern.setColor(QPalette.ToolTipBase, QColor(44, 49, 58))  # Dark tooltip background
        dark_modern.setColor(QPalette.ToolTipText, QColor(237, 240, 242))  # Light tooltip text
        
        # Text
        dark_modern.setColor(QPalette.Text, QColor(237, 240, 242))  # Off-white text
        
        # Gradients and shadows for depth
        dark_modern.setColor(QPalette.Light, QColor(55, 60, 70))
        dark_modern.setColor(QPalette.Midlight, QColor(45, 50, 60))
        dark_modern.setColor(QPalette.Mid, QColor(35, 40, 50))
        dark_modern.setColor(QPalette.Dark, QColor(18, 20, 24))
        dark_modern.setColor(QPalette.Shadow, QColor(10, 11, 13))
        
        # Disabled state - subtle but still visible
        dark_modern.setColor(QPalette.Disabled, QPalette.WindowText, QColor(128, 131, 136))
        dark_modern.setColor(QPalette.Disabled, QPalette.Text, QColor(128, 131, 136))
        dark_modern.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(128, 131, 136))
        dark_modern.setColor(QPalette.Disabled, QPalette.Highlight, QColor(40, 45, 52))
        dark_modern.setColor(QPalette.Disabled, QPalette.HighlightedText, QColor(128, 131, 136))

        QApplication.setPalette(dark_modern)
        
        # Apply comprehensive stylesheet for dark modern theme
        stylesheet = """
        /* Global styles */
        QWidget {
            background-color: #16191d;
            color: #edf0f2;
        }
        
        /* Main window */
        QMainWindow {
            background-color: #16191d;
        }
        
        /* Menus and status bar */
        QMenuBar, QMenu, QStatusBar {
            background-color: #1c2026;
            color: #edf0f2;
        }
        
        QMenuBar::item:selected, QMenu::item:selected {
            background-color: #3daeb7;
            color: #ffffff;
        }
        
        /* Buttons */
        QPushButton {
            background-color: #2c313a;
            color: #edf0f2;
            border: 1px solid #3c424d;
            border-radius: 3px;
            padding: 5px;
        }
        
        QPushButton:hover {
            background-color: #3c424d;
        }
        
        QPushButton:pressed {
            background-color: #3daeb7;
            color: #ffffff;
        }
        
        QPushButton:disabled {
            background-color: #16191d;
            color: #808388;
            border: 1px solid #2c313a;
        }
        
        /* Input widgets */
        QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background-color: #0f1114;
            color: #edf0f2;
            border: 1px solid #3c424d;
            border-radius: 3px;
            padding: 2px;
        }
        
        QComboBox QAbstractItemView {
            background-color: #0f1114;
            color: #edf0f2;
            selection-background-color: #3daeb7;
            selection-color: #ffffff;
        }
        
        /* Lists and trees */
        QListWidget, QTreeWidget, QTableWidget {
            background-color: #0f1114;
            color: #edf0f2;
            alternate-background-color: #1e2127;
        }
        
        QListWidget::item:selected, QTreeWidget::item:selected, QTableWidget::item:selected {
            background-color: #3daeb7;
            color: #ffffff;
        }
        
        /* Sliders */
        QSlider::groove:horizontal {
            background-color: #1e2127;
            height: 4px;
        }
        
        QSlider::handle:horizontal {
            background-color: #3daeb7;
            width: 10px;
            margin: -4px 0;
            border-radius: 5px;
        }
        
        /* Scroll bars */
        QScrollBar:vertical, QScrollBar:horizontal {
            background-color: #16191d;
            border: none;
        }
        
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
            background-color: #2c313a;
            border-radius: 4px;
        }
        
        QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
            background-color: #3daeb7;
        }
        
        /* Tool tips */
        QToolTip { 
            border: 1px solid #3c424d; 
            background-color: #2c313a; 
            color: #edf0f2; 
            padding: 5px;
            opacity: 200;
        }
        
        /* Canvas specific styles */
        VideoCanvas {
            background-color: #0f1114;
            border: 1px solid #3c424d;
        }
        
        /* Annotation dock specific styles */
        QDockWidget[objectName="annotationDock"] QWidget {
            background-color: #16191d;
            color: #edf0f2;
        }
        
        QDockWidget[objectName="annotationDock"] QTextEdit, 
        QDockWidget[objectName="annotationDock"] QPlainTextEdit,
        QDockWidget[objectName="annotationDock"] QLineEdit {
            background-color: #0f1114;
            color: #edf0f2;
            border: 1px solid #3c424d;
        }
        
        /* Fix for annotation dock - ensure text is visible */
        QDockWidget[objectName="annotationDock"] QTextEdit, 
        QDockWidget[objectName="annotationDock"] QPlainTextEdit,
        QDockWidget[objectName="annotationDock"] QLineEdit {
            background-color: #0f1114;
            color: #edf0f2;
            border: 1px solid #3c424d;
        }
        
        /* Dock widget styling */
        QDockWidget {
            titlebar-close-icon: url(close.png);
            titlebar-normal-icon: url(undock.png);
        }
        
        QDockWidget::title {
            background-color: #1c2026;
            padding-left: 5px;
            padding-top: 2px;
        }
        
        /* Tab widget styling */
        QTabWidget::pane {
            border: 1px solid #3c424d;
        }
        
        QTabBar::tab {
            background-color: #2c313a;
            color: #edf0f2;
            border: 1px solid #3c424d;
            padding: 5px 10px;
        }
        
        QTabBar::tab:selected {
            background-color: #3daeb7;
            color: #ffffff;
        }
        """
        
        QApplication.instance().setStyleSheet(stylesheet)
        return True

    @classmethod
    def get_available_styles(cls):
        """Get a list of all available styles."""
        return [ "Fusion", "Windows", "Dark", 
                "Light", "Blue", "Green", "Sunset", "DarkModern"]

    @classmethod
    def apply_style(cls, style_name):
        """Apply a style by name with error handling."""
        # First clear any existing stylesheet to avoid conflicts
        app = QApplication.instance()
        if app:
            app.setStyleSheet("")
            
        style_method = getattr(cls, f"set_{style_name.lower()}_style", None)

        if style_method is None:
            # Fallback to default if style not found
            return cls.set_darkmodern_style()

        try:
            return style_method()
        except Exception as e:
            print(f"Error applying style {style_name}: {e}")
            # Fallback to default if style application fails
            return cls.set_darkmodern_style()
