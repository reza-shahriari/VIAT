"""Style management for the Video Annotation Tool."""

from PyQt5.QtWidgets import QApplication, QStyleFactory
from PyQt5.QtGui import QPalette, QColor

class StyleManager:
    """Manages application styles and themes."""
    
    @staticmethod
    def set_default_style():
        """Set the default system style."""
        QApplication.setStyle(QStyleFactory.create("Fusion"))  # Use Fusion as base
        QApplication.setPalette(QApplication.style().standardPalette())
        return True
    
    @staticmethod
    def set_fusion_style():
        """Set the Fusion style."""
        QApplication.setStyle(QStyleFactory.create("Fusion"))
        QApplication.setPalette(QApplication.style().standardPalette())
        return True
    
    @staticmethod
    def set_windows_style():
        """Set the Windows style."""
        try:
            QApplication.setStyle(QStyleFactory.create("Windows"))
            QApplication.setPalette(QApplication.style().standardPalette())
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
        dark_palette.setColor(QPalette.Base, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, QColor(0, 0, 0))
        
        QApplication.setPalette(dark_palette)
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
        
        QApplication.setPalette(light_palette)
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
        return True
    
    @classmethod
    def get_available_styles(cls):
        """Get a list of all available styles."""
        return [
            "Default", "Fusion", "Windows", "Dark", "Light", "Blue", "Green"
        ]
    
    @classmethod
    def apply_style(cls, style_name):
        """Apply a style by name with error handling."""
        style_method = getattr(cls, f"set_{style_name.lower()}_style", None)
        
        if style_method is None:
            # Fallback to default if style not found
            return cls.set_default_style()
        
        try:
            return style_method()
        except Exception:
            # Fallback to default if style application fails
            return cls.set_default_style()
