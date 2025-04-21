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
        dark_palette.setColor(
            QPalette.Base, QColor(35, 35, 35)
        )  # Darker for better contrast
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

        # Additional palette colors for better contrast in dock widgets
        dark_palette.setColor(QPalette.Light, QColor(80, 80, 80))
        dark_palette.setColor(QPalette.Midlight, QColor(70, 70, 70))
        dark_palette.setColor(QPalette.Mid, QColor(60, 60, 60))
        dark_palette.setColor(QPalette.Dark, QColor(30, 30, 30))
        dark_palette.setColor(QPalette.Shadow, QColor(20, 20, 20))

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
        
        # Additional palette colors for better contrast
        sunset_palette.setColor(QPalette.Light, QColor(90, 45, 90))
        sunset_palette.setColor(QPalette.Midlight, QColor(80, 40, 80))
        sunset_palette.setColor(QPalette.Mid, QColor(70, 35, 70))
        sunset_palette.setColor(QPalette.Dark, QColor(40, 18, 40))
        sunset_palette.setColor(QPalette.Shadow, QColor(30, 15, 30))

        QApplication.setPalette(sunset_palette)
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
        
        # Additional stylesheet for finer control of specific widgets
        stylesheet = """
        QToolTip { 
            border: 1px solid #76797C; 
            background-color: #2A2C32; 
            color: #F0F0F0; 
            padding: 5px;
            opacity: 200;
        }
        
        QScrollBar:vertical {
            background: #1A1D21;
            width: 12px;
            margin: 0px;
        }
        
        QScrollBar::handle:vertical {
            background: #3D444D;
            min-height: 20px;
            border-radius: 4px;
            margin: 2px;
        }
        
        QScrollBar::handle:vertical:hover {
            background: #4A5157;
        }
        
        /* Fix for annotation dock - ensure text is visible */
        QDockWidget[objectName="annotationDock"] QTextEdit, 
        QDockWidget[objectName="annotationDock"] QPlainTextEdit,
        QDockWidget[objectName="annotationDock"] QLineEdit {
            background-color: #2A2C32;
            color: #E0E0E0;
            border: 1px solid #3D444D;
        }
        """
        
        # Get the instance of QApplication and apply the stylesheet
        app = QApplication.instance()
        if app:
            app.setStyleSheet(stylesheet)
        
        return True

    @classmethod
    def get_available_styles(cls):
        """Get a list of all available styles."""
        return ["Default", "Fusion", "Windows", "Dark", 
                "Light", "Blue", "Green", "Sunset","DarkModern"]

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

