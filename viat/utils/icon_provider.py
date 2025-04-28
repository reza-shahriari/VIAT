from PyQt5.QtGui import QIcon
import os
import sys
import qtawesome as qta

class IconProvider:
    """Centralized icon management for the application"""
    
    def __init__(self):
        self.theme = "light"
        
        # Map standard icon names to Font Awesome icons
        self.fa_icon_map = {
            "media-playback-start": "fa5s.play",
            "media-playback-pause": "fa5s.pause",
            "media-skip-backward": "fa5s.step-backward",
            "media-skip-forward": "fa5s.step-forward",
            "edit": "fa5s.edit",
            "add": "fa5s.plus",
            "delete": "fa5s.trash",
            "zoom-in": "fa5s.search-plus",
            "zoom-out": "fa5s.search-minus",
            "zoom-original": "fa5s.expand",
            # Add more mappings as needed
        }
        
        # Map standard icon names to custom icon files
        self.icon_map = {
            "media-playback-start": "play.png",
            "media-playback-pause": "pause.png",
            "media-skip-backward": "previous.png",
            "media-skip-forward": "next.png",
            "edit": "edit.png",
            "add": "add.png",
            "delete": "delete.png",
            "zoom-in": "zoom-in.png",
            "zoom-out": "zoom-out.png",
            "zoom-original": "zoom-reset.png"
        }
        
        # Determine the base path for icons
        if getattr(sys, 'frozen', False):
            # Running as compiled executable
            self.base_path = os.path.dirname(sys.executable)
            self.icon_base_path = os.path.join(self.base_path, "Icon")
        else:
            # Running in development environment
            self.base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.icon_base_path = os.path.join(self.base_path, "Icon")
    
    def set_theme(self, theme):
        """Set the current theme for icons"""
        self.theme = "dark" if theme.lower() == "dark" else "light"
        return self  # Return self for method chaining
        
    def get_icon(self, icon_name):
        """Get an icon by name, using QtAwesome icons with fallback to custom icons"""
        # Determine icon color based on theme
        icon_color = "white" if self.theme == "dark" else "black"
        icon_color = "gray"
        
        # Try to get icon from QtAwesome with theme-appropriate color
        if icon_name in self.fa_icon_map:
            try:
                return qta.icon(self.fa_icon_map[icon_name], color=icon_color)
            except Exception:
                # Fall back to custom icons if QtAwesome fails
                pass
                
        # Get the icon filename
        icon_file = self.icon_map.get(icon_name, f"{icon_name}.png")
        
        # Check if the icon file exists
        icon_path = os.path.join(self.icon_base_path, icon_file)
        if os.path.exists(icon_path):
            return QIcon(icon_path)
        
        # If not found, try system theme as fallback (works on Linux)
        return QIcon.fromTheme(icon_name, QIcon())
