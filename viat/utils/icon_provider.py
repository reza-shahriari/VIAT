from PyQt5.QtGui import QIcon
import os
import sys
import qtawesome as qta

def get_icon(icon_name):
    """
    Get an icon by name, using QtAwesome icons with fallback to custom icons.
    
    Args:
        icon_name (str): Name of the icon to retrieve
        
    Returns:
        QIcon: The requested icon
    """
    # Map standard icon names to Font Awesome icons
    fa_icon_map = {
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
    
    # Try to get icon from QtAwesome
    if icon_name in fa_icon_map:
        try:
            return qta.icon(fa_icon_map[icon_name])
        except Exception:
            # Fall back to custom icons if QtAwesome fails
            pass
    
    # Original custom icon logic as fallback
    # Map standard icon names to our custom icon files
    icon_map = {
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
    
    # Get the icon filename
    icon_file = icon_map.get(icon_name, f"{icon_name}.png")
    
    # Determine the base path based on whether we're running from source or installed
    if getattr(sys, 'frozen', False):
        # Running as compiled executable
        base_path = os.path.dirname(sys.executable)
        # Try multiple possible icon locations
        icon_paths = [
            os.path.join(base_path, "Icon", icon_file)
        ]
    else:
        # Running in development environment
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        icon_paths = [os.path.join(base_path, "Icon", icon_file)]
    
    # Try each possible path
    for icon_path in icon_paths:
        if os.path.exists(icon_path):
            return QIcon(icon_path)
    
    # If not found, try system theme as fallback (works on Linux)
    return QIcon.fromTheme(icon_name, QIcon())
