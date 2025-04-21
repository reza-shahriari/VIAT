"""Configuration settings for the Video Annotation Tool."""

# Default application settings
DEFAULT_SETTINGS = {
    "default_style": "Default",
    "default_playback_speed": 1.0,
    "window_geometry": (100, 100, 1200, 800),
    "auto_save_interval": 300,  # seconds
}

# Style configurations
STYLE_CONFIGS = {
    "Default": {"function": "set_default_style", "description": "System default style"},
    "Fusion": {
        "function": "set_fusion_style",
        "description": "Modern, flat appearance",
    },
    "Windows": {"function": "set_windows_style", "description": "Native Windows look"},
    "Dark": {
        "function": "set_dark_style",
        "description": "Dark theme for low-light environments",
    },
    "Light": {
        "function": "set_light_style",
        "description": "Light theme for bright environments",
    },
    "Blue": {"function": "set_blue_style", "description": "Blue-themed interface"},
    "Green": {"function": "set_green_style", "description": "Green-themed interface"},
}

# Export format configurations
EXPORT_FORMATS = {
    "COCO JSON": {
        "extension": "json",
        "format_id": "coco",
        "description": "Common Objects in Context format",
    },
    "YOLO TXT": {
        "extension": "txt",
        "format_id": "yolo",
        "description": "YOLO darknet format",
    },
    "Pascal VOC XML": {
        "extension": "xml",
        "format_id": "pascal_voc",
        "description": "Pascal Visual Object Classes format",
    },
}
