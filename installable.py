import sys
from cx_Freeze import setup, Executable

# Dependencies
build_exe_options = {
    "packages": ["os", "sys", "PyQt5", "cv2", "random"],
    "excludes": [],
    "include_files": [("vat/Icon", "Icon")]
}

# Base for GUI applications
base = None
if sys.platform == "win32":
    base = "Win32GUI"

setup(
    name="VideoAnnotationTool",
    version="1.0",
    description="Video Annotation Tool",
    options={"build_exe": build_exe_options},
    executables=[
        Executable(
            "vat/run.py", 
            base=base,
            target_name="VideoAnnotationTool.exe",
            icon="vat/Icon/Icon.png"
        )
    ]
)
