#!/usr/bin/env python3
"""
Installation script for VIAT on Ubuntu.
This script should be run with sudo privileges.
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

def install_viat():
    if os.geteuid() != 0:
        print("This script must be run with sudo privileges.")
        print("Please run: sudo python3 install.py")
        sys.exit(1)
    
    print("Installing VIAT...")
    
    # Get the current directory
    current_dir = Path(__file__).parent.absolute()
    
    # Check if the executable exists
    executable = current_dir.parent.parent / "dist" / "viat"
    if not executable.exists():
        print("Executable not found. Please run build_executable.py first.")
        sys.exit(1)
    
    # Copy the executable to /usr/local/bin
    print("Copying executable to /usr/local/bin...")
    os.makedirs("/usr/local/bin", exist_ok=True)
    shutil.copy2(executable, "/usr/local/bin/viat")
    os.chmod("/usr/local/bin/viat", 0o755)
    
    # Copy the desktop entry
    print("Installing desktop entry...")
    desktop_file = current_dir.parent.parent / "viat.desktop"
    if desktop_file.exists():
        shutil.copy2(desktop_file, "/usr/share/applications/")
    
    # Create icon directory if it doesn't exist
    os.makedirs("/usr/local/share/icons", exist_ok=True)
    
    # Check if an icon file exists in the current directory
    icon_files = list(current_dir.glob("*.png")) + list(current_dir.glob("*.svg"))
    if icon_files:
        # Use the first icon file found
        icon_file = icon_files[0]
        print(f"Using icon: {icon_file}")
        shutil.copy2(icon_file, "/usr/local/share/icons/viat.png")
    else:
        print("No icon file found. You may want to add one later.")
    
    print("\nVIAT has been successfully installed!")
    print("You can now run it by typing 'viat' in the terminal")
    print("or by finding it in your applications menu.")

if __name__ == "__main__":
    install_viat()