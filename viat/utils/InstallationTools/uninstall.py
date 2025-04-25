#!/usr/bin/env python3
"""
Uninstallation script for VIAT on Ubuntu.
This script should be run with sudo privileges.
"""
import os
import sys
from pathlib import Path

def uninstall_viat():
    if os.geteuid() != 0:
        print("This script must be run with sudo privileges.")
        print("Please run: sudo python3 uninstall.py")
        sys.exit(1)
    
    print("Uninstalling VIAT...")
    
    # Remove the executable
    executable = Path("/usr/local/bin/viat")
    if executable.exists():
        os.remove(executable)
        print("Removed executable.")
    
    # Remove the desktop entry
    desktop_file = Path("/usr/share/applications/viat.desktop")
    if desktop_file.exists():
        os.remove(desktop_file)
        print("Removed desktop entry.")
    
    # Remove the icon
    icon_file = Path("/usr/local/share/icons/viat.png")
    if icon_file.exists():
        os.remove(icon_file)
        print("Removed icon.")
    
    print("\nVIAT has been successfully uninstalled.")

if __name__ == "__main__":
    uninstall_viat()