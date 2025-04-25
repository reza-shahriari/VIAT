#!/usr/bin/env python3
"""
All-in-one packaging script for VIAT.
Detects the platform and creates the appropriate package.
"""
import os
import sys
import platform
import subprocess
from pathlib import Path

def erase_saved_data():
    """Erase all saved VIAT application data"""
    try:
        from viat.utils.file_operations import get_config_directory
        
        config_dir = get_config_directory()
        if os.path.exists(config_dir):
            print(f"Erasing saved application data from: {config_dir}")
            try:
                import shutil
                shutil.rmtree(config_dir)
                print("✓ Saved data successfully erased")
            except Exception as e:
                print(f"Error erasing data: {e}")
        else:
            print("No saved application data found")
    except ImportError:
        print("Could not import file_operations module. Skipping data erasure.")

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
    
    # Check for other dependencies based on platform
    if platform.system() == "Windows":
        # Check for Inno Setup on Windows
        inno_setup_path = None
        possible_paths = [
            r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
            r"C:\Program Files\Inno Setup 6\ISCC.exe",
            r"C:\Program Files (x86)\Inno Setup 5\ISCC.exe",
            r"C:\Program Files\Inno Setup 5\ISCC.exe"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                inno_setup_path = path
                break
        
        if not inno_setup_path:
            print("Warning: Inno Setup not found. You won't be able to create an installer.")
            print("Please install Inno Setup from: https://jrsoftware.org/isdl.php")
    
    elif platform.system() == "Linux":
        # Check for dpkg-deb on Linux
        try:
            subprocess.run(["dpkg-deb", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Warning: dpkg-deb not found. You won't be able to create a .deb package.")
            print("Install it with: sudo apt-get install dpkg-dev")

def package_application():
    """Package the application based on the detected platform"""
    system = platform.system()
    
    print(f"Detected platform: {system}")
    print("Packaging VIAT application...")
    
    # Erase saved data first
    erase_saved_data()
    
    # Check dependencies
    check_dependencies()
    
    # Get the current directory
    current_dir = Path(__file__).parent.absolute()
    
    if system == "Windows":
        # Build Windows executable
        print("\n=== Building Windows Executable ===")
        from build_executable_windows import build_executable
        build_executable()
        
        # Create Windows installer
        print("\n=== Creating Windows Installer ===")
        from create_installer import create_windows_installer
        create_windows_installer()
        
    elif system == "Linux":
        # Build Linux executable
        print("\n=== Building Linux Executable ===")
        from build_executable_ubuntu import build_executable
        build_executable()
        
        # Create DEB package
        print("\n=== Creating DEB Package ===")
        from create_deb_package import create_deb_package
        create_deb_package()
        
    elif system == "Darwin":  # macOS
        print("macOS packaging is not implemented yet.")
        # TODO: Implement macOS packaging
    
    else:
        print(f"Unsupported platform: {system}")
        return
    
    print("\nPackaging completed!")

if __name__ == "__main__":
    package_application()