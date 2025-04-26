"""
Main build script for VIAT that detects the platform and runs the appropriate build script
"""

import os
import platform
import sys
import subprocess
from pathlib import Path
import json
def get_config_directory():
    """
    Get the configuration directory for the application.
    
    Returns:
        str: Path to the configuration directory
    """
    # Use platform-specific config directory
    if os.name == 'nt':  # Windows
        config_dir = os.path.join(os.environ['APPDATA'], 'VideoAnnotationTool')
    else:  # macOS, Linux, etc.
        config_dir = os.path.join(os.path.expanduser('~'), '.config', 'VideoAnnotationTool')
    
    return config_dir

def clean_recent_projects():
    """Clean up recent projects configuration file"""
    print("Cleaning recent projects configuration...")
    
    # Get config directory using the exact same function as the application
    config_dir = get_config_directory()
    
    # Create the directory if it doesn't exist
    os.makedirs(config_dir, exist_ok=True)
    
    recent_projects_file = os.path.join(config_dir, "recent_projects.json")
    last_state_file = os.path.join(config_dir, "last_state.json")
    
    # Clear recent projects
    with open(recent_projects_file, "w") as f:
        json.dump([], f)
    print(f"Cleared recent projects in {recent_projects_file}")
    
    # Clear last state
    with open(last_state_file, "w") as f:
        json.dump({}, f)
    print(f"Cleared last state in {last_state_file}")
def main():
    """Main function to detect platform and build appropriate package"""
    print("VIAT Build Script")
    print("=================")
    
    # Ensure we're in the right directory
    os.chdir(Path(__file__).parent)
    
    # Check if PyInstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pyinstaller"], check=True)
    
    # Detect platform
    system = platform.system()
    clean_recent_projects()
    if system == "Windows":
        print("Detected Windows platform")
        from build_windows import build_executable, create_installer
        
        # if build_executable():
        create_installer()
    
    elif system == "Linux":
        print("Detected Linux platform")
        # Check if it's Ubuntu/Debian
        try:
            with open("/etc/os-release") as f:
                os_info = f.read()
            
            if "Ubuntu" in os_info or "Debian" in os_info:
                from build_ubuntu import build_executable, create_deb_package
                
                if build_executable():
                    create_deb_package()
            else:
                print("This Linux distribution is not supported for packaging.")
                print("Building standalone executable only...")
                from build_ubuntu import build_executable
                build_executable()
        except FileNotFoundError:
            print("Could not determine Linux distribution.")
            print("Building standalone executable only...")
            from build_ubuntu import build_executable
            build_executable()
    
    else:
        print(f"Platform {system} is not supported for packaging.")

if __name__ == "__main__":
    main()