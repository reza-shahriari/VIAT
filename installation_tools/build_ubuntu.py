"""
Build script for creating Ubuntu executable and installer for VIAT
"""

import os
import subprocess
import shutil
from pathlib import Path

def build_executable():
    """Build the Ubuntu executable using PyInstaller"""
    print("Building Ubuntu executable...")
    
    # Clean existing dist directory if it exists
    if os.path.exists("dist"):
        print("Cleaning existing dist directory...")
        shutil.rmtree("dist")
    
    # Run PyInstaller with the unified spec file
    spec_file = "viat.spec"
    if not os.path.exists(spec_file):
        print(f"ERROR: Spec file '{spec_file}' not found!")
        return False
        
    subprocess.run(["pyinstaller", spec_file], check=True)
    
    # Check if the build was successful
    if os.path.exists("dist") and "VIAT" in os.listdir("dist"):
        print("Ubuntu executable built successfully!")
        return True
    else:
        print("ERROR: PyInstaller did not create expected files")
        print("Current directory:", os.getcwd())
        print("Contents of dist directory (if it exists):")
        if os.path.exists("dist"):
            print(os.listdir("dist"))
        return False
    
def create_deb_package():
    """Create Debian package for Ubuntu"""
    print("Creating Debian package...")
    
    # Create directory structure for the package
    os.makedirs("deb_dist/DEBIAN", exist_ok=True)
    os.makedirs("deb_dist/usr/local/bin", exist_ok=True)
    os.makedirs("deb_dist/usr/share/applications", exist_ok=True)
    os.makedirs("deb_dist/usr/share/pixmaps", exist_ok=True)
    os.makedirs("deb_dist/usr/share/viat", exist_ok=True)
    
    # Check if dist/VIAT is a file or directory
    if os.path.isfile("dist/VIAT"):
        # It's a single executable file
        shutil.copy("dist/VIAT", "deb_dist/usr/share/viat/VIAT")
        os.chmod("deb_dist/usr/share/viat/VIAT", 0o755)
    elif os.path.isdir("dist/VIAT"):
        # It's a directory with multiple files
        shutil.copytree("dist/VIAT", "deb_dist/usr/share/viat", dirs_exist_ok=True)
    else:
        raise FileNotFoundError("Could not find the built executable at dist/VIAT")
    
    # Create executable symlink
    with open("deb_dist/usr/local/bin/viat", "w") as f:
        f.write("#!/bin/bash\n/usr/share/viat/VIAT \"$@\"\n")
    os.chmod("deb_dist/usr/local/bin/viat", 0o755)
    
    # Copy icon
    icon_path = "../viat/Icon/Icon.png"
    if os.path.exists(icon_path):
        shutil.copy(icon_path, "deb_dist/usr/share/pixmaps/viat.png")
    else:
        print(f"Warning: Icon file not found at {icon_path}")
    
    # Create desktop entry
    with open("deb_dist/usr/share/applications/viat.desktop", "w") as f:
        f.write("""[Desktop Entry]
Name=VIAT
Comment=Video Annotation Tool
Exec=/usr/local/bin/viat
Icon=viat
Terminal=false
Type=Application
Categories=Graphics;Video;
""")
    
    # Create control file
    with open("deb_dist/DEBIAN/control", "w") as f:
        f.write("""Package: viat
Version: 1.0.0
Section: graphics
Priority: optional
Architecture: amd64
Depends: libgl1-mesa-glx, libxkbcommon-x11-0
Maintainer: VIAT Team
Description: Video Annotation Tool
 A tool for annotating videos with bounding boxes and other annotations.
""")
    
    # Create postinst script
    with open("deb_dist/DEBIAN/postinst", "w") as f:
        f.write("""#!/bin/bash
chmod +x /usr/local/bin/viat
chmod +x /usr/share/viat/VIAT
""")
    os.chmod("deb_dist/DEBIAN/postinst", 0o755)
    
    # Build the package
    try:
        subprocess.run(["dpkg-deb", "--build", "deb_dist", "VIAT_1.0.0_amd64.deb"], check=True)
        print("Debian package created successfully!")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        print("Error: dpkg-deb not found. Please ensure you're on a Debian-based system.")
        return False

if __name__ == "__main__":
    # Ensure we're in the right directory
    os.chdir(Path(__file__).parent)
    
    if build_executable():
        create_deb_package()
