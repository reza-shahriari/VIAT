#!/usr/bin/env python3
"""
Create a .deb package for VIAT on Ubuntu
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path
import tempfile

def create_deb_package():
    """Create a .deb package for VIAT"""
    print("Creating .deb package for VIAT...")
    
    # Get the current directory
    current_dir = Path(__file__).parent.absolute()
    
    # Check if the executable exists
    executable = current_dir.parent.parent / "dist" / "viat"
    if not executable.exists():
        print("Executable not found. Please run build_executable_ubuntu.py first.")
        return
    
    # Create a temporary directory for the package structure
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create the package directory structure
        pkg_name = "viat"
        pkg_version = "1.0.0"
        pkg_dir = temp_path / f"{pkg_name}_{pkg_version}"
        
        # Create directories
        (pkg_dir / "DEBIAN").mkdir(parents=True)
        (pkg_dir / "usr/local/bin").mkdir(parents=True)
        (pkg_dir / "usr/share/applications").mkdir(parents=True)
        (pkg_dir / "usr/share/icons/hicolor/128x128/apps").mkdir(parents=True)
        
        # Create control file
        control_content = f"""Package: {pkg_name}
Version: {pkg_version}
Section: graphics
Priority: optional
Architecture: amd64
Depends: libc6, libgl1
Maintainer: VIAT Team <your-email@example.com>
Description: Video-Image Annotation Tool
 VIAT is a tool for annotating videos and images with bounding boxes
 and other metadata. It provides a user-friendly interface for creating,
 editing, and exporting annotations in various formats.
"""
        
        with open(pkg_dir / "DEBIAN/control", "w") as f:
            f.write(control_content)
        
        # Copy executable
        shutil.copy2(executable, pkg_dir / "usr/local/bin/viat")
        os.chmod(pkg_dir / "usr/local/bin/viat", 0o755)
        
        # Copy desktop file
        desktop_file = current_dir.parent.parent / "viat.desktop"
        if desktop_file.exists():
            shutil.copy2(desktop_file, pkg_dir / "usr/share/applications/")
        else:
            # Create desktop file if it doesn't exist
            desktop_content = """[Desktop Entry]
Name=VIAT
Comment=Video-Image Annotation Tool
Exec=/usr/local/bin/viat
Terminal=false
Type=Application
Categories=Graphics;VideoEditing;
"""
            with open(pkg_dir / "usr/share/applications/viat.desktop", "w") as f:
                f.write(desktop_content)
        
        # Look for an icon file
        icon_found = False
        for icon_file in current_dir.glob("*.png"):
            if icon_found:
                break
            # Copy the first PNG file found as the icon
            shutil.copy2(icon_file, pkg_dir / "usr/share/icons/hicolor/128x128/apps/viat.png")
            icon_found = True
        
        if not icon_found:
            print("Warning: No icon file found. The application will use the default icon.")
            # Create a simple placeholder icon
            try:
                from PIL import Image, ImageDraw
                img = Image.new('RGB', (128, 128), color=(73, 109, 137))
                d = ImageDraw.Draw(img)
                d.rectangle([10, 10, 118, 118], outline=(255, 255, 255), width=3)
                d.text((35, 55), "VIAT", fill=(255, 255, 255))
                img.save(pkg_dir / "usr/share/icons/hicolor/128x128/apps/viat.png")
            except ImportError:
                print("PIL not installed. Skipping icon creation.")
        
        # Create postinst script for additional setup if needed
        postinst_content = """#!/bin/bash
# Update icon cache
if [ -x /usr/bin/update-icon-caches ]; then
    /usr/bin/update-icon-caches /usr/share/icons/hicolor
fi
"""
        with open(pkg_dir / "DEBIAN/postinst", "w") as f:
            f.write(postinst_content)
        os.chmod(pkg_dir / "DEBIAN/postinst", 0o755)
        
        # Build the package
        print("Building .deb package...")
        output_deb = current_dir / f"{pkg_name}_{pkg_version}_amd64.deb"
        try:
            subprocess.run(["dpkg-deb", "--build", str(pkg_dir), str(output_deb)], check=True)
            print(f"Package created successfully: {output_deb}")
        except subprocess.CalledProcessError as e:
            print(f"Error creating .deb package: {e}")
            print("Make sure dpkg-deb is installed. You can install it with: sudo apt-get install dpkg-dev")

if __name__ == "__main__":
    create_deb_package()