#!/usr/bin/env python3
"""
Build script to create a standalone executable for VIAT on Windows.
Includes functionality to erase saved application data.
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

def erase_saved_data():
    """Erase all saved VIAT application data"""
    from utils.file_operations import get_config_directory
    
    config_dir = get_config_directory()
    if os.path.exists(config_dir):
        print(f"Erasing saved application data from: {config_dir}")
        try:
            shutil.rmtree(config_dir)
            print("✓ Saved data successfully erased")
        except Exception as e:
            print(f"Error erasing data: {e}")
    else:
        print("No saved application data found")

def build_executable():
    print("Building VIAT executable for Windows...")
    
    # Erase saved data first
    erase_saved_data()
    
    # Ensure PyInstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("PyInstaller not found. Installing...")
        subprocess.run(["pip", "install", "pyinstaller"], check=True)
    
    # Get the current directory
    current_dir = Path(__file__).parent.absolute()
    
    # Look for an icon file
    icon_path = None
    for icon_file in current_dir.glob("*.ico"):
        icon_path = icon_file
        break
    
    if not icon_path:
        print("No .ico file found. The executable will use the default icon.")
        icon_option = ""
    else:
        print(f"Using icon: {icon_path}")
        icon_option = f"--icon={icon_path}"
    
    # Create a spec file for PyInstaller
    spec_content = f"""
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['{current_dir.parent.parent / "run.py"}'],
    pathex=['{current_dir}'],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='VIAT',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon={"'" + str(icon_path) + "'" if icon_path else "None"},
)
"""
    
    # Write the spec file
    spec_file = current_dir / "viat_windows.spec"
    with open(spec_file, "w") as f:
        f.write(spec_content)
    
    # Run PyInstaller
    print("Running PyInstaller...")
    subprocess.run(
        ["pyinstaller", "--clean", "--noconfirm", "viat_windows.spec"], 
        cwd=current_dir,
        check=True
    )
    
    print("Build completed successfully!")
    print(f"Executable created at: {current_dir}/dist/VIAT.exe")

if __name__ == "__main__":
    build_executable()