#!/usr/bin/env python3
"""
Build a Windows executable from Ubuntu using PyInstaller with Wine
"""
import os
import sys
import subprocess
from pathlib import Path

def build_windows_executable():
    """Build a Windows executable using PyInstaller with Wine"""
    print("Building Windows executable for VIAT from Ubuntu...")
    
    # Check if Wine is installed
    try:
        subprocess.run(["wine", "--version"], stdout=subprocess.PIPE, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Wine is not installed. Please install it with: sudo apt install wine64")
        return
    
    # Install Python for Windows using Wine
    python_installed = False
    try:
        subprocess.run(["wine", "python", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        python_installed = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Python for Windows is not installed in Wine.")
        print("Please download and install Python for Windows using Wine first.")
        print("Example: wine python-3.9.7-amd64.exe")
        return
    
    if not python_installed:
        return
    
    # Install PyInstaller in the Windows Python environment
    try:
        subprocess.run(["wine", "pip", "install", "pyinstaller"], check=True)
    except subprocess.CalledProcessError:
        print("Failed to install PyInstaller in the Windows Python environment.")
        return
    
    # Get the current directory
    current_dir = Path(__file__).parent.absolute()
    
    # Create a spec file for PyInstaller
    spec_content = f"""
# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['{current_dir / "run.py"}'],
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
)
"""
    
    # Write the spec file
    spec_file = current_dir / "viat_windows.spec"
    with open(spec_file, "w") as f:
        f.write(spec_content)
    
    # Run PyInstaller through Wine
    print("Running PyInstaller through Wine...")
    try:
        subprocess.run(
            ["wine", "pyinstaller", "--clean", "--noconfirm", str(spec_file)], 
            cwd=current_dir,
            check=True
        )
        print("Build completed successfully!")
        print(f"Executable created at: {current_dir}/dist/VIAT.exe")
    except subprocess.CalledProcessError as e:
        print(f"Error building executable: {e}")

if __name__ == "__main__":
    build_windows_executable()
