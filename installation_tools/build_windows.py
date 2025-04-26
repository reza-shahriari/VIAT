"""
Build script for creating Windows executable and installer for VIAT
"""

import os
import subprocess
import shutil
from pathlib import Path

def build_executable():
    """Build the Windows executable using PyInstaller"""
    print("Building Windows executable...")
    
    # Clean existing dist directory if it exists
    if os.path.exists("dist"):
        print("Cleaning existing dist directory...")
        shutil.rmtree("dist")
    
    # Make sure PyQt5 is properly installed
    subprocess.run(["pip", "install", "PyQt5==5.12.3"], check=True)
    
    # Use the unified spec file
    spec_file = "viat.spec"
    if not os.path.exists(spec_file):
        print(f"ERROR: Spec file '{spec_file}' not found!")
        return False
    
    # Run PyInstaller with the spec file
    subprocess.run(["pyinstaller", spec_file], check=True)
    
    # Check if the build was successful
    if os.path.exists("dist") and "VIAT" in os.listdir("dist"):
        print("Windows executable built successfully!")
        return True
    else:
        print("ERROR: PyInstaller did not create expected files")
        print("Current directory:", os.getcwd())
        print("Contents of dist directory (if it exists):")
        if os.path.exists("dist"):
            print(os.listdir("dist"))
        return False

def create_installer():
    """Create Windows installer using NSIS"""
    print("Creating Windows installer...")
    
    installer_icon_path = os.path.join(Path(os.getcwd()).parent, "viat", "Icon", "installation_Icon.ico")
    # installer_icon_path = installer_icon_path.replace("\\", "/")
    app_icon_path = os.path.join(Path(os.getcwd()).parent, "viat", "Icon", "Icon.ico")
    # app_icon_path = app_icon_path.replace("\\", "/")
    dist_path = os.path.join(os.getcwd(), "dist")
    if not os.path.exists(dist_path) or not os.listdir(dist_path):
        print("Error: dist directory is empty or doesn't exist.")
        print("Cannot create installer without application files.")
        return False
    # dist_path = dist_path.replace("\\", "/")

    nsis_script = f"""
        !include "MUI2.nsh"
        !include "x64.nsh"

        Name "Video Annotation Tool (VIAT)"
        OutFile "VIAT_Setup.exe"
        InstallDir "$PROGRAMFILES64\\VIAT"
        InstallDirRegKey HKLM "Software\\VIAT" "Install_Dir"
        RequestExecutionLevel admin

        Icon {app_icon_path} 
        !define MUI_ICON {installer_icon_path}

        !insertmacro MUI_PAGE_WELCOME
        !insertmacro MUI_PAGE_DIRECTORY
        !insertmacro MUI_PAGE_INSTFILES
        !insertmacro MUI_PAGE_FINISH

        !insertmacro MUI_UNPAGE_CONFIRM
        !insertmacro MUI_UNPAGE_INSTFILES

        !insertmacro MUI_LANGUAGE "English"

        Section "Install"
            SetRegView 64
            SetOutPath "$INSTDIR"

            # Copy all files
            File /r {dist_path}*.*
            File /r {app_icon_path}

            # Create shortcuts AFTER copying files
            CreateDirectory "$SMPROGRAMS\\VIAT"
            CreateShortcut "$SMPROGRAMS\\VIAT\\VIAT.lnk" "$INSTDIR\\VIAT.exe" "" "$INSTDIR\\Icon.ico"
            CreateShortcut "$DESKTOP\\VIAT.lnk" "$INSTDIR\\VIAT.exe" "" "$INSTDIR\\Icon.ico"

            # Save install path
            WriteRegStr HKLM "Software\\VIAT" "Install_Dir" "$INSTDIR"

            # Create uninstaller
            WriteUninstaller "$INSTDIR\\Uninstall.exe"

            # Add uninstaller to Programs & Features
            WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\VIAT" "DisplayName" "Video Annotation Tool (VIAT)"
            WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\VIAT" "UninstallString" "$INSTDIR\\Uninstall.exe"
            WriteRegStr HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\VIAT" "DisplayIcon" "$INSTDIR\\Icon\\app_icon.ico"
            WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\VIAT" "NoModify" 1
            WriteRegDWORD HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\VIAT" "NoRepair" 1
        SectionEnd

        Section "Uninstall"
            SetRegView 64

            RMDir /r "$INSTDIR"
            Delete "$SMPROGRAMS\\VIAT\\VIAT.lnk"
            RMDir "$SMPROGRAMS\\VIAT"
            Delete "$DESKTOP\\VIAT.lnk"
            
            DeleteRegKey HKLM "Software\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\VIAT"
            DeleteRegKey HKLM "Software\\VIAT"
        SectionEnd 
    """
    
    viat_installer_path = os.path.join(os.getcwd(), "viat_installer.nsi")
    
    with open(viat_installer_path, "w") as f:
            f.write(nsis_script)

    try:
        subprocess.run(["makensis", viat_installer_path], check=True)
        print("Windows installer created successfully!")
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        try:
            subprocess.run(["C:\\Program Files (x86)\\NSIS\\makensis.exe", viat_installer_path], check=True)
            print("Windows installer created successfully!")
            return True
        except subprocess.SubprocessError:
            print("Error: NSIS compiler not found. Please install NSIS and try again.")
            return False
        

if __name__ == "__main__":
    # Ensure we're in the right directory
    os.chdir(Path(__file__).parent)
    
    # if build_executable():
    create_installer()
