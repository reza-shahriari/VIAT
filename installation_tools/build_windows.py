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
    
    dist_path = os.path.join(os.getcwd(), "dist")
    if os.path.exists(dist_path) and "VIAT.exe" in os.listdir(dist_path):
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

    app_icon_path = os.path.join(Path(os.getcwd()).parent, "viat", "Icon", "Icon.ico")
    icon_folder_path = os.path.join(Path(os.getcwd()).parent, "viat", "Icon",)

    dist_path = os.path.join(os.getcwd(), "dist")
    if not os.path.exists(dist_path) or not os.listdir(dist_path):
        print("Error: dist directory is empty or doesn't exist.")
        print("Cannot create installer without application files.")
        return False


    nsis_script = r"""
        !include "MUI2.nsh"
        !include "x64.nsh"

        Name "Video Annotation Tool (VIAT)"
        OutFile "VIAT_Setup.exe"
        InstallDir "$PROGRAMFILES64\VIAT"
        InstallDirRegKey HKLM "Software\VIAT" "Install_Dir"
        RequestExecutionLevel admin

        # Use relative paths instead of absolute paths
        !define ICON_PATH "..\viat\Icon"
        Icon "${ICON_PATH}\Icon.ico" 
        !define MUI_ICON "${ICON_PATH}\installation_Icon.ico"

         !insertmacro MUI_UNPAGE_CONFIRM
        !insertmacro MUI_UNPAGE_INSTFILES

        !insertmacro MUI_LANGUAGE "English"

        Section "Install"
            SetRegView 64
            SetOutPath "$INSTDIR"

            # Copy all files from the distribution
            File /r "..\installation_tools\dist\*.*"
            # Create Icon directory and copy icons
            CreateDirectory "$INSTDIR\icons"
            SetOutPath "$INSTDIR\icons"
            File /r "..\viat\icons\*.*"
            # Also copy to Icon directory for backward compatibility
            CreateDirectory "$INSTDIR\Icon"
            SetOutPath "$INSTDIR\Icon"
            File /r "..\viat\Icon\*.*"
            # Reset output path to install directory
            SetOutPath "$INSTDIR"
           # Create shortcuts AFTER copying files
            CreateDirectory "$SMPROGRAMS\VIAT"
            CreateShortcut "$SMPROGRAMS\VIAT\VIAT.lnk" "$INSTDIR\VIAT.exe" "" "$INSTDIR\Icon\Icon.ico"
            CreateShortcut "$DESKTOP\VIAT.lnk" "$INSTDIR\VIAT.exe" "" "$INSTDIR\Icon\Icon.ico"

            # Save install path
            WriteRegStr HKLM "Software\VIAT" "Install_Dir" "$INSTDIR"

            # Create uninstaller
            WriteUninstaller "$INSTDIR\Uninstall.exe"

            # Add uninstaller to Programs & Features
            WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\VIAT" "DisplayName" "Video Annotation Tool (VIAT)"
            WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\VIAT" "UninstallString" "$INSTDIR\Uninstall.exe"
            WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\VIAT" "DisplayIcon" "$INSTDIR\Icon\Icon.ico"
            WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\VIAT" "NoModify" 1
            WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\VIAT" "NoRepair" 1
        SectionEnd

        Section "Uninstall"
            SetRegView 64

            RMDir /r "$INSTDIR"
            Delete "$SMPROGRAMS\VIAT\VIAT.lnk"
            RMDir "$SMPROGRAMS\VIAT"
            Delete "$DESKTOP\VIAT.lnk"
            
            DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\VIAT"
            DeleteRegKey HKLM "Software\VIAT"
        SectionEnd 
    """
    
    viat_installer_path = os.path.join(os.getcwd(), "viat_installer.nsi")
    
    # with open(viat_installer_path, "w") as f:
            # f.write(nsis_script)

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
    
    if build_executable():
        create_installer()
