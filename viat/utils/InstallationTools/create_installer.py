#!/usr/bin/env python3
"""
Create an installable package for VIAT using Inno Setup (Windows)
"""
import os
import sys
import subprocess
from pathlib import Path
import tempfile

def create_windows_installer():
    """Create a Windows installer using Inno Setup"""
    print("Creating Windows installer for VIAT...")
    
    # Check if Inno Setup is installed
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
        print("Inno Setup not found. Please install it from: https://jrsoftware.org/isdl.php")
        print("After installation, run this script again.")
        return
    
    # Get the current directory
    current_dir = Path(__file__).parent.absolute()
    
    # Check if the executable exists
    exe_path = current_dir.parent.parent / "dist" / "VIAT.exe"
    if not exe_path.exists():
        print("VIAT.exe not found. Please run build_executable_windows.py first.")
        return
    
    # Create Inno Setup script
    script_content = f"""
#define MyAppName "VIAT"
#define MyAppVersion "1.0"
#define MyAppPublisher "VIAT Team"
#define MyAppURL "https://github.com/reza-shahriari/VIAT"
#define MyAppExeName "VIAT.exe"

[Setup]
AppId={{{{8A17D1E2-5C8F-4A47-B3B1-16A1E42F9EBD}}}}
AppName={{#MyAppName}}
AppVersion={{#MyAppVersion}}
AppPublisher={{#MyAppPublisher}}
AppPublisherURL={{#MyAppURL}}
AppSupportURL={{#MyAppURL}}
AppUpdatesURL={{#MyAppURL}}
DefaultDirName={{autopf}}\\{{#MyAppName}}
DisableProgramGroupPage=yes
LicenseFile={current_dir.parent.parent / "LICENSE"}
OutputDir={current_dir}
OutputBaseFilename=VIAT_Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{{cm:CreateDesktopIcon}}"; GroupDescription: "{{cm:AdditionalIcons}}"; Flags: unchecked

[Files]
Source: "{exe_path}"; DestDir: "{{app}}"; Flags: ignoreversion
Source: "{current_dir / "dist" / "*"}"; DestDir: "{{app}}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{{autoprograms}}\\{{#MyAppName}}"; Filename: "{{app}}\\{{#MyAppExeName}}"
Name: "{{autodesktop}}\\{{#MyAppName}}"; Filename: "{{app}}\\{{#MyAppExeName}}"; Tasks: desktopicon

[Run]
Filename: "{{app}}\\{{#MyAppExeName}}"; Description: "{{cm:LaunchProgram,{{#StringChange(MyAppName, '&', '&&')}}}}"; Flags: nowait postinstall skipifsilent
"""
    
    # Create a temporary file for the script
    script_file = current_dir / "viat_installer.iss"
    with open(script_file, "w") as f:
        f.write(script_content)
    
    # Run Inno Setup Compiler
    print("Running Inno Setup Compiler...")
    try:
        subprocess.run([inno_setup_path, str(script_file)], check=True)
        print("Installer created successfully!")
        print(f"Installer location: {current_dir / 'VIAT_Setup.exe'}")
    except subprocess.CalledProcessError as e:
        print(f"Error creating installer: {e}")
    finally:
        # Clean up the script file
        os.remove(script_file)

if __name__ == "__main__":
    create_windows_installer()