!include "MUI2.nsh"
!include "x64.nsh"

Name "Video Annotation Tool (VIAT)"
OutFile "VIAT_Setup.exe"
InstallDir "$PROGRAMFILES64\VIAT"
InstallDirRegKey HKLM "Software\VIAT" "Install_Dir"
RequestExecutionLevel admin

Icon "D:\VIAT\viat\Icon\icon.ico" ; <-- SET installer icon too if needed
!define MUI_ICON "D:\VIAT\viat\Icon\icon.ico"

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
    File /r "D:\VIAT\installation_tools\dist\*.*"

    # Create shortcuts AFTER copying files
    CreateDirectory "$SMPROGRAMS\VIAT"
    CreateShortcut "$SMPROGRAMS\VIAT\VIAT.lnk" "$INSTDIR\VIAT.exe" "" "$INSTDIR\Icon\app_icon.ico"
    CreateShortcut "$DESKTOP\VIAT.lnk" "$INSTDIR\VIAT.exe" "" "$INSTDIR\Icon\app_icon.ico"

    # Save install path
    WriteRegStr HKLM "Software\VIAT" "Install_Dir" "$INSTDIR"

    # Create uninstaller
    WriteUninstaller "$INSTDIR\Uninstall.exe"

    # Add uninstaller to Programs & Features
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\VIAT" "DisplayName" "Video Annotation Tool (VIAT)"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\VIAT" "UninstallString" "$INSTDIR\Uninstall.exe"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\VIAT" "DisplayIcon" "$INSTDIR\Icon\app_icon.ico"
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
