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

            # Copy all files from the distribution
            File /r "..\installation_tools\dist\*.*"
            
            # Create Icon directory and copy icons
            CreateDirectory "$INSTDIR\Icon"
            SetOutPath "$INSTDIR\Icon"
            File /r "${ICON_PATH}\*.*"
            
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
