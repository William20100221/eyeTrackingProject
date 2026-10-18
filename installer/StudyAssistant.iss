; Inno Setup script for Study Assistant (run by installer/build.py, or:
;   iscc /DAppVersion=0.1.0 installer\StudyAssistant.iss )
; Per-user install, no admin password: the app goes to %LOCALAPPDATA%\Programs,
; your data (%LOCALAPPDATA%\StudyAssistant) and settings are never touched by
; an upgrade or an uninstall. See docs/installer_dev_log.md.

#define AppName "Study Assistant"
#define AppExe "StudyAssistant.exe"
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif

[Setup]
; The AppId identifies the app across versions: a newer installer upgrades in place. Never change it.
AppId={{8D9CB1D4-F4D0-44EC-92FC-B47B2FB20ABA}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=William20100221
AppPublisherURL=https://github.com/William20100221/eyeTrackingProject
AppSupportURL=https://github.com/William20100221/eyeTrackingProject/issues
DefaultDirName={localappdata}\Programs\StudyAssistant
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=Output
OutputBaseFilename=StudyAssistant-Setup-{#AppVersion}
SetupIconFile=app.ico
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
; an open Study Assistant is closed before files are replaced, and started again after
CloseApplications=yes
RestartApplications=no

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Shortcuts:"; Flags: unchecked

[InstallDelete]
; the previous version's bundle goes first, so no stale library is left behind on upgrade
Type: filesandordirs; Name: "{app}\_internal"

[Files]
Source: "..\dist\StudyAssistant\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
; the same app id the app sets at start, so the taskbar groups the shortcut and the window together
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExe}"; AppUserModelID: "StudyAssistant.EyeTracking"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; AppUserModelID: "StudyAssistant.EyeTracking"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Start {#AppName}"; Flags: nowait postinstall skipifsilent

[Messages]
FinishedLabel=Study Assistant is installed. Your calibrations and chat history are kept in %LOCALAPPDATA%\StudyAssistant, so they stay when you upgrade or uninstall.
