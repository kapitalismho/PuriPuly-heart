#ifndef BetaPayload
  #error BetaPayload must name the verified local payload
#endif
#ifndef BetaCommit
  #error BetaCommit must name the source commit
#endif
#ifndef BetaOutput
  #error BetaOutput must name an isolated output directory
#endif

[Setup]
AppId=PuriPulyHeart-Fork-Desktop-Beta
AppName=PuriPuly Heart Beta
AppVersion=2.8.2-beta-{#BetaCommit}
AppPublisher=RICHARDwuxiaofei
DefaultDirName={localappdata}\Programs\PuriPulyHeartBeta
DefaultGroupName=PuriPuly Heart Beta
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir={#BetaOutput}
OutputBaseFilename=PuriPulyHeart-Beta-Setup-2.8.2-{#BetaCommit}
Compression=lzma2/fast
SolidCompression=yes
UninstallDisplayIcon={app}\PuriPulyHeartPreview.exe
SetupLogging=yes
WizardStyle=modern
LicenseFile=..\..\LICENSE

[Files]
Source: "{#BetaPayload}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\PuriPuly Heart Beta"; Filename: "{app}\PuriPulyHeartPreview.exe"
Name: "{autodesktop}\PuriPuly Heart Beta"; Filename: "{app}\PuriPulyHeartPreview.exe"

[Run]
Filename: "{app}\PuriPulyHeartPreview.exe"; Description: "Launch PuriPuly Heart Beta"; Flags: nowait postinstall skipifsilent unchecked
