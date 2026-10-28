; Inno Setup Script for PuriPuly <3

#include "installer\application-identity.iss"
#ifndef MyPackagedAppDir
  #error Pass MyPackagedAppDir from the native release builder.
#endif
#ifndef MyStagedOverlayDir
  #define MyStagedOverlayDir MyPackagedAppDir
#endif
#if VER < EncodeVer(7, 1, 0)
  #error Native packaging requires Inno Setup 7.1.0 or newer for extended-length bytecode paths.
#endif
#define PackagedApplicationRoot "app\"
#define AppShortcutIdentity 'AppUserModelID: "com.salee.PuriPulyHeart";'
#ifndef NativeCleanupInclude
  #define NativeCleanupInclude MyPackagedAppDir + "\..\native-installer-cleanup.iss"
#endif
#if !FileExists(NativeCleanupInclude)
  #error Native packaging requires the generated native cleanup include. Generate it from the official legacy ownership manifest and final native artifact manifest.
#endif
#ifndef LocalASRModelsInclude
  #define LocalASRModelsInclude MyPackagedAppDir + "\..\native-asr-models.iss"
#endif
#if !FileExists(LocalASRModelsInclude)
  #error Generate the canonical CPU ASR model include before compiling Setup.
#endif
#define AppIconRelativePath PackagedApplicationRoot + "puripuly_heart\data\icons\icon.ico"
#define NotoCjkFontRelativePath PackagedApplicationRoot + "puripuly_heart\data\fonts\NotoSansCJK-Medium.ttc"

#define InstallerPrivacyDir "installer\privacy"
#ifdef InstallerSmokeAppDataRoot
  #if (Len(InstallerSmokeAppDataRoot) < 3) || (Copy(InstallerSmokeAppDataRoot, 2, 2) != ":/")
    #error InstallerSmokeAppDataRoot must be an absolute drive path using forward slashes
  #endif
#endif

[Setup]
; NOTE: AppId uniquely identifies this application.
AppId={#MyAppId}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}/releases
VersionInfoVersion={#MyAppVersion}
VersionInfoProductVersion={#MyAppVersion}
VersionInfoProductName={#MyAppName}
VersionInfoProductTextVersion={#MyAppVersion}
VersionInfoDescription={#MyAppName} Setup
VersionInfoCompany={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppDirName}
DefaultGroupName={#MyAppGroupName}
AllowNoIcons=yes
LicenseFile=LICENSE
OutputDir=installer_output
OutputBaseFilename=PuriPulyHeart-Setup-{#MyAppVersion}
SetupIconFile=src\puripuly_heart\data\icons\icon.ico
UninstallDisplayIcon={app}\PuriPulyHeart.exe
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
; Auto-upgrade: remember previous install location
UsePreviousAppDir=yes
UsePreviousGroup=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"
Name: "japanese"; MessagesFile: "compiler:Languages\Japanese.isl"
Name: "chinesesimplified"; MessagesFile: "installer\Languages\ChineseSimplified.isl"
Name: "chinesetraditional"; MessagesFile: "installer\Languages\ChineseTraditional.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "quicklaunchicon"; Description: "{cm:CreateQuickLaunchIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked; OnlyBelowVersion: 6.1; Check: not IsAdminInstallMode
Name: "redownloadasr"; Description: "{cm:RedownloadAsrTask}"; GroupDescription: "{cm:AsrModelsGroup}"; Flags: unchecked

[CustomMessages]
english.LocalSttDownloadSize=ASR models will be downloaded.%nInstallation requires %1 of disk space.
english.LocalSttNoDownload=All ASR models are installed.
english.LocalSttDiskSpaceFailed=Not enough free space for the ASR model download. %1 is required, but only %2 is available on %3.
english.LocalSttDiskSpaceCheckFailed=Setup could not check free space on %1.
korean.LocalSttDownloadSize=ASR 모델을 다운로드합니다.%n설치에 필요한 용량은 %1입니다.
korean.LocalSttNoDownload=ASR 모델이 모두 설치되어있습니다.
korean.LocalSttDiskSpaceFailed=ASR 모델 다운로드를 위한 여유 공간이 부족합니다. %3에 %1이 필요하지만 %2만 사용할 수 있습니다.
korean.LocalSttDiskSpaceCheckFailed=%1의 여유 공간을 확인할 수 없습니다.
japanese.LocalSttDownloadSize=ASRモデルをダウンロードします。%nインストールには%1の空き容量が必要です。
japanese.LocalSttNoDownload=ASRモデルはすべてインストール済みです。
japanese.LocalSttDiskSpaceFailed=ASRモデルのダウンロードに必要な空き容量がありません。%3には%1が必要ですが、利用可能なのは%2です。
japanese.LocalSttDiskSpaceCheckFailed=%1の空き容量を確認できません。
chinesesimplified.LocalSttDownloadSize=将下载 ASR 模型。%n安装需要 %1 的空间。
chinesesimplified.LocalSttNoDownload=ASR 模型均已安装。
chinesesimplified.LocalSttDiskSpaceFailed=ASR 模型下载空间不足。%3 需要 %1，但只有 %2 可用。
chinesesimplified.LocalSttDiskSpaceCheckFailed=无法检查 %1 的可用空间。
chinesetraditional.LocalSttDownloadSize=將下載 ASR 模型。%n安裝需要 %1 的空間。
chinesetraditional.LocalSttNoDownload=ASR 模型均已安裝。
chinesetraditional.LocalSttDiskSpaceFailed=ASR 模型下載空間不足。%3 需要 %1，但只有 %2 可用。
chinesetraditional.LocalSttDiskSpaceCheckFailed=無法檢查 %1 的可用空間。
english.LocalSttDownloadTitle=Downloading ASR model
english.LocalSttDownloadDescription=
korean.LocalSttDownloadTitle=ASR 모델 다운로드 중
korean.LocalSttDownloadDescription=
japanese.LocalSttDownloadTitle=ASRモデルをダウンロード中
japanese.LocalSttDownloadDescription=
chinesesimplified.LocalSttDownloadTitle=正在下载 ASR 模型
chinesesimplified.LocalSttDownloadDescription=
chinesetraditional.LocalSttDownloadTitle=正在下載 ASR 模型
chinesetraditional.LocalSttDownloadDescription=
english.LocalSttDownloadFailed=Required ASR model download or verification failed. Installation cannot complete.
korean.LocalSttDownloadFailed=필수 ASR 모델 다운로드 또는 검증에 실패했습니다. 설치를 완료할 수 없습니다.
japanese.LocalSttDownloadFailed=必須ASRモデルのダウンロードまたは検証に失敗しました。インストールを完了できません。
chinesesimplified.LocalSttDownloadFailed=必需的 ASR 模型下载或验证失败。无法完成安装。
chinesetraditional.LocalSttDownloadFailed=必要的 ASR 模型下載或驗證失敗。無法完成安裝。
english.AsrModelsGroup=ASR Models
english.RedownloadAsrTask=Re-download ASR models
english.LocalSttRedownloadSize=Re-downloading ASR models.%nInstallation requires %1 of disk space.
korean.AsrModelsGroup=ASR 모델
korean.RedownloadAsrTask=ASR 모델 재다운로드
korean.LocalSttRedownloadSize=ASR 모델을 재다운로드합니다.%n설치에 필요한 용량은 %1입니다.
japanese.AsrModelsGroup=ASRモデル
japanese.RedownloadAsrTask=ASRモデルを再ダウンロード
japanese.LocalSttRedownloadSize=ASRモデルを再ダウンロードします。%nインストールには%1の空き容量が必要です。
chinesesimplified.AsrModelsGroup=ASR 模型
chinesesimplified.RedownloadAsrTask=重新下载 ASR 模型
chinesesimplified.LocalSttRedownloadSize=重新下载 ASR 模型。%n安装需要 %1 的空间。
chinesetraditional.AsrModelsGroup=ASR 模型
chinesetraditional.RedownloadAsrTask=重新下載 ASR 模型
chinesetraditional.LocalSttRedownloadSize=重新下載 ASR 模型。%n安裝需要 %1 的空間。
english.PrivacyPageTitle=Privacy Policy
english.PrivacyInstruction=If you agree to the policy, we send minimal data anonymously.
korean.PrivacyPageTitle=개인정보처리방침
korean.PrivacyInstruction=약관에 동의하면 최소 데이터를 익명으로 전송해요.
japanese.PrivacyPageTitle=プライバシーポリシー
japanese.PrivacyInstruction=規約に同意すると、最小限のデータを匿名で送信します。
chinesesimplified.PrivacyPageTitle=隐私政策
chinesesimplified.PrivacyInstruction=同意本政策后，我们会匿名发送最少量的数据。
chinesetraditional.PrivacyPageTitle=隱私權政策
chinesetraditional.PrivacyInstruction=同意本政策後，我們會匿名傳送最少量的資料。
english.TelemetryPreferenceReadFailed=Setup could not read the existing telemetry preference. Installation cannot continue without preserving it.
korean.TelemetryPreferenceReadFailed=기존 사용 통계 설정을 읽을 수 없습니다. 설정을 보존하지 않고 설치를 계속할 수 없습니다.
japanese.TelemetryPreferenceReadFailed=既存の利用統計設定を読み取れません。設定を維持できないため、インストールを続行できません。
chinesesimplified.TelemetryPreferenceReadFailed=无法读取现有使用统计设置。无法在不保留该设置的情况下继续安装。
chinesetraditional.TelemetryPreferenceReadFailed=無法讀取現有使用統計設定。無法在不保留該設定的情況下繼續安裝。
english.TelemetryPreferenceWriteFailed=Setup could not save the telemetry preference. Installation cannot complete.
korean.TelemetryPreferenceWriteFailed=사용 통계 설정을 저장할 수 없습니다. 설치를 완료할 수 없습니다.
japanese.TelemetryPreferenceWriteFailed=利用統計設定を保存できません。インストールを完了できません。
chinesesimplified.TelemetryPreferenceWriteFailed=无法保存使用统计设置。无法完成安装。
chinesetraditional.TelemetryPreferenceWriteFailed=無法儲存使用統計設定。無法完成安裝。
english.PrivacyPolicyLoadFailed=Setup could not load the Privacy Policy. Installation cannot continue.
korean.PrivacyPolicyLoadFailed=개인정보처리방침을 불러올 수 없습니다. 설치를 계속할 수 없습니다.
japanese.PrivacyPolicyLoadFailed=プライバシーポリシーを読み込めません。インストールを続行できません。
chinesesimplified.PrivacyPolicyLoadFailed=无法加载隐私政策。无法继续安装。
chinesetraditional.PrivacyPolicyLoadFailed=無法載入隱私權政策。無法繼續安裝。

[Files]
Source: "{#InstallerPrivacyDir}\en.txt"; Flags: dontcopy noencryption
Source: "{#InstallerPrivacyDir}\ko.txt"; Flags: dontcopy noencryption
Source: "{#InstallerPrivacyDir}\ja.txt"; Flags: dontcopy noencryption
Source: "{#InstallerPrivacyDir}\zh-CN.txt"; Flags: dontcopy noencryption
Source: "{#InstallerPrivacyDir}\zh-TW.txt"; Flags: dontcopy noencryption
Source: "installer\native-cleanup.ps1"; Flags: dontcopy noencryption
Source: "{#MyPackagedAppDir}\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#MyPackagedAppDir}\{#MyCliExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#MyStagedOverlayDir}\{#MyOverlayExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#MyPackagedAppDir}\{#MyASRWorkerExeName}"; DestDir: "{app}"; Flags: ignoreversion
; Installer build/install never resolves SteamVR paths for openvr_api.dll.
; Bundled CJK font is staged at {#MyPackagedAppDir}\{#NotoCjkFontRelativePath}; the recursive packaged-tree copy installs it to {app}\{#NotoCjkFontRelativePath}.
Source: "{#MyPackagedAppDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "{#MyAppExeName},{#MyCliExeName},{#MyOverlayExeName},{#MyASRWorkerExeName}"
; NOTE: Don't use "Flags: ignoreversion" on any shared system files

[Icons]
#ifndef InstallerSmokeAppDataRoot
Name: "{group}\{#MyAppGroupName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#AppIconRelativePath}"; {#AppShortcutIdentity}
Name: "{group}\{cm:UninstallProgram,{#MyAppGroupName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppGroupName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#AppIconRelativePath}"; Tasks: desktopicon; {#AppShortcutIdentity}
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppGroupName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#AppIconRelativePath}"; Tasks: quicklaunchicon; {#AppShortcutIdentity}
#endif

[UninstallDelete]
; Clean up user config on uninstall (optional)
#ifdef InstallerSmokeAppDataRoot
Type: filesandordirs; Name: "{code:ResolveInstallerSmokeAppDataRoot}"
#else
Type: filesandordirs; Name: "{#MyAppDataRoot}"
#endif

[Code]
var
  PrivacyPage: TWizardPage;
  PrivacyInstructionLabel: TNewStaticText;
  PrivacyPolicyMemo: TRichEditViewer;
  PrivacyAcceptedRadio: TNewRadioButton;
  PrivacyNotAcceptedRadio: TNewRadioButton;
  TelemetryEnabled: Boolean;

#include "installer\native-cleanup.iss"

function DirectoryLooksLikeRepositoryCheckout(Path: String): Boolean;
var
  ProbePath: String;
  ParentPath: String;
  Depth: Integer;
begin
  ProbePath := RemoveBackslashUnlessRoot(Path);
  Result := False;

  if ProbePath = '' then begin
    exit;
  end;

  for Depth := 0 to 8 do begin
    if DirExists(AddBackslash(ProbePath) + '.git') or
       FileExists(AddBackslash(ProbePath) + 'pyproject.toml') or
       FileExists(AddBackslash(ProbePath) + 'AGENTS.md') then begin
      Result := True;
      exit;
    end;

    ParentPath := ExtractFileDir(ProbePath);
    if (ParentPath = '') or (ParentPath = ProbePath) then begin
      exit;
    end;

    ProbePath := ParentPath;
  end;
end;

function PathEqualsOrIsUnder(Path: String; RootPath: String): Boolean;
var
  NormalizedPath: String;
  NormalizedRoot: String;
begin
  NormalizedPath := RemoveBackslashUnlessRoot(Path);
  NormalizedRoot := RemoveBackslashUnlessRoot(RootPath);

  if (NormalizedPath = '') or (NormalizedRoot = '') then begin
    Result := False;
    exit;
  end;

  if CompareText(NormalizedPath, NormalizedRoot) = 0 then begin
    Result := True;
    exit;
  end;

  Result :=
    (Length(NormalizedPath) > Length(NormalizedRoot)) and
    (CompareText(Copy(NormalizedPath, 1, Length(NormalizedRoot)), NormalizedRoot) = 0) and
    (
      (NormalizedRoot[Length(NormalizedRoot)] = '\') or
      (NormalizedPath[Length(NormalizedRoot) + 1] = '\')
    );
end;

function DirectoryLooksLikeTemporaryLocation(Path: String): Boolean;
var
  TempRoot: String;
begin
  Result := False;

  TempRoot := RemoveBackslashUnlessRoot(GetEnv('TEMP'));
  if PathEqualsOrIsUnder(Path, TempRoot) then begin
    Result := True;
    exit;
  end;

  TempRoot := RemoveBackslashUnlessRoot(GetEnv('TMP'));
  if PathEqualsOrIsUnder(Path, TempRoot) then begin
    Result := True;
    exit;
  end;

  TempRoot := RemoveBackslashUnlessRoot(ExpandConstant('{localappdata}\Temp'));
  if PathEqualsOrIsUnder(Path, TempRoot) then begin
    Result := True;
    exit;
  end;

  TempRoot := RemoveBackslashUnlessRoot(ExpandConstant('{tmp}'));
  if PathEqualsOrIsUnder(Path, TempRoot) then begin
    Result := True;
    exit;
  end;

  TempRoot := RemoveBackslashUnlessRoot(ExpandConstant('{win}\Temp'));
  if PathEqualsOrIsUnder(Path, TempRoot) then begin
    Result := True;
    exit;
  end;
end;

#ifdef InstallerSmokeAppDataRoot
function ResolveInstallerSmokeAppDataRoot(Param: String): String;
begin
  Result := '{#InstallerSmokeAppDataRoot}';
  StringChangeEx(Result, '/', '\', True);
  Result := RemoveBackslashUnlessRoot(Result);
  if (ExtractFileDrive(Result) = '') or
     (
       not DirectoryLooksLikeTemporaryLocation(Result) and
       (CompareText(Result, RemoveBackslashUnlessRoot(ExpandConstant('{#MyAppDataRoot}'))) <> 0)
     ) or
     DirectoryLooksLikeRepositoryCheckout(Result) then begin
    RaiseException('Installer smoke build refused an unsafe app-data root: ' + Result);
  end;
end;
#endif

procedure ResetSuspiciousInstallDir();
var
  CandidateDir: String;
  DefaultDir: String;
begin
  CandidateDir := RemoveBackslashUnlessRoot(WizardForm.DirEdit.Text);
  if CandidateDir = '' then begin
#ifdef InstallerSmokeAppDataRoot
    RaiseException('Installer smoke build refused an empty install directory.');
#else
    exit;
#endif
  end;

  DefaultDir := ExpandConstant('{autopf}\{#MyAppDirName}');
  if RemoveBackslashUnlessRoot(DefaultDir) = CandidateDir then begin
#ifdef InstallerSmokeAppDataRoot
    RaiseException('Installer smoke build refused the production default install directory: ' + CandidateDir);
#else
    exit;
#endif
  end;

  if DirectoryLooksLikeRepositoryCheckout(CandidateDir) then begin
#ifdef InstallerSmokeAppDataRoot
    RaiseException('Installer smoke build refused an install directory inside a repository checkout: ' + CandidateDir);
#else
    Log('Resetting suspicious install dir inside a repository checkout: ' + CandidateDir);
    WizardForm.DirEdit.Text := DefaultDir;
    exit;
#endif
  end;

  if DirectoryLooksLikeTemporaryLocation(CandidateDir) then begin
#ifdef InstallerSmokeAppDataRoot
    RaiseException('Installer smoke build refused an install directory inside a temporary directory: ' + CandidateDir);
#else
    Log('Resetting suspicious install dir inside a temporary directory: ' + CandidateDir);
    WizardForm.DirEdit.Text := DefaultDir;
    exit;
#endif
  end;
end;

function ResolveLocalSttAppDataRoot(): String;
#ifndef InstallerSmokeAppDataRoot
var
  OverrideRoot: String;
#endif
begin
#ifdef InstallerSmokeAppDataRoot
  Result := ResolveInstallerSmokeAppDataRoot('');
#else
  OverrideRoot := GetEnv('PURIPULY_HEART_LOCAL_STT_APPDATA_ROOT');
  if OverrideRoot <> '' then begin
    Result := OverrideRoot;
  end else begin
    Result := ExpandConstant('{#MyAppDataRoot}');
  end;
#endif
end;

#include "installer\local-asr.iss"


procedure CurPageChanged(CurPageID: Integer);
var
  DownloadSizeText: String;
begin
  if CurPageID = wpReady then begin
    PrepareLocalSttDownloadPlan();
    if LocalSttRequiredDownloadBytes > 0 then begin
      DownloadSizeText := FormatByteSize(LocalSttRequiredDownloadBytes);
      if WizardIsTaskSelected('redownloadasr') then begin
        WizardForm.ReadyMemo.Text := WizardForm.ReadyMemo.Text + #13#10 +
          FmtMessage(CustomMessage('LocalSttRedownloadSize'), [DownloadSizeText]);
      end else begin
        WizardForm.ReadyMemo.Text := WizardForm.ReadyMemo.Text + #13#10 +
          FmtMessage(CustomMessage('LocalSttDownloadSize'), [DownloadSizeText]);
      end;
    end else begin
      WizardForm.ReadyMemo.Text := WizardForm.ReadyMemo.Text + #13#10 +
        ExpandConstant('{cm:LocalSttNoDownload}');
    end;
  end;
end;

function PrivacyPolicyFileName(): String;
begin
  if ActiveLanguage = 'korean' then begin
    Result := 'ko.txt';
  end else if ActiveLanguage = 'japanese' then begin
    Result := 'ja.txt';
  end else if ActiveLanguage = 'chinesesimplified' then begin
    Result := 'zh-CN.txt';
  end else if ActiveLanguage = 'chinesetraditional' then begin
    Result := 'zh-TW.txt';
  end else begin
    Result := 'en.txt';
  end;
end;

#include "installer\telemetry-preference.iss"

procedure PrivacyChoiceChanged(Sender: TObject);
begin
  TelemetryEnabled := PrivacyAcceptedRadio.Checked;
end;

procedure ApplyTelemetryPreferenceToPrivacyControls();
begin
  PrivacyAcceptedRadio.Checked := TelemetryEnabled;
  PrivacyNotAcceptedRadio.Checked := not TelemetryEnabled;
end;

procedure CreatePrivacyPage();
var
  PolicyFileName: String;
  PolicyLines: TArrayOfString;
  Index: Integer;
begin
  PrivacyPage := CreateCustomPage(
    wpLicense,
    CustomMessage('PrivacyPageTitle'),
    SetupMessage(msgLicenseLabel)
  );
  PolicyFileName := PrivacyPolicyFileName();
  ExtractTemporaryFile(PolicyFileName);
  if not LoadStringsFromFile(AddBackslash(ExpandConstant('{tmp}')) + PolicyFileName, PolicyLines) then begin
    RaiseException(CustomMessage('PrivacyPolicyLoadFailed'));
  end;

  PrivacyInstructionLabel := TNewStaticText.Create(PrivacyPage);
  PrivacyInstructionLabel.Parent := PrivacyPage.Surface;
  PrivacyInstructionLabel.Caption := CustomMessage('PrivacyInstruction');
  PrivacyInstructionLabel.Font.Assign(WizardForm.LicenseLabel1.Font);
  PrivacyInstructionLabel.SetBounds(
    WizardForm.LicenseLabel1.Left,
    WizardForm.LicenseLabel1.Top,
    WizardForm.LicenseLabel1.Width,
    WizardForm.LicenseLabel1.Height
  );

  PrivacyPolicyMemo := TRichEditViewer.Create(PrivacyPage);
  PrivacyPolicyMemo.Parent := PrivacyPage.Surface;
  PrivacyPolicyMemo.ReadOnly := True;
  PrivacyPolicyMemo.ScrollBars := ssVertical;
  PrivacyPolicyMemo.WordWrap := True;
  PrivacyPolicyMemo.Font.Assign(WizardForm.LicenseMemo.Font);
  PrivacyPolicyMemo.SetBounds(
    WizardForm.LicenseMemo.Left,
    WizardForm.LicenseMemo.Top,
    WizardForm.LicenseMemo.Width,
    WizardForm.LicenseMemo.Height
  );
  for Index := 0 to GetArrayLength(PolicyLines) - 1 do begin
    PrivacyPolicyMemo.Lines.Add(PolicyLines[Index]);
  end;

  PrivacyAcceptedRadio := TNewRadioButton.Create(PrivacyPage);
  PrivacyAcceptedRadio.Parent := PrivacyPage.Surface;
  PrivacyAcceptedRadio.Caption := SetupMessage(msgLicenseAccepted);
  PrivacyAcceptedRadio.Font.Assign(WizardForm.LicenseAcceptedRadio.Font);
  PrivacyAcceptedRadio.SetBounds(
    WizardForm.LicenseAcceptedRadio.Left,
    WizardForm.LicenseAcceptedRadio.Top,
    WizardForm.LicenseAcceptedRadio.Width,
    WizardForm.LicenseAcceptedRadio.Height
  );
  PrivacyAcceptedRadio.OnClick := @PrivacyChoiceChanged;

  PrivacyNotAcceptedRadio := TNewRadioButton.Create(PrivacyPage);
  PrivacyNotAcceptedRadio.Parent := PrivacyPage.Surface;
  PrivacyNotAcceptedRadio.Caption := SetupMessage(msgLicenseNotAccepted);
  PrivacyNotAcceptedRadio.Font.Assign(WizardForm.LicenseNotAcceptedRadio.Font);
  PrivacyNotAcceptedRadio.SetBounds(
    WizardForm.LicenseNotAcceptedRadio.Left,
    WizardForm.LicenseNotAcceptedRadio.Top,
    WizardForm.LicenseNotAcceptedRadio.Width,
    WizardForm.LicenseNotAcceptedRadio.Height
  );
  PrivacyNotAcceptedRadio.OnClick := @PrivacyChoiceChanged;
end;

procedure LoadExistingTelemetryPreference();
var
  Exists: Boolean;
  Enabled: Boolean;
begin
  TelemetryEnabled := True;
  if not ProbeExistingTelemetryPreference(Exists, Enabled) then begin
    RaiseException(CustomMessage('TelemetryPreferenceReadFailed'));
  end;
  if Exists then begin
    TelemetryEnabled := Enabled;
  end;
  ApplyTelemetryPreferenceToPrivacyControls();
end;

procedure PersistTelemetryPreference();
var
  Action: String;
  Parameters: String;
  ResultCode: Integer;
begin
  if TelemetryEnabled then begin
    Action := 'enable';
  end else begin
    Action := 'disable';
  end;
  Parameters := 'installer-telemetry-preference ' + Action;
  if ResolveInstallerTelemetryConfigPath() <> '' then begin
    Parameters := '--config "' + ResolveInstallerTelemetryConfigPath() + '" ' + Parameters;
  end;
  Parameters := '--headless ' + Parameters;
  if not ExecAsOriginalUser(
    ExpandConstant('{app}\{#MyAppExeName}'),
    Parameters,
    ExpandConstant('{app}'),
    SW_HIDE,
    ewWaitUntilTerminated,
    ResultCode
  ) or (ResultCode <> 0) then begin
    RaiseException(CustomMessage('TelemetryPreferenceWriteFailed'));
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then begin
    PersistTelemetryPreference();
    CompleteNativeLegacyCleanup();
  end;
end;

procedure InitializeWizard();
begin
  ResetSuspiciousInstallDir();
  CreatePrivacyPage();
  LoadExistingTelemetryPreference();
  DownloadPage := CreateDownloadPage(
    ExpandConstant('{cm:LocalSttDownloadTitle}'),
    ExpandConstant('{cm:LocalSttDownloadDescription}'),
    @DownloadLocalSttProgress
  );
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  ResetSuspiciousInstallDir();
  Result := RunSetupLocalSttProvisioning();
end;
