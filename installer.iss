; Inno Setup Script for PuriPuly <3
; Compile with: ISCC installer.iss

#define MyAppName "PuriPuly <3"
#define MyAppDirName "PuriPulyHeart"
#define MyAppGroupName "PuriPulyHeart"
#define MyAppVersion "2.2.1"
#define MyAppPublisher "salee"
#define MyAppURL "https://github.com/kapitalismho/PuriPuly-heart"
#define MyAppExeName "PuriPulyHeart.exe"
#define MyOverlayExeName "PuriPulyHeartOverlay.exe"
#define MyPackagedAppDir "dist\PuriPulyHeart"
#define MyStagedOverlayDir "build\overlay"
#define NotoCjkFontRelativePath "puripuly_heart\data\fonts\NotoSansCJK-Medium.ttc"
#define LocalSttManifestRelativePath "puripuly_heart\data\models\qwen3-asr-0.6b-int8-sherpa.manifest.json"

#ifndef MyAppId
  #define MyAppId "{{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}"
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

[CustomMessages]
english.LocalSttPageTitle=ASR Model
english.LocalSttPageDescription=Download the built-in ASR model.
english.LocalSttReinstall=Redownload ASR model
korean.LocalSttPageTitle=ASR 모델
korean.LocalSttPageDescription=내장 ASR 모델을 다운로드 합니다.
korean.LocalSttReinstall=ASR 모델 재다운로드
japanese.LocalSttPageTitle=ASRモデル
japanese.LocalSttPageDescription=内蔵ASRモデルをダウンロードします。
japanese.LocalSttReinstall=ASRモデルを再ダウンロード
chinesesimplified.LocalSttPageTitle=ASR 模型
chinesesimplified.LocalSttPageDescription=下载内置 ASR 模型。
chinesesimplified.LocalSttReinstall=重新下载 ASR 模型
chinesetraditional.LocalSttPageTitle=ASR 模型
chinesetraditional.LocalSttPageDescription=下載內建 ASR 模型。
chinesetraditional.LocalSttReinstall=重新下載 ASR 模型
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
english.LocalSttDownloadFailed=ASR model download failed from both Hugging Face and ModelScope. Installation cannot complete.
korean.LocalSttDownloadFailed=Hugging Face와 ModelScope 모두에서 ASR 모델 다운로드에 실패했습니다. 설치를 완료할 수 없습니다.
japanese.LocalSttDownloadFailed=Hugging Face と ModelScope の両方でASRモデルのダウンロードに失敗しました。インストールを完了できません。
chinesesimplified.LocalSttDownloadFailed=从 Hugging Face 和 ModelScope 下载 ASR 模型均失败。无法完成安装。
chinesetraditional.LocalSttDownloadFailed=從 Hugging Face 和 ModelScope 下載 ASR 模型均失敗。無法完成安裝。

[Files]
Source: "{#MyPackagedAppDir}\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#MyStagedOverlayDir}\{#MyOverlayExeName}"; DestDir: "{app}"; Flags: ignoreversion
; Vendored OpenVR runtime DLL comes from dist\PuriPulyHeart\openvr_api.dll in the packaged tree built by build.spec.
; Installer build/install never resolves SteamVR paths for openvr_api.dll.
; Bundled CJK font is staged at {#MyPackagedAppDir}\{#NotoCjkFontRelativePath}; the recursive packaged-tree copy installs it to {app}\{#NotoCjkFontRelativePath}.
Source: "{#MyPackagedAppDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "{#MyAppExeName},{#MyOverlayExeName}"
; NOTE: Don't use "Flags: ignoreversion" on any shared system files

[Icons]
Name: "{group}\{#MyAppGroupName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppGroupName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppGroupName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppGroupName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: quicklaunchicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[InstallDelete]
; Remove the managed default-path VAD cache so the app can rehydrate it from the bundled model.
Type: files; Name: "{localappdata}\puripuly-heart\silero_vad.onnx"
; Remove stale legacy soxr runtime names before laying down the current packaged tree.
Type: files; Name: "{app}\soxr.dll"
Type: files; Name: "{app}\soxr\libsoxr.dll"

[UninstallDelete]
; Clean up user config on uninstall (optional)
Type: filesandordirs; Name: "{localappdata}\puripuly-heart"

[Code]
var
  LocalSttSourcePage: TWizardPage;
  LocalSttReinstallCheckBox: TNewCheckBox;
  DownloadPage: TDownloadWizardPage;

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

procedure ResetSuspiciousInstallDir();
var
  CandidateDir: String;
  DefaultDir: String;
begin
  CandidateDir := RemoveBackslashUnlessRoot(WizardForm.DirEdit.Text);
  if CandidateDir = '' then begin
    exit;
  end;

  DefaultDir := ExpandConstant('{autopf}\{#MyAppDirName}');
  if RemoveBackslashUnlessRoot(DefaultDir) = CandidateDir then begin
    exit;
  end;

  if DirectoryLooksLikeRepositoryCheckout(CandidateDir) then begin
    Log('Resetting suspicious install dir inside a repository checkout: ' + CandidateDir);
    WizardForm.DirEdit.Text := DefaultDir;
    exit;
  end;

  if DirectoryLooksLikeTemporaryLocation(CandidateDir) then begin
    Log('Resetting suspicious install dir inside a temporary directory: ' + CandidateDir);
    WizardForm.DirEdit.Text := DefaultDir;
    exit;
  end;
end;

procedure InitializeLocalSttWizardPage();
begin
  LocalSttSourcePage := CreateCustomPage(
    wpSelectTasks,
    ExpandConstant('{cm:LocalSttPageTitle}'),
    ExpandConstant('{cm:LocalSttPageDescription}')
  );

  LocalSttReinstallCheckBox := TNewCheckBox.Create(LocalSttSourcePage);
  LocalSttReinstallCheckBox.Parent := LocalSttSourcePage.Surface;
  LocalSttReinstallCheckBox.Left := 0;
  LocalSttReinstallCheckBox.Top := ScaleY(8);
  LocalSttReinstallCheckBox.Width := LocalSttSourcePage.SurfaceWidth;
  LocalSttReinstallCheckBox.Checked := False;
  LocalSttReinstallCheckBox.Caption := ExpandConstant('{cm:LocalSttReinstall}');
end;

function GetLocalSttReinstallEnabled(): Boolean;
begin
  Result := False;
  if LocalSttReinstallCheckBox <> nil then begin
    Result := LocalSttReinstallCheckBox.Checked;
  end;
end;

function ResolveLocalSttAppDataRoot(): String;
var
  OverrideRoot: String;
begin
  OverrideRoot := GetEnv('PURIPULY_HEART_LOCAL_STT_APPDATA_ROOT');
  if OverrideRoot <> '' then begin
    Result := OverrideRoot;
  end else begin
    Result := ExpandConstant('{localappdata}\puripuly-heart');
  end;
end;

function GetLocalSttInstallDir(): String;
begin
  Result := AddBackslash(ResolveLocalSttAppDataRoot()) + 'models\qwen3-asr-0.6b-int8-sherpa';
end;

function HuggingFaceLocalSttUrl(RelativePath: String): String;
begin
  Result := 'https://huggingface.co/csukuangfj2/sherpa-onnx-qwen3-asr-0.6B-int8-2026-03-25/resolve/2cc50d1abfe4d4f2df8d71f536d108bb40f943d2/' + RelativePath;
end;

function ModelScopeLocalSttRemotePath(RelativePath: String): String;
begin
  Result := RelativePath;
  if RelativePath = 'conv_frontend.onnx' then begin
    Result := 'model_0.6B/conv_frontend.onnx';
  end else if RelativePath = 'decoder.int8.onnx' then begin
    Result := 'model_0.6B/decoder.int8.onnx';
  end else if RelativePath = 'encoder.int8.onnx' then begin
    Result := 'model_0.6B/encoder.int8.onnx';
  end;
end;

function ModelScopeLocalSttUrl(RelativePath: String): String;
begin
  Result := 'https://www.modelscope.cn/api/v1/models/zengshuishui/Qwen3-ASR-onnx/repo?Revision=c69fb1666ccb59a82c09840c511a6c894e6a2482&FilePath=' + ModelScopeLocalSttRemotePath(RelativePath);
end;

function LocalSttDownloadUrl(SourceName: String; RelativePath: String): String;
begin
  if SourceName = 'modelscope' then begin
    Result := ModelScopeLocalSttUrl(RelativePath);
  end else begin
    Result := HuggingFaceLocalSttUrl(RelativePath);
  end;
end;

function LocalSttSourceRevision(SourceName: String): String;
begin
  if SourceName = 'modelscope' then begin
    Result := 'c69fb1666ccb59a82c09840c511a6c894e6a2482';
  end else begin
    Result := '2cc50d1abfe4d4f2df8d71f536d108bb40f943d2';
  end;
end;

function ValidateLocalSttAsset(BaseDir: String; RelativePath: String; Sha256: String; ExpectedSize: Int64): Boolean;
var
  AssetPath: String;
  ActualSize: Int64;
begin
  AssetPath := AddBackslash(BaseDir) + RelativePath;
  Result := False;
  if not FileExists(AssetPath) then begin
    Log('Local STT asset missing: ' + AssetPath);
    exit;
  end;
  if not FileSize64(AssetPath, ActualSize) then begin
    Log('Local STT asset size could not be read: ' + AssetPath);
    exit;
  end;
  if ActualSize <> ExpectedSize then begin
    Log('Local STT asset size mismatch: ' + AssetPath + ' expected ' + IntToStr(ExpectedSize) + ' found ' + IntToStr(ActualSize));
    exit;
  end;
  if CompareText(GetSHA256OfFile(AssetPath), Sha256) <> 0 then begin
    Log('Local STT asset SHA256 mismatch: ' + AssetPath);
    exit;
  end;
  Result := True;
end;

function ExpectedLocalSttInstalledManifest(SourceName: String): String;
begin
  Result := '{' + #13#10 +
    '  "manifest_version": 1,' + #13#10 +
    '  "model_id": "qwen3-asr-0.6b-int8-sherpa",' + #13#10 +
    '  "engine": "sherpa-onnx",' + #13#10 +
    '  "install_dirname": "qwen3-asr-0.6b-int8-sherpa",' + #13#10 +
    '  "selected_source": "' + SourceName + '",' + #13#10 +
    '  "selected_revision": "' + LocalSttSourceRevision(SourceName) + '"' + #13#10 +
    '}';
end;

function ValidateLocalSttInstalledManifest(BaseDir: String): Boolean;
var
  ManifestPath: String;
  ManifestText: AnsiString;
begin
  Result := False;
  ManifestPath := AddBackslash(BaseDir) + 'installed-manifest.json';
  if not FileExists(ManifestPath) then begin
    Log('Local STT installed manifest is missing: ' + ManifestPath);
    exit;
  end;
  if not LoadStringFromFile(ManifestPath, ManifestText) then begin
    Log('Local STT installed manifest could not be read: ' + ManifestPath);
    exit;
  end;

  Result :=
    (ManifestText = ExpectedLocalSttInstalledManifest('huggingface')) or
    (ManifestText = ExpectedLocalSttInstalledManifest('modelscope'));
  if not Result then begin
    Log('Local STT installed manifest content is invalid: ' + ManifestPath);
  end;
end;

function ValidateLocalSttInstall(BaseDir: String): Boolean;
begin
  Result :=
    ValidateLocalSttAsset(BaseDir, 'conv_frontend.onnx', 'd22dc4423e0940e49884e903d2ea2f7e5567c14fc1aed97e4e26d6b8f208ef9e', 44148281) and
    ValidateLocalSttAsset(BaseDir, 'decoder.int8.onnx', '61e5f8249f9e7c82d5e01e1938c79fb3f5b3135f91664928033029e42451bd18', 756563239) and
    ValidateLocalSttAsset(BaseDir, 'encoder.int8.onnx', '60748d3e6744a57c9c91e1b17424a6c2990567e8adceb0783940c03ed98fa9d9', 182491662) and
    ValidateLocalSttAsset(BaseDir, 'tokenizer\merges.txt', '8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5', 1671853) and
    ValidateLocalSttAsset(BaseDir, 'tokenizer\tokenizer_config.json', '4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c', 12487) and
    ValidateLocalSttAsset(BaseDir, 'tokenizer\vocab.json', 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910', 2776833) and
    ValidateLocalSttInstalledManifest(BaseDir);
end;

procedure AddLocalSttDownloads(SourceName: String);
begin
  DownloadPage.Add(LocalSttDownloadUrl(SourceName, 'conv_frontend.onnx'), 'conv_frontend.onnx', 'd22dc4423e0940e49884e903d2ea2f7e5567c14fc1aed97e4e26d6b8f208ef9e');
  DownloadPage.Add(LocalSttDownloadUrl(SourceName, 'decoder.int8.onnx'), 'decoder.int8.onnx', '61e5f8249f9e7c82d5e01e1938c79fb3f5b3135f91664928033029e42451bd18');
  DownloadPage.Add(LocalSttDownloadUrl(SourceName, 'encoder.int8.onnx'), 'encoder.int8.onnx', '60748d3e6744a57c9c91e1b17424a6c2990567e8adceb0783940c03ed98fa9d9');
  DownloadPage.Add(LocalSttDownloadUrl(SourceName, 'tokenizer/merges.txt'), 'merges.txt', '8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5');
  DownloadPage.Add(LocalSttDownloadUrl(SourceName, 'tokenizer/tokenizer_config.json'), 'tokenizer_config.json', '4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c');
  DownloadPage.Add(LocalSttDownloadUrl(SourceName, 'tokenizer/vocab.json'), 'vocab.json', 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910');
end;

function CopyLocalSttAsset(StagingDir: String; BaseName: String; RelativePath: String): Boolean;
var
  DestinationPath: String;
begin
  DestinationPath := AddBackslash(StagingDir) + RelativePath;
  ForceDirectories(ExtractFileDir(DestinationPath));
  Result := CopyFile(ExpandConstant('{tmp}\') + BaseName, DestinationPath, False);
  if not Result then begin
    Log('Failed to stage local STT asset: ' + DestinationPath);
  end;
end;

function StageLocalSttDownloads(StagingDir: String): Boolean;
begin
  Result :=
    CopyLocalSttAsset(StagingDir, 'conv_frontend.onnx', 'conv_frontend.onnx') and
    CopyLocalSttAsset(StagingDir, 'decoder.int8.onnx', 'decoder.int8.onnx') and
    CopyLocalSttAsset(StagingDir, 'encoder.int8.onnx', 'encoder.int8.onnx') and
    CopyLocalSttAsset(StagingDir, 'merges.txt', 'tokenizer\merges.txt') and
    CopyLocalSttAsset(StagingDir, 'tokenizer_config.json', 'tokenizer\tokenizer_config.json') and
    CopyLocalSttAsset(StagingDir, 'vocab.json', 'tokenizer\vocab.json');
end;

function WriteLocalSttInstalledManifest(StagingDir: String; SourceName: String): Boolean;
var
  ManifestJson: String;
begin
  ManifestJson := ExpectedLocalSttInstalledManifest(SourceName);
  Result := SaveStringToFile(AddBackslash(StagingDir) + 'installed-manifest.json', ManifestJson, False);
end;

function PromoteLocalSttInstall(StagingDir: String): Boolean;
var
  InstallDir: String;
begin
  InstallDir := GetLocalSttInstallDir();
  DelTree(InstallDir + '.backup', True, True, True);
  if DirExists(InstallDir) then begin
    if not RenameFile(InstallDir, InstallDir + '.backup') then begin
      Log('Failed to back up existing local STT install: ' + InstallDir);
      Result := False;
      exit;
    end;
  end;
  Result := RenameFile(StagingDir, InstallDir);
  if Result then begin
    DelTree(InstallDir + '.backup', True, True, True);
  end else begin
    Log('Failed to promote local STT staging directory: ' + StagingDir);
    if DirExists(InstallDir + '.backup') then begin
      RenameFile(InstallDir + '.backup', InstallDir);
    end;
  end;
end;

function DownloadLocalSttProgress(const Url, FileName: String; const Progress, ProgressMax: Int64): Boolean;
begin
  Result := True;
end;

function DownloadLocalSttSource(SourceName: String): Boolean;
var
  StagingDir: String;
begin
  Result := False;
  StagingDir := GetLocalSttInstallDir() + '.staging-' + SourceName;
  DelTree(StagingDir, True, True, True);
  ForceDirectories(StagingDir);
  try
    DownloadPage.Clear;
    AddLocalSttDownloads(SourceName);
    DownloadPage.Show;
    try
      DownloadPage.Download;
    finally
      DownloadPage.Hide;
    end;
    if StageLocalSttDownloads(StagingDir) and
       WriteLocalSttInstalledManifest(StagingDir, SourceName) and
       ValidateLocalSttInstall(StagingDir) then begin
      Result := PromoteLocalSttInstall(StagingDir);
    end else if not Result then begin
      Log('Failed to stage or validate local STT install for source: ' + SourceName);
    end;
  except
    Log('Local STT download failed for ' + SourceName + ': ' + GetExceptionMessage);
  end;
  if not Result then begin
    DelTree(StagingDir, True, True, True);
  end;
end;

function RunLocalSttModelInstall(): Boolean;
begin
  Result := False;
  if (not GetLocalSttReinstallEnabled()) and ValidateLocalSttInstall(GetLocalSttInstallDir()) then begin
    Log('Local STT model is already installed and valid.');
    Result := True;
    exit;
  end;

  if DownloadLocalSttSource('huggingface') then begin
    Log('Local STT provisioning completed successfully from Hugging Face.');
    Result := True;
    exit;
  end;

  Log('Hugging Face local STT provisioning failed; trying ModelScope.');
  if DownloadLocalSttSource('modelscope') then begin
    Log('Local STT provisioning completed successfully from ModelScope.');
    Result := True;
    exit;
  end;
end;

procedure InitializeWizard();
begin
  ResetSuspiciousInstallDir();
  InitializeLocalSttWizardPage();
  DownloadPage := CreateDownloadPage(
    ExpandConstant('{cm:LocalSttDownloadTitle}'),
    ExpandConstant('{cm:LocalSttDownloadDescription}'),
    @DownloadLocalSttProgress
  );
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  ResetSuspiciousInstallDir();
  Result := '';
  if not RunLocalSttModelInstall() then begin
    Log('Local STT provisioning did not complete; continuing app install without bundled ASR model. The app can retry the local STT model download at runtime.');
  end;
end;
