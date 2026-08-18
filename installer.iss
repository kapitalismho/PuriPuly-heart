; Inno Setup Script for PuriPuly <3
; Compile with: ISCC installer.iss

#define MyAppName "PuriPuly <3"
#define MyAppDirName "PuriPulyHeart"
#define MyAppGroupName "PuriPulyHeart"
#define MyAppVersion "2.4.0"
#define MyAppPublisher "salee"
#define MyAppURL "https://github.com/kapitalismho/PuriPuly-heart"
#ifndef MyAppExeName
  #define MyAppExeName "PuriPulyHeart.exe"
#endif
#define MyOverlayExeName "PuriPulyHeartOverlay.exe"
#define MyGpuWorkerExeName "PuriPulyHeartGpuWorker.exe"
#ifndef MyPackagedAppDir
  #define MyPackagedAppDir "dist\PuriPulyHeart"
#endif
#define MyStagedOverlayDir "build\overlay"
#define NotoCjkFontRelativePath "puripuly_heart\data\fonts\NotoSansCJK-Medium.ttc"
#define LocalSttManifestRelativePath "puripuly_heart\data\models\qwen3-asr-0.6b-int8-sherpa.manifest.json"
#define ParakeetV3ManifestRelativePath "puripuly_heart\data\models\parakeet-tdt-0.6b-v3-int8-sherpa.manifest.json"
#define ParakeetJapaneseManifestRelativePath "puripuly_heart\data\models\parakeet-tdt-ctc-0.6b-ja-int8-sherpa.manifest.json"

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
english.LocalSttDownloadFailed=ASR model download failed from both Hugging Face and ModelScope. Installation cannot complete.
korean.LocalSttDownloadFailed=Hugging Face와 ModelScope 모두에서 ASR 모델 다운로드에 실패했습니다. 설치를 완료할 수 없습니다.
japanese.LocalSttDownloadFailed=Hugging Face と ModelScope の両方でASRモデルのダウンロードに失敗しました。インストールを完了できません。
chinesesimplified.LocalSttDownloadFailed=从 Hugging Face 和 ModelScope 下载 ASR 模型均失败。无法完成安装。
chinesetraditional.LocalSttDownloadFailed=從 Hugging Face 和 ModelScope 下載 ASR 模型均失敗。無法完成安裝。
english.AsrModelsGroup=ASR Models
english.RedownloadAsrTask=Re-download all ASR models
english.LocalSttRedownloadSize=Re-downloading ASR models.%nInstallation requires %1 of disk space.
korean.AsrModelsGroup=ASR 모델
korean.RedownloadAsrTask=ASR 모델 전체 재다운로드
korean.LocalSttRedownloadSize=ASR 모델을 재다운로드합니다.%n설치에 필요한 용량은 %1입니다.
japanese.AsrModelsGroup=ASRモデル
japanese.RedownloadAsrTask=ASRモデルをすべて再ダウンロード
japanese.LocalSttRedownloadSize=ASRモデルを再ダウンロードします。%nインストールには%1の空き容量が必要です。
chinesesimplified.AsrModelsGroup=ASR 模型
chinesesimplified.RedownloadAsrTask=重新下载所有 ASR 模型
chinesesimplified.LocalSttRedownloadSize=重新下载 ASR 模型。%n安装需要 %1 的空间。
chinesetraditional.AsrModelsGroup=ASR 模型
chinesetraditional.RedownloadAsrTask=重新下載所有 ASR 模型
chinesetraditional.LocalSttRedownloadSize=重新下載 ASR 模型。%n安裝需要 %1 的空間。

[Files]
Source: "{#MyPackagedAppDir}\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#MyStagedOverlayDir}\{#MyOverlayExeName}"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#MyPackagedAppDir}\{#MyGpuWorkerExeName}"; DestDir: "{app}"; Flags: ignoreversion
; Vendored OpenVR runtime DLL comes from dist\PuriPulyHeart\openvr_api.dll in the packaged tree built by build.spec.
; Installer build/install never resolves SteamVR paths for openvr_api.dll.
; Bundled CJK font is staged at {#MyPackagedAppDir}\{#NotoCjkFontRelativePath}; the recursive packaged-tree copy installs it to {app}\{#NotoCjkFontRelativePath}.
Source: "{#MyPackagedAppDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: "{#MyAppExeName},{#MyOverlayExeName},{#MyGpuWorkerExeName}"
#ifdef ProcessCaptureSmokeArtifactRoot
Source: "{#ProcessCaptureSmokeArtifactRoot}\*"; DestDir: "{app}\process-capture-smoke"; Flags: ignoreversion recursesubdirs createallsubdirs
#endif
; NOTE: Don't use "Flags: ignoreversion" on any shared system files

[Icons]
Name: "{group}\{#MyAppGroupName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\{cm:UninstallProgram,{#MyAppGroupName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppGroupName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{userappdata}\Microsoft\Internet Explorer\Quick Launch\{#MyAppGroupName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: quicklaunchicon

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
  DownloadPage: TDownloadWizardPage;
  LocalSttPlanPrepared: Boolean;
  QwenNeedsDownload: Boolean;
  ParakeetV3NeedsDownload: Boolean;
  ParakeetJapaneseNeedsDownload: Boolean;
  LocalSttRequiredDownloadBytes: Int64;
  LocalSttCompletedDownloadBytes: Int64;
  LocalSttDisplayedDownloadBytes: Int64;
  LocalSttCurrentFileExpectedBytes: Int64;
  LocalSttCurrentFileLabel: String;

const
  QwenDownloadSize = 987664355;
  ParakeetV3DownloadSize = 670478772;
  ParakeetJapaneseDownloadSize = 655571161;
  LocalSttDiskSpaceMargin = 67108864;

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

function ValidateLocalSttAssets(BaseDir: String): Boolean;
begin
  Result :=
    ValidateLocalSttAsset(BaseDir, 'conv_frontend.onnx', 'd22dc4423e0940e49884e903d2ea2f7e5567c14fc1aed97e4e26d6b8f208ef9e', 44148281) and
    ValidateLocalSttAsset(BaseDir, 'decoder.int8.onnx', '61e5f8249f9e7c82d5e01e1938c79fb3f5b3135f91664928033029e42451bd18', 756563239) and
    ValidateLocalSttAsset(BaseDir, 'encoder.int8.onnx', '60748d3e6744a57c9c91e1b17424a6c2990567e8adceb0783940c03ed98fa9d9', 182491662) and
    ValidateLocalSttAsset(BaseDir, 'tokenizer\merges.txt', '8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5', 1671853) and
    ValidateLocalSttAsset(BaseDir, 'tokenizer\tokenizer_config.json', '4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c', 12487) and
    ValidateLocalSttAsset(BaseDir, 'tokenizer\vocab.json', 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910', 2776833);
end;

function ValidateLocalSttInstall(BaseDir: String): Boolean;
begin
  Result := ValidateLocalSttAssets(BaseDir) and ValidateLocalSttInstalledManifest(BaseDir);
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
    CopyLocalSttAsset(StagingDir, 'qwen-conv_frontend.onnx', 'conv_frontend.onnx') and
    CopyLocalSttAsset(StagingDir, 'qwen-decoder.int8.onnx', 'decoder.int8.onnx') and
    CopyLocalSttAsset(StagingDir, 'qwen-encoder.int8.onnx', 'encoder.int8.onnx') and
    CopyLocalSttAsset(StagingDir, 'qwen-merges.txt', 'tokenizer\merges.txt') and
    CopyLocalSttAsset(StagingDir, 'qwen-tokenizer_config.json', 'tokenizer\tokenizer_config.json') and
    CopyLocalSttAsset(StagingDir, 'qwen-vocab.json', 'tokenizer\vocab.json');
end;

function WriteLocalSttInstalledManifest(StagingDir: String; SourceName: String): Boolean;
var
  ManifestJson: String;
begin
  ManifestJson := ExpectedLocalSttInstalledManifest(SourceName);
  Result := SaveStringToFile(AddBackslash(StagingDir) + 'installed-manifest.json', ManifestJson, False);
end;

function PromoteLocalSttInstallTo(StagingDir: String; InstallDir: String): Boolean;
begin
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

function PromoteLocalSttInstall(StagingDir: String): Boolean;
begin
  Result := PromoteLocalSttInstallTo(StagingDir, GetLocalSttInstallDir());
end;

function FormatByteSize(Bytes: Int64): String;
var
  WholePart: Int64;
  FractionPart: Int64;
  FractionText: String;
begin
  if Bytes >= 1000000000 then begin
    WholePart := Bytes div 1000000000;
    FractionPart := ((Bytes mod 1000000000) * 100) div 1000000000;
    FractionText := IntToStr(FractionPart);
    if FractionPart < 10 then begin
      FractionText := '0' + FractionText;
    end;
    Result := IntToStr(WholePart) + '.' + FractionText + ' GB';
  end else begin
    WholePart := Bytes div 1000000;
    FractionPart := ((Bytes mod 1000000) * 100) div 1000000;
    FractionText := IntToStr(FractionPart);
    if FractionPart < 10 then begin
      FractionText := '0' + FractionText;
    end;
    Result := IntToStr(WholePart) + '.' + FractionText + ' MB';
  end;
end;

function DownloadLocalSttProgress(const Url, FileName: String; const Progress, ProgressMax: Int64): Boolean;
var
  CurrentBytes: Int64;
  OverallBytes: Int64;
  ProgressPosition: Integer;
begin
  CurrentBytes := Progress;
  if CurrentBytes > LocalSttCurrentFileExpectedBytes then begin
    CurrentBytes := LocalSttCurrentFileExpectedBytes;
  end;
  OverallBytes := LocalSttCompletedDownloadBytes + CurrentBytes;
  if OverallBytes > LocalSttDisplayedDownloadBytes then begin
    LocalSttDisplayedDownloadBytes := OverallBytes;
  end;
  if LocalSttRequiredDownloadBytes > 0 then begin
    ProgressPosition := (LocalSttDisplayedDownloadBytes * 10000) div LocalSttRequiredDownloadBytes;
    if ProgressPosition > 10000 then begin
      ProgressPosition := 10000;
    end;
    DownloadPage.SetProgress(ProgressPosition, 10000);
  end;
  DownloadPage.SetText(
    LocalSttCurrentFileLabel,
    FormatByteSize(LocalSttDisplayedDownloadBytes) + ' / ' + FormatByteSize(LocalSttRequiredDownloadBytes)
  );
  Result := not DownloadPage.AbortedByUser;
end;

function DownloadLocalSttFile(Url: String; BaseName: String; Sha256: String; ExpectedSize: Int64; DisplayName: String): Boolean;
begin
  Result := False;
  LocalSttCurrentFileExpectedBytes := ExpectedSize;
  LocalSttCurrentFileLabel := DisplayName;
  try
    DownloadTemporaryFile(Url, BaseName, Sha256, @DownloadLocalSttProgress);
    LocalSttCompletedDownloadBytes := LocalSttCompletedDownloadBytes + ExpectedSize;
    if LocalSttCompletedDownloadBytes > LocalSttDisplayedDownloadBytes then begin
      LocalSttDisplayedDownloadBytes := LocalSttCompletedDownloadBytes;
    end;
    Result := True;
  except
    Log('Local STT asset download failed: ' + DisplayName + ': ' + GetExceptionMessage);
  end;
end;

function DownloadLocalSttSourceFiles(SourceName: String): Boolean;
begin
  Result :=
    DownloadLocalSttFile(LocalSttDownloadUrl(SourceName, 'conv_frontend.onnx'), 'qwen-conv_frontend.onnx', 'd22dc4423e0940e49884e903d2ea2f7e5567c14fc1aed97e4e26d6b8f208ef9e', 44148281, 'Qwen3 ASR - conv_frontend.onnx') and
    DownloadLocalSttFile(LocalSttDownloadUrl(SourceName, 'decoder.int8.onnx'), 'qwen-decoder.int8.onnx', '61e5f8249f9e7c82d5e01e1938c79fb3f5b3135f91664928033029e42451bd18', 756563239, 'Qwen3 ASR - decoder.int8.onnx') and
    DownloadLocalSttFile(LocalSttDownloadUrl(SourceName, 'encoder.int8.onnx'), 'qwen-encoder.int8.onnx', '60748d3e6744a57c9c91e1b17424a6c2990567e8adceb0783940c03ed98fa9d9', 182491662, 'Qwen3 ASR - encoder.int8.onnx') and
    DownloadLocalSttFile(LocalSttDownloadUrl(SourceName, 'tokenizer/merges.txt'), 'qwen-merges.txt', '8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5', 1671853, 'Qwen3 ASR - merges.txt') and
    DownloadLocalSttFile(LocalSttDownloadUrl(SourceName, 'tokenizer/tokenizer_config.json'), 'qwen-tokenizer_config.json', '4942d005604266809309cabc9f4e9cb89ce855d59b14681fdc0e1cc62ea26c4c', 12487, 'Qwen3 ASR - tokenizer_config.json') and
    DownloadLocalSttFile(LocalSttDownloadUrl(SourceName, 'tokenizer/vocab.json'), 'qwen-vocab.json', 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910', 2776833, 'Qwen3 ASR - vocab.json');
end;

function GetParakeetV3InstallDir(): String;
begin
  Result := AddBackslash(ResolveLocalSttAppDataRoot()) + 'models\parakeet-tdt-0.6b-v3-int8-sherpa';
end;

function ParakeetV3DownloadUrl(RelativePath: String): String;
begin
  Result := 'https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8/resolve/2bda32ec70b097a55adaa07d9a7173915b43cc78/' + RelativePath;
end;

function ExpectedParakeetV3InstalledManifest(): String;
begin
  Result := '{' + #13#10 +
    '  "manifest_version": 1,' + #13#10 +
    '  "model_id": "parakeet-tdt-0.6b-v3-int8-sherpa",' + #13#10 +
    '  "engine": "sherpa-onnx",' + #13#10 +
    '  "install_dirname": "parakeet-tdt-0.6b-v3-int8-sherpa",' + #13#10 +
    '  "selected_source": "huggingface",' + #13#10 +
    '  "selected_revision": "2bda32ec70b097a55adaa07d9a7173915b43cc78"' + #13#10 +
    '}';
end;

function ValidateParakeetV3InstalledManifest(BaseDir: String): Boolean;
var
  ManifestText: AnsiString;
begin
  Result := LoadStringFromFile(AddBackslash(BaseDir) + 'installed-manifest.json', ManifestText) and
    (ManifestText = ExpectedParakeetV3InstalledManifest());
end;

function ValidateParakeetV3Assets(BaseDir: String): Boolean;
begin
  Result :=
    ValidateLocalSttAsset(BaseDir, 'decoder.int8.onnx', '179e50c43d1a9de79c8a24149a2f9bac6eb5981823f2a2ed88d655b24248db4e', 11845275) and
    ValidateLocalSttAsset(BaseDir, 'encoder.int8.onnx', 'acfc2b4456377e15d04f0243af540b7fe7c992f8d898d751cf134c3a55fd2247', 652184281) and
    ValidateLocalSttAsset(BaseDir, 'joiner.int8.onnx', '3164c13fc2821009440d20fcb5fdc78bff28b4db2f8d0f0b329101719c0948b3', 6355277) and
    ValidateLocalSttAsset(BaseDir, 'tokens.txt', 'd58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d', 93939);
end;

function ValidateParakeetV3Install(BaseDir: String): Boolean;
begin
  Result := ValidateParakeetV3Assets(BaseDir) and ValidateParakeetV3InstalledManifest(BaseDir);
end;

function DownloadParakeetV3(): Boolean;
var
  StagingDir: String;
begin
  Result := False;
  StagingDir := GetParakeetV3InstallDir() + '.staging-huggingface';
  DelTree(StagingDir, True, True, True);
  ForceDirectories(StagingDir);
  try
    if DownloadLocalSttFile(ParakeetV3DownloadUrl('decoder.int8.onnx'), 'parakeet-v3-decoder.int8.onnx', '179e50c43d1a9de79c8a24149a2f9bac6eb5981823f2a2ed88d655b24248db4e', 11845275, 'Parakeet v3 - decoder.int8.onnx') and
       DownloadLocalSttFile(ParakeetV3DownloadUrl('encoder.int8.onnx'), 'parakeet-v3-encoder.int8.onnx', 'acfc2b4456377e15d04f0243af540b7fe7c992f8d898d751cf134c3a55fd2247', 652184281, 'Parakeet v3 - encoder.int8.onnx') and
       DownloadLocalSttFile(ParakeetV3DownloadUrl('joiner.int8.onnx'), 'parakeet-v3-joiner.int8.onnx', '3164c13fc2821009440d20fcb5fdc78bff28b4db2f8d0f0b329101719c0948b3', 6355277, 'Parakeet v3 - joiner.int8.onnx') and
       DownloadLocalSttFile(ParakeetV3DownloadUrl('tokens.txt'), 'parakeet-v3-tokens.txt', 'd58544679ea4bc6ac563d1f545eb7d474bd6cfa467f0a6e2c1dc1c7d37e3c35d', 93939, 'Parakeet v3 - tokens.txt') and
       CopyLocalSttAsset(StagingDir, 'parakeet-v3-decoder.int8.onnx', 'decoder.int8.onnx') and
       CopyLocalSttAsset(StagingDir, 'parakeet-v3-encoder.int8.onnx', 'encoder.int8.onnx') and
       CopyLocalSttAsset(StagingDir, 'parakeet-v3-joiner.int8.onnx', 'joiner.int8.onnx') and
       CopyLocalSttAsset(StagingDir, 'parakeet-v3-tokens.txt', 'tokens.txt') and
       SaveStringToFile(AddBackslash(StagingDir) + 'installed-manifest.json', ExpectedParakeetV3InstalledManifest(), False) and
       ValidateParakeetV3Install(StagingDir) then begin
      Result := PromoteLocalSttInstallTo(StagingDir, GetParakeetV3InstallDir());
    end;
  except
    Log('Parakeet v3 download failed: ' + GetExceptionMessage);
  end;
  if not Result then begin
    DelTree(StagingDir, True, True, True);
  end;
end;

function RunParakeetV3LocalSttModelInstall(): Boolean;
begin
#ifdef SkipLocalSttProvisioning
  Result := True;
  exit;
#endif
  if not ParakeetV3NeedsDownload then begin
    Result := True;
    exit;
  end;
  Result := DownloadParakeetV3();
end;

function GetParakeetJapaneseInstallDir(): String;
begin
  Result := AddBackslash(ResolveLocalSttAppDataRoot()) + 'models\parakeet-tdt-ctc-0.6b-ja-int8-sherpa';
end;

function ParakeetJapaneseDownloadUrl(RelativePath: String): String;
begin
  Result := 'https://huggingface.co/csukuangfj/sherpa-onnx-nemo-parakeet-tdt_ctc-0.6b-ja-35000-int8/resolve/bef18eb066808c90bd0f5df5be685767b0732de8/' + RelativePath;
end;

function ExpectedParakeetJapaneseInstalledManifest(): String;
begin
  Result := '{' + #13#10 +
    '  "manifest_version": 1,' + #13#10 +
    '  "model_id": "parakeet-tdt-ctc-0.6b-ja-int8-sherpa",' + #13#10 +
    '  "engine": "sherpa-onnx",' + #13#10 +
    '  "install_dirname": "parakeet-tdt-ctc-0.6b-ja-int8-sherpa",' + #13#10 +
    '  "selected_source": "huggingface",' + #13#10 +
    '  "selected_revision": "bef18eb066808c90bd0f5df5be685767b0732de8"' + #13#10 +
    '}';
end;

function ValidateParakeetJapaneseInstalledManifest(BaseDir: String): Boolean;
var
  ManifestText: AnsiString;
begin
  Result := LoadStringFromFile(AddBackslash(BaseDir) + 'installed-manifest.json', ManifestText) and
    (ManifestText = ExpectedParakeetJapaneseInstalledManifest());
end;

function ValidateParakeetJapaneseAssets(BaseDir: String): Boolean;
begin
  Result :=
    ValidateLocalSttAsset(BaseDir, 'model.int8.onnx', '3addd00ef5bd1742078389e540b77394e4a508bdf2f4c9ad1b4a76d93e76598e', 655542604) and
    ValidateLocalSttAsset(BaseDir, 'tokens.txt', '732f64c53909f2620c713f4106b487d92e6f54a6915b3cd3d1dbd32f9f4f392a', 28557);
end;

function ValidateParakeetJapaneseInstall(BaseDir: String): Boolean;
begin
  Result := ValidateParakeetJapaneseAssets(BaseDir) and ValidateParakeetJapaneseInstalledManifest(BaseDir);
end;

function DownloadParakeetJapanese(): Boolean;
var
  StagingDir: String;
begin
  Result := False;
  StagingDir := GetParakeetJapaneseInstallDir() + '.staging-huggingface';
  DelTree(StagingDir, True, True, True);
  ForceDirectories(StagingDir);
  try
    if DownloadLocalSttFile(ParakeetJapaneseDownloadUrl('model.int8.onnx'), 'parakeet-ja-model.int8.onnx', '3addd00ef5bd1742078389e540b77394e4a508bdf2f4c9ad1b4a76d93e76598e', 655542604, 'Parakeet Japanese - model.int8.onnx') and
       DownloadLocalSttFile(ParakeetJapaneseDownloadUrl('tokens.txt'), 'parakeet-ja-tokens.txt', '732f64c53909f2620c713f4106b487d92e6f54a6915b3cd3d1dbd32f9f4f392a', 28557, 'Parakeet Japanese - tokens.txt') and
       CopyLocalSttAsset(StagingDir, 'parakeet-ja-model.int8.onnx', 'model.int8.onnx') and
       CopyLocalSttAsset(StagingDir, 'parakeet-ja-tokens.txt', 'tokens.txt') and
       SaveStringToFile(AddBackslash(StagingDir) + 'installed-manifest.json', ExpectedParakeetJapaneseInstalledManifest(), False) and
       ValidateParakeetJapaneseInstall(StagingDir) then begin
      Result := PromoteLocalSttInstallTo(StagingDir, GetParakeetJapaneseInstallDir());
    end;
  except
    Log('Parakeet Japanese download failed: ' + GetExceptionMessage);
  end;
  if not Result then begin
    DelTree(StagingDir, True, True, True);
  end;
end;

function RunParakeetJapaneseLocalSttModelInstall(): Boolean;
begin
#ifdef SkipLocalSttProvisioning
  Result := True;
  exit;
#endif
  if not ParakeetJapaneseNeedsDownload then begin
    Result := True;
    exit;
  end;
  Result := DownloadParakeetJapanese();
end;

function DownloadAndInstallLocalSttSource(SourceName: String): Boolean;
var
  StagingDir: String;
begin
  Result := False;
  StagingDir := GetLocalSttInstallDir() + '.staging-' + SourceName;
  DelTree(StagingDir, True, True, True);
  ForceDirectories(StagingDir);
  try
    if DownloadLocalSttSourceFiles(SourceName) and
       StageLocalSttDownloads(StagingDir) and
       WriteLocalSttInstalledManifest(StagingDir, SourceName) and
       ValidateLocalSttInstall(StagingDir) then begin
      Result := PromoteLocalSttInstall(StagingDir);
    end;
  except
    Log('Local STT download failed for ' + SourceName + ': ' + GetExceptionMessage);
  end;
  if not Result then begin
    DelTree(StagingDir, True, True, True);
  end;
end;

function RunLocalSttModelInstall(): Boolean;
var
  DownloadStartBytes: Int64;
begin
#ifdef SkipLocalSttProvisioning
  Log('Local STT provisioning skipped for isolated installer smoke.');
  Result := True;
  exit;
#endif
  Result := False;
  if not QwenNeedsDownload then begin
    Log('Local STT model is already installed and valid.');
    Result := True;
    exit;
  end;

  DownloadStartBytes := LocalSttCompletedDownloadBytes;
  if DownloadAndInstallLocalSttSource('huggingface') then begin
    Log('Local STT provisioning completed successfully from Hugging Face.');
    Result := True;
    exit;
  end;

  Log('Hugging Face local STT provisioning failed; trying ModelScope.');
  LocalSttCompletedDownloadBytes := DownloadStartBytes;
  if DownloadAndInstallLocalSttSource('modelscope') then begin
    Log('Local STT provisioning completed successfully from ModelScope.');
    Result := True;
    exit;
  end;
end;

procedure PrepareLocalSttDownloadPlan();
var
  ParakeetV3Dir: String;
  ParakeetJapaneseDir: String;
  QwenDir: String;
  ReDownloadSelected: Boolean;
begin
  if LocalSttPlanPrepared then begin
    exit;
  end;
  ReDownloadSelected := WizardIsTaskSelected('redownloadasr');
#ifdef SkipLocalSttProvisioning
  QwenNeedsDownload := False;
  ParakeetV3NeedsDownload := False;
  ParakeetJapaneseNeedsDownload := False;
#else
  ParakeetV3Dir := GetParakeetV3InstallDir();
  ParakeetV3NeedsDownload := ReDownloadSelected or not (ValidateParakeetV3InstalledManifest(ParakeetV3Dir) and
    FileExists(AddBackslash(ParakeetV3Dir) + 'encoder.int8.onnx'));

  ParakeetJapaneseDir := GetParakeetJapaneseInstallDir();
  ParakeetJapaneseNeedsDownload := ReDownloadSelected or not (ValidateParakeetJapaneseInstalledManifest(ParakeetJapaneseDir) and
    FileExists(AddBackslash(ParakeetJapaneseDir) + 'model.int8.onnx'));

  QwenDir := GetLocalSttInstallDir();
  QwenNeedsDownload := ReDownloadSelected or not (ValidateLocalSttInstalledManifest(QwenDir) and
    FileExists(AddBackslash(QwenDir) + 'decoder.int8.onnx'));
#endif
  LocalSttRequiredDownloadBytes := 0;
  if ParakeetV3NeedsDownload then begin
    LocalSttRequiredDownloadBytes := LocalSttRequiredDownloadBytes + ParakeetV3DownloadSize;
  end;
  if ParakeetJapaneseNeedsDownload then begin
    LocalSttRequiredDownloadBytes := LocalSttRequiredDownloadBytes + ParakeetJapaneseDownloadSize;
  end;
  if QwenNeedsDownload then begin
    LocalSttRequiredDownloadBytes := LocalSttRequiredDownloadBytes + QwenDownloadSize;
  end;
  LocalSttPlanPrepared := True;
end;

function LocalSttStorageVolume(Path: String): String;
begin
  Result := ExtractFileDrive(RemoveBackslashUnlessRoot(Path));
  if Result = '' then begin
    Result := RemoveBackslashUnlessRoot(Path);
  end;
end;

function CheckLocalSttDiskSpace(var ErrorMessage: String): Boolean;
var
  TempPath: String;
  ModelPath: String;
  TempFreeBytes: Int64;
  TempTotalBytes: Int64;
  ModelFreeBytes: Int64;
  ModelTotalBytes: Int64;
  RequiredBytes: Int64;
  RequiredText: String;
  AvailableText: String;
  VolumeText: String;
begin
  Result := False;
  ErrorMessage := '';
  PrepareLocalSttDownloadPlan();
  if LocalSttRequiredDownloadBytes = 0 then begin
    Result := True;
    exit;
  end;
  TempPath := ExpandConstant('{tmp}');
  ModelPath := ResolveLocalSttAppDataRoot();
  ForceDirectories(ModelPath);
  if not GetSpaceOnDisk64(TempPath, TempFreeBytes, TempTotalBytes) then begin
    ErrorMessage := FmtMessage(ExpandConstant('{cm:LocalSttDiskSpaceCheckFailed}'), [TempPath]);
    exit;
  end;
  if not GetSpaceOnDisk64(ModelPath, ModelFreeBytes, ModelTotalBytes) then begin
    ErrorMessage := FmtMessage(ExpandConstant('{cm:LocalSttDiskSpaceCheckFailed}'), [ModelPath]);
    exit;
  end;
  if CompareText(LocalSttStorageVolume(TempPath), LocalSttStorageVolume(ModelPath)) = 0 then begin
    RequiredBytes := (LocalSttRequiredDownloadBytes * 2) + LocalSttDiskSpaceMargin;
    if TempFreeBytes < RequiredBytes then begin
      RequiredText := FormatByteSize(RequiredBytes);
      AvailableText := FormatByteSize(TempFreeBytes);
      VolumeText := LocalSttStorageVolume(TempPath);
      ErrorMessage := FmtMessage(CustomMessage('LocalSttDiskSpaceFailed'), [RequiredText, AvailableText, VolumeText]);
      exit;
    end;
  end else begin
    RequiredBytes := LocalSttRequiredDownloadBytes + LocalSttDiskSpaceMargin;
    if TempFreeBytes < RequiredBytes then begin
      RequiredText := FormatByteSize(RequiredBytes);
      AvailableText := FormatByteSize(TempFreeBytes);
      VolumeText := LocalSttStorageVolume(TempPath);
      ErrorMessage := FmtMessage(CustomMessage('LocalSttDiskSpaceFailed'), [RequiredText, AvailableText, VolumeText]);
      exit;
    end;
    if ModelFreeBytes < RequiredBytes then begin
      RequiredText := FormatByteSize(RequiredBytes);
      AvailableText := FormatByteSize(ModelFreeBytes);
      VolumeText := LocalSttStorageVolume(ModelPath);
      ErrorMessage := FmtMessage(CustomMessage('LocalSttDiskSpaceFailed'), [RequiredText, AvailableText, VolumeText]);
      exit;
    end;
  end;
  Result := True;
end;

procedure CompleteLocalSttDownloadSegment(SegmentStart: Int64; SegmentSize: Int64);
var
  ProgressPosition: Integer;
begin
  LocalSttCompletedDownloadBytes := SegmentStart + SegmentSize;
  if LocalSttCompletedDownloadBytes > LocalSttDisplayedDownloadBytes then begin
    LocalSttDisplayedDownloadBytes := LocalSttCompletedDownloadBytes;
  end;
  if LocalSttRequiredDownloadBytes > 0 then begin
    ProgressPosition := (LocalSttDisplayedDownloadBytes * 10000) div LocalSttRequiredDownloadBytes;
    if ProgressPosition > 10000 then begin
      ProgressPosition := 10000;
    end;
    DownloadPage.SetProgress(ProgressPosition, 10000);
  end;
end;

function RunRequiredCpuLocalSttModelInstalls(): Boolean;
var
  ParakeetV3Installed: Boolean;
  ParakeetJapaneseInstalled: Boolean;
  QwenInstalled: Boolean;
  SegmentStart: Int64;
begin
  PrepareLocalSttDownloadPlan();
  if LocalSttRequiredDownloadBytes = 0 then begin
#ifdef SkipLocalSttProvisioning
    Log('Local STT provisioning skipped for isolated installer smoke.');
#endif
    Result := True;
    exit;
  end;
  LocalSttCompletedDownloadBytes := 0;
  LocalSttDisplayedDownloadBytes := 0;
  DownloadPage.SetProgress(0, 10000);
  DownloadPage.Show;
  try
    SegmentStart := LocalSttCompletedDownloadBytes;
  ParakeetV3Installed := RunParakeetV3LocalSttModelInstall();
    if ParakeetV3NeedsDownload then begin
      CompleteLocalSttDownloadSegment(SegmentStart, ParakeetV3DownloadSize);
    end;
    SegmentStart := LocalSttCompletedDownloadBytes;
  ParakeetJapaneseInstalled := RunParakeetJapaneseLocalSttModelInstall();
    if ParakeetJapaneseNeedsDownload then begin
      CompleteLocalSttDownloadSegment(SegmentStart, ParakeetJapaneseDownloadSize);
    end;
    SegmentStart := LocalSttCompletedDownloadBytes;
  if not RunLocalSttModelInstall() then begin
    QwenInstalled := False;
  end else begin
    QwenInstalled := True;
  end;
    if QwenNeedsDownload then begin
      CompleteLocalSttDownloadSegment(SegmentStart, QwenDownloadSize);
    end;
  finally
    DownloadPage.Hide;
  end;
  Result := ParakeetV3Installed and ParakeetJapaneseInstalled and QwenInstalled;
end;

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

procedure InitializeWizard();
begin
  ResetSuspiciousInstallDir();
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
  if not CheckLocalSttDiskSpace(Result) then begin
    exit;
  end;
  if not RunRequiredCpuLocalSttModelInstalls() then begin
    Log('Local STT provisioning did not complete; continuing app install without bundled ASR model. The app can retry the local STT model download at runtime.');
  end;
end;
