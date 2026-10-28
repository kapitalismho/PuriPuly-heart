var
  DownloadPage: TDownloadWizardPage;
  LocalSttPlanPrepared: Boolean;
  AsrNeedsDownload: array of Boolean;
  LocalSttRequiredDownloadBytes: Int64;
  LocalSttCompletedDownloadBytes: Int64;
  LocalSttDisplayedDownloadBytes: Int64;
  LocalSttCurrentFileExpectedBytes: Int64;
  LocalSttCurrentFileLabel: String;
  LocalSttProgressFailed: Boolean;

const
  LocalSttDiskSpaceMargin = 67108864;

#include LocalASRModelsInclude
function AsrCreateFile(Name: String; Access, Sharing: DWORD; Security: NativeUInt;
  Creation, Flags: DWORD; Template: THandle): THandle;
  external 'CreateFileW@kernel32.dll stdcall';
function AsrLockFile(Handle: THandle; OffsetLow, OffsetHigh, BytesLow, BytesHigh: DWORD): Boolean;
  external 'LockFile@kernel32.dll stdcall';
function AsrUnlockFile(Handle: THandle; OffsetLow, OffsetHigh, BytesLow, BytesHigh: DWORD): Boolean;
  external 'UnlockFile@kernel32.dll stdcall';
function AsrCloseHandle(Handle: THandle): Boolean;
  external 'CloseHandle@kernel32.dll stdcall';
function AsrGetFileAttributes(Name: String): DWORD;
  external 'GetFileAttributesW@kernel32.dll stdcall';

function AsrPathIsPlain(Path: String): Boolean;
var
  Parent: String;
  Attributes: DWORD;
begin
  Result := False;
  Path := RemoveBackslashUnlessRoot(ExpandFileName(Path));
  while Path <> '' do begin
    Attributes := AsrGetFileAttributes(Path);
    if (Attributes <> $FFFFFFFF) and ((Attributes and $400) <> 0) then exit;
    Parent := ExtractFileDir(Path);
    if Parent = Path then break;
    Path := Parent;
  end;
  Result := True;
end;

function AcquireAsrProvisioningLease(var Handle: THandle): Boolean;
var
  Root: String;
  LockPath: String;
begin
  Result := False;
  Root := AddBackslash(ResolveLocalSttAppDataRoot()) + 'models';
  if not AsrPathIsPlain(Root) or not ForceDirectories(Root) then exit;
  LockPath := AddBackslash(Root) + '.local-asr-provisioning.lock';
  if not AsrPathIsPlain(LockPath) then exit;
  Handle := AsrCreateFile(LockPath, $C0000000, 3, 0, 4, $80, 0);
  if Handle = THandle(-1) then exit;
  Result := AsrLockFile(Handle, 0, 0, 1, 0);
  if not Result then AsrCloseHandle(Handle);
end;

procedure ReleaseAsrProvisioningLease(Handle: THandle);
begin
  if not AsrUnlockFile(Handle, 0, 0, 1, 0) then
    Log('Local STT provisioning lease unlock failed.');
  if not AsrCloseHandle(Handle) then
    Log('Local STT provisioning lease handle close failed.');
end;


function PreferredLocalSttSource(ChineseSource: String): String;
begin
  if (ActiveLanguage = 'chinesesimplified') or
     (ActiveLanguage = 'chinesetraditional') then begin
    Result := ChineseSource;
  end else begin
    Result := 'huggingface';
  end;
end;

function ValidateLocalSttAsset(BaseDir: String; RelativePath: String; Sha256: String; ExpectedSize: Int64): Boolean;
var
  AssetPath: String;
  ActualSize: Int64;
begin
  AssetPath := AddBackslash(BaseDir) + RelativePath;
  Result := False;
  if not AsrPathIsPlain(AssetPath) then exit;
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
  try
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
  except
    LocalSttProgressFailed := True;
    RaiseException(GetExceptionMessage);
  end;
end;

function DownloadLocalSttFile(Url: String; BaseName: String; Sha256: String; ExpectedSize: Int64; DisplayName: String): Boolean;
begin
  Result := False;
  if DownloadPage.AbortedByUser then begin
    exit;
  end;
  LocalSttProgressFailed := False;
  LocalSttCurrentFileExpectedBytes := ExpectedSize;
  LocalSttCurrentFileLabel := DisplayName;
  try
    DownloadTemporaryFile(Url, BaseName, Sha256, @DownloadLocalSttProgress);
  except
    if LocalSttProgressFailed then begin
      RaiseException(GetExceptionMessage);
    end;
    Log('Local STT asset download failed: ' + DisplayName + ': ' + GetExceptionMessage);
    exit;
  end;
  LocalSttCompletedDownloadBytes := LocalSttCompletedDownloadBytes + ExpectedSize;
  if LocalSttCompletedDownloadBytes > LocalSttDisplayedDownloadBytes then begin
    LocalSttDisplayedDownloadBytes := LocalSttCompletedDownloadBytes;
  end;
  Result := True;
end;

function GetAsrModelInstallDir(ModelIndex: Integer): String;
begin
  Result := AddBackslash(ResolveLocalSttAppDataRoot()) + 'models\' + AsrInstallDirname(ModelIndex);
end;

function AsrInstalledIdentityMatches(BaseDir: String; ModelIndex: Integer): Boolean;
var
  ManifestText: AnsiString;
  Normalized: String;
begin
  Result := False;
  if not AsrPathIsPlain(BaseDir) then exit;
  if not LoadStringFromFile(AddBackslash(BaseDir) + 'installed-manifest.json', ManifestText) then exit;
  Normalized := String(ManifestText);
  StringChangeEx(Normalized, #13#10, #10, True);
  Result := (Trim(Normalized) = Trim(AsrInstalledManifest(ModelIndex, 'huggingface'))) or
    (Trim(Normalized) = Trim(AsrInstalledManifest(ModelIndex, 'hf_mirror')));
end;

function ValidateAsrModelInstall(BaseDir: String; ModelIndex: Integer): Boolean;
var
  FileIndex: Integer;
begin
  Result := False;
  if not AsrInstalledIdentityMatches(BaseDir, ModelIndex) then exit;
  for FileIndex := 0 to AsrFileCount(ModelIndex) - 1 do begin
    if not ValidateLocalSttAsset(BaseDir, AsrFilePath(ModelIndex, FileIndex),
      AsrFileSha256(ModelIndex, FileIndex), AsrFileSize(ModelIndex, FileIndex)) then exit;
  end;
  Result := True;
end;

function AsrOwnedEntries(BaseDir, Prefix: String; ModelIndex: Integer): Boolean;
var
  Entry: TFindRec;
  Relative: String;
  Expected: String;
  Known: Boolean;
  FileIndex: Integer;
begin
  Result := False;
  if not AsrPathIsPlain(BaseDir) then exit;
  if FindFirst(AddBackslash(BaseDir) + '*', Entry) then begin
    try
      repeat
        if (Entry.Name <> '.') and (Entry.Name <> '..') then begin
          if (Entry.Attributes and $400) <> 0 then exit;
          Relative := Prefix + Entry.Name;
          Known := CompareText(Relative, 'installed-manifest.json') = 0;
          for FileIndex := 0 to AsrFileCount(ModelIndex) - 1 do begin
            Expected := AsrFilePath(ModelIndex, FileIndex);
            if (Entry.Attributes and $10) <> 0 then begin
              if CompareText(Copy(Expected, 1, Length(Relative) + 1), Relative + '\') = 0 then Known := True;
            end else if CompareText(Relative, Expected) = 0 then Known := True;
          end;
          if not Known then exit;
          if ((Entry.Attributes and $10) <> 0) and
             not AsrOwnedEntries(AddBackslash(BaseDir) + Entry.Name, Relative + '\', ModelIndex) then exit;
        end;
      until not FindNext(Entry);
    finally
      FindClose(Entry);
    end;
  end;
  Result := True;
end;

function RemoveAsrOwnedDirectory(Path: String; ModelIndex: Integer): Boolean;
begin
  Result := False;
  if not DirExists(Path) then begin
    Result := not FileExists(Path);
    exit;
  end;
  if not AsrOwnedEntries(Path, '', ModelIndex) then exit;
  Result := DelTree(Path, True, True, True);
end;

function PromoteLocalSttInstallTo(StagingDir, InstallDir: String; ModelIndex: Integer): Boolean;
var
  BackupDir: String;
begin
  Result := False;
  BackupDir := InstallDir + '.backup';
  if not AsrPathIsPlain(InstallDir) or not AsrPathIsPlain(BackupDir) then exit;
  if DirExists(BackupDir) then begin
    if not AsrInstalledIdentityMatches(BackupDir, ModelIndex) then exit;
    if not DirExists(InstallDir) then begin
      if not RenameFile(BackupDir, InstallDir) then exit;
    end else if not RemoveAsrOwnedDirectory(BackupDir, ModelIndex) then exit;
  end;
  if DirExists(InstallDir) then begin
    if not AsrInstalledIdentityMatches(InstallDir, ModelIndex) or
       not AsrOwnedEntries(InstallDir, '', ModelIndex) then exit;
    if not RenameFile(InstallDir, BackupDir) then exit;
  end;
  Result := RenameFile(StagingDir, InstallDir);
  if Result then begin
    if not RemoveAsrOwnedDirectory(BackupDir, ModelIndex) then begin
      Log('Local STT model promoted; backup deletion remains pending.');
      Result := False;
    end;
  end else if DirExists(BackupDir) then begin
    if not RenameFile(BackupDir, InstallDir) then
      Log('Local STT promotion failed; previous model remains in backup.');
  end;
end;

function DownloadAsrModelSource(ModelIndex: Integer; SourceName: String; var SourceFailed: Boolean): Boolean;
var
  StagingDir: String;
  FileIndex: Integer;
  TemporaryName: String;
  Url: String;
begin
  Result := False;
  SourceFailed := False;
  StagingDir := GetAsrModelInstallDir(ModelIndex) + '.staging-' +
    ExtractFileName(GenerateUniqueName(ExtractFileDir(GetAsrModelInstallDir(ModelIndex)), ''));
  if DirExists(StagingDir) or FileExists(StagingDir) or not AsrPathIsPlain(StagingDir) then exit;
  try
    if not ForceDirectories(StagingDir) then exit;
    for FileIndex := 0 to AsrFileCount(ModelIndex) - 1 do begin
      if SourceName = 'hf_mirror' then
        Url := AsrMirrorUrl(ModelIndex, FileIndex)
      else
        Url := AsrPrimaryUrl(ModelIndex, FileIndex);
      TemporaryName := 'asr-' + IntToStr(ModelIndex) + '-' + IntToStr(FileIndex) + '.download';
      if not DownloadLocalSttFile(Url, TemporaryName, AsrFileSha256(ModelIndex, FileIndex),
        AsrFileSize(ModelIndex, FileIndex), AsrModelId(ModelIndex)) then begin
        SourceFailed := not DownloadPage.AbortedByUser;
        exit;
      end;
      if not CopyLocalSttAsset(StagingDir, TemporaryName, AsrFilePath(ModelIndex, FileIndex)) then exit;
    end;
    if not SaveStringToFile(AddBackslash(StagingDir) + 'installed-manifest.json',
      AsrInstalledManifest(ModelIndex, SourceName), False) then exit;
    SourceFailed := not ValidateAsrModelInstall(StagingDir, ModelIndex);
    if SourceFailed or DownloadPage.AbortedByUser then exit;
    Result := PromoteLocalSttInstallTo(StagingDir, GetAsrModelInstallDir(ModelIndex), ModelIndex);
  finally
    if not Result and DirExists(StagingDir) then
      if not RemoveAsrOwnedDirectory(StagingDir, ModelIndex) then
        Log('Local STT staging deletion remains pending.');
  end;
end;

function RunAsrModelInstall(ModelIndex: Integer): Boolean;
var
  SourceName: String;
  AlternateSource: String;
  SourceFailed: Boolean;
  DownloadStartBytes: Int64;
begin
  if not AsrNeedsDownload[ModelIndex] then begin
    Result := True;
    exit;
  end;
  SourceName := PreferredLocalSttSource('hf_mirror');
  if SourceName = 'huggingface' then AlternateSource := 'hf_mirror'
  else AlternateSource := 'huggingface';
  DownloadStartBytes := LocalSttCompletedDownloadBytes;
  Result := DownloadAsrModelSource(ModelIndex, SourceName, SourceFailed);
  if Result or not SourceFailed or DownloadPage.AbortedByUser then exit;
  LocalSttCompletedDownloadBytes := DownloadStartBytes;
  Result := DownloadAsrModelSource(ModelIndex, AlternateSource, SourceFailed);
end;
procedure PrepareLocalSttDownloadPlan();
var
  ModelIndex: Integer;
  ReDownloadSelected: Boolean;
begin
  if LocalSttPlanPrepared then exit;
  ReDownloadSelected := WizardIsTaskSelected('redownloadasr');
  SetArrayLength(AsrNeedsDownload, AsrModelCount());
  LocalSttRequiredDownloadBytes := 0;
  for ModelIndex := 0 to AsrModelCount() - 1 do begin
#ifdef SkipLocalSttProvisioning
    AsrNeedsDownload[ModelIndex] := False;
#else
    AsrNeedsDownload[ModelIndex] := ReDownloadSelected or
      not ValidateAsrModelInstall(GetAsrModelInstallDir(ModelIndex), ModelIndex);
#endif
    if AsrNeedsDownload[ModelIndex] then
      LocalSttRequiredDownloadBytes := LocalSttRequiredDownloadBytes + AsrModelSize(ModelIndex);
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
  ModelIndex: Integer;
  ModelInstalled: Boolean;
  SegmentStart: Int64;
begin
  PrepareLocalSttDownloadPlan();
  Result := True;
  if LocalSttRequiredDownloadBytes = 0 then exit;
  LocalSttCompletedDownloadBytes := 0;
  LocalSttDisplayedDownloadBytes := 0;
  DownloadPage.SetProgress(0, 10000);
  DownloadPage.Show;
  try
    for ModelIndex := 0 to AsrModelCount() - 1 do begin
      SegmentStart := LocalSttCompletedDownloadBytes;
      ModelInstalled := RunAsrModelInstall(ModelIndex);
      if DownloadPage.AbortedByUser then begin
        Result := False;
        exit;
      end;
      if not ModelInstalled then Result := False;
      if AsrNeedsDownload[ModelIndex] and ModelInstalled then
        CompleteLocalSttDownloadSegment(SegmentStart, AsrModelSize(ModelIndex));
    end;
  finally
    DownloadPage.Hide;
  end;
end;

function RunSetupLocalSttProvisioning(): String;
var
  Lease: THandle;
begin
  Result := '';
#ifdef SkipLocalSttProvisioning
  Log('Local STT provisioning skipped for isolated installer smoke.');
  exit;
#endif
  if not AcquireAsrProvisioningLease(Lease) then begin
    Log('Local STT provisioning lease is unavailable or the model root is protected.');
    Result := CustomMessage('LocalSttDownloadFailed');
    exit;
  end;
  try
    LocalSttPlanPrepared := False;
    if not CheckLocalSttDiskSpace(Result) then exit;
    if not RunRequiredCpuLocalSttModelInstalls() then
      Result := CustomMessage('LocalSttDownloadFailed');
  finally
    ReleaseAsrProvisioningLease(Lease);
  end;
end;
