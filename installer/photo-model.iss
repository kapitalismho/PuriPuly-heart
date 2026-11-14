#define PhotoModelRevision "12c0ab260ba27a0bffb5d7a51ced06e36b04a1c0"
#define PhotoModelRelativePath "assets\photo\mbnv4_text_seg.onnx"
#ifndef PhotoModelSize
  #define PhotoModelSize 48103781
#endif
#ifndef PhotoModelPrimaryUrl
  #define PhotoModelPrimaryUrl "https://huggingface.co/salee1233/mobilenetv4-text-seg-onnx/resolve/" + PhotoModelRevision + "/mbnv4_text_seg.onnx"
#endif
#ifndef PhotoModelMirrorUrl
  #define PhotoModelMirrorUrl "https://hf-mirror.net/salee1233/mobilenetv4-text-seg-onnx/resolve/" + PhotoModelRevision + "/mbnv4_text_seg.onnx"
#endif
#ifdef SkipPhotoModelProvisioning
  #ifndef InstallerSmokeAppDataRoot
    #error Skipping photo provisioning requires an isolated installer smoke profile.
  #endif
#endif

[Files]
Source: "{tmp}\photo-mask.download"; DestDir: "{app}\assets\photo"; DestName: "mbnv4_text_seg.onnx"; ExternalSize: {#PhotoModelSize}; Flags: external ignoreversion; Check: ShouldInstallPhotoModel

[Code]
var
  PhotoModelNeedsDownload: Boolean;

function ShouldInstallPhotoModel(): Boolean;
begin
  Result := PhotoModelNeedsDownload;
end;

function PreparePhotoModelDownloadPlan(): Int64;
begin
#ifdef SkipPhotoModelProvisioning
  PhotoModelNeedsDownload := False;
#else
  PhotoModelNeedsDownload := WizardIsTaskSelected('redownloadasr') or
    not ValidateLocalSttAsset(ExpandConstant('{app}'), '{#PhotoModelRelativePath}', {#PhotoModelSize});
#endif
  Result := 0;
  if PhotoModelNeedsDownload then Result := {#PhotoModelSize};
end;

function CheckPhotoModelDiskSpace(var ErrorMessage: String): Boolean;
var
  TempVolume, InstallVolume: String;
  FreeBytes, TotalBytes, RequiredBytes: Int64;
begin
  Result := False;
  TempVolume := LocalSttStorageVolume(ExpandConstant('{tmp}'));
  InstallVolume := LocalSttStorageVolume(ExpandConstant('{app}'));
  RequiredBytes := {#PhotoModelSize} + LocalSttDiskSpaceMargin;
  if CompareText(TempVolume, InstallVolume) = 0 then
    RequiredBytes := RequiredBytes + {#PhotoModelSize};
  if not GetSpaceOnDisk64(TempVolume, FreeBytes, TotalBytes) then begin
    ErrorMessage := FmtMessage(CustomMessage('LocalSttDiskSpaceCheckFailed'), [TempVolume]);
    exit;
  end;
  if FreeBytes < RequiredBytes then begin
    ErrorMessage := FmtMessage(CustomMessage('LocalSttDiskSpaceFailed'), [
      FormatByteSize(RequiredBytes), FormatByteSize(FreeBytes), TempVolume]);
    exit;
  end;
  if CompareText(TempVolume, InstallVolume) <> 0 then begin
    if not GetSpaceOnDisk64(InstallVolume, FreeBytes, TotalBytes) then begin
      ErrorMessage := FmtMessage(CustomMessage('LocalSttDiskSpaceCheckFailed'), [InstallVolume]);
      exit;
    end;
    if FreeBytes < RequiredBytes then begin
      ErrorMessage := FmtMessage(CustomMessage('LocalSttDiskSpaceFailed'), [
        FormatByteSize(RequiredBytes), FormatByteSize(FreeBytes), InstallVolume]);
      exit;
    end;
  end;
  Result := True;
end;

function DownloadPhotoModelSource(Url: String): Boolean;
var
  TemporaryPath: String;
begin
  Result := False;
  TemporaryPath := ExpandConstant('{tmp}\photo-mask.download');
  if FileExists(TemporaryPath) and not DeleteFile(TemporaryPath) then exit;
  LocalSttCompletedDownloadBytes := 0;
  if not DownloadLocalSttFile(Url, 'photo-mask.download', {#PhotoModelSize},
    'MobileNetV4 Text Segmentation (ONNX)') then exit;
  Result := ValidateLocalSttAsset(ExpandConstant('{tmp}'), 'photo-mask.download', {#PhotoModelSize});
end;

function RunSetupPhotoModelProvisioning(): String;
var
  PrimaryUrl, AlternateUrl: String;
  Downloaded: Boolean;
begin
  Result := '';
  if PreparePhotoModelDownloadPlan() = 0 then exit;
  Result := CustomMessage('LocalSttDownloadFailed');
  if not AsrPathIsPlain(ExpandConstant('{app}\{#PhotoModelRelativePath}')) then exit;
  if not CheckPhotoModelDiskSpace(Result) then exit;
  Result := CustomMessage('LocalSttDownloadFailed');
  if PreferredLocalSttSource('hf_mirror') = 'hf_mirror' then begin
    PrimaryUrl := '{#PhotoModelMirrorUrl}';
    AlternateUrl := '{#PhotoModelPrimaryUrl}';
  end else begin
    PrimaryUrl := '{#PhotoModelPrimaryUrl}';
    AlternateUrl := '{#PhotoModelMirrorUrl}';
  end;
  LocalSttRequiredDownloadBytes := {#PhotoModelSize};
  LocalSttCompletedDownloadBytes := 0;
  LocalSttDisplayedDownloadBytes := 0;
  DownloadPage.SetProgress(0, 10000);
  DownloadPage.Show;
  try
    Downloaded := DownloadPhotoModelSource(PrimaryUrl);
    if not Downloaded and not DownloadPage.AbortedByUser then
      Downloaded := DownloadPhotoModelSource(AlternateUrl);
    if Downloaded and not DownloadPage.AbortedByUser then Result := '';
  finally
    DownloadPage.Hide;
    LocalSttPlanPrepared := False;
  end;
end;
