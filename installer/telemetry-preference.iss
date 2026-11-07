function ResolveInstallerTelemetryConfigPath(): String;
begin
#ifdef InstallerSmokeAppDataRoot
  Result := AddBackslash(ResolveInstallerSmokeAppDataRoot('')) + 'settings.json';
#else
  Result := '';
#endif
end;

function ProbeExistingTelemetryPreference(
  var Exists: Boolean;
  var Enabled: Boolean
): Boolean;
var
  PowerShellPath: String;
  PowerShellScript: String;
  ResultCode: Integer;
begin
  Exists := False;
  Enabled := True;
  PowerShellPath := ExpandConstant('{sys}\WindowsPowerShell\v1.0\powershell.exe');
  if ResolveInstallerTelemetryConfigPath() <> '' then begin
    PowerShellScript := '$p=''' + ResolveInstallerTelemetryConfigPath() + ''';';
  end else begin
    PowerShellScript :=
      '$base=$env:LOCALAPPDATA;' +
      'if ([string]::IsNullOrWhiteSpace($base)) { $base=$env:APPDATA };' +
      'if ([string]::IsNullOrWhiteSpace($base)) { $base=Join-Path $HOME ''AppData\Local'' };' +
      '$p=Join-Path $base ''puripuly-heart\settings.json'';';
  end;
  PowerShellScript := PowerShellScript +
    'if (!(Test-Path -LiteralPath $p)) { exit 22 };' +
    'try {' +
    '$j=Get-Content -LiteralPath $p -Raw -Encoding UTF8 | ConvertFrom-Json -ErrorAction Stop;' +
    'if (($null -eq $j) -or ($j -isnot [pscustomobject])) { exit 23 };' +
    '$ip=$j.PSObject.Properties[''intent''];$sp=$j.PSObject.Properties[''state''];' +
    'if (($null -eq $ip) -and ($null -eq $sp)) { exit 20 };' +
    'if (($null -eq $ip) -or ($null -eq $sp)) { exit 23 };' +
    '$vp=$j.PSObject.Properties[''settings_version''];' +
    'if (($null -eq $vp) -or ' +
    '((($vp.Value -isnot [int]) -and ($vp.Value -isnot [long]))) -or ' +
    '($vp.Value -lt 1) -or ($vp.Value -gt {#CanonicalSettingsVersion})) { exit 23 };' +
    '$tp=$j.intent.PSObject.Properties[''telemetry''];' +
    'if ($null -eq $tp) { exit 20 };$t=$tp.Value;' +
    'if ($t -isnot [pscustomobject]) { exit 23 };' +
    '$ep=$t.PSObject.Properties[''enabled''];$cp=$t.PSObject.Properties[''consent''];' +
    'if (($null -ne $cp) -and ($cp.Value -eq ''decline'')) { exit 21 };' +
    'if ($null -eq $ep) {' +
    'if (($null -eq $cp) -or ($cp.Value -eq ''allow'') -or ($cp.Value -eq ''unknown'')) { exit 20 } else { exit 21 }};' +
    'if ($ep.Value -is [bool]) { if ($ep.Value) { exit 20 } else { exit 21 }};' +
    'exit 23' +
    '} catch { exit 23 }';
  Result := ExecAsOriginalUser(
    PowerShellPath,
    '-NoProfile -NonInteractive -Command "' + PowerShellScript + '"',
    '',
    SW_HIDE,
    ewWaitUntilTerminated,
    ResultCode
  );
  if not Result then begin
    exit;
  end;
  Exists := ResultCode <> 22;
  if ResultCode = 20 then begin
    Enabled := True;
  end else if ResultCode = 21 then begin
    Enabled := False;
  end else if ResultCode <> 22 then begin
    Result := False;
  end;
end;
