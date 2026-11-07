[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ToolPython,

    [Parameter(Mandatory = $true)]
    [string]$PinnedInputRoot,

    [Parameter(Mandatory = $true)]
    [string]$PythonEmbedArchive,

    [Parameter()]
    [string]$InnoSetupVersion = "7.1.0",

    [Parameter()]
    [string]$InnoSetupCompiler = "",

    [Parameter()]
    [string]$SoxrBuildEnvironment = ".venv",

    [Parameter()]
    [string]$BuildRoot = "C:\d177\native-integration\release\build",

    [Parameter()]
    [string]$OutputDir = "C:\d177\native-integration\release\dist",

    [Parameter()]
    [string]$InstallerOutputDir = "installer_output\native",

    [Parameter()]
    [switch]$SkipNativeBuild,

    [Parameter()]
    [switch]$VerifyInstallerLifecycle,

    [Parameter()]
    [string]$InstallerSmokeRoot = "",

    [Parameter()]
    [string]$InstallerSmokeInstallDir = "",

    [Parameter()]
    [string]$InstallerSmokeAppId = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if ($PSVersionTable.PSVersion.Major -lt 7 -or [System.Environment]::OSVersion.Platform -ne "Win32NT") {
    throw "Native releases require PowerShell 7 on Windows x64."
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter()][string[]]$ArgumentList = @()
    )
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    Write-Host "Starting: $FilePath $($ArgumentList -join ' ')"
    try {
        & $FilePath @ArgumentList
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($ArgumentList -join ' ')"
        }
    } finally {
        $timer.Stop()
        Write-Host ("Ended after {0:F3}s: {1}" -f $timer.Elapsed.TotalSeconds, $FilePath)
    }
}

function Resolve-RepoPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $Path))
}

function Invoke-IsolatedProcess {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter(Mandatory = $true)][string]$StateRoot,
        [Parameter()][string]$RuntimeRoot = "",
        [Parameter()][switch]$AllowFailure
    )
    $profile = Join-Path $StateRoot "profile"
    foreach ($directory in @($profile, "$profile\localappdata", "$profile\appdata")) {
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
    }
    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $FilePath
    $startInfo.WorkingDirectory = $StateRoot
    $startInfo.UseShellExecute = $false
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $startInfo.StandardErrorEncoding = [System.Text.Encoding]::UTF8
    $startInfo.Environment.Clear()
    foreach ($name in @("SystemRoot", "WINDIR", "COMSPEC", "ProgramFiles", "ProgramFiles(x86)")) {
        $value = [Environment]::GetEnvironmentVariable($name)
        if ($value) { $startInfo.Environment[$name] = $value }
    }
    $startInfo.Environment["PATH"] = "$env:SystemRoot\System32"
    $startInfo.Environment["HOME"] = $profile
    $startInfo.Environment["USERPROFILE"] = $profile
    $startInfo.Environment["APPDATA"] = "$profile\appdata"
    $startInfo.Environment["LOCALAPPDATA"] = "$profile\localappdata"
    $startInfo.Environment["TEMP"] = $profile
    $startInfo.Environment["TMP"] = $profile
    $startInfo.Environment["PYTHON_KEYRING_BACKEND"] = "keyring.backends.fail.Keyring"
    $startInfo.Environment["PYTHONNOUSERSITE"] = "1"
    $startInfo.Environment["PYTHONSAFEPATH"] = "1"
    $startInfo.Environment["PYTHONDONTWRITEBYTECODE"] = "1"
    $startInfo.Environment["PYTHONUTF8"] = "1"
    if ($RuntimeRoot) {
        $resources = Join-Path $RuntimeRoot "app"
        $startInfo.Environment["PYTHONHOME"] = $RuntimeRoot
        $startInfo.Environment["PYTHONPATH"] = "$resources\python.zip;$RuntimeRoot\python314.zip;$resources;$RuntimeRoot\site-packages"
        $startInfo.Environment["PATH"] = "$RuntimeRoot;$RuntimeRoot\DLLs;$RuntimeRoot\site-packages;$env:SystemRoot\System32"
        $startInfo.Environment["PURIPULY_HEART_NATIVE_RESOURCE_ROOT"] = $resources
        $startInfo.Environment["PURIPULY_HEART_NATIVE_RUNTIME_ROOT"] = $RuntimeRoot
        $startInfo.Environment["PURIPULY_HEART_NATIVE_HOST_EXECUTABLE"] = Join-Path $RuntimeRoot "PuriPulyHeart.exe"
        $startInfo.Environment["PURIPULY_HEART_NATIVE_PYTHON_EXECUTABLE"] = Join-Path $RuntimeRoot "python.exe"
    }
    foreach ($argument in $ArgumentList) { $startInfo.ArgumentList.Add($argument) }
    $timer = [System.Diagnostics.Stopwatch]::StartNew()
    Write-Host "Starting isolated process: $FilePath $($ArgumentList -join ' ')"
    $process = [System.Diagnostics.Process]::Start($startInfo)
    try {
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        $process.WaitForExit()
        $result = [pscustomobject]@{
            ExitCode = $process.ExitCode
            Output = $stdout.GetAwaiter().GetResult().Trim()
            Errors = $stderr.GetAwaiter().GetResult().Trim()
        }
        if ($result.ExitCode -ne 0 -and -not $AllowFailure) {
            throw "Isolated process failed: exit $($result.ExitCode)`n$($result.Output)`n$($result.Errors)"
        }
        return $result
    } finally {
        $process.Dispose()
        $timer.Stop()
        Write-Host ("Ended after {0:F3}s: {1}" -f $timer.Elapsed.TotalSeconds, $FilePath)
    }
}

function Invoke-Headless {
    param(
        [Parameter(Mandatory = $true)][string[]]$ArgumentList,
        [Parameter()][string]$RuntimeRoot = $OutputDir,
        [Parameter()][string]$StateRoot = $smokeRoot
    )
    $result = Invoke-IsolatedProcess -FilePath (Join-Path $RuntimeRoot "PuriPulyHeart.exe") `
        -ArgumentList (@("--headless") + $ArgumentList) -StateRoot $StateRoot -RuntimeRoot $RuntimeRoot
    if ($result.Errors) { Write-Host $result.Errors }
    return $result.Output
}

function Assert-NativeInstalledRuntime {
    param(
        [Parameter(Mandatory = $true)][string]$RuntimeRoot,
        [Parameter(Mandatory = $true)][string]$StateRoot,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $version = Invoke-Headless -RuntimeRoot $RuntimeRoot -StateRoot $StateRoot -ArgumentList @("--version")
    if ($version -ne $appVersion) { throw "$Label runtime version mismatch: $version" }
    foreach ($command in @("gui-startup-check", "local-sherpa-runtime-check", "hf-xet-runtime-check")) {
        Write-Host (Invoke-Headless -RuntimeRoot $RuntimeRoot -StateRoot $StateRoot -ArgumentList @($command))
    }
    [void](Invoke-IsolatedProcess -FilePath (Join-Path $RuntimeRoot "PuriPulyHeartOverlay.exe") `
        -ArgumentList @("--check-startup-contract") -StateRoot $StateRoot)
    $gpu = Invoke-IsolatedProcess -FilePath (Join-Path $RuntimeRoot "PuriPulyHeartASRWorker.exe") `
        -ArgumentList @("--version") -StateRoot $StateRoot
    if ($gpu.Output -ne $appVersion) { throw "$Label ASR worker version mismatch." }
    $reportPath = Join-Path $StateRoot "$Label-soxr.json"
    $soxrCheck = "import os,sys; os.environ['PURIPULY_HEART_SOXR_RUNTIME_REPORT_PATH']=sys.argv[1]; from puripuly_heart.app.soxr_runtime_check import run_soxr_runtime_check; raise SystemExit(run_soxr_runtime_check())"
    [void](Invoke-IsolatedProcess -FilePath (Join-Path $RuntimeRoot "python.exe") `
        -ArgumentList @("-c", $soxrCheck, $reportPath) -StateRoot $StateRoot -RuntimeRoot $RuntimeRoot)
    $report = Get-Content -LiteralPath $reportPath -Raw -Encoding utf8 | ConvertFrom-Json
    $soxrRoot = Join-Path $RuntimeRoot "site-packages\soxr"
    $extension = @(Get-ChildItem -LiteralPath $soxrRoot -Filter "soxr_ext*.pyd" -File)
    if ($extension.Count -ne 1 -or $report.imported_extension_path -ne $extension[0].FullName -or
        $report.loaded_sibling_dll_path -ne (Join-Path $soxrRoot "soxr.dll")) {
        throw "$Label soxr loaded outside its intended native payload."
    }
    $captureReport = Join-Path $StateRoot "$Label-process-capture.json"
    [void](Invoke-IsolatedProcess -FilePath (Join-Path $RuntimeRoot "python.exe") `
        -ArgumentList @("-m", "puripuly_heart.release_evidence.process_capture_packaged_smoke", "--artifact-root", $RuntimeRoot, "--report", $captureReport) `
        -StateRoot $StateRoot -RuntimeRoot $RuntimeRoot)
    $validateCapture = "import json,sys; from pathlib import Path; from puripuly_heart.release_evidence.windows_process_distribution import validate_runtime_report; validate_runtime_report(json.loads(Path(sys.argv[1]).read_text(encoding='utf-8')), expected_root=Path(sys.argv[2]))"
    [void](Invoke-IsolatedProcess -FilePath $ToolPython -ArgumentList @("-c", $validateCapture, $captureReport, $RuntimeRoot) -StateRoot $StateRoot)
    [void](Invoke-IsolatedProcess -FilePath $ToolPython -ArgumentList @(
        "-m", "puripuly_heart.release_evidence.managed_gemma_distribution", "verify-package", $RuntimeRoot, "--launch"
    ) -StateRoot $StateRoot)
    Copy-Item -LiteralPath $reportPath, $captureReport -Destination $evidenceRoot -Force
}

function Remove-InstallerLifecycleInstallations {
    param(
        [Parameter(Mandatory = $true)][Guid]$Identity,
        [Parameter(Mandatory = $true)][AllowEmptyCollection()][string[]]$KnownTargets,
        [Parameter(Mandatory = $true)][string]$DefaultDirectoryName,
        [Parameter(Mandatory = $true)][string]$ExpectedHostSha256,
        [Parameter(Mandatory = $true)][string]$StateRoot,
        [Parameter(Mandatory = $true)][string]$WorkspaceRoot
    )
    if ($Identity -eq [Guid]"A1B2C3D4-E5F6-7890-ABCD-EF1234567890" -or
        $DefaultDirectoryName -notmatch '^PuriPuly-native-smoke-[0-9a-f]{32}$' -or
        $ExpectedHostSha256 -notmatch '^[0-9a-fA-F]{64}$') {
        throw "Installer lifecycle cleanup requires its unique alternate identity and candidate hash."
    }
    $registration = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\{$Identity}_is1"
    $targets = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($path in $KnownTargets) { [void]$targets.Add([IO.Path]::GetFullPath($path).TrimEnd('\')) }
    $results = [Collections.Generic.List[object]]::new()
    if (Test-Path -LiteralPath $registration) {
        try {
            $registered = [IO.Path]::GetFullPath(
                (Get-ItemProperty -LiteralPath $registration -Name InstallLocation -ErrorAction Stop).InstallLocation
            ).TrimEnd('\')
            if (-not $targets.Contains($registered) -and [IO.Path]::GetFileName($registered) -ne $DefaultDirectoryName) {
                throw "Alternate AppId installation is outside the current run's exact owned targets."
            }
            [void]$targets.Add($registered)
        } catch {
            $results.Add(@{ target = $registration; disposition = "failed"; error = $_.Exception.Message })
        }
    }
    foreach ($target in $targets) {
        try {
            if ([IO.Path]::GetFileName($target) -eq "PuriPulyHeart" -or $target -eq [IO.Path]::GetPathRoot($target) -or
                $target -eq $WorkspaceRoot -or $target.StartsWith("$WorkspaceRoot\", [StringComparison]::OrdinalIgnoreCase) -or
                $WorkspaceRoot.StartsWith("$target\", [StringComparison]::OrdinalIgnoreCase)) {
                throw "Refusing installer lifecycle cleanup at or across the checkout."
            }
            $ancestor = [IO.DirectoryInfo]::new($target)
            while ($null -ne $ancestor) {
                if ($ancestor.Exists -and ($ancestor.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
                    throw "Refusing installer lifecycle cleanup through a reparse point."
                }
                $ancestor = $ancestor.Parent
            }
            $uninstaller = Join-Path $target "unins000.exe"
            $hostPath = Join-Path $target "PuriPulyHeart.exe"
            if (-not (Test-Path -LiteralPath $uninstaller -PathType Leaf)) {
                if (Test-Path -LiteralPath $hostPath) { throw "Owned payload has no uninstaller; retained for explicit recovery." }
                $results.Add(@{ target = $target; disposition = "absent"; error = $null })
                continue
            }
            if (-not (Test-Path -LiteralPath $hostPath -PathType Leaf) -or
                (Get-FileHash -LiteralPath $hostPath -Algorithm SHA256).Hash -ne $ExpectedHostSha256) {
                throw "Refusing lifecycle uninstaller for an unrecognized installed candidate."
            }
            [void](Invoke-IsolatedProcess -FilePath $uninstaller `
                -ArgumentList @("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART") -StateRoot $StateRoot)
            $deadline = [DateTime]::UtcNow.AddSeconds(15)
            while ((Test-Path -LiteralPath $uninstaller) -and [DateTime]::UtcNow -lt $deadline) { Start-Sleep -Milliseconds 100 }
            if ((Test-Path -LiteralPath $uninstaller) -or (Test-Path -LiteralPath $hostPath)) {
                throw "Owned uninstaller left its executable payload."
            }
            $results.Add(@{ target = $target; disposition = "uninstalled"; error = $null })
        } catch {
            $results.Add(@{ target = $target; disposition = "failed"; error = $_.Exception.Message })
        }
    }
    if (Test-Path -LiteralPath $registration) {
        $results.Add(@{ target = $registration; disposition = "failed"; error = "Alternate AppId registration remains; no registry deletion attempted." })
    }
    return $results.ToArray()
}

function Invoke-InstallerLifecycle {
    $runId = [Guid]::NewGuid().ToString("N")
    $state = if ($InstallerSmokeRoot) { Resolve-RepoPath $InstallerSmokeRoot } else { Join-Path ([IO.Path]::GetTempPath()) "PuriPuly-native-lifecycle-$runId" }
    $install = if ($InstallerSmokeInstallDir) { Resolve-RepoPath $InstallerSmokeInstallDir } else {
        Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) "Programs\PuriPuly-native-lifecycle-$runId"
    }
    $identity = if ($InstallerSmokeAppId) { [Guid]::Parse($InstallerSmokeAppId.Trim("{}")) } else { [Guid]::NewGuid() }
    if ($identity -eq [Guid]"A1B2C3D4-E5F6-7890-ABCD-EF1234567890") {
        throw "Installer lifecycle requires a non-production AppId."
    }
    $child = "puripuly-heart-native-smoke-$runId"
    $appId = "{{$($identity.ToString().ToUpperInvariant())}"
    $registration = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\{$identity}_is1"
    foreach ($path in @($state, $install)) {
        if (Test-Path -LiteralPath $path) { throw "Installer lifecycle refuses preexisting occupancy: $path" }
        if ($path.StartsWith("$repoRoot\", [StringComparison]::OrdinalIgnoreCase) -or $path -eq $repoRoot) {
            throw "Installer lifecycle must remain outside the checkout."
        }
    }
    if ($install -eq $state -or $install.StartsWith("$state\", [StringComparison]::OrdinalIgnoreCase) -or $state.StartsWith("$install\", [StringComparison]::OrdinalIgnoreCase)) {
        throw "Installer lifecycle install and state roots must be disjoint."
    }
    if (Test-Path -LiteralPath $registration) { throw "Installer lifecycle refuses an occupied AppId." }
    New-Item -ItemType Directory -Path $state | Out-Null
    $data = Join-Path $state "profile\localappdata\$child"
    $config = Join-Path $data "settings.json"
    $protectedRoot = Join-Path $state "profile\localappdata\puripuly-heart"
    New-Item -ItemType Directory -Path $protectedRoot -Force | Out-Null
    $sentinel = Join-Path $protectedRoot "synthetic-production.sentinel"
    [IO.File]::WriteAllText($sentinel, $runId)
    $smokeInstaller = Join-Path $state "PuriPulyHeart-Setup-$appVersion.exe"
    $uninstaller = Join-Path $install "unins000.exe"
    $defaultDirectoryName = "PuriPuly-native-smoke-$runId"
    $checkoutTarget = Join-Path $state "synthetic-checkout"
    $temporaryTarget = Join-Path $state "profile\unsafe-install"
    New-Item -ItemType Directory -Path $checkoutTarget | Out-Null
    [IO.File]::WriteAllText((Join-Path $checkoutTarget "pyproject.toml"), "[project]")
    $attemptedTargets = [Collections.Generic.List[string]]::new()
    $expectedHostHash = (Get-FileHash -LiteralPath (Join-Path $OutputDir "PuriPulyHeart.exe") -Algorithm SHA256).Hash
    $primaryFailure = $null
    $cleanupResults = @()
    $lifecyclePassed = $false
    try {
        Invoke-Checked -FilePath $iscc -ArgumentList @(
            "/DMyAppId=$appId", "/DMyAppDataDirName=$child",
            "/DMyAppGroupName=PuriPuly-native-smoke-$runId",
            "/DMyAppDirName=$defaultDirectoryName",
            "/DInstallerSmokeAppDataRoot=$($data.Replace('\', '/'))",
            "/DSkipLocalSttProvisioning=1", "/DMyPackagedAppDir=$OutputDir",
            "/DNativeCleanupInclude=$cleanupInclude", "/DLocalASRModelsInclude=$asrModelsInclude", "/O$state", (Join-Path $repoRoot "installer.iss")
        )
        $negativeCases = @(
            @{ name = "checkout"; directory = $checkoutTarget; marker = "Installer smoke build refused an install directory inside a repository checkout" },
            @{ name = "temporary"; directory = $temporaryTarget; marker = "Installer smoke build refused an install directory inside a temporary directory" },
            @{ name = "default"; directory = $null; marker = "Installer smoke build refused the production default install directory" }
        )
        foreach ($case in $negativeCases) {
            $rejectionLog = Join-Path $state "rejected-$($case.name).log"
            $arguments = @("/CURRENTUSER", "/VERYSILENT", "/SUPPRESSMSGBOXES", "/LOG=$rejectionLog")
            if ($case.directory) {
                $attemptedTargets.Add($case.directory)
                $arguments += "/DIR=$($case.directory)"
            }
            $rejection = Invoke-IsolatedProcess -FilePath $smokeInstaller `
                -ArgumentList $arguments -StateRoot $state -AllowFailure
            $rejectionText = Get-Content -LiteralPath $rejectionLog -Raw
            if ($rejection.ExitCode -eq 0 -or $rejectionText -notmatch [regex]::Escape($case.marker)) {
                throw "Installer did not reject the $($case.name) smoke target at its safety boundary."
            }
        }
        $manifest = Get-Content -LiteralPath (Join-Path $OutputDir "native-artifact-manifest.json") -Raw -Encoding utf8 | ConvertFrom-Json
        $repairPaths = @(
            "openvr_api.dll", "site-packages\soxr\soxr.dll", "app\puripuly_heart\data\fonts\NotoSansCJK-Medium.ttc",
            "third_party\soxr\COPYING.LGPL-2.1.txt"
        ) + @($manifest.inventory | Where-Object { $_.path -like "third_party/soxr/*.zip" } | ForEach-Object { $_.path })
        foreach ($phase in @("installed", "reinstalled")) {
            $log = Join-Path $state "$phase.log"
            if (-not $attemptedTargets.Contains($install)) { $attemptedTargets.Add($install) }
            [void](Invoke-IsolatedProcess -FilePath $smokeInstaller `
                -ArgumentList @("/CURRENTUSER", "/VERYSILENT", "/SUPPRESSMSGBOXES", "/DIR=$install", "/LOG=$log") -StateRoot $state)
            if (-not (Test-Path -LiteralPath $registration)) { throw "Installer did not register the isolated AppId." }
            $logText = Get-Content -LiteralPath $log -Raw
            if ($logText -notmatch [regex]::Escape("Local STT provisioning skipped for isolated installer smoke.") -or
                $logText -match "Local STT provisioning (failed|completed successfully)") {
                throw "Installer lifecycle did not preserve the controlled no-network provisioning boundary."
            }
            foreach ($record in $manifest.inventory) {
                $path = Join-Path $install $record.path
                if (-not (Test-Path -LiteralPath $path -PathType Leaf) -or
                    (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $record.sha256) {
                    throw "$phase native payload identity mismatch: $($record.path)"
                }
            }
            $settings = Get-Content -LiteralPath $config -Raw -Encoding utf8 | ConvertFrom-Json
            if ($phase -eq "installed") {
                if ($settings.intent.telemetry.enabled -ne $true) { throw "Fresh silent install did not persist telemetry ON." }
            } elseif ($settings.intent.telemetry.enabled -ne $false -or $null -ne $settings.state.telemetry.anonymous_id -or
                $settings.intent.ui.locale -ne "ja" -or [IO.File]::ReadAllText("$data\custom-prompt.md") -ne $runId -or
                [IO.File]::ReadAllText("$data\synthetic-keys.json") -ne $runId) {
                throw "Reinstall failed to preserve telemetry opt-out and unrelated settings/user content."
            }
            Assert-NativeInstalledRuntime -RuntimeRoot $install -StateRoot $state -Label $phase
            if ($phase -eq "installed") {
                $settings.intent.ui.locale = "ja"
                $settings | ConvertTo-Json -Depth 100 | Set-Content -LiteralPath $config -Encoding utf8NoBOM
                [IO.File]::WriteAllText("$data\custom-prompt.md", $runId)
                [IO.File]::WriteAllText("$data\synthetic-keys.json", $runId)
                [void](Invoke-Headless -RuntimeRoot $install -StateRoot $state -ArgumentList @("--config", $config, "installer-telemetry-preference", "disable"))
                $disabled = Get-Content -LiteralPath $config -Raw -Encoding utf8 | ConvertFrom-Json
                if ($disabled.intent.telemetry.enabled -ne $false -or $null -ne $disabled.state.telemetry.anonymous_id) {
                    throw "Telemetry OFF seed failed to clear identity."
                }
                foreach ($relative in $repairPaths) {
                    $path = Join-Path $install $relative
                    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Repair target missing: $relative" }
                    [IO.File]::WriteAllText($path, "synthetic replacement $runId")
                }
            }
        }
        [void](Invoke-IsolatedProcess -FilePath $uninstaller -ArgumentList @("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART") -StateRoot $state)
        $deadline = [DateTime]::UtcNow.AddSeconds(15)
        while ((Test-Path -LiteralPath $uninstaller) -and [DateTime]::UtcNow -lt $deadline) { Start-Sleep -Milliseconds 100 }
        if ((Test-Path -LiteralPath $install) -or (Test-Path -LiteralPath $data) -or (Test-Path -LiteralPath $registration)) {
            throw "Isolated uninstall left its install, data, or AppId namespace."
        }
        if ([IO.File]::ReadAllText($sentinel) -ne $runId) { throw "Isolated lifecycle changed the protected synthetic production namespace." }
        $lifecyclePassed = $true
    } catch {
        $primaryFailure = $_
    } finally {
        $cleanupResults = @(Remove-InstallerLifecycleInstallations -Identity $identity `
            -KnownTargets @($attemptedTargets) -DefaultDirectoryName $defaultDirectoryName `
            -ExpectedHostSha256 $expectedHostHash -StateRoot $state -WorkspaceRoot $repoRoot)
        $cleanupErrors = @($cleanupResults | Where-Object { $_.disposition -eq "failed" })
        @{
            passed = $lifecyclePassed -and $cleanupErrors.Count -eq 0; app_id = $identity.ToString();
            install_root = $install; data_root = $data; protected_namespace = $protectedRoot; state_root = $state;
            model_provisioning = "skipped"; gui_presentation = "not_observed";
            attempted_targets = @($attemptedTargets); cleanup = $cleanupResults;
            default_directory_name = $defaultDirectoryName; default_target_selector = "Inno DefaultDirName without /DIR";
            primary_failure = if ($primaryFailure) { $primaryFailure.Exception.Message } else { $null }
        } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $evidenceRoot "installer-lifecycle.json") -Encoding utf8
        Get-ChildItem -LiteralPath $state -Filter "*.log" -File | Copy-Item -Destination $evidenceRoot -Force
        if ($lifecyclePassed -and $cleanupErrors.Count -eq 0) {
            Remove-Item -LiteralPath $state -Recurse -Force
        } else {
            Write-Warning "Installer lifecycle failed; isolated state and logs retained at $state"
        }
    }
    if ($primaryFailure) { throw $primaryFailure }
    if ($cleanupErrors.Count -ne 0) { throw "Installer lifecycle owned cleanup failed; inspect installer-lifecycle.json." }
}

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$ToolPython = Resolve-RepoPath $ToolPython
$PinnedInputRoot = Resolve-RepoPath $PinnedInputRoot
$PythonEmbedArchive = Resolve-RepoPath $PythonEmbedArchive
$SoxrBuildEnvironment = Resolve-RepoPath $SoxrBuildEnvironment
$BuildRoot = Resolve-RepoPath $BuildRoot
$OutputDir = Resolve-RepoPath $OutputDir
$InstallerOutputDir = Resolve-RepoPath $InstallerOutputDir
if ([string]::IsNullOrWhiteSpace($InnoSetupCompiler)) {
    $iscc = @(
        (Join-Path $env:ProgramFiles "Inno Setup 7\ISCC.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Inno Setup 7\ISCC.exe")
    ) | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
} else {
    $iscc = Resolve-RepoPath $InnoSetupCompiler
}
if ([string]::IsNullOrWhiteSpace($iscc) -or -not (Test-Path -LiteralPath $iscc -PathType Leaf)) {
    throw "ISCC.exe not found: $iscc. Install Inno Setup $InnoSetupVersion or pass -InnoSetupCompiler."
}
$compilerProbe = @"
[Setup]
AppName=PuriPuly compiler identity
AppVersion=1
DefaultDirName={tmp}\PuriPulyCompilerIdentity
Uninstallable=no
Output=no
"@ | & $iscc "/O-" "-" | Out-String
$versionMatch = [regex]::Match($compilerProbe, 'Compiler engine version: Inno Setup (\d+\.\d+\.\d+)')
if ($LASTEXITCODE -ne 0 -or -not $versionMatch.Success) {
    throw "Could not read the Inno Setup compiler engine version."
}
$currentInnoVersion = $versionMatch.Groups[1].Value
if ($currentInnoVersion -ne $InnoSetupVersion) {
    throw "Inno Setup version mismatch: expected $InnoSetupVersion, found $currentInnoVersion"
}

$previousPythonPath = $env:PYTHONPATH
$previousDontWriteBytecode = $env:PYTHONDONTWRITEBYTECODE
$smokeRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("PuriPuly native release " + [Guid]::NewGuid().ToString("N"))
Push-Location $repoRoot
try {
    $env:PYTHONPATH = Join-Path $repoRoot "src"
    $env:PYTHONDONTWRITEBYTECODE = "1"
    $appVersion = (& $ToolPython (Join-Path $PSScriptRoot "read-project-version.py") | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($appVersion)) {
        throw "Could not read the project version."
    }
    $installerPath = Join-Path $InstallerOutputDir "PuriPulyHeart-Setup-$appVersion.exe"
    $cleanupInclude = Join-Path (Split-Path -Parent $OutputDir) "native-installer-cleanup.iss"
    $asrModelsInclude = Join-Path (Split-Path -Parent $OutputDir) "native-asr-models.iss"
    $evidenceRoot = Join-Path $BuildRoot "evidence"
    if (-not $SkipNativeBuild) {
        Invoke-Checked -FilePath (Join-Path $PSHOME "pwsh.exe") -ArgumentList @(
            "-NoProfile", "-File", (Join-Path $PSScriptRoot "build-native-experimental.ps1"),
            "-ToolPython", $ToolPython,
            "-PinnedInputRoot", $PinnedInputRoot,
            "-PythonEmbedArchive", $PythonEmbedArchive,
            "-SoxrBuildEnvironment", $SoxrBuildEnvironment,
            "-BuildRoot", $BuildRoot,
            "-OutputDir", $OutputDir
        )
    }
    $nativeModule = "puripuly_heart.release_evidence.native_distribution"
    Invoke-Checked -FilePath $ToolPython -ArgumentList @(
        "-m", $nativeModule, "validate-target",
        "--target-root", $OutputDir,
        "--layout", (Join-Path $repoRoot "native\windows_host\artifact-layout.json"),
        "--requirements", (Join-Path $BuildRoot "requirements-export.txt"),
        "--vc-runtime", (Join-Path $evidenceRoot "vc-runtime.json"),
        "--output", (Join-Path $evidenceRoot "release-target-validation.json")
    )
    Invoke-Checked -FilePath $ToolPython -ArgumentList @(
        "-m", $nativeModule, "validate-compliance",
        "--target-root", $OutputDir, "--repo-root", $repoRoot,
        "--soxr-manifest", (Join-Path $BuildRoot "soxr-release-inputs\manifest.json"),
        "--output", (Join-Path $evidenceRoot "release-compliance-validation.json")
    )
    $metadataCheck = "import sys; from pathlib import Path; from puripuly_heart.release_evidence.release_identity import verify_pe_product_metadata; [verify_pe_product_metadata(Path(p), expected_version=sys.argv[1]) for p in sys.argv[2:]]"
    $hostExe = Join-Path $OutputDir "PuriPulyHeart.exe"
    Invoke-Checked -FilePath $ToolPython -ArgumentList @(
        "-c", $metadataCheck, $appVersion, $hostExe,
        (Join-Path $OutputDir "PuriPulyHeartOverlay.exe"),
        (Join-Path $OutputDir "PuriPulyHeartASRWorker.exe")
    )
    New-Item -ItemType Directory -Path $smokeRoot | Out-Null
    Assert-NativeInstalledRuntime -RuntimeRoot $OutputDir -StateRoot $smokeRoot -Label "packaged"
    $config = Join-Path $smokeRoot "설정 with spaces.json"
    [void](Invoke-Headless -ArgumentList @("--config", $config, "installer-telemetry-preference", "disable"))
    $settings = Get-Content -LiteralPath $config -Raw -Encoding utf8 | ConvertFrom-Json
    if ($settings.intent.telemetry.enabled -ne $false -or $null -ne $settings.state.telemetry.anonymous_id) {
        throw "Native installer telemetry smoke did not persist the canonical OFF invariant."
    }
    Invoke-Checked -FilePath $ToolPython -ArgumentList @(
        "-m", "puripuly_heart.release_evidence.native_installer_cleanup",
        "--legacy-manifest", (Join-Path $repoRoot "native\windows_host\legacy-pyinstaller-v2.7.0.json"),
        "--native-manifest", (Join-Path $OutputDir "native-artifact-manifest.json"),
        "--output", $cleanupInclude
    )
    Invoke-Checked -FilePath $ToolPython -ArgumentList @(
        "-m", "puripuly_heart.release_evidence.local_asr_installer",
        "--output", $asrModelsInclude
    )
    New-Item -ItemType Directory -Path $InstallerOutputDir -Force | Out-Null
    Invoke-Checked -FilePath $iscc -ArgumentList @(
        "/DMyPackagedAppDir=$OutputDir",
        "/DMyStagedOverlayDir=$OutputDir",
        "/DNativeCleanupInclude=$cleanupInclude",
        "/DLocalASRModelsInclude=$asrModelsInclude",
        "/O$InstallerOutputDir",
        (Join-Path $repoRoot "installer.iss")
    )
    Invoke-Checked -FilePath $ToolPython -ArgumentList @("-c", $metadataCheck, $appVersion, $installerPath)
    $hash = (Get-FileHash -LiteralPath $installerPath -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-Host "Native installer: $installerPath"
    Write-Host "SHA256: $hash"
    Write-Host "Build evidence: $evidenceRoot"
    if ($VerifyInstallerLifecycle) { Invoke-InstallerLifecycle }
} finally {
    Pop-Location
    if ($null -eq $previousPythonPath) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    } else {
        $env:PYTHONPATH = $previousPythonPath
    }
    if ($null -eq $previousDontWriteBytecode) {
        Remove-Item Env:PYTHONDONTWRITEBYTECODE -ErrorAction SilentlyContinue
    } else {
        $env:PYTHONDONTWRITEBYTECODE = $previousDontWriteBytecode
    }
    if (Test-Path -LiteralPath $smokeRoot) {
        Remove-Item -LiteralPath $smokeRoot -Recurse -Force
    }
}
