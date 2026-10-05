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
    [switch]$SkipNativeBuild
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
    & $FilePath @ArgumentList
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($ArgumentList -join ' ')"
    }
}

function Resolve-RepoPath {
    param([Parameter(Mandatory = $true)][string]$Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $Path))
}

function Invoke-Headless {
    param([Parameter(Mandatory = $true)][string[]]$ArgumentList)
    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $hostExe
    $startInfo.WorkingDirectory = $smokeRoot
    $startInfo.UseShellExecute = $false
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.StandardOutputEncoding = [System.Text.Encoding]::UTF8
    $startInfo.StandardErrorEncoding = [System.Text.Encoding]::UTF8
    $startInfo.ArgumentList.Add("--headless")
    foreach ($argument in $ArgumentList) {
        $startInfo.ArgumentList.Add($argument)
    }
    $process = [System.Diagnostics.Process]::Start($startInfo)
    try {
        $stdout = $process.StandardOutput.ReadToEndAsync()
        $stderr = $process.StandardError.ReadToEndAsync()
        $process.WaitForExit()
        $text = $stdout.GetAwaiter().GetResult()
        $errors = $stderr.GetAwaiter().GetResult()
        if ($process.ExitCode -ne 0) {
            throw "Native headless smoke failed ($($ArgumentList -join ' ')): exit $($process.ExitCode)`n$text`n$errors"
        }
        if ($errors) { Write-Host $errors }
        return $text.Trim()
    } finally {
        $process.Dispose()
    }
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
        (Join-Path $OutputDir "PuriPulyHeartGpuWorker.exe")
    )
    New-Item -ItemType Directory -Path $smokeRoot | Out-Null
    $version = Invoke-Headless -ArgumentList @("--version")
    if ($version -ne $appVersion) {
        throw "Native runtime version mismatch: expected $appVersion, found $version"
    }
    foreach ($command in @("gui-startup-check", "local-qwen-runtime-check", "soxr-runtime-check", "hf-xet-runtime-check")) {
        Write-Host "Checking native $command..."
        Write-Host (Invoke-Headless -ArgumentList @($command))
    }
    $config = Join-Path $smokeRoot "설정 with spaces.json"
    [void](Invoke-Headless -ArgumentList @("--config", $config, "installer-telemetry-preference", "disable"))
    $settings = Get-Content -LiteralPath $config -Raw -Encoding utf8 | ConvertFrom-Json
    if ($settings.intent.telemetry.enabled -ne $false -or $null -ne $settings.state.telemetry.anonymous_id) {
        throw "Native installer telemetry smoke did not persist the canonical OFF invariant."
    }
    Invoke-Checked -FilePath $ToolPython -ArgumentList @(
        "-m", "puripuly_heart.release_evidence.managed_gemma_distribution", "verify-package", $OutputDir, "--launch"
    )
    Invoke-Checked -FilePath $ToolPython -ArgumentList @(
        "-m", "puripuly_heart.release_evidence.native_installer_cleanup",
        "--legacy-manifest", (Join-Path $repoRoot "native\windows_host\legacy-pyinstaller-v2.7.0.json"),
        "--native-manifest", (Join-Path $OutputDir "native-artifact-manifest.json"),
        "--output", $cleanupInclude
    )
    New-Item -ItemType Directory -Path $InstallerOutputDir -Force | Out-Null
    Invoke-Checked -FilePath $iscc -ArgumentList @(
        "/DNativeExperimental=1",
        "/DMyPackagedAppDir=$OutputDir",
        "/DMyStagedOverlayDir=$OutputDir",
        "/DNativeCleanupInclude=$cleanupInclude",
        "/O$InstallerOutputDir",
        (Join-Path $repoRoot "installer.iss")
    )
    Invoke-Checked -FilePath $ToolPython -ArgumentList @("-c", $metadataCheck, $appVersion, $installerPath)
    $hash = (Get-FileHash -LiteralPath $installerPath -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  $([System.IO.Path]::GetFileName($installerPath))" | Set-Content -LiteralPath "$installerPath.sha256" -Encoding ascii
    Copy-Item -LiteralPath (Join-Path $BuildRoot "soxr-release-inputs\PuriPulyHeart-soxr-third-party-source-bundle.zip") -Destination $InstallerOutputDir -Force
    Write-Host "Native installer: $installerPath"
    Write-Host "SHA256: $installerPath.sha256"
    Write-Host "Build evidence: $evidenceRoot"
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
