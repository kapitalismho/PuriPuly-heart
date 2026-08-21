[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$AppVersion,

    [Parameter(Mandatory = $true)]
    [string]$InnoSetupVersion
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$PinnedOpenVrVendorDllSha256 = "bab8ac6ef64e68a9ca53315b0014d131088584b2efdfa6db511d67ec03cfcb4a"
$PinnedNotoCjkFontSha256 = "197d5e1e019faca33a4d55931c7d68b8056f3b97cb862049f5cb8de9efdfb8ce"
$PrepareFletRuntimeScript = Join-Path $PSScriptRoot "prepare-flet-runtime.ps1"
$ManagedGemmaDistributionModule = "puripuly_heart.release_evidence.managed_gemma_distribution"

function Invoke-External {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [Parameter()]
        [string[]]$ArgumentList = @()
    )

    & $FilePath @ArgumentList
    $exitCode = Get-Variable -Name LASTEXITCODE -ValueOnly -ErrorAction SilentlyContinue
    if ($null -eq $exitCode) {
        $exitCode = 0
    }
    if ($exitCode -ne 0) {
        throw "Command failed with exit code ${exitCode}: $FilePath $($ArgumentList -join ' ')"
    }
}

function Invoke-ExternalProcess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [Parameter()]
        [string[]]$ArgumentList = @(),

        [Parameter()]
        [string]$WorkingDirectory = $PWD
    )

    $process = Start-Process -FilePath $FilePath -ArgumentList $ArgumentList -WorkingDirectory $WorkingDirectory -Wait -PassThru
    if ($process.ExitCode -ne 0) {
        throw "Command failed with exit code $($process.ExitCode): $FilePath $($ArgumentList -join ' ')"
    }
}

function Get-InnoSetupVersion {
    foreach ($registryPath in @(
        "HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1",
        "HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall\Inno Setup 6_is1"
    )) {
        if (Test-Path $registryPath) {
            return (Get-ItemProperty $registryPath).DisplayVersion
        }
    }
    return $null
}

function Resolve-ProjectEnvironmentScriptsPath {
    $projectEnvironment = $env:UV_PROJECT_ENVIRONMENT
    if ([string]::IsNullOrWhiteSpace($projectEnvironment)) {
        $projectEnvironment = ".venv"
    }
    if (-not [System.IO.Path]::IsPathRooted($projectEnvironment)) {
        $projectEnvironment = Join-Path $PWD $projectEnvironment
    }
    return Join-Path $projectEnvironment "Scripts"
}

function Resolve-ProjectEnvironmentPath {
    $scriptsPath = Resolve-ProjectEnvironmentScriptsPath
    return Split-Path -Parent $scriptsPath
}

function Resolve-CommandPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter()]
        [string[]]$Fallbacks = @()
    )

    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }

    foreach ($candidate in $Fallbacks) {
        if ([string]::IsNullOrWhiteSpace($candidate)) {
            continue
        }
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    throw "Required command not found on PATH: $Name"
}

function Get-CanonicalPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw "Canonical path cannot be resolved from a blank path value."
    }

    return (Get-Item -LiteralPath $Path).FullName
}

function Resolve-PackagedLocalQwenRuntimeDir {
    param(
        [Parameter(Mandatory = $true)]
        [string]$DistDir
    )

    $packagedLocalQwenRuntimeDir = Join-Path $DistDir "_runtime\local_qwen"
    $packagedLocalQwenFallbackRuntimeDir = Join-Path $DistDir "_internal\_runtime\local_qwen"
    if (Test-Path $packagedLocalQwenRuntimeDir) {
        return $packagedLocalQwenRuntimeDir
    }

    return $packagedLocalQwenFallbackRuntimeDir
}

function Get-FileSha256 {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path $Path)) {
        throw "File not found for SHA256 calculation: $Path"
    }

    return (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-PinnedSha256FromFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter()]
        [string]$ExpectedFileName = "openvr_api.dll"
    )

    if (-not (Test-Path $Path)) {
        throw "Pinned SHA256 file not found: $Path"
    }

    $sha256Line = (Get-Content -Path $Path -Raw -Encoding utf8).Trim()
    $sha256Match = [regex]::Match($sha256Line, '^(?<sha>[0-9A-Fa-f]{64})\s+\*(?<name>.+)$')
    if (-not $sha256Match.Success) {
        throw "Pinned SHA256 file must be a single sha256sum line for ${ExpectedFileName}: $Path"
    }

    $resolvedFileName = $sha256Match.Groups["name"].Value
    if ($resolvedFileName -ne $ExpectedFileName) {
        throw "Pinned SHA256 file must target $ExpectedFileName; found $resolvedFileName"
    }

    return $sha256Match.Groups["sha"].Value.ToLowerInvariant()
}

function Assert-FileSha256Equals {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$ExpectedSha256,

        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    if (-not (Test-Path $Path)) {
        throw "$Label not found: $Path"
    }

    $normalizedExpectedSha256 = $ExpectedSha256.Trim().ToLowerInvariant()
    $actualSha256 = Get-FileSha256 -Path $Path
    if ($actualSha256 -ne $normalizedExpectedSha256) {
        throw "$Label sha256 mismatch: expected $normalizedExpectedSha256, found $actualSha256"
    }
}

function Assert-SoxrRuntimeReport {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ReportPath,

        [Parameter(Mandatory = $true)]
        [string]$ExpectedExtensionPath,

        [Parameter(Mandatory = $true)]
        [string]$ExpectedSoxrDllPath,

        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    if (-not (Test-Path $ReportPath)) {
        throw "$Label soxr runtime report not found: $ReportPath"
    }

    $soxrRuntimeReport = Get-Content -Path $ReportPath -Raw -Encoding utf8 | ConvertFrom-Json
    $reportedExpectedExtensionPath = [string]$soxrRuntimeReport.expected_extension_path
    $reportedExpectedSoxrDllPath = [string]$soxrRuntimeReport.expected_sibling_dll_path
    $reportedImportedExtensionPath = [string]$soxrRuntimeReport.imported_extension_path
    $reportedLoadedSoxrDllPath = [string]$soxrRuntimeReport.loaded_sibling_dll_path

    $canonicalExpectedExtensionPath = Get-CanonicalPath -Path $ExpectedExtensionPath
    $canonicalExpectedSoxrDllPath = Get-CanonicalPath -Path $ExpectedSoxrDllPath
    $canonicalReportedExpectedExtensionPath = Get-CanonicalPath -Path $reportedExpectedExtensionPath
    $canonicalReportedExpectedSoxrDllPath = Get-CanonicalPath -Path $reportedExpectedSoxrDllPath
    $canonicalReportedImportedExtensionPath = Get-CanonicalPath -Path $reportedImportedExtensionPath
    $canonicalReportedLoadedSoxrDllPath = Get-CanonicalPath -Path $reportedLoadedSoxrDllPath

    if ($canonicalReportedExpectedExtensionPath -ne $canonicalExpectedExtensionPath) {
        throw "$Label soxr runtime report expected_extension_path mismatch: $reportedExpectedExtensionPath"
    }
    if ($canonicalReportedExpectedSoxrDllPath -ne $canonicalExpectedSoxrDllPath) {
        throw "$Label soxr runtime report expected_sibling_dll_path mismatch: $reportedExpectedSoxrDllPath"
    }
    if ($canonicalReportedImportedExtensionPath -ne $canonicalExpectedExtensionPath) {
        throw "$Label soxr runtime report imported_extension_path mismatch: $reportedImportedExtensionPath"
    }
    if ($canonicalReportedLoadedSoxrDllPath -ne $canonicalExpectedSoxrDllPath) {
        throw "$Label soxr runtime report loaded_sibling_dll_path mismatch: $reportedLoadedSoxrDllPath"
    }
}

function Invoke-SoxrRuntimeSmokeCheck {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ExePath,

        [Parameter(Mandatory = $true)]
        [string]$ReportPath,

        [Parameter(Mandatory = $true)]
        [string]$ExpectedExtensionPath,

        [Parameter(Mandatory = $true)]
        [string]$ExpectedSoxrDllPath,

        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    $previousSoxrRuntimeReportPath = $env:PURIPULY_HEART_SOXR_RUNTIME_REPORT_PATH
    $env:PURIPULY_HEART_SOXR_RUNTIME_REPORT_PATH = $ReportPath
    try {
        $soxrRuntimeSmokeTest = Start-Process -FilePath $ExePath -ArgumentList @("soxr-runtime-check") -Wait -PassThru
    } finally {
        if ($null -eq $previousSoxrRuntimeReportPath) {
            Remove-Item Env:PURIPULY_HEART_SOXR_RUNTIME_REPORT_PATH -ErrorAction SilentlyContinue
        } else {
            $env:PURIPULY_HEART_SOXR_RUNTIME_REPORT_PATH = $previousSoxrRuntimeReportPath
        }
    }

    if ($soxrRuntimeSmokeTest.ExitCode -ne 0) {
        throw "$Label soxr runtime smoke test failed with exit code $($soxrRuntimeSmokeTest.ExitCode)"
    }

    Assert-SoxrRuntimeReport -ReportPath $ReportPath -ExpectedExtensionPath $ExpectedExtensionPath -ExpectedSoxrDllPath $ExpectedSoxrDllPath -Label $Label
}

function Invoke-GuiStartupSmokeCheck {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ExePath,

        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    $smokeTest = Start-Process -FilePath $ExePath -ArgumentList @("gui-startup-check") -Wait -PassThru
    if ($smokeTest.ExitCode -ne 0) {
        throw "$Label GUI startup smoke test failed with exit code $($smokeTest.ExitCode)"
    }
}

function Invoke-ProcessCaptureRuntimeSmokeCheck {
    param(
        [Parameter(Mandatory = $true)]
        [string]$HelperExePath,

        [Parameter(Mandatory = $true)]
        [string]$ArtifactRoot,

        [Parameter(Mandatory = $true)]
        [string]$ReportPath,

        [Parameter(Mandatory = $true)]
        [string]$Label
    )

    $smokeTest = Start-Process -FilePath $HelperExePath -ArgumentList @(
        "--artifact-root",
        $ArtifactRoot,
        "--report",
        $ReportPath
    ) -Wait -PassThru

    if ($smokeTest.ExitCode -ne 0) {
        throw "$Label process-capture runtime smoke test failed with exit code $($smokeTest.ExitCode)"
    }
    if (-not (Test-Path $ReportPath)) {
        throw "$Label process-capture runtime report not found: $ReportPath"
    }
    $report = Get-Content -Path $ReportPath -Raw -Encoding utf8 | ConvertFrom-Json
    if ($report.status -ne "passed" -or
        $report.proctap_version -ne "1.1.1" -or
        $report.native_process_specific -ne $true -or
        $report.capture_started -ne $true -or
        $report.release_only_helper -ne $true -or
        $report.device_fallback_used -ne $false -or
        $report.credentials_used -ne $false -or
        $report.network_used -ne $false) {
        throw "$Label process-capture runtime report failed strict validation"
    }
    if (-not (Test-Path ([string]$report.artifact_native_module))) {
        throw "$Label packaged ProcTap native module not found: $($report.artifact_native_module)"
    }
    $nativeHash = Get-FileSha256 -Path ([string]$report.artifact_native_module)
    if ($nativeHash -ne ([string]$report.artifact_native_sha256)) {
        throw "$Label packaged ProcTap native module hash mismatch"
    }
    $helperHash = Get-FileSha256 -Path $HelperExePath
    if ($helperHash -ne ([string]$report.helper_executable_sha256)) {
        throw "$Label release-only process-capture smoke helper hash mismatch"
    }
}

$projectEnvironmentScripts = Resolve-ProjectEnvironmentScriptsPath
$projectEnvironmentPath = Resolve-ProjectEnvironmentPath
if (Test-Path $projectEnvironmentScripts) {
    $env:PATH = "$projectEnvironmentScripts;$env:PATH"
}
if ([string]::IsNullOrWhiteSpace($env:LIBCLANG_PATH)) {
    $bundledLibclangPath = Join-Path $projectEnvironmentPath "Lib\site-packages\clang\native"
    if (Test-Path $bundledLibclangPath) {
        $env:LIBCLANG_PATH = $bundledLibclangPath
    }
}

$cargoCommand = Resolve-CommandPath -Name "cargo" -Fallbacks @(
    (Join-Path $env:USERPROFILE ".cargo\bin\cargo.exe")
)
$cmakeCommand = Resolve-CommandPath -Name "cmake" -Fallbacks @(
    (Join-Path $projectEnvironmentScripts "cmake.exe"),
    "C:\Program Files\CMake\bin\cmake.exe"
)
$pythonCommand = Resolve-CommandPath -Name "python" -Fallbacks @(
    (Join-Path $projectEnvironmentScripts "python.exe")
)
$cmakeCommandDirectory = Split-Path -Parent $cmakeCommand
if (-not [string]::IsNullOrWhiteSpace($cmakeCommandDirectory)) {
    $env:PATH = "$cmakeCommandDirectory;$env:PATH"
}
$env:CMAKE = $cmakeCommand

$overlayManifestPath = Join-Path $PWD "native/overlay/Cargo.toml"
$gpuWorkerManifestPath = Join-Path $PWD "native/gpu_worker/Cargo.toml"
$releaseBuildRoot = $env:PURIPULY_HEART_RELEASE_BUILD_ROOT
if ([string]::IsNullOrWhiteSpace($releaseBuildRoot)) {
    $releaseBuildRoot = Join-Path $env:TEMP "PuriPulyHeart-ReleaseBuild-$AppVersion"
}
$overlayTargetDir = Join-Path $releaseBuildRoot "overlay-target"
$overlayBuildDir = Join-Path $PWD "build/overlay"
$overlayReleasePath = Join-Path $overlayTargetDir "release/PuriPulyHeartOverlay.exe"
$overlayStagedPath = Join-Path $overlayBuildDir "PuriPulyHeartOverlay.exe"
$overlayBundledDllPath = Join-Path $overlayBuildDir "openvr_api.dll"
$gpuWorkerTargetDir = Join-Path $releaseBuildRoot "gpu-worker-target"
$gpuWorkerBuildDir = Join-Path $PWD "build/gpu_worker"
$gpuWorkerReleasePath = Join-Path $gpuWorkerTargetDir "release/PuriPulyHeartGpuWorker.exe"
$gpuWorkerStagedPath = Join-Path $gpuWorkerBuildDir "PuriPulyHeartGpuWorker.exe"
$openVrVendorDllPath = Join-Path $PWD "third_party/openvr/win64/openvr_api.dll"
$openVrVendorSha256Path = Join-Path $PWD "third_party/openvr/win64/openvr_api.dll.sha256"
$notoCjkFontSourcePath = Join-Path $PWD "src/puripuly_heart/data/fonts/NotoSansCJK-Medium.ttc"
$notoCjkFontLicensePath = Join-Path $PWD "third_party/noto-sans-cjk/OFL.txt"
$notoCjkFontReadmePath = Join-Path $PWD "third_party/noto-sans-cjk/README.md"
$notoCjkFontSha256SumsPath = Join-Path $PWD "third_party/noto-sans-cjk/SHA256SUMS.txt"
$pyInstallerBuildDir = Join-Path $PWD "build/build"
$distDir = Join-Path $PWD "dist/PuriPulyHeart"
$soxrLicenseTextPath = Join-Path $PWD "src\puripuly_heart\data\licenses\COPYING.LGPL-2.1.txt"
$soxrReleaseInputsManifestPath = Join-Path $PWD "build/soxr-release-inputs/manifest.json"
$packagedSoxrRuntimeDir = Join-Path $distDir "soxr"
$soxrRuntimeReportDir = Join-Path $PWD "build/soxr-runtime-smoke"
$packagedSoxrDllPath = Join-Path $packagedSoxrRuntimeDir "soxr.dll"
$packagedSoxrRuntimeReportPath = Join-Path $soxrRuntimeReportDir "packaged-soxr-runtime-check.json"
$installedSoxrRuntimeReportPath = Join-Path $soxrRuntimeReportDir "installed-soxr-runtime-check.json"
$reinstalledSoxrRuntimeReportPath = Join-Path $soxrRuntimeReportDir "reinstalled-soxr-runtime-check.json"
$processCaptureRuntimeReportDir = Join-Path $PWD "build/process-capture-runtime-smoke"
$packagedProcessCaptureRuntimeReportPath = Join-Path $processCaptureRuntimeReportDir "packaged-process-capture-runtime-check.json"
$installedProcessCaptureRuntimeReportPath = Join-Path $processCaptureRuntimeReportDir "installed-process-capture-runtime-check.json"
$reinstalledProcessCaptureRuntimeReportPath = Join-Path $processCaptureRuntimeReportDir "reinstalled-process-capture-runtime-check.json"
$processCaptureSmokeBuildRoot = Join-Path $releaseBuildRoot "process-capture-smoke"
$processCaptureSmokeDistDir = Join-Path $processCaptureSmokeBuildRoot "dist"
$processCaptureSmokeWorkDir = Join-Path $processCaptureSmokeBuildRoot "work"
$processCaptureSmokeArtifactRoot = Join-Path $processCaptureSmokeDistDir "PuriPulyHeartProcessCaptureSmoke"
$processCaptureSmokeHelperPath = Join-Path $processCaptureSmokeArtifactRoot "PuriPulyHeartProcessCaptureSmoke.exe"
$packagedSoxrComplianceDir = Join-Path $distDir "third_party\soxr"
$packagedSoxrLicensePath = Join-Path $packagedSoxrComplianceDir "COPYING.LGPL-2.1.txt"
$packagedSoxrSourceBundlePath = $null
$packagedOverlayDllPath = Join-Path $distDir "openvr_api.dll"
$packagedNotoCjkFontPath = Join-Path $distDir "puripuly_heart\data\fonts\NotoSansCJK-Medium.ttc"
$packagedNotoCjkProvenanceDir = Join-Path $distDir "third_party\noto-sans-cjk"
$packagedNotoCjkLicensePath = Join-Path $packagedNotoCjkProvenanceDir "OFL.txt"
$packagedNotoCjkReadmePath = Join-Path $packagedNotoCjkProvenanceDir "README.md"
$packagedNotoCjkSha256SumsPath = Join-Path $packagedNotoCjkProvenanceDir "SHA256SUMS.txt"
$packagedTranslationExamplesDir = Join-Path $distDir "examples\http_extensions"
$packagedMyMemoryExamplePath = Join-Path $packagedTranslationExamplesDir "mymemory.json"
$pinnedOpenVrVendorDllSha256FromFile = Get-PinnedSha256FromFile -Path $openVrVendorSha256Path
if ($pinnedOpenVrVendorDllSha256FromFile -ne $PinnedOpenVrVendorDllSha256) {
    throw "Vendored OpenVR runtime DLL pinned SHA256 literal drifted from $openVrVendorSha256Path"
}
Assert-FileSha256Equals -Path $openVrVendorDllPath -ExpectedSha256 $PinnedOpenVrVendorDllSha256 -Label "Vendored OpenVR runtime DLL"
Assert-FileSha256Equals -Path $notoCjkFontSourcePath -ExpectedSha256 $PinnedNotoCjkFontSha256 -Label "Source Noto Sans CJK Medium TTC"
foreach ($notoCjkProvenancePath in @($notoCjkFontLicensePath, $notoCjkFontReadmePath, $notoCjkFontSha256SumsPath)) {
    if (-not (Test-Path $notoCjkProvenancePath)) {
        throw "Noto Sans CJK provenance file not found: $notoCjkProvenancePath"
    }
}

Write-Host "Building Rust overlay executable..."
Invoke-External -FilePath $cargoCommand -ArgumentList @(
    "build",
    "--manifest-path",
    $overlayManifestPath,
    "--locked",
    "--release",
    "--bin",
    "PuriPulyHeartOverlay",
    "--target-dir",
    $overlayTargetDir
)

if (-not (Test-Path $overlayReleasePath)) {
    throw "Rust overlay executable not found: $overlayReleasePath"
}
$overlayVersion = (& $overlayReleasePath --version | Out-String).Trim()
if ($overlayVersion -ne $AppVersion) {
    throw "Rust overlay version mismatch: expected $AppVersion, found $overlayVersion"
}
$overlayReleaseSha256 = Get-FileSha256 -Path $overlayReleasePath
Write-Host "Built Rust overlay version=$overlayVersion sha256=$overlayReleaseSha256"

New-Item -ItemType Directory -Force -Path $overlayBuildDir | Out-Null
Copy-Item -Path $overlayReleasePath -Destination $overlayStagedPath -Force
Copy-Item -Path $openVrVendorDllPath -Destination $overlayBundledDllPath -Force

if (-not (Test-Path $overlayStagedPath)) {
    throw "Staged overlay executable not found: $overlayStagedPath"
}
if (-not (Test-Path $overlayBundledDllPath)) {
    throw "Staged OpenVR runtime DLL not found: $overlayBundledDllPath"
}
Assert-FileSha256Equals -Path $overlayBundledDllPath -ExpectedSha256 $PinnedOpenVrVendorDllSha256 -Label "Staged OpenVR runtime DLL"

Write-Host "Building Rust GPU worker executable..."
Invoke-External -FilePath $cargoCommand -ArgumentList @(
    "build",
    "--manifest-path",
    $gpuWorkerManifestPath,
    "--locked",
    "--release",
    "--bin",
    "PuriPulyHeartGpuWorker",
    "--target-dir",
    $gpuWorkerTargetDir
)

if (-not (Test-Path $gpuWorkerReleasePath)) {
    throw "Rust GPU worker executable not found: $gpuWorkerReleasePath"
}
$gpuWorkerVersion = (& $gpuWorkerReleasePath --version | Out-String).Trim()
if ($gpuWorkerVersion -ne $AppVersion) {
    throw "Rust GPU worker version mismatch: expected $AppVersion, found $gpuWorkerVersion"
}
$gpuWorkerReleaseSha256 = Get-FileSha256 -Path $gpuWorkerReleasePath
Write-Host "Built Rust GPU worker version=$gpuWorkerVersion sha256=$gpuWorkerReleaseSha256"

New-Item -ItemType Directory -Force -Path $gpuWorkerBuildDir | Out-Null
Copy-Item -Path $gpuWorkerReleasePath -Destination $gpuWorkerStagedPath -Force
if (-not (Test-Path $gpuWorkerStagedPath)) {
    throw "Staged GPU worker executable not found: $gpuWorkerStagedPath"
}

Write-Host "Smoke-testing staged overlay executable..."
Invoke-External -FilePath $overlayStagedPath -ArgumentList @("--check-startup-contract")

Write-Host "Cleaning previous PyInstaller outputs..."
Remove-Item -Recurse -Force $pyInstallerBuildDir -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force $distDir -ErrorAction SilentlyContinue

Write-Host "Preparing pinned Flet desktop runtime..."
Invoke-External -FilePath "pwsh" -ArgumentList @("-File", $PrepareFletRuntimeScript)

Write-Host "Preparing pinned llama.cpp runtime..."
Invoke-External -FilePath $pythonCommand -ArgumentList @(
    "-m",
    $ManagedGemmaDistributionModule,
    "prepare"
)

Write-Host "Building Windows executable..."
Invoke-ExternalProcess -FilePath $pythonCommand -ArgumentList @(
    "-m",
    "PyInstaller",
    "--clean",
    "--noconfirm",
    "build.spec"
)

Write-Host "Building release-only process-capture smoke helper..."
$previousProcessCaptureSmokeBuild = $env:PURIPULY_HEART_RELEASE_PROCESS_CAPTURE_SMOKE
$env:PURIPULY_HEART_RELEASE_PROCESS_CAPTURE_SMOKE = "1"
try {
    Invoke-ExternalProcess -FilePath $pythonCommand -ArgumentList @(
        "-m",
        "PyInstaller",
        "--clean",
        "--noconfirm",
        "--distpath",
        $processCaptureSmokeDistDir,
        "--workpath",
        $processCaptureSmokeWorkDir,
        "build.spec"
    )
} finally {
    if ($null -eq $previousProcessCaptureSmokeBuild) {
        Remove-Item Env:PURIPULY_HEART_RELEASE_PROCESS_CAPTURE_SMOKE -ErrorAction SilentlyContinue
    } else {
        $env:PURIPULY_HEART_RELEASE_PROCESS_CAPTURE_SMOKE = $previousProcessCaptureSmokeBuild
    }
}
if (-not (Test-Path $processCaptureSmokeHelperPath)) {
    throw "Release-only process-capture smoke helper not found: $processCaptureSmokeHelperPath"
}

Invoke-External -FilePath $pythonCommand -ArgumentList @(
    "-m",
    $ManagedGemmaDistributionModule,
    "verify-installer",
    "installer.iss"
)
Invoke-External -FilePath $pythonCommand -ArgumentList @(
    "-m",
    $ManagedGemmaDistributionModule,
    "verify-package",
    $distDir,
    "--launch"
)

$exePath = Join-Path $PWD "dist/PuriPulyHeart/PuriPulyHeart.exe"
if (-not (Test-Path $exePath)) {
    throw "Packaged executable not found: $exePath"
}
$packagedMyMemoryExamplePath = [System.IO.Path]::GetFullPath($packagedMyMemoryExamplePath)
if (-not (Test-Path -LiteralPath $packagedMyMemoryExamplePath -PathType Leaf)) {
    throw "Packaged MyMemory example not found: $packagedMyMemoryExamplePath"
}
$productionPackagedSmokeHelperPath = Join-Path $distDir "PuriPulyHeartProcessCaptureSmoke.exe"
if (Test-Path $productionPackagedSmokeHelperPath) {
    throw "Production packaged application must omit the release-only process-capture smoke helper"
}

$packagedLocalQwenRuntimeDir = Resolve-PackagedLocalQwenRuntimeDir -DistDir $distDir
$packagedOnnxRuntimeDllPath = Join-Path $packagedLocalQwenRuntimeDir "onnxruntime.dll"
$packagedOnnxRuntimeProvidersSharedDllPath = Join-Path $packagedLocalQwenRuntimeDir "onnxruntime_providers_shared.dll"

$numpyCoreExtensions = @(Get-ChildItem -Path $distDir -Filter "_multiarray_umath*.pyd" -Recurse -File -ErrorAction SilentlyContinue)
if ($numpyCoreExtensions.Count -eq 0) {
    throw "Packaged executable is missing numpy._core._multiarray_umath in $distDir"
}
$packagedProcTapNativeExtensions = @(Get-ChildItem -Path (Join-Path $distDir "proctap") -Filter "_native*.pyd" -File -ErrorAction SilentlyContinue)
if ($packagedProcTapNativeExtensions.Count -ne 1) {
    throw "Packaged ProcTap runtime must contain exactly one proctap/_native extension; found $($packagedProcTapNativeExtensions.Count)"
}
if (-not (Test-Path $packagedOnnxRuntimeDllPath)) {
    throw "Packaged Local Qwen runtime DLL not found: $packagedOnnxRuntimeDllPath"
}
if (-not (Test-Path $packagedOnnxRuntimeProvidersSharedDllPath)) {
    throw "Packaged Local Qwen runtime providers DLL not found: $packagedOnnxRuntimeProvidersSharedDllPath"
}
if (-not (Test-Path $soxrReleaseInputsManifestPath)) {
    throw "Prepared soxr release inputs manifest not found: $soxrReleaseInputsManifestPath"
}
Assert-FileSha256Equals -Path $packagedNotoCjkFontPath -ExpectedSha256 $PinnedNotoCjkFontSha256 -Label "Packaged Noto Sans CJK Medium TTC"
foreach ($packagedNotoCjkProvenancePath in @($packagedNotoCjkLicensePath, $packagedNotoCjkReadmePath, $packagedNotoCjkSha256SumsPath)) {
    if (-not (Test-Path $packagedNotoCjkProvenancePath)) {
        throw "Packaged Noto Sans CJK provenance file not found: $packagedNotoCjkProvenancePath"
    }
}

$soxrReleaseInputsManifest = Get-Content -Path $soxrReleaseInputsManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$soxrSourceBundlePath = Join-Path $PWD $soxrReleaseInputsManifest.third_party_source_bundle_path
$packagedSoxrSourceBundlePath = Join-Path $packagedSoxrComplianceDir ([System.IO.Path]::GetFileName($soxrSourceBundlePath))

if (-not (Test-Path $packagedSoxrDllPath)) {
    throw "Packaged soxr runtime DLL not found: $packagedSoxrDllPath"
}

$packagedSoxrExtensions = @(Get-ChildItem -Path $packagedSoxrRuntimeDir -Filter "soxr_ext*.pyd" -File -ErrorAction SilentlyContinue)
if ($packagedSoxrExtensions.Count -ne 1) {
    throw "Packaged soxr runtime must contain exactly one soxr extension module in $packagedSoxrRuntimeDir; found $($packagedSoxrExtensions.Count)"
}
$packagedSoxrExtensionPath = $packagedSoxrExtensions[0].FullName

$packagedSoxrDlls = @(Get-ChildItem -Path $distDir -Filter "soxr.dll" -Recurse -File -ErrorAction SilentlyContinue)
if ($packagedSoxrDlls.Count -ne 1) {
    throw "Packaged soxr runtime must contain exactly one soxr.dll copy in $packagedSoxrRuntimeDir; found $($packagedSoxrDlls.Count)"
}

$expectedPackagedSoxrDllPath = [System.IO.Path]::GetFullPath($packagedSoxrDllPath)
$actualPackagedSoxrDllPath = [System.IO.Path]::GetFullPath($packagedSoxrDlls[0].FullName)
if ($actualPackagedSoxrDllPath -ne $expectedPackagedSoxrDllPath) {
    throw "Packaged soxr runtime DLL must only be staged at $packagedSoxrDllPath; found $actualPackagedSoxrDllPath"
}

$stalePackagedLibsoxrDlls = @(Get-ChildItem -Path $distDir -Filter "libsoxr.dll" -Recurse -File -ErrorAction SilentlyContinue)
if ($stalePackagedLibsoxrDlls.Count -ne 0) {
    throw "Packaged soxr runtime must not contain legacy libsoxr.dll copies: $($stalePackagedLibsoxrDlls.FullName -join ', ')"
}

if (-not (Test-Path $soxrSourceBundlePath)) {
    throw "soxr third-party source bundle not found: $soxrSourceBundlePath"
}
if (-not (Test-Path $soxrLicenseTextPath)) {
    throw "soxr LGPL license text not found: $soxrLicenseTextPath"
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
$sourceBundleArchive = [System.IO.Compression.ZipFile]::OpenRead($soxrSourceBundlePath)
try {
    $sourceBundleEntries = @($sourceBundleArchive.Entries | ForEach-Object { $_.FullName })
    $sourceBundleManifestEntry = $sourceBundleArchive.GetEntry("manifest.json")
    if ($null -eq $sourceBundleManifestEntry) {
        throw "soxr third-party source bundle is missing manifest.json"
    }

    $sourceBundleManifestReader = New-Object System.IO.StreamReader($sourceBundleManifestEntry.Open())
    try {
        $sourceBundleManifest = $sourceBundleManifestReader.ReadToEnd() | ConvertFrom-Json
    } finally {
        $sourceBundleManifestReader.Dispose()
    }

    $requiredSourceFilenames = @($sourceBundleManifest.sources | ForEach-Object { $_.filename })
    if ($requiredSourceFilenames.Count -eq 0) {
        throw "soxr third-party source bundle manifest is missing source entries"
    }

    foreach ($requiredSourceFilename in $requiredSourceFilenames) {
        if ([string]::IsNullOrWhiteSpace($requiredSourceFilename)) {
            throw "soxr third-party source bundle manifest contains a blank source filename"
        }
        if ($sourceBundleEntries -notcontains $requiredSourceFilename) {
            throw "soxr third-party source bundle is missing source archive: $requiredSourceFilename"
        }
    }
} finally {
    $sourceBundleArchive.Dispose()
}

New-Item -ItemType Directory -Force -Path $packagedSoxrComplianceDir | Out-Null
Copy-Item -Path $soxrLicenseTextPath -Destination $packagedSoxrLicensePath -Force
Copy-Item -Path $soxrSourceBundlePath -Destination $packagedSoxrSourceBundlePath -Force

if (-not (Test-Path $packagedSoxrLicensePath)) {
    throw "Packaged soxr LGPL license text not found: $packagedSoxrLicensePath"
}
if (-not (Test-Path $packagedSoxrSourceBundlePath)) {
    throw "Packaged soxr source bundle not found: $packagedSoxrSourceBundlePath"
}

$packagedOverlayPath = Join-Path $PWD "dist/PuriPulyHeart/PuriPulyHeartOverlay.exe"
$packagedGpuWorkerPath = Join-Path $PWD "dist/PuriPulyHeart/PuriPulyHeartGpuWorker.exe"
Copy-Item -Path $overlayStagedPath -Destination $packagedOverlayPath -Force

if (-not (Test-Path $packagedOverlayPath)) {
    throw "Packaged overlay executable not found: $packagedOverlayPath"
}
if (-not (Test-Path $packagedGpuWorkerPath)) {
    throw "Packaged GPU worker executable not found: $packagedGpuWorkerPath"
}
if (-not (Test-Path $packagedOverlayDllPath)) {
    throw "Packaged OpenVR runtime DLL not found: $packagedOverlayDllPath"
}
Assert-FileSha256Equals -Path $packagedOverlayDllPath -ExpectedSha256 $PinnedOpenVrVendorDllSha256 -Label "Packaged OpenVR runtime DLL"

Remove-Item -Recurse -Force $soxrRuntimeReportDir -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $soxrRuntimeReportDir | Out-Null
Remove-Item -Recurse -Force $processCaptureRuntimeReportDir -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $processCaptureRuntimeReportDir | Out-Null

Write-Host "Smoke-testing packaged executable..."
$versionSmokeTest = Start-Process -FilePath $exePath -ArgumentList @("--version") -Wait -PassThru
if ($versionSmokeTest.ExitCode -ne 0) {
    throw "Packaged executable version smoke test failed with exit code $($versionSmokeTest.ExitCode)"
}

Invoke-GuiStartupSmokeCheck -ExePath $exePath -Label "Packaged"

$localQwenRuntimeSmokeTest = Start-Process -FilePath $exePath -ArgumentList @("local-qwen-runtime-check") -Wait -PassThru
if ($localQwenRuntimeSmokeTest.ExitCode -ne 0) {
    throw "Local Qwen runtime smoke test failed with exit code $($localQwenRuntimeSmokeTest.ExitCode)"
}

Invoke-SoxrRuntimeSmokeCheck -ExePath $exePath -ReportPath $packagedSoxrRuntimeReportPath -ExpectedExtensionPath $packagedSoxrExtensionPath -ExpectedSoxrDllPath $packagedSoxrDllPath -Label "Packaged"
Invoke-ProcessCaptureRuntimeSmokeCheck -HelperExePath $processCaptureSmokeHelperPath -ArtifactRoot $processCaptureSmokeArtifactRoot -ReportPath $packagedProcessCaptureRuntimeReportPath -Label "Packaged production-collected smoke"

Write-Host "Smoke-testing packaged overlay executable..."
Invoke-External -FilePath $packagedOverlayPath -ArgumentList @("--check-startup-contract")

Write-Host "Smoke-testing packaged GPU worker executable..."
$packagedGpuWorkerVersion = (& $packagedGpuWorkerPath --version | Out-String).Trim()
if ($packagedGpuWorkerVersion -ne $AppVersion) {
    throw "Packaged GPU worker version mismatch: expected $AppVersion, found $packagedGpuWorkerVersion"
}

$isccPath = Join-Path ([Environment]::GetFolderPath("ProgramFilesX86")) "Inno Setup 6\ISCC.exe"
$currentInnoVersion = Get-InnoSetupVersion

if ($currentInnoVersion -eq $InnoSetupVersion -and (Test-Path $isccPath)) {
    Write-Host "Using installed Inno Setup $currentInnoVersion."
} else {
    $choco = Get-Command choco -ErrorAction SilentlyContinue
    if ($null -eq $choco) {
        throw "Chocolatey is required to install Inno Setup $InnoSetupVersion."
    }

    Write-Host "Installing Inno Setup $InnoSetupVersion..."
    Invoke-External -FilePath $choco.Source -ArgumentList @(
        "install",
        "innosetup",
        "--version=$InnoSetupVersion",
        "--no-progress",
        "-y"
    )

    $currentInnoVersion = Get-InnoSetupVersion
}

if (-not (Test-Path $isccPath)) {
    throw "ISCC.exe not found after Inno Setup install: $isccPath"
}

if ($currentInnoVersion -ne $InnoSetupVersion) {
    throw "Inno Setup version mismatch: expected $InnoSetupVersion, found $currentInnoVersion"
}

$installerPath = Join-Path $PWD "installer_output/PuriPulyHeart-Setup-$AppVersion.exe"
$installerHashPath = "$installerPath.sha256"
$InstallerTestAppId = "{{C2E4A7B1-59F3-4C89-9D21-7E6B5A4032F8}"
$InstallerSmokeBuildDir = Join-Path $env:TEMP "PuriPulyHeart-Installer-Smoke"
$InstallerSmokeDir = Join-Path $env:LOCALAPPDATA "Programs\PuriPulyHeart-LocalSTT-Test"
$InstallerSmokeAppDataRoot = Join-Path $env:TEMP "PuriPulyHeart-LocalSTT-Test-AppData"
$InstallerSmokeLogPath = Join-Path $env:TEMP "PuriPulyHeart-LocalSTT-Test.log"
$InstallerReinstallSmokeLogPath = Join-Path $env:TEMP "PuriPulyHeart-LocalSTT-Test-reinstall.log"
$installedProcessCaptureSmokeArtifactRoot = Join-Path $InstallerSmokeDir "process-capture-smoke"
$installedProcessCaptureSmokeHelperPath = Join-Path $installedProcessCaptureSmokeArtifactRoot "PuriPulyHeartProcessCaptureSmoke.exe"
$installedExePath = Join-Path $InstallerSmokeDir "PuriPulyHeart.exe"
$installedOpenVrDllPath = Join-Path $InstallerSmokeDir "openvr_api.dll"
$installedNotoCjkFontPath = Join-Path $InstallerSmokeDir "puripuly_heart\data\fonts\NotoSansCJK-Medium.ttc"
$installedNotoCjkProvenanceDir = Join-Path $InstallerSmokeDir "third_party\noto-sans-cjk"
$installedNotoCjkLicensePath = Join-Path $installedNotoCjkProvenanceDir "OFL.txt"
$installedNotoCjkReadmePath = Join-Path $installedNotoCjkProvenanceDir "README.md"
$installedNotoCjkSha256SumsPath = Join-Path $installedNotoCjkProvenanceDir "SHA256SUMS.txt"
$installedSoxrDllPath = Join-Path $InstallerSmokeDir "soxr\soxr.dll"
$installedSoxrExtensionPath = Join-Path $InstallerSmokeDir (Join-Path "soxr" ([System.IO.Path]::GetFileName($packagedSoxrExtensionPath)))
$legacyRootLevelSoxrDllPath = Join-Path $InstallerSmokeDir "soxr.dll"
$installedLegacySoxrDllPath = Join-Path $InstallerSmokeDir "soxr\libsoxr.dll"
$installedSoxrComplianceDir = Join-Path $InstallerSmokeDir "third_party\soxr"
$installedSoxrLicensePath = Join-Path $installedSoxrComplianceDir "COPYING.LGPL-2.1.txt"
$installedSoxrSourceBundlePath = Join-Path $installedSoxrComplianceDir ([System.IO.Path]::GetFileName($soxrSourceBundlePath))
$smokeInstallerPath = Join-Path $InstallerSmokeBuildDir "PuriPulyHeart-Setup-$AppVersion.exe"
if (Test-Path $installerPath) {
    Remove-Item -Path $installerPath -Force
}
if (Test-Path $installerHashPath) {
    Remove-Item -Path $installerHashPath -Force
}
if (Test-Path $InstallerSmokeBuildDir) {
    Remove-Item -Recurse -Force $InstallerSmokeBuildDir -ErrorAction SilentlyContinue
}
if (Test-Path $InstallerSmokeDir) {
    Remove-Item -Recurse -Force $InstallerSmokeDir -ErrorAction SilentlyContinue
}
if (Test-Path $InstallerSmokeAppDataRoot) {
    Remove-Item -Recurse -Force $InstallerSmokeAppDataRoot -ErrorAction SilentlyContinue
}
if (Test-Path $InstallerSmokeLogPath) {
    Remove-Item -Path $InstallerSmokeLogPath -Force -ErrorAction SilentlyContinue
}
if (Test-Path $InstallerReinstallSmokeLogPath) {
    Remove-Item -Path $InstallerReinstallSmokeLogPath -Force -ErrorAction SilentlyContinue
}

Write-Host "Building installer..."
Invoke-ExternalProcess -FilePath $isccPath -ArgumentList @("installer.iss") -WorkingDirectory $PWD

if (-not (Test-Path $installerPath)) {
    throw "Installer not found: $installerPath"
}

if (-not (Test-Path $packagedOverlayPath)) {
    Copy-Item -Path $overlayStagedPath -Destination $packagedOverlayPath -Force
}

if (-not (Test-Path $packagedOverlayPath)) {
    throw "Packaged overlay executable not found after installer build: $packagedOverlayPath"
}

Write-Host "Building smoke-test installer with alternate AppId..."
Invoke-ExternalProcess -FilePath $isccPath -ArgumentList @(
    "/DMyAppId=$InstallerTestAppId",
    "/DSkipLocalSttProvisioning=1",
    "/DProcessCaptureSmokeArtifactRoot=$processCaptureSmokeArtifactRoot",
    "/O$InstallerSmokeBuildDir",
    "installer.iss"
) -WorkingDirectory $PWD

if (-not (Test-Path $smokeInstallerPath)) {
    throw "Smoke installer not found: $smokeInstallerPath"
}

Write-Host "Smoke-testing installer with alternate AppId and isolated directory..."
$previousLocalSttAppDataRoot = $env:PURIPULY_HEART_LOCAL_STT_APPDATA_ROOT
$env:PURIPULY_HEART_LOCAL_STT_APPDATA_ROOT = $InstallerSmokeAppDataRoot
try {
    $installerSmoke = Start-Process -FilePath $smokeInstallerPath -ArgumentList @(
        "/CURRENTUSER",
        "/VERYSILENT",
        "/SUPPRESSMSGBOXES",
        "/DIR=$InstallerSmokeDir",
        "/LOG=$InstallerSmokeLogPath"
    ) -Wait -PassThru
} finally {
    if ($null -eq $previousLocalSttAppDataRoot) {
        Remove-Item Env:PURIPULY_HEART_LOCAL_STT_APPDATA_ROOT -ErrorAction SilentlyContinue
    } else {
        $env:PURIPULY_HEART_LOCAL_STT_APPDATA_ROOT = $previousLocalSttAppDataRoot
    }
}
if ($installerSmoke.ExitCode -ne 0) {
    throw "Installer smoke test failed with exit code $($installerSmoke.ExitCode)"
}
if (-not (Test-Path $InstallerSmokeLogPath)) {
    throw "Installer smoke log not found: $InstallerSmokeLogPath"
}
$installerSmokeLog = Get-Content -Path $InstallerSmokeLogPath -Raw
if ($installerSmokeLog -match "Local STT provisioning failed" -or $installerSmokeLog -match "Failed to launch local STT provisioning script") {
    throw "Installer smoke test did not complete local STT provisioning successfully"
}
if ($installerSmokeLog -notmatch [regex]::Escape("Local STT provisioning skipped for isolated installer smoke.")) {
    throw "Installer smoke log is missing isolated no-network provisioning skip marker"
}
if ($installerSmokeLog -match [regex]::Escape("Local STT provisioning completed successfully.")) {
    throw "Installer smoke unexpectedly performed Local STT network provisioning"
}
if (-not (Test-Path $installedExePath)) {
    throw "Installed app executable not found after installer smoke: $installedExePath"
}
if (-not (Test-Path $installedProcessCaptureSmokeHelperPath)) {
    throw "Installed release-only process-capture smoke helper not found: $installedProcessCaptureSmokeHelperPath"
}
Invoke-External -FilePath $pythonCommand -ArgumentList @(
    "-m",
    $ManagedGemmaDistributionModule,
    "verify-package",
    $InstallerSmokeDir,
    "--launch"
)
if (-not (Test-Path $installedOpenVrDllPath)) {
    throw "Installed OpenVR runtime DLL not found after installer smoke: $installedOpenVrDllPath"
}
Assert-FileSha256Equals -Path $installedOpenVrDllPath -ExpectedSha256 $PinnedOpenVrVendorDllSha256 -Label "Installed OpenVR runtime DLL"
Assert-FileSha256Equals -Path $installedNotoCjkFontPath -ExpectedSha256 $PinnedNotoCjkFontSha256 -Label "Installed Noto Sans CJK Medium TTC"
foreach ($installedNotoCjkProvenancePath in @($installedNotoCjkLicensePath, $installedNotoCjkReadmePath, $installedNotoCjkSha256SumsPath)) {
    if (-not (Test-Path $installedNotoCjkProvenancePath)) {
        throw "Installed Noto Sans CJK provenance file not found after installer smoke: $installedNotoCjkProvenancePath"
    }
}
if (-not (Test-Path $installedSoxrDllPath)) {
    throw "Installed app soxr runtime DLL not found after installer smoke: $installedSoxrDllPath"
}
if (Test-Path $installedLegacySoxrDllPath) {
    throw "Installed app still contains stale legacy soxr runtime DLL after installer smoke: $installedLegacySoxrDllPath"
}
if (-not (Test-Path $installedSoxrComplianceDir)) {
    throw "Installed soxr compliance bundle directory not found after installer smoke: $installedSoxrComplianceDir"
}
if (-not (Test-Path $installedSoxrLicensePath)) {
    throw "Installed soxr LGPL license text not found after installer smoke: $installedSoxrLicensePath"
}
if (-not (Test-Path $installedSoxrSourceBundlePath)) {
    throw "Installed soxr source bundle not found after installer smoke: $installedSoxrSourceBundlePath"
}

Invoke-SoxrRuntimeSmokeCheck -ExePath $installedExePath -ReportPath $installedSoxrRuntimeReportPath -ExpectedExtensionPath $installedSoxrExtensionPath -ExpectedSoxrDllPath $installedSoxrDllPath -Label "Installed"
Invoke-GuiStartupSmokeCheck -ExePath $installedExePath -Label "Installed"
Invoke-ProcessCaptureRuntimeSmokeCheck -HelperExePath $installedProcessCaptureSmokeHelperPath -ArtifactRoot $installedProcessCaptureSmokeArtifactRoot -ReportPath $installedProcessCaptureRuntimeReportPath -Label "Installed"

$expectedInstalledSoxrDllHash = (Get-FileHash -Path $packagedSoxrDllPath -Algorithm SHA256).Hash
$expectedInstalledSoxrLicenseHash = (Get-FileHash -Path $packagedSoxrLicensePath -Algorithm SHA256).Hash
$expectedInstalledSoxrSourceBundleHash = (Get-FileHash -Path $packagedSoxrSourceBundlePath -Algorithm SHA256).Hash
$installedSoxrLicenseHash = (Get-FileHash -Path $installedSoxrLicensePath -Algorithm SHA256).Hash
if ($installedSoxrLicenseHash -ne $expectedInstalledSoxrLicenseHash) {
    throw "Installed soxr LGPL license text hash does not match packaged compliance bundle after installer smoke"
}
$installedSoxrSourceBundleHash = (Get-FileHash -Path $installedSoxrSourceBundlePath -Algorithm SHA256).Hash
if ($installedSoxrSourceBundleHash -ne $expectedInstalledSoxrSourceBundleHash) {
    throw "Installed soxr source bundle hash does not match packaged compliance bundle after installer smoke"
}
[System.IO.File]::WriteAllBytes($installedOpenVrDllPath, [System.Text.Encoding]::ASCII.GetBytes("manual replacement openvr smoke payload"))
$mutatedInstalledOpenVrDllHash = Get-FileSha256 -Path $installedOpenVrDllPath
if ($mutatedInstalledOpenVrDllHash -eq $PinnedOpenVrVendorDllSha256) {
    throw "Failed to mutate installed OpenVR runtime DLL before reinstall smoke"
}
[System.IO.File]::WriteAllBytes($installedSoxrDllPath, [System.Text.Encoding]::ASCII.GetBytes("manual replacement smoke payload"))
$mutatedInstalledSoxrDllHash = (Get-FileHash -Path $installedSoxrDllPath -Algorithm SHA256).Hash
if ($mutatedInstalledSoxrDllHash -eq $expectedInstalledSoxrDllHash) {
    throw "Failed to mutate installed soxr runtime DLL before reinstall smoke"
}
[System.IO.File]::WriteAllText($installedSoxrLicensePath, "manual compliance license smoke payload", [System.Text.Encoding]::ASCII)
$mutatedInstalledSoxrLicenseHash = (Get-FileHash -Path $installedSoxrLicensePath -Algorithm SHA256).Hash
if ($mutatedInstalledSoxrLicenseHash -eq $expectedInstalledSoxrLicenseHash) {
    throw "Failed to mutate installed soxr LGPL license text before reinstall smoke"
}
[System.IO.File]::WriteAllBytes($installedSoxrSourceBundlePath, [System.Text.Encoding]::ASCII.GetBytes("manual compliance source bundle smoke payload"))
$mutatedInstalledSoxrSourceBundleHash = (Get-FileHash -Path $installedSoxrSourceBundlePath -Algorithm SHA256).Hash
if ($mutatedInstalledSoxrSourceBundleHash -eq $expectedInstalledSoxrSourceBundleHash) {
    throw "Failed to mutate installed soxr source bundle before reinstall smoke"
}
[System.IO.File]::WriteAllBytes($installedLegacySoxrDllPath, [System.Text.Encoding]::ASCII.GetBytes("stale legacy runtime smoke payload"))
if (-not (Test-Path $installedLegacySoxrDllPath)) {
    throw "Failed to seed stale legacy soxr runtime DLL before reinstall smoke"
}
[System.IO.File]::WriteAllBytes($legacyRootLevelSoxrDllPath, [System.Text.Encoding]::ASCII.GetBytes("stale root-level runtime smoke payload"))
if (-not (Test-Path $legacyRootLevelSoxrDllPath)) {
    throw "Failed to seed stale root-level soxr runtime DLL before reinstall smoke"
}

Write-Host "Smoke-testing installer reinstall replaces installed soxr runtime DLL..."
$previousLocalSttAppDataRoot = $env:PURIPULY_HEART_LOCAL_STT_APPDATA_ROOT
$env:PURIPULY_HEART_LOCAL_STT_APPDATA_ROOT = $InstallerSmokeAppDataRoot
try {
    $installerReinstallSmoke = Start-Process -FilePath $smokeInstallerPath -ArgumentList @(
        "/CURRENTUSER",
        "/VERYSILENT",
        "/SUPPRESSMSGBOXES",
        "/DIR=$InstallerSmokeDir",
        "/LOG=$InstallerReinstallSmokeLogPath"
    ) -Wait -PassThru
} finally {
    if ($null -eq $previousLocalSttAppDataRoot) {
        Remove-Item Env:PURIPULY_HEART_LOCAL_STT_APPDATA_ROOT -ErrorAction SilentlyContinue
    } else {
        $env:PURIPULY_HEART_LOCAL_STT_APPDATA_ROOT = $previousLocalSttAppDataRoot
    }
}
if ($installerReinstallSmoke.ExitCode -ne 0) {
    throw "Installer reinstall smoke test failed with exit code $($installerReinstallSmoke.ExitCode)"
}
if (-not (Test-Path $InstallerReinstallSmokeLogPath)) {
    throw "Installer reinstall smoke log not found: $InstallerReinstallSmokeLogPath"
}
$installerReinstallSmokeLog = Get-Content -Path $InstallerReinstallSmokeLogPath -Raw
if ($installerReinstallSmokeLog -match "Local STT provisioning failed" -or $installerReinstallSmokeLog -match "Failed to launch local STT provisioning script") {
    throw "Installer reinstall smoke test did not complete local STT provisioning successfully"
}
if ($installerReinstallSmokeLog -notmatch [regex]::Escape("Local STT provisioning skipped for isolated installer smoke.")) {
    throw "Installer reinstall smoke log is missing isolated no-network provisioning skip marker"
}

$reinstalledOpenVrDllHash = Get-FileSha256 -Path $installedOpenVrDllPath
if ($reinstalledOpenVrDllHash -ne $PinnedOpenVrVendorDllSha256) {
    throw "Installed OpenVR runtime DLL reinstall smoke failed to restore pinned hash"
}
Assert-FileSha256Equals -Path $installedOpenVrDllPath -ExpectedSha256 $PinnedOpenVrVendorDllSha256 -Label "Reinstalled OpenVR runtime DLL"
$reinstalledSoxrDllHash = (Get-FileHash -Path $installedSoxrDllPath -Algorithm SHA256).Hash
if ($reinstalledSoxrDllHash -ne $expectedInstalledSoxrDllHash) {
    throw "Installed soxr runtime DLL reinstall smoke failed to restore bundled hash"
}
if (-not (Test-Path $installedSoxrLicensePath)) {
    throw "Installed soxr LGPL license text not found after reinstall smoke: $installedSoxrLicensePath"
}
if (-not (Test-Path $installedSoxrSourceBundlePath)) {
    throw "Installed soxr source bundle not found after reinstall smoke: $installedSoxrSourceBundlePath"
}
$reinstalledSoxrLicenseHash = (Get-FileHash -Path $installedSoxrLicensePath -Algorithm SHA256).Hash
if ($reinstalledSoxrLicenseHash -ne $expectedInstalledSoxrLicenseHash) {
    throw "Installed soxr LGPL license text reinstall smoke failed to restore bundled hash"
}
$reinstalledSoxrSourceBundleHash = (Get-FileHash -Path $installedSoxrSourceBundlePath -Algorithm SHA256).Hash
if ($reinstalledSoxrSourceBundleHash -ne $expectedInstalledSoxrSourceBundleHash) {
    throw "Installed soxr source bundle reinstall smoke failed to restore bundled hash"
}
if (Test-Path $installedLegacySoxrDllPath) {
    throw "Installed stale legacy soxr runtime DLL was not removed by reinstall smoke: $installedLegacySoxrDllPath"
}
if (Test-Path $legacyRootLevelSoxrDllPath) {
    throw "Installed stale root-level soxr runtime DLL was not removed by reinstall smoke: $legacyRootLevelSoxrDllPath"
}

Invoke-SoxrRuntimeSmokeCheck -ExePath $installedExePath -ReportPath $reinstalledSoxrRuntimeReportPath -ExpectedExtensionPath $installedSoxrExtensionPath -ExpectedSoxrDllPath $installedSoxrDllPath -Label "Reinstalled"
Invoke-GuiStartupSmokeCheck -ExePath $installedExePath -Label "Reinstalled"
Invoke-ProcessCaptureRuntimeSmokeCheck -HelperExePath $installedProcessCaptureSmokeHelperPath -ArtifactRoot $installedProcessCaptureSmokeArtifactRoot -ReportPath $reinstalledProcessCaptureRuntimeReportPath -Label "Reinstalled"
Invoke-External -FilePath $pythonCommand -ArgumentList @(
    "-m",
    $ManagedGemmaDistributionModule,
    "verify-package",
    $InstallerSmokeDir,
    "--launch"
)

$installedUninstallerPath = Join-Path $InstallerSmokeDir "unins000.exe"
if (-not (Test-Path $installedUninstallerPath)) {
    throw "Isolated installer smoke uninstaller not found: $installedUninstallerPath"
}
$uninstallSmoke = Start-Process -FilePath $installedUninstallerPath -ArgumentList @(
    "/VERYSILENT",
    "/SUPPRESSMSGBOXES",
    "/NORESTART"
) -Wait -PassThru
if ($uninstallSmoke.ExitCode -ne 0) {
    throw "Isolated installer smoke cleanup failed with exit code $($uninstallSmoke.ExitCode)"
}
Start-Sleep -Seconds 1
if (Test-Path $InstallerSmokeDir) {
    throw "Isolated installer smoke directory remains after cleanup: $InstallerSmokeDir"
}

Write-Host "Generating SHA256..."
$hash = (Get-FileHash -Path $installerPath -Algorithm SHA256).Hash
"$hash  PuriPulyHeart-Setup-$AppVersion.exe" | Out-File -FilePath $installerHashPath -Encoding ascii
