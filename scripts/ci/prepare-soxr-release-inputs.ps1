[CmdletBinding()]
param(
    [Parameter()]
    [string]$OutputRoot = "",

    [Parameter()]
    [string]$PackagedRuntimeRelativeDir = "soxr"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$PinnedSoxrSpecifier = "soxr==1.1.0"
$SoxrVersion = "1.1.0"
$LibsoxrVersion = "0.1.3"
$SoxrSdistUrl = "https://files.pythonhosted.org/packages/ed/11/27cebce4a108f77afea7c80545115536b45e3f11ebfb914f638fdd9ba847/soxr-1.1.0.tar.gz"
$SoxrSdistSha256 = "9f228ae21c78fa9359ca98d8a5e8e91f30639e438e574133dace62c5b5309e44"
$LibsoxrSourceUrl = "https://sourceforge.net/projects/soxr/files/soxr-0.1.3-Source.tar.xz/download"
$expectedLibsoxrSourceSha256 = "b111c15fdc8c029989330ff559184198c161100a59312f5dc19ddeb9b5a15889"
$ZeroconfVersion = "0.150.0"
$ZeroconfSdistUrl = "https://files.pythonhosted.org/packages/09/ea/34bb185645ecaa18d34e5883bffea71aa9bffbbb994634884e8b2f3ad0c4/zeroconf-0.150.0.tar.gz"
$ZeroconfSdistSha256 = "a5fe7feab1de6ef5e541e0a3d07e534fd91629b813fc27281593584100f63164"
$SoxrBuildPatchName = "python-soxr-1.1.0-release-input.patch"
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $ReleaseInputsRoot = Join-Path $PWD "build/soxr-release-inputs"
} else {
    $ReleaseInputsRoot = [System.IO.Path]::GetFullPath($OutputRoot)
}
$ManifestPath = Join-Path $ReleaseInputsRoot "manifest.json"
$SourceBundleName = "PuriPulyHeart-soxr-third-party-source-bundle.zip"
$SourceBundlePath = Join-Path $ReleaseInputsRoot $SourceBundleName

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

function Get-FileSha256 {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    return (Get-FileHash -Path $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-RepoRelativePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $fullPath = [System.IO.Path]::GetFullPath($Path)
    $repoRoot = [System.IO.Path]::GetFullPath($PWD)
    if (-not $repoRoot.EndsWith([System.IO.Path]::DirectorySeparatorChar)) {
        $repoRoot += [System.IO.Path]::DirectorySeparatorChar
    }
    $repoRootUri = [System.Uri]::new($repoRoot)
    $fullPathUri = [System.Uri]::new($fullPath)
    return [System.Uri]::UnescapeDataString($repoRootUri.MakeRelativeUri($fullPathUri).ToString())
}

$projectEnvironmentScripts = Resolve-ProjectEnvironmentScriptsPath
if (Test-Path $projectEnvironmentScripts) {
    $env:PATH = "$projectEnvironmentScripts;$env:PATH"
}

$pythonCommand = Resolve-CommandPath -Name "python" -Fallbacks @(
    (Join-Path $projectEnvironmentScripts "python.exe")
)
$cmakeCommand = Resolve-CommandPath -Name "cmake" -Fallbacks @(
    (Join-Path $projectEnvironmentScripts "cmake.exe"),
    "C:\Program Files\CMake\bin\cmake.exe"
)
$curlCommand = Resolve-CommandPath -Name "curl.exe" -Fallbacks @(
    (Join-Path $env:SystemRoot "System32\curl.exe")
)
$tarCommand = Resolve-CommandPath -Name "tar"
$pythonVersion = (& $pythonCommand --version 2>&1).Trim()
$cmakeVersion = ((& $cmakeCommand --version 2>&1) | Select-Object -First 1).Trim()


$pyproject = Get-Content -Path (Join-Path $PWD "pyproject.toml") -Raw -Encoding utf8
if ($pyproject -notmatch [regex]::Escape($PinnedSoxrSpecifier)) {
    throw "pyproject.toml no longer pins $PinnedSoxrSpecifier"
}

$uvLock = Get-Content -Path (Join-Path $PWD "uv.lock") -Raw -Encoding utf8
if ($uvLock -notmatch '(?ms)\[\[package\]\]\s+name = "soxr"\s+version = "1\.1\.0"') {
    throw "uv.lock no longer pins soxr 1.1.0"
}
if ($uvLock -notmatch '(?ms)\[\[package\]\]\s+name = "zeroconf"\s+version = "0\.150\.0".*?sdist = \{ url = "https://files\.pythonhosted\.org/packages/09/ea/34bb185645ecaa18d34e5883bffea71aa9bffbbb994634884e8b2f3ad0c4/zeroconf-0\.150\.0\.tar\.gz", hash = "sha256:a5fe7feab1de6ef5e541e0a3d07e534fd91629b813fc27281593584100f63164"') {
    throw "uv.lock no longer pins the expected zeroconf 0.150.0 source distribution"
}


$downloadsDir = Join-Path $ReleaseInputsRoot "downloads"
$soxrExtractRoot = Join-Path $ReleaseInputsRoot "soxr-src"
$libsoxrExtractRoot = Join-Path $ReleaseInputsRoot "libsoxr-src"
$libsoxrBuildDir = Join-Path $ReleaseInputsRoot "libsoxr-build"
$libsoxrInstallDir = Join-Path $ReleaseInputsRoot "libsoxr-install"
$wheelOutputDir = Join-Path $ReleaseInputsRoot "wheel"
$wheelExtractDir = Join-Path $ReleaseInputsRoot "wheel-extract"
$runtimeStageDir = Join-Path $ReleaseInputsRoot "runtime"
$sourceBundleStageDir = Join-Path $ReleaseInputsRoot "source-bundle"

Write-Host "Preparing soxr release inputs in $ReleaseInputsRoot"
Remove-Item -Recurse -Force $ReleaseInputsRoot -ErrorAction SilentlyContinue
foreach ($path in @(
    $downloadsDir,
    $soxrExtractRoot,
    $libsoxrExtractRoot,
    $libsoxrBuildDir,
    $libsoxrInstallDir,
    $wheelOutputDir,
    $wheelExtractDir,
    $runtimeStageDir,
    $sourceBundleStageDir
)) {
    New-Item -ItemType Directory -Path $path -Force | Out-Null
}

$soxrSdistPath = Join-Path $downloadsDir "soxr-$SoxrVersion.tar.gz"
$libsoxrSourcePath = Join-Path $downloadsDir "soxr-$LibsoxrVersion-Source.tar.xz"
$zeroconfSdistPath = Join-Path $downloadsDir "zeroconf-$ZeroconfVersion.tar.gz"

Write-Host "Downloading python-soxr source distribution..."
Invoke-WebRequest -Uri $SoxrSdistUrl -OutFile $soxrSdistPath
$actualSoxrSdistSha256 = Get-FileSha256 -Path $soxrSdistPath
if ($actualSoxrSdistSha256 -ne $SoxrSdistSha256) {
    throw "python-soxr source hash mismatch: expected $SoxrSdistSha256, found $actualSoxrSdistSha256"
}
Write-Host "Downloading python-zeroconf source distribution..."
Invoke-WebRequest -Uri $ZeroconfSdistUrl -OutFile $zeroconfSdistPath
$actualZeroconfSdistSha256 = Get-FileSha256 -Path $zeroconfSdistPath
if ($actualZeroconfSdistSha256 -ne $ZeroconfSdistSha256) {
    throw "python-zeroconf source hash mismatch: expected $ZeroconfSdistSha256, found $actualZeroconfSdistSha256"
}


Write-Host "Extracting python-soxr source distribution..."
Invoke-External -FilePath $tarCommand -ArgumentList @("-xf", $soxrSdistPath, "-C", $soxrExtractRoot)
$soxrSourceRoot = Get-ChildItem -Path $soxrExtractRoot -Directory | Select-Object -First 1
if ($null -eq $soxrSourceRoot) {
    throw "Could not locate extracted python-soxr source directory in $soxrExtractRoot"
}
$soxrCMakeListsPath = Join-Path $soxrSourceRoot.FullName "CMakeLists.txt"
$soxrCMakeLists = Get-Content -Path $soxrCMakeListsPath -Raw -Encoding utf8
if ($soxrCMakeLists -notmatch [regex]::Escape("nanobind_add_stub(soxr_ext_stub")) {
    throw "python-soxr CMakeLists.txt no longer contains nanobind_add_stub(soxr_ext_stub)"
}
if ($soxrCMakeLists -notmatch [regex]::Escape("if (NOT CMAKE_CROSSCOMPILING)")) {
    throw "python-soxr CMakeLists.txt no longer contains if (NOT CMAKE_CROSSCOMPILING)"
}
$soxrCMakeLists = $soxrCMakeLists -replace [regex]::Escape("if (NOT CMAKE_CROSSCOMPILING)"), "if (FALSE) # release-input wheel build disables stub generation"
Set-Content -Path $soxrCMakeListsPath -Value $soxrCMakeLists -Encoding utf8
$soxrBuildPatch = @"
--- a/CMakeLists.txt
+++ b/CMakeLists.txt
@@ -65,7 +65,7 @@ nanobind_add_module(soxr_ext STABLE_ABI FREE_THREADED NB_STATIC
     `${CSOXR_VER_C}
 )
 
-if (NOT CMAKE_CROSSCOMPILING)
+if (FALSE) # release-input wheel build disables stub generation
     # nanobind's stub generation requires importing the module, so skip it when cross-compiling
     nanobind_add_stub(soxr_ext_stub
         MODULE soxr_ext
"@
$soxrBuildPatchPath = Join-Path $sourceBundleStageDir $SoxrBuildPatchName
[System.IO.File]::WriteAllText($soxrBuildPatchPath, "$soxrBuildPatch`n", [System.Text.UTF8Encoding]::new($false))


Write-Host "Downloading libsoxr source archive..."
Invoke-External -FilePath $curlCommand -ArgumentList @(
    "-L",
    $LibsoxrSourceUrl,
    "-o",
    $libsoxrSourcePath
)
$libsoxrSourceSha256 = Get-FileSha256 -Path $libsoxrSourcePath
if ($libsoxrSourceSha256 -ne $expectedLibsoxrSourceSha256) {
    throw "libsoxr source hash mismatch: expected $expectedLibsoxrSourceSha256, found $libsoxrSourceSha256"
}

Write-Host "Extracting libsoxr source archive..."
Invoke-External -FilePath $tarCommand -ArgumentList @("-xf", $libsoxrSourcePath, "-C", $libsoxrExtractRoot)
$libsoxrSourceRoot = Get-ChildItem -Path $libsoxrExtractRoot -Directory | Select-Object -First 1
if ($null -eq $libsoxrSourceRoot) {
    throw "Could not locate extracted libsoxr source directory in $libsoxrExtractRoot"
}

Write-Host "Building shared libsoxr runtime..."
Invoke-External -FilePath $cmakeCommand -ArgumentList @(
    "-S",
    $libsoxrSourceRoot.FullName,
    "-B",
    $libsoxrBuildDir,
    "-DBUILD_SHARED_LIBS=ON",
    "-DBUILD_TESTS=OFF",
    "-DBUILD_EXAMPLES=OFF",
    "-DWITH_OPENMP=OFF",
    "-DWITH_LSR_BINDINGS=OFF",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
    "-DCMAKE_INSTALL_PREFIX=$libsoxrInstallDir"
)
$compilerMetadataPath = Get-ChildItem -Path (Join-Path $libsoxrBuildDir "CMakeFiles") -Filter "CMakeCCompiler.cmake" -File -Recurse | Select-Object -First 1
if ($null -eq $compilerMetadataPath) {
    throw "CMake compiler metadata was not generated under $libsoxrBuildDir"
}
$compilerMetadata = Get-Content -Path $compilerMetadataPath.FullName -Raw -Encoding utf8
if ($compilerMetadata -notmatch 'set\(CMAKE_C_COMPILER_ID "([^"]+)"\)') {
    throw "CMake compiler metadata is missing CMAKE_C_COMPILER_ID"
}
$compilerId = $Matches[1]
if ($compilerMetadata -notmatch 'set\(CMAKE_C_COMPILER_VERSION "([^"]+)"\)') {
    throw "CMake compiler metadata is missing CMAKE_C_COMPILER_VERSION"
}
$compilerVersion = $Matches[1]
$cmakeCache = Get-Content -Path (Join-Path $libsoxrBuildDir "CMakeCache.txt") -Raw -Encoding utf8
if ($cmakeCache -notmatch '(?m)^CMAKE_GENERATOR:INTERNAL=(.+)$') {
    throw "CMake cache is missing CMAKE_GENERATOR"
}
$cmakeGenerator = $Matches[1].Trim()
$generatedProjectPath = Get-ChildItem -Path $libsoxrBuildDir -Filter "*.vcxproj" -File | Select-Object -First 1
if (($null -eq $generatedProjectPath) -or ((Get-Content -Path $generatedProjectPath.FullName -Raw -Encoding utf8) -notmatch '<WindowsTargetPlatformVersion>([^<]+)</WindowsTargetPlatformVersion>')) {
    throw "Generated libsoxr project is missing WindowsTargetPlatformVersion"
}
$windowsSdkVersion = $Matches[1]

Invoke-External -FilePath $cmakeCommand -ArgumentList @(
    "--build",
    $libsoxrBuildDir,
    "--config",
    "Release"
)
Invoke-External -FilePath $cmakeCommand -ArgumentList @(
    "--install",
    $libsoxrBuildDir,
    "--config",
    "Release"
)

$libsoxrBuiltDllPath = Join-Path $libsoxrInstallDir "bin\soxr.dll"
$libsoxrBinDir = Join-Path $libsoxrInstallDir "bin"
$libsoxrImportLibPath = Join-Path $libsoxrInstallDir "lib\soxr.lib"
$libsoxrHeaderPath = Join-Path $libsoxrInstallDir "include\soxr.h"
if (-not (Test-Path $libsoxrBuiltDllPath)) {
    throw "Built libsoxr DLL not found: $libsoxrBuiltDllPath"
}
if (-not (Test-Path $libsoxrImportLibPath)) {
    throw "Built libsoxr import library not found: $libsoxrImportLibPath"
}
if (-not (Test-Path $libsoxrHeaderPath)) {
    throw "Installed libsoxr header not found: $libsoxrHeaderPath"
}

$env:PATH = "$libsoxrBinDir;$env:PATH"

Write-Host "Bootstrapping pip into the prepared project environment..."
Invoke-External -FilePath $pythonCommand -ArgumentList @(
    "-m",
    "ensurepip",
    "--upgrade"
)
$pipVersion = ((& $pythonCommand -m pip --version 2>&1) | Out-String).Trim()
$buildBackendVersions = (& $pythonCommand -c "import importlib.metadata as m; print('nanobind=' + m.version('nanobind') + ';scikit-build-core=' + m.version('scikit-build-core'))").Trim()


$nanobindCmakeDir = (& $pythonCommand -c "import pathlib, nanobind; print((pathlib.Path(nanobind.__file__).resolve().parent / 'cmake'))").Trim()
if ([string]::IsNullOrWhiteSpace($nanobindCmakeDir)) {
    throw "Could not resolve nanobind CMake directory from the prepared project environment."
}
$nanobindConfigPath = Join-Path $nanobindCmakeDir "nanobind-config.cmake"
if (-not (Test-Path $nanobindConfigPath)) {
    throw "nanobind-config.cmake not found: $nanobindConfigPath"
}

Write-Host "Building custom system-linked python-soxr source wheel using the prepared project environment (no build isolation)..."
$wheelBuildSoxrDllPath = Join-Path $projectEnvironmentScripts "soxr.dll"
$wheelBuildSoxrDllBackupPath = Join-Path $ReleaseInputsRoot "wheel-build-python-soxr.dll.backup"
$hadExistingWheelBuildSoxrDll = Test-Path $wheelBuildSoxrDllPath
$wheelBuildError = $null
$wheelBuildCleanupActionError = $null
try {
    if ($hadExistingWheelBuildSoxrDll) {
        Move-Item -Path $wheelBuildSoxrDllPath -Destination $wheelBuildSoxrDllBackupPath -Force
    }

    Copy-Item -Path $libsoxrBuiltDllPath -Destination $wheelBuildSoxrDllPath -Force

    Invoke-External -FilePath $pythonCommand -ArgumentList @(
        "-m",
        "pip",
        "wheel",
        "--no-build-isolation",
        "--no-deps",
        "--wheel-dir",
        $wheelOutputDir,
        "--config-settings=cmake.define.USE_SYSTEM_LIBSOXR=ON",
        "--config-settings=cmake.define.CMAKE_PREFIX_PATH=$libsoxrInstallDir",
        "--config-settings=cmake.define.nanobind_DIR=$nanobindCmakeDir",
        "--config-settings=cmake.define.SOXR_LIBRARY=$libsoxrImportLibPath",
        "--config-settings=cmake.define.SOXR_INCLUDE_DIR=$(Join-Path $libsoxrInstallDir 'include')",
        $soxrSourceRoot.FullName
    )
}
catch {
    $wheelBuildError = $_
}
finally {
    try {
        Remove-Item -Path $wheelBuildSoxrDllPath -Force -ErrorAction SilentlyContinue
        if ($hadExistingWheelBuildSoxrDll -and (Test-Path $wheelBuildSoxrDllBackupPath)) {
            Move-Item -Path $wheelBuildSoxrDllBackupPath -Destination $wheelBuildSoxrDllPath -Force
        }
    }
    catch {
        $wheelBuildCleanupActionError = $_
    }
}

$wheelBuildCleanupIssues = @()
if ($null -ne $wheelBuildCleanupActionError) {
    $wheelBuildCleanupIssues += "Wheel-build soxr.dll cleanup action failed: $($wheelBuildCleanupActionError.Exception.Message)"
}
if ($hadExistingWheelBuildSoxrDll) {
    if (-not (Test-Path $wheelBuildSoxrDllPath)) {
        $wheelBuildCleanupIssues += "Preexisting wheel-build soxr.dll was not restored: $wheelBuildSoxrDllPath"
    }
    if (Test-Path $wheelBuildSoxrDllBackupPath) {
        $wheelBuildCleanupIssues += "Temporary wheel-build soxr.dll backup was not removed after restore: $wheelBuildSoxrDllBackupPath"
    }
}
elseif (Test-Path $wheelBuildSoxrDllPath) {
    $wheelBuildCleanupIssues += "Temporary wheel-build soxr.dll staging was not removed: $wheelBuildSoxrDllPath"
}
if ($wheelBuildCleanupIssues.Count -gt 0) {
    if ($null -ne $wheelBuildError) {
        $wheelBuildCleanupIssues += "Original wheel build failure: $($wheelBuildError.Exception.Message)"
    }
    throw ($wheelBuildCleanupIssues -join [Environment]::NewLine)
}
if ($null -ne $wheelBuildError) {
    $PSCmdlet.ThrowTerminatingError($wheelBuildError)
}

$wheelPath = Get-ChildItem -Path $wheelOutputDir -Filter "soxr-$SoxrVersion-*.whl" | Select-Object -First 1
if ($null -eq $wheelPath) {
    throw "Custom system-linked soxr wheel was not produced in $wheelOutputDir"
}
Invoke-External -FilePath $pythonCommand -ArgumentList @(
    "-m",
    "puripuly_heart.release_evidence.native_distribution",
    "finalize-soxr-wheel",
    "--wheel",
    $wheelPath.FullName,
    "--dll",
    $libsoxrBuiltDllPath,
    "--output",
    $wheelPath.FullName
)

Write-Host "Extracting wheel contents for runtime staging..."
Invoke-External -FilePath $pythonCommand -ArgumentList @(
    "-c",
    "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])",
    $wheelPath.FullName,
    $wheelExtractDir
)

$stagedSoxrExtensionPath = Join-Path $runtimeStageDir "soxr_ext.pyd"
$stagedSoxrDllPath = Join-Path $runtimeStageDir "soxr.dll"
$wheelExtensionPath = Join-Path $wheelExtractDir "soxr\soxr_ext.pyd"
$wheelDllPath = Join-Path $wheelExtractDir "soxr\soxr.dll"
if (-not (Test-Path $wheelExtensionPath)) {
    throw "Extracted wheel is missing soxr/soxr_ext.pyd: $wheelExtensionPath"
}
if (-not (Test-Path $wheelDllPath)) {
    throw "Extracted wheel is missing soxr/soxr.dll: $wheelDllPath"
}

Copy-Item -Path $wheelExtensionPath -Destination $stagedSoxrExtensionPath -Force
Copy-Item -Path $wheelDllPath -Destination $stagedSoxrDllPath -Force

Copy-Item -Path $soxrSdistPath -Destination (Join-Path $sourceBundleStageDir ([System.IO.Path]::GetFileName($soxrSdistPath))) -Force
Copy-Item -Path $libsoxrSourcePath -Destination (Join-Path $sourceBundleStageDir ([System.IO.Path]::GetFileName($libsoxrSourcePath))) -Force
Copy-Item -Path $zeroconfSdistPath -Destination (Join-Path $sourceBundleStageDir ([System.IO.Path]::GetFileName($zeroconfSdistPath))) -Force

$sourceBundleManifestPath = Join-Path $sourceBundleStageDir "manifest.json"
$sourceBundleManifest = [ordered]@{
    bundle_scope = "redistributable LGPL source inputs for the PuriPuly Heart Windows release"
    soxr_version = $SoxrVersion
    libsoxr_version = $LibsoxrVersion
    zeroconf_version = $ZeroconfVersion
    packaged_runtime_relative_dir = $PackagedRuntimeRelativeDir
    wheel = [ordered]@{
        filename = $wheelPath.Name
        sha256 = (Get-FileSha256 -Path $wheelPath.FullName)
        linkage = "system-linked"
    }
    build = [ordered]@{
        python = $pythonVersion
        cmake = $cmakeVersion
        pip = $pipVersion
        backends = $buildBackendVersions
        compiler = "$compilerId $compilerVersion"
        generator = $cmakeGenerator
        windows_sdk = $windowsSdkVersion
        python_soxr_patch = [ordered]@{
            filename = $SoxrBuildPatchName
            sha256 = (Get-FileSha256 -Path $soxrBuildPatchPath)
            apply_from = "soxr-1.1.0 source root"
            command = "git apply python-soxr-1.1.0-release-input.patch"
        }
        libsoxr_cmake_arguments = @(
            "-DBUILD_SHARED_LIBS=ON",
            "-DBUILD_TESTS=OFF",
            "-DBUILD_EXAMPLES=OFF",
            "-DWITH_OPENMP=OFF",
            "-DWITH_LSR_BINDINGS=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"
        )
        python_soxr_wheel_arguments = @(
            "--no-build-isolation",
            "--no-deps",
            "--config-settings=cmake.define.USE_SYSTEM_LIBSOXR=ON"
        )
    }
    sources = @(
        [ordered]@{
            name = "python-soxr"
            filename = [System.IO.Path]::GetFileName($soxrSdistPath)
            url = $SoxrSdistUrl
            sha256 = $actualSoxrSdistSha256
            license = "LGPL-2.1-or-later"
            modifications = @($SoxrBuildPatchName)
        },
        [ordered]@{
            name = "libsoxr"
            filename = [System.IO.Path]::GetFileName($libsoxrSourcePath)
            url = $LibsoxrSourceUrl
            sha256 = $libsoxrSourceSha256
            license = "LGPL-2.1-or-later"
            modifications = @()
        },
        [ordered]@{
            name = "python-zeroconf"
            filename = [System.IO.Path]::GetFileName($zeroconfSdistPath)
            url = $ZeroconfSdistUrl
            sha256 = $actualZeroconfSdistSha256
            license = "LGPL-2.1-or-later"
            modifications = @()
        }
    )
}
$sourceBundleManifest | ConvertTo-Json -Depth 6 | Set-Content -Path $sourceBundleManifestPath -Encoding utf8

Write-Host "Creating third-party source bundle..."
Compress-Archive -Path (Join-Path $sourceBundleStageDir "*") -DestinationPath $SourceBundlePath -Force

$releaseInputsManifest = [ordered]@{
    soxr_version = $SoxrVersion
    wheel_path = (Get-RepoRelativePath -Path $wheelPath.FullName)
    third_party_source_bundle_path = (Get-RepoRelativePath -Path $SourceBundlePath)
    runtime = [ordered]@{
        packaged_relative_dir = $PackagedRuntimeRelativeDir
        extension_path = (Get-RepoRelativePath -Path $stagedSoxrExtensionPath)
        dll_path = (Get-RepoRelativePath -Path $stagedSoxrDllPath)
    }
}
$releaseInputsManifest | ConvertTo-Json -Depth 6 | Set-Content -Path $ManifestPath -Encoding utf8

Write-Host "Prepared soxr release inputs:"
Write-Host "- Manifest: $ManifestPath"
Write-Host "- Wheel: $($wheelPath.FullName)"
Write-Host "- Runtime DLL: $stagedSoxrDllPath"
Write-Host "- Source bundle: $SourceBundlePath"
