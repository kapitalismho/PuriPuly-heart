[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$TargetDir,
    [Parameter(Mandatory = $true)]
    [string]$OutputDir
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
if ([string]::IsNullOrWhiteSpace($env:VULKAN_SDK)) { throw "VULKAN_SDK is required" }
foreach ($relative in @("Include\vulkan\vulkan.h", "Lib\vulkan-1.lib", "Bin\glslc.exe")) {
    if (-not (Test-Path -LiteralPath (Join-Path $env:VULKAN_SDK $relative) -PathType Leaf)) {
        throw "Vulkan SDK input is missing: $relative"
    }
}
$TargetDir = [System.IO.Path]::GetFullPath($TargetDir)
$OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
$buildId = [Guid]::NewGuid().ToString("N")
$freshTarget = Join-Path $TargetDir $buildId
New-Item -ItemType Directory -Path $freshTarget | Out-Null
$environment = @{}
$names = @(Get-ChildItem Env: | Where-Object {
    $_.Name -match '^(CMAKE.*|TRANSCRIBE_.*|GGML_.*|(?:(?:HOST|TARGET)_)?(?:CFLAGS.*|CXXFLAGS.*|CPPFLAGS.*|LDFLAGS.*|CC.*|CXX.*|CMAKE.*)|CL|_CL_|LINK|_LINK_|RUSTFLAGS|CARGO_ENCODED_RUSTFLAGS|RUSTC.*|CARGO_BUILD_RUST.*|CARGO_BUILD_TARGET|CARGO_TARGET_.*)$'
} | ForEach-Object { $_.Name })
$names += @("TRANSCRIBE_CMAKE_ARGS", "CMAKE_GENERATOR", "CMAKE_PREFIX_PATH", "RUSTFLAGS")
foreach ($name in ($names | Sort-Object -Unique)) {
    $environment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
    Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
}
Push-Location $repoRoot
try {
    $env:TRANSCRIBE_CMAKE_ARGS = "-DTRANSCRIBE_USE_SYSTEM_BLAS=OFF -DTRANSCRIBE_USE_OPENMP=OFF -DCMAKE_SUPPRESS_REGENERATION=ON"
    $env:CMAKE_GENERATOR = "Visual Studio 17 2022"
    $env:CMAKE_PREFIX_PATH = $env:VULKAN_SDK
    $env:RUSTFLAGS = "-C target-cpu=x86-64"
    & cargo build --manifest-path native/gpu_worker/Cargo.toml --locked --release --bin PuriPulyHeartGpuWorker --target x86_64-pc-windows-msvc --target-dir $freshTarget
    if ($LASTEXITCODE -ne 0) { throw "GPU worker Cargo build failed: $LASTEXITCODE" }
    $release = Join-Path $freshTarget "x86_64-pc-windows-msvc\release"
    $sources = Get-Content -LiteralPath (Join-Path $release "gpu-worker-runtime-sources.json") -Raw | ConvertFrom-Json
    if ($sources.schema_version -ne 1) { throw "Unsupported GPU runtime source inventory" }
    $files = [ordered]@{}
    $worker = Join-Path $release "PuriPulyHeartGpuWorker.exe"
    $files["PuriPulyHeartGpuWorker.exe"] = $worker
    foreach ($property in $sources.files.PSObject.Properties) {
        $source = [string]$property.Value
        if (-not (Test-Path -LiteralPath $source -PathType Leaf)) { throw "Missing current-build runtime: $source" }
        $files[$property.Name] = $source
    }
    foreach ($name in @("transcribe.dll", "ggml.dll", "ggml-base.dll", "ggml-vulkan.dll", "ggml-cpu-x64.dll")) {
        if (-not $files.Contains($name)) { throw "Missing required GPU runtime: $name" }
    }
    $actual = @(Get-ChildItem -LiteralPath $release -Filter "*.dll" -File | ForEach-Object { $_.Name })
    $expected = @($sources.files.PSObject.Properties.Name)
    if (@(Compare-Object $actual $expected).Count -ne 0) { throw "Current-build runtime inventory differs from installed DLLs" }
    $linkManifest = Get-Content -LiteralPath $sources.link_manifest -Raw | ConvertFrom-Json
    if (-not $linkManifest.shared -or [string]::IsNullOrWhiteSpace($linkManifest.module_dir)) {
        throw "Current-build transcribe manifest does not describe dynamic shared backends"
    }
    $installManifest = Join-Path (Split-Path -Parent (Split-Path -Parent $sources.link_manifest)) "build\install_manifest.txt"
    $installedDlls = @(Get-Content -LiteralPath $installManifest | Where-Object {
        [IO.Path]::GetExtension($_) -eq ".dll"
    } | ForEach-Object { [IO.Path]::GetFileName($_) } | Sort-Object -Unique)
    if (@(Compare-Object $installedDlls $expected).Count -ne 0) {
        throw "GPU runtime inventory differs from the current CMake installation manifest"
    }
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    $oldInventory = Join-Path $OutputDir "gpu-worker-runtime.json"
    if (Test-Path -LiteralPath $oldInventory) {
        $old = Get-Content -LiteralPath $oldInventory -Raw | ConvertFrom-Json
        foreach ($entry in $old.files) {
            if ([IO.Path]::GetFileName($entry.path) -ne $entry.path) { throw "Unsafe previous GPU inventory path" }
            Remove-Item -LiteralPath (Join-Path $OutputDir $entry.path) -Force -ErrorAction SilentlyContinue
        }
    }
    $inventory = @()
    foreach ($name in $files.Keys) {
        $source = $files[$name]
        $digest = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash.ToLowerInvariant()
        $destination = Join-Path $OutputDir $name
        Copy-Item -LiteralPath $source -Destination $destination -Force
        if ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant() -ne $digest) {
            throw "GPU runtime copy hash mismatch: $name"
        }
        $inventory += @{ path = $name; sha256 = $digest }
    }
    $stagedNames = @(Get-ChildItem -LiteralPath $OutputDir -File | Where-Object {
        $_.Name -eq "PuriPulyHeartGpuWorker.exe" -or $_.Name -eq "transcribe.dll" -or
        ($_.Name -like "ggml*.dll") -or $files.Contains($_.Name)
    } | ForEach-Object { $_.Name })
    if (@(Compare-Object $stagedNames @($files.Keys)).Count -ne 0) {
        throw "Staged GPU runtime contains stale or missing files"
    }
    $record = [ordered]@{
        schema_version = 1
        build_id = $buildId
        target_dir = $freshTarget
        cpu_selection = "upstream-cpuid"
        files = $inventory
    }
    $record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $oldInventory -Encoding utf8NoBOM
    Write-Host "GPU worker runtime staged from $freshTarget to $OutputDir"
} finally {
    Pop-Location
    foreach ($name in $environment.Keys) {
        if ($null -eq $environment[$name]) {
            Remove-Item -LiteralPath "Env:$name" -ErrorAction SilentlyContinue
        } else {
            [Environment]::SetEnvironmentVariable($name, $environment[$name], "Process")
        }
    }
}
