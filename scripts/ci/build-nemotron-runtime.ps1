[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [string]$CacheDir = (Join-Path $env:LOCALAPPDATA 'PuriPulyHeart/nemotron-build'),
    [string]$Python = 'python',
    [int]$Jobs = 4,
    [string]$VulkanSdk = $env:VULKAN_SDK
)
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
if ($Jobs -lt 1) { throw 'Jobs must be positive' }
$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$OutputDir = [IO.Path]::GetFullPath($OutputDir)
$CacheDir = [IO.Path]::GetFullPath($CacheDir)
$Prepare = Join-Path $RepoRoot 'native/nemotron/prepare.py'
$Prepared = & $Python $Prepare prepare --repo $RepoRoot --cache $CacheDir
if ($LASTEXITCODE -ne 0) { throw 'Pinned source preparation failed' }
$Trees = $Prepared | ConvertFrom-Json
$SourceDir = $Trees.source
$VcpkgRoot = $Trees.vcpkg
$VsWhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
$VsPath = & $VsWhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $VsPath) { throw 'MSVC x64 build tools unavailable' }
$VcVars = Join-Path $VsPath 'VC/Auxiliary/Build/vcvars64.bat'
cmd /c "`"$VcVars`" >NUL 2>&1 && set" | ForEach-Object {
    if ($_ -match '^([^=]+)=(.*)$') { Set-Item -Path "env:$($matches[1])" -Value $matches[2] }
}
if (-not $VulkanSdk) { throw 'Explicit Vulkan SDK required' }
foreach ($Required in @('Bin/glslc.exe', 'Lib/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake')) {
    if (-not (Test-Path (Join-Path $VulkanSdk $Required))) { throw "Missing Vulkan SDK file $Required" }
}
$env:VCPKG_DISABLE_METRICS = '1'
$env:VCPKG_MAX_CONCURRENCY = "$Jobs"
$VcpkgExe = Join-Path $VcpkgRoot 'vcpkg.exe'
if (-not (Test-Path $VcpkgExe)) {
    & (Join-Path $VcpkgRoot 'bootstrap-vcpkg.bat') -disableMetrics
    if ($LASTEXITCODE -ne 0) { throw 'Pinned vcpkg bootstrap failed' }
}
$Dependencies = Join-Path $CacheDir 'installed'
& $VcpkgExe install sentencepiece:x64-windows-static-md --classic "--x-install-root=$Dependencies" "--vcpkg-root=$VcpkgRoot"
if ($LASTEXITCODE -ne 0) { throw 'Pinned SentencePiece dependencies failed' }
$BuildDir = Join-Path $CacheDir 'build'
$CMakeArgs = @('-S', $SourceDir, '-B', $BuildDir, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
    "-DCMAKE_INSTALL_PREFIX=$OutputDir", "-DCMAKE_PREFIX_PATH=$Dependencies/x64-windows-static-md",
    '-DCMAKE_C_FLAGS=/utf-8', '-DCMAKE_CXX_FLAGS=/utf-8 /EHsc',
    '-DNEMO_SPEECH_BUILD_ASR=OFF', '-DNEMO_SPEECH_BUILD_DIAR=ON', '-DNEMO_SPEECH_BUILD_TTS=OFF',
    '-DNEMO_SPEECH_BUILD_NMT=OFF', '-DNEMO_SPEECH_BUILD_S2S=OFF', '-DNEMO_SPEECH_BUILD_CLI=OFF',
    '-DNEMO_SPEECH_BUILD_MIC_CAPTURE=OFF', '-DNEMO_SPEECH_BUILD_HTTP=OFF', '-DNEMO_SPEECH_BUILD_GRPC=OFF',
    '-DNEMO_SPEECH_BUILD_TESTS=OFF', '-DBUILD_TESTING=OFF', '-DNEMO_SPEECH_BUILD_EXAMPLES=OFF',
    '-DNEMO_SPEECH_BUILD_TOOLS=OFF', '-DNEMO_SPEECH_WITH_NORM=OFF', '-DNEMO_SPEECH_WITH_FLASHLIGHT=OFF',
    '-DNEMO_SPEECH_GGML_PATCHED=OFF', '-DGGML_CUDA=OFF', '-DGGML_VULKAN=ON', '-DGGML_METAL=OFF',
    '-DGGML_BLAS=OFF', '-DGGML_NATIVE=OFF', '-DGGML_OPENMP=OFF',
    "-DVulkan_GLSLC_EXECUTABLE=$VulkanSdk/Bin/glslc.exe", "-DSPIRV-Headers_DIR=$VulkanSdk/Lib/cmake/SPIRV-Headers")
& cmake @CMakeArgs
if ($LASTEXITCODE -ne 0) { throw 'Nemotron configure failed' }
& cmake --build $BuildDir --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw 'Nemotron build failed' }
& cmake --install $BuildDir
if ($LASTEXITCODE -ne 0) { throw 'Nemotron install failed' }
$Notices = Join-Path $OutputDir 'share/licenses/nemo-speech/third_party'
Get-ChildItem (Join-Path $Dependencies 'x64-windows-static-md/share') -Directory | ForEach-Object {
    $Copyright = Join-Path $_.FullName 'copyright'
    if (Test-Path $Copyright) {
        $Dest = Join-Path $Notices $_.Name
        New-Item -ItemType Directory -Force -Path $Dest | Out-Null
        Copy-Item $Copyright (Join-Path $Dest 'copyright') -Force
    }
}
& $Python $Prepare record --repo $RepoRoot --output $OutputDir --build $BuildDir
if ($LASTEXITCODE -ne 0) { throw 'Nemotron provenance recording failed' }
Write-Output $OutputDir
