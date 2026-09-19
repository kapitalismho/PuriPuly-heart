[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ToolPython,

    [Parameter(Mandatory = $true)]
    [string]$PinnedInputRoot,

    [Parameter(Mandatory = $true)]
    [string]$PythonEmbedArchive,

    [Parameter()]
    [string]$SoxrBuildEnvironment = ".venv",

    [Parameter()]
    [string]$BuildRoot = "C:\d177\native-integration",

    [Parameter()]
    [string]$OutputDir = "dist\native\PuriPulyHeart"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,
        [Parameter()]
        [string[]]$ArgumentList = @(),
        [Parameter()]
        [string]$WorkingDirectory = $PWD
    )
    Push-Location $WorkingDirectory
    try {
        & $FilePath @ArgumentList
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($ArgumentList -join ' ')"
        }
    } finally {
        Pop-Location
    }
}

function Resolve-OneFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Root,
        [Parameter(Mandatory = $true)]
        [string]$Name
    )
    $direct = Join-Path $Root $Name
    if (Test-Path -LiteralPath $direct -PathType Leaf) {
        return (Get-Item -LiteralPath $direct).FullName
    }
    $matches = @(Get-ChildItem -LiteralPath $Root -Filter $Name -File -Recurse)
    if ($matches.Count -ne 1) {
        throw "Expected exactly one $Name below $Root; found $($matches.Count)"
    }
    return $matches[0].FullName
}

function Copy-Tree {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Source,
        [Parameter(Mandatory = $true)]
        [string]$Destination
    )
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
    Copy-Item -Path (Join-Path $Source "*") -Destination $Destination -Recurse -Force
}

function Test-PathWithin {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Candidate,
        [Parameter(Mandatory = $true)]
        [string]$Root
    )
    $candidatePath = [System.IO.Path]::GetFullPath($Candidate)
    $rootPath = [System.IO.Path]::GetFullPath($Root).TrimEnd(
        [System.IO.Path]::DirectorySeparatorChar,
        [System.IO.Path]::AltDirectorySeparatorChar
    )
    return $candidatePath.Equals($rootPath, [System.StringComparison]::OrdinalIgnoreCase) -or
        $candidatePath.StartsWith(
            $rootPath + [System.IO.Path]::DirectorySeparatorChar,
            [System.StringComparison]::OrdinalIgnoreCase
        )
}

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$BuildRoot = [System.IO.Path]::GetFullPath($BuildRoot)
$SoxrBuildEnvironmentPath = if ([System.IO.Path]::IsPathRooted($SoxrBuildEnvironment)) {
    [System.IO.Path]::GetFullPath($SoxrBuildEnvironment)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $repoRoot $SoxrBuildEnvironment))
}
$OutputDir = [System.IO.Path]::GetFullPath((Join-Path $repoRoot $OutputDir))
$allowedBuildRoot = [System.IO.Path]::GetFullPath("C:\d177\native-integration")
$allowedScratchRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot ".tmp\issue-177-native-integration"))
$allowedOutputRoot = [System.IO.Path]::GetFullPath((Join-Path $repoRoot "dist\native"))
if (-not ((Test-PathWithin -Candidate $BuildRoot -Root $allowedBuildRoot) -or (Test-PathWithin -Candidate $BuildRoot -Root $allowedScratchRoot))) {
    throw "BuildRoot must be below $allowedBuildRoot or $allowedScratchRoot"
}
if (-not (Test-PathWithin -Candidate $OutputDir -Root $allowedOutputRoot)) {
    throw "OutputDir must be below $allowedOutputRoot"
}
foreach ($required in @($ToolPython, $PinnedInputRoot, $PythonEmbedArchive, $SoxrBuildEnvironmentPath)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required native build input not found: $required"
    }
}

$layoutPath = Join-Path $repoRoot "native\windows_host\artifact-layout.json"
$inputSpecPath = Join-Path $repoRoot "native\windows_host\upstream-inputs.json"
$overlayRoot = Join-Path $repoRoot "native\windows_host\template"
$pythonBootstrapPath = Join-Path $repoRoot "native\windows_host\python_bootstrap.py.in"
$inputIndex = Join-Path $BuildRoot "input-index"
$templateRoot = Join-Path $BuildRoot "template"
$fixtureRoot = Join-Path $BuildRoot "fixture"
$fletOutput = Join-Path $BuildRoot "flet-output"
$fletCache = Join-Path $BuildRoot "flet-cache"
$flutterSdkRoot = Join-Path $BuildRoot "flutter-sdk"
$pythonEmbedRoot = Join-Path $BuildRoot "official-python-embed"
$requirementsPath = Join-Path $BuildRoot "requirements-native.txt"
$requirementsExportPath = Join-Path $BuildRoot "requirements-export.txt"
$soxrRoot = Join-Path $BuildRoot "soxr-release-inputs"
$llamaRoot = Join-Path $BuildRoot "llama.cpp-b10423"
$llamaCache = Join-Path $BuildRoot "llama-cache"
$artifactRoot = Join-Path $BuildRoot "artifact"
$evidenceRoot = Join-Path $BuildRoot "evidence"
$overlayTarget = Join-Path $BuildRoot "overlay-target"
$gpuTarget = Join-Path $BuildRoot "gpu-target"

New-Item -ItemType Directory -Path $BuildRoot -Force | Out-Null
foreach ($transient in @(
    $inputIndex,
    $templateRoot,
    $fixtureRoot,
    $fletOutput,
    $pythonEmbedRoot,
    $soxrRoot,
    $llamaRoot,
    $artifactRoot,
    $evidenceRoot,
    $overlayTarget,
    $gpuTarget
)) {
    Remove-Item -LiteralPath $transient -Recurse -Force -ErrorAction SilentlyContinue
}
New-Item -ItemType Directory -Path $inputIndex, $evidenceRoot -Force | Out-Null
$spec = Get-Content -LiteralPath $inputSpecPath -Raw -Encoding utf8 | ConvertFrom-Json
foreach ($property in $spec.inputs.PSObject.Properties) {
    $filename = $property.Name
    $source = if ($filename -eq "python-3.14.7-embed-amd64.zip") { $PythonEmbedArchive } else { Resolve-OneFile -Root $PinnedInputRoot -Name $filename }
    try {
        New-Item -ItemType HardLink -Path (Join-Path $inputIndex $filename) -Target $source | Out-Null
    } catch {
        Copy-Item -LiteralPath $source -Destination (Join-Path $inputIndex $filename) -Force
    }
}

$previousPythonPath = $env:PYTHONPATH
$env:PYTHONPATH = Join-Path $repoRoot "src"
try {
    Invoke-Checked -FilePath $ToolPython -ArgumentList @(
        "-m", "puripuly_heart.release_evidence.native_distribution", "verify-inputs",
        "--spec", $inputSpecPath, "--input-root", $inputIndex,
        "--output", (Join-Path $evidenceRoot "upstream-inputs.json")
    ) -WorkingDirectory $repoRoot

    Expand-Archive -LiteralPath (Join-Path $inputIndex "flet-build-template.zip") -DestinationPath $templateRoot
    $cookiecutterTemplateRoot = Join-Path $templateRoot "build"
    Invoke-Checked -FilePath $ToolPython -ArgumentList @(
        "-m", "puripuly_heart.release_evidence.native_distribution", "render-template",
        "--template-root", $cookiecutterTemplateRoot,
        "--overlay-root", $overlayRoot,
        "--layout", $layoutPath,
        "--python-bootstrap", $pythonBootstrapPath,
        "--output", (Join-Path $evidenceRoot "template-render.json")
    ) -WorkingDirectory $repoRoot

    $flutterCommand = Join-Path $flutterSdkRoot "flutter\bin\flutter.bat"
    if (-not (Test-Path -LiteralPath $flutterCommand -PathType Leaf)) {
        Remove-Item -LiteralPath $flutterSdkRoot -Recurse -Force -ErrorAction SilentlyContinue
        Expand-Archive -LiteralPath (Join-Path $inputIndex "flutter_windows_3.44.8-stable.zip") -DestinationPath $flutterSdkRoot
    }
    $flutterVersionOutput = (& $flutterCommand --version --machine | Out-String)
    $flutterJsonOffset = $flutterVersionOutput.IndexOf("{")
    if ($flutterJsonOffset -lt 0) {
        throw "Pinned Flutter version output did not contain JSON"
    }
    $flutterVersion = $flutterVersionOutput.Substring($flutterJsonOffset) | ConvertFrom-Json
    if ($flutterVersion.frameworkVersion -ne $spec.versions.flutter -or $flutterVersion.dartSdkVersion -notlike "$($spec.versions.dart)*") {
        throw "Pinned Flutter/Dart identity mismatch"
    }

    $sourceCache = Join-Path $PinnedInputRoot "cache"
    if (-not (Test-Path -LiteralPath $sourceCache -PathType Container)) {
        throw "Pinned Flet cache not found: $sourceCache"
    }
    Copy-Tree -Source $sourceCache -Destination $fletCache
    New-Item -ItemType Directory -Path $fixtureRoot -Force | Out-Null
    Copy-Item -LiteralPath (Join-Path $repoRoot "native\windows_host\product_bootstrap.py") -Destination (Join-Path $fixtureRoot "product_bootstrap.py") -Force
    @"
[project]
name = "puripuly-heart-native-bootstrap"
version = "2.7.0"
requires-python = ">=3.14,<3.15"

[tool.flet]
org = "com.salee"
product = "PuriPuly <3"
company = "salee"
"@ | Set-Content -LiteralPath (Join-Path $fixtureRoot "pyproject.toml") -Encoding utf8

    $env:FLET_CACHE_DIR = $fletCache
    $env:PATH = "$(Join-Path $flutterSdkRoot "flutter\bin");$env:PATH"
    $env:PIP_CACHE_DIR = Join-Path $BuildRoot "pip-cache"
    $env:PYTHONDONTWRITEBYTECODE = "1"
    $env:PYTHONOPTIMIZE = "0"
    Invoke-Checked -FilePath $ToolPython -ArgumentList @(
        "-m", "flet.cli", "build", "windows", $fixtureRoot,
        "--output", $fletOutput,
        "--project", "puripuly-heart",
        "--artifact", "PuriPulyHeart",
        "--product", "PuriPuly <3",
        "--company", "salee",
        "--org", "com.salee",
        "--description", "Real-time multilingual speech translation",
        "--build-version", "2.7.0",
        "--build-number", "0",
        "--module-name", "product_bootstrap",
        "--template", $cookiecutterTemplateRoot,
        "--python-version", "3.14",
        "--no-compile-app",
        "--no-compile-packages",
        "--no-rich-output",
        "--yes"
    ) -WorkingDirectory $repoRoot

    Copy-Tree -Source $fletOutput -Destination $artifactRoot
    Remove-Item -LiteralPath (Join-Path $artifactRoot "app") -Recurse -Force
    Remove-Item -LiteralPath (Join-Path $artifactRoot "site-packages") -Recurse -Force
    New-Item -ItemType Directory -Path (Join-Path $artifactRoot "app"), (Join-Path $artifactRoot "site-packages") | Out-Null
    Expand-Archive -LiteralPath $PythonEmbedArchive -DestinationPath $pythonEmbedRoot
    Copy-Item -LiteralPath (Join-Path $pythonEmbedRoot "python.exe") -Destination (Join-Path $artifactRoot "python.exe") -Force

    $uvCommand = (Get-Command uv -ErrorAction Stop).Source
    Invoke-Checked -FilePath $uvCommand -ArgumentList @(
        "export", "--locked", "--no-dev", "--no-emit-project", "--format", "requirements-txt", "--output-file", $requirementsExportPath
    ) -WorkingDirectory $repoRoot
    Invoke-Checked -FilePath $ToolPython -ArgumentList @(
        "-m", "puripuly_heart.release_evidence.native_distribution", "filter-requirements", $requirementsExportPath, $requirementsPath
    ) -WorkingDirectory $repoRoot
    Invoke-Checked -FilePath $uvCommand -ArgumentList @(
        "pip", "install",
        "--python", $ToolPython,
        "--python-version", "3.14.7",
        "--python-platform", "x86_64-pc-windows-msvc",
        "--require-hashes",
        "--no-deps",
        "--no-build",
        "--target", (Join-Path $artifactRoot "site-packages"),
        "--requirements", $requirementsPath
    ) -WorkingDirectory $repoRoot

    $previousProjectEnvironment = $env:UV_PROJECT_ENVIRONMENT
    $env:UV_PROJECT_ENVIRONMENT = $SoxrBuildEnvironmentPath
    try {
        & (Join-Path $repoRoot "scripts\ci\prepare-soxr-release-inputs.ps1") -OutputRoot $soxrRoot -PackagedRuntimeRelativeDir "site-packages/soxr"
    } finally {
        if ($null -eq $previousProjectEnvironment) {
            Remove-Item Env:UV_PROJECT_ENVIRONMENT -ErrorAction SilentlyContinue
        } else {
            $env:UV_PROJECT_ENVIRONMENT = $previousProjectEnvironment
        }
    }
    $soxrWheel = Resolve-OneFile -Root (Join-Path $soxrRoot "wheel") -Name "soxr-1.1.0-*.whl"
    Invoke-Checked -FilePath $uvCommand -ArgumentList @(
        "pip", "install",
        "--python", $ToolPython,
        "--no-deps",
        "--target", (Join-Path $artifactRoot "site-packages"),
        $soxrWheel
    ) -WorkingDirectory $repoRoot

    Copy-Tree -Source (Join-Path $repoRoot "src\puripuly_heart") -Destination (Join-Path $artifactRoot "app\puripuly_heart")
    Copy-Tree -Source (Join-Path $repoRoot "prompts") -Destination (Join-Path $artifactRoot "app\prompts")
    Get-ChildItem -LiteralPath (Join-Path $artifactRoot "app") -Directory -Filter "__pycache__" -Recurse | Remove-Item -Recurse -Force
    Get-ChildItem -LiteralPath (Join-Path $artifactRoot "app") -File -Filter "*.pyc" -Recurse | Remove-Item -Force
    Copy-Item -LiteralPath (Join-Path $repoRoot "native\windows_host\product_bootstrap.py") -Destination (Join-Path $artifactRoot "app\product_bootstrap.py") -Force
    Copy-Tree -Source (Join-Path $repoRoot "examples\http_extensions") -Destination (Join-Path $artifactRoot "examples\http_extensions")

    Invoke-Checked -FilePath $ToolPython -ArgumentList @(
        "-m", "puripuly_heart.release_evidence.native_distribution",
        "stage-product-metadata",
        "--site-packages", (Join-Path $artifactRoot "site-packages"),
        "--pyproject", (Join-Path $repoRoot "pyproject.toml"),
        "--output", (Join-Path $evidenceRoot "product-metadata.json")
    ) -WorkingDirectory $repoRoot
    $cargoCommand = (Get-Command cargo -ErrorAction Stop).Source
    Invoke-Checked -FilePath $cargoCommand -ArgumentList @(
        "build", "--manifest-path", (Join-Path $repoRoot "native\overlay\Cargo.toml"), "--locked", "--release", "--bin", "PuriPulyHeartOverlay", "--target-dir", $overlayTarget
    ) -WorkingDirectory $repoRoot
    Invoke-Checked -FilePath $cargoCommand -ArgumentList @(
        "build", "--manifest-path", (Join-Path $repoRoot "native\gpu_worker\Cargo.toml"), "--locked", "--release", "--bin", "PuriPulyHeartGpuWorker", "--target-dir", $gpuTarget
    ) -WorkingDirectory $repoRoot
    Copy-Item -LiteralPath (Join-Path $overlayTarget "release\PuriPulyHeartOverlay.exe") -Destination $artifactRoot -Force
    Copy-Item -LiteralPath (Join-Path $gpuTarget "release\PuriPulyHeartGpuWorker.exe") -Destination $artifactRoot -Force
    Copy-Item -LiteralPath (Join-Path $repoRoot "third_party\openvr\win64\openvr_api.dll") -Destination $artifactRoot -Force

    Invoke-Checked -FilePath $ToolPython -ArgumentList @(
        "-m", "puripuly_heart.release_evidence.managed_gemma_distribution", "prepare",
        "--repo-root", $repoRoot, "--cache-dir", $llamaCache, "--output-root", $llamaRoot
    ) -WorkingDirectory $repoRoot
    Copy-Tree -Source $llamaRoot -Destination (Join-Path $artifactRoot "_runtime\llama.cpp-b10423")
    Copy-Tree -Source (Join-Path $repoRoot "third_party\llama.cpp") -Destination (Join-Path $artifactRoot "third_party\llama.cpp")
    $localQwenRoot = Join-Path $artifactRoot "_runtime\local_qwen"
    New-Item -ItemType Directory -Path $localQwenRoot -Force | Out-Null
    foreach ($dll in @("onnxruntime.dll", "onnxruntime_providers_shared.dll")) {
        $source = Resolve-OneFile -Root (Join-Path $artifactRoot "site-packages\onnxruntime") -Name $dll
        Copy-Item -LiteralPath $source -Destination $localQwenRoot -Force
    }

    Copy-Tree -Source (Join-Path $soxrRoot "source-bundle") -Destination (Join-Path $artifactRoot "third_party\soxr")
    Copy-Item -LiteralPath (Join-Path $soxrRoot "PuriPulyHeart-soxr-third-party-source-bundle.zip") -Destination (Join-Path $artifactRoot "third_party\soxr") -Force
    Copy-Item -LiteralPath (Join-Path $repoRoot "src\puripuly_heart\data\licenses\COPYING.LGPL-2.1.txt") -Destination (Join-Path $artifactRoot "third_party\soxr") -Force
    Copy-Tree -Source (Join-Path $repoRoot "third_party\noto-sans-cjk") -Destination (Join-Path $artifactRoot "third_party\noto-sans-cjk")

    Invoke-Checked -FilePath $ToolPython -ArgumentList @(
        "-m", "puripuly_heart.release_evidence.native_distribution", "compile-app",
        "--application-root", (Join-Path $artifactRoot "app"), "--output", (Join-Path $evidenceRoot "bytecode.json")
    ) -WorkingDirectory $repoRoot
    Invoke-Checked -FilePath $ToolPython -ArgumentList @(
        "-m", "puripuly_heart.release_evidence.native_distribution", "validate-target",
        "--target-root", $artifactRoot, "--layout", $layoutPath,
        "--output", (Join-Path $evidenceRoot "target-validation.json")
    ) -WorkingDirectory $repoRoot
    Invoke-Checked -FilePath $ToolPython -ArgumentList @(
        "-m", "puripuly_heart.release_evidence.native_distribution", "validate-compliance",
        "--target-root", $artifactRoot, "--repo-root", $repoRoot,
        "--soxr-manifest", (Join-Path $soxrRoot "manifest.json"),
        "--output", (Join-Path $evidenceRoot "compliance-validation.json")
    ) -WorkingDirectory $repoRoot
    Invoke-Checked -FilePath $ToolPython -ArgumentList @(
        "-m", "puripuly_heart.release_evidence.managed_gemma_distribution",
        "verify-package", $artifactRoot, "--launch"
    ) -WorkingDirectory $repoRoot
    Invoke-Checked -FilePath $ToolPython -ArgumentList @(
        "-m", "puripuly_heart.release_evidence.native_distribution", "manifest",
        "--target-root", $artifactRoot, "--layout", $layoutPath,
        "--provenance", (Join-Path $evidenceRoot "upstream-inputs.json"),
        "--bytecode", (Join-Path $evidenceRoot "bytecode.json"),
        "--output", (Join-Path $artifactRoot "native-artifact-manifest.json")
    ) -WorkingDirectory $repoRoot

    Remove-Item -LiteralPath $OutputDir -Recurse -Force -ErrorAction SilentlyContinue
    New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
    Copy-Item -Path (Join-Path $artifactRoot "*") -Destination $OutputDir -Recurse -Force
} finally {
    if ($null -eq $previousPythonPath) {
        Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
    } else {
        $env:PYTHONPATH = $previousPythonPath
    }
}

Write-Host "Experimental native artifact: $OutputDir"
Write-Host "Build evidence: $evidenceRoot"
