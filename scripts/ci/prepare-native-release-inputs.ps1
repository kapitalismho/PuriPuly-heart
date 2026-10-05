[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ToolPython,

    [Parameter(Mandatory = $true)]
    [string]$PinnedInputRoot
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
Set-StrictMode -Version Latest

function Test-PinnedInput {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [psobject]$Expected
    )
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }
    if ((Get-Item -LiteralPath $Path).Length -ne $Expected.bytes) {
        return $false
    }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() -eq $Expected.sha256
}

foreach ($absolutePath in @($ToolPython, $PinnedInputRoot)) {
    if (-not [System.IO.Path]::IsPathFullyQualified($absolutePath)) {
        throw "Native input preparation requires absolute paths: $absolutePath"
    }
}
if (-not (Test-Path -LiteralPath $ToolPython -PathType Leaf)) {
    throw "ToolPython not found: $ToolPython"
}

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$ToolPython = [System.IO.Path]::GetFullPath($ToolPython)
$PinnedInputRoot = [System.IO.Path]::GetFullPath($PinnedInputRoot)
$inputSpecPath = Join-Path $repoRoot "native\windows_host\upstream-inputs.json"
$spec = Get-Content -LiteralPath $inputSpecPath -Raw -Encoding utf8 | ConvertFrom-Json
$cacheRoot = Join-Path $PinnedInputRoot "cache"
New-Item -ItemType Directory -Path $PinnedInputRoot, $cacheRoot -Force | Out-Null

foreach ($property in $spec.inputs.PSObject.Properties) {
    $filename = $property.Name
    $expected = $property.Value
    $destination = Join-Path $PinnedInputRoot $filename
    if (Test-PinnedInput -Path $destination -Expected $expected) {
        Write-Host "Verified cached native input: $filename"
    } else {
        $temporaryPath = "$destination.download"
        try {
            Write-Host "Downloading pinned native input: $filename"
            Invoke-WebRequest -Uri $expected.url -OutFile $temporaryPath -ErrorAction Stop
            if (-not (Test-PinnedInput -Path $temporaryPath -Expected $expected)) {
                throw "Pinned native input mismatch: $filename; expected $($expected.bytes) bytes and SHA256 $($expected.sha256)"
            }
            Move-Item -LiteralPath $temporaryPath -Destination $destination -Force
        } finally {
            if (Test-Path -LiteralPath $temporaryPath) {
                Remove-Item -LiteralPath $temporaryPath -Force
            }
        }
    }

    $cacheProperty = $expected.PSObject.Properties["cache_path"]
    if ($null -ne $cacheProperty) {
        $cachePath = Join-Path $cacheRoot $cacheProperty.Value
        if (Test-PinnedInput -Path $cachePath -Expected $expected) {
            Write-Host "Verified Flet cache input: $($cacheProperty.Value)"
        } else {
            New-Item -ItemType Directory -Path (Split-Path -Parent $cachePath) -Force | Out-Null
            $temporaryPath = "$cachePath.download"
            try {
                Copy-Item -LiteralPath $destination -Destination $temporaryPath -Force
                if (-not (Test-PinnedInput -Path $temporaryPath -Expected $expected)) {
                    throw "Pinned Flet cache input mismatch: $cachePath"
                }
                Move-Item -LiteralPath $temporaryPath -Destination $cachePath -Force
            } finally {
                if (Test-Path -LiteralPath $temporaryPath) {
                    Remove-Item -LiteralPath $temporaryPath -Force
                }
            }
        }
    }
}

$inputIndex = Join-Path $PinnedInputRoot (".verify-native-inputs-" + [guid]::NewGuid().ToString("N"))
$previousPythonPath = $env:PYTHONPATH
New-Item -ItemType Directory -Path $inputIndex | Out-Null
try {
    foreach ($property in $spec.inputs.PSObject.Properties) {
        $cacheProperty = $property.Value.PSObject.Properties["cache_path"]
        $source = if ($null -ne $cacheProperty) {
            Join-Path $cacheRoot $cacheProperty.Value
        } else {
            Join-Path $PinnedInputRoot $property.Name
        }
        $indexedPath = Join-Path $inputIndex $property.Name
        try {
            New-Item -ItemType HardLink -Path $indexedPath -Target $source | Out-Null
        } catch {
            Copy-Item -LiteralPath $source -Destination $indexedPath -Force
        }
    }

    $env:PYTHONPATH = Join-Path $repoRoot "src"
    Push-Location $repoRoot
    try {
        & $ToolPython -m puripuly_heart.release_evidence.native_distribution verify-inputs --spec $inputSpecPath --input-root $inputIndex --output (Join-Path $PinnedInputRoot "verified-inputs.json")
        if ($LASTEXITCODE -ne 0) {
            throw "Native upstream input verification failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
} finally {
    $env:PYTHONPATH = $previousPythonPath
    Remove-Item -LiteralPath $inputIndex -Recurse -Force
}

Write-Host "Prepared pinned native inputs and Flet cache: $PinnedInputRoot"
Write-Output $PinnedInputRoot
