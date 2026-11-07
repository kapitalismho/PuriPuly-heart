[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PinnedInputRoot
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"
Set-StrictMode -Version Latest

foreach ($absolutePath in @($PinnedInputRoot)) {
    if (-not [System.IO.Path]::IsPathFullyQualified($absolutePath)) {
        throw "Native input preparation requires absolute paths: $absolutePath"
    }
}

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$PinnedInputRoot = [System.IO.Path]::GetFullPath($PinnedInputRoot)
$inputSpecPath = Join-Path $repoRoot "native\windows_host\upstream-inputs.json"
$spec = Get-Content -LiteralPath $inputSpecPath -Raw -Encoding utf8 | ConvertFrom-Json
$cacheRoot = Join-Path $PinnedInputRoot "cache"
New-Item -ItemType Directory -Path $PinnedInputRoot, $cacheRoot -Force | Out-Null

foreach ($property in $spec.inputs.PSObject.Properties) {
    $filename = $property.Name
    $expected = $property.Value
    $destination = Join-Path $PinnedInputRoot $filename
    if (Test-Path -LiteralPath $destination -PathType Leaf) {
        Write-Host "Using cached native input: $filename"
    } else {
        $temporaryPath = "$destination.download"
        try {
            Write-Host "Downloading pinned native input: $filename"
            Invoke-WebRequest -Uri $expected.url -OutFile $temporaryPath -ErrorAction Stop
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
        if (Test-Path -LiteralPath $cachePath -PathType Leaf) {
            Write-Host "Using Flet cache input: $($cacheProperty.Value)"
        } else {
            New-Item -ItemType Directory -Path (Split-Path -Parent $cachePath) -Force | Out-Null
            $temporaryPath = "$cachePath.download"
            try {
                Copy-Item -LiteralPath $destination -Destination $temporaryPath -Force
                Move-Item -LiteralPath $temporaryPath -Destination $cachePath -Force
            } finally {
                if (Test-Path -LiteralPath $temporaryPath) {
                    Remove-Item -LiteralPath $temporaryPath -Force
                }
            }
        }
    }
}

Write-Host "Prepared pinned native inputs and Flet cache: $PinnedInputRoot"
Write-Output $PinnedInputRoot
