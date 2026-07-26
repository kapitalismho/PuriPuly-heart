[CmdletBinding()]
param(
    [Parameter()]
    [string]$OutputPath = "build/flet/flet-windows.zip"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$FletVersion = "0.86.1"
$ExpectedSha256 = "2cf0865b31bd0e394a24a6c2d270e084cf9dad9c711e0b5d0cf9fa9bfac31e14"
$DownloadUrl = "https://github.com/flet-dev/flet/releases/download/v$FletVersion/flet-windows.zip"
$ResolvedOutputPath = [System.IO.Path]::GetFullPath((Join-Path $PWD $OutputPath))
$OutputDirectory = Split-Path -Parent $ResolvedOutputPath

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

if (Test-Path -LiteralPath $ResolvedOutputPath) {
    $ExistingSha256 = (Get-FileHash -LiteralPath $ResolvedOutputPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ExistingSha256 -eq $ExpectedSha256) {
        Write-Output $ResolvedOutputPath
        exit 0
    }
}

$TemporaryPath = Join-Path $OutputDirectory "flet-windows.zip.download"
try {
    Invoke-WebRequest -Uri $DownloadUrl -OutFile $TemporaryPath
    $ActualSha256 = (Get-FileHash -LiteralPath $TemporaryPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($ActualSha256 -ne $ExpectedSha256) {
        throw "Flet Windows runtime checksum mismatch: expected $ExpectedSha256, found $ActualSha256"
    }
    Move-Item -LiteralPath $TemporaryPath -Destination $ResolvedOutputPath -Force
} finally {
    if (Test-Path -LiteralPath $TemporaryPath) {
        Remove-Item -LiteralPath $TemporaryPath -Force
    }
}

Write-Output $ResolvedOutputPath
