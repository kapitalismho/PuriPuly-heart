[CmdletBinding()]
param([Parameter(Mandatory = $true)][string]$EnvironmentDir)
$ErrorActionPreference = 'Stop'
$EnvironmentDir = [IO.Path]::GetFullPath($EnvironmentDir)
& uv venv --python 3.12 $EnvironmentDir
if ($LASTEXITCODE -ne 0) { throw 'Converter Python 3.12 environment creation failed' }
$Python = Join-Path $EnvironmentDir 'Scripts/python.exe'
& uv pip sync --python $Python --require-hashes (Join-Path $PSScriptRoot 'conversion-requirements.lock')
if ($LASTEXITCODE -ne 0) { throw 'Hash-locked converter dependencies failed' }
Write-Output $Python
