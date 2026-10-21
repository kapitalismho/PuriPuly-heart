[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ManifestPath,

    [Parameter(Mandatory = $true)]
    [string]$AppDataRoot,

    [Parameter(Mandatory = $true)]
    [ValidateSet("huggingface", "modelscope", "hf_mirror")]
    [string]$SelectedSource,

    [switch]$Reinstall
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$DefaultInstalledManifestFilename = "installed-manifest.json"

function Read-JsonObject {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "JSON file not found: $Path"
    }

    try {
        return (Get-Content -LiteralPath $Path -Raw -Encoding UTF8) | ConvertFrom-Json
    } catch {
        throw "Invalid JSON file: $Path"
    }
}

function Get-NamedPropertyValue {
    param(
        [Parameter(Mandatory = $true)]
        $InputObject,

        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    if ($null -eq $InputObject) {
        throw "Object is null when resolving property '$Name'"
    }

    if ($InputObject -is [System.Collections.IDictionary]) {
        if (-not $InputObject.Contains($Name)) {
            throw "Required property '$Name' was not found"
        }
        return $InputObject[$Name]
    }

    $property = $InputObject.PSObject.Properties[$Name]
    if ($null -eq $property) {
        throw "Required property '$Name' was not found"
    }

    return $property.Value
}

function Get-InstalledManifestFilename {
    param(
        [Parameter(Mandatory = $true)]
        $Manifest
    )

    $property = $Manifest.PSObject.Properties["installed_manifest_filename"]
    if ($null -ne $property -and -not [string]::IsNullOrWhiteSpace([string]$property.Value)) {
        return [string]$property.Value
    }

    return $DefaultInstalledManifestFilename
}

function Resolve-AssetRemotePath {
    param(
        [Parameter(Mandatory = $true)]
        $Asset,

        [Parameter(Mandatory = $true)]
        [string]$SourceName
    )

    $overridesProperty = $Asset.PSObject.Properties["source_path_overrides"]
    if ($null -ne $overridesProperty -and $null -ne $overridesProperty.Value) {
        $overrides = $overridesProperty.Value
        if ($overrides -is [System.Collections.IDictionary]) {
            if ($overrides.Contains($SourceName) -and -not [string]::IsNullOrWhiteSpace([string]$overrides[$SourceName])) {
                return [string]$overrides[$SourceName]
            }
        } else {
            $override = $overrides.PSObject.Properties[$SourceName]
            if ($null -ne $override -and -not [string]::IsNullOrWhiteSpace([string]$override.Value)) {
                return [string]$override.Value
            }
        }
    }

    return [string]$Asset.relative_path
}

function Get-InstallDir {
    param(
        [Parameter(Mandatory = $true)]
        $Manifest,

        [Parameter(Mandatory = $true)]
        [string]$AppDataRoot
    )

    return (Join-Path -Path (Join-Path -Path $AppDataRoot -ChildPath "models") -ChildPath ([string]$Manifest.install_dirname))
}

function Get-SourceNames {
    param(
        [Parameter(Mandatory = $true)]
        $Manifest
    )

    $sources = $Manifest.sources
    if ($sources -is [System.Collections.IDictionary]) {
        return @($sources.Keys | ForEach-Object { [string]$_ })
    }

    return @($sources.PSObject.Properties | ForEach-Object { [string]$_.Name })
}

function Get-SourceOrder {
    param(
        [Parameter(Mandatory = $true)]
        $Manifest,

        [Parameter(Mandatory = $true)]
        [string]$SelectedSource
    )

    $names = New-Object System.Collections.Generic.List[string]
    if ((Get-SourceNames -Manifest $Manifest) -contains $SelectedSource) {
        $names.Add($SelectedSource)
    }

    foreach ($sourceName in Get-SourceNames -Manifest $Manifest) {
        if (-not $names.Contains($sourceName)) {
            $names.Add($sourceName)
        }
    }

    return @($names)
}

function Write-BomlessJson {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        $Value
    )

    $parent = Split-Path -Path $Path -Parent
    if (-not [string]::IsNullOrWhiteSpace($parent)) {
        [System.IO.Directory]::CreateDirectory($parent) | Out-Null
    }

    $encoding = New-Object System.Text.UTF8Encoding($false)
    $json = $Value | ConvertTo-Json -Depth 16
    [System.IO.File]::WriteAllText($Path, $json, $encoding)
}

function Get-Sha256Hex {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $getFileHashCommand = Get-Command -Name Get-FileHash -ErrorAction SilentlyContinue
    if ($null -ne $getFileHashCommand) {
        return (& $getFileHashCommand -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    }

    $stream = [System.IO.File]::OpenRead($Path)
    try {
        $sha256 = [System.Security.Cryptography.SHA256]::Create()
        try {
            $hashBytes = $sha256.ComputeHash($stream)
            return ([System.BitConverter]::ToString($hashBytes).Replace("-", "").ToLowerInvariant())
        } finally {
            $sha256.Dispose()
        }
    } finally {
        $stream.Dispose()
    }
}

function Validate-InstalledManifest {
    param(
        [Parameter(Mandatory = $true)]
        $Manifest,

        [Parameter(Mandatory = $true)]
        [string]$InstallDir,

        [Parameter(Mandatory = $true)]
        [string]$ExpectedSource
    )

    $installedManifestPath = Join-Path -Path $InstallDir -ChildPath (Get-InstalledManifestFilename -Manifest $Manifest)
    $installedManifest = Read-JsonObject -Path $installedManifestPath

    if ([int]$installedManifest.manifest_version -ne [int]$Manifest.installed_manifest_version) {
        throw "stale installed manifest version"
    }
    if ([string]$installedManifest.model_id -ne [string]$Manifest.model_id) {
        throw "installed manifest model_id does not match"
    }
    if ([string]$installedManifest.engine -ne [string]$Manifest.engine) {
        throw "installed manifest engine does not match"
    }
    if ([string]$installedManifest.install_dirname -ne [string]$Manifest.install_dirname) {
        throw "installed manifest install_dirname does not match"
    }
    if ([string]$installedManifest.selected_source -ne $ExpectedSource) {
        throw "installed manifest selected_source does not match"
    }

    $expectedSourceConfig = Get-NamedPropertyValue -InputObject $Manifest.sources -Name $ExpectedSource
    if ([string]$installedManifest.selected_revision -ne [string](Get-NamedPropertyValue -InputObject $expectedSourceConfig -Name "revision")) {
        throw "stale installed manifest revision"
    }

    return $installedManifest
}

function Validate-LocalSttInstall {
    param(
        [Parameter(Mandatory = $true)]
        $Manifest,

        [Parameter(Mandatory = $true)]
        [string]$InstallDir,

        [Parameter(Mandatory = $true)]
        [string]$ExpectedSource
    )

    if (-not (Test-Path -LiteralPath $InstallDir -PathType Container)) {
        throw "local STT model directory is missing"
    }

    $null = Validate-InstalledManifest -Manifest $Manifest -InstallDir $InstallDir -ExpectedSource $ExpectedSource

    foreach ($asset in @($Manifest.files)) {
        $assetPath = Join-Path -Path $InstallDir -ChildPath ([string]$asset.relative_path)
        if (-not (Test-Path -LiteralPath $assetPath -PathType Leaf)) {
            throw "missing required model file: $($asset.relative_path)"
        }

        $actualHash = Get-Sha256Hex -Path $assetPath
        if ($actualHash -ne ([string]$asset.sha256).ToLowerInvariant()) {
            throw "checksum mismatch for required model file: $($asset.relative_path)"
        }

        $sizeProperty = $asset.PSObject.Properties["size_bytes"]
        if ($null -ne $sizeProperty -and $null -ne $sizeProperty.Value) {
            $actualSize = (Get-Item -LiteralPath $assetPath).Length
            if ([int64]$actualSize -ne [int64]$sizeProperty.Value) {
                throw "size mismatch for required model file: $($asset.relative_path)"
            }
        }
    }

    return $true
}

function Test-InstalledLocalSttInstall {
    param(
        [Parameter(Mandatory = $true)]
        $Manifest,

        [Parameter(Mandatory = $true)]
        [string]$InstallDir,

        [Parameter(Mandatory = $true)]
        [string]$ExpectedSource
    )

    try {
        Validate-LocalSttInstall -Manifest $Manifest -InstallDir $InstallDir -ExpectedSource $ExpectedSource | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Test-InstalledLocalSttInstallForAnySource {
    param(
        [Parameter(Mandatory = $true)]
        $Manifest,

        [Parameter(Mandatory = $true)]
        [string]$InstallDir,

        [Parameter(Mandatory = $true)]
        [string]$SelectedSource
    )

    foreach ($sourceName in Get-SourceOrder -Manifest $Manifest -SelectedSource $SelectedSource) {
        if (Test-InstalledLocalSttInstall -Manifest $Manifest -InstallDir $InstallDir -ExpectedSource $sourceName) {
            return $true
        }
    }

    return $false
}

function Download-SourceIntoStaging {
    param(
        [Parameter(Mandatory = $true)]
        $Manifest,

        [Parameter(Mandatory = $true)]
        [string]$SourceName,

        [Parameter(Mandatory = $true)]
        [string]$StagingDir
    )

    $source = Get-NamedPropertyValue -InputObject $Manifest.sources -Name $SourceName
    [System.IO.Directory]::CreateDirectory($StagingDir) | Out-Null

    foreach ($asset in @($Manifest.files)) {
        $assetPath = Join-Path -Path $StagingDir -ChildPath ([string]$asset.relative_path)
        $assetParent = Split-Path -Path $assetPath -Parent
        if (-not [string]::IsNullOrWhiteSpace($assetParent)) {
            [System.IO.Directory]::CreateDirectory($assetParent) | Out-Null
        }

        $remotePath = Resolve-AssetRemotePath -Asset $asset -SourceName $SourceName
        $downloadUrl = ([string](Get-NamedPropertyValue -InputObject $source -Name "download_url_template")).Replace("{path}", $remotePath)
        try {
            Invoke-WebRequest -Uri $downloadUrl -OutFile $assetPath -UseBasicParsing
        } catch {
            $exception = $_.Exception
            $exceptionType = $exception.GetType().FullName
            $transferFailed = (
                ($exception -is [System.Net.WebException] -and $exception.Status -ne [System.Net.WebExceptionStatus]::RequestCanceled) -or
                $exceptionType -eq "System.Net.Http.HttpRequestException" -or
                $exceptionType -eq "Microsoft.PowerShell.Commands.HttpResponseException"
            )
            if (-not $transferFailed) {
                throw
            }
            throw [System.IO.InvalidDataException]::new("download failed for required model file: $($asset.relative_path)", $exception)
        }

        $downloadHash = Get-Sha256Hex -Path $assetPath
        if ($downloadHash -ne ([string]$asset.sha256).ToLowerInvariant()) {
            throw [System.IO.InvalidDataException]::new("checksum mismatch for required model file: $($asset.relative_path)")
        }

        $sizeProperty = $asset.PSObject.Properties["size_bytes"]
        if ($null -ne $sizeProperty -and $null -ne $sizeProperty.Value) {
            $downloadSize = (Get-Item -LiteralPath $assetPath).Length
            if ([int64]$downloadSize -ne [int64]$sizeProperty.Value) {
                throw [System.IO.InvalidDataException]::new("size mismatch for required model file: $($asset.relative_path)")
            }
        }
    }

    $installedManifest = [ordered]@{
        manifest_version = [int]$Manifest.installed_manifest_version
        model_id = [string]$Manifest.model_id
        engine = [string]$Manifest.engine
        install_dirname = [string]$Manifest.install_dirname
        selected_source = $SourceName
        selected_revision = [string](Get-NamedPropertyValue -InputObject $source -Name "revision")
    }

    $installedManifestPath = Join-Path -Path $StagingDir -ChildPath (Get-InstalledManifestFilename -Manifest $Manifest)
    Write-BomlessJson -Path $installedManifestPath -Value $installedManifest
    Validate-LocalSttInstall -Manifest $Manifest -InstallDir $StagingDir -ExpectedSource $SourceName | Out-Null
}

function Promote-StagingInstall {
    param(
        [Parameter(Mandatory = $true)]
        [string]$StagingDir,

        [Parameter(Mandatory = $true)]
        [string]$InstallDir
    )

    $backupDir = "$InstallDir.backup"
    $installParent = Split-Path -Path $InstallDir -Parent
    if (-not [string]::IsNullOrWhiteSpace($installParent)) {
        [System.IO.Directory]::CreateDirectory($installParent) | Out-Null
    }

    if (Test-Path -LiteralPath $backupDir) {
        Remove-Item -LiteralPath $backupDir -Recurse -Force
    }

    $hadExistingInstall = Test-Path -LiteralPath $InstallDir
    if ($hadExistingInstall) {
        Move-Item -Path $InstallDir -Destination $backupDir -Force
    }

    try {
        Move-Item -Path $StagingDir -Destination $InstallDir -Force
    } catch {
        if (Test-Path -LiteralPath $InstallDir) {
            Remove-Item -LiteralPath $InstallDir -Recurse -Force -ErrorAction SilentlyContinue
        }
        if ($hadExistingInstall -and (Test-Path -LiteralPath $backupDir)) {
            Move-Item -Path $backupDir -Destination $InstallDir -Force
        }
        throw
    }

    if (Test-Path -LiteralPath $backupDir) {
        Remove-Item -LiteralPath $backupDir -Recurse -Force
    }
}

function Install-LocalSttModel {
    param(
        [Parameter(Mandatory = $true)]
        $Manifest,

        [Parameter(Mandatory = $true)]
        [string]$AppDataRoot,

        [Parameter(Mandatory = $true)]
        [string]$SelectedSource,

        [switch]$Reinstall
    )

    $installDir = Get-InstallDir -Manifest $Manifest -AppDataRoot $AppDataRoot
    if (-not $Reinstall.IsPresent -and (Test-InstalledLocalSttInstallForAnySource -Manifest $Manifest -InstallDir $installDir -SelectedSource $SelectedSource)) {
        return
    }

    $modelRoot = Join-Path -Path $AppDataRoot -ChildPath "models"
    [System.IO.Directory]::CreateDirectory($modelRoot) | Out-Null

    $failures = New-Object System.Collections.Generic.List[string]
    foreach ($sourceName in Get-SourceOrder -Manifest $Manifest -SelectedSource $SelectedSource) {
        $stagingDir = Join-Path -Path $modelRoot -ChildPath ("{0}.staging-{1}" -f [string]$Manifest.install_dirname, [guid]::NewGuid().ToString("N"))

        if (Test-Path -LiteralPath $stagingDir) {
            Remove-Item -LiteralPath $stagingDir -Recurse -Force
        }

        try {
            try {
                Download-SourceIntoStaging -Manifest $Manifest -SourceName $sourceName -StagingDir $stagingDir
            } catch [System.IO.InvalidDataException] {
                $failures.Add(("{0}: {1}" -f $sourceName, $_.Exception.Message))
                continue
            }
            Promote-StagingInstall -StagingDir $stagingDir -InstallDir $installDir
            return
        } finally {
            if (Test-Path -LiteralPath $stagingDir) {
                Remove-Item -LiteralPath $stagingDir -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
    }

    throw ($failures -join "; ")
}

$manifest = Read-JsonObject -Path $ManifestPath

try {
    Install-LocalSttModel -Manifest $manifest -AppDataRoot $AppDataRoot -SelectedSource $SelectedSource -Reinstall:$Reinstall.IsPresent
    exit 0
} catch {
    Write-Error $_
    exit 1
}
