[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$ToolPython,
    [Parameter(Mandatory = $true)][string]$SigningRoot,
    [Parameter(Mandatory = $true)][string]$EvidenceRoot,
    [Parameter(Mandatory = $true)][string]$Version
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if ($env:RUNNER_ENVIRONMENT -ne "github-hosted" -or $env:GITHUB_ACTIONS -ne "true") {
    throw "Actual release installer validation requires an isolated GitHub-hosted runner."
}
$localData = [Environment]::GetFolderPath("LocalApplicationData")
$installed = Join-Path $localData "Programs\PuriPulyHeart-SignedValidation-$env:GITHUB_RUN_ID-$env:GITHUB_RUN_ATTEMPT"
$registry = "HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall\{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}_is1"
$appData = Join-Path $localData "puripuly-heart"
if ((Test-Path -LiteralPath $installed) -or (Test-Path -LiteralPath $registry) -or (Test-Path -LiteralPath $appData)) {
    throw "Release install validation refuses preexisting application state."
}
New-Item -ItemType Directory -Path $EvidenceRoot -Force | Out-Null
$installer = Join-Path $SigningRoot "signed-installer\PuriPulyHeart-Setup-$Version.exe"
$installLog = Join-Path $EvidenceRoot "install.log"
$uninstallLog = Join-Path $EvidenceRoot "uninstall.log"
$installationStarted = $false
$uninstallPassed = $false
try {
    $installationStarted = $true
    $process = Start-Process -FilePath $installer -ArgumentList @(
        "/VERYSILENT", "/CURRENTUSER", "/SP-", "/SUPPRESSMSGBOXES", "/NORESTART", "/NOICONS", "/TASKS=",
        "/DIR=`"$installed`"", "/LOG=`"$installLog`""
    ) -PassThru
    if (-not $process.WaitForExit(1200000)) {
        $process.Kill($true)
        throw "Signed installer validation timed out."
    }
    if ($process.ExitCode -ne 0) { throw "Signed installer failed: $($process.ExitCode)" }
    & $ToolPython -m puripuly_heart.release_evidence.signing_release validate-installed `
        --version $Version --root $SigningRoot --directory $installed
    if ($LASTEXITCODE -ne 0) { throw "Installed signature or byte validation failed." }
    $versionResult = & (Join-Path $installed "PuriPulyHeart.exe") --headless --version | Out-String
    if ($LASTEXITCODE -ne 0 -or $versionResult.Trim() -ne $Version) {
        throw "Installed native runtime version failed."
    }
} finally {
    if ($installationStarted -and (Test-Path -LiteralPath $installed)) {
        $uninstallers = @(Get-ChildItem -LiteralPath $installed -Filter "unins*.exe" -File)
        if ($uninstallers.Count -ne 1) { throw "Cannot verify clean uninstall: generated uninstaller missing or ambiguous." }
        $process = Start-Process -FilePath $uninstallers[0].FullName -ArgumentList @(
            "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/LOG=`"$uninstallLog`""
        ) -PassThru
        if (-not $process.WaitForExit(300000)) {
            $process.Kill($true)
            throw "Signed uninstaller timed out."
        }
        if ($process.ExitCode -ne 0 -or (Test-Path -LiteralPath $registry)) {
            throw "Clean signed uninstall failed."
        }
        foreach ($target in @("PuriPulyHeart.exe", "PuriPulyHeartOverlay.exe", "PuriPulyHeartGpuWorker.exe", $uninstallers[0].Name)) {
            if (Test-Path -LiteralPath (Join-Path $installed $target)) { throw "Signed target remains after uninstall: $target" }
        }
        $uninstallPassed = $true
    }
    @{ uninstall = $(if ($uninstallPassed) { "passed" } else { "not-passed" }); installed_directory = $installed } |
        ConvertTo-Json | Set-Content -LiteralPath (Join-Path $EvidenceRoot "install-result.json") -Encoding utf8
}
if (-not $uninstallPassed) { throw "Clean signed uninstall evidence missing." }
