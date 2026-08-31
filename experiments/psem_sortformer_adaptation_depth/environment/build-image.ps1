[CmdletBinding()]
param(
    [Parameter()]
    [string]$Tag = "puripuly-heart/issue-107-runtime:local",

    [Parameter()]
    [string]$Distribution = "Ubuntu",

    [Parameter()]
    [switch]$NoCache
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$WslExecutable = (Get-Command "wsl.exe" -ErrorAction Stop).Source
$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..\..")).Path
$normalizedRepositoryRoot = $RepositoryRoot.Replace("\", "/")
$pathMatch = [regex]::Match($normalizedRepositoryRoot, '^(?<drive>[A-Za-z]):/(?<rest>.+)$')
if (-not $pathMatch.Success) {
    throw "Repository root is not a WSL-accessible Windows drive path: $RepositoryRoot"
}
$drive = $pathMatch.Groups["drive"].Value.ToLowerInvariant()
$LinuxRepositoryRoot = "/mnt/$drive/$($pathMatch.Groups['rest'].Value)"

$Dockerfile = "$LinuxRepositoryRoot/experiments/psem_sortformer_adaptation_depth/environment/Dockerfile"
$BuildContext = "$LinuxRepositoryRoot/experiments/psem_sortformer_adaptation_depth/environment"
$DockerArguments = @(
    "docker",
    "build",
    "--platform",
    "linux/amd64",
    "--progress",
    "plain",
    "--file",
    $Dockerfile,
    "--tag",
    $Tag
)
if ($NoCache) {
    $DockerArguments += "--no-cache"
}
$DockerArguments += $BuildContext

& $WslExecutable -d $Distribution -- @DockerArguments
if ($LASTEXITCODE -ne 0) {
    throw "Docker build failed with exit code $LASTEXITCODE."
}

& $WslExecutable -d $Distribution -- docker image inspect $Tag
if ($LASTEXITCODE -ne 0) {
    throw "Docker image inspection failed with exit code $LASTEXITCODE."
}

& $WslExecutable -d $Distribution -- docker run --rm --entrypoint python $Tag /opt/psem/validate-runtime.py --mode build --receipt /tmp/issue-107-build-validation.json
if ($LASTEXITCODE -ne 0) {
    throw "Container build validation failed with exit code $LASTEXITCODE."
}
