[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$FilePath
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
if ([System.Environment]::OSVersion.Platform -ne "Win32NT") {
    throw "Authenticode validation requires Windows."
}
$publisher = $env:SIGNPATH_EXPECTED_PUBLISHER
$expectedRoot = $env:SIGNPATH_EXPECTED_ROOT_SHA256
if ([string]::IsNullOrWhiteSpace($publisher) -or $expectedRoot -notmatch '^[0-9a-fA-F]{64}$') {
    throw "Pinned publisher and root certificate SHA256 authority are required."
}
$tools = @(Get-ChildItem -Path "${env:ProgramFiles(x86)}\Windows Kits\10\bin\*\x64\signtool.exe" -File | Sort-Object FullName -Descending)
if ($tools.Count -eq 0) {
    throw "Windows SDK x64 signtool.exe is required."
}
$signature = Get-AuthenticodeSignature -LiteralPath $FilePath
if ($signature.Status -ne "Valid" -or $signature.SignatureType -ne "Authenticode") {
    throw "Authenticode trust validation failed: $($signature.Status)"
}
if ($signature.SignerCertificate.Subject -cne $publisher) {
    throw "Unexpected signing publisher."
}
if ($null -eq $signature.TimeStamperCertificate) {
    throw "An applicable Authenticode timestamp is required."
}
$verification = & $tools[0].FullName verify /pa /all /tw /v $FilePath 2>&1 | Out-String
if ($LASTEXITCODE -ne 0) {
    throw "Windows Authenticode chain/timestamp verification failed: $verification"
}
$chain = [System.Security.Cryptography.X509Certificates.X509Chain]::new()
try {
    $chain.ChainPolicy.RevocationMode = [System.Security.Cryptography.X509Certificates.X509RevocationMode]::Online
    $chain.ChainPolicy.VerificationFlags = [System.Security.Cryptography.X509Certificates.X509VerificationFlags]::IgnoreNotTimeValid
    $chain.ChainPolicy.UrlRetrievalTimeout = [TimeSpan]::FromSeconds(30)
    [void]$chain.ChainPolicy.ApplicationPolicy.Add([System.Security.Cryptography.Oid]::new("1.3.6.1.5.5.7.3.3"))
    if (-not $chain.Build($signature.SignerCertificate)) {
        throw "Signing certificate chain could not be established."
    }
    $root = $chain.ChainElements[$chain.ChainElements.Count - 1].Certificate
    $rootHash = [Convert]::ToHexString([System.Security.Cryptography.SHA256]::HashData($root.RawData)).ToLowerInvariant()
    if ($rootHash -ne $expectedRoot.ToLowerInvariant()) {
        throw "Unexpected signing trust root."
    }
    @{
        status = "Valid"
        publisher = $signature.SignerCertificate.Subject
        signer_sha256 = [Convert]::ToHexString([System.Security.Cryptography.SHA256]::HashData($signature.SignerCertificate.RawData)).ToLowerInvariant()
        root_sha256 = $rootHash
        timestamp = @{
            publisher = $signature.TimeStamperCertificate.Subject
            certificate_sha256 = [Convert]::ToHexString([System.Security.Cryptography.SHA256]::HashData($signature.TimeStamperCertificate.RawData)).ToLowerInvariant()
        }
        wintrust = $verification.Trim()
    } | ConvertTo-Json -Depth 5
} finally {
    $chain.Dispose()
}
