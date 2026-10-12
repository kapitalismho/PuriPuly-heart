[CmdletBinding()]
param(
    [Parameter()]
    [switch]$TestSigning
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
if ($env:SIGNPATH_EXPECTED_ROOT_SHA256 -notmatch '^[0-9a-fA-F]{64}$' -or [string]::IsNullOrWhiteSpace($env:SIGNPATH_EXPECTED_PUBLISHER)) {
    throw "Independently verified publisher/root variables are required."
}
if ($TestSigning) {
    if ([string]::IsNullOrWhiteSpace($env:SIGNPATH_TEST_ROOT_CERTIFICATE_BASE64)) {
        throw "Test signing requires an independently obtained, pinned test root certificate."
    }
    $bytes = [Convert]::FromBase64String($env:SIGNPATH_TEST_ROOT_CERTIFICATE_BASE64)
    $actual = [Convert]::ToHexString([System.Security.Cryptography.SHA256]::HashData($bytes)).ToLowerInvariant()
    if ($actual -ne $env:SIGNPATH_EXPECTED_ROOT_SHA256.ToLowerInvariant()) {
        throw "Test trust anchor differs from the pinned SHA256."
    }
    $certificate = [System.Security.Cryptography.X509Certificates.X509Certificate2]::new($bytes)
    try {
        if ($certificate.Subject -cne $certificate.Issuer) {
            throw "The independently pinned test authority must be self-issued."
        }
        $store = [System.Security.Cryptography.X509Certificates.X509Store]::new("Root", "CurrentUser")
        try {
            $store.Open("ReadWrite")
            $store.Add($certificate)
        } finally {
            $store.Dispose()
        }
    } finally {
        $certificate.Dispose()
    }
}
