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
    if ([System.Environment]::OSVersion.Platform -ne "Win32NT" -or $env:RUNNER_ENVIRONMENT -ne "github-hosted" -or $env:GITHUB_ACTIONS -ne "true") {
        throw "Test trust import requires an isolated GitHub-hosted Windows runner."
    }
    $identity = [System.Security.Principal.WindowsIdentity]::GetCurrent()
    try {
        $principal = [System.Security.Principal.WindowsPrincipal]::new($identity)
        if (-not $principal.IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)) {
            throw "Test trust import requires an elevated GitHub-hosted Windows runner."
        }
    } finally {
        $identity.Dispose()
    }
    $certificate = [System.Security.Cryptography.X509Certificates.X509Certificate2]::new($bytes)
    try {
        if ($certificate.Subject -cne $certificate.Issuer) {
            throw "The independently pinned test authority must be self-issued."
        }
        Write-Host "Test signing trust: importing pinned authority into LocalMachine/Root."
        $store = [System.Security.Cryptography.X509Certificates.X509Store]::new("Root", "LocalMachine")
        try {
            $store.Open("ReadWrite")
            $store.Add($certificate)
        } finally {
            $store.Dispose()
        }
        Write-Host "Test signing trust: import returned; verifying LocalMachine/Root readback."
        $readback = [System.Security.Cryptography.X509Certificates.X509Store]::new("Root", "LocalMachine")
        try {
            $readback.Open([System.Security.Cryptography.X509Certificates.OpenFlags]::ReadOnly -bor [System.Security.Cryptography.X509Certificates.OpenFlags]::OpenExistingOnly)
            $matches = @($readback.Certificates | Where-Object {
                [Convert]::ToHexString([System.Security.Cryptography.SHA256]::HashData($_.RawData)).ToLowerInvariant() -eq $actual
            })
            if ($matches.Count -ne 1 -or [Convert]::ToBase64String($matches[0].RawData) -cne [Convert]::ToBase64String($bytes)) {
                throw "Pinned test authority LocalMachine/Root readback mismatch."
            }
        } finally {
            $readback.Dispose()
        }
        Write-Host "Test signing trust: exact pinned authority readback verified."
        @{ store = "LocalMachine/Root"; root_sha256 = $actual; status = "verified" } | ConvertTo-Json
    } finally {
        $certificate.Dispose()
    }
}
