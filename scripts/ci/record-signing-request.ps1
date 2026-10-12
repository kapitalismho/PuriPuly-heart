[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("components", "installer")]
    [string]$Phase,

    [Parameter(Mandatory = $true)]
    [string]$OutputDir
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
$path = Join-Path $OutputDir "$Phase-request.json"
$record = @{
    artifact_id = $env:ARTIFACT_ID
    request_id = $env:REQUEST_ID
    request_url = $env:REQUEST_URL
    action_outcome = $env:REQUEST_RESULT
    metadata_status = "not-requested"
}
function Save-Record {
    $text = $record | ConvertTo-Json -Depth 12
    if (-not [string]::IsNullOrEmpty($env:SIGNPATH_API_TOKEN)) {
        $text = $text.Replace($env:SIGNPATH_API_TOKEN, "[REDACTED]")
    }
    $text | Set-Content -LiteralPath $path -Encoding utf8
}
Save-Record
if ([string]::IsNullOrWhiteSpace($env:REQUEST_ID)) {
    if ($env:REQUEST_RESULT -eq "success") { throw "Signing returned no request ID." }
    return
}
if ($env:REQUEST_ID -notmatch '^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$') {
    $record.metadata_status = "invalid-request-id"
    Save-Record
    throw "Malformed signing request ID."
}
try {
    $uri = "https://app.signpath.io/Api/v1/41a8602a-e07d-4d35-876f-8876d5acb31d/SigningRequests/$env:REQUEST_ID"
    $response = Invoke-RestMethod -Uri $uri -Headers @{ Authorization = "Bearer $env:SIGNPATH_API_TOKEN" } -TimeoutSec 60
    $metadata = @{}
    foreach ($field in @("signingRequestId", "status", "workflowStatus", "isFinalStatus", "projectSlug", "artifactConfigurationSlug", "signingPolicySlug", "parameters", "origin", "webLink")) {
        $property = $response.PSObject.Properties[$field]
        if ($null -ne $property) { $metadata[$field] = $property.Value }
    }
    $record.metadata = $metadata
    $record.metadata_status = "retrieved"
    Save-Record
} catch {
    $record.metadata_status = "retrieval-failed"
    Save-Record
    throw "Signing request metadata retrieval failed; request ID preserved in evidence."
}
