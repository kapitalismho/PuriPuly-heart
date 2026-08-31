[CmdletBinding(DefaultParameterSetName = "Remote")]
param(
    [Parameter(Mandatory = $true)]
    [string]$PodId,

    [Parameter(Mandatory = $true)]
    [string]$RunPodCliPath,

    [Parameter(Mandatory = $true)]
    [string]$ReceiptPath,

    [Parameter(Mandatory = $true, ParameterSetName = "Remote")]
    [string]$SshTarget,

    [Parameter(Mandatory = $true, ParameterSetName = "Remote")]
    [string]$RemoteRunRoot,

    [Parameter(ParameterSetName = "Remote")]
    [int]$SshPort = 22,

    [Parameter(ParameterSetName = "Remote")]
    [string]$IdentityFile,

    [Parameter(Mandatory = $true, ParameterSetName = "Local")]
    [string]$LocalControlDirectory,

    [Parameter()]
    [ValidateRange(5, 300)]
    [int]$PollSeconds = 15,

    [Parameter()]
    [ValidateRange(30, 900)]
    [int]$StaleHeartbeatSeconds = 90,

    [Parameter()]
    [ValidateRange(0, 1800)]
    [int]$InitialGraceSeconds = 180,

    [Parameter()]
    [ValidateRange(0.0, 72.0)]
    [double]$MaxRuntimeHours = 20.0,

    [Parameter()]
    [string]$AbsoluteDeadlineUtc,

    [Parameter()]
    [switch]$DryRun,

    [Parameter()]
    [switch]$Once
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-AtomicJson {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [hashtable]$Value
    )

    $parent = Split-Path -Parent $Path
    if (-not [string]::IsNullOrWhiteSpace($parent) -and -not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent | Out-Null
    }
    $temporary = "$Path.$([Guid]::NewGuid().ToString('N')).tmp"
    $Value | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $temporary -Encoding utf8NoBOM
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Read-JsonFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Control file is absent: $Path"
    }
    return Get-Content -LiteralPath $Path -Raw -Encoding utf8 | ConvertFrom-Json
}

function ConvertTo-IsoTimestamp {
    param(
        [Parameter()]
        [object]$Value
    )

    if ($null -eq $Value) {
        return $null
    }
    if ($Value -is [DateTimeOffset]) {
        return $Value.ToString("o")
    }
    if ($Value -is [DateTime]) {
        return ([DateTimeOffset]$Value).ToString("o")
    }
    return [string]$Value
}

function Invoke-Captured {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [Parameter(Mandatory = $true)]
        [string[]]$ArgumentList
    )

    $output = & $FilePath @ArgumentList 2>&1 | Out-String
    return @{
        ExitCode = $LASTEXITCODE
        Output = $output.Trim()
    }
}

function Get-ControlSnapshot {
    if ($PSCmdlet.ParameterSetName -eq "Local") {
        return @{
            State = Read-JsonFile -Path (Join-Path $LocalControlDirectory "state.json")
            Heartbeat = Read-JsonFile -Path (Join-Path $LocalControlDirectory "heartbeat.json")
        }
    }

    if ($RemoteRunRoot -notmatch '^/[A-Za-z0-9._/-]+$') {
        throw "RemoteRunRoot must be an absolute path containing no whitespace or shell metacharacters."
    }
    $scp = (Get-Command "scp.exe" -ErrorAction Stop).Source
    $temporary = Join-Path ([System.IO.Path]::GetTempPath()) "issue-107-watch-$([Guid]::NewGuid().ToString('N'))"
    New-Item -ItemType Directory -Path $temporary | Out-Null
    try {
        $arguments = @("-q", "-P", $SshPort.ToString(), "-o", "BatchMode=yes", "-o", "ConnectTimeout=15")
        if (-not [string]::IsNullOrWhiteSpace($IdentityFile)) {
            $arguments += @("-i", $IdentityFile)
        }
        $stateCopy = Invoke-Captured -FilePath $scp -ArgumentList ($arguments + @("${SshTarget}:$RemoteRunRoot/control/state.json", (Join-Path $temporary "state.json")))
        if ($stateCopy.ExitCode -ne 0) {
            throw "Failed to fetch remote state.json with exit code $($stateCopy.ExitCode)."
        }
        $heartbeatCopy = Invoke-Captured -FilePath $scp -ArgumentList ($arguments + @("${SshTarget}:$RemoteRunRoot/control/heartbeat.json", (Join-Path $temporary "heartbeat.json")))
        if ($heartbeatCopy.ExitCode -ne 0) {
            throw "Failed to fetch remote heartbeat.json with exit code $($heartbeatCopy.ExitCode)."
        }
        return @{
            State = Read-JsonFile -Path (Join-Path $temporary "state.json")
            Heartbeat = Read-JsonFile -Path (Join-Path $temporary "heartbeat.json")
        }
    }
    finally {
        Remove-Item -LiteralPath $temporary -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Get-PodStatus {
    $result = Invoke-Captured -FilePath $RunPodCliPath -ArgumentList @("pod", "get", $PodId, "--output", "json")
    if ($result.ExitCode -ne 0) {
        return @{
            query_exit_code = $result.ExitCode
            id = $PodId
            desired_status = $null
            runtime_status = $null
            runtime_status_reason = $null
        }
    }
    try {
        $value = $result.Output | ConvertFrom-Json
        return @{
            query_exit_code = 0
            id = [string]$value.id
            desired_status = [string]$value.desiredStatus
            runtime_status = [string]$value.runtimeStatus
            runtime_status_reason = [string]$value.runtimeStatusReason
        }
    }
    catch {
        return @{
            query_exit_code = 0
            id = $PodId
            desired_status = $null
            runtime_status = $null
            runtime_status_reason = "unparseable_status"
        }
    }
}

function Stop-WatchedPod {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Reason,

        [Parameter()]
        [object]$ObservedState,

        [Parameter()]
        [object]$ObservedHeartbeat
    )

    $firedAt = [DateTimeOffset]::UtcNow
    if ($DryRun) {
        $receipt = @{
            schema_version = 1
            artifact_role = "issue_107_external_watchdog_receipt"
            pod_id = $PodId
            state = "would_stop"
            reason = $Reason
            armed_at = $script:ArmedAt.ToString("o")
            deadline = $script:Deadline.ToString("o")
            deadline_source = $script:DeadlineSource
            fired_at = $firedAt.ToString("o")
            observed_control_status = if ($null -eq $ObservedState) { $null } else { [string]$ObservedState.status }
            observed_heartbeat_at = if ($null -eq $ObservedHeartbeat) { $null } else { ConvertTo-IsoTimestamp -Value $ObservedHeartbeat.updated_at }
        }
        Write-AtomicJson -Path $ReceiptPath -Value $receipt
        return
    }

    $attemptCount = 0
    while ($true) {
        if ([string]::IsNullOrWhiteSpace($env:RUNPOD_API_KEY)) {
            $receipt = @{
                schema_version = 1
                artifact_role = "issue_107_external_watchdog_receipt"
                pod_id = $PodId
                state = "stop_failed"
                reason = $Reason
                failure = "missing_local_runpod_api_key"
                armed_at = $script:ArmedAt.ToString("o")
                deadline = $script:Deadline.ToString("o")
                deadline_source = $script:DeadlineSource
                fired_at = $firedAt.ToString("o")
                attempt_count = $attemptCount
                observed_control_status = if ($null -eq $ObservedState) { $null } else { [string]$ObservedState.status }
                observed_heartbeat_at = if ($null -eq $ObservedHeartbeat) { $null } else { ConvertTo-IsoTimestamp -Value $ObservedHeartbeat.updated_at }
            }
            Write-AtomicJson -Path $ReceiptPath -Value $receipt
            throw "RUNPOD_API_KEY disappeared from the local watchdog environment before a confirmed live stop."
        }

        $attemptCount += 1
        $attemptedAt = [DateTimeOffset]::UtcNow
        $stop = Invoke-Captured -FilePath $RunPodCliPath -ArgumentList @("pod", "stop", $PodId, "--output", "json")
        $podStatus = Get-PodStatus
        $runtimeStatus = [string]$podStatus.runtime_status
        if (-not [string]::IsNullOrWhiteSpace($runtimeStatus) -and $runtimeStatus.ToLowerInvariant() -in @("stopped", "exited", "terminated")) {
            $confirmedAt = [DateTimeOffset]::UtcNow
            $receipt = @{
                schema_version = 1
                artifact_role = "issue_107_external_watchdog_receipt"
                pod_id = $PodId
                state = "stop_confirmed"
                reason = $Reason
                armed_at = $script:ArmedAt.ToString("o")
                deadline = $script:Deadline.ToString("o")
                deadline_source = $script:DeadlineSource
                fired_at = $firedAt.ToString("o")
                confirmed_at = $confirmedAt.ToString("o")
                attempt_count = $attemptCount
                stop_exit_code = $stop.ExitCode
                observed_control_status = if ($null -eq $ObservedState) { $null } else { [string]$ObservedState.status }
                observed_heartbeat_at = if ($null -eq $ObservedHeartbeat) { $null } else { ConvertTo-IsoTimestamp -Value $ObservedHeartbeat.updated_at }
                pod_status = $podStatus
            }
            Write-AtomicJson -Path $ReceiptPath -Value $receipt
            return
        }

        $receipt = @{
            schema_version = 1
            artifact_role = "issue_107_external_watchdog_receipt"
            pod_id = $PodId
            state = "stop_retrying"
            reason = $Reason
            armed_at = $script:ArmedAt.ToString("o")
            deadline = $script:Deadline.ToString("o")
            deadline_source = $script:DeadlineSource
            fired_at = $firedAt.ToString("o")
            last_attempt_at = $attemptedAt.ToString("o")
            attempt_count = $attemptCount
            stop_exit_code = $stop.ExitCode
            observed_control_status = if ($null -eq $ObservedState) { $null } else { [string]$ObservedState.status }
            observed_heartbeat_at = if ($null -eq $ObservedHeartbeat) { $null } else { ConvertTo-IsoTimestamp -Value $ObservedHeartbeat.updated_at }
            pod_status = $podStatus
        }
        Write-AtomicJson -Path $ReceiptPath -Value $receipt
        Start-Sleep -Seconds $PollSeconds
    }
}

if (-not (Test-Path -LiteralPath $RunPodCliPath -PathType Leaf)) {
    throw "RunPod CLI not found: $RunPodCliPath"
}
if ($PSCmdlet.ParameterSetName -eq "Local" -and -not (Test-Path -LiteralPath $LocalControlDirectory -PathType Container)) {
    throw "Local control directory not found: $LocalControlDirectory"
}
if ($StaleHeartbeatSeconds -le $PollSeconds) {
    throw "StaleHeartbeatSeconds must be greater than PollSeconds."
}
$absoluteDeadlineProvided = $PSBoundParameters.ContainsKey("AbsoluteDeadlineUtc")
$parsedAbsoluteDeadline = [DateTimeOffset]::MinValue
if ($absoluteDeadlineProvided) {
    $deadlineText = [string]$AbsoluteDeadlineUtc
    if ([string]::IsNullOrWhiteSpace($deadlineText)) {
        throw "AbsoluteDeadlineUtc must be a timezone-aware timestamp."
    }
    $deadlineText = $deadlineText.Trim()
    if ($deadlineText -notmatch '(?i)(Z|[+-]\d{2}:\d{2})$') {
        throw "AbsoluteDeadlineUtc must be a timezone-aware timestamp."
    }
    if (-not [DateTimeOffset]::TryParse($deadlineText, [Globalization.CultureInfo]::InvariantCulture, [Globalization.DateTimeStyles]::None, [ref]$parsedAbsoluteDeadline)) {
        throw "AbsoluteDeadlineUtc must be a valid timezone-aware timestamp."
    }
    $parsedAbsoluteDeadline = $parsedAbsoluteDeadline.ToUniversalTime()
}
if (-not $DryRun -and $InitialGraceSeconds -lt 30) {
    throw "InitialGraceSeconds must be at least 30 for a live watchdog."
}
if (-not $DryRun -and -not $absoluteDeadlineProvided -and $MaxRuntimeHours -lt 0.1) {
    throw "MaxRuntimeHours must be at least 0.1 for a live watchdog."
}
if (-not $DryRun -and $absoluteDeadlineProvided -and $parsedAbsoluteDeadline -le [DateTimeOffset]::UtcNow) {
    throw "AbsoluteDeadlineUtc must be in the future before arming a live watchdog."
}
if (-not $DryRun -and [string]::IsNullOrWhiteSpace($env:RUNPOD_API_KEY)) {
    throw "RUNPOD_API_KEY must be present in the local watchdog environment before arming."
}

$script:ArmedAt = [DateTimeOffset]::UtcNow
if ($absoluteDeadlineProvided) {
    $script:Deadline = $parsedAbsoluteDeadline
    $script:DeadlineSource = "absolute_utc"
}
else {
    $script:Deadline = $script:ArmedAt.AddHours($MaxRuntimeHours)
    $script:DeadlineSource = "armed_at_plus_max_runtime"
}
Write-AtomicJson -Path $ReceiptPath -Value @{
    schema_version = 1
    artifact_role = "issue_107_external_watchdog_receipt"
    pod_id = $PodId
    state = "armed"
    armed_at = $script:ArmedAt.ToString("o")
    deadline = $script:Deadline.ToString("o")
    deadline_source = $script:DeadlineSource
    poll_seconds = $PollSeconds
    stale_heartbeat_seconds = $StaleHeartbeatSeconds
    initial_grace_seconds = $InitialGraceSeconds
    mode = if ($PSCmdlet.ParameterSetName -eq "Local") { "local_control_files" } else { "remote_scp_poll" }
    dry_run = [bool]$DryRun
}

while ($true) {
    $now = [DateTimeOffset]::UtcNow
    if ($now -ge $script:Deadline) {
        Stop-WatchedPod -Reason "absolute_deadline" -ObservedState $null -ObservedHeartbeat $null
        break
    }

    $snapshot = $null
    try {
        $snapshot = Get-ControlSnapshot
    }
    catch {
        $graceElapsed = ($now - $script:ArmedAt).TotalSeconds -ge $InitialGraceSeconds
        if ($graceElapsed) {
            Stop-WatchedPod -Reason "control_unreachable_after_initial_grace" -ObservedState $null -ObservedHeartbeat $null
            break
        }
    }

    if ($null -ne $snapshot) {
        $state = $snapshot.State
        $heartbeat = $snapshot.Heartbeat
        $status = [string]$state.status
        if ($status -in @("WAITING_FOR_DECISION", "ERROR", "COMPLETED")) {
            Stop-WatchedPod -Reason "control_status_$($status.ToLowerInvariant())" -ObservedState $state -ObservedHeartbeat $heartbeat
            break
        }
        if ($status -notin @("STARTING", "RUNNING")) {
            Stop-WatchedPod -Reason "unknown_control_status" -ObservedState $state -ObservedHeartbeat $heartbeat
            break
        }
        try {
            $heartbeatAt = [DateTimeOffset]::Parse([string]$heartbeat.updated_at)
            $heartbeatAge = ($now - $heartbeatAt).TotalSeconds
        }
        catch {
            $heartbeatAge = [double]::PositiveInfinity
        }
        $graceElapsed = ($now - $script:ArmedAt).TotalSeconds -ge $InitialGraceSeconds
        if ($graceElapsed -and $heartbeatAge -gt $StaleHeartbeatSeconds) {
            Stop-WatchedPod -Reason "stale_heartbeat" -ObservedState $state -ObservedHeartbeat $heartbeat
            break
        }
    }

    if ($Once) {
        break
    }
    Start-Sleep -Seconds $PollSeconds
}
