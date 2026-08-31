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

    [Parameter(ParameterSetName = "Remote")]
    [string]$KnownHostsFile,

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
    [string]$AbsoluteDeadlineUtc,

    [Parameter()]
    [switch]$DryRun,

    [Parameter()]
    [switch]$Once
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
$script:WatchdogParameterSetName = $PSCmdlet.ParameterSetName

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

function Get-FileSha256 {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-ConfigDeadlineText {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path
    )

    $raw = Get-Content -LiteralPath $Path -Raw -Encoding utf8
    $matches = [regex]::Matches($raw, '"absolute_deadline_utc"\s*:\s*"([^"]+)"')
    if ($matches.Count -ne 1) {
        throw "Run config must contain exactly one absolute_deadline_utc string."
    }
    return $matches[0].Groups[1].Value
}

function ConvertFrom-ConfigDeadline {
    param(
        [Parameter(Mandatory = $true)]
        [object]$Value
    )

    $text = [string]$Value
    if (
        [string]::IsNullOrWhiteSpace($text) -or
        $text -notmatch '^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{6})?\+00:00$' -or
        $text -match '\.000000\+00:00$'
    ) {
        throw "Run config absolute_deadline_utc must be a canonical UTC timestamp."
    }
    $parsed = [DateTimeOffset]::MinValue
    if (-not [DateTimeOffset]::TryParse($text, [Globalization.CultureInfo]::InvariantCulture, [Globalization.DateTimeStyles]::None, [ref]$parsed)) {
        throw "Run config absolute_deadline_utc must be a canonical UTC timestamp."
    }
    if ($parsed.Offset -ne [TimeSpan]::Zero) {
        throw "Run config absolute_deadline_utc must be UTC."
    }
    return $parsed.ToUniversalTime()
}

function Assert-ControlSnapshot {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$Snapshot
    )

    $config = $Snapshot.Config
    $state = $Snapshot.State
    $heartbeat = $Snapshot.Heartbeat
    $configHash = [string]$Snapshot.ConfigSha256
    if ([int]$config.schema_version -ne 2) {
        throw "Run config schema_version must be 2."
    }
    if ($configHash -notmatch '^[0-9a-f]{64}$') {
        throw "Run config SHA-256 is invalid."
    }
    if ([string]$state.config_sha256 -cne $configHash) {
        throw "State is not bound to the durable run config."
    }
    if ([string]$heartbeat.config_sha256 -cne $configHash) {
        throw "Heartbeat is not bound to the durable run config."
    }
    $runId = [string]$config.run_id
    if ([string]::IsNullOrWhiteSpace($runId) -or [string]$state.run_id -cne $runId -or [string]$heartbeat.run_id -cne $runId) {
        throw "Config, state, and heartbeat run IDs differ."
    }
    return @{
        RunId = $runId
        ConfigSha256 = $configHash
        Deadline = ConvertFrom-ConfigDeadline -Value $Snapshot.ConfigDeadlineText
    }
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
        [string[]]$ArgumentList,

        [Parameter()]
        [object]$DeadlineUtc,

        [Parameter()]
        [ValidateRange(0, 300)]
        [int]$TimeoutSeconds = 0
    )

    $timeoutMilliseconds = -1
    if ($null -ne $DeadlineUtc) {
        $remainingMilliseconds = ([DateTimeOffset]$DeadlineUtc - [DateTimeOffset]::UtcNow).TotalMilliseconds
        if ($remainingMilliseconds -le 0) {
            return @{ ExitCode = 124; Output = "command_deadline_reached" }
        }
        $timeoutMilliseconds = [int][Math]::Min([int]::MaxValue, [Math]::Ceiling($remainingMilliseconds))
    }
    if ($TimeoutSeconds -gt 0) {
        $fixedMilliseconds = $TimeoutSeconds * 1000
        if ($timeoutMilliseconds -lt 0 -or $fixedMilliseconds -lt $timeoutMilliseconds) {
            $timeoutMilliseconds = $fixedMilliseconds
        }
    }

    $startInfo = [System.Diagnostics.ProcessStartInfo]::new()
    $startInfo.FileName = $FilePath
    $startInfo.UseShellExecute = $false
    $startInfo.RedirectStandardOutput = $true
    $startInfo.RedirectStandardError = $true
    $startInfo.CreateNoWindow = $true
    foreach ($argument in $ArgumentList) {
        $startInfo.ArgumentList.Add($argument)
    }
    $process = [System.Diagnostics.Process]::new()
    $process.StartInfo = $startInfo
    try {
        if (-not $process.Start()) {
            return @{ ExitCode = 125; Output = "command_start_failed" }
        }
        $stdoutTask = $process.StandardOutput.ReadToEndAsync()
        $stderrTask = $process.StandardError.ReadToEndAsync()
        $completed = if ($timeoutMilliseconds -lt 0) {
            $process.WaitForExit()
            $true
        }
        else {
            $process.WaitForExit($timeoutMilliseconds)
        }
        if (-not $completed) {
            $killFailed = $false
            try {
                $process.Kill($true)
            }
            catch {
                $killFailed = $true
            }
            $killConfirmed = $process.WaitForExit(1000)
            if ($killFailed -or -not $killConfirmed) {
                return @{ ExitCode = 126; Output = "command_kill_unconfirmed" }
            }
            return @{ ExitCode = 124; Output = "command_timeout" }
        }
        $stdout = $stdoutTask.GetAwaiter().GetResult()
        $stderr = $stderrTask.GetAwaiter().GetResult()
        return @{
            ExitCode = $process.ExitCode
            Output = (($stdout + $stderr) | Out-String).Trim()
        }
    }
    finally {
        $process.Dispose()
    }
}

function Get-ControlSnapshot {
    param(
        [Parameter()]
        [object]$DeadlineUtc
    )

    if ($script:WatchdogParameterSetName -eq "Local") {
        $configPath = Join-Path $LocalControlDirectory "run_config.json"
        return @{
            Config = Read-JsonFile -Path $configPath
            ConfigSha256 = Get-FileSha256 -Path $configPath
            ConfigDeadlineText = Get-ConfigDeadlineText -Path $configPath
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
        if (-not [string]::IsNullOrWhiteSpace($KnownHostsFile)) {
            $arguments += @("-o", "UserKnownHostsFile=$KnownHostsFile", "-o", "StrictHostKeyChecking=yes")
        }
        $configPath = Join-Path $temporary "run_config.json"
        $configCopy = Invoke-Captured -FilePath $scp -ArgumentList ($arguments + @("${SshTarget}:$RemoteRunRoot/control/run_config.json", $configPath)) -DeadlineUtc $DeadlineUtc
        if ($configCopy.ExitCode -ne 0) {
            throw "Failed to fetch remote run_config.json with exit code $($configCopy.ExitCode)."
        }
        $stateCopy = Invoke-Captured -FilePath $scp -ArgumentList ($arguments + @("${SshTarget}:$RemoteRunRoot/control/state.json", (Join-Path $temporary "state.json"))) -DeadlineUtc $DeadlineUtc
        if ($stateCopy.ExitCode -ne 0) {
            throw "Failed to fetch remote state.json with exit code $($stateCopy.ExitCode)."
        }
        $heartbeatCopy = Invoke-Captured -FilePath $scp -ArgumentList ($arguments + @("${SshTarget}:$RemoteRunRoot/control/heartbeat.json", (Join-Path $temporary "heartbeat.json"))) -DeadlineUtc $DeadlineUtc
        if ($heartbeatCopy.ExitCode -ne 0) {
            throw "Failed to fetch remote heartbeat.json with exit code $($heartbeatCopy.ExitCode)."
        }
        return @{
            Config = Read-JsonFile -Path $configPath
            ConfigSha256 = Get-FileSha256 -Path $configPath
            ConfigDeadlineText = Get-ConfigDeadlineText -Path $configPath
            State = Read-JsonFile -Path (Join-Path $temporary "state.json")
            Heartbeat = Read-JsonFile -Path (Join-Path $temporary "heartbeat.json")
        }
    }
    finally {
        Remove-Item -LiteralPath $temporary -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Get-PodStatus {
    $result = Invoke-Captured -FilePath $RunPodCliPath -ArgumentList @("pod", "get", $PodId, "--output", "json") -TimeoutSeconds 30
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
            run_id = $script:RunId
            config_sha256 = $script:ConfigSha256
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
                run_id = $script:RunId
                config_sha256 = $script:ConfigSha256
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
        $stop = Invoke-Captured -FilePath $RunPodCliPath -ArgumentList @("pod", "stop", $PodId, "--output", "json") -TimeoutSeconds 30
        $podStatus = Get-PodStatus
        $runtimeStatus = [string]$podStatus.runtime_status
        if (-not [string]::IsNullOrWhiteSpace($runtimeStatus) -and $runtimeStatus.ToLowerInvariant() -in @("stopped", "exited", "terminated")) {
            $confirmedAt = [DateTimeOffset]::UtcNow
            $receipt = @{
                schema_version = 1
                artifact_role = "issue_107_external_watchdog_receipt"
                pod_id = $PodId
                run_id = $script:RunId
                config_sha256 = $script:ConfigSha256
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
            run_id = $script:RunId
            config_sha256 = $script:ConfigSha256
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
if ($script:WatchdogParameterSetName -eq "Local" -and -not (Test-Path -LiteralPath $LocalControlDirectory -PathType Container)) {
    throw "Local control directory not found: $LocalControlDirectory"
}
if (
    $script:WatchdogParameterSetName -eq "Remote" -and
    -not [string]::IsNullOrWhiteSpace($KnownHostsFile) -and
    -not (Test-Path -LiteralPath $KnownHostsFile -PathType Leaf)
) {
    throw "Pinned known-hosts file not found: $KnownHostsFile"
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
if (-not $DryRun -and -not $absoluteDeadlineProvided) {
    throw "AbsoluteDeadlineUtc is required as a local cross-check for a live watchdog."
}
if (-not $DryRun -and [string]::IsNullOrWhiteSpace($env:RUNPOD_API_KEY)) {
    throw "RUNPOD_API_KEY must be present in the local watchdog environment before arming."
}

$initialFetchDeadline = if ($absoluteDeadlineProvided) { $parsedAbsoluteDeadline } else { $null }
try {
    $initialSnapshot = Get-ControlSnapshot -DeadlineUtc $initialFetchDeadline
    $initialBinding = Assert-ControlSnapshot -Snapshot $initialSnapshot
}
catch {
    if (-not $DryRun -and $absoluteDeadlineProvided -and [DateTimeOffset]::UtcNow -ge $parsedAbsoluteDeadline) {
        $script:RunId = $null
        $script:ConfigSha256 = $null
        $script:Deadline = $parsedAbsoluteDeadline
        $script:DeadlineSource = "required_operator_cross_check_before_config_binding"
        $script:ArmedAt = [DateTimeOffset]::UtcNow
        Stop-WatchedPod -Reason "absolute_deadline_before_config_binding" -ObservedState $null -ObservedHeartbeat $null
        return
    }
    throw "Cannot arm without an exact config/state/heartbeat binding: $($_.Exception.Message)"
}
$script:RunId = [string]$initialBinding.RunId
$script:ConfigSha256 = [string]$initialBinding.ConfigSha256
$script:Deadline = [DateTimeOffset]$initialBinding.Deadline
$script:DeadlineSource = "immutable_run_config"
if ($absoluteDeadlineProvided -and $parsedAbsoluteDeadline -ne $script:Deadline) {
    $configDeadline = $script:Deadline
    $script:Deadline = if ($parsedAbsoluteDeadline -lt $configDeadline) { $parsedAbsoluteDeadline } else { $configDeadline }
    $script:DeadlineSource = "deadline_mismatch_conservative_minimum"
    $script:ArmedAt = [DateTimeOffset]::UtcNow
    Stop-WatchedPod -Reason "deadline_mismatch_before_arm" -ObservedState $initialSnapshot.State -ObservedHeartbeat $initialSnapshot.Heartbeat
    return
}
$script:ArmedAt = [DateTimeOffset]::UtcNow
if ($script:Deadline -le $script:ArmedAt) {
    Stop-WatchedPod -Reason "absolute_deadline_before_arm" -ObservedState $initialSnapshot.State -ObservedHeartbeat $initialSnapshot.Heartbeat
    return
}
Write-AtomicJson -Path $ReceiptPath -Value @{
    schema_version = 1
    artifact_role = "issue_107_external_watchdog_receipt"
    pod_id = $PodId
    run_id = $script:RunId
    config_sha256 = $script:ConfigSha256
    state = "armed"
    armed_at = $script:ArmedAt.ToString("o")
    deadline = $script:Deadline.ToString("o")
    deadline_source = $script:DeadlineSource
    poll_seconds = $PollSeconds
    stale_heartbeat_seconds = $StaleHeartbeatSeconds
    initial_grace_seconds = $InitialGraceSeconds
    mode = if ($script:WatchdogParameterSetName -eq "Local") { "local_control_files" } else { "remote_scp_poll" }
    dry_run = [bool]$DryRun
}

$pendingSnapshot = $initialSnapshot
while ($true) {
    $now = [DateTimeOffset]::UtcNow
    if ($now -ge $script:Deadline) {
        Stop-WatchedPod -Reason "absolute_deadline" -ObservedState $null -ObservedHeartbeat $null
        break
    }

    $snapshot = $pendingSnapshot
    $pendingSnapshot = $null
    try {
        if ($null -eq $snapshot) {
            $snapshot = Get-ControlSnapshot -DeadlineUtc $script:Deadline
        }
    }
    catch {
        $now = [DateTimeOffset]::UtcNow
        if ($now -ge $script:Deadline) {
            Stop-WatchedPod -Reason "absolute_deadline" -ObservedState $null -ObservedHeartbeat $null
            break
        }
        $graceElapsed = ($now - $script:ArmedAt).TotalSeconds -ge $InitialGraceSeconds
        if ($graceElapsed) {
            Stop-WatchedPod -Reason "control_unreachable_after_initial_grace" -ObservedState $null -ObservedHeartbeat $null
            break
        }
    }

    if ($null -ne $snapshot) {
        try {
            $binding = Assert-ControlSnapshot -Snapshot $snapshot
        }
        catch {
            Stop-WatchedPod -Reason "control_binding_invalid" -ObservedState $snapshot.State -ObservedHeartbeat $snapshot.Heartbeat
            break
        }
        if ([string]$binding.RunId -cne $script:RunId -or [string]$binding.ConfigSha256 -cne $script:ConfigSha256 -or [DateTimeOffset]$binding.Deadline -ne $script:Deadline) {
            Stop-WatchedPod -Reason "control_binding_changed" -ObservedState $snapshot.State -ObservedHeartbeat $snapshot.Heartbeat
            break
        }
        $now = [DateTimeOffset]::UtcNow
        $state = $snapshot.State
        $heartbeat = $snapshot.Heartbeat
        if ($now -ge $script:Deadline) {
            Stop-WatchedPod -Reason "absolute_deadline" -ObservedState $state -ObservedHeartbeat $heartbeat
            break
        }
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
    $remainingSeconds = ($script:Deadline - [DateTimeOffset]::UtcNow).TotalSeconds
    if ($remainingSeconds -le 0) {
        continue
    }
    $sleepMilliseconds = [int][Math]::Ceiling([Math]::Min($PollSeconds, $remainingSeconds) * 1000.0)
    Start-Sleep -Milliseconds $sleepMilliseconds
}
