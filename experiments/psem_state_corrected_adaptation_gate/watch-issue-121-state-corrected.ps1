[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$StatePath,

    [Parameter()]
    [string]$PodId,

    [Parameter()]
    [string]$RunPodCliPath,

    [Parameter()]
    [string]$ReceiptPath,

    [Parameter()]
    [ValidateRange(1, 300)]
    [int]$PollSeconds = 10,

    [Parameter()]
    [ValidateRange(1, 3600)]
    [int]$SummarySeconds = 60,

    [Parameter()]
    [ValidateRange(60, 21600)]
    [int]$UnhandledSeconds = 5400,

    [Parameter()]
    [ValidateRange(1, 120)]
    [int]$MaxStopAttempts = 30,

    [Parameter()]
    [switch]$Silent,

    [Parameter()]
    [switch]$DryRun,

    [Parameter()]
    [switch]$Once
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($SummarySeconds -lt $PollSeconds) { $SummarySeconds = $PollSeconds }
if ([string]::IsNullOrWhiteSpace($ReceiptPath)) {
    $receiptDir = Split-Path -Parent $StatePath
    if ([string]::IsNullOrWhiteSpace($receiptDir)) { $receiptDir = "." }
    $ReceiptPath = Join-Path $receiptDir "issue-121-watchdog-receipt.json"
}
$script:LiveGuard = (-not $DryRun) -and (-not [string]::IsNullOrWhiteSpace($RunPodCliPath))
$script:GuardSince = $null
$script:GuardReason = $null
$script:GuardPodId = $null
$script:GuardKey = $null
$script:ArmedAt = [DateTimeOffset]::UtcNow

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
    $Value | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $temporary -Encoding utf8
    Move-Item -LiteralPath $temporary -Destination $Path -Force
}

function Invoke-Captured {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [Parameter(Mandatory = $true)]
        [string[]]$ArgumentList,

        [Parameter()]
        [ValidateRange(0, 300)]
        [int]$TimeoutSeconds = 0
    )
    $timeoutMilliseconds = -1
    if ($TimeoutSeconds -gt 0) {
        $timeoutMilliseconds = $TimeoutSeconds * 1000
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

function Invoke-GateBeep {
    try {
        [Console]::Beep(880, 160)
        Start-Sleep -Milliseconds 70
        [Console]::Beep(1480, 220)
    } catch {
    }
}

function Read-GateState {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
    try {
        return Get-Content -LiteralPath $Path -Raw -Encoding utf8 | ConvertFrom-Json
    } catch {
        return $null
    }
}

function Get-TransitionKey {
    param($State)
    if ($null -eq $State) { return $null }
    $branch = [string]$State.branch
    $gate = [string]$State.gate
    if ([string]::IsNullOrWhiteSpace($branch) -or [string]::IsNullOrWhiteSpace($gate)) { return $null }
    return "$branch|$gate"
}

function Get-GuardContentKey {
    param($State)
    $branch = ""
    $gate = ""
    $statePod = ""
    if ($null -ne $State) {
        $branch = [string]$State.branch
        $gate = [string]$State.gate
        $statePod = [string]$State.pod
    }
    $pinned = ""
    if (-not [string]::IsNullOrWhiteSpace($PodId)) { $pinned = [string]$PodId }
    return ("{0}|{1}|{2}|{3}" -f $branch, $gate, $statePod, $pinned)
}

function Get-GuardClassification {
    param($State)
    $pinned = ""
    if (-not [string]::IsNullOrWhiteSpace($PodId)) { $pinned = [string]$PodId }
    if ($null -eq $State) {
        return @{ Handled = $false; Reason = "control_missing_or_invalid"; Pod = $pinned; Branch = ""; Gate = "" }
    }
    $branch = [string]$State.branch
    $gate = [string]$State.gate
    $statePod = [string]$State.pod
    $effective = if ([string]::IsNullOrWhiteSpace($statePod)) { $pinned } else { $statePod }
    if ([string]::IsNullOrWhiteSpace($branch) -or [string]::IsNullOrWhiteSpace($gate)) {
        return @{ Handled = $false; Reason = "control_missing_or_invalid"; Pod = $effective; Branch = $branch; Gate = $gate }
    }
    if ($gate -in @("P0-RUNNING", "P0-REPAIR", "P1-RUNNING", "P1-REPAIR", "P1-PROFILE", "P2-RUNNING", "P2-REPAIR", "P2-PROFILE", "P3-RUNNING", "P3-REPAIR", "P3-PROFILE")) {
        return @{ Handled = $true; Reason = ""; Pod = $effective; Branch = $branch; Gate = $gate }
    }
    $sanitized = $gate.ToUpperInvariant() -replace '[^A-Z0-9]+', '_'
    return @{ Handled = $false; Reason = ("unhandled_gate_" + $sanitized); Pod = $effective; Branch = $branch; Gate = $gate }
}

function Get-StatusLine {
    param($State, [string]$Timestamp)
    if ($null -eq $State) {
        return "issue-121 status $Timestamp waiting (state missing or incomplete)"
    }
    return ("issue-121 status {0} branch={1} gate={2}" -f $Timestamp, [string]$State.branch, [string]$State.gate)
}

function Get-GuardStatusText {
    if ($null -eq $script:GuardSince) { return " guard=handled" }
    $age = [math]::Floor(([DateTimeOffset]::UtcNow - $script:GuardSince).TotalSeconds)
    return (" guard=unhandled({0} {1}s/{2}s pod={3})" -f $script:GuardReason, $age, $UnhandledSeconds, [string]$script:GuardPodId)
}

function Write-GuardReceipt {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ReceiptState,

        [Parameter()]
        [string]$Reason,

        [Parameter()]
        [hashtable]$Observed,

        [Parameter()]
        [hashtable]$Extra
    )
    $branch = ""
    $gate = ""
    $pod = ""
    if ($null -ne $Observed) {
        $branch = [string]$Observed.Branch
        $gate = [string]$Observed.Gate
        $pod = [string]$Observed.Pod
    }
    if ([string]::IsNullOrWhiteSpace($Reason)) { $Reason = "" }
    $value = @{
        schema_version = 1
        artifact_role = "issue_121_watchdog_receipt"
        state = $ReceiptState
        reason = $Reason
        armed_at = $script:ArmedAt.ToString("o")
        maximum_unhandled_seconds = $UnhandledSeconds
        poll_seconds = $PollSeconds
        dry_run = [bool]$DryRun
        live = [bool]$script:LiveGuard
        api_key_present = -not [string]::IsNullOrWhiteSpace($env:RUNPOD_API_KEY)
        branch = $branch
        gate = $gate
        pod_id = $pod
        unhandled_since = $null
        unhandled_seconds = $null
    }
    if ($null -ne $script:GuardSince) {
        $value.unhandled_since = $script:GuardSince.ToString("o")
        $value.unhandled_seconds = [math]::Floor(([DateTimeOffset]::UtcNow - $script:GuardSince).TotalSeconds)
    }
    if ($null -ne $Extra) {
        foreach ($entry in $Extra.GetEnumerator()) { $value[$entry.Key] = $entry.Value }
    }
    Write-AtomicJson -Path $ReceiptPath -Value $value
}

function Clear-GuardState {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$Observed
    )
    $script:GuardSince = $null
    $script:GuardReason = $null
    $script:GuardPodId = $null
    $script:GuardKey = $null
    Write-GuardReceipt -ReceiptState "armed" -Observed $Observed
}

function Get-121PodStatus {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TargetPodId
    )
    $result = Invoke-Captured -FilePath $RunPodCliPath -ArgumentList @("pod", "get", $TargetPodId, "--output", "json") -TimeoutSeconds 30
    if ($result.ExitCode -ne 0) {
        return @{ query_exit_code = $result.ExitCode; id = $TargetPodId; desired_status = $null; runtime_status = $null }
    }
    try {
        $value = $result.Output | ConvertFrom-Json
        return @{ query_exit_code = 0; id = [string]$value.id; desired_status = [string]$value.desiredStatus; runtime_status = [string]$value.runtimeStatus }
    }
    catch {
        return @{ query_exit_code = 0; id = $TargetPodId; desired_status = $null; runtime_status = $null }
    }
}

function Stop-121Pod {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Reason
    )
    $target = [string]$script:GuardPodId
    $fresh = Read-GateState -Path $StatePath
    $check = Get-GuardClassification -State $fresh
    $freshKey = Get-GuardContentKey -State $fresh
    if ([string]::IsNullOrWhiteSpace($target) -or $check.Handled -or $check.Reason -cne $Reason -or [string]$check.Pod -cne $target -or $freshKey -cne $script:GuardKey) {
        if ($check.Handled) {
            Clear-GuardState -Observed $check
        }
        else {
            $script:GuardSince = [DateTimeOffset]::UtcNow
            $script:GuardReason = [string]$check.Reason
            $script:GuardPodId = [string]$check.Pod
            $script:GuardKey = $freshKey
            Write-GuardReceipt -ReceiptState "unhandled_waiting" -Reason $script:GuardReason -Observed $check
        }
        return $false
    }
    $firedAt = [DateTimeOffset]::UtcNow
    $stopReason = "unhandled_timeout_$Reason"
    $observed = @{ Branch = [string]$check.Branch; Gate = [string]$check.Gate; Pod = $target }
    if (-not $script:LiveGuard) {
        Write-GuardReceipt -ReceiptState "would_stop" -Reason $stopReason -Observed $observed -Extra @{ fired_at = $firedAt.ToString("o"); attempt_count = 0 }
        return $true
    }
    if ([string]::IsNullOrWhiteSpace($env:RUNPOD_API_KEY)) {
        Write-GuardReceipt -ReceiptState "stop_failed" -Reason $stopReason -Observed $observed -Extra @{ fired_at = $firedAt.ToString("o"); attempt_count = 0; failure = "missing_local_runpod_api_key" }
        throw "RUNPOD_API_KEY disappeared from the local watchdog environment before a confirmed live stop."
    }
    $attemptCount = 0
    while ($attemptCount -lt $MaxStopAttempts) {
        $attemptCount += 1
        $attemptedAt = [DateTimeOffset]::UtcNow
        $stop = Invoke-Captured -FilePath $RunPodCliPath -ArgumentList @("pod", "stop", $target, "--output", "json") -TimeoutSeconds 30
        $podStatus = Get-121PodStatus -TargetPodId $target
        $runtimeStatus = [string]$podStatus.runtime_status
        if (-not [string]::IsNullOrWhiteSpace($runtimeStatus) -and $runtimeStatus.ToLowerInvariant() -in @("stopped", "exited", "terminated")) {
            Write-GuardReceipt -ReceiptState "stop_confirmed" -Reason $stopReason -Observed $observed -Extra @{ fired_at = $firedAt.ToString("o"); confirmed_at = [DateTimeOffset]::UtcNow.ToString("o"); attempt_count = $attemptCount; stop_exit_code = $stop.ExitCode; pod_status = $podStatus }
            return $true
        }
        Write-GuardReceipt -ReceiptState "stop_retrying" -Reason $stopReason -Observed $observed -Extra @{ fired_at = $firedAt.ToString("o"); last_attempt_at = $attemptedAt.ToString("o"); attempt_count = $attemptCount; stop_exit_code = $stop.ExitCode; pod_status = $podStatus }
        Start-Sleep -Seconds $PollSeconds
    }
    Write-GuardReceipt -ReceiptState "stop_failed" -Reason $stopReason -Observed $observed -Extra @{ fired_at = $firedAt.ToString("o"); attempt_count = $attemptCount; failure = "stop_unconfirmed_after_bounded_retries" }
    return $true
}

if ($script:LiveGuard -and -not (Test-Path -LiteralPath $RunPodCliPath -PathType Leaf)) {
    throw "RunPod CLI not found: $RunPodCliPath"
}
$initialState = Read-GateState -Path $StatePath
$initialClassification = Get-GuardClassification -State $initialState
Write-GuardReceipt -ReceiptState "armed" -Observed $initialClassification

$priorKey = $null
$lastState = $null
$lastSummary = [DateTime]::UtcNow
$started = $false
while ($true) {
    $state = Read-GateState -Path $StatePath
    $key = Get-TransitionKey -State $state
    if ($null -ne $key) { $lastState = $state }
    if (-not $started) {
        $started = $true
        if ($null -ne $key) {
            $priorKey = $key
            Write-Output "issue-121 watching: $key"
        } else {
            Write-Output "issue-121 waiting: $StatePath (missing or incomplete)"
        }
    } elseif ($null -ne $key -and $key -ne $priorKey) {
        $priorKey = $key
        Write-Output "issue-121 transition: $key"
        if (-not $Silent) { Invoke-GateBeep }
    }
    $classification = Get-GuardClassification -State $state
    $contentKey = Get-GuardContentKey -State $state
    if ($classification.Handled) {
        if ($null -ne $script:GuardSince) {
            Clear-GuardState -Observed $classification
        }
    } else {
        $now = [DateTimeOffset]::UtcNow
        if ($null -eq $script:GuardSince -or [string]$classification.Reason -cne [string]$script:GuardReason -or $contentKey -cne $script:GuardKey -or [string]$classification.Pod -cne [string]$script:GuardPodId) {
            $script:GuardSince = $now
            $script:GuardReason = [string]$classification.Reason
            $script:GuardPodId = [string]$classification.Pod
            $script:GuardKey = $contentKey
        }
        $ageSeconds = ([DateTimeOffset]::UtcNow - $script:GuardSince).TotalSeconds
        if ($ageSeconds -ge $UnhandledSeconds) {
            if (Stop-121Pod -Reason $script:GuardReason) { break }
        } else {
            Write-GuardReceipt -ReceiptState "unhandled_waiting" -Reason $script:GuardReason -Observed $classification
        }
    }
    if ($Once) { break }
    if (([DateTime]::UtcNow - $lastSummary).TotalSeconds -ge $SummarySeconds) {
        $lastSummary = [DateTime]::UtcNow
        $utc = [DateTimeOffset]::UtcNow.ToString("HH:mm:ss'Z'")
        Write-Output ((Get-StatusLine -State $lastState -Timestamp $utc) + (Get-GuardStatusText))
    }
    Start-Sleep -Seconds $PollSeconds
}
