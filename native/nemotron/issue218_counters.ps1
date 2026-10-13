param([int]$Samples = 180)
$ErrorActionPreference = 'Stop'
$Paths = @('\GPU Adapter Memory(*)\Dedicated Usage', '\GPU Adapter Memory(*)\Shared Usage', '\GPU Process Memory(*)\Dedicated Usage', '\GPU Engine(*)\Utilization Percentage')
Get-Counter -Counter $Paths -SampleInterval 1 -MaxSamples $Samples | ForEach-Object {
    $Rows = @($_.CounterSamples | ForEach-Object { @{path=$_.Path; value=$_.CookedValue; status=$_.Status} })
    @{observed_utc=$_.Timestamp.ToUniversalTime().ToString('o'); counters=$Rows} | ConvertTo-Json -Depth 5 -Compress
}
