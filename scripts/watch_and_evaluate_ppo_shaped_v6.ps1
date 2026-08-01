param(
    [int]$TrainingPid = 0,
    [string]$RunDir = "logs\ppo\ppo_compact_shaped_v6_from_1500_to_3k",
    [string]$BaselineModel = ".\logs\ppo\ppo_compact_v8_continue_to_3k\checkpoints\junimo_ppo_ep001500_steps124430.zip",
    [int]$Episodes = 20,
    [int]$PollSeconds = 15,
    [string]$Out = "logs\ppo\ppo_compact_shaped_v6_from_1500_to_3k\evaluation_after_1500_v6_deterministic_20ep.csv"
)

$ErrorActionPreference = "Stop"
$PollSeconds = [Math]::Max($PollSeconds, 5)

function Write-Status([string]$Message) {
    Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] $Message"
}

if ($TrainingPid -le 0) {
    $training = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(\.exe)?$' -and
        $_.CommandLine -match 'train_ppo\.py' -and
        $_.CommandLine -match 'ppo_compact_shaped_v6_from_1500_to_3k'
    } | Select-Object -First 1
    if (-not $training) {
        throw "The shaped_v6 training process was not found."
    }
    $TrainingPid = $training.ProcessId
}

Write-Status "Watching shaped_v6 training PID $TrainingPid."
while (Get-Process -Id $TrainingPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds $PollSeconds
}

Write-Status "Training process exited. Verifying the episode-3000 checkpoint."
$checkpointDir = Join-Path $RunDir "checkpoints"
$finalCheckpoint = Get-ChildItem -LiteralPath $checkpointDir -Filter "junimo_ppo_ep003000_steps*.zip" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if (-not $finalCheckpoint) {
    throw "Training stopped before a final episode-3000 checkpoint was saved. Automatic evaluation was not started."
}
if (-not (Test-Path -LiteralPath $BaselineModel)) {
    throw "Baseline model was not found: $BaselineModel"
}

$v6Checkpoints = @(Get-ChildItem -LiteralPath $checkpointDir -Filter "junimo_ppo_ep*.zip" | Sort-Object Name)
$models = @($BaselineModel) + @($v6Checkpoints.FullName)
Write-Status "Starting deterministic evaluation of $($models.Count) models, $Episodes episodes each."

$arguments = [System.Collections.Generic.List[string]]::new()
$arguments.Add(".\scripts\evaluate_ppo_models.py")
$arguments.AddRange([string[]]$models)
$arguments.AddRange([string[]]@(
    "--episodes", "$Episodes",
    "--frame-skip", "1",
    "--observation-mode", "compact",
    "--reward-version", "shaped_v6",
    "--action-mode", "binary",
    "--score-reward-coef", "0",
    "--gap-landing-confirm-steps", "2",
    "--gap-landing-base-reward", "5",
    "--gap-landing-width-coef", "0.015",
    "--progress-reward-coef", "0.01",
    "--death-penalty", "5",
    "--jump-start-penalty", "0.02",
    "--gap-miss-penalty-coef", "2",
    "--level-complete-reward", "50",
    "--game-complete-reward", "200",
    "--coin-reward-value", "0.2",
    "--fruit-reward-value", "2",
    "--max-steps-per-episode", "1000",
    "--out", $Out
))

& python @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Automatic PPO evaluation failed with exit code $LASTEXITCODE."
}
Write-Status "Automatic evaluation completed: $Out"
