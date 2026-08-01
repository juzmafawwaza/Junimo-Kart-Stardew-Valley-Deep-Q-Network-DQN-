param(
    [int] $Episodes = 200,
    [int] $SaveEpisodeFreq = 100,
    [int] $FrameSkip = 2,
    [int] $RecentActionHistory = 12,
    [double] $LearningRate = 0.0001,
    [double] $EntropyCoef = 0.01,
    [double] $ScoreRewardCoef = 0.0,
    [int] $GapLandingConfirmSteps = 2,
    [int] $LstmHiddenSize = 128,
    [int] $NLstmLayers = 1,
    [int] $NSteps = 256,
    [int] $NEpochs = 3,
    [int] $BatchSize = 64,
    [int] $TracePrintFreq = 1,
    [int] $TraceCsvFreq = 1,
    [int] $TraceMaxRows = 0,
    [string] $TraceFormat = "simple",
    [switch] $TraceSimpleRaw,
    [string] $RunName = "ppo_lstm_multiinput_binary_v6_trace_debug",
    [string] $ModelPath = "models\ppo_lstm\junimo_ppo_lstm_multiinput_binary_v6_trace_debug",
    [string] $TraceCsv = "logs\ppo_lstm\ppo_lstm_multiinput_binary_v6_trace_debug\state_trace.csv",
    [string] $LoadModel = "",
    [int] $EpisodeOffset = -1,
    [switch] $NoAutoResume
)

$resolvedLoadModel = $LoadModel
if (-not $NoAutoResume -and [string]::IsNullOrWhiteSpace($resolvedLoadModel)) {
    $checkpointCandidates = @(
        Get-ChildItem -Path "logs\ppo_lstm" -Recurse -Filter "junimo_ppo_lstm_ep*.zip" -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -like "*multiinput*binary*v6*" } |
            Sort-Object LastWriteTime -Descending
    )
    if ($checkpointCandidates.Count -gt 0) {
        $resolvedLoadModel = $checkpointCandidates[0].FullName
    }
}
if (-not $NoAutoResume -and [string]::IsNullOrWhiteSpace($resolvedLoadModel)) {
    $fallbackModel = "models\ppo_lstm\junimo_ppo_lstm_multiinput_binary_v6.zip"
    if (Test-Path $fallbackModel) {
        $resolvedLoadModel = $fallbackModel
    }
}

$resolvedEpisodeOffset = $EpisodeOffset
if ($resolvedEpisodeOffset -lt 0) {
    $resolvedEpisodeOffset = 0
    if (-not [string]::IsNullOrWhiteSpace($resolvedLoadModel)) {
        $fileName = [System.IO.Path]::GetFileName($resolvedLoadModel)
        if ($fileName -match "ep(\d+)") {
            $resolvedEpisodeOffset = [int]$Matches[1]
        }
    }
}

Write-Host "PPO-LSTM MultiInput v6 trace debug"
Write-Host "Run name: $RunName"
Write-Host "Episodes this run: $Episodes"
Write-Host "Episode offset: $resolvedEpisodeOffset"
Write-Host "Trace print freq: $TracePrintFreq"
Write-Host "Trace format: $TraceFormat"
Write-Host "Trace CSV: $TraceCsv"
Write-Host "n_steps: $NSteps"
Write-Host "n_epochs: $NEpochs"
if (-not [string]::IsNullOrWhiteSpace($resolvedLoadModel)) {
    Write-Host "Loading model: $resolvedLoadModel"
} else {
    Write-Host "No v6 checkpoint found. Starting a new debug model."
}

$trainArgs = @(
    ".\scripts\train_ppo_lstm.py",
    "--episodes", "$Episodes",
    "--episode-offset", "$resolvedEpisodeOffset",
    "--save-episode-freq", "$SaveEpisodeFreq",
    "--save-freq", "0",
    "--frame-skip", "$FrameSkip",
    "--observation-mode", "multi",
    "--recent-action-history", "$RecentActionHistory",
    "--semantic-features",
    "--temporal-features",
    "--reward-version", "shaped_v3",
    "--action-mode", "binary",
    "--score-reward-coef", "$ScoreRewardCoef",
    "--gap-landing-confirm-steps", "$GapLandingConfirmSteps",
    "--learning-rate", "$LearningRate",
    "--ent-coef", "$EntropyCoef",
    "--lstm-hidden-size", "$LstmHiddenSize",
    "--n-lstm-layers", "$NLstmLayers",
    "--n-steps", "$NSteps",
    "--n-epochs", "$NEpochs",
    "--batch-size", "$BatchSize",
    "--model-path", "$ModelPath",
    "--run-name", "$RunName"
)

if ($TracePrintFreq -gt 0) {
    $trainArgs += @(
        "--trace-state-print-freq", "$TracePrintFreq",
        "--trace-state-format", "$TraceFormat",
        "--trace-state-simple-action"
    )
}

if (-not [string]::IsNullOrWhiteSpace($TraceCsv)) {
    $trainArgs += @(
        "--trace-state-csv", "$TraceCsv",
        "--trace-state-csv-freq", "$TraceCsvFreq",
        "--trace-state-max-rows", "$TraceMaxRows"
    )
}

if ($TraceSimpleRaw) {
    $trainArgs += @("--trace-state-simple-raw")
}

if (-not [string]::IsNullOrWhiteSpace($resolvedLoadModel)) {
    $trainArgs += @("--load-model", "$resolvedLoadModel")
}

python @trainArgs
