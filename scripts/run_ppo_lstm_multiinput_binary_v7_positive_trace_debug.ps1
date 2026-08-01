param(
    [int] $Episodes = 200,
    [int] $SaveEpisodeFreq = 100,
    [int] $FrameSkip = 2,
    [int] $RecentActionHistory = 12,
    [double] $LearningRate = 0.0001,
    [double] $EntropyCoef = 0.01,
    [double] $ScoreRewardCoef = 0.0,
    [double] $CoinRewardCoef = 0.0005,
    [double] $FruitRewardCoef = 0.003,
    [double] $FruitScoreThreshold = 100.0,
    [int] $GapLandingConfirmSteps = 2,
    [double] $GapLandingBaseReward = 8.0,
    [double] $GapLandingWidthCoef = 0.04,
    [int] $LstmHiddenSize = 128,
    [int] $NLstmLayers = 1,
    [int] $NSteps = 256,
    [int] $NEpochs = 3,
    [int] $BatchSize = 64,
    [int] $TracePrintFreq = 1,
    [int] $TraceCsvFreq = 1,
    [int] $TraceMaxRows = 0,
    [string] $RunName = "ppo_lstm_multiinput_binary_v7_positive_trace_debug",
    [string] $ModelPath = "models\ppo_lstm\junimo_ppo_lstm_multiinput_binary_v7_positive_trace_debug",
    [string] $TraceCsv = "logs\ppo_lstm\ppo_lstm_multiinput_binary_v7_positive_trace_debug\state_trace.csv",
    [string] $LoadModel = "models\ppo_lstm\junimo_ppo_lstm_multiinput_binary_v6.zip"
)

$trainArgs = @(
    ".\scripts\train_ppo_lstm.py",
    "--episodes", "$Episodes",
    "--save-episode-freq", "$SaveEpisodeFreq",
    "--save-freq", "0",
    "--frame-skip", "$FrameSkip",
    "--observation-mode", "multi",
    "--recent-action-history", "$RecentActionHistory",
    "--semantic-features",
    "--temporal-features",
    "--reward-version", "shaped_v4",
    "--action-mode", "binary",
    "--score-reward-coef", "$ScoreRewardCoef",
    "--coin-reward-coef", "$CoinRewardCoef",
    "--fruit-reward-coef", "$FruitRewardCoef",
    "--fruit-score-threshold", "$FruitScoreThreshold",
    "--gap-landing-confirm-steps", "$GapLandingConfirmSteps",
    "--gap-landing-base-reward", "$GapLandingBaseReward",
    "--gap-landing-width-coef", "$GapLandingWidthCoef",
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
        "--trace-state-format", "simple",
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

if (-not [string]::IsNullOrWhiteSpace($LoadModel) -and (Test-Path $LoadModel)) {
    Write-Host "Loading model: $LoadModel"
    $trainArgs += @("--load-model", "$LoadModel")
} else {
    Write-Host "No load model found. Starting v7 positive debug from scratch."
}

python @trainArgs
