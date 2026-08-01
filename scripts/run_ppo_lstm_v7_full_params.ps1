param(
    # Bridge / game connection
    [string] $HostAddress = "127.0.0.1",
    [int] $Port = 8765,

    # Training duration
    [int] $Timesteps = 100000,
    # Set Episodes to -1 to disable the episode target and train by Timesteps only.
    [int] $Episodes = 5000,
    [int] $EpisodeOffset = 0,

    # Save/load paths
    [string] $ModelPath = "models\ppo_lstm\junimo_ppo_lstm_v7_full",
    [string] $LoadModel = "models\ppo_lstm\junimo_ppo_lstm_multiinput_binary_v6.zip",
    [string] $LogDir = "logs\ppo_lstm",
    [string] $RunName = "ppo_lstm_v7_full_params_5k",

    # Checkpointing / progress
    [int] $SaveFreq = 0,
    [int] $SaveEpisodeFreq = 1000,
    [int] $ProgressEpisodeFreq = 0,
    [int] $ProgressWindow = 20,

    # Live state trace
    [int] $TracePrintFreq = 0,
    [ValidateSet("full", "simple")]
    [string] $TraceFormat = "simple",
    [bool] $TraceSimpleRaw = $false,
    [bool] $TraceSimpleAction = $true,
    [string] $TraceCsv = "",
    [int] $TraceCsvFreq = 1,
    [int] $TraceMaxRows = 0,

    # Environment / observation
    [int] $FrameSkip = 2,
    [ValidateSet("flat", "multi")]
    [string] $ObservationMode = "multi",
    [int] $RecentActionHistory = 12,
    [bool] $SemanticFeatures = $true,
    [bool] $TemporalFeatures = $true,

    # Reward / action design
    [ValidateSet("legacy", "shaped_v1", "shaped_v2", "shaped_v3", "shaped_v4")]
    [string] $RewardVersion = "shaped_v4",
    [ValidateSet("binary", "macro", "tap_macro")]
    [string] $ActionMode = "binary",
    [int] $MacroActionFrames = 8,
    [int] $MacroReleaseFrames = 1,
    [double] $ScoreRewardCoef = 0.0,
    [double] $CoinRewardCoef = 0.0005,
    [double] $FruitRewardCoef = 0.005,
    [double] $FruitScoreThreshold = 100.0,
    [int] $GapLandingConfirmSteps = 2,
    [double] $GapLandingBaseReward = 12.0,
    [double] $GapLandingWidthCoef = 0.05,

    # PPO hyperparameters
    [double] $LearningRate = 0.0001,
    [int] $NSteps = 512,
    [int] $BatchSize = 64,
    [int] $NEpochs = 5,
    [double] $Gamma = 0.99,
    [double] $GaeLambda = 0.95,
    [double] $ClipRange = 0.2,
    [double] $EntropyCoef = 0.01,
    [double] $VfCoef = 0.5,
    [double] $MaxGradNorm = 0.5,

    # LSTM architecture
    [int] $LstmHiddenSize = 128,
    [int] $NLstmLayers = 1,

    # Helper only for this launcher
    [bool] $PrintCommand = $true,
    [switch] $DryRun
)

$trainArgs = @(
    ".\scripts\train_ppo_lstm.py",
    "--host", "$HostAddress",
    "--port", "$Port",
    "--timesteps", "$Timesteps",
    "--episode-offset", "$EpisodeOffset",
    "--model-path", "$ModelPath",
    "--log-dir", "$LogDir",
    "--run-name", "$RunName",
    "--save-freq", "$SaveFreq",
    "--save-episode-freq", "$SaveEpisodeFreq",
    "--progress-episode-freq", "$ProgressEpisodeFreq",
    "--progress-window", "$ProgressWindow",
    "--frame-skip", "$FrameSkip",
    "--observation-mode", "$ObservationMode",
    "--recent-action-history", "$RecentActionHistory",
    "--reward-version", "$RewardVersion",
    "--action-mode", "$ActionMode",
    "--macro-action-frames", "$MacroActionFrames",
    "--macro-release-frames", "$MacroReleaseFrames",
    "--score-reward-coef", "$ScoreRewardCoef",
    "--coin-reward-coef", "$CoinRewardCoef",
    "--fruit-reward-coef", "$FruitRewardCoef",
    "--fruit-score-threshold", "$FruitScoreThreshold",
    "--gap-landing-confirm-steps", "$GapLandingConfirmSteps",
    "--gap-landing-base-reward", "$GapLandingBaseReward",
    "--gap-landing-width-coef", "$GapLandingWidthCoef",
    "--learning-rate", "$LearningRate",
    "--n-steps", "$NSteps",
    "--batch-size", "$BatchSize",
    "--n-epochs", "$NEpochs",
    "--gamma", "$Gamma",
    "--gae-lambda", "$GaeLambda",
    "--clip-range", "$ClipRange",
    "--ent-coef", "$EntropyCoef",
    "--vf-coef", "$VfCoef",
    "--max-grad-norm", "$MaxGradNorm",
    "--lstm-hidden-size", "$LstmHiddenSize",
    "--n-lstm-layers", "$NLstmLayers"
)

if ($Episodes -ge 0) {
    $trainArgs += @("--episodes", "$Episodes")
}

if ($SemanticFeatures) {
    $trainArgs += @("--semantic-features")
}

if ($TemporalFeatures) {
    $trainArgs += @("--temporal-features")
}

if ($TracePrintFreq -gt 0) {
    $trainArgs += @(
        "--trace-state-print-freq", "$TracePrintFreq",
        "--trace-state-format", "$TraceFormat"
    )

    if ($TraceSimpleRaw) {
        $trainArgs += @("--trace-state-simple-raw")
    }

    if ($TraceSimpleAction) {
        $trainArgs += @("--trace-state-simple-action")
    }
}

if (-not [string]::IsNullOrWhiteSpace($TraceCsv)) {
    $trainArgs += @(
        "--trace-state-csv", "$TraceCsv",
        "--trace-state-csv-freq", "$TraceCsvFreq",
        "--trace-state-max-rows", "$TraceMaxRows"
    )

    if ($TracePrintFreq -le 0) {
        $trainArgs += @("--trace-state-format", "$TraceFormat")
    }
}

if (-not [string]::IsNullOrWhiteSpace($LoadModel)) {
    if (Test-Path $LoadModel) {
        Write-Host "Loading model: $LoadModel"
        $trainArgs += @("--load-model", "$LoadModel")
    } else {
        Write-Host "LoadModel was set but file was not found: $LoadModel"
        Write-Host "Starting from scratch. Set -LoadModel '' if this is intentional."
    }
} else {
    Write-Host "LoadModel is empty. Starting from scratch."
}

Write-Host "Run name: $RunName"
Write-Host "Model path: $ModelPath"
Write-Host "Log dir: $LogDir\$RunName"
Write-Host "Reward: $RewardVersion | action: $ActionMode | observation: $ObservationMode"
Write-Host "Coin coef: $CoinRewardCoef | fruit coef: $FruitRewardCoef"
Write-Host "Gap landing reward: $GapLandingBaseReward + $GapLandingWidthCoef * min(gap_width, 120)"

if ($PrintCommand) {
    Write-Host "Command:"
    Write-Host ("python " + ($trainArgs -join " "))
}

if ($DryRun) {
    Write-Host "DryRun enabled. Command was printed but training was not started."
    exit 0
}

python @trainArgs
