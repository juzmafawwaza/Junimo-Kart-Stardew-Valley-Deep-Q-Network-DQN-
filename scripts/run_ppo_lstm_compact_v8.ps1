param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8765,
    [int]$Timesteps = 1000000,
    [int]$Episodes = 5000,
    [int]$EpisodeOffset = 0,
    [string]$ModelPath = "models\ppo_lstm\junimo_ppo_lstm_compact_v8",
    [string]$LoadModel = "",
    [string]$LogDir = "logs\ppo_lstm",
    [string]$RunName = "ppo_lstm_compact_shaped_v5_binary_5k",
    [int]$SaveFreq = 0,
    [int]$SaveEpisodeFreq = 500,
    [int]$ProgressEpisodeFreq = 0,
    [int]$ProgressWindow = 100,
    [int]$TracePrintFreq = 0,
    [ValidateSet("full", "simple")][string]$TraceFormat = "simple",
    [switch]$TraceSimpleRaw,
    [switch]$TraceSimpleAction,
    [string]$TraceCsv = "",
    [int]$TraceCsvFreq = 10,
    [int]$TraceMaxRows = 0,
    [int]$FrameSkip = 1,
    [int]$GapLandingConfirmSteps = 2,
    [double]$GapLandingBaseReward = 5.0,
    [double]$GapLandingWidthCoef = 0.015,
    [double]$ProgressRewardCoef = 0.01,
    [double]$DeathPenalty = 5.0,
    [double]$LevelCompleteReward = 50.0,
    [double]$GameCompleteReward = 200.0,
    [double]$CoinRewardValue = 0.2,
    [double]$FruitRewardValue = 2.0,
    [double]$LearningRate = 0.00025,
    [int]$NSteps = 512,
    [int]$BatchSize = 64,
    [int]$NEpochs = 3,
    [double]$Gamma = 0.99,
    [double]$GaeLambda = 0.95,
    [double]$ClipRange = 0.2,
    [double]$EntropyCoef = 0.003,
    [double]$VfCoef = 0.5,
    [double]$MaxGradNorm = 0.5,
    [int]$LstmHiddenSize = 64,
    [int]$NLstmLayers = 1,
    [switch]$PrintCommand,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$arguments = [System.Collections.Generic.List[string]]::new()
$arguments.Add(".\scripts\train_ppo_lstm.py")
$arguments.AddRange([string[]]@(
    "--host", $HostAddress,
    "--port", "$Port",
    "--timesteps", "$Timesteps",
    "--episode-offset", "$EpisodeOffset",
    "--model-path", $ModelPath,
    "--log-dir", $LogDir,
    "--run-name", $RunName,
    "--save-freq", "$SaveFreq",
    "--save-episode-freq", "$SaveEpisodeFreq",
    "--progress-episode-freq", "$ProgressEpisodeFreq",
    "--progress-window", "$ProgressWindow",
    "--trace-state-print-freq", "$TracePrintFreq",
    "--trace-state-format", $TraceFormat,
    "--trace-state-csv-freq", "$TraceCsvFreq",
    "--trace-state-max-rows", "$TraceMaxRows",
    "--frame-skip", "$FrameSkip",
    "--observation-mode", "compact",
    "--gap-detection-mode", "legacy",
    "--reward-version", "shaped_v5",
    "--action-mode", "binary",
    "--score-reward-coef", "0",
    "--gap-landing-confirm-steps", "$GapLandingConfirmSteps",
    "--gap-landing-base-reward", "$GapLandingBaseReward",
    "--gap-landing-width-coef", "$GapLandingWidthCoef",
    "--progress-reward-coef", "$ProgressRewardCoef",
    "--death-penalty", "$DeathPenalty",
    "--level-complete-reward", "$LevelCompleteReward",
    "--game-complete-reward", "$GameCompleteReward",
    "--coin-reward-value", "$CoinRewardValue",
    "--fruit-reward-value", "$FruitRewardValue",
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
))

if ($Episodes -ge 0) { $arguments.AddRange([string[]]@("--episodes", "$Episodes")) }
if ($LoadModel) { $arguments.AddRange([string[]]@("--load-model", $LoadModel)) }
if ($TraceCsv) { $arguments.AddRange([string[]]@("--trace-state-csv", $TraceCsv)) }
if ($TraceSimpleRaw) { $arguments.Add("--trace-state-simple-raw") }
if ($TraceSimpleAction) { $arguments.Add("--trace-state-simple-action") }

$display = "python " + (($arguments | ForEach-Object { if ($_ -match '\s') { '"' + $_ + '"' } else { $_ } }) -join " ")
if ($PrintCommand -or $DryRun) { Write-Host $display }
if ($DryRun) { return }

Write-Host "Starting a fresh compact PPO-LSTM v8 run (27 inputs, shaped_v5, binary control)."
& python @arguments
exit $LASTEXITCODE
