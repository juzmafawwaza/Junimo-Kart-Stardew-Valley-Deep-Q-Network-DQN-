param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8765,
    [int]$Episodes = 1500,
    [int]$EpisodeOffset = 1500,
    [string]$LoadModel = ".\logs\ppo\ppo_compact_v8_continue_to_3k\checkpoints\junimo_ppo_ep001500_steps124430.zip",
    [string]$ModelPath = "models\ppo\junimo_ppo_compact_anchored_v9_3k",
    [string]$LogDir = "logs\ppo",
    [string]$RunName = "ppo_compact_anchored_v9_from_1500_to_3k",
    [int]$SaveEpisodeFreq = 250,
    [int]$FrameSkip = 1,
    [int]$GapLandingConfirmSteps = 2,
    [double]$GapLandingBaseReward = 5.0,
    [double]$GapLandingWidthCoef = 0.015,
    [double]$ProgressRewardCoef = 0.01,
    [double]$DeathPenalty = 5.0,
    [double]$JumpStartPenalty = 0.05,
    [double]$GapMissPenaltyCoef = 2.0,
    [double]$LevelCompleteReward = 50.0,
    [double]$GameCompleteReward = 200.0,
    [double]$CoinRewardValue = 0.2,
    [double]$FruitRewardValue = 2.0,
    [double]$LearningRate = 0.0003,
    [int]$NSteps = 1024,
    [int]$BatchSize = 64,
    [int]$NEpochs = 5,
    [double]$Gamma = 0.99,
    [double]$GaeLambda = 0.95,
    [double]$ClipRange = 0.2,
    [double]$EntropyCoef = 0.003,
    [double]$VfCoef = 0.5,
    [double]$MaxGradNorm = 0.5,
    [switch]$PrintCommand,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $LoadModel)) {
    throw "Source checkpoint was not found: $LoadModel"
}

$arguments = [System.Collections.Generic.List[string]]::new()
$arguments.AddRange([string[]]@(
    ".\scripts\train_ppo.py",
    "--host", $HostAddress,
    "--port", "$Port",
    "--episodes", "$Episodes",
    "--episode-offset", "$EpisodeOffset",
    "--load-model", $LoadModel,
    "--model-path", $ModelPath,
    "--log-dir", $LogDir,
    "--run-name", $RunName,
    "--save-freq", "0",
    "--save-episode-freq", "$SaveEpisodeFreq",
    "--progress-episode-freq", "0",
    "--progress-window", "100",
    "--trace-state-print-freq", "0",
    "--frame-skip", "$FrameSkip",
    "--observation-mode", "compact",
    "--gap-detection-mode", "anchored",
    "--reward-version", "shaped_v6",
    "--action-mode", "binary",
    "--score-reward-coef", "0",
    "--gap-landing-confirm-steps", "$GapLandingConfirmSteps",
    "--gap-landing-base-reward", "$GapLandingBaseReward",
    "--gap-landing-width-coef", "$GapLandingWidthCoef",
    "--progress-reward-coef", "$ProgressRewardCoef",
    "--death-penalty", "$DeathPenalty",
    "--jump-start-penalty", "$JumpStartPenalty",
    "--gap-miss-penalty-coef", "$GapMissPenaltyCoef",
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
    "--max-grad-norm", "$MaxGradNorm"
))

$display = "python " + (($arguments | ForEach-Object {
    if ($_ -match "\s") { '"' + $_ + '"' } else { $_ }
}) -join " ")
if ($PrintCommand -or $DryRun) { Write-Host $display }
if ($DryRun) { return }

Write-Host "Continuing checkpoint episode $EpisodeOffset with anchored gap detection and persistent airborne geometry."
& python @arguments
exit $LASTEXITCODE
