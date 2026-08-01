param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8765,
    [int]$Episodes = 3000,
    [int]$EpisodeOffset = 0,
    [string]$LoadModel = "",
    [string]$ModelPath = "models\ppo\junimo_ppo_compact_tip_v11_fresh_3k",
    [string]$LogDir = "logs\ppo",
    [string]$RunName = "ppo_compact_tip_v11_fresh_3k",
    [int]$SaveEpisodeFreq = 250,
    [int]$FrameSkip = 1,
    [int]$GapLandingConfirmSteps = 2,
    [double]$GapLandingBaseReward = 5.0,
    [double]$GapLandingWidthCoef = 0.015,
    [double]$ProgressRewardCoef = 0.01,
    [double]$DeathPenalty = 5.0,
    [double]$JumpStartPenalty = 0.05,
    [double]$GapMissPenaltyCoef = 2.0,
    [int]$AirborneHoldFreeSteps = 4,
    [double]$AirborneHoldPenalty = 0.02,
    [double]$GroundedProgressBonusCoef = 0.005,
    [double]$UnnecessaryJumpPenalty = 0.15,
    [double]$NonGapAirbornePenalty = 0.01,
    [double]$GapTipTechniqueReward = 1.5,
    [double]$TakeoffTipTargetDistance = 12.0,
    [double]$TakeoffTipTolerance = 48.0,
    [double]$LandingTipTargetDepth = 16.0,
    [double]$LandingTipTolerance = 64.0,
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
    [double]$EntropyCoef = 0.01,
    [double]$VfCoef = 0.5,
    [double]$MaxGradNorm = 0.5,
    [switch]$PrintCommand,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

if ($LoadModel -and -not (Test-Path -LiteralPath $LoadModel)) {
    throw "Source checkpoint was not found: $LoadModel"
}

$arguments = [System.Collections.Generic.List[string]]::new()
$arguments.AddRange([string[]]@(
    ".\scripts\train_ppo.py",
    "--host", $HostAddress,
    "--port", "$Port",
    "--episodes", "$Episodes",
    "--episode-offset", "$EpisodeOffset",
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
    "--reward-version", "shaped_v8",
    "--action-mode", "binary",
    "--score-reward-coef", "0",
    "--gap-landing-confirm-steps", "$GapLandingConfirmSteps",
    "--gap-landing-base-reward", "$GapLandingBaseReward",
    "--gap-landing-width-coef", "$GapLandingWidthCoef",
    "--progress-reward-coef", "$ProgressRewardCoef",
    "--death-penalty", "$DeathPenalty",
    "--jump-start-penalty", "$JumpStartPenalty",
    "--gap-miss-penalty-coef", "$GapMissPenaltyCoef",
    "--airborne-hold-free-steps", "$AirborneHoldFreeSteps",
    "--airborne-hold-penalty", "$AirborneHoldPenalty",
    "--grounded-progress-bonus-coef", "$GroundedProgressBonusCoef",
    "--unnecessary-jump-penalty", "$UnnecessaryJumpPenalty",
    "--non-gap-airborne-penalty", "$NonGapAirbornePenalty",
    "--gap-tip-technique-reward", "$GapTipTechniqueReward",
    "--takeoff-tip-target-distance", "$TakeoffTipTargetDistance",
    "--takeoff-tip-tolerance", "$TakeoffTipTolerance",
    "--landing-tip-target-depth", "$LandingTipTargetDepth",
    "--landing-tip-tolerance", "$LandingTipTolerance",
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
if ($LoadModel) {
    $arguments.Add("--load-model")
    $arguments.Add($LoadModel)
}

$display = "python " + (($arguments | ForEach-Object {
    if ($_ -match "\s") { '"' + $_ + '"' } else { $_ }
}) -join " ")
if ($PrintCommand -or $DryRun) { Write-Host $display }
if ($DryRun) { return }

if ($LoadModel) {
    Write-Warning "Warm-starting from $LoadModel can retain its learned jump habit. A fresh v11 policy is recommended for a clean comparison."
} else {
    Write-Host "Starting a fresh shaped_v8 PPO policy with confirmed tip technique and anti-bunny-hop shaping."
}
& python @arguments
exit $LASTEXITCODE
