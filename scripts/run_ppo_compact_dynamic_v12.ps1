param(
    [string]$HostAddress = "127.0.0.1",
    [int]$Port = 8765,
    [int]$Episodes = 3000,
    [int]$EpisodeOffset = 0,
    [string]$LoadModel = "",
    [string]$ModelPath = "models\ppo\junimo_ppo_compact_dynamic_v12_fresh_3k",
    [string]$LogDir = "logs\ppo",
    [string]$RunName = "ppo_compact_dynamic_v12_fresh_3k",
    [int]$SaveEpisodeFreq = 250,
    [int]$FrameSkip = 1,
    [int]$TraceCsvFreq = 5,
    [int]$TraceMaxRows = 0,
    [int]$GapLandingConfirmSteps = 2,
    [double]$GapLandingBaseReward = 6.0,
    [double]$GapLandingWidthCoef = 0.02,
    [double]$ProgressRewardCoef = 0.0075,
    [double]$DeathPenalty = 6.0,
    [double]$JumpStartPenalty = 0.03,
    [double]$GapMissPenaltyCoef = 2.0,
    [int]$AirborneHoldFreeSteps = 6,
    [double]$AirborneHoldPenalty = 0.01,
    [double]$GroundedProgressBonusCoef = 0.002,
    [double]$NonGapAirbornePenalty = 0.005,
    [double]$GapTechniqueReward = 0.5,
    [double]$LandingTargetDepth = 16.0,
    [double]$LandingTolerance = 96.0,
    [double]$TakeoffWidthCoef = 0.65,
    [double]$TakeoffUphillCoef = 0.25,
    [double]$TakeoffDownhillCoef = 0.10,
    [double]$TakeoffMinDistance = 32.0,
    [double]$TakeoffMaxDistance = 112.0,
    [double]$TakeoffTolerance = 64.0,
    [double]$GapInactionMargin = 24.0,
    [double]$GapInactionPenalty = 0.05,
    [double]$LevelCompleteReward = 50.0,
    [double]$GameCompleteReward = 200.0,
    [double]$CoinRewardValue = 0.2,
    [double]$FruitRewardValue = 2.0,
    [double]$LearningRate = 0.0002,
    [int]$NSteps = 2048,
    [int]$BatchSize = 128,
    [int]$NEpochs = 5,
    [double]$Gamma = 0.99,
    [double]$GaeLambda = 0.95,
    [double]$ClipRange = 0.2,
    [double]$EntropyCoef = 0.02,
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
    "--reward-version", "shaped_v9",
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
    "--non-gap-airborne-penalty", "$NonGapAirbornePenalty",
    "--gap-tip-technique-reward", "$GapTechniqueReward",
    "--landing-tip-target-depth", "$LandingTargetDepth",
    "--landing-tip-tolerance", "$LandingTolerance",
    "--takeoff-target-width-coef", "$TakeoffWidthCoef",
    "--takeoff-target-uphill-coef", "$TakeoffUphillCoef",
    "--takeoff-target-downhill-coef", "$TakeoffDownhillCoef",
    "--takeoff-target-min-distance", "$TakeoffMinDistance",
    "--takeoff-target-max-distance", "$TakeoffMaxDistance",
    "--takeoff-dynamic-tolerance", "$TakeoffTolerance",
    "--gap-inaction-margin", "$GapInactionMargin",
    "--gap-inaction-penalty", "$GapInactionPenalty",
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

if ($TraceCsvFreq -gt 0) {
    $tracePath = Join-Path (Join-Path $LogDir $RunName) "step_trace.csv"
    $arguments.Add("--trace-state-csv")
    $arguments.Add($tracePath)
    $arguments.Add("--trace-state-csv-freq")
    $arguments.Add("$TraceCsvFreq")
    $arguments.Add("--trace-state-max-rows")
    $arguments.Add("$TraceMaxRows")
}

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
    Write-Warning "Warm-starting from $LoadModel can retain its learned jump/no-jump habit. Fresh v12 is recommended."
} else {
    Write-Host "Starting fresh shaped_v9 PPO with dynamic takeoff timing and collapse-resistant exploration."
}
if ($TraceCsvFreq -gt 0) {
    Write-Host "Per-step trace sampling: one row every $TraceCsvFreq environment steps."
}
& python @arguments
exit $LASTEXITCODE

