param(
    [string[]]$Models = @(
        ".\logs\ppo\ppo_compact_anchored_v10_fresh_3k\checkpoints\junimo_ppo_ep000500_steps*.zip",
        ".\logs\ppo\ppo_compact_tip_v11_fresh_3k\checkpoints\junimo_ppo_ep*.zip"
    ),
    [int]$Episodes = 20,
    [string]$Out = "logs\ppo\ppo_compact_tip_v11_fresh_3k\evaluation_deterministic_20ep.csv",
    [int]$MaxStepsPerEpisode = 300,
    [switch]$Stochastic
)

$ErrorActionPreference = "Stop"

$arguments = [System.Collections.Generic.List[string]]::new()
$arguments.Add(".\scripts\evaluate_ppo_models.py")
foreach ($model in $Models) { $arguments.Add($model) }
$arguments.AddRange([string[]]@(
    "--episodes", "$Episodes",
    "--frame-skip", "1",
    "--observation-mode", "compact",
    "--gap-detection-mode", "anchored",
    "--reward-version", "shaped_v8",
    "--action-mode", "binary",
    "--score-reward-coef", "0",
    "--gap-landing-confirm-steps", "2",
    "--gap-landing-base-reward", "5.0",
    "--gap-landing-width-coef", "0.015",
    "--progress-reward-coef", "0.01",
    "--death-penalty", "5.0",
    "--jump-start-penalty", "0.05",
    "--gap-miss-penalty-coef", "2.0",
    "--airborne-hold-free-steps", "4",
    "--airborne-hold-penalty", "0.02",
    "--grounded-progress-bonus-coef", "0.005",
    "--unnecessary-jump-penalty", "0.15",
    "--non-gap-airborne-penalty", "0.01",
    "--gap-tip-technique-reward", "1.5",
    "--takeoff-tip-target-distance", "12.0",
    "--takeoff-tip-tolerance", "48.0",
    "--landing-tip-target-depth", "16.0",
    "--landing-tip-tolerance", "64.0",
    "--level-complete-reward", "50.0",
    "--game-complete-reward", "200.0",
    "--coin-reward-value", "0.2",
    "--fruit-reward-value", "2.0",
    "--max-steps-per-episode", "$MaxStepsPerEpisode",
    "--out", $Out
))
if ($Stochastic) { $arguments.Add("--stochastic") }

& python @arguments
exit $LASTEXITCODE
