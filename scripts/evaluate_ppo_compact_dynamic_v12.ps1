param(
    [string[]]$Models = @(
        ".\logs\ppo\ppo_compact_anchored_v10_fresh_3k\checkpoints\junimo_ppo_ep001000_steps*.zip",
        ".\logs\ppo\ppo_compact_tip_v11_fresh_3k\checkpoints\junimo_ppo_ep000500_steps*.zip",
        ".\logs\ppo\ppo_compact_tip_v11_fresh_3k\checkpoints\junimo_ppo_ep001000_steps*.zip",
        ".\logs\ppo\ppo_compact_dynamic_v12_fresh_3k\checkpoints\junimo_ppo_ep*.zip"
    ),
    [int]$Episodes = 20,
    [string]$Out = "logs\ppo\ppo_compact_dynamic_v12_fresh_3k\evaluation_deterministic_20ep.csv",
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
    "--reward-version", "shaped_v9",
    "--action-mode", "binary",
    "--score-reward-coef", "0",
    "--gap-landing-confirm-steps", "2",
    "--gap-landing-base-reward", "6.0",
    "--gap-landing-width-coef", "0.02",
    "--progress-reward-coef", "0.0075",
    "--death-penalty", "6.0",
    "--jump-start-penalty", "0.03",
    "--gap-miss-penalty-coef", "2.0",
    "--airborne-hold-free-steps", "6",
    "--airborne-hold-penalty", "0.01",
    "--grounded-progress-bonus-coef", "0.002",
    "--non-gap-airborne-penalty", "0.005",
    "--gap-tip-technique-reward", "0.5",
    "--landing-tip-target-depth", "16.0",
    "--landing-tip-tolerance", "96.0",
    "--takeoff-target-width-coef", "0.65",
    "--takeoff-target-uphill-coef", "0.25",
    "--takeoff-target-downhill-coef", "0.10",
    "--takeoff-target-min-distance", "32.0",
    "--takeoff-target-max-distance", "112.0",
    "--takeoff-dynamic-tolerance", "64.0",
    "--gap-inaction-margin", "24.0",
    "--gap-inaction-penalty", "0.05",
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
