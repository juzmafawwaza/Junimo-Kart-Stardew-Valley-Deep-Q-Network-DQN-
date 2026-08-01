param(
    [string[]]$Models = @(
        ".\logs\ppo\ppo_compact_v8_continue_to_3k\checkpoints\junimo_ppo_ep001500_steps124430.zip",
        ".\logs\ppo\ppo_compact_anchored_v10_fresh_3k\checkpoints\junimo_ppo_ep*.zip"
    ),
    [int]$Episodes = 20,
    [string]$Out = "logs\ppo\ppo_compact_anchored_v10_fresh_3k\evaluation_deterministic_20ep.csv",
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
    "--reward-version", "shaped_v7",
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
