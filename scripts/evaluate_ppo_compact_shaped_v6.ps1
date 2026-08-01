param(
    [string]$Models = ".\logs\ppo\ppo_compact_shaped_v6_from_1500_to_3k\checkpoints\junimo_ppo_ep*.zip",
    [int]$Episodes = 20,
    [string]$Out = "logs\ppo\ppo_compact_shaped_v6_from_1500_to_3k\evaluation_deterministic_20ep.csv",
    [double]$JumpStartPenalty = 0.02,
    [double]$GapMissPenaltyCoef = 2.0,
    [switch]$Stochastic
)

$arguments = [System.Collections.Generic.List[string]]::new()
$arguments.AddRange([string[]]@(
    ".\scripts\evaluate_ppo_models.py", $Models,
    "--episodes", "$Episodes",
    "--frame-skip", "1",
    "--observation-mode", "compact",
    "--gap-detection-mode", "legacy",
    "--reward-version", "shaped_v6",
    "--action-mode", "binary",
    "--score-reward-coef", "0",
    "--gap-landing-confirm-steps", "2",
    "--gap-landing-base-reward", "5",
    "--gap-landing-width-coef", "0.015",
    "--progress-reward-coef", "0.01",
    "--death-penalty", "5",
    "--jump-start-penalty", "$JumpStartPenalty",
    "--gap-miss-penalty-coef", "$GapMissPenaltyCoef",
    "--level-complete-reward", "50",
    "--game-complete-reward", "200",
    "--coin-reward-value", "0.2",
    "--fruit-reward-value", "2",
    "--max-steps-per-episode", "1000",
    "--out", $Out
))
if ($Stochastic) { $arguments.Add("--stochastic") }

& python @arguments
exit $LASTEXITCODE
