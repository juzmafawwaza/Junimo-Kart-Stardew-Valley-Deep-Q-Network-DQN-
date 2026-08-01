param(
    [string]$Models = ".\logs\ppo_lstm\ppo_lstm_compact_shaped_v5_binary_5k\checkpoints\junimo_ppo_lstm_ep*.zip",
    [int]$Episodes = 100,
    [string]$Out = "logs\ppo_lstm\ppo_lstm_compact_shaped_v5_binary_5k\evaluation_100ep.csv",
    [switch]$Stochastic
)

$arguments = [System.Collections.Generic.List[string]]::new()
$arguments.AddRange([string[]]@(
    ".\scripts\evaluate_ppo_lstm_models.py", $Models,
    "--episodes", "$Episodes",
    "--frame-skip", "1",
    "--observation-mode", "compact",
    "--gap-detection-mode", "legacy",
    "--reward-version", "shaped_v5",
    "--action-mode", "binary",
    "--score-reward-coef", "0",
    "--gap-landing-confirm-steps", "2",
    "--gap-landing-base-reward", "5",
    "--gap-landing-width-coef", "0.015",
    "--progress-reward-coef", "0.01",
    "--death-penalty", "5",
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
