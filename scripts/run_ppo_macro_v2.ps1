param(
    [int] $Episodes = 5000,
    [int] $SaveEpisodeFreq = 1000,
    [int] $FrameSkip = 2,
    [int] $MacroActionFrames = 8,
    [string] $RunName = "ppo_semantic_shaped_v2_macro_5k",
    [string] $ModelPath = "models\ppo\junimo_ppo_macro_v2"
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path -LiteralPath (Join-Path $scriptDir "..")
Set-Location -LiteralPath $repoRoot

$trainArgs = @(
    ".\scripts\train_ppo.py",
    "--episodes", $Episodes.ToString(),
    "--save-episode-freq", $SaveEpisodeFreq.ToString(),
    "--save-freq", "0",
    "--frame-skip", $FrameSkip.ToString(),
    "--semantic-features",
    "--reward-version", "shaped_v2",
    "--action-mode", "macro",
    "--macro-action-frames", $MacroActionFrames.ToString(),
    "--model-path", $ModelPath,
    "--run-name", $RunName
)

python @trainArgs
