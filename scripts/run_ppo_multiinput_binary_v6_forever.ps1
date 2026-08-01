param(
    [int64] $Timesteps = 2147483647,
    [int] $SaveEpisodeFreq = 1000,
    [int] $FrameSkip = 2,
    [int] $RecentActionHistory = 12,
    [double] $LearningRate = 0.00025,
    [double] $EntropyCoef = 0.02,
    [double] $ScoreRewardCoef = 0.0,
    [int] $GapLandingConfirmSteps = 2,
    [string] $RunName = "ppo_multiinput_semantic_spatial_memory_binary_v6_forever",
    [string] $ModelPath = "models\ppo\junimo_ppo_multiinput_binary_v6_forever",
    [string] $LoadModel = "",
    [int] $EpisodeOffset = -1,
    [switch] $NoAutoResume
)

$resolvedLoadModel = $LoadModel
if (-not $NoAutoResume -and [string]::IsNullOrWhiteSpace($resolvedLoadModel)) {
    $checkpointCandidates = @(
        Get-ChildItem -Path "logs\ppo" -Recurse -Filter "junimo_ppo_ep*.zip" -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -like "*multiinput*binary*v6*" } |
            Sort-Object LastWriteTime -Descending
    )
    if ($checkpointCandidates.Count -gt 0) {
        $resolvedLoadModel = $checkpointCandidates[0].FullName
    }
}
if (-not $NoAutoResume -and [string]::IsNullOrWhiteSpace($resolvedLoadModel)) {
    $fallbackModel = "models\ppo\junimo_ppo_multiinput_binary_v6.zip"
    if (Test-Path $fallbackModel) {
        $resolvedLoadModel = $fallbackModel
    }
}

$resolvedEpisodeOffset = $EpisodeOffset
if ($resolvedEpisodeOffset -lt 0) {
    $resolvedEpisodeOffset = 0
    if (-not [string]::IsNullOrWhiteSpace($resolvedLoadModel)) {
        $fileName = [System.IO.Path]::GetFileName($resolvedLoadModel)
        if ($fileName -match "ep(\d+)") {
            $resolvedEpisodeOffset = [int]$Matches[1]
        }
    }
}

Write-Host "PPO MultiInput v6 continuous training"
Write-Host "Run name: $RunName"
Write-Host "Timesteps limit: $Timesteps"
Write-Host "Save every episodes: $SaveEpisodeFreq"
Write-Host "Episode offset: $resolvedEpisodeOffset"
if (-not [string]::IsNullOrWhiteSpace($resolvedLoadModel)) {
    Write-Host "Loading model: $resolvedLoadModel"
} else {
    Write-Host "No v6 checkpoint found. Starting a new model."
}

$trainArgs = @(
    ".\scripts\train_ppo.py",
    "--timesteps", "$Timesteps",
    "--episode-offset", "$resolvedEpisodeOffset",
    "--save-episode-freq", "$SaveEpisodeFreq",
    "--save-freq", "0",
    "--frame-skip", "$FrameSkip",
    "--observation-mode", "multi",
    "--recent-action-history", "$RecentActionHistory",
    "--semantic-features",
    "--temporal-features",
    "--reward-version", "shaped_v3",
    "--action-mode", "binary",
    "--score-reward-coef", "$ScoreRewardCoef",
    "--gap-landing-confirm-steps", "$GapLandingConfirmSteps",
    "--learning-rate", "$LearningRate",
    "--ent-coef", "$EntropyCoef",
    "--model-path", "$ModelPath",
    "--run-name", "$RunName"
)

if (-not [string]::IsNullOrWhiteSpace($resolvedLoadModel)) {
    $trainArgs += @("--load-model", "$resolvedLoadModel")
}

python @trainArgs
