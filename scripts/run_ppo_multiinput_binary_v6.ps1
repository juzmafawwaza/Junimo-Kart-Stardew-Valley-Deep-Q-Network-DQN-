param(
    [int] $Episodes = 2000,
    [int] $SaveEpisodeFreq = 250,
    [int] $FrameSkip = 2,
    [int] $RecentActionHistory = 12,
    [double] $LearningRate = 0.00025,
    [double] $EntropyCoef = 0.02,
    [double] $ScoreRewardCoef = 0.0,
    [int] $GapLandingConfirmSteps = 2,
    [string] $RunName = "ppo_multiinput_semantic_spatial_memory_binary_v6_2k_save250",
    [string] $ModelPath = "models\ppo\junimo_ppo_multiinput_binary_v6"
)

python .\scripts\train_ppo.py `
    --episodes $Episodes `
    --save-episode-freq $SaveEpisodeFreq `
    --save-freq 0 `
    --frame-skip $FrameSkip `
    --observation-mode multi `
    --recent-action-history $RecentActionHistory `
    --semantic-features `
    --temporal-features `
    --reward-version shaped_v3 `
    --action-mode binary `
    --score-reward-coef $ScoreRewardCoef `
    --gap-landing-confirm-steps $GapLandingConfirmSteps `
    --learning-rate $LearningRate `
    --ent-coef $EntropyCoef `
    --model-path $ModelPath `
    --run-name $RunName
