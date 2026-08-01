param(
    [int] $Episodes = 5000,
    [int] $SaveEpisodeFreq = 1000,
    [int] $FrameSkip = 2,
    [int] $MacroActionFrames = 6,
    [int] $MacroReleaseFrames = 1,
    [double] $LearningRate = 0.0003,
    [double] $EntropyCoef = 0.003,
    [double] $ScoreRewardCoef = 0.0,
    [int] $GapLandingConfirmSteps = 2,
    [string] $RunName = "ppo_semantic_temporal_shaped_v3_tap_macro_5k",
    [string] $ModelPath = "models\ppo\junimo_ppo_tap_macro_v4"
)

python .\scripts\train_ppo.py `
    --episodes $Episodes `
    --save-episode-freq $SaveEpisodeFreq `
    --save-freq 0 `
    --frame-skip $FrameSkip `
    --semantic-features `
    --temporal-features `
    --reward-version shaped_v3 `
    --action-mode tap_macro `
    --macro-action-frames $MacroActionFrames `
    --macro-release-frames $MacroReleaseFrames `
    --score-reward-coef $ScoreRewardCoef `
    --gap-landing-confirm-steps $GapLandingConfirmSteps `
    --learning-rate $LearningRate `
    --ent-coef $EntropyCoef `
    --model-path $ModelPath `
    --run-name $RunName
