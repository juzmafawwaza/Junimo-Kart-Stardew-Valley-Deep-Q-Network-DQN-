param(
    [int] $Episodes = 1000,
    [int] $SaveEpisodeFreq = 100,
    [int] $FrameSkip = 2,
    [int] $RecentActionHistory = 12,
    [double] $LearningRate = 0.0001,
    [double] $EntropyCoef = 0.01,
    [double] $ScoreRewardCoef = 0.0,
    [int] $GapLandingConfirmSteps = 2,
    [int] $LstmHiddenSize = 128,
    [int] $NLstmLayers = 1,
    [string] $RunName = "ppo_lstm_multiinput_semantic_spatial_memory_binary_v6_1k_save100",
    [string] $ModelPath = "models\ppo_lstm\junimo_ppo_lstm_multiinput_binary_v6"
)

python .\scripts\train_ppo_lstm.py `
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
    --lstm-hidden-size $LstmHiddenSize `
    --n-lstm-layers $NLstmLayers `
    --model-path $ModelPath `
    --run-name $RunName
