param(
    [int] $Episodes = 5000,
    [int] $SaveEpisodeFreq = 1000,
    [int] $FrameSkip = 2,
    [int] $MacroActionFrames = 6,
    [double] $LearningRate = 0.0001,
    [double] $EntropyCoef = 0.003,
    [double] $ScoreRewardCoef = 0.0,
    [int] $GapLandingConfirmSteps = 2,
    [int] $LstmHiddenSize = 128,
    [int] $NLstmLayers = 1,
    [string] $RunName = "ppo_lstm_semantic_shaped_v3_macro6_5k",
    [string] $ModelPath = "models\ppo_lstm\junimo_ppo_lstm_macro6_v3"
)

python .\scripts\train_ppo_lstm.py `
    --episodes $Episodes `
    --save-episode-freq $SaveEpisodeFreq `
    --save-freq 0 `
    --frame-skip $FrameSkip `
    --semantic-features `
    --reward-version shaped_v3 `
    --action-mode macro `
    --macro-action-frames $MacroActionFrames `
    --score-reward-coef $ScoreRewardCoef `
    --gap-landing-confirm-steps $GapLandingConfirmSteps `
    --learning-rate $LearningRate `
    --ent-coef $EntropyCoef `
    --lstm-hidden-size $LstmHiddenSize `
    --n-lstm-layers $NLstmLayers `
    --model-path $ModelPath `
    --run-name $RunName
