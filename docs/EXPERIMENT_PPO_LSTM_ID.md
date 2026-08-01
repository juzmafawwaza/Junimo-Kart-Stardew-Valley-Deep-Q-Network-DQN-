# Catatan Eksperimen: PPO-LSTM Macro 6 + `shaped_v3`

Dokumen ini menjelaskan script PPO-LSTM yang dibuat untuk membandingkan PPO biasa vs PPO dengan memory pada Junimo Kart.

Eksperimen ini dibuat sejajar dengan PPO v3:

```text
semantic features: aktif
reward_version: shaped_v3
action_mode: macro
macro_action_frames: 6
score_reward_coef: 0.0
gap_landing_confirm_steps: 2
entropy coefficient: 0.003
```

Perbedaannya: policy PPO biasa diganti menjadi policy recurrent/LSTM.

## Kenapa PPO-LSTM?

PPO biasa melihat state saat ini saja.

```text
state_t -> policy -> action_t
```

PPO-LSTM melihat state saat ini sambil membawa memory dari step sebelumnya.

```text
state_t + lstm_memory_t -> policy -> action_t + lstm_memory_(t+1)
```

Untuk Junimo Kart, memory ini penting karena action yang benar sering bergantung pada trajectory:

- beberapa frame lalu agent masih grounded atau sudah lompat;
- durasi hold jump sebelumnya short/medium/long;
- cart sedang naik atau turun;
- landing target lebih tinggi/rendah;
- gap yang sedang diseberangi mulai dari mana dan selebar apa.

Dengan PPO biasa, sebagian informasi temporal memang bisa diwakili velocity dan `jumpHeld`, tetapi LSTM memberi model ruang untuk menyimpan konteks beberapa step sebelumnya.

## Dependency baru

PPO-LSTM memakai `RecurrentPPO` dari `sb3-contrib`.

Dependency ini sudah ditambahkan ke:

```text
pyproject.toml
```

Kalau command error karena `sb3_contrib` belum ada, jalankan:

```powershell
pip install -e ".[train,analysis]"
```

Atau minimal:

```powershell
pip install sb3-contrib
```

## File yang ditambahkan

### `scripts/train_ppo_lstm.py`

Script utama training PPO-LSTM.

Bagian penting:

```python
from sb3_contrib import RecurrentPPO
```

Model dibuat dengan:

```python
RecurrentPPO(
    "MlpLstmPolicy",
    env,
    learning_rate=args.learning_rate,
    n_steps=args.n_steps,
    batch_size=args.batch_size,
    n_epochs=args.n_epochs,
    gamma=args.gamma,
    gae_lambda=args.gae_lambda,
    clip_range=args.clip_range,
    ent_coef=args.ent_coef,
    vf_coef=args.vf_coef,
    max_grad_norm=args.max_grad_norm,
    policy_kwargs=policy_kwargs,
    tensorboard_log=str(tensorboard_dir),
    verbose=1,
)
```

`MlpLstmPolicy` berarti input tetap vector numerik, bukan pixel, tetapi network-nya punya LSTM memory.

Policy kwargs default:

```python
policy_kwargs = {
    "lstm_hidden_size": 128,
    "n_lstm_layers": 1,
}
```

Default learning rate dibuat lebih kecil dari PPO v3 biasa:

```text
PPO v3:      0.0003
PPO-LSTM v3: 0.0001
```

Alasannya: recurrent policy biasanya lebih sensitif dan lebih mudah unstable. Environment dan reward tetap sama, tetapi optimizer dibuat lebih konservatif.

### `scripts/evaluate_ppo_lstm_models.py`

Evaluator khusus PPO-LSTM.

Evaluator PPO biasa tidak cukup karena LSTM perlu membawa hidden state antar-step.

Bagian penting:

```python
lstm_states = None
episode_starts = np.ones((1,), dtype=bool)

action, lstm_states = model.predict(
    obs,
    state=lstm_states,
    episode_start=episode_starts,
    deterministic=True,
)
```

Setelah episode selesai, memory harus reset. Karena itu `episode_start` dikirim ke model.

### `scripts/run_ppo_lstm_macro6_v3.ps1`

Launcher PowerShell agar command lebih pendek.

Default-nya:

```powershell
.\scripts\run_ppo_lstm_macro6_v3.ps1
```

Setara dengan:

```powershell
python .\scripts\train_ppo_lstm.py --episodes 5000 --save-episode-freq 1000 --save-freq 0 --frame-skip 2 --semantic-features --reward-version shaped_v3 --action-mode macro --macro-action-frames 6 --score-reward-coef 0.0 --gap-landing-confirm-steps 2 --learning-rate 0.0001 --ent-coef 0.003 --lstm-hidden-size 128 --n-lstm-layers 1 --model-path models\ppo_lstm\junimo_ppo_lstm_macro6_v3 --run-name ppo_lstm_semantic_shaped_v3_macro6_5k
```

## Output folder

Training PPO-LSTM disimpan terpisah:

```text
logs/ppo_lstm/<run-name>/
models/ppo_lstm/
```

Default:

```text
logs/ppo_lstm/ppo_lstm_semantic_shaped_v3_macro6_5k/
models/ppo_lstm/junimo_ppo_lstm_macro6_v3.zip
```

Checkpoints:

```text
logs/ppo_lstm/ppo_lstm_semantic_shaped_v3_macro6_5k/checkpoints/
```

Checkpoint name:

```text
junimo_ppo_lstm_ep001000_stepsXXXXX.zip
junimo_ppo_lstm_ep002000_stepsXXXXX.zip
...
```

## Evaluation command

Setelah checkpoint tersedia:

```powershell
python .\scripts\evaluate_ppo_lstm_models.py ".\logs\ppo_lstm\ppo_lstm_semantic_shaped_v3_macro6_5k\checkpoints\junimo_ppo_lstm_ep*.zip" --episodes 20 --frame-skip 2 --semantic-features --reward-version shaped_v3 --action-mode macro --macro-action-frames 6 --score-reward-coef 0.0 --gap-landing-confirm-steps 2 --out logs\ppo_lstm\ppo_lstm_semantic_shaped_v3_macro6_5k\evaluation.csv
```

## Metrics yang dibandingkan dengan PPO v3 biasa

Untuk fair comparison, jangan hanya pakai reward mean. Bandingkan:

- `mean_length`
- `mean_max_episode_x`
- `gap_landing_rate`
- `death_near_gap_rate`
- `death_near_obstacle_rate`
- `completion_rate`
- action ratios: `action_0_ratio`, `action_1_ratio`, `action_2_ratio`, `action_3_ratio`

Hipotesis:

```text
Kalau Junimo Kart memang butuh temporal memory,
PPO-LSTM harus lebih baik di gap_landing_rate dan mean_max_episode_x
dibanding PPO biasa dengan reward/action yang sama.
```

Kalau PPO-LSTM tidak membaik, kemungkinan masalah utamanya bukan memory, melainkan:

- reward masih kurang tepat;
- semantic gap detection kurang akurat;
- action abstraction masih belum pas;
- training terlalu lambat/noisy karena game real-time;
- model butuh curriculum atau deterministic baseline sebagai bantuan.
