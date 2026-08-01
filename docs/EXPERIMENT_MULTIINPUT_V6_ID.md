# Catatan Eksperimen: MultiInput Semantic + Spatial + Memory V6

Dokumen ini menjelaskan eksperimen baru yang terinspirasi dari pendekatan PWhiddy/PokemonRedExperiments, tetapi disesuaikan dengan keterbatasan Stardew Valley live game.

## Kenapa v6 dibuat?

Eksperimen sebelumnya menunjukkan masalah:

```text
PPO macro v4b deterministic policy collapse.
Checkpoint awal cenderung selalu action 3.
Checkpoint akhir cenderung selalu action 0.
```

Artinya model belum belajar memilih action berbeda sesuai kondisi. Menambah episode saja belum tentu menyelesaikan masalah.

PWhiddy/PokemonRedExperiments berbeda karena memakai emulator PyBoy:

- bisa headless;
- bisa tick game secepat CPU;
- bisa menjalankan banyak environment paralel;
- observation-nya multi-modal: screen, coordinate/map, memory/recent actions.

Project Junimo Kart kita belum punya emulator/headless Stardew, jadi speed 1000x belum realistis. Tetapi bagian multi-modal observation bisa kita adaptasi.

## Apa yang ditambah di v6?

Mode baru:

```powershell
--observation-mode multi
```

Kalau mode ini aktif, observation bukan satu vector besar lagi. Observation menjadi dictionary:

```text
state
semantic
temporal
recent_actions
spatial
```

## Isi masing-masing input

### `state`

Raw internal-state vector lama:

```text
base game state
player position/velocity/status
tracks ahead
entities ahead
```

Ini shape lama:

```text
306
```

### `semantic`

Feature hasil engineering:

```text
next_track_dx
next_track_y
next_gap_present
next_gap_start_dx
next_gap_width
landing_y
landing_delta_y
next_obstacle_dx
next_pickup_dx
distance_to_finish
progress_fraction
```

Ini memberi model sinyal yang lebih “manusiawi” tentang gap dan landing.

### `temporal`

Sensor timing:

```text
jump_held_steps
airborne_steps
grounded_steps
last_action
last_action_holds_jump
```

Ini membantu model memahami durasi hold/release.

### `recent_actions`

Memory action beberapa decision terakhir.

Default:

```text
--recent-action-history 12
```

Untuk binary action, shape-nya:

```text
12 * 2 = 24
```

Ini memberi policy konteks seperti:

```text
apakah tadi baru hold jump?
apakah sudah lama release?
apakah action belakangan monoton?
```

### `spatial`

Ini bagian “visual-like”.

Penting: ini bukan screenshot layar Stardew.

`spatial` adalah grid kecil yang dirender dari koordinat internal game:

```text
channel 0 = track/rel
channel 1 = obstacle
channel 2 = pickup/coin/fruit
channel 3 = player/cart
```

Shape:

```text
4 x 16 x 64
```

Kenapa tidak langsung screenshot?

Karena screenshot live Stardew:

- lebih lambat;
- tergantung posisi window/focus;
- raw pixel lebih sulit dianalisis;
- belum tentu stabil saat game unfocused.

Semantic spatial map lebih cocok untuk tahap sekarang karena tetap memberi bentuk “visual” gap/track/obstacle tetapi dari data internal yang bersih.

## Kenapa action balik ke binary?

Ini bukan balik ke v1.

v1:

```text
binary action
flat vector
belum punya spatial map
belum punya recent action memory
belum punya temporal features
```

v6:

```text
binary action
multi-input observation
semantic features
temporal features
recent action memory
spatial map
track bounds support
```

Untuk Junimo Kart, action asli game memang:

```text
release jump
hold jump
```

Short jump dan long jump muncul dari berapa lama model memilih hold berturut-turut. Karena sekarang model melihat `recent_actions` dan `jump_held_steps`, binary action menjadi lebih masuk akal daripada macro action yang sempat collapse.

## File yang diubah

### `junimo_rl/env.py`

Perubahan:

- menambah `OBSERVATION_MODES = {"flat", "multi"}`;
- menambah `multi_observation_space()`;
- menambah `spatial_feature_map()`;
- menambah recent action memory;
- `JunimoKartEnv` menerima:

```python
observation_mode: str = "flat"
recent_action_history: int = 12
```

### `scripts/train_ppo.py`

Flag baru:

```powershell
--observation-mode multi
--recent-action-history 12
```

Kalau observation mode `multi`, script otomatis memakai:

```text
MultiInputPolicy
```

### `scripts/train_ppo_lstm.py`

Flag baru sama seperti PPO.

Kalau observation mode `multi`, script otomatis memakai:

```text
MultiInputLstmPolicy
```

### Evaluation dan trace scripts

File:

```text
scripts/evaluate_ppo_models.py
scripts/evaluate_ppo_lstm_models.py
scripts/trace_policy_rollout.py
```

sekarang menerima:

```powershell
--observation-mode multi
--recent-action-history 12
```

## Command training yang direkomendasikan

Mulai dari PPO MultiInput v6:

```powershell
python .\scripts\train_ppo.py --episodes 2000 --save-episode-freq 250 --save-freq 0 --frame-skip 2 --observation-mode multi --recent-action-history 12 --semantic-features --temporal-features --reward-version shaped_v3 --action-mode binary --score-reward-coef 0.0 --gap-landing-confirm-steps 2 --learning-rate 0.00025 --ent-coef 0.02 --model-path models\ppo\junimo_ppo_multiinput_binary_v6 --run-name ppo_multiinput_semantic_spatial_memory_binary_v6_2k_save250
```

Atau:

```powershell
.\scripts\run_ppo_multiinput_binary_v6.ps1
```

PPO-LSTM MultiInput v6:

```powershell
python .\scripts\train_ppo_lstm.py --episodes 1000 --save-episode-freq 100 --save-freq 0 --frame-skip 2 --observation-mode multi --recent-action-history 12 --semantic-features --temporal-features --reward-version shaped_v3 --action-mode binary --score-reward-coef 0.0 --gap-landing-confirm-steps 2 --learning-rate 0.0001 --ent-coef 0.01 --lstm-hidden-size 128 --n-lstm-layers 1 --model-path models\ppo_lstm\junimo_ppo_lstm_multiinput_binary_v6 --run-name ppo_lstm_multiinput_semantic_spatial_memory_binary_v6_1k_save100
```

Atau:

```powershell
.\scripts\run_ppo_lstm_multiinput_binary_v6.ps1
```

## Command evaluasi

PPO MultiInput v6:

```powershell
python .\scripts\evaluate_ppo_models.py ".\logs\ppo\ppo_multiinput_semantic_spatial_memory_binary_v6_2k_save250\checkpoints\junimo_ppo_ep*.zip" --episodes 5 --frame-skip 2 --observation-mode multi --recent-action-history 12 --semantic-features --temporal-features --reward-version shaped_v3 --action-mode binary --score-reward-coef 0.0 --gap-landing-confirm-steps 2 --max-steps-per-episode 300 --out logs\ppo\ppo_multiinput_semantic_spatial_memory_binary_v6_2k_save250\evaluation_5ep.csv
```

PPO-LSTM MultiInput v6:

```powershell
python .\scripts\evaluate_ppo_lstm_models.py ".\logs\ppo_lstm\ppo_lstm_multiinput_semantic_spatial_memory_binary_v6_1k_save100\checkpoints\junimo_ppo_lstm_ep*.zip" --episodes 5 --frame-skip 2 --observation-mode multi --recent-action-history 12 --semantic-features --temporal-features --reward-version shaped_v3 --action-mode binary --score-reward-coef 0.0 --gap-landing-confirm-steps 2 --max-steps-per-episode 300 --out logs\ppo_lstm\ppo_lstm_multiinput_semantic_spatial_memory_binary_v6_1k_save100\evaluation_5ep.csv
```

## Continuous training / jalan terus

Launcher baru:

```powershell
.\scripts\run_ppo_multiinput_binary_v6_forever.ps1
.\scripts\run_ppo_lstm_multiinput_binary_v6_forever.ps1
```

Bedanya dengan launcher biasa:

```text
launcher biasa   = berhenti setelah target episode tertentu
launcher forever = tidak memakai target episode, berhenti hanya kalau dihentikan user / game bridge disconnect / timesteps sangat besar tercapai
```

Secara teknis launcher forever mengirim:

```powershell
--timesteps 2147483647
--save-episode-freq 1000
```

Angka `2147483647` dipakai sebagai batas aman yang sangat jauh. Dengan kecepatan Stardew live, batas ini praktis tidak akan tercapai dalam eksperimen normal.

Launcher forever juga mencoba auto-resume dari checkpoint v6 terakhir. Contoh:

```text
junimo_ppo_lstm_ep010000_steps300000.zip
```

akan membuat checkpoint berikutnya bernama:

```text
junimo_ppo_lstm_ep011000_...
junimo_ppo_lstm_ep012000_...
```

Kalau belum ada checkpoint v6, launcher akan start model baru dari episode 0. Jangan pakai `--episode-offset 10000` hanya untuk mengganti nama checkpoint kalau model belum benar-benar pernah training 10000 episode, karena itu akan membuat analisis eksperimen misleading.

Kalau training dihentikan dengan `Ctrl+C`, script training sekarang tetap menyimpan model terakhir ke `--model-path`. Namun untuk analisis yang rapi, tetap gunakan checkpoint episode 1000/2000/3000/dst karena checkpoint itulah yang comparable.

## Live state/telemetry table dari monitor.csv

Untuk melihat apakah training membaik tanpa membuka Excel, gunakan:

```powershell
python .\scripts\watch_monitor_table.py --latest --watch --every-episodes 100 --history 10
```

Script ini membaca `monitor.csv`, bukan bridge. Jadi boleh dijalankan sambil training berjalan.

Setiap 100 episode, script menampilkan tabel ringkas:

```text
iter, episodes, rew_mean, len_mean, best_len, max_x_mean,
grounded_pct, jump_held_pct, gap_visible_pct, gap_near_pct,
gap_dx_mean, gap_w_mean, landing_dy_mean, obs_near_pct,
pickup_visible_pct, gap_att_ep, gap_land_rate, death_gap_rate,
pickup_ep, action ratio, final_gap_dx, final_gap_w, final_obs_dx
```

Kalau ingin hanya menampilkan window yang benar-benar penuh:

```powershell
python .\scripts\watch_monitor_table.py --latest --watch --every-episodes 100 --history 10 --no-partial
```

## Per-step state trace

Kalau ingin melihat state per pergerakan seperti:

```text
x=628.7 | grounded=False | gap=True | gap_dx=0.0 | gap_width=96.0 | landing_y=176.0
```

gunakan trace debug:

```powershell
.\scripts\run_ppo_lstm_multiinput_binary_v6_trace_debug.ps1
```

Script ini menjalankan PPO-LSTM v6 dengan format simple:

```text
x: 2, y: 1, reward: 4.235, score: 170, generation: 28
```

Untuk debug, launcher menambahkan action:

```text
x: 2, y: 1, action: 1, hold: 1, reward: 4.235, score: 170, generation: 28
```

Makna `x` dan `y` di format simple:

```text
x = bin jarak horizontal ke target terdekat, prioritas gap -> obstacle -> pickup
y = bin beda tinggi target, terutama landing_delta_y untuk gap
reward = reward kumulatif episode sampai step itu
```

Di CSV trace:

```text
step_reward = reward pada step itu saja
reward      = reward kumulatif episode
```

Flag utama:

```powershell
--trace-state-print-freq 1
--trace-state-format simple
--trace-state-simple-action
--trace-state-csv logs\ppo_lstm\ppo_lstm_multiinput_binary_v6_trace_debug\state_trace.csv
```

Jika terminal terlalu ramai, pakai:

```powershell
.\scripts\run_ppo_lstm_multiinput_binary_v6_trace_debug.ps1 -TracePrintFreq 10
```

CSV trace bisa dibuka di Excel untuk melihat state setiap step.

Launcher debug juga memakai:

```powershell
--n-steps 256
--n-epochs 3
```

agar jeda saat PPO-LSTM update model lebih pendek daripada setting normal.

## V7 positive reward / `shaped_v4`

Setelah trace per-step menunjukkan reward sering tampak turun karena penalti kecil action timing, dibuat versi reward baru:

```text
--reward-version shaped_v4
```

Perubahan:

```text
- tidak ada penalti kecil untuk lompat di rel yang masih nyambung;
- tidak ada penalti kecil untuk tidak lompat saat gap dekat;
- death penalty tetap ada;
- gap landing reward dibuat lebih besar;
- bonus life positif saat reset dihapus;
- trace menampilkan event reward.
```

Command debug:

```powershell
.\scripts\run_ppo_lstm_multiinput_binary_v7_positive_trace_debug.ps1
```

Command training:

```powershell
.\scripts\run_ppo_lstm_multiinput_binary_v7_positive.ps1
```

Contoh output yang dicari:

```text
x: 2, y: -2, action: 1, hold: 1, reward: 11.84, event: gap_landing, score: 180, generation: 12
```

Kalau event `gap_landing` muncul, berarti reward melewati jurang sudah benar-benar terbayar.

### Gap landing reward parameter

Reward berhasil mendarat setelah jurang sekarang bisa diatur:

```powershell
.\scripts\run_ppo_lstm_multiinput_binary_v7_positive.ps1 -GapLandingBaseReward 12 -GapLandingWidthCoef 0.05
```

Rumus:

```text
gap_landing_reward = GapLandingBaseReward + GapLandingWidthCoef * min(gap_width, 120)
```

Default v7:

```text
GapLandingBaseReward = 8.0
GapLandingWidthCoef  = 0.04
```

Contoh:

```text
gap_width 96 -> 8.0 + 0.04 * 96 = 11.84
```

Reward ini tidak keluar untuk lompat biasa di rel yang masih nyambung. Reward hanya keluar kalau agent pernah masuk area gap lalu berhasil grounded lagi di landing track.

### Coin vs fruit reward

V7 juga memisahkan reward coin dan fruit:

```powershell
.\scripts\run_ppo_lstm_multiinput_binary_v7_positive.ps1 -CoinRewardCoef 0.0005 -FruitRewardCoef 0.003
```

Default launcher v7:

```text
coin  = score_delta * 0.0005
fruit = score_delta * 0.003
```

Fruit berarti 6x lebih penting daripada coin. Trace akan menampilkan:

```text
event: coin
event: fruit
```

Catatan: split reward ini mencoba mendeteksi pickup dekat cart dari entity type/bounds, bukan sekadar setiap score naik. Ini supaya score/progress biasa tidak otomatis dianggap coin.

### Full parameter launcher

Launcher lengkap untuk eksperimen PPO-LSTM v7:

```powershell
.\scripts\run_ppo_lstm_v7_full_params.ps1
```

Default-nya sama dengan setup v7 positive terbaru:

```text
observation_mode   = multi
reward_version     = shaped_v4
action_mode        = binary
fruit_reward_coef  = 0.005
gap_landing_reward = 12.0 + 0.05 * min(gap_width, 120)
```

Untuk melihat trace kumulatif reward di terminal:

```powershell
.\scripts\run_ppo_lstm_v7_full_params.ps1 -TracePrintFreq 1
```

Untuk cek command tanpa mulai training:

```powershell
.\scripts\run_ppo_lstm_v7_full_params.ps1 -DryRun
```

## Metric yang perlu dilihat

Jangan cuma lihat reward mean.

Lihat:

```text
action_0_ratio
jump_hold_ratio
mean_length
mean_max_episode_x
gap_landing_rate
death_near_gap_rate
capped_episode_rate
```

Kalau model collapse lagi, biasanya terlihat:

```text
action_0_ratio = 1.0
```

atau:

```text
jump_hold_ratio = 1.0
```

Kalau v6 lebih sehat, action ratio tidak ekstrem dan `mean_length`/`mean_max_episode_x` mulai naik.
