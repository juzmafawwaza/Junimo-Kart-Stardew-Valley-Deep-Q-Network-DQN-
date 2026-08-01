# Catatan Perubahan: PPO/PPO-LSTM V4 `tap_macro` + Temporal Features

Dokumen ini menjelaskan improvement setelah hasil `shaped_v3 + macro` dan PPO-LSTM awal masih belum jauh berbeda dari eksperimen sebelumnya.

Kesimpulan sementara dari log dan video:

1. Semantic coordinate dari bridge terlihat cukup masuk akal.
   - Saat kamu main manual, nilai `gap_dx`, `gap_width`, `landing_y`, dan `landing_delta_y` terlihat mengikuti gap di layar.
   - Jadi issue terbesar saat ini kemungkinan bukan “koordinat kebalik total”.

2. Agent masih sering mati dekat gap.
   - Telemetry v3 menunjukkan `death_near_gap` masih sangat tinggi.
   - `gap_landing_rate` masih rendah.

3. Action `macro` lama terlalu gampang menjadi “hold terus”.
   - Pada v3, action long hold sering dominan.
   - Kalau model memilih action long berkali-kali, tombol jump bisa terasa seperti terus ditahan.
   - Untuk Junimo Kart, ini berbahaya karena kontrol utama bukan hanya “lompat atau tidak”, tapi “kapan tekan dan kapan lepas”.

4. PPO-LSTM belum otomatis memperbaiki masalah ini.
   - LSTM memberi memory internal, tapi kalau action space-nya masih membuat hold/release ambigu, model tetap bisa kesulitan.
   - Karena itu v4 memperbaiki action design dan observation dulu.

## Perubahan utama v4

### 1. Action mode baru: `tap_macro`

File:

```text
junimo_rl/env.py
```

Mode lama:

```text
binary:
  0 = release
  1 = hold

macro:
  0 = release
  1 = short hold
  2 = medium hold
  3 = long hold
```

Mode baru:

```text
tap_macro:
  0 = release selama seluruh macro window
  1 = short hold, lalu release
  2 = medium hold, lalu release
  3 = long hold, lalu release tail kecil
```

Contoh jika:

```text
macro_action_frames = 6
macro_release_frames = 1
```

Maka action kira-kira menjadi:

```text
0 = release 6 frame
1 = hold sekitar 2 frame, release 4 frame
2 = hold sekitar 3 frame, release 3 frame
3 = hold 5 frame, release 1 frame
```

Kenapa ini penting?

Karena agent tidak lagi bisa “nyangkut” menahan jump tanpa jeda. Setiap jump macro dipaksa punya release di akhir, sehingga action lebih mirip input manusia: tekan pendek/sedang/panjang, lalu lepas.

Ini masih bukan rule-based bot. Kita tidak memberi tahu “kalau gap sekian harus action 2”. Kita hanya memperbaiki bentuk tombol agar action yang dipelajari lebih masuk akal secara mekanik game.

### 2. Temporal features

File:

```text
junimo_rl/env.py
```

Flag baru:

```powershell
--temporal-features
```

Kalau flag ini aktif, observation ditambah 5 angka:

```text
jump_held_steps
airborne_steps
grounded_steps
last_action
last_action_holds_jump
```

Semua dinormalisasi supaya masuk ke neural network dengan skala kecil.

Kenapa ini penting?

Junimo Kart sangat bergantung pada timing. Dua state yang secara posisi mirip bisa butuh action berbeda kalau:

- tombol jump sudah ditahan lama;
- cart baru mulai naik;
- cart sudah lama airborne;
- cart baru mendarat;
- action sebelumnya short/medium/long.

PPO biasa tidak punya memory internal seperti LSTM. Dengan temporal features, PPO diberi “jam kecil” agar bisa belajar timing tanpa harus menebak dari satu snapshot saja.

### 3. Script trace rollout

File:

```text
scripts/trace_policy_rollout.py
```

Script ini menjalankan model terlatih dan menulis CSV per-step.

Kolom penting:

```text
episode
step
action
reward
episode_reward
player_x
player_y
velocity_x
velocity_y
grounded
jumping
jump_held
next_gap_present
next_gap_start_dx
next_gap_width
landing_y
landing_delta_y
next_obstacle_dx
next_pickup_dx
progress_fraction
```

Tujuannya:

- melihat apakah model benar-benar melompat saat gap dekat;
- melihat apakah ia terlalu lama hold;
- melihat apakah ia release terlalu cepat;
- memvalidasi apakah semantic features sesuai dengan kondisi game;
- membuat bahan analisis untuk chart/paper.

### 4. Track bounds dari SMAPI bridge

File:

```text
src/JunimoKartRLBridge/Protocol.cs
src/JunimoKartRLBridge/ModEntry.cs
junimo_rl/env.py
```

Sebelumnya gap dihitung dari jarak antar posisi track. Itu cukup membantu, tetapi posisi track belum tentu sama dengan edge/collision area track.

Sekarang bridge mengirim:

```text
track.bounds
```

Kalau bounds tersedia, Python menghitung:

```text
gap_start_dx = right_edge_track_kiri - player_x
gap_width    = left_edge_track_kanan - right_edge_track_kiri
landing_y    = y dari track kanan / landing track
```

Kalau bounds tidak tersedia, Python tetap fallback ke cara lama berbasis `track.dx`. Jadi perubahan ini backward-compatible.

## File yang diubah

### `junimo_rl/env.py`

Perubahan:

- menambah `TEMPORAL_FEATURES`;
- menambah `SEMANTIC_TEMPORAL_OBSERVATION_SIZE`;
- `observation_size()` sekarang bisa menghitung observation size untuk kombinasi legacy/semantic/temporal;
- `snapshot_to_vector()` sekarang bisa append temporal values;
- gap detection memakai `track.bounds` jika tersedia, lalu fallback ke `track.dx`;
- `JunimoKartEnv` menerima:

```python
use_temporal_features: bool = False
macro_release_frames: int = 1
```

- action mode baru `tap_macro`;
- temporal counters internal:

```python
jump_held_steps
airborne_steps
grounded_steps
last_action
last_action_holds_jump
```

### `scripts/train_ppo.py`

Flag baru:

```powershell
--temporal-features
--action-mode tap_macro
--macro-release-frames
```

### `scripts/train_ppo_lstm.py`

Flag baru yang sama seperti PPO biasa.

### `scripts/train_dqn.py`

DQN juga diberi flag yang sama agar comparative experiment tetap bisa dibuat fair kalau nanti ingin DQN versi v4.

### Evaluation scripts

File:

```text
scripts/evaluate_models.py
scripts/evaluate_ppo_models.py
scripts/evaluate_ppo_lstm_models.py
```

Semua sekarang menerima:

```powershell
--temporal-features
--action-mode tap_macro
--macro-release-frames
```

Ingat: evaluation harus memakai flag observation/action yang sama dengan training.

### Launcher scripts

File:

```text
scripts/run_ppo_tap_macro_v4.ps1
scripts/run_ppo_lstm_tap_macro_v4.ps1
```

## Koreksi penting: `tap_macro` bisa memotong long jump

Setelah dicoba di game, ada detail mekanik Junimo Kart yang penting:

```text
Kalau tombol jump dilepas di tengah udara, press lagi saat masih di udara tidak otomatis menyambung hold jump yang sama.
```

Artinya, `tap_macro` bisa membuat action 3 terasa seperti long jump yang kepotong:

```text
hold beberapa frame -> release di udara -> press lagi, tetapi input berikutnya baru efektif saat sudah grounded
```

Ini menjelaskan kenapa perilakunya bisa terlihat seperti spam small jump.

Karena itu rekomendasi yang lebih aman adalah `macro + temporal-features`, bukan `tap_macro`.

Dengan `macro`:

```text
action 0 = release
action 1 = short hold lalu release
action 2 = medium hold lalu release
action 3 = hold full macro window
```

Kalau model memilih action 3 beberapa kali berturut-turut, jump bisa tetap ditahan kontinu. Bedanya dari v3 lama: sekarang model juga punya temporal features seperti `jump_held_steps` dan `airborne_steps`, jadi ia punya informasi untuk belajar kapan harus release.

`tap_macro` tetap ada sebagai mode eksperimen, tetapi jangan jadikan baseline utama dulu untuk Junimo Kart.

## Command training yang direkomendasikan

Mulai dari PPO v4b dulu:

```powershell
python .\scripts\train_ppo.py --episodes 5000 --save-episode-freq 1000 --save-freq 0 --frame-skip 2 --semantic-features --temporal-features --reward-version shaped_v3 --action-mode macro --macro-action-frames 6 --score-reward-coef 0.0 --gap-landing-confirm-steps 2 --learning-rate 0.0003 --ent-coef 0.003 --model-path models\ppo\junimo_ppo_macro6_temporal_v4b --run-name ppo_semantic_temporal_shaped_v3_macro6_5k
```

Atau lewat PowerShell launcher:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\scripts\run_ppo_macro_temporal_v4b.ps1
```

Command `tap_macro` lama, hanya untuk eksperimen tambahan:

```powershell
python .\scripts\train_ppo.py --episodes 5000 --save-episode-freq 1000 --save-freq 0 --frame-skip 2 --semantic-features --temporal-features --reward-version shaped_v3 --action-mode tap_macro --macro-action-frames 6 --macro-release-frames 1 --score-reward-coef 0.0 --gap-landing-confirm-steps 2 --learning-rate 0.0003 --ent-coef 0.003 --model-path models\ppo\junimo_ppo_tap_macro_v4 --run-name ppo_semantic_temporal_shaped_v3_tap_macro_5k
```

Kalau mau PPO-LSTM v4b, gunakan checkpoint lebih sering dulu. Ini sengaja 1.000 episode dan save tiap 100 episode supaya kalau training ke-close sebelum selesai, tetap ada model `.zip` untuk dievaluate:

```powershell
python .\scripts\train_ppo_lstm.py --episodes 1000 --save-episode-freq 100 --save-freq 0 --frame-skip 2 --semantic-features --temporal-features --reward-version shaped_v3 --action-mode macro --macro-action-frames 6 --score-reward-coef 0.0 --gap-landing-confirm-steps 2 --learning-rate 0.0001 --ent-coef 0.003 --lstm-hidden-size 128 --n-lstm-layers 1 --model-path models\ppo_lstm\junimo_ppo_lstm_macro6_temporal_v4b --run-name ppo_lstm_semantic_temporal_shaped_v3_macro6_1k_save100
```

Atau:

```powershell
.\scripts\run_ppo_lstm_macro_temporal_v4b.ps1
```

Command PPO-LSTM `tap_macro` lama, hanya untuk eksperimen tambahan:

```powershell
python .\scripts\train_ppo_lstm.py --episodes 5000 --save-episode-freq 1000 --save-freq 0 --frame-skip 2 --semantic-features --temporal-features --reward-version shaped_v3 --action-mode tap_macro --macro-action-frames 6 --macro-release-frames 1 --score-reward-coef 0.0 --gap-landing-confirm-steps 2 --learning-rate 0.0001 --ent-coef 0.003 --lstm-hidden-size 128 --n-lstm-layers 1 --model-path models\ppo_lstm\junimo_ppo_lstm_tap_macro_v4 --run-name ppo_lstm_semantic_temporal_shaped_v3_tap_macro_5k
```

## Command evaluasi

Evaluation di project ini bukan test set offline. Model benar-benar dimainkan lagi di Junimo Kart live, lalu metric dihitung dari episode baru.

Evaluator sekarang punya safety cap:

```powershell
--max-steps-per-episode 300
```

Default-nya 300 supaya kalau game/state nyangkut dan episode tidak mengirim `gameOver/completed`, evaluator tetap lanjut. Kalau episode kena cap, kolom `capped_episode_rate` di CSV akan naik. Untuk debugging, nilai ini harusnya dekat 0.

Kalau wildcard checkpoint tidak menemukan file apa pun, evaluator sekarang akan berhenti dengan pesan seperti:

```text
No PPO-LSTM checkpoint matched pattern: ...
```

Ini biasanya berarti:

- nama folder run salah;
- training belum pernah mencapai episode checkpoint;
- training ke-close sebelum final model tersimpan;
- folder `checkpoints` kosong.

PPO v4:

```powershell
python .\scripts\evaluate_ppo_models.py ".\logs\ppo\ppo_semantic_temporal_shaped_v3_macro6_5k\checkpoints\junimo_ppo_ep*.zip" --episodes 20 --frame-skip 2 --semantic-features --temporal-features --reward-version shaped_v3 --action-mode macro --macro-action-frames 6 --score-reward-coef 0.0 --gap-landing-confirm-steps 2 --max-steps-per-episode 300 --out logs\ppo\ppo_semantic_temporal_shaped_v3_macro6_5k\evaluation.csv
```

PPO-LSTM v4:

```powershell
python .\scripts\evaluate_ppo_lstm_models.py ".\logs\ppo_lstm\ppo_lstm_semantic_temporal_shaped_v3_macro6_1k_save100\checkpoints\junimo_ppo_lstm_ep*.zip" --episodes 5 --frame-skip 2 --semantic-features --temporal-features --reward-version shaped_v3 --action-mode macro --macro-action-frames 6 --score-reward-coef 0.0 --gap-landing-confirm-steps 2 --max-steps-per-episode 300 --out logs\ppo_lstm\ppo_lstm_semantic_temporal_shaped_v3_macro6_1k_save100\evaluation_5ep.csv
```

## Command trace/debug policy

Setelah punya checkpoint, jalankan:

```powershell
python .\scripts\trace_policy_rollout.py --algorithm ppo --model .\logs\ppo\ppo_semantic_temporal_shaped_v3_tap_macro_5k\checkpoints\junimo_ppo_ep001000_stepsXXXXX.zip --episodes 5 --frame-skip 2 --semantic-features --temporal-features --reward-version shaped_v3 --action-mode tap_macro --macro-action-frames 6 --macro-release-frames 1 --score-reward-coef 0.0 --gap-landing-confirm-steps 2 --out logs\ppo\ppo_semantic_temporal_shaped_v3_tap_macro_5k\trace_ep1000.csv
```

Ganti `stepsXXXXX` sesuai nama checkpoint yang benar.

## Cara menilai apakah v4 membaik

Jangan hanya lihat `ep_rew_mean`.

Lihat juga:

```text
mean_length
mean_max_episode_x
gap_landing_rate
death_near_gap_rate
action_0_ratio
action_1_ratio
action_2_ratio
action_3_ratio
```

Target awal yang sehat:

- `death_near_gap_rate` turun;
- `gap_landing_rate` naik;
- `mean_max_episode_x` naik;
- action ratio lebih seimbang, bukan action 3 mendominasi terus;
- panjang episode naik konsisten, bukan cuma sesekali outlier.
