# PPO Compact / PPO-LSTM Compact v8

Dokumen ini menjelaskan perubahan setelah audit v7, alasan matematisnya, isi kode, dan cara menjalankan eksperimen baru.

## 1. Masalah yang ditemukan pada v7

### Observation terlalu besar

`observation_mode=multi` pada v7 berisi 4.448 nilai:

```text
legacy state       306
semantic            17
temporal              5
recent actions        24
spatial map         4096
total               4448
```

Spatial map tersebut diproses dengan `Flatten`, bukan CNN. PPO-LSTM v7 akhirnya memiliki sekitar 4.712.899 parameter, sedangkan data yang terkumpul hanya puluhan ribu step.

### Tick bridge masuk ke observation

`snapshot.version` adalah nomor update bridge yang terus naik. Nilai ini bukan keadaan permainan. Model dapat mengasosiasikan keputusan dengan waktu proses dan menerima nilai yang jauh berbeda ketika evaluasi dilakukan kemudian.

Slot version pada observation lama sekarang selalu diisi `0.0` agar dimensi model lama tetap konsisten tanpa membocorkan tick.

### Progress lama tidak valid

`distanceToTravel` bukan panjang total level. Pada trace pernah ditemukan `player_x > 600` ketika `distanceToTravel` sekitar 150, sehingga nilai bernama `progress_fraction` menjadi lebih besar dari satu.

V8 tidak memakai nilai tersebut untuk compact observation. Slot diagnostik lama dibatasi ke `[0, 1]` menggunakan world-x yang dinormalisasi.

### Score bukan bukti pickup

Junimo Kart menambah score ketika kereta bergerak. Karena itu kondisi berikut salah sebagai detector coin/fruit:

```python
score_delta > 0
```

Bridge v8 mengirim `entity.id`. Python baru menyatakan pickup berhasil jika:

1. entity pickup ada pada state lama;
2. entity cukup dekat atau overlap dengan player;
3. ID entity tersebut hilang pada state baru.

### Kematian dihitung berulang

Pada v4, jatuh dekat jurang dapat menerima penalti life, game-over, dan gap-death sekaligus. V5 hanya menggunakan satu penalti terminal.

### Input jump dapat ter-buffer ketika airborne

Bridge lama memanggil `QueueJump()` setiap rising edge, termasuk ketika player masih di udara. V8 hanya mengizinkan rising edge memulai jump ketika `grounded=true`. Menahan tombol tetap dapat memperpanjang jump; release tetap memotong jump sesuai mekanik game.

## 2. Compact observation

Implementasi berada di `junimo_rl/env.py`, fungsi `compact_feature_vector()`.

V8 menggunakan 27 fitur yang semuanya dibatasi ke `[-1, 1]`:

| No. | Fitur | Arti |
|---:|---|---|
| 1 | `vx` | Kecepatan horizontal |
| 2 | `vy` | Kecepatan vertikal |
| 3 | `player_track_delta_y` | Posisi vertikal player relatif terhadap rel |
| 4 | `grounded` | Player menyentuh rel |
| 5 | `jumping` | Status jumping dari game |
| 6 | `jump_held` | Tombol jump sedang ditahan |
| 7 | `jump_ready` | Jump baru boleh dimulai |
| 8 | `current_track_present` | Rel penopang berhasil ditemukan |
| 9 | `current_track_end_dx` | Jarak ke ujung rel saat ini |
| 10 | `current_track_type_id` | Jenis rel saat ini |
| 11 | `gap_present` | Ada jurang di depan |
| 12 | `gap_start_dx` | Jarak ke awal jurang |
| 13 | `gap_width` | Lebar jurang |
| 14 | `gap_end_dx` | Jarak ke ujung jurang/awal landing |
| 15 | `landing_delta_y` | Perbedaan tinggi rel tujuan |
| 16 | `obstacle_present` | Ada obstacle di depan |
| 17 | `obstacle_dx` | Jarak horizontal obstacle |
| 18 | `obstacle_delta_y` | Posisi vertikal obstacle relatif player |
| 19 | `pickup_present` | Ada pickup di depan |
| 20 | `pickup_dx` | Jarak horizontal pickup |
| 21 | `pickup_delta_y` | Posisi vertikal pickup relatif player |
| 22 | `jump_held_steps` | Lama tombol ditahan |
| 23 | `airborne_steps` | Lama berada di udara |
| 24 | `grounded_steps` | Lama berada di tanah |
| 25 | `last_action_holds_jump` | Aksi sebelumnya hold/release |
| 26 | `current_theme` | Tema/stage Junimo Kart |
| 27 | `levels_beat` | Jumlah level yang selesai |

Fitur bersifat egocentric: jarak dihitung relatif terhadap player, bukan koordinat level yang random. Karena itu pola track boleh berubah, tetapi konsep "jurang berjarak 120 pixel dengan landing lebih tinggi" tetap memiliki representasi yang sama.

## 3. Reward shaped_v5

Reward utama:

\[
r_t =
c_p \Delta x_{\max}
+ r_{pickup}
+ r_{gap}
+ r_{level}
+ r_{complete}
- r_{death}
\]

Dengan default launcher:

```text
progress_reward_coef = 0.01
death_penalty        = 5.0
level reward         = 50.0
completion reward    = 200.0
coin reward          = 0.2
fruit reward         = 2.0
gap base reward      = 5.0
gap width coef       = 0.015
```

Progress memakai posisi terjauh yang pernah dicapai dalam episode:

\[
\Delta x_{\max} = \max(x_{t+1}-x_{\max,t},0)
\]

Dengan demikian bergerak mundur lalu maju ke lokasi lama tidak menghasilkan reward kedua kali.

Reward landing jurang:

\[
r_{gap}=5+0.015\min(w_{gap},120)
\]

Contoh jurang 96 pixel:

\[
r_{gap}=5+0.015(96)=6.44
\]

Kematian hanya `-5`, bukan sekitar `-115` seperti kombinasi penalti v4. Tujuannya agar critic dapat membedakan episode yang melewati satu atau lebih jurang sebelum akhirnya mati.

## 4. Reward trace

Setiap `info` environment sekarang memiliki `reward_components`:

```text
progress
pickup
gap
level
completion
death
```

CSV state trace memiliki kolom terpisah untuk komponen tersebut. Ini memungkinkan analisis apakah kenaikan episode reward berasal dari progress nyata, landing, pickup, atau sekadar perubahan terminal.

## 5. Ukuran model

Hasil pemeriksaan lokal:

```text
PPO-LSTM v7 multi-input : 4,712,899 parameter
PPO compact v8          :    12,099 parameter
PPO-LSTM compact v8     :    64,451 parameter
```

PPO-LSTM v8 berkurang sekitar 98,63% dibanding v7. Ini membuat setiap update jauh lebih cepat dan jumlah data per parameter jauh lebih masuk akal.

## 6. Mengapa run harus mulai dari nol

Model v7 tidak dapat dilanjutkan ke v8 karena:

- observation berubah dari 4.448 menjadi 27 nilai;
- definisi reward berubah;
- bridge mengubah semantics rising-edge jump;
- arsitektur policy berubah.

Checkpoint v7 tetap berguna sebagai hasil eksperimen lama, tetapi tidak kompatibel untuk `--load-model` pada v8.

## 7. Urutan instalasi

Training v7 yang masih berjalan harus dihentikan dengan `Ctrl+C`. Script training akan menyimpan final model.

Setelah itu tutup Stardew Valley/SMAPI dan jalankan:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\scripts\install_bridge_v8.ps1
```

Buka kembali Stardew melalui SMAPI dan load save sampai masuk world.

## 8. Training yang direkomendasikan

Mulai dari PPO compact terlebih dahulu:

```powershell
.\scripts\run_ppo_compact_v8.ps1
```

Default-nya:

```text
episodes        = 5000
checkpoint      = setiap 500 episode
frame_skip      = 1
observation     = compact (27 fitur)
action          = binary
reward          = shaped_v5
learning_rate   = 0.0003
n_steps         = 1024
n_epochs        = 5
entropy_coef    = 0.003
```

PPO-LSTM compact tersedia untuk eksperimen berikutnya:

```powershell
.\scripts\run_ppo_lstm_compact_v8.ps1
```

PPO-LSTM sebaiknya dijalankan setelah PPO compact terbukti mengalahkan baseline first-gap success sekitar 10–12%.

## 9. Debug singkat

Jangan print setiap step pada training berjam-jam. Untuk debug 50–100 episode:

```powershell
.\scripts\run_ppo_compact_v8.ps1 `
  -Episodes 100 `
  -RunName "ppo_compact_v8_debug" `
  -ModelPath "models\ppo\junimo_ppo_compact_v8_debug" `
  -TracePrintFreq 1 `
  -TraceSimpleAction `
  -TraceCsv "logs\ppo\ppo_compact_v8_debug\state_trace.csv"
```

Untuk training utama biarkan `TracePrintFreq=0`; jika memerlukan CSV, gunakan `TraceCsvFreq=10` atau lebih.

## 10. Evaluasi

Deterministic PPO:

```powershell
.\scripts\evaluate_ppo_compact_v8.ps1
```

Stochastic PPO:

```powershell
.\scripts\evaluate_ppo_compact_v8.ps1 `
  -Stochastic `
  -Out "logs\ppo\ppo_compact_shaped_v5_binary_5k\evaluation_stochastic_100ep.csv"
```

PPO-LSTM menggunakan launcher yang setara:

```powershell
.\scripts\evaluate_ppo_lstm_compact_v8.ps1
```

Gunakan minimal 100 episode karena layout bersifat random. Bandingkan:

- `gap_landing_rate`;
- `mean_max_episode_x`;
- `mean_levels_beat`;
- `completion_rate`;
- `action_0_ratio` dan `action_1_ratio`;
- deterministic versus stochastic.

Gate awal yang disarankan:

```text
first-gap/gap landing rate > 30%
deterministic policy tidak collapse ke satu action
explained_variance mulai jelas di atas 0
mean_max_episode_x meningkat antarkelompok checkpoint
```

## 11. File yang berubah

```text
junimo_rl/env.py
    Compact observation, actual pickup detection, shaped_v5, reward components.

junimo_rl/state_trace.py
    Kolom reward component, jump_ready, track end, dan gap end.

src/JunimoKartRLBridge/Protocol.cs
    Entity ID dan JumpReady.

src/JunimoKartRLBridge/ModEntry.cs
    Entity runtime ID dan grounded-only jump rising edge.

scripts/train_ppo.py
scripts/train_ppo_lstm.py
    Dukungan compact/shaped_v5 dan seluruh parameter reward baru.

scripts/evaluate_ppo_models.py
scripts/evaluate_ppo_lstm_models.py
    Dukungan v8 dan evaluasi deterministic/stochastic.

scripts/run_ppo_compact_v8.ps1
scripts/run_ppo_lstm_compact_v8.ps1
    Launcher training baru.

scripts/evaluate_ppo_compact_v8.ps1
scripts/evaluate_ppo_lstm_compact_v8.ps1
    Launcher evaluasi 100 episode.

scripts/install_bridge_v8.ps1
    Build dan deploy bridge setelah game ditutup.

tests/test_env_v8.py
    Unit test observation dan reward v8.
```
