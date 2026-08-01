# Eksperimen `shaped_v6`: anti-spam jump dan distance-aware gap death

Dokumen ini menjelaskan perubahan reward setelah PPO compact v8 memperlihatkan pola long jump berulang. `shaped_v5` tetap tersedia dan tidak diubah, sehingga hasil lama masih dapat direproduksi.

## 1. Baseline sebelum perubahan

Continuation `shaped_v5` dihentikan setelah checkpoint episode 1.500 tersimpan. Evaluasi deterministic memakai 20 episode per checkpoint dan konfigurasi observation/action yang sama.

| Checkpoint | Mean reward | Mean length | Mean score | Mean max X | Gap landing rate | Jump hold ratio | Completion |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.000 | 4.466 | 56.2 | 719.0 | 488.6 | 37.5% | 41.1% | 0% |
| 1.250 | 7.296 | 76.0 | 1,217.5 | 546.4 | 34.4% | 51.3% | 0% |
| 1.500 | **8.731** | 75.0 | 890.5 | **635.7** | **51.3%** | 49.1% | 0% |

Checkpoint 1.500 dipilih sebagai sumber next run karena unggul pada kombinasi reward, jarak, dan keberhasilan jurang. Checkpoint 1.250 memiliki satu max score 14.980 yang membuat mean score lebih rentan terhadap outlier.

CSV baseline:

```text
logs/ppo/ppo_compact_v8_continue_to_3k/evaluation_1000_1250_1500_deterministic_20ep.csv
```

## 2. Masalah yang ingin diselesaikan

`shaped_v5` membayar progress, pickup, gap landing, level, dan completion serta memberi satu death penalty. Reward itu tidak mempunyai biaya untuk memulai jump. Akibatnya, strategi “sering long jump” dapat menjadi local optimum karena:

1. jump pada rel lurus sering tidak langsung menyebabkan kematian;
2. progress reward tetap diterima selama bergerak maju;
3. long jump kadang berhasil melewati gap tanpa membutuhkan timing presisi.

Penalti berdasarkan jarak miss saja tidak cukup untuk memperbaiki spam. Penalti tersebut mengajari akurasi landing, tetapi dapat mendorong model menahan jump lebih lama agar bergerak semakin dekat. Karena itu v6 memisahkan dua sinyal.

## 3. Rumus reward v6

Reward dasar tetap `shaped_v5`:

```text
r_v6 = r_v5 + jump_start_reward + gap_miss_reward
```

### 3.1 Biaya memulai jump

Jump baru hanya valid ketika action berubah dari release menjadi hold, cart grounded, dan bridge melaporkan `jumpReady`.

```text
jump_start_reward = -lambda_jump * I(real_grounded_jump_start)
lambda_jump       = 0.02
```

Airborne re-press tidak dikenai biaya karena bridge v8 mengabaikannya dan tidak membuat jump baru. Penalti ini juga tidak memeriksa gap, sehingga bukan aturan “jika gap sekian maka jump”. Model tetap menentukan sendiri kapan manfaat jump lebih besar daripada biaya kecilnya.

### 3.2 Penalti kematian berdasarkan jarak miss

Untuk satu gap aktif:

```text
x_furthest = posisi horizontal terjauh selama gap attempt
[L, R]     = seluruh interval horizontal rel landing
W          = lebar gap
```

Jarak ke interval landing:

```text
d = L - x_furthest, jika x_furthest < L
d = 0,              jika L <= x_furthest <= R
d = x_furthest - R, jika x_furthest > R
```

Normalisasi dan reward:

```text
miss_ratio      = clip(d / max(W, 32), 0, 1)
gap_miss_reward = -lambda_miss * miss_ratio^2
lambda_miss     = 2.0
```

Total death penalty ketika gagal pada gap menjadi:

```text
death_total = -5.0 - 2.0 * miss_ratio^2
```

Dengan demikian:

| `miss_ratio` | Base death | Tambahan miss | Total |
|---:|---:|---:|---:|
| 0.00 | -5.0 | 0.000 | -5.000 |
| 0.25 | -5.0 | -0.125 | -5.125 |
| 0.50 | -5.0 | -0.500 | -5.500 |
| 1.00 | -5.0 | -2.000 | -7.000 |

Near miss tidak dibuat lebih ringan daripada death penalty v5. Alasannya, progress reward sudah membayar gerakan mendekati landing. Mengurangi near-death menjadi -1 atau -2 dapat membuat episode mati tetap terlalu menarik.

## 4. Mengapa memakai interval landing

Landing bukan satu titik. Rel tujuan mempunyai sisi kiri dan kanan. Jika cart berada secara horizontal di atas bagian mana pun dari rel tetapi jatuh karena trajectory terlalu rendah, horizontal miss distance harus nol.

`semantic_feature_snapshot()` sekarang menghitung:

```text
landing_end_dx
landing_width
```

Nilai tersebut dipakai untuk telemetry/reward, tetapi tidak menambah ukuran compact observation. Model tetap menggunakan 27 input yang sama, sehingga checkpoint v5 dapat dimuat untuk fine-tuning v6.

## 5. Telemetry baru

`monitor.csv` menambahkan kolom:

```text
jump_start_events
jump_start_near_gap_events
jump_start_without_near_gap_events
jump_start_penalty_total
gap_miss_deaths
gap_miss_distance_total
gap_miss_ratio_total
gap_miss_penalty_total
```

Evaluator juga menghasilkan:

```text
mean_jump_starts
jump_start_near_gap_rate
jump_start_without_near_gap_rate
mean_jump_start_penalty_total
mean_gap_miss_deaths
mean_gap_miss_distance
mean_gap_miss_ratio
mean_gap_miss_penalty_total
```

Keberhasilan v6 tidak boleh dinilai hanya dari mean reward karena skala reward berubah. Bandingkan terutama:

1. completion rate dan levels beat;
2. gap landing rate;
3. mean/max score dan mean max X;
4. mean jump starts;
5. jump-start-without-near-gap rate;
6. mean gap miss ratio.

## 6. File yang berubah

```text
junimo_rl/env.py
    shaped_v6, landing interval, jump-start detector, miss-distance telemetry.

junimo_rl/state_trace.py
    reward component jump_start dan gap_miss.

scripts/train_ppo.py
scripts/train_ppo_lstm.py
    parameter training shaped_v6.

scripts/evaluate_ppo_models.py
scripts/evaluate_ppo_lstm_models.py
    parameter dan metrik evaluasi shaped_v6.

scripts/trace_policy_rollout.py
    dukungan compact observation serta shaped_v5/v6.

scripts/run_ppo_compact_shaped_v6.ps1
    launcher continuation dari checkpoint 1.500.

scripts/evaluate_ppo_compact_shaped_v6.ps1
    launcher evaluasi deterministic/stochastic v6.
```

## 7. Menjalankan next run

Pastikan Stardew dibuka lewat SMAPI dan Junimo Kart Progress Mode sudah aktif.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\scripts\run_ppo_compact_shaped_v6.ps1
```

Default launcher:

```text
source checkpoint = episode 1.500 shaped_v5
additional run    = 1.500 episode
final numbering   = episode 3.000
checkpoint        = setiap 250 episode
observation       = compact, 27 fitur
action            = binary, frame_skip 1
reward            = shaped_v6
jump penalty      = 0.02 per real jump start
gap miss coef     = 2.0 quadratic
```

Cek command tanpa menjalankan:

```powershell
.\scripts\run_ppo_compact_shaped_v6.ps1 -DryRun
```

## 8. Evaluasi

```powershell
.\scripts\evaluate_ppo_compact_shaped_v6.ps1 -Episodes 20
```

Stochastic evaluation:

```powershell
.\scripts\evaluate_ppo_compact_shaped_v6.ps1 -Episodes 20 -Stochastic `
  -Out "logs\ppo\ppo_compact_shaped_v6_from_1500_to_3k\evaluation_stochastic_20ep.csv"
```

## 9. Desain eksperimen paper

Untuk ablation yang adil, semua varian harus dimulai dari checkpoint 1.500 yang sama dan menggunakan hyperparameter PPO yang sama:

```text
A: shaped_v5
B: shaped_v5 + jump-start cost
C: shaped_v5 + distance-aware gap death
D: shaped_v6, kedua improvement
```

Jalankan beberapa seed dan lebih dari 20 episode evaluasi untuk hasil paper. Evaluasi 20 episode cukup untuk screening awal, tetapi belum cukup untuk kesimpulan statistik kuat karena layout Junimo Kart random dan score memiliki outlier besar.

## 10. Evaluasi otomatis setelah 1.500 episode v6

Watcher berikut menunggu proses training selesai tanpa membuka koneksi bridge. Evaluasi hanya dimulai jika checkpoint episode 3.000 benar-benar tersedia:

```powershell
.\scripts\watch_and_evaluate_ppo_shaped_v6.ps1 -TrainingPid <PID>
```

Model yang diuji memakai reward/observation/action evaluation yang sama:

```text
baseline shaped_v5 episode 1.500
checkpoint shaped_v6 episode 1.750
checkpoint shaped_v6 episode 2.000
checkpoint shaped_v6 episode 2.250
checkpoint shaped_v6 episode 2.500
checkpoint shaped_v6 episode 2.750
checkpoint shaped_v6 episode 3.000
```

Output default:

```text
logs/ppo/ppo_compact_shaped_v6_from_1500_to_3k/evaluation_after_1500_v6_deterministic_20ep.csv
```

Jika training dihentikan sebelum checkpoint 3.000 atau game/SMAPI ditutup sebelum evaluasi, watcher tidak dapat menyelesaikan evaluasi. Game harus tetap terbuka setelah training berakhir.
