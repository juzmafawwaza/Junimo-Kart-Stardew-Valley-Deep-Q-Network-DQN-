# Eksperimen v10: memecah habit spam long jump

## Diagnosis dari run v9

Run yang dianalisis:

```text
logs/ppo/ppo_compact_anchored_v9_from_1500_to_3k/monitor.csv
```

Pada sekitar 180 episode pertama, perilaku visual spam long jump didukung oleh telemetry:

| Window | Reward mean | Length mean | Gap landing rate | Jump tanpa near-gap | Max X mean |
|---|---:|---:|---:|---:|---:|
| Episode 1–50 | 3.278 | 95.8 | 43.4% | 54.7% | 553.0 |
| 50 episode tengah | 1.587 | 68.9 | 27.1% | 60.9% | 500.4 |
| 50 episode terakhir | 2.004 | 62.0 | 26.7% | 57.8% | 497.3 |

Jadi belum ada bukti bahwa menambah episode v9 sedang memperbaiki perilaku. Gap landing dan episode length justru turun dibanding awal run.

TensorBoard menunjukkan policy entropy sekitar `0.16`. Untuk dua action, entropy maksimum adalah:

```text
H_max = ln(2) = 0.693
```

Nilai yang jauh lebih rendah berarti policy warisan episode 1.500 sudah sangat yakin pada action tertentu. Ini membuat fine-tuning sulit keluar dari habit lama.

## Kekurangan reward v6

`shaped_v6` hanya membebankan biaya ketika jump dimulai:

```text
jump_start_cost = -0.05
```

Action binary diputuskan setiap frame environment:

```text
0 = release
1 = hold
```

Setelah action 1 berhasil memulai jump, menahan action 1 selama seluruh fase airborne tidak mempunyai biaya tambahan. Artinya short jump dan long jump dapat menerima biaya start yang sama walaupun long jump memakai lebih banyak keputusan `hold`.

Telemetry v9 menunjukkan biaya jump start hanya sekitar `-0.20` per episode, sedangkan satu landing jurang memberi sekitar `+6.4`. Reward tersebut terlalu kecil untuk membedakan durasi jump.

## Reward `shaped_v7`

`shaped_v7` mempertahankan seluruh komponen v6 dan menambahkan regularisasi durasi hold:

```text
r_v7 = r_v6 + r_hold
```

dengan:

```text
r_hold(t) = 0
    jika cart grounded
    atau action = release
    atau hold_steps <= free_steps

r_hold(t) = -lambda_hold
    jika cart airborne
    dan action = hold
    dan hold_steps > free_steps
```

Default v10:

```text
free_steps = 4
lambda_hold = 0.02
```

Empat keputusan hold pertama gratis. Mulai keputusan airborne hold kelima, setiap hold tambahan dikenai `-0.02`.

Contoh:

| Durasi hold | Biaya durasi |
|---:|---:|
| 3 step | 0.00 |
| 4 step | 0.00 |
| 8 step | -0.08 |
| 14 step | -0.20 |
| 24 step | -0.40 |

Ini bukan aturan yang menentukan bahwa suatu gap harus memakai delapan frame. Model tetap bebas melakukan long jump jika reward survival dan landing lebih besar dari biaya hold. Biaya hanya membuat short jump lebih disukai ketika short dan long sama-sama berhasil.

## Mengapa default v10 mulai dari policy fresh

V9 memuat checkpoint episode 1.500 yang sudah mempunyai habit long jump dan entropy rendah. Selain itu, checkpoint tersebut dahulu belajar dari detector `legacy`, sedangkan v9 menggunakan detector `anchored`. Walaupun jumlah input tetap 27, arti fitur gap berubah.

V10 default tidak memuat checkpoint lama:

```text
LoadModel = ""
EpisodeOffset = 0
```

Keuntungannya:

- policy awal tidak membawa preferensi long jump;
- entropy awal binary mendekati `0.693`;
- model belajar langsung dari detector anchored dan reward v7;
- eksperimen lebih mudah dipertanggungjawabkan dalam paper.

Jika ingin menguji warm-start sebagai ablation, launcher tetap mendukung:

```powershell
.\scripts\run_ppo_compact_anchored_v10.ps1 `
  -LoadModel ".\logs\ppo\ppo_compact_v8_continue_to_3k\checkpoints\junimo_ppo_ep001500_steps124430.zip" `
  -EpisodeOffset 1500 `
  -RunName "ppo_compact_anchored_v10_warm_from_1500" `
  -ModelPath "models\ppo\junimo_ppo_compact_anchored_v10_warm"
```

Namun fresh run adalah default yang direkomendasikan berdasarkan data v9.

## Exploration

V10 menaikkan entropy coefficient:

```text
v9  ent_coef = 0.003
v10 ent_coef = 0.010
```

Tujuannya bukan membuat gameplay random selamanya. Entropy bonus memberi kesempatan policy awal mencoba release pada durasi yang berbeda. Jika setelah ribuan episode policy telah menemukan timing yang baik, entropy secara alami dapat turun karena keuntungan action mulai berbeda.

## Telemetry baru

`monitor.csv` v10 menambahkan:

```text
airborne_hold_penalty_steps
airborne_hold_penalty_total
max_jump_hold_steps
```

Interpretasi:

- `airborne_hold_penalty_steps`: berapa keputusan long-hold yang melewati free steps;
- `airborne_hold_penalty_total`: total biaya durasi dalam episode;
- `max_jump_hold_steps`: streak hold terpanjang dalam episode.

`summarize_monitor.py` dan `watch_monitor_table.py` juga menampilkan jump-start, jump tanpa near-gap, max hold, dan biaya hold.

## Menjalankan v10

Training v9 yang sedang berjalan memakai kode lama yang sudah dimuat ke memory. Mengedit file Python tidak mengubah proses tersebut. Hentikan v9 dengan `Ctrl+C`, kemudian jalankan:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\scripts\run_ppo_compact_anchored_v10.ps1
```

Default:

```text
episodes              = 3000
save every            = 250 episodes
observation           = compact, 27 features
gap detector          = anchored
reward                = shaped_v7
action                = binary
frame_skip            = 1
airborne free hold    = 4 steps
airborne hold penalty = 0.02 per extra step
ent_coef              = 0.01
```

Log:

```text
logs/ppo/ppo_compact_anchored_v10_fresh_3k
```

Final model:

```text
models/ppo/junimo_ppo_compact_anchored_v10_fresh_3k.zip
```

## Evaluasi

```powershell
.\scripts\evaluate_ppo_compact_anchored_v10.ps1 -Episodes 20
```

Jangan hanya memilih checkpoint dengan reward tertinggi. Minimal periksa:

```text
mean_length
mean_max_episode_x
gap_landing_rate
jump_start_without_near_gap_rate
mean_max_jump_hold_steps
mean_airborne_hold_penalty_steps
completion_rate
```

Target awal bukan langsung completion. Bukti pertama bahwa v10 bekerja adalah `max_jump_hold_steps` dan jump tanpa near-gap turun tanpa menghancurkan gap landing rate.
