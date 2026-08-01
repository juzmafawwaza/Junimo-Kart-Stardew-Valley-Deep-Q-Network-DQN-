# Eksperimen PPO v11: Teknik Takeoff/Landing di Ujung Rel

## 1. Masalah yang hendak diperbaiki

PPO v10 menaikkan reward dan gap landing rate, tetapi menemukan local optimum berupa long-jump spam. Data episode 1-100 dibanding 601-662 menunjukkan:

| Metrik | Episode 1-100 | Episode 601-662 |
|---|---:|---:|
| Grounded time | 20,74% | 6,66% |
| Jump-held time | 49,53% | 63,20% |
| Mean maximum hold | 4,97 | 8,08 |
| Gap landing rate | 2,2% | 31,0% |
| Mean reward | -2,299 | 2,090 |

Agent tidak gagal belajar. Ia menemukan bahwa terus melompat memberikan return lebih besar daripada biaya jump v10. Oleh karena itu, menambah episode dengan reward yang sama berisiko memperkuat kebiasaan tersebut.

## 2. Apakah reward berdasarkan action valid?

Secara matematis, reward reinforcement learning boleh bergantung pada state, action, dan state berikutnya:

$$
r_t = R(s_t, a_t, s_{t+1})
$$

Jadi action-based reward bukan sesuatu yang tidak valid. Masalahnya adalah reward langsung pada action mudah menghasilkan perilaku yang tampak benar tetapi tidak mencapai outcome. Contoh yang dihindari v11:

```text
jump tepat di ujung -> langsung diberi reward -> kemudian jatuh
```

Pada v11, takeoff hanya dicatat. Bonus teknik baru dibayar jika gap yang sama benar-benar berhasil dilalui dan landing bertahan selama confirmation steps.

## 3. Definisi zona ujung rel

Reward tidak memakai satu pixel exact karena update bridge, ukuran collision box cart, elevasi rel, dan frame timing dapat berubah. Kualitas menggunakan fungsi triangular yang kontinu.

### 3.1 Kualitas takeoff

Jarak sisi depan cart terhadap ujung rel:

$$
d_T = x_{gap\_start} - x_{cart\_front}
$$

Dengan target default $d_T^*=12$ pixel dan toleransi $\tau_T=48$ pixel:

$$
q_T = \operatorname{clip}\left(1-\frac{|d_T-d_T^*|}{\tau_T},0,1\right)
$$

- $q_T=1$: takeoff berada pada target aman dekat tip.
- $0<q_T<1$: takeoff masih berada dalam zona tip.
- $q_T=0$: takeoff terlalu awal atau terlalu terlambat.

Jika agent melakukan beberapa jump sebelum gap, takeoff terbaru yang dipakai. Ini mencegah lompatan awal di rel mengambil kredit untuk lompatan sebenarnya di ujung.

### 3.2 Kualitas landing

Kedalaman pusat cart setelah kontak pertama dengan rel tujuan:

$$
d_L = x_{cart\_center} - x_{landing\_start}
$$

Dengan target default $d_L^*=16$ pixel dan toleransi $\tau_L=64$ pixel:

$$
q_L = \operatorname{clip}\left(1-\frac{|d_L-d_L^*|}{\tau_L},0,1\right)
$$

Target 16 pixel berarti cart sudah masuk ke rel, bukan menggantung tepat pada satu pixel pertama. Ini lebih aman daripada memberi nilai maksimum pada kontak paling tipis.

### 3.3 Bonus teknik yang outcome-conditioned

Kualitas takeoff dan landing digabungkan memakai geometric mean:

$$
R_{tip} = B_{tip}\sqrt{q_Tq_L}
$$

Default $B_{tip}=1{,}5$. Jika salah satu kualitas nol, bonus teknik juga nol. Reward hanya dibayar setelah confirmed gap landing. Takeoff bagus yang berakhir dengan kematian tidak mendapat bonus.

## 4. Anti-bunny-hop shaping

`shaped_v8` mempertahankan seluruh reward `shaped_v7`, kemudian menambahkan tiga sinyal kecil.

### 4.1 Grounded progress bonus

$$
R_{ground}=c_g\Delta x_{max}
$$

Default $c_g=0{,}005$. Reward hanya aktif jika state sebelum dan sesudah transition sama-sama grounded dan cart tidak mati. Ini memberi alasan ekonomis untuk tetap di rel ketika melompat tidak diperlukan.

### 4.2 Unnecessary jump penalty

Tambahan `-0,15` dibebankan ketika jump benar-benar dimulai, tidak ada obstacle dekat, dan takeoff tidak berada dalam zona tip gap. Ini berbeda dari menghukum setiap action `1`: re-press di udara tidak dihitung sebagai jump baru.

### 4.3 Non-gap airborne penalty

Tambahan `-0,01` per transition diberikan ketika cart terus airborne tanpa takeoff gap yang berkualitas dan tanpa obstacle relevan. Lompatan gap yang dimulai dari zona tip tidak terkena komponen ini.

### 4.4 Reward total

Secara ringkas:

$$
R_{v11}=R_{v10}+R_{ground}+R_{tip}-C_{unnecessary}-C_{non\_gap\_air}
$$

Penalti long hold v10 tetap berlaku setelah empat hold steps gratis.

## 5. Skala reward default

| Event | Nilai default |
|---|---:|
| Progress baru | `0.01 × dx` |
| Grounded progress tambahan | `0.005 × dx` |
| Confirmed gap landing | `5 + 0.015 × min(gap_width, 120)` |
| Teknik tip sempurna | maksimum `+1.5` |
| Coin | `+0.2` |
| Fruit | `+2.0` |
| Jump start umum | `-0.05` |
| Jump tidak relevan | tambahan `-0.15` |
| Non-gap airtime | `-0.01/step` |
| Hold setelah free steps | `-0.02/step` |
| Death | `-5` ditambah bounded gap-miss penalty bila relevan |

Bonus tip sengaja lebih kecil daripada base gap landing. Prioritas pertama tetap selamat; teknik takeoff/landing hanya menjadi refinement.

## 6. Perubahan kode

### `junimo_rl/env.py`

- Menambahkan reward version `shaped_v8`.
- Menambahkan `_record_gap_takeoff()` untuk merekam jarak dan kualitas takeoff tanpa langsung memberi reward.
- Menambahkan `_prepare_gap_tip_technique()` untuk menghitung landing depth, landing quality, dan bonus gabungan.
- Menambahkan `_pay_gap_landing()` agar telemetry tip hanya dicatat saat landing benar-benar dibayar.
- Menambahkan `_shaped_v8_reward()` untuk grounded progress, unnecessary jump, dan non-gap airtime.
- Mempertahankan anchored gap geometry selama airborne.

### `scripts/train_ppo.py` dan `scripts/evaluate_ppo_models.py`

Keduanya menerima seluruh parameter v11 dan evaluator menghasilkan metrik teknik tambahan.

### `junimo_rl/state_trace.py`

Trace per-step memisahkan komponen:

```text
grounded_progress_reward
unnecessary_jump_reward
non_gap_airborne_reward
gap_tip_technique_reward
```

### Script launcher

```text
scripts/run_ppo_compact_tip_v11.ps1
scripts/evaluate_ppo_compact_tip_v11.ps1
```

## 7. Telemetry baru

| Kolom | Arti |
|---|---|
| `grounded_progress_bonus_total` | Total bonus progress saat tetap grounded |
| `unnecessary_jump_events` | Jumlah jump start di luar konteks berguna |
| `unnecessary_jump_penalty_total` | Total biaya jump tersebut |
| `non_gap_airborne_steps` | Airtime tanpa gap takeoff berkualitas/obstacle |
| `gap_takeoff_events` | Jump start yang tercatat saat ada active gap |
| `takeoff_tip_distance_total` | Total jarak sisi depan cart ke tip saat takeoff |
| `takeoff_tip_quality_total` | Total kualitas seluruh gap takeoff |
| `edge_qualified_landings` | Landing sukses yang memiliki takeoff record |
| `successful_takeoff_tip_quality_total` | Kualitas takeoff untuk landing sukses |
| `landing_tip_depth_total` | Kedalaman landing sukses |
| `landing_tip_quality_total` | Kualitas posisi landing sukses |
| `edge_technique_reward_total` | Total bonus teknik yang benar-benar dibayar |

## 8. Menjalankan training

Tutup training lama atau tunggu checkpoint yang diinginkan, lalu:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\scripts\run_ppo_compact_tip_v11.ps1
```

Launcher default memulai model fresh. Hal ini penting untuk membandingkan v10 dan v11 tanpa mewarisi policy long-jump v10.

Dry run untuk melihat command lengkap tanpa menjalankan:

```powershell
.\scripts\run_ppo_compact_tip_v11.ps1 -DryRun
```

## 9. Kriteria keberhasilan awal

Jangan menilai hanya dari `mean_reward`, karena formula reward v10 dan v11 berbeda. Setelah 250-500 episode, bandingkan:

- `grounded_pct` naik dari baseline v10 sekitar 6-8%.
- `jump_held_pct` turun dari sekitar 63%.
- `unneeded_jump_ep` turun antar-window.
- `takeoff_tip_q` dan `landing_tip_q` naik.
- `gap_land_rate` tidak jatuh tajam.
- `max_x_mean` dan episode length tetap meningkat.

Jika grounded time naik tetapi gap landing runtuh, shaping terlalu keras. Jika grounded time tetap rendah dan unnecessary jump tidak turun, biaya anti-spam masih terlalu kecil atau fitur tip tidak cukup akurat.

## 10. Evaluasi

Setelah checkpoint tersedia dan training berhenti:

```powershell
.\scripts\evaluate_ppo_compact_tip_v11.ps1 -Episodes 20
```

Evaluator default membandingkan checkpoint v10 episode 500 dengan seluruh checkpoint v11 menggunakan environment v11 yang sama. Jalankan deterministic sebagai hasil utama dan stochastic sebagai pemeriksaan tambahan:

```powershell
.\scripts\evaluate_ppo_compact_tip_v11.ps1 -Episodes 20 -Stochastic `
  -Out logs\ppo\ppo_compact_tip_v11_fresh_3k\evaluation_stochastic_20ep.csv
```

