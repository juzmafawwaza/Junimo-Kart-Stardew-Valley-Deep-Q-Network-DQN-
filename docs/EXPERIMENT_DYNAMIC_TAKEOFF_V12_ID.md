# Eksperimen PPO v12: Dynamic Takeoff dan Pencegahan No-Jump Collapse

## 1. Diagnosis v11

V11 berhasil mengurangi bunny-hop, tetapi kemudian memasuki fase no-jump collapse: cart tetap grounded hingga jatuh pada gap pertama.

| Metrik | Episode 701-800 | Episode 1101-1200 |
|---|---:|---:|
| Mean episode length | 78,53 | 18,34 |
| Mean maximum X | 403,1 | 348,8 |
| Gap landing rate | 25,5% | 8,0% |
| Jump starts per episode | 1,86 | 1,02 |
| Grounded time | 44,93% | 56,43% |

Entropy policy turun dari sekitar `0.622` pada 10.240 steps menjadi `0.222` pada 69.632 steps. Karena entropy maksimum untuk dua action sekitar `0.693`, policy telah menjadi hampir deterministik ketika collapse terjadi.

Pengamatan lanjutan menunjukkan policy kemudian pulih pada episode 1401-1476: mean length kembali sekitar 65, gap landing rate sekitar 31,6%, dan mean reward sekitar +1,26. Karena itu fase tersebut bukan kerusakan permanen, melainkan policy oscillation. Namun pemulihan juga membawa maximum hold kembali ke sekitar 13,3 steps. V11 masih berpindah antara dua ekstrem—terlalu sedikit jump dan hold terlalu panjang—sehingga v12 ditujukan untuk memperbaiki stabilitas, bukan hanya satu snapshot performa.

Dalam 100 episode terakhir yang dianalisis:

```text
mean progress reward        = +3.265
mean grounded bonus         = +0.889
mean technique reward       = +0.061
death penalty               = -5.000
mean actual takeoff distance= 66.789 px
fixed v11 takeoff target    = 12 px
```

V11 memberi dense reward untuk tetap grounded, tetapi bonus teknik sangat jarang. Lebih buruk lagi, takeoff rata-rata 66,8 px sebelum tip dianggap berkualitas nol oleh target tetap 12 px dengan toleransi 48 px. Kombinasi ini mendorong fase no-jump; PPO kemudian keluar dari fase tersebut, tetapi kembali menuju hold panjang.

## 2. Tujuan v12

V12 harus memenuhi empat hal:

1. Tidak memberi reward untuk berjalan grounded sampai masuk zona gap berbahaya.
2. Tidak menghukum takeoff yang masuk akal hanya karena tidak tepat 12 px dari tip.
3. Tetap mengurangi jump spam pada rel aman.
4. Mempertahankan eksplorasi PPO agar policy tidak cepat mengunci pada satu action.

Nama reward internal v12 adalah `shaped_v9`.

## 3. Dynamic takeoff target

Takeoff target kini bergantung pada lebar gap dan beda tinggi landing:

$$
d_T^* = \operatorname{clip}\left(
0.65w
+0.25\max(-\Delta y,0)
-0.10\max(\Delta y,0),
32,
112
\right)
$$

Keterangan:

- $w$: lebar gap.
- $\Delta y<0$: landing lebih tinggi, sehingga takeoff dimajukan.
- $\Delta y>0$: landing lebih rendah, sehingga takeoff boleh sedikit lebih lambat.
- Output dibatasi antara 32 dan 112 pixel.

Contoh gap 96 px dengan landing setinggi sama:

$$
d_T^*=0.65(96)=62.4\text{ px}
$$

Nilai tersebut dekat dengan takeoff aktual v11 sekitar 66,8 px. Untuk landing 32 px lebih tinggi:

$$
d_T^*=0.65(96)+0.25(32)=70.4\text{ px}
$$

Kualitas tetap kontinu:

$$
q_T=\operatorname{clip}\left(1-\frac{|d_T-d_T^*|}{64},0,1\right)
$$

Dynamic target bukan action rule yang memaksa agent melompat pada satu koordinat. Nilai tersebut hanya memengaruhi bonus teknik setelah gap benar-benar berhasil dilalui.

## 4. Perubahan reward

### 4.1 Grounded bonus hanya pada rel aman

Bonus progress grounded turun dari `0.005 × dx` menjadi `0.002 × dx`. Bonus tidak diberikan jika:

- gap telah memasuki dynamic decision zone;
- obstacle berada dekat;
- transition berakhir dengan death.

Dengan demikian, berjalan di rel lurus tetap lebih efisien daripada spam jump, tetapi menunggu sampai jatuh tidak lagi terus diberi bonus.

### 4.2 Penalti jump di luar tip dihapus

V11 memberi tambahan `-0.15` ketika jump dimulai di luar fixed tip zone. V12 menghapus komponen ini sepenuhnya. Semua jump masih membayar energy cost umum `-0.03`, tetapi jump yang diperlukan tidak mendapat label “unnecessary” berdasarkan target sempit.

### 4.3 Gap airtime tidak bergantung pada tip quality

Pada v11, airtime hanya dianggap valid ketika `takeoff_tip_quality > 0`. Akibatnya, takeoff valid sekitar 67 px dapat tetap dihukum sebagai non-gap airtime.

Pada v12:

```text
active gap + takeoff tercatat
-> seluruh airtime percobaan tersebut dianggap gap airtime
```

Kualitas takeoff tidak menentukan apakah airtime valid.

### 4.4 Imminent-gap inaction cost

Zona keputusan dimulai pada:

$$
d_{trigger}=d_T^*+8+24
$$

Angka 8 mengubah jarak dari sisi depan cart menjadi jarak dari pusat cart. Angka 24 adalah margin default.

Jika cart grounded, jump-ready, sudah berada dalam zona ini, tetapi memilih release, diberikan biaya kecil `-0.05` per step. Ini bukan reward untuk action jump; sinyal ini hanya menyatakan bahwa terus menunggu di zona yang segera berakhir dengan gap adalah keputusan berisiko.

### 4.5 Bonus teknik diperkecil

Bonus teknik maksimum diturunkan dari `+1.5` menjadi `+0.5`. Formula outcome-conditioned tetap:

$$
R_{technique}=0.5\sqrt{q_Tq_L}
$$

Takeoff yang mati tetap mendapat bonus nol. Survival dan confirmed gap landing tetap menjadi objective utama.

### 4.6 Hold dan non-gap airtime dibuat lebih lunak

```text
airborne hold free steps : 4 -> 6
extra hold cost          : -0.02 -> -0.01 per step
non-gap airtime cost     : -0.01 -> -0.005 per step
jump-start cost          : -0.05 -> -0.03
```

Perubahan ini memberi ruang untuk gap lebar tanpa mengembalikan insentif long-jump spam penuh.

## 5. Reward scale v12

| Event | Default |
|---|---:|
| New max-X progress | `0.0075 × dx` |
| Safe grounded progress | `0.002 × dx` |
| Confirmed gap landing | `6 + 0.02 × min(width,120)` |
| Dynamic technique | maksimum `+0.5` |
| Gap-zone inaction | `-0.05/step` |
| General jump start | `-0.03` |
| Hold setelah 6 free steps | `-0.01/step` |
| Non-gap airtime | `-0.005/step` |
| Death | `-6`, ditambah bounded miss penalty bila relevan |
| Coin | `+0.2` |
| Fruit | `+2.0` |

## 6. PPO hyperparameter v12

| Parameter | v11 | v12 | Alasan |
|---|---:|---:|---|
| Learning rate | 0.0003 | 0.0002 | Update lebih konservatif |
| `n_steps` | 1024 | 2048 | Rollout berisi layout/episode lebih beragam |
| Batch size | 64 | 128 | Gradient lebih stabil |
| Entropy coefficient | 0.01 | 0.02 | Menahan policy agar tidak cepat deterministik |
| Epochs | 5 | 5 | Tidak menambah reuse rollout |

Hyperparameter tidak dapat memperbaiki reward yang salah, sehingga perubahan reward dilakukan terlebih dahulu. Entropy lebih tinggi hanya membantu eksplorasi setelah objective diperbaiki.

## 7. Telemetry baru

```text
takeoff_target_distance_total
successful_takeoff_tip_distance_total
successful_takeoff_target_distance_total
gap_inaction_steps
gap_inaction_penalty_total
```

Metrik yang perlu dibandingkan:

- Actual takeoff distance versus dynamic target.
- Successful takeoff distance versus successful target.
- Gap inaction per episode.
- Gap landing rate.
- Grounded percentage.
- Jump-held percentage.
- Entropy loss.

## 8. Per-step trace

Launcher v12 menyimpan `step_trace.csv` secara default setiap 5 environment steps. Ini memberi cukup detail untuk diagnosis tanpa menghasilkan file sebesar logging setiap step.

Lokasi:

```text
logs/ppo/ppo_compact_dynamic_v12_fresh_3k/step_trace.csv
```

Untuk setiap step:

```powershell
.\scripts\run_ppo_compact_dynamic_v12.ps1 -TraceCsvFreq 1
```

Untuk menonaktifkan per-step trace:

```powershell
.\scripts\run_ppo_compact_dynamic_v12.ps1 -TraceCsvFreq 0
```

## 9. Menjalankan training

V11 yang sedang berjalan harus dihentikan terlebih dahulu agar port bridge `8765` bebas. Checkpoint v11 episode 1000 sudah tersedia.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\scripts\run_ppo_compact_dynamic_v12.ps1
```

V12 default dimulai fresh. Jangan mewarisi checkpoint v11 terbaru karena policy tersebut telah memiliki no-jump habit dan entropy rendah.

Dry run:

```powershell
.\scripts\run_ppo_compact_dynamic_v12.ps1 -DryRun
```

Checkpoint tersimpan setiap 250 episode:

```text
logs/ppo/ppo_compact_dynamic_v12_fresh_3k/checkpoints
```

## 10. Evaluasi

Setelah training berhenti:

```powershell
.\scripts\evaluate_ppo_compact_dynamic_v12.ps1 -Episodes 20
```

Evaluator default membandingkan:

- v10 episode 1000;
- v11 episode 500;
- v11 episode 1000;
- seluruh checkpoint v12.

Seluruh model memakai observation dan action space yang sama, sehingga policy dapat dimuat pada environment evaluasi v12. Behavioral metrics seperti length, max-X, gap landing, grounded ratio, dan action ratio lebih penting daripada membandingkan raw reward dari run training yang memakai formula berbeda.

## 11. Checkpoint keputusan awal

Lakukan review pada episode 250 dan 500. V12 dianggap bergerak ke arah benar jika:

- grounded time tidak kembali ke 6-8% seperti v10;
- grounded time tidak terus naik di atas 55% sambil gap landing turun seperti v11;
- gap landing rate naik menuju atau melampaui 20%;
- successful takeoff distance mendekati dynamic target;
- episode length dan max-X tidak mengalami collapse beruntun;
- entropy tidak turun mendekati 0.2 terlalu dini.

Jangan menunggu sampai 3000 jika tiga window berturut-turut menunjukkan episode length, max-X, dan gap landing semuanya menurun.
