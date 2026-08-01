# Eksperimen compact anchored-gap v9

Dokumen ini menjelaskan perubahan deteksi jurang setelah eksperimen `shaped_v6` berhenti membaik. Perubahan ini tidak menambah jumlah fitur observasi: PPO tetap menerima 27 angka ter-normalisasi dan action tetap binary `release/hold`.

## Masalah pada detektor lama

Mode `legacy` mengurutkan seluruh objek rel berdasarkan koordinat X, lalu memeriksa setiap pasangan yang bersebelahan:

```text
track[0] -> track[1]
track[1] -> track[2]
...
```

Jika selisih X lebih besar dari 36 piksel, pasangan tersebut dianggap jurang. Ini bekerja pada jalur sederhana, tetapi ambigu pada layout bertingkat. Rel di atas dan rel jalur utama dapat bersebelahan dalam urutan X walaupun cart tidak sedang berada pada rel tersebut. Akibatnya:

- `gap_dx` dapat menunjuk jurang milik cabang lain;
- `gap_width` dapat berasal dari dua rel yang bukan pasangan takeoff–landing;
- lompatan yang sebenarnya dilakukan di rel lurus dapat tercatat sebagai `near_gap`;
- target landing dapat berubah atau hilang ketika cart mulai airborne;
- reward landing dan penalti miss dapat memakai geometri yang keliru.

Mode lama tetap tersedia melalui:

```text
--gap-detection-mode legacy
```

Hal ini penting agar eksperimen lama masih dapat direproduksi.

## Algoritma baru: `anchored`

Mode baru diaktifkan dengan:

```text
--gap-detection-mode anchored
```

Alurnya adalah sebagai berikut.

### 1. Menentukan supporting track

Detektor hanya memilih rel asal ketika bridge menyatakan cart `grounded`. Kandidat rel harus menutupi posisi horizontal cart, dengan toleransi 10 piksel. Jika beberapa rel bertumpuk pada X yang sama, rel dengan selisih vertikal terkecil terhadap cart dipilih.

Dengan demikian, rel tidak lagi dipilih hanya karena posisinya muncul lebih awal dalam daftar.

### 2. Mengikuti rangkaian rel yang tersambung

Dari supporting track, algoritma bergerak maju ke potongan rel berikutnya selama:

```text
horizontal_gap <= 36 px
abs(vertical_delta) <= 48 px
```

Batas vertikal tetap mengizinkan slope dan perubahan ketinggian kecil, tetapi mencegah algoritma berpindah ke platform atas yang tidak tersambung. Ujung seluruh rangkaian ini menjadi `current_track_end_dx`, bukan lagi ujung satu potongan rel tepat di bawah cart.

### 3. Memilih landing pertama yang masuk akal

Setelah ujung rangkaian asal ditemukan, kandidat landing harus:

- berada lebih dari 36 piksel di depan ujung takeoff;
- mempunyai perbedaan tinggi maksimum 112 piksel;
- merupakan kandidat paling dekat secara horizontal.

Rangkaian rel landing juga ditelusuri sampai ujung. Karena itu interval landing sekarang benar-benar memiliki awal dan akhir:

```text
landing interval = [gap_end_x, landing_end_x]
```

Ini memperbaiki perhitungan penalti miss yang sebelumnya sering hanya mempunyai satu titik landing karena `track.bounds` Junimo Kart tidak tersedia.

### 4. Menyimpan geometri selama airborne

Saat masih grounded, koordinat relatif diubah menjadi koordinat dunia absolut:

```text
gap_start_x
gap_end_x
landing_end_x
landing_y
takeoff_y
```

Selama airborne, pasangan yang sama dikonversi kembali relatif terhadap posisi cart saat ini:

```text
gap_start_dx(t) = gap_start_x - player_x(t)
gap_end_dx(t) = gap_end_x - player_x(t)
```

Akibatnya, target tidak berubah hanya karena daftar `tracksAhead` berubah ketika cart bergerak. Nilai `gap_start_dx` boleh menjadi negatif setelah cart melewati ujung takeoff; ini justru memberi tahu model seberapa jauh cart telah bergerak di atas jurang.

Tracker juga mempertahankan target pada satu frame transisi ketika game masih melaporkan `grounded=True`, tetapi tidak ada supporting track. Ini menangani keterlambatan flag grounded tepat setelah cart meninggalkan ujung rel.

## Hubungan dengan reward `shaped_v6`

Rumus reward tetap sama:

```text
r_v6 = r_v5
       - jump_start_penalty
       - gap_miss_penalty_coef * miss_ratio^2  (hanya saat gap death)
```

Launcher v9 memakai:

```text
jump_start_penalty = 0.05
gap_miss_penalty_coef = 2.0
```

Nilai jump-start dinaikkan dari 0.02 menjadi 0.05 karena evaluasi v6 menunjukkan spam jump belum turun. Nilainya masih kecil dibanding reward landing sekitar 5–6.8 dan death penalty dasar -5, sehingga lompatan yang memang menyelamatkan cart tetap layak dilakukan.

Untuk ablation yang hanya menguji perubahan detektor tanpa perubahan bobot anti-spam:

```powershell
.\scripts\run_ppo_compact_anchored_v9.ps1 -JumpStartPenalty 0.02
```

## Kompatibilitas model

- Dimensi compact observation tetap 27.
- Action tetap binary: action 0 melepas jump, action 1 menahan jump.
- Bridge DLL tidak berubah, sehingga tidak perlu menjalankan `install_bridge_v8.ps1` lagi.
- Model episode 1.500 dapat dimuat karena bentuk neural network sama.
- Walaupun bentuk input sama, arti beberapa fitur gap menjadi lebih akurat. Karena itu hasil v9 harus disimpan di folder baru dan tidak dicampur dengan log v5/v6.

## Menjalankan training

Pastikan Stardew Valley dibuka melalui SMAPI, save sudah loaded, dan Junimo Kart dapat dimulai oleh bridge. Lalu jalankan:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\scripts\run_ppo_compact_anchored_v9.ps1
```

Default launcher:

- memuat checkpoint v5 episode 1.500;
- melatih 1.500 episode tambahan;
- memberi nomor episode 1.500 sampai 3.000;
- menyimpan checkpoint setiap 250 episode;
- menulis log ke `logs/ppo/ppo_compact_anchored_v9_from_1500_to_3k`;
- menyimpan final model ke `models/ppo/junimo_ppo_compact_anchored_v9_3k.zip`.

Untuk melihat command tanpa menjalankan training:

```powershell
.\scripts\run_ppo_compact_anchored_v9.ps1 -DryRun
```

## Evaluasi

Setelah checkpoint tersedia:

```powershell
.\scripts\evaluate_ppo_compact_anchored_v9.ps1 -Episodes 20
```

Evaluator wajib memakai `--gap-detection-mode anchored`, karena observasinya harus sama dengan saat training.

Secara default launcher mengevaluasi checkpoint sumber episode 1.500 **dan** seluruh checkpoint v9. Baseline tersebut juga dijalankan dengan detector `anchored` dan skala reward v9. Ini memisahkan perubahan perilaku yang langsung terjadi akibat representasi input baru dari improvement yang benar-benar muncul setelah gradient update v9.

Metrik utama yang perlu dibandingkan dengan baseline episode 1.500:

- `completion_rate` dan `max_levels_beat`;
- `mean_max_episode_x` dan `mean_length`;
- `gap_landing_rate`;
- `mean_gap_miss_distance`;
- `jump_start_without_near_gap_rate`;
- `mean_jump_starts`;
- `jump_hold_ratio`.

Perubahan dianggap menjanjikan bila gap landing dan jarak maksimum naik, sedangkan miss distance dan jump tanpa jurang turun. Mean reward saja tidak cukup karena skala penalti jump v9 berbeda dari baseline.

## Pengujian otomatis

`tests/test_env_v8.py` sekarang menguji:

1. platform atas yang tidak reachable tidak dipilih sebagai pasangan jurang;
2. `current_track_end_dx` berasal dari ujung rangkaian rel, bukan satu tile;
3. interval landing mencakup seluruh rangkaian landing;
4. geometri absolut tetap konsisten saat airborne;
5. satu frame `grounded` tanpa supporting track tidak menghapus target;
6. detector tidak mengarang supporting track ketika cart airborne tanpa riwayat.

Jalankan dengan:

```powershell
python -m unittest discover -s tests -v
```
