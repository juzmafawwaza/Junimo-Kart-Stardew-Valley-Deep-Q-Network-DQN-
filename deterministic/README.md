# Deterministic Junimo Kart

Folder ini adalah project sampingan untuk pendekatan deterministik Junimo Kart.
Tujuannya bukan mengganti project RL, tapi membuat baseline yang bisa dihitung dan dibandingkan.

Project ini tetap memakai bridge SMAPI yang sudah ada:

```text
Stardew Valley + SMAPI
  -> JunimoKartRLBridge
  -> Python
  -> calibration CSV / rule-based controller
```

## Kenapa Perlu Folder Ini

RL belajar dari reward dan trial-and-error. Pendekatan deterministik mencoba menjawab pertanyaan yang lebih fisik:

```text
Kalau velocity cart sekian, gap mulai pada dx sekian, gap width sekian,
kapan jump dimulai dan berapa frame jump harus ditahan?
```

Untuk menjawab itu, kita butuh data per-frame, bukan hanya reward episode.

## Struktur

```text
deterministic/
  README.md
  scripts/
    collect_jump_calibration.py
    collect_gap_timing.py
    summarize_jump_calibration.py
    summarize_gap_timing.py
    run_rule_live.py
    summarize_rule_live.py
    run_rule_baseline.py

junimo_det/
  calibration.py
  controller.py
```

Output default masuk ke:

```text
outputs/deterministic/
```

Folder `outputs/` sudah di-ignore oleh Git.

## Setup

Pakai setup Python yang sama dengan project utama:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[train,analysis]"
```

Build dan jalankan mod seperti biasa:

```powershell
dotnet build .\src\JunimoKartRLBridge\JunimoKartRLBridge.csproj -c Release
& "C:\Program Files (x86)\Steam\steamapps\common\Stardew Valley\StardewModdingAPI.exe"
```

Load save sampai masuk farm/world, lalu cek bridge:

```powershell
python .\scripts\smoke_test.py --start --hold 0.2
```

## 1. Ambil Data Jump Kalibrasi

Script ini menjalankan percobaan terkontrol:

```text
hold jump 1 frame
hold jump 2 frame
hold jump 4 frame
...
```

Lalu mencatat posisi, velocity, grounded/jumping, gap, obstacle, dan progress per frame.

```powershell
python .\deterministic\scripts\collect_jump_calibration.py --trials-per-hold 3 --hold-frames "1,2,4,6,8,10,12,16,20,24,30"
```

Output contoh:

```text
outputs/deterministic/jump_calibration_20260730_153000.csv
```

Kolom penting:

```text
hold_frames
frame_index
jump_commanded
player_x
player_y
velocity_x
velocity_y
grounded
jumping
next_gap_start_dx
next_gap_width
landing_delta_y
```

## 2. Ringkas Data Jump

Setelah CSV terkumpul:

```powershell
python .\deterministic\scripts\summarize_jump_calibration.py ".\outputs\deterministic\jump_calibration_*.csv"
```

Ringkasan yang dicari:

```text
hold_frames -> airtime_frames
hold_frames -> horizontal_distance
hold_frames -> peak_height
hold_frames -> avg_velocity_x_per_frame
```

Itu bahan awal untuk menurunkan rumus atau lookup table.

## 3. Ambil Data Timing Gap

Setelah tahu bentuk dasar lompatan, langkah berikutnya adalah mencari kapan jump harus dimulai sebelum gap.

Script ini mencoba grid:

```text
trigger_dx = jarak gap saat jump mulai
hold_frames = berapa frame jump ditahan
```

Contoh awal yang tidak terlalu besar:

```powershell
python .\deterministic\scripts\collect_gap_timing.py --trigger-dx "10,20,30,40,50,60,80" --hold-frames "8,10,12,16" --trials-per-combo 2
```

Output contoh:

```text
outputs/deterministic/gap_timing_20260730_171500.csv
```

Kolom penting:

```text
trigger_dx
hold_frames
trigger_gap_start_dx
trigger_gap_width
trigger_landing_delta_y
crossed_gap
landed_after_gap
survived_until_margin
game_over
reason
```

Kalau perlu membedah frame-by-frame untuk trial tertentu, aktifkan trace:

```powershell
python .\deterministic\scripts\collect_gap_timing.py --trigger-dx "10,20,30,40" --hold-frames "8,10,12" --trials-per-combo 1 --trace
```

`survived_until_margin = 1` adalah sinyal paling berguna untuk rule awal: cart melewati gap, mendarat, dan masih hidup beberapa frame setelahnya.

Setelah CSV terkumpul, ranking kombinasi terbaik:

```powershell
python .\deterministic\scripts\summarize_gap_timing.py ".\outputs\deterministic\gap_timing_*.csv" --top 10 --out outputs\deterministic\gap_timing_summary.csv
```

Kolom paling penting:

```text
survived_rate
landed_rate
gameover_rate
mean_final_player_x
mean_required_jump_distance
mean_final_shortfall
```

## 4. Jalankan Rule-Based Baseline

Untuk kontrol deterministik yang lebih presisi, pakai runner per-frame. Ini membaca state dan mengirim hold/release jump tiap frame, sehingga cocok untuk trigger kecil seperti 5-25px sebelum gap.

```powershell
python .\deterministic\scripts\run_rule_live.py --episodes 20 --out outputs\deterministic\rule_live.csv
```

Baseline Gym macro lama masih tersedia untuk pembanding, tapi lebih kasar karena setiap action bisa menahan kontrol beberapa frame sebelum observasi berikutnya.

```powershell
python .\deterministic\scripts\run_rule_baseline.py --episodes 20 --action-mode macro --macro-action-frames 16 --gap-trigger-dx 25 --out outputs\deterministic\rule_baseline.csv
```

Rule awal:

```text
kalau grounded dan gap dekat:
  pilih short/medium/long hold berdasarkan gap width dan landing delta

kalau grounded dan obstacle dekat:
  pilih jump

selain itu:
  release
```

## Catatan Penting

Data lama dari training RL belum cukup untuk rumus deterministik karena lognya episode-level.
Untuk formula jump, kita perlu trace per-frame seperti yang dikumpulkan script kalibrasi ini.

Manual play bisa berguna nanti untuk imitation learning, tapi untuk rumus dasar jump lebih bersih memakai automated calibration.
