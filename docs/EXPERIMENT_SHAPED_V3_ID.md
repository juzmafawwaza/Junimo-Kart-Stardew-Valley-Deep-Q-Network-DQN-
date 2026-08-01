# Catatan Perubahan: PPO Macro 6 + `shaped_v3`

Dokumen ini menjelaskan perubahan yang dibuat setelah hasil PPO macro `shaped_v2` terlihat plateau dari sekitar episode 1.000 sampai 9.000.

Masalah yang terlihat dari log dan video:

1. Episode masih pendek.
   - Banyak run mati dalam beberapa detik pertama.
   - Ini berarti agent belum konsisten melewati obstacle/gap awal.

2. `macro_action_frames=8` kemungkinan terlalu kasar.
   - Satu keputusan agent mengontrol sekitar 8 frame game.
   - Pada 60 FPS, itu sekitar 0,13 detik.
   - Junimo Kart butuh timing lompat yang kecil, jadi 8 frame bisa membuat short/medium/long jump terasa kaku.

3. Reward gap landing terlalu cepat dibayar.
   - Pada `shaped_v2`, agent bisa menerima reward ketika terdeteksi grounded setelah gap.
   - Masalahnya, landing yang tampak sukses belum tentu benar-benar aman; agent bisa langsung mati sesaat setelahnya.

4. Reward score/coin/fruit masih bisa mengganggu survival.
   - Tujuan utama kita adalah menyelesaikan Progress Mode untuk arcade machine.
   - Coin/fruit boleh jadi secondary objective nanti, tetapi untuk fase ini survival lebih penting.

5. `monitor.csv` sebelumnya kurang diagnostik.
   - Kita hanya melihat `reward` dan `length`.
   - Dari dua angka itu, kita belum tahu agent mati karena tidak lompat, terlalu sering lompat, gagal gap, atau kena obstacle.

## File yang diubah

### `junimo_rl/env.py`

Perubahan utama:

1. Menambah `REWARD_VERSIONS`.

```python
REWARD_VERSIONS = {"legacy", "shaped_v1", "shaped_v2", "shaped_v3"}
```

Ini membuat validasi reward lebih eksplisit. Kalau salah mengetik reward version, script langsung error.

2. Menambah telemetry columns.

```python
TELEMETRY_INFO_KEYS = (
    "action_0_count",
    "action_1_count",
    "action_2_count",
    "action_3_count",
    "gap_attempts",
    "gap_landings",
    "gap_failures",
    "gap_deaths",
    "death_near_gap",
    "death_near_obstacle",
    "pickup_events",
    "score_delta_total",
    "max_episode_x",
    "final_gap_start_dx",
    "final_gap_width",
    "final_obstacle_dx",
)
```

Kolom ini akan masuk ke `monitor.csv` untuk training run baru.

Maknanya:

- `action_0_count`: berapa kali agent memilih release/no jump.
- `action_1_count`: berapa kali agent memilih short hold atau hold pada mode binary.
- `action_2_count`: berapa kali agent memilih medium hold.
- `action_3_count`: berapa kali agent memilih long hold.
- `gap_attempts`: berapa kali env mendeteksi agent sedang menghadapi gap valid.
- `gap_landings`: berapa kali agent sukses melewati gap.
- `gap_failures`: berapa kali gap attempt gagal/expired.
- `gap_deaths`: berapa kali agent mati saat gap attempt aktif.
- `death_near_gap`: episode berakhir ketika gap dekat.
- `death_near_obstacle`: episode berakhir ketika obstacle dekat.
- `pickup_events`: berapa kali score naik dalam episode.
- `score_delta_total`: total perubahan score selama episode.
- `max_episode_x`: posisi x terjauh dalam episode.
- `final_gap_start_dx`, `final_gap_width`, `final_obstacle_dx`: kondisi sekitar agent tepat sebelum akhir episode.

3. Menambah `score_reward_coef`.

```python
score_reward_coef: float | None = None
```

Kalau `None`, reward memakai default masing-masing versi.

Untuk `shaped_v3`, default score reward adalah `0.0`, jadi coin/fruit tidak menjadi target utama.

4. Menambah `gap_landing_confirm_steps`.

```python
gap_landing_confirm_steps: int = 2
```

Pada `shaped_v3`, gap landing reward tidak langsung dibayar saat grounded. Agent harus tetap hidup beberapa environment step lagi. Ini mencegah reward palsu dari landing yang langsung berakhir mati.

5. Menambah reward baru `_shaped_v3_reward`.

Formula sederhananya:

```text
reward =
  0.006 * delta_x
+ score_reward_coef * delta_score
+ 150 * level_delta
+ 30 * life_delta
+ 0.035 jika masih hidup dalam gameplay
+ 0.02 jika grounded dan hold jump saat mendekati gap valid
- 0.05 jika gap sangat dekat tapi agent tidak hold jump
+ 0.015 jika grounded dan hold jump saat obstacle dekat
- 0.035 jika hold jump saat grounded padahal tidak ada gap/obstacle dekat
+ confirmed_gap_landing_reward
+ 700 jika completed
- 80 jika game over mulai
- 2 jika keluar minigame
```

Tujuannya bukan membuat rule-based bot. Rule kecil ini hanya menjadi sinyal latihan. Keputusan final tetap dipilih oleh neural network PPO berdasarkan state.

### `scripts/train_ppo.py`

Perubahan:

1. `--reward-version` sekarang menerima `shaped_v3`.
2. Menambah:

```powershell
--score-reward-coef
--gap-landing-confirm-steps
```

3. `Monitor` sekarang menerima `info_keywords=TELEMETRY_INFO_KEYS`, sehingga telemetry masuk ke `monitor.csv`.

### `scripts/train_dqn.py`

Perubahan sama seperti PPO agar DQN bisa menjadi baseline yang fair bila nanti ingin dibandingkan ulang.

### `scripts/evaluate_ppo_models.py` dan `scripts/evaluate_models.py`

Evaluation sekarang menambahkan ringkasan telemetry:

- `mean_gap_attempts`
- `mean_gap_landings`
- `gap_landing_rate`
- `mean_gap_failures`
- `death_near_gap_rate`
- `death_near_obstacle_rate`
- `mean_pickup_events`
- `mean_score_delta_total`
- `mean_max_episode_x`
- `action_0_ratio`
- `action_1_ratio`
- `action_2_ratio`
- `action_3_ratio`

Ini penting untuk paper karena kamu bisa menunjukkan bukan cuma “reward naik/turun”, tetapi juga “perilaku agent berubah seperti apa”.

### `scripts/run_ppo_macro6_v3.ps1`

Launcher eksperimen baru.

Default command-nya setara dengan:

```powershell
python .\scripts\train_ppo.py --episodes 5000 --save-episode-freq 1000 --save-freq 0 --frame-skip 2 --semantic-features --reward-version shaped_v3 --action-mode macro --macro-action-frames 6 --score-reward-coef 0.0 --gap-landing-confirm-steps 2 --learning-rate 0.0003 --ent-coef 0.003 --model-path models\ppo\junimo_ppo_macro6_v3 --run-name ppo_semantic_shaped_v3_macro6_5k
```

### `scripts/summarize_monitor.py`

Helper untuk membaca `monitor.csv` tanpa membuka Excel.

Contoh:

```powershell
python .\scripts\summarize_monitor.py .\logs\ppo\ppo_semantic_shaped_v3_macro6_5k\monitor.csv --window 100
```

Output-nya merangkum:

- mean reward
- mean length
- best length
- action ratio
- gap landing rate
- death-near-gap rate
- death-near-obstacle rate
- max x-position rata-rata

## Cara membaca hasil run baru

Setelah training berjalan, buka `monitor.csv`.

Kalau agent terlalu jarang lompat:

```text
action_0_ratio sangat tinggi
death_near_gap_rate tinggi
gap_landing_rate rendah
```

Kalau agent terlalu sering lompat:

```text
action_1_ratio/action_2_ratio/action_3_ratio tinggi
death_near_obstacle_rate tidak turun
mean_length tetap pendek
```

Kalau gap detection/reward mulai bekerja:

```text
gap_attempts > 0
gap_landings naik
gap_landing_rate naik
mean_max_episode_x naik
```

Kalau coin/fruit masih mendominasi:

```text
mean_pickup_events naik
mean_score_delta_total naik
tetapi mean_length tidak ikut naik
```

Untuk fase survival-first, sinyal bagus adalah:

```text
mean_length naik
mean_max_episode_x naik
gap_landing_rate naik
death_near_gap_rate turun
```

Score bisa dinaikkan lagi nanti setelah survival membaik.
