# Stardew Valley Junimo Kart RL Bridge

Eksperimen ini memakai opsi “proper”: sebuah SMAPI mod membaca state internal Junimo Kart dan membuka local TCP bridge untuk Python RL agent.

Target awalnya bukan langsung “agent jago”, tapi pipeline yang benar:

1. Stardew Valley + SMAPI menjalankan mod.
2. Python bisa start Junimo Kart Progress Mode.
3. Python menerima observation internal, bukan screen capture.
4. Python mengirim action `release jump` / `hold jump`.
5. Training RL bisa dilakukan di atas environment Gymnasium.

## Struktur

- `src/JunimoKartRLBridge/` — SMAPI mod C#.
- `junimo_rl/` — Python TCP client + Gymnasium environment.
- `scripts/smoke_test.py` — cek koneksi bridge dan start minigame.
- `scripts/train_dqn.py` — contoh training Stable-Baselines3 DQN.

## Setup cepat

Pastikan:

- Stardew Valley sudah terinstall.
- SMAPI sudah terinstall dan game bisa dibuka lewat `StardewModdingAPI.exe`.
- .NET 6 SDK tersedia.
- Python 3.10+ tersedia.

Build mod:

```powershell
dotnet build .\src\JunimoKartRLBridge\JunimoKartRLBridge.csproj -c Release
```

`Pathoschild.Stardew.ModBuildConfig` biasanya otomatis deploy hasil build ke folder `Stardew Valley\Mods`. Kalau tidak, copy folder hasil build dari `src\JunimoKartRLBridge\bin\Release\net6.0\JunimoKartRLBridge` ke:

```text
C:\Program Files (x86)\Steam\steamapps\common\Stardew Valley\Mods\JunimoKartRLBridge
```

Install dependency Python:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[train]
```

Jalankan game lewat SMAPI, load save, lalu di terminal project:

```powershell
python .\scripts\smoke_test.py --start
```

Kalau berhasil, script akan menerima JSON state dari Junimo Kart.

## Training

Dengan game tetap berjalan dan save sudah loaded:

```powershell
python .\scripts\train_dqn.py --timesteps 100000 --model-path models\junimo_dqn
```

Setiap run training menyimpan:

- `logs\<run-name>\monitor.csv` — reward dan panjang tiap episode.
- `logs\<run-name>\hparams.txt` — hyperparameter run.
- `logs\<run-name>\checkpoints\` — checkpoint model berkala.
- `logs\<run-name>\tensorboard\` — log TensorBoard.

Plot cepat dari CSV:

```powershell
pip install -e .[analysis]
python .\scripts\plot_training.py .\logs\<run-name>\monitor.csv
```

TensorBoard:

```powershell
tensorboard --logdir .\logs
```

Save model setiap N episode:

```powershell
python .\scripts\train_dqn.py --episodes 1000 --save-episode-freq 100 --model-path models\junimo_dqn --run-name ep_compare_01
```

Checkpoint akan muncul di `logs\ep_compare_01\checkpoints\` dengan nama seperti:

```text
junimo_dqn_ep000100_steps12345.zip
junimo_dqn_ep000200_steps23456.zip
```

Lanjut dari model yang sudah pernah dilatih 1000 episode, lalu simpan checkpoint kumulatif 2000, 3000, dst:

```powershell
python .\scripts\train_dqn.py --load-model models\junimo_dqn.zip --episodes 9000 --episode-offset 1000 --save-episode-freq 1000 --frame-skip 2 --model-path models\junimo_dqn --run-name continue_to_10k
```

Bandingkan beberapa checkpoint:

```powershell
python .\scripts\evaluate_models.py .\logs\ep_compare_01\checkpoints\junimo_dqn_ep000100_*.zip .\logs\ep_compare_01\checkpoints\junimo_dqn_ep001000_*.zip --episodes 20 --out logs\ep_compare_01\evaluation.csv
```

Catatan: training real-time di game asli akan lambat karena game tetap berjalan sekitar 60 FPS. Ini fondasi yang bersih; kalau nanti mau training cepat, langkah berikutnya adalah menambah mode “accelerated simulation” di mod.

## Protocol singkat

Bridge listen di `127.0.0.1:8765` dan memakai JSON-lines.

Request:

```json
{"type":"ping"}
{"type":"state"}
{"type":"start","mode":"progress"}
{"type":"action","jump":true}
{"type":"action","jump":false}
```

Response selalu:

```json
{"ok":true,"type":"state","message":null,"state":{...}}
```

## Catatan etika/achievement

SMAPI sendiri kompatibel dengan Steam achievements, tapi ini tetap automation/modding. Pakai di single-player save pribadi saja. Kalau tujuanmu hanya mendapat arcade machine, ini tidak perlu menyentuh leaderboard Endless Mode.
