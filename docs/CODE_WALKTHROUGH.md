# Code Walkthrough — Junimo Kart RL Bridge

Dokumen ini menjelaskan code yang dibuat untuk project Junimo Kart Reinforcement Learning. Tujuannya supaya project ini tidak terasa seperti “vibe code”: setiap file punya alasan, alur data jelas, dan kamu bisa membaca ulang untuk memahami apa yang sebenarnya terjadi.

Ke depan, setiap kali Codex menambah atau mengubah code di project ini, dokumen ini perlu ikut di-update dengan:

1. file apa yang berubah,
2. fungsi perubahan itu,
3. alur data/logika yang terpengaruh,
4. cara menjalankan atau mengetesnya,
5. caveat atau risiko yang perlu kamu tahu.

## Gambaran besar

Project ini terdiri dari dua sisi:

1. SMAPI mod C# di dalam Stardew Valley.
2. Python RL environment di luar game.

Alurnya:

```text
Python DQN agent
  -> pilih action: release jump / hold jump
  -> kirim JSON ke TCP bridge localhost
  -> SMAPI mod menerima action
  -> mod mengubah state jump internal Junimo Kart
  -> game berjalan beberapa frame
  -> mod membaca state internal Junimo Kart
  -> Python menerima observation + hitung reward
  -> DQN belajar dari pengalaman itu
```

Bridge memakai TCP lokal:

```text
127.0.0.1:8765
```

Format komunikasinya JSON-lines: satu JSON per baris.

## Folder penting

```text
src/JunimoKartRLBridge/   SMAPI mod C#
junimo_rl/                Python client + Gymnasium environment
scripts/                  script smoke test, training, plotting, evaluation
docs/                     dokumentasi detail code
logs/                     output training
models/                   final model
```

Folder/file generated seperti `logs/`, `models/`, `bin/`, `obj/`, `.scratch/`, `__pycache__/`, dan `*.egg-info/` di-ignore oleh Git. Yang dipush ke GitHub adalah source code, script, config, dan dokumentasi; bukan model hasil training atau build artifact.

## SMAPI mod C#

### `src/JunimoKartRLBridge/manifest.json`

Manifest SMAPI. Ini file yang membuat SMAPI mengenali mod.

Isi penting:

- `Name`: nama mod.
- `UniqueID`: ID unik mod.
- `EntryDll`: DLL yang dijalankan SMAPI.
- `MinimumApiVersion`: versi minimal SMAPI.

Kalau file ini salah, SMAPI tidak akan load mod.

### `src/JunimoKartRLBridge/JunimoKartRLBridge.csproj`

Project file .NET untuk build mod.

Bagian penting:

- target framework: `net6.0`, karena Stardew Valley 1.6 berjalan di .NET 6.
- package `Pathoschild.Stardew.ModBuildConfig`, supaya build mod bisa otomatis deploy ke folder `Stardew Valley/Mods`.
- `GamePath`, diarahkan ke instalasi Stardew lokal.

Command build:

```powershell
dotnet build .\src\JunimoKartRLBridge\JunimoKartRLBridge.csproj -c Release
```

Kalau Stardew sedang terbuka, deploy bisa gagal karena DLL sedang dikunci. Tutup game dulu.

### `src/JunimoKartRLBridge/Config.cs`

Config mod yang bisa disimpan oleh SMAPI.

Field penting:

- `BindAddress`: alamat TCP server, default `127.0.0.1`.
- `Port`: port bridge, default `8765`.
- `StartServerOnLaunch`: apakah bridge langsung hidup saat mod load.
- `MaxTracks`: jumlah track depan yang dikirim ke Python.
- `MaxEntities`: jumlah entity depan yang dikirim ke Python.
- `LookaheadPixels`: jarak depan yang diamati agent.
- `LookbehindPixels`: sedikit area belakang yang masih dikirim.
- `AutoAdvanceTitleAfterStart`: auto-skip layar Title Junimo Kart setelah start.
- `AutoContinueProgressModeNonGameplayStates`: auto-continue state non-gameplay seperti Title/Map/Cutscene.
- `ForceRunWhenUnfocused`: memaksa opsi Stardew `pauseWhenOutOfFocus` menjadi `false`, supaya game tetap berjalan saat window tidak aktif.

Tujuan config ini: membuat behavior bridge bisa diubah tanpa hardcode ulang.

### `src/JunimoKartRLBridge/BridgeServer.cs`

TCP server lokal yang menerima request dari Python.

Tanggung jawab:

- listen di `127.0.0.1:8765`,
- menerima koneksi Python,
- membaca JSON line-by-line,
- parse request,
- panggil `ModEntry.HandleBridgeRequest`,
- kirim response JSON.

Bridge ini berjalan di background thread. Karena Stardew game state tidak aman dimutasi dari thread TCP langsung, perubahan seperti start/reset/jump disimpan sebagai pending state, lalu diproses pada main update tick Stardew di `ModEntry`.

### `src/JunimoKartRLBridge/Protocol.cs`

Berisi class data untuk request/response JSON.

Class penting:

- `ClientRequest`
  - `Type`: `ping`, `state`, `start`, `reset`, `action`, `advance`.
  - `Mode`: `progress` atau `endless`.
  - `Jump`: `true` untuk tahan lompat, `false` untuk lepas.

- `BridgeResponse`
  - `Ok`: request berhasil atau tidak.
  - `Type`: jenis response.
  - `Message`: pesan opsional.
  - `State`: snapshot game.

- `BridgeSnapshot`
  - semua state Junimo Kart yang dikirim ke Python.

- `PlayerSnapshot`
  - posisi, velocity, grounded/jumping, bounds player.

- `TrackSnapshot`
  - posisi track depan, tipe track, obstacle di track.

- `EntitySnapshot`
  - coin, fruit, obstacle, dekor, dan entity lain di depan kart.

### `src/JunimoKartRLBridge/ReflectionUtil.cs`

Helper untuk membaca/menulis field private Stardew Valley.

Kenapa perlu reflection?

Junimo Kart menyimpan banyak state penting sebagai private field, misalnya:

- `player`
- `_tracks`
- `_entities`
- `isJumpPressed`
- `gameState`
- `levelsBeat`

SMAPI mod tidak bisa mengakses field private langsung dengan normal C#. `ReflectionUtil` mencari field/method tersebut via reflection dan cache hasil pencariannya supaya tidak lambat setiap frame.

Fungsi penting:

- `Field(target, name)`: baca field private/public.
- `Field<T>(target, name)`: baca field dan convert ke tipe tertentu.
- `SetField(target, name, value)`: tulis field private/public.
- `Invoke(target, name, args)`: panggil method private/public.
- `BoolMethod(target, name)`: panggil method boolean.
- `VectorField(target, name)`: baca `Vector2`.
- `Bounds(target)`: panggil `GetBounds()` dan convert ke DTO.
- `Enumerate(value)`: flatten list/dictionary internal Stardew.
- `InheritsTypeName(value, typeName)`: cek class inheritance berdasarkan nama.

Bagian paling penting untuk bug jump:

```text
SetField(mineCart, "isJumpPressed", desiredJump)
Invoke(player, "QueueJump")
Invoke(player, "ReleaseJump")
```

Awalnya bridge mencoba `receiveKeyPress(Keys.Space)`, tapi Junimo Kart tidak memakai method itu untuk jump. Fix-nya adalah mengubah state jump internal langsung.

### `src/JunimoKartRLBridge/ModEntry.cs`

File utama mod.

Tanggung jawab:

1. load config,
2. start TCP bridge,
3. handle request dari Python,
4. start Junimo Kart Progress Mode,
5. auto-continue Title/Map/Cutscene,
6. memastikan game tidak pause saat window tidak fokus,
7. apply action jump,
8. buat snapshot observation.

#### Entry point

`Entry(IModHelper helper)` dipanggil SMAPI saat mod load.

Yang dilakukan:

- baca config,
- register event `GameLoop.UpdateTicked`,
- register console command `jkrl_start`,
- register console command `jkrl_release`,
- start `BridgeServer`.

#### Request handling

`HandleBridgeRequest(ClientRequest? request)` menerima request dari TCP server.

Request penting:

- `ping`: cek bridge hidup.
- `state`: ambil snapshot terakhir.
- `start` / `reset`: minta Junimo Kart Progress Mode dimulai ulang.
- `action`: set target jump held/released.
- `advance`: minta mod lanjut dari state non-gameplay ke gameplay.

#### Kenapa ada pending state?

TCP server berjalan di thread terpisah. Stardew game state harus disentuh dari game update thread. Jadi request tidak langsung mengubah game, tapi menulis:

- `pendingStartMode`
- `pendingAdvance`
- `desiredJumpHeld`

Lalu `OnUpdateTicked` membaca pending state dan menerapkannya secara aman.

#### Start Junimo Kart

`StartMineCart(string mode)` membuat instance minigame:

```csharp
Game1.currentMinigame = new MineCart(0, modeId);
```

Mode:

- `2`: Progress Mode.
- `3`: Infinite/Endless Mode.

Untuk target arcade machine, yang dipakai adalah Progress Mode.

#### Run while unfocused

`EnsureRunsWhenUnfocused()` mematikan opsi internal Stardew:

```csharp
Game1.options.pauseWhenOutOfFocus = false;
```

Tanpa ini, game bisa pause/freeze ketika kamu klik aplikasi lain. Training RL butuh game loop tetap berjalan, jadi bridge memaksa opsi ini off selama mod aktif. Ini tidak selalu menjamin game tetap berjalan kalau window diminimize total, karena OS/renderer masih bisa melakukan throttling. Untuk training stabil, lebih aman gunakan Windowed/Borderless dan biarkan window Stardew tetap terbuka walau tidak aktif.

#### Auto-continue

`AutoContinueNonGameplayStates` dan `ForceProgressModeGameplay` memastikan training tidak nyangkut di layar Title/Map/Cutscene.

State internal Junimo Kart:

```text
0 = Title
1 = Ingame
2 = FruitsSummary
3 = Map
4 = Cutscene
```

Logic:

- kalau Title, panggil `restartLevel(true)`;
- kalau Map, panggil `ShowCutscene()`;
- kalau Cutscene, panggil `EndCutscene()`;
- kalau Ingame, biarkan agent main.

#### Apply jump

`ApplyJump(MineCart mineCart, bool desiredJump)` adalah fungsi yang mengubah action Python menjadi behavior game.

Logic:

- set `isJumpPressed` sesuai action,
- kalau baru mulai hold jump, panggil `QueueJump()`,
- kalau baru release jump, panggil `ReleaseJump()`.

Action space Python:

```text
0 = release jump
1 = hold jump
```

#### Snapshot observation

`CreateSnapshot` membaca internal state:

- score,
- lives,
- level,
- game state,
- current theme,
- player position,
- player velocity,
- grounded/jumping,
- track depan,
- entity depan seperti coin/fruit/obstacle.

Track dan entity difilter agar hanya yang dekat dengan player dikirim ke Python. Ini menjaga observation tetap kecil dan relevan.

## Python package

### `junimo_rl/client.py`

Client TCP sederhana untuk bicara dengan mod.

Method penting:

- `connect()`: buka koneksi TCP.
- `request(payload)`: kirim JSON dan baca response.
- `ping()`: cek bridge hidup.
- `state()`: ambil state terakhir.
- `start(mode="progress")`: start Progress Mode.
- `advance()`: paksa lanjut dari non-gameplay state.
- `action(jump: bool)`: kirim action jump.

Semua request dikirim sebagai satu baris JSON.

Contoh:

```json
{"type":"action","jump":true}
```

Client juga punya retry ringan: kalau koneksi TCP putus saat request, client menutup socket lama, reconnect, lalu retry request sekali. Ini membantu saat bridge sempat reset. Kalau Stardew/SMAPI memang sudah ditutup, error dibuat lebih jelas: buka Stardew lewat SMAPI dan load save dulu.

### `junimo_rl/env.py`

Gymnasium environment untuk RL.

Stable-Baselines3 butuh environment dengan interface:

- `reset()`
- `step(action)`
- `observation_space`
- `action_space`

#### Observation vector

Game state dari JSON tidak langsung bisa dipakai neural network. Maka `snapshot_to_vector` mengubahnya menjadi array angka `np.float32`.

Isi vector:

1. base features:
   - inMinigame,
   - score,
   - lives,
   - levelsBeat,
   - gameMode,
   - gameState,
   - gameOver,
   - completed,
   - jumpHeld,
   - player position,
   - player velocity,
   - grounded/jumping.

2. track features:
   - jarak track dari player (`dx`),
   - posisi y,
   - tipe track,
   - ada obstacle atau tidak,
   - tipe obstacle.

3. entity features:
   - jarak entity,
   - posisi y,
   - ukuran bounds,
   - tipe entity,
   - obstacle/pickup flag.

Jumlah track/entity dibatasi:

```text
MAX_TRACKS = 24
MAX_ENTITIES = 24
```

Kalau jumlah aktual kurang, sisanya diisi nol. Ini penting karena neural network perlu input size tetap.

#### Action space

```python
spaces.Discrete(2)
```

Artinya hanya ada dua action:

```text
0 = release jump
1 = hold jump
```

#### Reset

`reset()`:

1. kirim `start progress`,
2. tunggu sampai game benar-benar `Ingame`,
3. kalau masih Title/Map/Cutscene, kirim `advance`,
4. return observation awal.

#### Step

`step(action)`:

1. ambil snapshot lama,
2. kirim action jump,
3. tunggu beberapa frame sesuai `frame_skip`,
4. ambil snapshot baru,
5. hitung reward,
6. return `(obs, reward, terminated, truncated, info)`.

#### Reward logic

Reward dihitung di `_reward`.

Komponen:

- maju ke kanan (`dx`): reward kecil positif,
- score naik: reward positif,
- level selesai: reward besar,
- life berkurang: reward negatif,
- Progress Mode selesai: reward sangat besar,
- game over: penalti besar.

Tujuannya supaya agent belajar:

- bergerak sejauh mungkin,
- jangan mati,
- ambil score/coin kalau mungkin,
- selesaikan level,
- akhirnya complete Progress Mode.

## Scripts

### `scripts/smoke_test.py`

Script sanity check.

Gunanya:

- cek bridge bisa di-ping,
- optional start Junimo Kart,
- optional hold jump untuk test apakah kart benar-benar lompat,
- print JSON state.

Contoh:

```powershell
python .\scripts\smoke_test.py --start --hold 0.4
```

Kalau jump berhasil, di output `duringHold` harus terlihat `jumpHeld: true`, dan biasanya `player.jumping` atau `velocity.y` berubah.

### `scripts/train_dqn.py`

Script training DQN.

Fitur:

- training berdasarkan timestep atau episode,
- save final model,
- save monitor CSV,
- save TensorBoard logs,
- save checkpoint per timestep,
- save checkpoint per episode,
- bisa continue dari model lama.

Command contoh:

```powershell
python .\scripts\train_dqn.py --episodes 1000 --save-episode-freq 100 --model-path models\junimo_dqn --run-name ep_compare_01
```

Output:

```text
logs/ep_compare_01/monitor.csv
logs/ep_compare_01/hparams.txt
logs/ep_compare_01/checkpoints/
logs/ep_compare_01/tensorboard/
models/junimo_dqn.zip
```

#### `EpisodeCheckpointCallback`

Callback custom untuk save model setiap N episode.

Kenapa dibuat?

Stable-Baselines3 default checkpoint berdasarkan timestep. Kamu ingin compare “model episode 100 vs 1000 vs 10000”, jadi checkpoint episode lebih cocok.

Nama checkpoint:

```text
junimo_dqn_ep000100_steps12345.zip
junimo_dqn_ep001000_steps99999.zip
```

### `scripts/plot_training.py`

Membuat line chart dari `monitor.csv`.

Chart yang dibuat:

- episode reward,
- rolling mean reward,
- episode length,
- rolling mean length.

Command:

```powershell
python .\scripts\plot_training.py .\logs\ep_compare_01\monitor.csv
```

Output:

```text
logs/ep_compare_01/training_plot.png
```

### `scripts/evaluate_models.py`

Membandingkan beberapa checkpoint secara deterministic.

Training memakai exploration/random. Evaluation sebaiknya deterministic supaya model dibandingkan lebih adil.

Command:

```powershell
python .\scripts\evaluate_models.py ".\logs\ep_compare_01\checkpoints\junimo_dqn_ep*.zip" --episodes 20 --out logs\ep_compare_01\evaluation.csv
```

Output CSV berisi:

- model path,
- jumlah episode evaluasi,
- mean reward,
- mean episode length,
- completion rate,
- max levels beat.

Evaluator tetap membutuhkan Stardew + SMAPI + bridge hidup, sama seperti training. Kalau game ditutup di tengah evaluasi, script akan berhenti dengan pesan singkat yang meminta kamu membuka Stardew via SMAPI dan load save lagi.

## Troubleshooting umum

### `ConnectionResetError [WinError 10054]`

Artinya koneksi Python ke bridge SMAPI diputus oleh sisi game/mod. Penyebab paling umum:

1. Stardew/SMAPI ditutup.
2. Game kembali ke title screen atau save belum loaded.
3. Bridge belum listen di `127.0.0.1:8765`.
4. Mod belum ter-load setelah build/restart.

Cek cepat:

```powershell
Get-Process | Where-Object { $_.ProcessName -like '*Stardew*' }
Test-NetConnection -ComputerName 127.0.0.1 -Port 8765
```

Kalau `TcpTestSucceeded` false, buka ulang Stardew lewat SMAPI, load save, lalu ulang command Python.

## Cara DQN belajar

DQN mencoba mempelajari fungsi:

```text
Q(state, action)
```

Maknanya:

```text
Seberapa bagus action tertentu jika dilakukan dari state sekarang?
```

Contoh:

```text
State:
- kart grounded,
- velocity X stabil,
- gap 200 pixel di depan,
- track berikutnya lebih tinggi.

Action:
- release jump
- hold jump
```

DQN mencoba menebak action mana yang menghasilkan reward masa depan lebih tinggi.

Awalnya neural network belum tahu apa-apa, jadi agent banyak random. Ini disebut exploration.

Setiap step:

1. agent melihat observation,
2. agent memilih action,
3. game merespons,
4. environment menghitung reward,
5. pengalaman disimpan ke replay buffer,
6. DQN sampling pengalaman lama,
7. neural network diupdate agar prediksi Q lebih dekat dengan reward aktual + prediksi masa depan.

Secara intuisi:

```text
Kalau action A sering bikin mati, Q(A) turun.
Kalau action B sering bikin bertahan lebih lama / score naik / level selesai, Q(B) naik.
```

## Hyperparameter penting

### `--episodes`

Target jumlah episode.

Contoh:

```powershell
--episodes 1000
```

Cocok kalau kamu ingin compare model berdasarkan episode.

Jika menjalankan ulang training dari model lama, `--episodes` berarti jumlah episode tambahan di run baru, bukan total kumulatif.

### `--episode-offset`

Episode awal untuk penamaan checkpoint saat melanjutkan training.

Contoh: kamu sudah punya model 1000 episode, lalu ingin lanjut 9000 episode lagi dan checkpoint berikutnya dinamai 2000, 3000, dst:

```powershell
python .\scripts\train_dqn.py --load-model models\junimo_dqn.zip --episodes 9000 --episode-offset 1000 --save-episode-freq 1000 --model-path models\junimo_dqn --run-name continue_to_10k
```

Dengan `--episode-offset 1000`, callback checkpoint mulai menghitung dari 1000, sehingga setelah 1000 episode tambahan file pertama bernama sekitar:

```text
junimo_dqn_ep002000_steps....zip
```

### `--timesteps`

Target jumlah step RL.

Kalau `--episodes` dipakai, script stop berdasarkan episode dan timesteps dibuat sangat besar secara internal.

### `--save-episode-freq`

Save model setiap N episode.

Contoh:

```powershell
--save-episode-freq 100
```

### `--save-freq`

Save model setiap N timestep.

Ini checkpoint tambahan berbasis step.

### `--frame-skip`

Agent memilih action setiap N frame.

- lebih kecil: kontrol lebih presisi, training lebih lambat;
- lebih besar: training lebih cepat, tapi timing lompat lebih kasar.

Default:

```text
4
```

Untuk Junimo Kart, `2` mungkin lebih responsif.

### `--learning-rate`

Seberapa besar update neural network setiap training step.

Default:

```text
1e-4
```

Kalau training tidak stabil, coba `5e-5`.

### `--exploration-fraction`

Porsi training saat agent masih menurunkan random action dari tinggi ke rendah.

Default:

```text
0.25
```

Kalau agent terlalu cepat “percaya diri” padahal belum jago, naikkan ke `0.5`.

### `--exploration-final-eps`

Random action minimum setelah exploration turun.

Default:

```text
0.05
```

Untuk level procedural/bervariasi, `0.1` bisa dicoba.

### `--buffer-size`

Jumlah pengalaman yang disimpan replay buffer.

Default:

```text
50000
```

Lebih besar berarti agent bisa belajar dari sejarah lebih panjang, tapi memakai lebih banyak memori.

### `--learning-starts`

Jumlah timestep awal sebelum DQN mulai update network.

Default:

```text
2000
```

Tujuannya agar replay buffer terisi pengalaman dulu.

## Cara run yang direkomendasikan

Setelah patch jump aktif dan Stardew dibuka via SMAPI:

```powershell
python .\scripts\smoke_test.py --start --hold 0.4
```

Pastikan kart lompat.

Lalu training episode checkpoint:

```powershell
python .\scripts\train_dqn.py --episodes 1000 --save-episode-freq 100 --frame-skip 2 --model-path models\junimo_dqn --run-name ep_compare_01
```

Plot:

```powershell
python .\scripts\plot_training.py .\logs\ep_compare_01\monitor.csv
```

Evaluate checkpoints:

```powershell
python .\scripts\evaluate_models.py ".\logs\ep_compare_01\checkpoints\junimo_dqn_ep*.zip" --episodes 20 --out logs\ep_compare_01\evaluation.csv
```

## Caveat saat ini

1. Training masih real-time di game asli, jadi lambat.
2. DQN murni dari nol mungkin butuh sangat banyak episode.
3. Reward shaping masih sederhana.
4. Agent hanya punya dua action: hold/release jump.
5. Model lama sebelum fix jump sebaiknya tidak dipercaya, karena action jump kemungkinan belum benar-benar memengaruhi kart.

## Ide improvement berikutnya

1. Tambah rule-based heuristic teacher untuk warm start.
2. Tambah observation yang lebih semantik, misalnya “gap distance” dan “landing track y”.
3. Tambah accelerated mode di mod agar training tidak real-time.
4. Tambah curriculum: mulai dari level/gap sederhana dulu.
5. Simpan replay/evaluation video atau screenshot untuk debugging.
