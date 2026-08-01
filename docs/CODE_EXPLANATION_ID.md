# Penjelasan Kode - Junimo Kart Reinforcement Learning

Dokumen ini dibuat sebagai antidote dari "vibe coding": bukan cuma tahu command apa yang harus dijalankan, tapi paham file apa yang dibuat, kenapa file itu ada, fungsi tiap bagian kode, dan bagaimana semuanya nyambung sampai agent bisa belajar.

Bahasanya sengaja dibuat Indonesia dan agak naratif. Anggap ini catatan belajar project, bukan dokumentasi formal untuk publikasi paper. Untuk dokumentasi English yang lebih ringkas dan repo-facing, lihat `docs/CODE_WALKTHROUGH.md`.

## Gambaran besar project

Project ini punya tujuan:

```text
Melatih agent Reinforcement Learning untuk memainkan Junimo Kart di Stardew Valley.
```

Tapi Python tidak bisa langsung "masuk" ke Stardew Valley. Karena itu project ini dibagi jadi dua dunia:

```text
1. Dunia Stardew Valley / SMAPI mod
   Bahasa: C#
   Tugas: membaca state internal Junimo Kart dan menekan/menahan jump.

2. Dunia Reinforcement Learning
   Bahasa: Python
   Tugas: menerima state, memilih action, menghitung reward, dan melatih model seperti DQN atau PPO.
```

Alurnya seperti ini:

```text
Stardew Valley sedang jalan
        |
        v
SMAPI mod membaca posisi cart, rel, obstacle, coin, score, lives
        |
        v
SMAPI mod mengirim data itu lewat TCP localhost
        |
        v
Python menerima data sebagai observation/state
        |
        v
Model RL memilih action: release jump atau hold jump
        |
        v
Python mengirim action ke SMAPI mod
        |
        v
SMAPI mod menahan/melepas jump di Junimo Kart
        |
        v
Game lanjut beberapa frame
        |
        v
Python hitung reward: maju bagus, mati buruk, score bagus
        |
        v
Model belajar dari pengalaman itu
```

Kalau disingkat:

```text
state -> action -> game berubah -> reward -> model update
```

Itulah loop dasar Reinforcement Learning.

## Kenapa tidak pakai screenshot / pixel?

Pendekatan umum untuk game AI adalah membaca pixel layar:

```text
screen image -> CNN / neural network -> action
```

Tapi project ini tidak begitu. Project ini membaca internal state langsung dari Junimo Kart, misalnya:

```text
cart x, cart y, velocity x, velocity y,
grounded/jumping,
track pieces ahead,
entities ahead,
score,
lives,
level,
game state
```

Keuntungannya:

- training lebih ringan daripada image-based RL;
- model tidak perlu belajar "mengenali pixel rel" dari nol;
- data lebih bersih untuk eksperimen awal;
- lebih mudah dianalisis dan dijelaskan di paper.

Kekurangannya:

- butuh SMAPI mod dan reflection;
- sangat tergantung struktur internal Stardew Valley;
- kalau update game mengubah nama field internal, mod bisa rusak.

## Struktur folder

```text
src/JunimoKartRLBridge/
  Kode C# SMAPI mod yang hidup di dalam Stardew Valley.

junimo_rl/
  Kode Python package: TCP client dan Gymnasium environment.

scripts/
  Script untuk smoke test, training, plotting, dan evaluasi model.

docs/
  Dokumentasi project.

logs/
  Output training. Di-ignore oleh Git.

models/
  Model hasil training. Di-ignore oleh Git.
```

File yang paling penting:

```text
src/JunimoKartRLBridge/ModEntry.cs
src/JunimoKartRLBridge/BridgeServer.cs
src/JunimoKartRLBridge/Protocol.cs
src/JunimoKartRLBridge/ReflectionUtil.cs
junimo_rl/client.py
junimo_rl/env.py
scripts/train_dqn.py
scripts/evaluate_models.py
scripts/train_ppo.py
scripts/evaluate_ppo_models.py
scripts/plot_training.py
scripts/smoke_test.py
scripts/inspect_semantic_features.py
```

## Bagian C#: SMAPI mod

Bagian C# adalah "tangan dan mata" agent di dalam Stardew Valley.

Mata:

```text
Membaca posisi cart, rel, obstacle, pickup, score, lives, dan status game.
```

Tangan:

```text
Mengubah input jump menjadi hold/release.
```

Bridge:

```text
Membuka server lokal 127.0.0.1:8765 supaya Python bisa berkomunikasi.
```

### `manifest.json`

File:

```text
src/JunimoKartRLBridge/manifest.json
```

Ini adalah file identitas mod untuk SMAPI.

SMAPI butuh tahu:

- nama mod;
- unique ID;
- DLL mana yang harus diload;
- minimum versi SMAPI.

Tanpa file ini, SMAPI tidak tahu bahwa folder tersebut adalah mod.

Secara konsep, ini mirip `package.json` di Node.js atau `pyproject.toml` di Python, tapi untuk SMAPI mod.

### `JunimoKartRLBridge.csproj`

File:

```text
src/JunimoKartRLBridge/JunimoKartRLBridge.csproj
```

Ini file project .NET.

Isinya mengatur:

- target framework;
- dependency SMAPI/Stardew mod build config;
- path game Stardew Valley;
- proses build dan deploy ke folder Mods.

Command build:

```powershell
dotnet build .\src\JunimoKartRLBridge\JunimoKartRLBridge.csproj -c Release
```

Hasil akhirnya adalah DLL mod. DLL ini yang dibaca SMAPI saat Stardew dibuka lewat `StardewModdingAPI.exe`.

### `Config.cs`

File:

```text
src/JunimoKartRLBridge/Config.cs
```

Ini class konfigurasi mod.

Contoh setting yang penting:

```text
BindAddress = "127.0.0.1"
Port = 8765
MaxTracks = jumlah track pieces yang dikirim ke Python
MaxEntities = jumlah entity yang dikirim ke Python
LookaheadPixels = seberapa jauh ke depan agent boleh "melihat"
LookbehindPixels = konteks sedikit di belakang player
ForceRunWhenUnfocused = supaya game tetap jalan saat window tidak fokus
```

Kenapa konfigurasi dipisah?

Supaya angka-angka seperti port, jumlah object, atau behavior auto-continue tidak ditanam mati di kode utama.

### `Protocol.cs`

File:

```text
src/JunimoKartRLBridge/Protocol.cs
```

Ini mendefinisikan bentuk pesan JSON antara Python dan C#.

Bayangkan Python dan C# itu dua orang yang ngobrol. Mereka butuh bahasa yang sama. File ini adalah kamusnya.

Request dari Python berbentuk:

```json
{"type":"action","jump":true}
```

Atau:

```json
{"type":"start","mode":"progress"}
```

Class penting:

```text
ClientRequest
BridgeResponse
BridgeSnapshot
PlayerSnapshot
TrackSnapshot
EntitySnapshot
VectorSnapshot
BoundsSnapshot
```

#### `ClientRequest`

Ini pesan dari Python ke mod.

Field:

```text
Type
Mode
Jump
```

Contoh nilai `Type`:

```text
ping
state
start
reset
action
advance
```

Maknanya:

```text
ping    -> cek bridge hidup atau tidak
state   -> minta snapshot terbaru
start   -> mulai Junimo Kart
reset   -> mulai ulang Junimo Kart
action  -> tahan/lepas jump
advance -> paksa lanjut dari title/map/cutscene ke gameplay
```

#### `BridgeResponse`

Ini balasan dari mod ke Python.

Field:

```text
Ok
Type
Message
State
```

Kalau `Ok = false`, artinya request gagal.

Kalau `Ok = true`, Python bisa mengambil `State` dan mengubahnya menjadi observation vector.

#### `BridgeSnapshot`

Ini snapshot besar dari keadaan Junimo Kart.

Isinya antara lain:

```text
InMinigame
Score
LevelsBeat
GameMode
LivesLeft
CurrentTheme
GameState
ReachedFinish
GameOver
Completed
JumpHeld
Player
TracksAhead
EntitiesAhead
```

Ini adalah data utama yang dikonsumsi Python.

### `BridgeServer.cs`

File:

```text
src/JunimoKartRLBridge/BridgeServer.cs
```

Ini adalah server TCP lokal.

Alamat default:

```text
127.0.0.1:8765
```

Artinya server hanya menerima koneksi dari komputer sendiri, bukan dari internet.

Tugasnya:

```text
1. Membuka port.
2. Menunggu Python connect.
3. Membaca satu baris JSON.
4. Mengubah JSON jadi ClientRequest.
5. Mengirim request ke ModEntry.
6. Mengubah BridgeResponse jadi JSON.
7. Mengirim balik ke Python.
```

Kode intinya secara konsep:

```text
while server running:
    accept client
    while client connected:
        read line
        parse JSON
        handle request
        write JSON response
```

Kenapa pakai JSON-lines?

Karena TCP itu stream, bukan message-based. Kalau kita kirim JSON biasa tanpa pemisah, penerima bisa bingung satu pesan selesai di mana. Dengan newline:

```text
{"type":"state"}\n
{"type":"action","jump":true}\n
```

setiap baris adalah satu pesan.

### `ReflectionUtil.cs`

File:

```text
src/JunimoKartRLBridge/ReflectionUtil.cs
```

Ini salah satu file paling "ajaib" di project.

Masalahnya: banyak data Junimo Kart tersimpan di field private/internal Stardew Valley, bukan public API.

Contoh field internal:

```text
player
_tracks
_entities
isJumpPressed
gameState
levelsBeat
score
livesLeft
```

Kode C# biasa tidak bisa asal akses private field:

```csharp
mineCart.player
```

kalau field itu private.

Reflection memungkinkan kita bilang:

```text
"ambil field bernama player dari object mineCart, walaupun private"
```

Fungsi penting:

```text
Field(target, name)
SetField(target, name, value)
Field<T>(target, name, fallback)
Invoke(target, name, args)
BoolMethod(target, name)
VectorField(target, name)
EnumId(value)
Enumerate(value)
InheritsTypeName(value, typeName)
Bounds(target)
```

#### `Field`

Membaca field dari object.

Konsep:

```text
target = object yang mau dibaca
name   = nama field
```

Contoh:

```csharp
ReflectionUtil.Field<int>(mineCart, "score")
```

Maknanya:

```text
ambil score dari object mineCart sebagai int
```

#### `SetField`

Menulis nilai ke field.

Contoh:

```csharp
ReflectionUtil.SetField(mineCart, "isJumpPressed", desiredJump)
```

Maknanya:

```text
set field isJumpPressed di Junimo Kart menjadi true/false
```

Ini dipakai untuk input jump.

#### `Invoke`

Memanggil method private/internal.

Contoh:

```csharp
ReflectionUtil.Invoke(player, "QueueJump")
ReflectionUtil.Invoke(player, "ReleaseJump")
```

Maknanya:

```text
suruh player mulai jump
suruh player release jump
```

#### Kenapa ada cache?

Reflection relatif mahal. Kalau setiap frame harus cari field/method dari nol, bisa berat.

Karena itu ada:

```csharp
ConcurrentDictionary<string, FieldInfo?> FieldCache
ConcurrentDictionary<string, MethodInfo?> MethodCache
```

Jadi pencarian field/method disimpan. Setelah pertama kali ditemukan, pemanggilan berikutnya lebih cepat.

### `ModEntry.cs`

File:

```text
src/JunimoKartRLBridge/ModEntry.cs
```

Ini file utama mod.

Kalau project ini adalah tubuh, `ModEntry.cs` adalah otaknya di sisi Stardew.

Tugas besarnya:

```text
1. Load config.
2. Start TCP bridge.
3. Handle request dari Python.
4. Start Junimo Kart Progress Mode.
5. Auto-continue dari title/map/cutscene.
6. Apply jump.
7. Create snapshot.
8. Simpan snapshot terbaru untuk Python.
```

#### `Entry`

Method:

```csharp
public override void Entry(IModHelper helper)
```

Ini method pertama yang dipanggil SMAPI saat mod diload.

Di sini mod melakukan:

```text
read config
register update tick event
register console command
start bridge server
```

Event penting:

```csharp
helper.Events.GameLoop.UpdateTicked += this.OnUpdateTicked;
```

Artinya:

```text
setiap game tick, panggil OnUpdateTicked
```

Game tick inilah yang membuat bridge bisa terus memperbarui snapshot dan menerapkan action.

#### `HandleBridgeRequest`

Method:

```csharp
internal BridgeResponse HandleBridgeRequest(ClientRequest? request)
```

Ini menerima request dari `BridgeServer`.

Logikanya:

```text
if request type ping:
    balas pong

if request type state:
    balas snapshot terbaru

if request type start/reset:
    minta game start Junimo Kart

if request type action:
    simpan desiredJumpHeld true/false

if request type advance:
    minta game lanjut dari title/map/cutscene
```

Perhatikan: method ini tidak langsung mengubah game di banyak kasus. Ia sering hanya menyimpan "pending request".

#### Kenapa ada `pendingStartMode`, `pendingAdvance`, `desiredJumpHeld`?

Karena request TCP datang dari thread server, sedangkan game state Stardew sebaiknya diubah dari main game thread.

Kalau thread TCP langsung mengubah game, bisa muncul bug sulit:

```text
race condition
crash
state tidak konsisten
```

Jadi alurnya dibuat aman:

```text
Thread TCP:
    simpan request ke variable pending

Main game tick:
    baca variable pending
    ubah game state
```

Inilah gunanya:

```csharp
private readonly object sync = new();
```

Setiap akses state bersama dibungkus:

```csharp
lock (this.sync)
{
    ...
}
```

`lock` memastikan tidak ada dua thread mengubah data yang sama pada waktu bersamaan.

#### `OnUpdateTicked`

Method ini berjalan setiap game tick.

Alur besarnya:

```text
1. Pastikan game tetap jalan saat unfocused.
2. Kalau world belum ready, snapshot = idle.
3. Ambil pending start / advance / desired jump.
4. Kalau ada pending start, mulai Junimo Kart.
5. Kalau sedang di Junimo Kart:
   - auto continue non-gameplay state
   - apply jump
   - create snapshot
   - simpan snapshot terbaru
6. Kalau tidak sedang di Junimo Kart:
   - snapshot = idle
```

Ini method terpenting di sisi C# karena dia adalah jembatan antara request Python dan game loop asli.

#### `StartMineCart`

Method:

```csharp
private void StartMineCart(string mode)
```

Ini yang langsung membuka Junimo Kart.

Kode penting:

```csharp
Game1.currentMinigame = new MineCart(0, modeId);
```

Mode:

```text
2 = Progress Mode
3 = Infinite / Endless Mode
```

Karena target kamu adalah arcade machine / achievement Progress Mode, mode default adalah:

```text
progress
```

#### `AutoContinueNonGameplayStates`

Junimo Kart tidak selalu langsung gameplay. Ada state seperti:

```text
0 = Title
1 = Ingame
2 = FruitsSummary
3 = Map
4 = Cutscene
```

Kalau Python training dan game berhenti di title/map/cutscene, agent bisa stuck karena tidak ada gameplay.

Maka mod otomatis lanjut:

```text
Title -> restartLevel(true)
Map -> ShowCutscene()
Cutscene -> EndCutscene()
```

Tujuannya supaya training bisa dibiarkan jalan tanpa kamu harus pencet tombol manual.

#### `ApplyJump`

Method:

```csharp
private void ApplyJump(MineCart mineCart, bool desiredJump)
```

Ini mengubah action Python menjadi input Junimo Kart.

Action Python:

```text
0 = release jump
1 = hold jump
```

Kalau `desiredJump = true`:

```csharp
ReflectionUtil.SetField(mineCart, "isJumpPressed", true);
ReflectionUtil.Invoke(player, "QueueJump");
```

Kalau `desiredJump = false`:

```csharp
ReflectionUtil.SetField(mineCart, "isJumpPressed", false);
ReflectionUtil.Invoke(player, "ReleaseJump");
```

Kenapa ada `actualJumpHeld`?

Supaya `QueueJump()` tidak dipanggil terus-menerus setiap frame saat tombol masih ditahan.

Logikanya:

```text
kalau desiredJump true dan sebelumnya belum hold:
    mulai jump

kalau desiredJump true dan sebelumnya sudah hold:
    tetap tahan, jangan panggil QueueJump lagi

kalau desiredJump false dan sebelumnya hold:
    release jump
```

Ini penting karena jump di Junimo Kart bukan cuma "sekali pencet". Tinggi/jauh jump dipengaruhi durasi hold.

#### `CreateSnapshot`

Method:

```csharp
private BridgeSnapshot CreateSnapshot(MineCart mineCart, bool desiredJump)
```

Ini mengambil data internal Junimo Kart dan membungkusnya menjadi `BridgeSnapshot`.

Yang dibaca:

```text
player position
player velocity
score
lives
levels beat
game mode
current theme
game state
game over
reached finish
tracks ahead
entities ahead
```

Hasil snapshot ini dikirim ke Python.

#### `GetTracksAhead`

Method ini mengambil rel di sekitar player.

Konsep:

```text
ambil semua track
hitung dx = track.x - player.x
saring yang dekat dengan player
urutkan dari yang terdekat di depan
ambil maksimal MaxTracks
```

`dx` penting karena agent tidak terlalu butuh koordinat absolut rel di seluruh map. Yang penting:

```text
berapa jauh rel/object dari cart sekarang?
```

Contoh:

```text
dx = 50 berarti object 50 pixel di depan cart
dx = -20 berarti object 20 pixel di belakang cart
```

#### `GetEntitiesAhead`

Mirip `GetTracksAhead`, tapi untuk entity:

```text
coin
fruit
obstacle
pickup
object lain
```

Player sendiri dan track tidak dihitung sebagai entity target.

## Bagian Python: komunikasi dan environment

Bagian Python adalah "otak learning".

Tugasnya:

```text
1. Connect ke bridge C#.
2. Minta state dari game.
3. Ubah JSON state menjadi vector angka.
4. Definisikan action space.
5. Hitung reward.
6. Jalankan training DQN.
7. Simpan model, log, checkpoint, dan plot.
```

### `junimo_rl/client.py`

File:

```text
junimo_rl/client.py
```

Ini TCP client di sisi Python.

Class utama:

```python
JunimoKartBridgeClient
```

#### `connect`

Membuka koneksi ke:

```text
127.0.0.1:8765
```

Kalau `_sock` sudah ada, dia tidak membuat koneksi baru.

#### `close`

Menutup file object dan socket.

Ini penting supaya koneksi tidak menggantung.

#### `request`

Method umum untuk mengirim payload ke bridge.

Contoh payload:

```python
{"type": "state"}
```

Method ini melakukan retry sekali kalau koneksi putus.

Kenapa retry?

Karena komunikasi TCP bisa putus kalau SMAPI restart, game close, atau bridge reconnect.

#### `_request_once`

Ini versi satu percobaan request.

Alurnya:

```text
1. connect
2. json.dumps(payload)
3. tambah newline
4. encode UTF-8
5. write ke socket
6. readline response
7. json.loads response
8. cek ok true/false
```

Kalau bridge tidak bisa dicapai, error message dibuat lebih manusiawi:

```text
Pastikan Stardew dibuka lewat SMAPI dan save sudah loaded.
```

#### Method pendek

Ada wrapper supaya code lain lebih enak dibaca:

```python
ping()
state()
start(mode="progress")
advance()
action(jump: bool)
```

Contoh:

```python
client.action(True)
```

lebih jelas daripada:

```python
client.request({"type": "action", "jump": True})
```

### `junimo_rl/env.py`

File:

```text
junimo_rl/env.py
```

Ini file paling penting di sisi Python.

Stable-Baselines3 tidak tahu apa itu Junimo Kart. Dia hanya tahu interface Gymnasium:

```text
reset()
step(action)
observation_space
action_space
```

Maka tugas `JunimoKartEnv` adalah membuat Junimo Kart terlihat seperti RL environment standar.

#### Konstanta observation

Di atas file ada:

```python
MAX_TRACKS = 24
MAX_ENTITIES = 24
TRACK_FEATURES = 5
ENTITY_FEATURES = 7
BASE_FEATURES = 18
OBSERVATION_SIZE = BASE_FEATURES + MAX_TRACKS * TRACK_FEATURES + MAX_ENTITIES * ENTITY_FEATURES
```

Artinya:

```text
base features = 18 angka
track features = 24 track * 5 angka = 120 angka
entity features = 24 entity * 7 angka = 168 angka
total = 18 + 120 + 168 = 306 angka
```

Jadi setiap state yang masuk ke neural network adalah vector:

```text
306 angka float32
```

#### Helper `_num`

Fungsi:

```python
def _num(value, default=0.0) -> float
```

Tugasnya mengubah value menjadi float dengan aman.

Kalau value `None` atau tidak bisa dikonversi, pakai default.

Kenapa perlu?

Karena JSON dari game kadang field-nya tidak ada, null, atau tipe datanya tidak sesuai dugaan.

#### Helper `_bool`

Mengubah boolean menjadi angka:

```text
true  -> 1.0
false -> 0.0
```

Neural network tidak menerima boolean mentah. Semua harus angka.

#### Helper `_stable_unit`

Fungsi ini mengubah label string menjadi angka stabil antara 0 sampai 1.

Contoh label:

```text
"Coin"
"Fruit"
"Obstacle"
"Slime"
```

Kenapa tidak langsung pakai string?

Karena neural network butuh angka.

Kenapa pakai hash?

Karena kita belum punya mapping manual semua tipe object. Hash membuat string apapun bisa jadi angka.

Catatan penting: ini praktis, tapi bukan representasi terbaik. Untuk riset lebih rapi, lebih baik pakai mapping eksplisit atau one-hot encoding.

#### `snapshot_to_vector`

Fungsi:

```python
def snapshot_to_vector(snapshot: dict[str, Any]) -> np.ndarray
```

Ini mengubah JSON snapshot dari C# menjadi vector fixed-size.

Bagian base features:

```text
inMinigame
version
score
livesLeft
levelsBeat
gameMode
currentTheme
gameState
gameOver
reachedFinish
completed
jumpHeld
player x
player y
velocity x
velocity y
grounded
jumping
```

Beberapa nilai dibagi angka tertentu:

```python
score / 10000.0
lives / 10.0
levelsBeat / 6.0
position.x / 10000.0
position.y / 1000.0
velocity.x / 1000.0
velocity.y / 1000.0
```

Ini disebut normalisasi.

Tujuannya supaya angka tidak terlalu besar. Neural network biasanya lebih stabil kalau input berada di kisaran kecil.

Bagian track features:

Untuk setiap track:

```text
dx
y
typeId
hasObstacle
obstacleType
```

Jika track kurang dari 24, sisanya diisi nol.

Bagian entity features:

Untuk setiap entity:

```text
dx
y
width
height
type
isObstacle
isPickup
```

Jika entity kurang dari 24, sisanya diisi nol.

Kenapa harus padding?

Neural network butuh ukuran input tetap. Tidak bisa episode A punya 80 angka dan episode B punya 306 angka.

#### Semantic features

Versi awal model hanya melihat list track dan entity mentah. Itu bisa dipelajari, tapi berat. Manusia langsung melihat:

```text
ada jurang di depan
jurangnya lebar
landing-nya lebih tinggi
ada obstacle sebelum landing
ada coin/fruit di depan
```

Model awal tidak langsung diberi konsep itu. Dia cuma melihat deretan angka track:

```text
track dx, track y, track type, ...
```

Maka sekarang ditambahkan opsi:

```powershell
--semantic-features
```

Kalau flag ini aktif, Python akan menghitung fitur tambahan dari `tracksAhead` dan `entitiesAhead`.

Fitur semantic yang ditambahkan:

```text
next_track_dx
  Jarak ke track/rel terdekat di depan cart.

next_track_y
  Tinggi/y track terdekat di depan.

next_track_type_id
  Jenis track di depan.

next_track_has_obstacle
  Apakah track terdekat punya obstacle.

next_gap_present
  Apakah ada gap/jurang yang terdeteksi.

next_gap_start_dx
  Jarak kira-kira ke awal gap dari posisi cart sekarang.

next_gap_width
  Lebar gap/jurang yang diperkirakan dari jarak antar track piece.

landing_y
  Tinggi/y track tempat landing setelah gap.

landing_delta_y
  Selisih tinggi landing dibanding track sebelum gap.
  Di koordinat game 2D biasanya Y makin besar berarti makin bawah di layar.

next_obstacle_present
  Apakah ada obstacle di depan.

next_obstacle_dx
  Jarak obstacle terdekat.

next_obstacle_y
  Tinggi/y obstacle terdekat.

next_pickup_present
  Apakah ada pickup seperti coin/fruit.

next_pickup_dx
  Jarak pickup terdekat.

next_pickup_y
  Tinggi/y pickup terdekat.

distance_to_finish
  Perkiraan jarak ke finish.

progress_fraction
  Perkiraan progress posisi cart terhadap panjang level.
```

Jadi jawaban pertanyaan kamu: iya, `gap width` itu lebar jurang, `next_gap_start_dx` itu koordinat relatif awal jurang dari cart, dan `landing_y` itu tinggi landing setelah jurang.

Catatan kecil: karena bridge saat ini belum mengirim lebar fisik tiap track piece, deteksi gap masih heuristic. Python mengurutkan track berdasarkan `dx`, lalu kalau jarak antar track piece lebih besar dari threshold tertentu, itu dianggap gap.

Untuk reward `shaped_v2`, syaratnya dibuat lebih ketat:

```text
gap_width >= 56 pixel
next_gap_start_dx <= 180 pixel
```

Jadi gap kecil/ambigu tidak langsung dianggap percobaan melewati jurang untuk bonus landing.

Catatan penting: model lama DQN 10k tidak bisa langsung memakai semantic features, karena ukuran observation berubah. Untuk model lama, jangan pakai `--semantic-features` saat evaluate. Untuk run baru, pakai flag itu dari awal training sampai evaluation.

#### `JunimoKartEnv`

Class:

```python
class JunimoKartEnv(gym.Env[np.ndarray, int])
```

Ini environment RL.

#### `__init__`

Mengatur:

```text
host
port
mode
frame_skip
fps
client
action_space
observation_space
action_mode
macro_action_frames
```

Default action space:

```python
spaces.Discrete(2)
```

Artinya hanya ada dua action:

```text
0 = release jump
1 = hold jump
```

Ini disebut:

```text
action_mode = binary
```

Sekarang ada mode baru:

```text
action_mode = macro
```

Kalau `--action-mode macro`, action space berubah menjadi:

```python
spaces.Discrete(4)
```

Artinya:

```text
0 = release jump
1 = short hold
2 = medium hold
3 = long/continue hold
```

Kenapa ini lebih worth?

Karena Junimo Kart bukan cuma soal "lompat atau tidak". Tinggi/jarak lompatan sangat tergantung seberapa lama tombol jump ditahan.

Dengan mode lama:

```text
1, 1, 1, 0
```

baru berarti agent menahan jump beberapa step.

Dengan macro action:

```text
action 1 = tahan sebentar
action 2 = tahan sedang
action 3 = tahan lama
```

Jadi model lebih mudah mengekspresikan durasi jump.

Catatan penting:

```text
checkpoint binary tidak bisa dilanjutkan sebagai macro
checkpoint macro tidak bisa dievaluate sebagai binary
```

Karena ukuran output policy berubah dari 2 action menjadi 4 action.

Observation space:

```python
spaces.Box(shape=(OBSERVATION_SIZE,), dtype=np.float32)
```

Artinya observation adalah vector float dengan panjang tetap.

#### `reset`

Dipanggil saat episode baru mulai.

Alurnya:

```text
1. Start Junimo Kart Progress Mode.
2. Tunggu sampai game benar-benar masuk gameplay.
3. Kalau masih title/map/cutscene, panggil advance.
4. Simpan snapshot pertama.
5. Return observation vector.
```

Dalam RL, `reset()` harus mengembalikan state awal episode.

#### `step`

Dipanggil setiap agent memilih action.

Alur:

```text
1. Simpan snapshot lama.
2. Terapkan action ke game.
3. Untuk binary: kirim jump true/false lalu tunggu frame_skip.
4. Untuk macro: tahan jump sebagian/seluruh macro_action_frames.
5. Ambil snapshot baru.
6. Hitung reward dari perubahan old -> new.
7. Tentukan episode selesai atau belum.
8. Return observation, reward, terminated, truncated, info.
```

Kode konsep:

```python
old = last_snapshot
apply_action(action)
new = client.state()
reward = reward(old, new)
```

Untuk binary:

```python
client.action(action == 1)
sleep(frame_skip / fps)
```

Untuk macro:

```text
action 0 -> release sepanjang window
action 1 -> hold sekitar 25% window, lalu release
action 2 -> hold sekitar 50% window, lalu release
action 3 -> hold sepanjang window
```

Kalau:

```text
macro_action_frames = 8
fps = 60
```

maka satu macro action mengontrol:

```text
8 / 60 = 0.133 detik gameplay
```

#### `frame_skip` dan `macro_action_frames`

Kalau `frame_skip = 2`, artinya agent memilih action tiap 2 frame.

Dengan FPS 60:

```text
2 frame = 2 / 60 detik = 0.033 detik
```

Semakin kecil:

```text
kontrol lebih presisi, training lebih lambat
```

Semakin besar:

```text
training lebih cepat, timing jump lebih kasar
```

Untuk `action_mode = macro`, durasi action lebih banyak dikontrol oleh:

```text
macro_action_frames
```

Default-nya:

```text
8 frame
```

Jadi binary mode cocok untuk kontrol halus, sedangkan macro mode cocok untuk memberi model pilihan durasi jump yang lebih eksplisit.

#### `terminated` dan `truncated`

Di Gymnasium:

```text
terminated = episode selesai karena tujuan tercapai
truncated  = episode berhenti karena batas/kegagalan eksternal
```

Di kode:

```python
terminated = completed
truncated = gameOver and not completed
```

Artinya:

```text
kalau Progress Mode selesai -> terminated
kalau game over -> truncated
```

#### `_reward`

Ini fungsi yang menentukan "bagus atau buruk".

Sekarang project punya tiga versi reward:

```text
legacy
  Reward lama. Ini yang dipakai oleh training DQN kamu sebelumnya.

shaped_v1
  Reward baru yang lebih dense. Ini yang gue tambahkan untuk eksperimen semantic features.

shaped_v2
  Reward yang lebih outcome-based. Fokusnya bukan "lompat dekat gap", tapi "berhasil melewati gap dan landing aman".
```

Default-nya tetap `legacy`, supaya model lama tidak rusak.

#### Reward versi `legacy`

Formula legacy:

```text
reward =
  0.001 * delta_x
+ 0.01  * delta_score
+ 50    * max(delta_level, 0)
+ 10    * delta_life
+ 250   jika completed
- 100   jika baru game over
- 1     jika keluar dari minigame
```

Makna tiap komponen:

```text
delta_x
  Cart bergerak maju. Makin maju, makin bagus.

delta_score
  Score naik, biasanya karena coin, fruit, atau objective lain.

delta_level
  Berhasil mengalahkan level.

delta_life
  Jika lives berkurang, nilainya negatif, jadi penalti.

completed
  Reward besar jika Progress Mode selesai.

gameOver
  Penalti besar saat mati/game over.
```

Contoh:

```text
Cart maju 40 pixel
Score tidak naik
Level tidak naik
Lives tetap
Belum mati

reward = 0.001 * 40 = 0.04
```

Contoh mati:

```text
Cart maju 20 pixel
Lives turun dari 1 ke 0
Game over

delta_x = 20
life_delta = -1

reward =
  0.001 * 20
+ 10 * (-1)
- 100
= 0.02 - 10 - 100
= -109.98
```

Makanya log kamu sering di sekitar:

```text
-105 sampai -107
```

Itu biasanya pola:

```text
sempat maju sedikit -> mati -> kena penalti besar
```

#### Reward versi `shaped_v1`

Formula barunya:

```text
reward =
  0.003 * delta_x
+ 0.02  * delta_score
+ 100   * max(delta_level, 0)
+ 25    * delta_life
+ 0.02  jika masih hidup dan sedang di gameplay
+ 500   jika completed
- 80    jika baru game over
- 2     jika keluar dari minigame

tambahan semantic shaping:
+ 0.08  jika agent hold jump saat grounded dan ada gap dekat
- 0.08  jika agent tidak jump saat grounded dan gap sudah sangat dekat
- 0.015 jika agent hold jump saat grounded padahal tidak ada gap/obstacle dekat
+ 0.20  jika agent berhasil landing aman setelah sebelumnya jumping
```

Kenapa dibuat begitu?

```text
legacy reward
  Agent sering cuma tahu "mati = buruk", tapi tidak selalu tahu aksi kecil mana yang membuat dia mati.

shaped_v1 reward
  Agent diberi sinyal yang lebih rapat:
  - maju sedikit tetap dihargai;
  - survive per step tetap dihargai kecil;
  - jump dekat jurang dihargai;
  - tidak jump dekat jurang diberi penalti kecil;
  - landing aman diberi bonus kecil.
```

Ini bukan berarti model langsung pintar. Tapi dibanding cuma menunggu sinyal mati, model dapat petunjuk lebih sering.

#### Reward versi `shaped_v2`

Formula shaped v2:

```text
reward =
  0.004 * delta_x
+ 0.005 * delta_score
+ 120   * max(delta_level, 0)
+ 25    * delta_life
+ 0.03  jika masih hidup dan sedang di gameplay
+ 600   jika completed
- 80    jika baru game over
- 2     jika keluar dari minigame

tambahan gap-specific shaping:
+ bonus jika berhasil melewati tracked gap dan landing aman
- penalti kecil jika tidak hold jump saat gap sungguhan sudah sangat dekat
- penalti kecil jika hold jump saat grounded padahal tidak ada gap/obstacle dekat
- penalti kecil tambahan jika percobaan melewati gap berakhir game over
```

Perubahan penting dibanding `shaped_v1`:

```text
1. Reward score diperkecil:
   shaped_v1 = 0.02 * delta_score
   shaped_v2 = 0.005 * delta_score

2. Bonus landing biasa dihapus.
   Agent tidak lagi diberi bonus hanya karena lompat-lompat lalu landing di rel lurus.

3. Bonus landing hanya diberikan kalau sebelumnya ada gap yang dilacak.
   Jadi reward-nya lebih dekat ke outcome:
   "berhasil melewati jurang" bukan "asal lompat".
```

Gap landing reward memakai memory kecil di environment:

```text
1. Python melihat ada gap dekat di depan.
2. Environment menyimpan active_gap_attempt.
3. Kalau cart melewati ujung gap dan grounded lagi tanpa gameOver:
   beri bonus landing.
4. Kalau mati saat active_gap_attempt:
   beri penalti kecil tambahan.
```

Bonus landing setelah gap:

```text
3.0 + 0.02 * min(gap_width, 120)
```

Contoh:

```text
gap_width = 80

bonus = 3.0 + 0.02 * 80
bonus = 4.6
```

Ini cukup besar untuk memberi sinyal "melewati gap itu bagus", tapi masih jauh lebih kecil daripada reward menyelesaikan stage.

Catatan penting:

```text
Reward legacy, shaped_v1, dan shaped_v2 tidak apple-to-apple kalau dibandingkan dari angka mean_reward saja.
```

Untuk paper/comparison, lebih adil pakai metric game seperti:

```text
mean_score
mean_length
max_score
levels_beat
completion_rate
gameover_rate
```

Atau latih semua model dengan setting observation dan reward yang sama.

## Bagian script

Folder:

```text
scripts/
```

Isi folder ini adalah program-program kecil yang menjalankan workflow project.

### `scripts/smoke_test.py`

Tujuan:

```text
Mengecek apakah bridge jalan dan apakah jump bisa dikontrol.
```

Command:

```powershell
python .\scripts\smoke_test.py --start --hold 0.4
```

Yang dilakukan:

```text
1. Connect ke bridge.
2. Ping bridge.
3. Kalau --start, mulai Junimo Kart.
4. Kalau --hold, tahan jump beberapa detik.
5. Print state akhir.
```

Pakai ini sebelum training. Kalau smoke test gagal, training hampir pasti tidak valid.

### `scripts/inspect_semantic_features.py`

Tujuan:

```text
Melihat fitur semantic secara live, supaya kita tahu "mata bot" sedang membaca apa.
```

Command:

```powershell
python .\scripts\inspect_semantic_features.py --interval 0.25
```

Output-nya kira-kira berisi:

```text
x
grounded
gap
gap_dx
gap_width
landing_y
landing_dy
obstacle_dx
pickup_dx
progress
```

Maknanya:

```text
gap_dx
  Jarak kira-kira dari cart ke awal gap/jurang.

gap_width
  Lebar gap berdasarkan jarak antar track piece yang terdeteksi.

landing_y
  Posisi Y track pertama setelah gap.

landing_dy
  Selisih tinggi landing dibanding track sebelum gap.
```

Ini script debugging saja. Dia tidak melatih model dan tidak menekan tombol jump.

### `scripts/train_dqn.py`

Ini script training utama.

Command contoh:

```powershell
python .\scripts\train_dqn.py --episodes 1000 --save-episode-freq 100 --frame-skip 2 --model-path models\junimo_dqn --run-name ep_compare_01
```

Tugas script:

```text
1. Parse command-line arguments.
2. Buat folder log.
3. Buat environment JunimoKartEnv.
4. Bungkus environment dengan Monitor.
5. Simpan hyperparameter ke hparams.txt.
6. Buat model DQN baru atau load model lama.
7. Pasang callbacks untuk checkpoint.
8. Jalankan model.learn().
9. Simpan final model.
```

#### Argumen penting

```text
--episodes
  Berapa episode training.

--timesteps
  Berapa step training kalau tidak pakai episode target.

--model-path
  Lokasi final model disimpan.

--load-model
  Model lama yang ingin dilanjutkan.

--run-name
  Nama folder log.

--save-freq
  Save checkpoint per jumlah timestep.

--save-episode-freq
  Save checkpoint per jumlah episode.

--frame-skip
  Berapa frame game dilewati per action.

--learning-rate
  Kecepatan update neural network.

--buffer-size
  Ukuran replay buffer.

--learning-starts
  Jumlah step awal sebelum model mulai update.

--batch-size
  Jumlah sample per update.

--gamma
  Discount factor future reward.

--exploration-fraction
  Porsi training untuk menurunkan eksplorasi random.

--exploration-final-eps
  Sisa randomness minimum.
```

#### `EpisodeCheckpointCallback`

Stable-Baselines3 sudah punya checkpoint berdasarkan timestep. Tapi kamu ingin compare:

```text
model episode 100
model episode 1000
model episode 2000
model episode 10000
```

Maka dibuat callback custom:

```python
class EpisodeCheckpointCallback(BaseCallback)
```

Tugasnya:

```text
setiap ada episode selesai:
    hitung episode_count
    kalau sudah mencapai kelipatan save_freq_episodes:
        save model
        save replay buffer
```

Nama checkpoint:

```text
junimo_dqn_ep001000_steps58633.zip
```

Maknanya:

```text
episode kumulatif = 1000
total timesteps saat save = 58633
```

Kalau lanjut training dari 1000 episode:

```powershell
--episode-offset 1000
--episodes 9000
--save-episode-freq 1000
```

maka checkpoint berikutnya jadi:

```text
ep002000
ep003000
...
ep010000
```

#### `Monitor`

Kode:

```python
env = Monitor(JunimoKartEnv(...), filename="monitor.csv")
```

`Monitor` mencatat:

```text
r = total reward per episode
l = panjang episode dalam step
t = waktu sejak training mulai
```

Itulah sumber data log yang kemarin kita baca.

### `scripts/train_ppo.py`

Ini script training PPO.

PPO memakai environment yang sama persis dengan DQN:

```text
JunimoKartEnv
```

Untuk PPO baseline, bagian Stardew/SMAPI bridge dan environment-nya sama. Kalau memakai `--semantic-features` atau `--action-mode macro`, yang berubah adalah representasi observation/action di Python, bukan SMAPI mod-nya.

Command contoh:

```powershell
python .\scripts\train_ppo.py --episodes 10000 --save-episode-freq 1000 --frame-skip 2 --model-path models\ppo\junimo_ppo --run-name ppo_10k
```

Command yang sekarang menurut gue lebih worth dicoba:

```powershell
python .\scripts\train_ppo.py --episodes 5000 --save-episode-freq 1000 --save-freq 0 --frame-skip 2 --semantic-features --reward-version shaped_v2 --action-mode macro --macro-action-frames 8 --model-path models\ppo\junimo_ppo_macro_v2 --run-name ppo_semantic_shaped_v2_macro_5k
```

Versi script siap jalan:

```powershell
.\scripts\run_ppo_macro_v2.ps1
```

Isi script itu hanya membungkus command di atas supaya kamu tidak perlu copy command panjang setiap kali mulai eksperimen.

Ini start dari nol, bukan continue dari checkpoint lama, karena action space-nya berubah:

```text
binary = 2 action
macro  = 4 action
```

Default folder PPO:

```text
logs/ppo/<run-name>/
models/ppo/
```

Contoh output:

```text
logs/ppo/ppo_10k/monitor.csv
logs/ppo/ppo_10k/hparams.txt
logs/ppo/ppo_10k/checkpoints/
logs/ppo/ppo_10k/tensorboard/
models/ppo/junimo_ppo.zip
```

Kenapa PPO dibuat folder terpisah?

Supaya hasil DQN dan PPO tidak campur:

```text
logs/continue_to_10k       -> hasil DQN lama
logs/ppo/ppo_10k           -> hasil PPO
models/junimo_dqn.zip      -> final DQN
models/ppo/junimo_ppo.zip  -> final PPO
```

PPO tidak menyimpan replay buffer seperti DQN. Ini sengaja supaya storage C tidak cepat habis lagi. Untuk analisis dan evaluasi, file `.zip` checkpoint sudah cukup.

Script ini juga punya:

```python
EpisodeProgressCallback
```

Fungsinya mencetak progress ringkas setiap N episode selesai.

Awalnya default-nya setiap 4 episode, tapi itu terlalu berisik di terminal. Sekarang default-nya dimatikan supaya yang muncul cukup table bawaan Stable-Baselines3.

Kalau diaktifkan manual, bentuknya:

```text
PPO episode progress | episodes=4 | total_timesteps=... | recent_reward_mean=... | recent_length_mean=...
```

Opsi yang relevan:

```text
--progress-episode-freq
  Print progress setiap N episode. Default 0, artinya disable.

--progress-window
  Jumlah episode terakhir yang dipakai untuk recent_reward_mean dan recent_length_mean.
```

#### Bedanya PPO dengan DQN di kode

DQN:

```python
from stable_baselines3 import DQN
model = DQN("MlpPolicy", env, ...)
```

PPO:

```python
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, ...)
```

Secara konsep:

```text
DQN belajar Q(state, action)
PPO belajar policy(action | state)
```

Secara default, action-nya:

```text
0 = release jump
1 = hold jump
```

Kalau memakai `--action-mode macro`, action-nya menjadi:

```text
0 = release jump
1 = short hold
2 = medium hold
3 = long/continue hold
```

Yang berbeda adalah cara update neural network-nya.

#### Hyperparameter penting PPO

```text
--n-steps
  Berapa step gameplay dikumpulkan sebelum PPO update model.

--batch-size
  Ukuran minibatch saat update.

--n-epochs
  Berapa kali PPO mengulang belajar dari rollout yang sama.

--gae-lambda
  Mengatur bias/variance advantage estimation.

--clip-range
  Membatasi perubahan policy supaya training lebih stabil.

--ent-coef
  Bonus exploration. Lebih tinggi berarti policy lebih didorong untuk tidak terlalu cepat kaku.
```

Default awal yang dibuat:

```text
learning_rate = 3e-4
n_steps = 1024
batch_size = 64
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
ent_coef = 0.01
```

Ini bukan pasti optimal. Ini baseline PPO pertama untuk dibandingkan melawan DQN.

### `scripts/plot_training.py`

Tujuan:

```text
Mengubah monitor.csv menjadi chart PNG.
```

Command:

```powershell
python .\scripts\plot_training.py .\logs\ep_compare_01\monitor.csv
```

Output:

```text
logs\ep_compare_01\training_plot.png
```

Fungsi penting:

```text
read_monitor_csv
rolling_mean
main
```

#### `read_monitor_csv`

Membaca `monitor.csv`.

File monitor punya baris metadata yang diawali `#`, jadi script melewati baris itu.

#### `rolling_mean`

Menghaluskan grafik.

Kalau reward per episode naik turun liar, rolling mean membuat trend lebih mudah dilihat.

Contoh window 20:

```text
nilai episode sekarang = rata-rata 20 episode terakhir
```

#### Chart yang dibuat

```text
1. episode reward
2. reward rolling mean
3. episode length
4. length rolling mean
```

Interpretasi:

```text
reward naik -> model makin bagus
episode length naik -> model hidup lebih lama
```

Tapi hati-hati: episode length naik tidak selalu berarti bagus kalau agent cuma hidup lama tanpa progress.

### `scripts/evaluate_models.py`

Tujuan:

```text
Membandingkan beberapa checkpoint model secara deterministic.
```

Training DQN memakai exploration, jadi kadang action random masih terjadi. Evaluation harus lebih fair:

```python
model.predict(obs, deterministic=True)
```

Artinya:

```text
pilih action terbaik menurut model, bukan random exploration
```

Command:

```powershell
python .\scripts\evaluate_models.py ".\logs\ep_compare_01\checkpoints\junimo_dqn_ep*.zip" --episodes 20 --out logs\ep_compare_01\evaluation.csv
```

Output CSV:

```text
algorithm
model
semantic_features
reward_version
action_mode
macro_action_frames
episodes
mean_reward
mean_length
mean_score
max_score
mean_levels_beat
max_levels_beat
completion_rate
gameover_rate
jump_hold_ratio
```

Ini lebih cocok untuk compare:

```text
checkpoint 100 episode vs 1000 episode vs 2000 episode
```

daripada hanya melihat training log.

### `scripts/evaluate_ppo_models.py`

Ini evaluator untuk checkpoint PPO.

Command:

```powershell
python .\scripts\evaluate_ppo_models.py ".\logs\ppo\ppo_10k\checkpoints\junimo_ppo_ep*.zip" --episodes 20 --frame-skip 2 --out logs\ppo\ppo_10k\evaluation.csv
```

Kalau model dilatih dengan semantic + shaped + macro, evaluasinya juga harus memakai flag yang sama:

```powershell
python .\scripts\evaluate_ppo_models.py ".\logs\ppo\ppo_semantic_shaped_v2_macro_5k\checkpoints\junimo_ppo_ep*.zip" --episodes 20 --frame-skip 2 --semantic-features --reward-version shaped_v2 --action-mode macro --macro-action-frames 8 --out logs\ppo\ppo_semantic_shaped_v2_macro_5k\evaluation.csv
```

Output CSV-nya sengaja dibuat sama formatnya dengan evaluator DQN supaya nanti bisa dibandingkan di Excel/paper:

```text
DQN checkpoint vs PPO checkpoint
```

Evaluator tetap butuh Stardew dibuka lewat SMAPI, karena environment-nya masih live game, bukan simulator offline.

## Bagaimana DQN belajar di project ini?

DQN belajar fungsi:

```text
Q(state, action)
```

Artinya:

```text
seberapa bagus action tertentu kalau dilakukan di state sekarang?
```

Dalam binary mode, action hanya dua:

```text
0 = release jump
1 = hold jump
```

model output-nya kira-kira:

```text
Q(state, release) = angka
Q(state, hold)    = angka
```

Dalam macro mode, DQN akan punya empat output:

```text
Q(state, release)
Q(state, short_hold)
Q(state, medium_hold)
Q(state, long_hold)
```

Agent memilih action dengan nilai Q lebih tinggi.

Contoh:

```text
Q(state, release) = -20
Q(state, hold)    = -5
```

Karena `-5` lebih tinggi daripada `-20`, agent memilih:

```text
hold jump
```

### Bellman equation

Rumus inti DQN:

```text
target = reward + gamma * max(Q(next_state, next_action))
```

Dalam bahasa manusia:

```text
Nilai action sekarang =
reward langsung sekarang
+
perkiraan reward terbaik dari state berikutnya
```

Loss:

```text
loss = (Q(state, action) - target)^2
```

Model dilatih agar prediksi Q-nya semakin dekat ke target.

### Replay buffer

DQN tidak langsung belajar hanya dari pengalaman terbaru.

Ia menyimpan pengalaman:

```text
state, action, reward, next_state, done
```

ke replay buffer.

Lalu saat training, model mengambil batch acak dari buffer.

Kenapa?

Karena data game berurutan sangat berkorelasi. Replay buffer membuat sample lebih beragam dan training lebih stabil.

### Exploration

Di awal training, agent harus banyak coba-coba.

Kalau dari awal selalu pilih action yang menurut model "terbaik", padahal model belum tahu apa-apa, dia bisa stuck.

DQN memakai epsilon-greedy:

```text
dengan probabilitas epsilon:
    pilih action random

selain itu:
    pilih action terbaik menurut Q-value
```

Di script:

```text
exploration_fraction = 0.25
exploration_final_eps = 0.05
```

Artinya randomness turun selama 25% awal training, lalu tetap menyisakan 5% random action.

## Kenapa hasil DQN legacy sebelumnya belum bagus?

Dari log sebelumnya, reward masih sekitar:

```text
-105 sampai -106
```

Ini menunjukkan pola:

```text
agent main sebentar -> mati -> kena penalti besar
```

Ada beberapa kemungkinan penyebab:

### 1. Reward terlalu sparse / keras

Penalti game over `-100` sangat besar.

Reward maju:

```text
0.001 * dx
```

Kalau cart maju 50 pixel:

```text
reward = 0.05
```

Tapi mati:

```text
-100
```

Jadi sinyal kecil "aku tadi sempat lebih baik sedikit" tenggelam oleh penalti mati.

### 2. State masih terlalu mentah

Model melihat list track pieces dan entities.

Di awal project, model belum langsung diberi fitur semantic seperti:

```text
next_gap_start
next_gap_width
landing_y
next_obstacle_dx
next_pickup_dx
```

Manusia melihat "ada jurang". Model legacy hanya melihat deretan angka track.

Dia bisa belajar dari itu, tapi lebih sulit.

Sekarang sudah ada opsi:

```text
--semantic-features
--reward-version shaped_v2
```

Tujuannya adalah memberi model versi "sudah diringkas" dari situasi game, misalnya jarak ke jurang, lebar jurang, dan tinggi landing.

### 3. Action terlalu miskin untuk timing jump

Binary action cuma:

```text
release
hold
```

Secara teori agent bisa belajar durasi jump dari urutan action:

```text
hold, hold, hold, release
```

Tapi di game real-time, ini lebih sulit karena timing-nya halus. Macro action memberi pilihan durasi langsung:

```text
short hold
medium hold
long hold
```

Ini bukan membuat rule-based policy. Model tetap memilih action sendiri, tapi action vocabulary-nya lebih cocok dengan mekanik game.

### 4. DQN mungkin bukan algoritma terbaik

DQN bisa, tapi game real-time dengan timing jump sering lebih enak dicoba dengan:

```text
PPO
Recurrent PPO
Behavioral Cloning + RL
```

### 5. Training real-time lambat

Karena game berjalan sungguhan, 1000 episode tidak selalu banyak untuk RL.

Di tutorial Flappy Bird, training bisa dipercepat ribuan frame per detik. Di sini kita jauh lebih lambat.

## Cara membaca log training

Log `monitor.csv` punya kolom:

```text
r = reward total episode
l = length episode dalam step
t = waktu
```

Contoh:

```text
r=-106.98, l=63
```

Maknanya:

```text
episode itu selesai dengan total reward -106.98
panjang episode 63 environment step
```

Kalau `frame_skip = 2` dan FPS 60:

```text
1 step = 2 frame = 0.033 detik
63 step = sekitar 2.1 detik gameplay
```

Jadi episode length 63 artinya cart biasanya mati sangat cepat.

Target yang kita mau:

```text
reward rolling mean naik
episode length naik secara sehat
max levels beat naik
completion rate naik
```

Jangan hanya melihat satu episode terbaik. RL noisy. Yang penting trend rata-rata.

## Apa yang harus dipahami kalau ingin modifikasi kode

Kalau mau mengubah apa yang agent lihat:

```text
ubah C# snapshot di ModEntry.cs / Protocol.cs
ubah Python snapshot_to_vector di env.py
ubah observation_size / semantic feature list kalau jumlah feature berubah
```

Kalau mau mengubah reward:

```text
ubah _reward di junimo_rl/env.py
tambah versi baru, misalnya shaped_v2, agar eksperimen lama tetap bisa direproduksi
```

Kalau mau mengubah action:

```text
ubah action_space di env.py
ubah step(action)
ubah action handling di ModEntry.cs jika action baru butuh kontrol game baru
```

Kalau mau mengganti algoritma DQN ke PPO:

```text
pakai scripts/train_ppo.py
pakai environment JunimoKartEnv yang sama
atur hyperparameter PPO
```

Kalau mau menambah logging:

```text
bisa tambah info di env.step()
bisa tambah custom callback di train_dqn.py
bisa tambah custom callback di train_ppo.py
bisa tambah field snapshot dari C#
```

## Mental model coding project ini

Kalau kamu merasa "ini banyak banget", pecah jadi lima konsep kecil:

### 1. Socket communication

Python dan C# ngobrol lewat TCP.

```text
Python kirim JSON
C# balas JSON
```

### 2. Game bridge

C# mod hidup di dalam Stardew dan punya akses ke object game.

```text
MineCart object -> reflection -> snapshot
```

### 3. Gymnasium environment

Python membungkus game sebagai environment standar RL.

```text
reset()
step()
reward()
observation_space
action_space
```

### 4. Model training

Stable-Baselines3 DQN/PPO mengambil environment dan belajar dari interaksi.

```text
model.learn()
```

### 5. Experiment management

Script menyimpan:

```text
model
checkpoint
monitor.csv
hparams.txt
plot
evaluation.csv
```

Ini penting untuk paper/LinkedIn karena kamu butuh bukti eksperimen, bukan cuma cerita.

## Peta file cepat

```text
README.md
  Cara install, build, run, train, plot, evaluate.

docs/CODE_WALKTHROUGH.md
  Dokumentasi English untuk repo/public.

docs/CODE_EXPLANATION_ID.md
  Dokumen ini. Penjelasan belajar dalam bahasa Indonesia.

src/JunimoKartRLBridge/manifest.json
  Identitas SMAPI mod.

src/JunimoKartRLBridge/JunimoKartRLBridge.csproj
  Konfigurasi build .NET mod.

src/JunimoKartRLBridge/Config.cs
  Setting bridge dan snapshot.

src/JunimoKartRLBridge/Protocol.cs
  Bentuk JSON request/response.

src/JunimoKartRLBridge/BridgeServer.cs
  TCP server localhost.

src/JunimoKartRLBridge/ReflectionUtil.cs
  Helper baca/tulis field private Stardew.

src/JunimoKartRLBridge/ModEntry.cs
  Main logic mod: start game, apply jump, create snapshot.

junimo_rl/client.py
  TCP client Python.

junimo_rl/env.py
  Gymnasium environment + observation vector + reward.

scripts/smoke_test.py
  Test bridge dan jump.

scripts/train_dqn.py
  Training DQN dan checkpoint.

scripts/train_ppo.py
  Training PPO dengan folder output terpisah.

scripts/plot_training.py
  Plot reward/length dari monitor.csv.

scripts/evaluate_models.py
  Compare checkpoint DQN secara deterministic.

scripts/evaluate_ppo_models.py
  Compare checkpoint PPO secara deterministic.

pyproject.toml
  Metadata package dan dependency Python.
```

## Checklist belajar ulang coding dari project ini

Kalau kamu ingin mendalami pelan-pelan, urutan belajarnya menurutku begini:

1. Baca `junimo_rl/client.py`.
   - Fokus: socket, JSON, request/response.

2. Baca `scripts/smoke_test.py`.
   - Fokus: cara client dipakai.

3. Baca `junimo_rl/env.py`.
   - Fokus: `reset`, `step`, `snapshot_to_vector`, `_reward`.

4. Baca `scripts/train_dqn.py`.
   - Fokus: argparse, DQN, callback, logging.

5. Baca `scripts/train_ppo.py`.
   - Fokus: perbedaan PPO vs DQN, folder output, dan hyperparameter PPO.

6. Baca `Protocol.cs`.
   - Fokus: data shape dari C# ke Python.

7. Baca `BridgeServer.cs`.
   - Fokus: TCP server.

8. Baca `ModEntry.cs`.
   - Fokus: game loop, start minigame, apply jump, snapshot.

9. Baca `ReflectionUtil.cs`.
   - Fokus: reflection dan kenapa butuh private field access.

Kalau kamu cuma punya waktu 30 menit:

```text
client.py -> env.py -> train_dqn.py -> train_ppo.py
```

Kalau kamu mau benar-benar paham project:

```text
Protocol.cs -> ModEntry.cs -> env.py -> train_dqn.py -> train_ppo.py
```

## Ringkasan super pendek

Project ini bukan sekadar "bot main game".

Secara engineering, project ini membuat:

```text
Stardew Valley internal state
    -> C# SMAPI bridge
    -> TCP JSON protocol
    -> Python Gymnasium environment
    -> DQN/PPO model
    -> action jump
    -> kembali ke Stardew Valley
```

Secara machine learning, project ini mengajarkan:

```text
Agent tidak langsung pintar hanya karena pakai neural network.
State representation, reward design, exploration, dan evaluation jauh lebih penting daripada sekadar menjalankan model.learn().
```

Dan secara belajar coding, project ini bagus karena mencakup banyak konsep nyata:

```text
C# modding
Python packaging
socket programming
JSON protocol
Gymnasium environment design
Stable-Baselines3 training
logging
checkpointing
evaluation
experiment analysis
```

Kalau kamu merasa "kok banyak banget ya", itu bukan kamu yang lambat. Project ini memang lintas domain. Justru bagus dijadikan bahan LinkedIn/paper karena problemnya nyata, messy, dan penuh tradeoff - bukan tutorial steril.

## Update eksperimen: PPO macro 6 + `shaped_v3`

Setelah video dan log menunjukkan PPO macro `shaped_v2` masih plateau, project ini ditambah satu eksperimen diagnosis baru:

- reward version baru: `shaped_v3`
- macro action lebih halus: `--macro-action-frames 6`
- score/coin/fruit reward dimatikan default untuk fokus survival
- gap landing reward harus terkonfirmasi beberapa step
- `monitor.csv` sekarang punya telemetry action/gap/death

Dokumen detailnya ada di:

```text
docs/EXPERIMENT_SHAPED_V3_ID.md
```

Command launcher:

```powershell
.\scripts\run_ppo_macro6_v3.ps1
```

## Update eksperimen: PPO-LSTM macro 6 + `shaped_v3`

Project ini juga punya eksperimen PPO-LSTM yang memakai environment/reward/action sama seperti v3, tetapi policy-nya punya memory LSTM.

Tujuannya untuk menguji apakah Junimo Kart memang membutuhkan temporal memory, misalnya untuk memahami trajectory lompat dan durasi hold jump sebelumnya.

Command launcher:

```powershell
.\scripts\run_ppo_lstm_macro6_v3.ps1
```

Dokumen detail:

```text
docs/EXPERIMENT_PPO_LSTM_ID.md
```

## Update eksperimen: V4/V4b temporal features

Setelah hasil v3 dan PPO-LSTM awal masih belum jauh berbeda, issue paling kuat bukan sekadar hyperparameter.

Yang terlihat:

- koordinat semantic dari video manual cukup masuk akal;
- agent masih sering mati dekat gap;
- action `macro` lama membuat model mudah terlalu sering memilih long hold;
- model belum diberi fitur eksplisit tentang durasi hold/airborne/grounded.
- gap lama dihitung dari posisi track, belum dari edge/collision bounds track.

Karena itu v4 menambahkan:

```text
action-mode tap_macro
temporal-features
trace_policy_rollout.py
track.bounds dari SMAPI bridge
```

Catatan koreksi: setelah dicoba, `tap_macro` bisa memotong long jump karena release di tengah udara tidak selalu bisa disambung lagi oleh press berikutnya. Karena itu rekomendasi utama sekarang adalah v4b:

```text
action-mode macro + temporal-features
```

Dengan ini action 3 tetap bisa menjadi long hold kontinu, tetapi model diberi sensor durasi seperti `jump_held_steps` dan `airborne_steps` supaya bisa belajar kapan harus release.

`temporal-features` menambahkan lima sensor waktu:

```text
jump_held_steps
airborne_steps
grounded_steps
last_action
last_action_holds_jump
```

Script trace baru:

```text
scripts/trace_policy_rollout.py
```

berguna untuk menganalisis model by-step: ketika gap muncul, action apa yang dipilih, apakah grounded, velocity berapa, dan reward apa yang diterima.

Bridge C# juga sekarang mengirim `track.bounds`. Kalau data bounds tersedia, Python menghitung gap dari edge track, bukan hanya dari posisi/anchor track. Ini membuat `gap_start_dx`, `gap_width`, dan `landing_y` lebih dekat ke kondisi fisik di game.

Command launcher:

```powershell
.\scripts\run_ppo_macro_temporal_v4b.ps1
.\scripts\run_ppo_lstm_macro_temporal_v4b.ps1
```

Dokumen detail:

```text
docs/EXPERIMENT_V4_ACTION_OBSERVATION_ID.md
```

## Update eksperimen: MultiInput V6

Setelah melihat pendekatan PWhiddy/PokemonRedExperiments, project ini menambahkan observation mode baru:

```text
--observation-mode multi
```

Mode ini membuat observation menjadi dictionary, bukan satu vector besar.

Isinya:

```text
state          = raw internal vector lama
semantic       = gap/landing/obstacle/pickup features
temporal       = durasi jump held, airborne, grounded, last action
recent_actions = one-hot memory action beberapa step terakhir
spatial        = grid visual-like dari track/obstacle/pickup/player
```

`spatial` bukan screenshot asli. Itu peta kecil dari koordinat internal game. Tujuannya memberi model bentuk visual gap/track tanpa harus menangkap layar Stardew secara live.

Kalau memakai PPO biasa, script otomatis memilih:

```text
MultiInputPolicy
```

Kalau memakai PPO-LSTM, script otomatis memilih:

```text
MultiInputLstmPolicy
```

Command launcher:

```powershell
.\scripts\run_ppo_multiinput_binary_v6.ps1
.\scripts\run_ppo_lstm_multiinput_binary_v6.ps1
```

Launcher continuous:

```powershell
.\scripts\run_ppo_multiinput_binary_v6_forever.ps1
.\scripts\run_ppo_lstm_multiinput_binary_v6_forever.ps1
```

Launcher continuous memakai `--timesteps 2147483647` dan tidak mengirim `--episodes`, jadi training tidak berhenti di episode tertentu. Checkpoint tetap disimpan tiap 1000 episode. Script ini juga mencoba mencari checkpoint v6 terakhir dari folder log, mengambil nomor episode dari nama file, lalu memasukkannya sebagai `--episode-offset` agar checkpoint lanjut dari nomor sebenarnya.

`scripts/train_ppo.py` dan `scripts/train_ppo_lstm.py` juga sekarang menangani `Ctrl+C` lebih aman. Kalau training dihentikan manual, model saat itu tetap disimpan ke `--model-path` sebelum proses keluar.

### Monitoring state/telemetry table

File:

```text
scripts/watch_monitor_table.py
```

Script ini tidak membuka koneksi ke SMAPI bridge. Dia hanya membaca `monitor.csv`, jadi aman dijalankan saat training sedang memakai bridge.

Yang ditampilkan adalah ringkasan state/telemetry per window episode:

```text
rew_mean          = rata-rata reward
len_mean          = rata-rata episode length
best_len          = episode length terbaik dalam window
max_x_mean        = rata-rata posisi X terjauh
gap_att_ep        = rata-rata gap attempt per episode
gap_land_rate     = rasio gap landing sukses / gap attempt
death_gap_rate    = rasio episode mati dekat gap
death_obs_rate    = rasio episode mati dekat obstacle
pickup_ep         = rata-rata pickup events
score_mean        = rata-rata score delta total
a0_pct..a3_pct    = distribusi action
final_gap_dx      = rata-rata jarak gap terakhir saat episode berakhir
final_gap_w       = rata-rata width gap terakhir saat episode berakhir
final_obs_dx      = rata-rata jarak obstacle terakhir saat episode berakhir
```

Training run baru juga menulis state telemetry tambahan ke `monitor.csv`:

```text
state_samples            = jumlah transition/state sample dalam episode
gap_visible_steps        = berapa step gap terlihat di depan
gap_near_steps           = berapa step gap valid berada di zona lompat
obstacle_visible_steps   = berapa step obstacle terlihat di depan
obstacle_near_steps      = berapa step obstacle dekat
pickup_visible_steps     = berapa step pickup terlihat
grounded_steps_total     = berapa step cart grounded
jump_held_steps_total    = berapa step tombol jump sedang held
sum_gap_start_dx         = akumulasi gap_dx saat gap terlihat
sum_gap_width            = akumulasi gap width saat gap terlihat
sum_landing_delta_y      = akumulasi beda tinggi landing saat gap terlihat
sum_obstacle_dx          = akumulasi obstacle_dx saat obstacle terlihat
sum_pickup_dx            = akumulasi pickup_dx saat pickup terlihat
```

`watch_monitor_table.py` mengubah raw counter ini menjadi kolom tabel seperti `grounded_pct`, `jump_held_pct`, `gap_visible_pct`, `gap_near_pct`, `gap_dx_mean`, `gap_w_mean`, `landing_dy_mean`, `obs_near_pct`, dan `pickup_visible_pct`.

Command:

```powershell
python .\scripts\watch_monitor_table.py --latest --watch --every-episodes 100 --history 10
```

Kalau ingin tabel hanya setiap 100 episode penuh, tanpa partial episode window:

```powershell
python .\scripts\watch_monitor_table.py --latest --watch --every-episodes 100 --history 10 --no-partial
```

### Per-step state trace saat training

File:

```text
junimo_rl/state_trace.py
scripts/run_ppo_lstm_multiinput_binary_v6_trace_debug.ps1
```

`state_trace.py` adalah callback Stable-Baselines yang membaca `info["snapshot"]` dari environment setiap step. Karena snapshot ini berasal dari env training yang sama, script ini tidak membuka koneksi bridge kedua.

Kolom yang ditulis:

```text
timestep, episode, episode_step, action, action_holds_jump,
step_reward, reward, done, x, y, vx, vy, grounded, jump_held, jumping,
gap_present, gap_dx, gap_width, landing_y, landing_dy,
obstacle_present, obstacle_dx, obstacle_y,
pickup_present, pickup_dx, pickup_y,
progress, score, lives_left, levels_beat
```

Catatan:

```text
step_reward = reward pada step itu saja
reward      = reward kumulatif episode sampai step tersebut
```

Flag training:

```powershell
--trace-state-print-freq 1
--trace-state-format simple
--trace-state-simple-action
--trace-state-csv logs\...\state_trace.csv
--trace-state-csv-freq 1
--trace-state-max-rows 0
```

Format simple menghasilkan log seperti:

```text
x: 2, y: 1, reward: 4.235, score: 170, generation: 28
```

Untuk debug no-jump, launcher trace memakai `--trace-state-simple-action`, jadi output menjadi:

```text
x: 2, y: 1, action: 1, hold: 1, reward: 4.235, score: 170, generation: 28
```

Ini membedakan dua kasus:

```text
action: 0, hold: 0 = model memang memilih tidak lompat
action: 1, hold: 1 = model mengirim hold jump; kalau di game tidak terlihat lompat, masalahnya bukan policy output
```

Pada format simple:

```text
x = bin jarak horizontal ke target terdekat
    prioritas target: gap, obstacle, pickup
y = bin beda tinggi target
    untuk gap: landing_delta_y / 16
reward = reward kumulatif episode sampai step itu
         reward step mentah tetap ada di CSV sebagai step_reward
score = score game saat itu
generation = episode training
```

Default simple memakai bin diskrit agar mirip contoh Flappy Bird. Jika ingin angka raw:

```powershell
--trace-state-simple-raw
```

Contoh debug:

```powershell
.\scripts\run_ppo_lstm_multiinput_binary_v6_trace_debug.ps1
```

Untuk menghindari terminal terlalu berat, bisa print tiap 5 atau 10 step:

```powershell
.\scripts\run_ppo_lstm_multiinput_binary_v6_trace_debug.ps1 -TracePrintFreq 10
```

Launcher trace debug memakai `n_steps=256` dan `n_epochs=3` agar jeda update PPO-LSTM lebih pendek daripada training normal `n_steps=1024`.

## Update eksperimen: `shaped_v4` positive reward

Setelah melihat trace per-step, penalti kecil pada `shaped_v3` membuat reward tampak turun walaupun action agent masih terlihat masuk akal. Karena itu ditambahkan reward version baru:

```text
shaped_v4
```

Tujuan `shaped_v4`:

```text
1. menghapus penalti kecil untuk jump/no-jump;
2. mempertahankan death penalty;
3. menghapus bonus life positif saat reset episode;
4. membuat gap landing reward lebih besar dan lebih terlihat;
5. menambahkan reward event ke trace.
```

Perbedaan utama dari `shaped_v3`:

```text
shaped_v3:
- penalti jika hold jump di rel normal tanpa gap/obstacle
- penalti jika tidak hold saat gap sangat dekat
- bonus kecil jika hold dekat gap/obstacle
- gap landing sekitar 3.0 + 0.02 * gap_width

shaped_v4:
- tidak ada penalti kecil action timing
- tidak ada bonus kecil action timing
- progress tetap diberi reward
- death tetap negatif
- gap landing default sekitar 8.0 + 0.04 * gap_width
- positive life_delta tidak diberi reward agar reset episode tidak membuat reward spike palsu
```

Trace sekarang memiliki kolom/event:

```text
progress
score_delta
gap_landing
gap_death
death
life_lost
completed
not_in_minigame
```

Contoh log simple:

```text
x: 2, y: -2, action: 1, hold: 1, reward: 11.84, event: gap_landing, score: 180, generation: 12
x: 0, y: -2, action: 0, hold: 0, reward: -114.99, event: gap_death, score: 320, generation: 12
```

Launcher debug:

```powershell
.\scripts\run_ppo_lstm_multiinput_binary_v7_positive_trace_debug.ps1
```

Launcher training:

```powershell
.\scripts\run_ppo_lstm_multiinput_binary_v7_positive.ps1
```

Keduanya memakai observation/action yang sama dengan v6:

```text
observation_mode = multi
action_mode = binary
```

Secara default launcher v7 akan mencoba load model:

```text
models\ppo_lstm\junimo_ppo_lstm_multiinput_binary_v6.zip
```

Jadi training v7 tidak harus mulai dari nol selama file v6 itu tersedia.

### Parameter reward berhasil mendarat setelah jurang

Reward `gap_landing` sekarang bisa diatur dari command, tanpa perlu edit langsung di `junimo_rl/env.py`.

Parameter CLI:

```powershell
--gap-landing-base-reward
--gap-landing-width-coef
```

Parameter PowerShell launcher v7:

```powershell
-GapLandingBaseReward
-GapLandingWidthCoef
```

Rumusnya:

```text
gap_landing_reward = GapLandingBaseReward + GapLandingWidthCoef * min(gap_width, 120)
```

Default:

```text
GapLandingBaseReward = 8.0
GapLandingWidthCoef  = 0.04
```

Contoh jika `gap_width = 96`:

```text
8.0 + 0.04 * 96 = 11.84
```

Kalau ingin reward mendarat setelah jurang lebih besar:

```powershell
.\scripts\run_ppo_lstm_multiinput_binary_v7_positive.ps1 -GapLandingBaseReward 12 -GapLandingWidthCoef 0.05
```

Dengan `gap_width = 96`, reward-nya menjadi:

```text
12 + 0.05 * 96 = 16.8
```

Catatan penting:

```text
- Reward ini hanya keluar kalau environment mendeteksi agent sempat melewati gap, lalu grounded lagi di track berikutnya.
- Lompat-lompat di rel yang masih tersambung tidak mendapat reward gap_landing.
- Gap width dibatasi dengan min(gap_width, 120) supaya jurang yang sangat lebar tidak membuat reward terlalu eksplosif.
```

### Split reward untuk coin dan fruit

Reward pickup sekarang bisa dipisah:

```powershell
--coin-reward-coef
--fruit-reward-coef
--fruit-score-threshold
```

Di launcher v7 PowerShell:

```powershell
-CoinRewardCoef
-FruitRewardCoef
-FruitScoreThreshold
```

Default launcher v7:

```text
CoinRewardCoef  = 0.0005
FruitRewardCoef = 0.003
```

Artinya fruit diberi bobot 6x coin. Contoh:

```text
coin score_delta 10  -> 10 * 0.0005 = 0.005
fruit score_delta 100 -> 100 * 0.003 = 0.300
```

Kenapa tidak langsung pakai `score_delta` untuk semua score? Karena dari trace terlihat score bisa naik sangat sering. Kalau semua kena reward, agent bisa belajar mengejar score/progress kecil, bukan pickup. Karena itu jika `CoinRewardCoef` atau `FruitRewardCoef` diset, Python masuk mode split pickup:

```text
1. cek entity pickup dekat cart dari snapshot lama/baru;
2. jika type mengandung "fruit", pakai fruit coefficient;
3. jika type mengandung "coin" atau "gem", pakai coin coefficient;
4. jika type tidak jelas tetapi score_delta >= FruitScoreThreshold, fallback sebagai fruit;
5. kalau tidak ada pickup dekat cart, tidak ada reward pickup.
```

Trace event baru:

```text
event: coin
event: fruit
```

Contoh command:

```powershell
.\scripts\run_ppo_lstm_multiinput_binary_v7_positive.ps1 -CoinRewardCoef 0.0005 -FruitRewardCoef 0.003 -TracePrintFreq 1
```

Kalau fruit masih kurang menarik:

```powershell
.\scripts\run_ppo_lstm_multiinput_binary_v7_positive.ps1 -CoinRewardCoef 0.0005 -FruitRewardCoef 0.005
```

Dokumen detail:

```text
docs/EXPERIMENT_MULTIINPUT_V6_ID.md
```

## Launcher lengkap semua parameter PPO-LSTM v7

File:

```text
scripts/run_ppo_lstm_v7_full_params.ps1
```

Tujuan file ini adalah menyediakan satu launcher PowerShell yang mengekspos hampir seluruh argumen `scripts/train_ppo_lstm.py`, sehingga eksperimen bisa diubah tanpa mengedit Python.

Default launcher mengikuti eksperimen terakhir:

```text
algorithm          = PPO-LSTM
observation_mode   = multi
semantic_features  = true
temporal_features  = true
reward_version     = shaped_v4
action_mode        = binary
coin_reward_coef   = 0.0005
fruit_reward_coef  = 0.005
gap_landing_reward = 12.0 + 0.05 * min(gap_width, 120)
```

Contoh run standar:

```powershell
.\scripts\run_ppo_lstm_v7_full_params.ps1
```

Contoh cek command tanpa menjalankan training:

```powershell
.\scripts\run_ppo_lstm_v7_full_params.ps1 -DryRun
```

Contoh training berdasarkan timesteps saja, tanpa batas episode:

```powershell
.\scripts\run_ppo_lstm_v7_full_params.ps1 -Episodes -1 -Timesteps 1000000
```

Contoh run dengan trace terminal setiap step:

```powershell
.\scripts\run_ppo_lstm_v7_full_params.ps1 -TracePrintFreq 1
```

Contoh run dengan trace CSV:

```powershell
.\scripts\run_ppo_lstm_v7_full_params.ps1 -TraceCsv "logs\ppo_lstm\v7_full_state_trace.csv" -TraceCsvFreq 1
```

Contoh mengubah reward:

```powershell
.\scripts\run_ppo_lstm_v7_full_params.ps1 -CoinRewardCoef 0.0005 -FruitRewardCoef 0.005 -GapLandingBaseReward 12 -GapLandingWidthCoef 0.05
```

Contoh mulai dari nol:

```powershell
.\scripts\run_ppo_lstm_v7_full_params.ps1 -LoadModel ""
```

Contoh lanjut dari checkpoint tertentu:

```powershell
.\scripts\run_ppo_lstm_v7_full_params.ps1 -LoadModel "logs\ppo_lstm\nama_run\checkpoints\nama_checkpoint.zip" -EpisodeOffset 5000
```

Parameter dikelompokkan di script sebagai:

```text
1. Bridge/game connection
2. Training duration
3. Save/load paths
4. Checkpointing/progress
5. Live state trace
6. Environment/observation
7. Reward/action design
8. PPO hyperparameters
9. LSTM architecture
```

## Compact agent v8

Audit v7 menemukan input 4.448 nilai, model PPO-LSTM sekitar 4,7 juta parameter, bridge tick di observation, score yang keliru dianggap pickup, serta death penalty yang dihitung berulang. Implementasi penggantinya dijelaskan lengkap di:

```text
docs/EXPERIMENT_COMPACT_V8_ID.md
```

Ringkasnya, v8 menambahkan:

```text
observation_mode = compact (27 fitur bounded)
reward_version   = shaped_v5
action_mode      = binary, frame_skip=1
PPO parameters   = sekitar 12 ribu
LSTM parameters  = sekitar 64 ribu
```

Launcher utama:

```powershell
.\scripts\run_ppo_compact_v8.ps1
.\scripts\run_ppo_lstm_compact_v8.ps1
```

## Addendum: reward `shaped_v6`

Setelah compact PPO mencapai episode 1.500, evaluasi menunjukkan gap landing membaik tetapi perilaku masih sering membentuk long jump berulang. `shaped_v5` sengaja tidak diubah. Reward baru ditambahkan sebagai versi terpisah agar eksperimen lama tetap reproducible.

Alur `shaped_v6` di `JunimoKartEnv.step()` tetap:

```text
old snapshot
-> apply binary action
-> new snapshot
-> calculate shaped_v5 components
-> subtract one real jump-start cost
-> if death during active gap, subtract bounded miss-distance cost
-> update telemetry and monitor.csv
```

Deteksi jump baru memakai empat syarat:

```python
holds_jump
and not previous_holds
and grounded
and jump_ready
```

Karena itu action `1` berulang selama satu long jump tidak dikenai penalti start berulang. Airborne re-press juga tidak dikenai biaya karena bridge tidak menjalankan input tersebut sebagai jump baru.

Gap attempt menyimpan:

```text
start_x
end_x / landing_start_x
landing_end_x
width
furthest_x
```

Jarak miss dihitung terhadap seluruh interval `[landing_start_x, landing_end_x]`, lalu dinormalisasi terhadap lebar gap. Formula lengkap, telemetry, launcher, dan desain ablation dijelaskan di:

```text
docs/EXPERIMENT_SHAPED_V6_ID.md
```

Next run:

```powershell
.\scripts\run_ppo_compact_shaped_v6.ps1
```

Evaluasi otomatis setelah 1.500 episode tambahan tidak menghubungi bridge saat menunggu. Script memantau PID training, memverifikasi checkpoint total episode 3.000, lalu menjalankan evaluator untuk baseline 1.500 dan seluruh checkpoint v6:

```powershell
.\scripts\watch_and_evaluate_ppo_shaped_v6.ps1 -TrainingPid <PID>
```

## Addendum: current-track-anchored gap detector v9

Evaluasi `shaped_v6` menemukan bahwa masalahnya bukan hanya bobot reward. Fungsi lama `semantic_feature_snapshot()` memasangkan semua rel berdasarkan urutan X. Pada stage bertingkat, pasangan tersebut belum tentu merupakan rel yang sedang dipakai cart.

Implementasi baru menambahkan parameter:

```text
gap_detection_mode = legacy | anchored
```

Default tetap `legacy` untuk menjaga reproducibility. Launcher v9 secara eksplisit memilih `anchored`.

Fungsi penting yang ditambahkan di `junimo_rl/env.py`:

```text
_supporting_track()
    memilih rel yang benar-benar menopang cart grounded

_extend_connected_track_run()
    mengikuti tile rel yang masih tersambung sampai ujung jalur

_anchored_gap_geometry()
    mencari landing pertama dari ujung jalur tersebut dan menghitung interval landing

GapGeometryTracker
    menyimpan gap_start_x, gap_end_x, landing_end_x, landing_y, takeoff_y
    selama cart airborne
```

`JunimoKartEnv` memiliki satu `GapGeometryTracker`. Pada `reset()`, tracker dikosongkan lalu diisi dari snapshot pertama. Pada setiap `step()`:

```text
old snapshot + tracked geometry
-> action
-> new snapshot
-> reward dan telemetry memakai target yang konsisten
-> tracker diperbarui dari new snapshot
-> observation dan info mengirim semantic hasil tracker
```

`compact_feature_vector()` sekarang dapat menerima semantic snapshot yang sudah dihitung environment. Ini penting agar fungsi tersebut tidak menghitung ulang jurang memakai mode legacy. Bentuk outputnya tetap `(27,)`.

`info["semantic"]` juga dikirim oleh environment. `StateTraceCallback` dan `trace_policy_rollout.py` memakai nilai ini supaya CSV debug mencatat fitur yang benar-benar dilihat model, bukan hasil perhitungan ulang yang berbeda.

Semua entry point PPO/PPO-LSTM/evaluator/trace mendapat argumen `--gap-detection-mode`. Detail algoritma, batas geometri, pengujian, dan command eksperimen terdapat di:

```text
docs/EXPERIMENT_ANCHORED_GAP_V9_ID.md
```

Launcher utama:

```powershell
.\scripts\run_ppo_compact_anchored_v9.ps1
.\scripts\evaluate_ppo_compact_anchored_v9.ps1 -Episodes 20
```

## Addendum: `shaped_v7` dan anti-long-jump v10

Analisis v9 menunjukkan biaya jump-start tidak membedakan short dan long jump. Keduanya hanya membayar sekali ketika action berubah dari release menjadi hold saat grounded. Karena itu ditambahkan `JunimoKartEnv._shaped_v7_reward()`.

Fungsi tersebut memanggil `_shaped_v6_reward()` terlebih dahulu agar progress, pickup, landing, death, jump-start, dan miss-distance tetap identik. Setelah itu fungsi membaca:

```text
old.player.grounded
current action holds jump atau tidak
temporal_state.jump_held_steps
```

Jika cart airborne dan terus hold melewati `airborne_hold_free_steps`, reward dikurangi `airborne_hold_penalty` pada setiap step. Komponen ini dicatat sebagai `reward_components["airborne_hold"]` agar terlihat pada trace CSV.

Telemetry baru:

```text
airborne_hold_penalty_steps
airborne_hold_penalty_total
max_jump_hold_steps
```

Semua script train/evaluate PPO dan PPO-LSTM menerima parameter:

```text
--airborne-hold-free-steps
--airborne-hold-penalty
```

Launcher fresh PPO:

```powershell
.\scripts\run_ppo_compact_anchored_v10.ps1
```

Penjelasan formula, diagnosis data v9, alasan policy fresh, dan evaluasi terdapat di:

```text
docs/EXPERIMENT_ANTI_LONG_JUMP_V10_ID.md
```

## Addendum: `shaped_v8` dan eksperimen teknik tip v11

Gameplay dan monitor v10 menunjukkan policy semakin jarang grounded walaupun reward dan gap landing meningkat. Ini berarti agent mengeksploitasi reward dengan long-jump spam. `JunimoKartEnv._shaped_v8_reward()` menambahkan shaping yang membedakan progress grounded dari airtime yang tidak relevan.

Takeoff dekat ujung rel tidak langsung mendapat reward. `_record_gap_takeoff()` hanya menyimpan jarak sisi depan cart ke `active_gap.start_x` beserta kualitas kontinu. Ketika `_gap_landing_reward()` mendeteksi landing, `_prepare_gap_tip_technique()` menghitung kualitas kedalaman landing. `_pay_gap_landing()` baru membayar dan mencatat bonus setelah confirmation selesai.

Formula bonus teknik:

```text
takeoff_quality = clip(1 - abs(takeoff_distance - target) / tolerance, 0, 1)
landing_quality = clip(1 - abs(landing_depth - target) / tolerance, 0, 1)
tip_reward = max_tip_reward * sqrt(takeoff_quality * landing_quality)
```

Dengan geometric mean, takeoff dan landing harus sama-sama baik. Jump bagus yang berakhir mati tidak mendapatkan bonus.

Komponen tambahan v11:

```text
grounded_progress
unnecessary_jump
non_gap_airborne
gap_tip_technique
```

Launcher dan dokumen eksperimen:

```powershell
.\scripts\run_ppo_compact_tip_v11.ps1
.\scripts\evaluate_ppo_compact_tip_v11.ps1
```

```text
docs/EXPERIMENT_TIP_TECHNIQUE_V11_ID.md
```

## Addendum: `shaped_v9` dan dynamic-takeoff v12

V11 memperlihatkan policy oscillation: sempat mengalami no-jump collapse karena fixed takeoff target 12 px, grounded bonus yang tetap aktif dekat gap, dan penalti langsung untuk jump di luar tip zone; kemudian pulih dengan hold panjang. V12 memperbaiki kedua ekstrem tersebut melalui `JunimoKartEnv._shaped_v9_reward()`.

Dynamic takeoff target dihitung oleh `_dynamic_takeoff_target()`:

```text
target = clip(
    0.65 * gap_width
    + 0.25 * uphill_height
    - 0.10 * downhill_height,
    32,
    112,
)
```

`_record_gap_takeoff()` memakai target tersebut saat reward version adalah `shaped_v9`. Takeoff distance, target, dan kualitas disimpan pada active gap. Bonus teknik tetap dibayar oleh `_pay_gap_landing()` hanya setelah successful confirmed landing.

`_shaped_v9_reward()` memanggil `_shaped_v7_reward()` untuk mempertahankan progress, pickup, death, gap landing, miss-distance, jump start, dan hold-duration. Setelah itu fungsi:

```text
memberi grounded bonus hanya di rel aman
memberi small inaction cost jika gap sudah mendesak
menganggap seluruh active-gap airtime valid setelah takeoff tercatat
tidak memberi fixed-tip unnecessary-jump penalty
```

Parameter PPO v12 juga dibuat lebih konservatif dan exploratory: learning rate `0.0002`, `n_steps=2048`, batch size `128`, dan entropy coefficient `0.02`.

Launcher utama dan evaluasi:

```powershell
.\scripts\run_ppo_compact_dynamic_v12.ps1
.\scripts\evaluate_ppo_compact_dynamic_v12.ps1
```

Penjelasan formula, diagnosis, telemetry, dan kriteria evaluasi lengkap:

```text
docs/EXPERIMENT_DYNAMIC_TAKEOFF_V12_ID.md
```
