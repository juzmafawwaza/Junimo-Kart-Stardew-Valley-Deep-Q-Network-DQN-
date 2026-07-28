# Code Walkthrough — Junimo Kart RL Bridge

This document explains the code in this project in detail. The purpose is to avoid "vibe coding": every file should have a reason, the data flow should be understandable, and future changes should be documented.

Whenever code is added or changed, this document should be updated with:

1. which files changed,
2. what the change does,
3. which data flow or logic it affects,
4. how to run or test it,
5. caveats and risks.

## High-level architecture

The project has two main parts:

1. a C# SMAPI mod running inside Stardew Valley;
2. a Python reinforcement learning environment running outside the game.

Data flow:

```text
Python DQN agent
  -> chooses action: release jump / hold jump
  -> sends JSON to the local TCP bridge
  -> SMAPI mod receives the action
  -> mod changes Junimo Kart's internal jump state
  -> game advances for a few frames
  -> mod reads Junimo Kart's internal state
  -> Python receives an observation and computes reward
  -> DQN learns from the transition
```

The bridge listens on:

```text
127.0.0.1:8765
```

The protocol is JSON-lines: one JSON object per line.

## Important folders

```text
src/JunimoKartRLBridge/   C# SMAPI mod
junimo_rl/                Python TCP client and Gymnasium environment
scripts/                  Smoke test, training, plotting, and evaluation scripts
docs/                     Detailed project documentation
logs/                     Training outputs, ignored by Git
models/                   Saved models, ignored by Git
```

Generated files such as `logs/`, `models/`, `bin/`, `obj/`, `.scratch/`, `__pycache__/`, and `*.egg-info/` are ignored by Git. The GitHub repository contains source code, scripts, configuration, and documentation, not trained models or build artifacts.

## C# SMAPI mod

### `src/JunimoKartRLBridge/manifest.json`

This is the SMAPI manifest. SMAPI uses it to discover and load the mod.

Important fields:

- `Name`: display name of the mod.
- `UniqueID`: unique mod ID.
- `EntryDll`: DLL loaded by SMAPI.
- `MinimumApiVersion`: minimum required SMAPI version.

If this file is missing or invalid, SMAPI will not load the bridge.

### `src/JunimoKartRLBridge/JunimoKartRLBridge.csproj`

This is the .NET project file for the mod.

Important details:

- Target framework is `net6.0`, matching Stardew Valley 1.6.
- `Pathoschild.Stardew.ModBuildConfig` handles SMAPI mod build/deploy behavior.
- `GamePath` points to the local Stardew Valley installation.

Build command:

```powershell
dotnet build .\src\JunimoKartRLBridge\JunimoKartRLBridge.csproj -c Release
```

If Stardew Valley is running, the DLL in the Mods folder may be locked. Close the game before building if deployment fails.

### `src/JunimoKartRLBridge/Config.cs`

Configuration for the mod.

Fields:

- `BindAddress`: TCP bind address, default `127.0.0.1`.
- `Port`: TCP port, default `8765`.
- `StartServerOnLaunch`: starts the bridge automatically when the mod loads.
- `MaxTracks`: number of track entries sent to Python.
- `MaxEntities`: number of entity entries sent to Python.
- `LookaheadPixels`: how far ahead of the cart the snapshot should include objects.
- `LookbehindPixels`: how far behind the cart to keep context.
- `AutoAdvanceTitleAfterStart`: automatically moves past the Junimo Kart title state.
- `AutoContinueProgressModeNonGameplayStates`: automatically skips non-gameplay states such as Title, Map, and Cutscene during Progress Mode.
- `ForceRunWhenUnfocused`: forces Stardew's `pauseWhenOutOfFocus` option to `false`, so training continues when the game window is not focused.

### `src/JunimoKartRLBridge/BridgeServer.cs`

This file implements the TCP server.

Responsibilities:

- listen on `127.0.0.1:8765`;
- accept a Python client connection;
- read JSON-lines requests;
- deserialize them into `ClientRequest`;
- call `ModEntry.HandleBridgeRequest`;
- write JSON-lines responses back to Python.

The TCP server runs on a background thread. It does not directly mutate Stardew game state. Instead, it sets pending values that are later processed on the main Stardew update tick. This avoids touching game state from the wrong thread.

### `src/JunimoKartRLBridge/Protocol.cs`

This file contains DTO classes used by the JSON protocol.

Important classes:

- `ClientRequest`
  - `Type`: `ping`, `state`, `start`, `reset`, `action`, or `advance`.
  - `Mode`: `progress`, `endless`, or `infinite`.
  - `Jump`: `true` means hold jump, `false` means release jump.

- `BridgeResponse`
  - `Ok`: whether the request succeeded.
  - `Type`: response type.
  - `Message`: optional message.
  - `State`: latest game snapshot.

- `BridgeSnapshot`
  - score, lives, level, game state, player state, tracks ahead, entities ahead.

- `PlayerSnapshot`
  - player position, velocity, bounds, grounded/jumping state.

- `TrackSnapshot`
  - track position, distance from player, track type, obstacle information.

- `EntitySnapshot`
  - pickups, obstacles, coins, fruit, and other entities near the player.

### `src/JunimoKartRLBridge/ReflectionUtil.cs`

This helper reads and writes private Stardew Valley fields and methods.

Why reflection is needed:

Junimo Kart stores important information in non-public fields, such as:

- `player`
- `_tracks`
- `_entities`
- `isJumpPressed`
- `gameState`
- `levelsBeat`

Normal C# code cannot directly access those fields. Reflection allows the mod to inspect them at runtime. Reflection lookups are cached for performance.

Important functions:

- `Field(target, name)`: reads a field.
- `Field<T>(target, name)`: reads a field and converts it to a type.
- `SetField(target, name, value)`: writes a field.
- `Invoke(target, name, args)`: calls a method.
- `BoolMethod(target, name)`: calls a method returning a boolean.
- `VectorField(target, name)`: reads an XNA `Vector2`.
- `Bounds(target)`: calls `GetBounds()` and converts the result.
- `Enumerate(value)`: flattens lists and dictionaries.
- `InheritsTypeName(value, typeName)`: checks inheritance by type name.

The jump-control fix depends on these internal calls:

```text
SetField(mineCart, "isJumpPressed", desiredJump)
Invoke(player, "QueueJump")
Invoke(player, "ReleaseJump")
```

The earlier approach used `receiveKeyPress(Keys.Space)`, but Junimo Kart does not use that method for jump input. The fixed version changes the internal jump state directly.

### `src/JunimoKartRLBridge/ModEntry.cs`

This is the main mod file.

Responsibilities:

1. load config;
2. start the TCP bridge;
3. handle Python requests;
4. start Junimo Kart Progress Mode;
5. auto-continue non-gameplay states;
6. keep the game running while unfocused;
7. apply jump actions;
8. build observations.

#### Entry point

`Entry(IModHelper helper)` runs when SMAPI loads the mod.

It:

- reads config;
- registers `GameLoop.UpdateTicked`;
- registers console command `jkrl_start`;
- registers console command `jkrl_release`;
- starts the bridge server.

#### Request handling

`HandleBridgeRequest(ClientRequest? request)` processes commands sent by Python.

Supported requests:

- `ping`: check whether the bridge is alive.
- `state`: return the latest snapshot.
- `start` / `reset`: request a new Junimo Kart run.
- `action`: set desired jump state.
- `advance`: request progress from Title/Map/Cutscene into gameplay.

#### Pending state

The TCP server thread should not directly change game state. Instead it stores pending values:

- `pendingStartMode`
- `pendingAdvance`
- `desiredJumpHeld`

`OnUpdateTicked` reads these values and applies them on the main game thread.

#### Starting Junimo Kart

`StartMineCart(string mode)` creates the minigame:

```csharp
Game1.currentMinigame = new MineCart(0, modeId);
```

Mode IDs:

```text
2 = Progress Mode
3 = Infinite / Endless Mode
```

The arcade-machine goal uses Progress Mode.

#### Run while unfocused

`EnsureRunsWhenUnfocused()` sets:

```csharp
Game1.options.pauseWhenOutOfFocus = false;
```

This helps training continue while Stardew is not the active window. It may still be best not to fully minimize the game window, since rendering or update throttling can still happen depending on OS/window behavior.

#### Auto-continue

`AutoContinueNonGameplayStates` and `ForceProgressModeGameplay` prevent training from getting stuck outside gameplay.

Junimo Kart game states:

```text
0 = Title
1 = Ingame
2 = FruitsSummary
3 = Map
4 = Cutscene
```

Logic:

- Title -> call `restartLevel(true)`;
- Map -> call `ShowCutscene()`;
- Cutscene -> call `EndCutscene()`;
- Ingame -> let the agent play.

#### Applying jump

`ApplyJump(MineCart mineCart, bool desiredJump)` converts the Python action into game behavior.

It:

1. sets `isJumpPressed`;
2. calls `QueueJump()` when jump starts;
3. calls `ReleaseJump()` when jump ends.

Python action space:

```text
0 = release jump
1 = hold jump
```

Jump duration emerges from repeated actions. For example:

```text
1, 1, 1, 1, 0
```

means the agent held jump for four environment steps and then released it.

#### Snapshot creation

`CreateSnapshot` reads internal game state:

- score;
- lives;
- levels beaten;
- game mode;
- current theme;
- game state;
- player position;
- player velocity;
- grounded/jumping state;
- tracks ahead;
- entities ahead.

Tracks and entities are filtered to a window around the player to keep observations small and relevant.

## Python package

### `junimo_rl/client.py`

This is the TCP client used by Python.

Important methods:

- `connect()`: opens the TCP connection.
- `request(payload)`: sends JSON and reads JSON response.
- `ping()`: checks bridge availability.
- `state()`: gets latest snapshot.
- `start(mode="progress")`: starts Progress Mode.
- `advance()`: asks the bridge to move from non-gameplay state to gameplay.
- `action(jump: bool)`: sends hold/release jump.

Example request:

```json
{"type":"action","jump":true}
```

The client retries once if the TCP connection breaks. If Stardew/SMAPI is closed, it raises a clearer error telling the user to open Stardew through SMAPI and load a save.

### `junimo_rl/env.py`

This is the Gymnasium environment used by Stable-Baselines3.

Stable-Baselines3 expects:

- `reset()`;
- `step(action)`;
- `observation_space`;
- `action_space`.

#### Observation vector

The raw JSON snapshot is converted into a fixed-size `np.float32` vector by `snapshot_to_vector`.

The vector contains:

1. base features:
   - whether the minigame is active;
   - score;
   - lives;
   - levels beaten;
   - game mode;
   - game state;
   - game over flag;
   - completed flag;
   - jump held flag;
   - player position;
   - player velocity;
   - grounded/jumping flags.

2. track features:
   - distance from player (`dx`);
   - y position;
   - track type;
   - obstacle flag;
   - obstacle type.

3. entity features:
   - distance from player;
   - y position;
   - bounds size;
   - entity type;
   - obstacle/pickup flags.

Limits:

```text
MAX_TRACKS = 24
MAX_ENTITIES = 24
```

If there are fewer entries, the rest are padded with zeros. Neural networks need a fixed input size.

#### Action space

```python
spaces.Discrete(2)
```

Meaning:

```text
0 = release jump
1 = hold jump
```

#### Reset

`reset()`:

1. sends `start progress`;
2. waits until Junimo Kart is actually in gameplay;
3. sends `advance` if the game is still in Title/Map/Cutscene;
4. returns the first observation.

#### Step

`step(action)`:

1. stores the previous snapshot;
2. sends the jump action;
3. waits `frame_skip / fps` seconds;
4. reads the new snapshot;
5. computes reward;
6. returns `(obs, reward, terminated, truncated, info)`.

#### Reward function

Current reward:

```text
reward =
  0.001 * delta_x
+ 0.01  * delta_score
+ 50    * max(delta_level, 0)
+ 10    * delta_life
+ 250   if Progress Mode is completed
- 100   if game over just happened
- 1     if the minigame is no longer active
```

Interpretation:

- moving forward is mildly good;
- gaining score is good;
- beating a level is very good;
- completing Progress Mode is excellent;
- losing lives and game over are bad.

There is no explicit penalty for missing a coin or fruit. The agent simply does not get the score reward.

## Scripts

### `scripts/smoke_test.py`

Sanity-check script.

Use it to:

- ping the bridge;
- start Junimo Kart;
- print the internal snapshot;
- optionally hold jump for a short time.

Example:

```powershell
python .\scripts\smoke_test.py --start --hold 0.4
```

If jump control works, the cart should visibly jump and the `duringHold` state should show `jumpHeld: true`.

### `scripts/train_dqn.py`

Training script using Stable-Baselines3 DQN.

Features:

- train by timesteps or episodes;
- save final model;
- save monitor CSV;
- save TensorBoard logs;
- save checkpoints by timestep;
- save checkpoints by episode;
- continue from an existing model.

Example:

```powershell
python .\scripts\train_dqn.py --episodes 1000 --save-episode-freq 100 --model-path models\junimo_dqn --run-name ep_compare_01
```

Outputs:

```text
logs/ep_compare_01/monitor.csv
logs/ep_compare_01/hparams.txt
logs/ep_compare_01/checkpoints/
logs/ep_compare_01/tensorboard/
models/junimo_dqn.zip
```

#### `EpisodeCheckpointCallback`

This custom callback saves model checkpoints every N completed episodes.

Stable-Baselines3 already has timestep-based checkpoints. This project also needs episode-based checkpoints so models like "episode 1000", "episode 2000", and "episode 10000" can be compared directly.

Checkpoint names:

```text
junimo_dqn_ep001000_steps58633.zip
junimo_dqn_ep002000_steps119000.zip
```

### `scripts/plot_training.py`

Plots the training curve from `monitor.csv`.

It creates:

- episode reward;
- rolling mean reward;
- episode length;
- rolling mean episode length.

Command:

```powershell
python .\scripts\plot_training.py .\logs\ep_compare_01\monitor.csv
```

Output:

```text
logs/ep_compare_01/training_plot.png
```

### `scripts/evaluate_models.py`

Evaluates one or more checkpoints deterministically.

Training uses exploration/randomness. Evaluation should be deterministic so checkpoints are compared more fairly.

Command:

```powershell
python .\scripts\evaluate_models.py ".\logs\ep_compare_01\checkpoints\junimo_dqn_ep*.zip" --episodes 20 --out logs\ep_compare_01\evaluation.csv
```

Output columns:

- model path;
- number of evaluation episodes;
- mean reward;
- mean episode length;
- completion rate;
- max levels beaten.

Evaluation still requires Stardew + SMAPI + bridge to be running. If the game closes, the script stops with a clear message.

## Mathematical learning logic

DQN learns:

```text
Q(state, action)
```

This estimates the expected future reward if the agent takes an action from the current state.

The total discounted return is:

```text
G_t = r_t + gamma * r_{t+1} + gamma^2 * r_{t+2} + ...
```

The Bellman target used by DQN is:

```text
target = r + gamma * max_a' Q(next_state, a')
```

The network is trained to reduce:

```text
loss = (Q(state, action) - target)^2
```

With two actions, the network outputs two values:

```text
Q(state, release)
Q(state, hold)
```

The agent usually picks:

```text
argmax_a Q(state, a)
```

but during exploration it sometimes picks a random action.

## Why random levels can still be learned

The model does not memorize a sequence of jumps.

It learns a mapping:

```text
observation -> action
```

Even if the level layout changes, similar situations appear repeatedly:

- gap nearby;
- landing track is higher;
- landing track is lower;
- obstacle is ahead;
- cart is airborne and falling;
- cart is grounded and approaching a gap.

The neural network learns which action tends to work in those situations.

## Track representation

Tracks are not represented as one long rail. The game stores them as many small track pieces.

Example:

```text
track 1: x=0,   y=160, type=Straight
track 2: x=16,  y=160, type=Straight
track 3: x=32,  y=160, type=Straight
track 4: x=48,  y=160, type=Straight
track 5: x=64,  y=144, type=UpSlope
track 6: x=80,  y=128, type=UpSlope
```

A gap appears when there is a large jump in `dx` between track pieces:

```text
dx=16
dx=32
dx=48
dx=160
dx=176
```

The current model receives this raw list. A future improvement would add semantic features like:

```text
next_gap_start
next_gap_width
landing_y
landing_height_delta
next_obstacle_dx
next_obstacle_type
```

That would make learning easier.

## Hyperparameters

### `--episodes`

Number of completed episodes to train for.

If continuing from an existing model, this is the number of additional episodes for the new run.

### `--episode-offset`

Used for cumulative checkpoint names when continuing training.

Example:

```powershell
python .\scripts\train_dqn.py --load-model models\junimo_dqn.zip --episodes 9000 --episode-offset 1000 --save-episode-freq 1000 --model-path models\junimo_dqn --run-name continue_to_10k
```

The first new checkpoint will be named around:

```text
junimo_dqn_ep002000_steps....zip
```

### `--timesteps`

Number of RL environment steps.

If `--episodes` is used, the script stops based on completed episodes instead.

### `--save-episode-freq`

Save model every N completed episodes.

### `--save-freq`

Save model every N timesteps.

### `--frame-skip`

The agent chooses an action every N frames.

- lower value: more precise control, slower training;
- higher value: faster training, rougher jump timing.

Recommended for Junimo Kart:

```text
2
```

### `--learning-rate`

How large neural network updates are.

Default:

```text
1e-4
```

Try `5e-5` if learning is unstable.

### `--exploration-fraction`

Fraction of training where epsilon decays from high randomness to final randomness.

Default:

```text
0.25
```

Try `0.5` if the agent needs more exploration.

### `--exploration-final-eps`

Minimum probability of random action after exploration decays.

Default:

```text
0.05
```

Try `0.1` for more persistent exploration.

### `--buffer-size`

Replay buffer size.

Default:

```text
50000
```

Try `100000` for longer training.

### `--learning-starts`

Number of timesteps before DQN starts training the network.

Default:

```text
2000
```

This allows the replay buffer to collect some experience first.

## Recommended run sequence

1. Close Stardew Valley.
2. Build the mod:

```powershell
dotnet build .\src\JunimoKartRLBridge\JunimoKartRLBridge.csproj -c Release
```

3. Open Stardew through SMAPI.
4. Load a save.
5. Smoke-test bridge and jump:

```powershell
python .\scripts\smoke_test.py --start --hold 0.4
```

6. Train:

```powershell
python .\scripts\train_dqn.py --episodes 10000 --save-episode-freq 1000 --frame-skip 2 --model-path models\junimo_dqn --run-name fresh_10k
```

7. Plot:

```powershell
python .\scripts\plot_training.py .\logs\fresh_10k\monitor.csv
```

8. Evaluate checkpoints:

```powershell
python .\scripts\evaluate_models.py ".\logs\fresh_10k\checkpoints\junimo_dqn_ep*.zip" --episodes 20 --out logs\fresh_10k\evaluation.csv
```

## Troubleshooting

### `ConnectionResetError [WinError 10054]`

Python lost connection to the SMAPI bridge.

Common causes:

1. Stardew/SMAPI was closed.
2. The save is not loaded.
3. The bridge is not listening on `127.0.0.1:8765`.
4. The mod did not load after build/restart.

Check:

```powershell
Get-Process | Where-Object { $_.ProcessName -like '*Stardew*' }
Test-NetConnection -ComputerName 127.0.0.1 -Port 8765
```

If `TcpTestSucceeded` is false, reopen Stardew through SMAPI, load a save, then rerun the Python command.

### The game freezes when clicking another window

Disable Stardew's "pause when unfocused" setting. The mod also tries to force:

```csharp
Game1.options.pauseWhenOutOfFocus = false;
```

For best stability, keep Stardew in Windowed or Borderless mode and avoid fully minimizing it.

### The cart does not jump

Run:

```powershell
python .\scripts\smoke_test.py --start --hold 0.4
```

If the cart does not jump, rebuild the mod and restart Stardew via SMAPI.

## Current caveats

1. Training runs in real time inside the actual game, so it is slow.
2. DQN from scratch may require many episodes.
3. Reward shaping is still simple.
4. The action space only contains hold/release jump.
5. Models trained before the jump-control fix should not be trusted.

## Possible next improvements

1. Add a heuristic teacher for warm starts.
2. Add semantic track features such as gap width and landing height.
3. Add accelerated simulation mode in the mod.
4. Add curriculum training by starting from specific levels/themes.
5. Save gameplay snapshots or videos for debugging.
