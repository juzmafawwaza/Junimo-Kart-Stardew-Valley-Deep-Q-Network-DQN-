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
Python RL agent
  -> chooses action: release jump / hold jump
  -> sends JSON to the local TCP bridge
  -> SMAPI mod receives the action
  -> mod changes Junimo Kart's internal jump state
  -> game advances for a few frames
  -> mod reads Junimo Kart's internal state
  -> Python receives an observation and computes reward
  -> DQN or PPO learns from the transition
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

Optional macro action mode is enabled with:

```powershell
--action-mode macro
```

Macro action space:

```text
0 = release jump
1 = short hold
2 = medium hold
3 = long/continue hold
```

With `--macro-action-frames 8`, each macro action controls an eight-frame window. Action `1` holds jump for roughly 25% of that window, action `2` for roughly 50%, and action `3` for the full window. This gives the agent a cleaner way to express jump duration.

Binary-action checkpoints and macro-action checkpoints are not compatible because the policy output size changes.

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

Optional semantic features can be enabled through `use_semantic_features=True` or the training flag `--semantic-features`.

These features are derived from the same track/entity snapshot, but they describe useful gameplay concepts directly:

- next visible track distance, height, type, and obstacle flag;
- whether a gap is visible soon;
- next gap start distance;
- next gap width;
- landing height;
- landing height delta compared to takeoff;
- next obstacle distance and height;
- next pickup distance and height;
- distance to finish;
- progress fraction.

Gap detection is currently heuristic. The bridge exposes track positions, not the physical width of each track piece, so Python infers a gap when the distance between two forward track pieces is larger than `GAP_PIXEL_THRESHOLD`.

The `shaped_v2` gap-landing reward uses a stricter tracked-attempt rule: the inferred gap must be at least `GAP_LANDING_REWARD_MIN_WIDTH` (`56px`) and its start must be within `GAP_LANDING_REWARD_ACTIVATION_DX` (`180px`) ahead of the cart.

This changes the observation vector size. Legacy models trained without semantic features must be evaluated without `--semantic-features`; models trained with semantic features must be evaluated with it.

#### Action space

Default binary mode:

```python
spaces.Discrete(2)
```

Meaning:

```text
0 = release jump
1 = hold jump
```

Optional macro mode:

```python
spaces.Discrete(4)
```

Meaning:

```text
0 = release jump
1 = short hold
2 = medium hold
3 = long/continue hold
```

Macro mode is enabled with `--action-mode macro`.

#### Reset

`reset()`:

1. sends `start progress`;
2. waits until Junimo Kart is actually in gameplay;
3. sends `advance` if the game is still in Title/Map/Cutscene;
4. returns the first observation.

#### Step

`step(action)`:

1. stores the previous snapshot;
2. applies the action;
3. in binary mode, sends hold/release and waits `frame_skip / fps` seconds;
4. in macro mode, holds jump for part or all of the macro action window;
5. reads the new snapshot;
6. computes reward;
7. returns `(obs, reward, terminated, truncated, info)`.

#### Reward function

Legacy reward:

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

Optional shaped reward `shaped_v1` is enabled with:

```powershell
--reward-version shaped_v1
```

It uses denser signals:

```text
reward =
  0.003 * delta_x
+ 0.02  * delta_score
+ 100   * max(delta_level, 0)
+ 25    * delta_life
+ 0.02  per alive gameplay step
+ small bonus for jumping near a detected gap
- small penalty for not jumping near a close gap
- small penalty for unnecessary grounded jump holds
+ small bonus for safe landing
+ 500   if Progress Mode is completed
- 80    if game over just happened
- 2     if the minigame is no longer active
```

Optional shaped reward `shaped_v2` is enabled with:

```powershell
--reward-version shaped_v2
```

It is more outcome-based:

```text
reward =
  0.004 * delta_x
+ 0.005 * delta_score
+ 120   * max(delta_level, 0)
+ 25    * delta_life
+ 0.03  per alive gameplay step
+ bonus only when a tracked gap is crossed and the cart lands safely
- small penalty for not holding jump when a real gap is very close
- small penalty for unnecessary grounded jump holds
+ 600   if Progress Mode is completed
- 80    if game over just happened
- extra small penalty if a tracked gap attempt ends in game over
- 2     if the minigame is no longer active
```

`shaped_v2` reduces score reward so coins/fruits are not over-prioritized when survival is still weak. It also removes the general safe-landing bonus from `shaped_v1`; landing reward is only given for a tracked gap attempt.

This reward is not directly comparable to legacy reward values. For paper comparisons across reward versions, prefer game metrics such as score, levels beaten, and completion rate, or train all compared algorithms with the same reward version.

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

### `scripts/inspect_semantic_features.py`

Live debugging script for the engineered semantic features.

It repeatedly reads the bridge state and prints:

- cart x position;
- grounded flag;
- next gap presence;
- next gap start distance;
- next gap width;
- landing height;
- landing height delta;
- next obstacle distance;
- next pickup distance;
- progress fraction.

Example:

```powershell
python .\scripts\inspect_semantic_features.py --interval 0.25
```

This is useful before semantic training because it lets you verify whether the Python side is detecting gaps and landing points in a reasonable way.

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

Semantic/shaped DQN example:

```powershell
python .\scripts\train_dqn.py --episodes 10000 --save-episode-freq 1000 --frame-skip 2 --semantic-features --reward-version shaped_v1 --log-dir logs\dqn_semantic --model-path models\dqn_semantic\junimo_dqn_semantic --run-name dqn_semantic_shaped_10k
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

Important disk note: DQN can save replay buffers. Replay buffer files are useful for exact training continuation, but they are much larger than model `.zip` checkpoints. For analysis and deterministic evaluation, the model `.zip`, `monitor.csv`, `hparams.txt`, and `evaluation.csv` are usually enough.

The DQN script now saves replay buffers only when `--save-replay-buffer` is passed.

### `scripts/train_ppo.py`

Training script using Stable-Baselines3 PPO.

It uses the same `JunimoKartEnv` environment as DQN. The SMAPI bridge and observation vector do not change; only the learning algorithm changes.

Default PPO output folders:

```text
logs/ppo/<run-name>/
models/ppo/
```

Example:

```powershell
python .\scripts\train_ppo.py --episodes 10000 --save-episode-freq 1000 --frame-skip 2 --model-path models\ppo\junimo_ppo --run-name ppo_10k
```

Semantic/shaped PPO example:

```powershell
python .\scripts\train_ppo.py --episodes 10000 --save-episode-freq 1000 --frame-skip 2 --semantic-features --reward-version shaped_v1 --model-path models\ppo\junimo_ppo_semantic --run-name ppo_semantic_shaped_10k
```

Recommended macro-action PPO example:

```powershell
python .\scripts\train_ppo.py --episodes 5000 --save-episode-freq 1000 --save-freq 0 --frame-skip 2 --semantic-features --reward-version shaped_v2 --action-mode macro --macro-action-frames 8 --model-path models\ppo\junimo_ppo_macro_v2 --run-name ppo_semantic_shaped_v2_macro_5k
```

Equivalent launcher script:

```powershell
.\scripts\run_ppo_macro_v2.ps1
```

Outputs:

```text
logs/ppo/ppo_10k/monitor.csv
logs/ppo/ppo_10k/hparams.txt
logs/ppo/ppo_10k/checkpoints/
logs/ppo/ppo_10k/tensorboard/
models/ppo/junimo_ppo.zip
```

PPO does not use a DQN-style replay buffer, so checkpoints are much smaller. This is intentional to avoid filling the C: drive with large `*_replay_buffer_*.pkl` files.

The script includes an optional `EpisodeProgressCallback`, which can print a compact progress line every N completed episodes.

It is disabled by default so the terminal shows only the standard Stable-Baselines3 training tables.

If enabled, the extra line looks like:

```text
PPO episode progress | episodes=4 | total_timesteps=... | recent_reward_mean=... | recent_length_mean=...
```

Useful options:

- `--progress-episode-freq`: print every N completed episodes; default `0` disables it.
- `--progress-window`: number of recent episodes used for the printed rolling means.

Important PPO hyperparameters:

- `n_steps`: rollout length before each PPO update.
- `batch_size`: minibatch size during policy updates.
- `n_epochs`: how many passes PPO makes over each rollout.
- `gae_lambda`: bias/variance tradeoff for advantage estimation.
- `clip_range`: limits how much the policy can change per update.
- `ent_coef`: entropy bonus; higher values encourage more exploration.

Conceptually:

```text
DQN learns Q(state, action).
PPO learns policy(action | state).
```

By default, both algorithms choose between the same two actions:

```text
0 = release jump
1 = hold jump
```

With `--action-mode macro`, they choose between four duration-oriented actions:

```text
0 = release jump
1 = short hold
2 = medium hold
3 = long/continue hold
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

Evaluates one or more DQN checkpoints deterministically.

Training uses exploration/randomness. Evaluation should be deterministic so checkpoints are compared more fairly.

Command:

```powershell
python .\scripts\evaluate_models.py ".\logs\ep_compare_01\checkpoints\junimo_dqn_ep*.zip" --episodes 20 --out logs\ep_compare_01\evaluation.csv
```

Output columns:

- algorithm;
- model path;
- semantic feature flag;
- reward version;
- action mode;
- macro action frames;
- number of evaluation episodes;
- mean reward;
- mean episode length;
- mean score;
- max score;
- mean levels beaten;
- max levels beaten;
- completion rate;
- game over rate;
- jump hold ratio.

### `scripts/evaluate_ppo_models.py`

Evaluates one or more PPO checkpoints deterministically.

Command:

```powershell
python .\scripts\evaluate_ppo_models.py ".\logs\ppo\ppo_10k\checkpoints\junimo_ppo_ep*.zip" --episodes 20 --frame-skip 2 --out logs\ppo\ppo_10k\evaluation.csv
```

The output columns intentionally match the DQN evaluator, so DQN and PPO can be compared in the same Excel workbook or paper table.

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

In binary mode, the network outputs two values:

```text
Q(state, release)
Q(state, hold)
```

In macro mode, the network outputs four values:

```text
Q(state, release)
Q(state, short_hold)
Q(state, medium_hold)
Q(state, long_hold)
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

Legacy models receive this raw list. New semantic-feature runs append engineered features like:

```text
next_gap_start
next_gap_width
landing_y
landing_height_delta
next_obstacle_dx
next_obstacle_type
```

These features make the geometry easier for a dense network to interpret without learning every gap concept from raw track-piece sequences.

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

### `--action-mode`

Selects the action representation:

```text
binary
macro
```

`binary` is the legacy two-action mode:

```text
0 = release jump
1 = hold jump
```

`macro` exposes jump duration as four discrete actions:

```text
0 = release jump
1 = short hold
2 = medium hold
3 = long/continue hold
```

Models trained with `binary` cannot be loaded with `macro`, and macro models cannot be loaded with `binary`.

### `--macro-action-frames`

Total frame window for each macro action. The default is:

```text
8
```

This option only affects `--action-mode macro`.

### `--semantic-features`

Appends engineered gap, landing, obstacle, pickup, finish-distance, and progress-fraction features to the observation vector.

Use this only for new models:

```powershell
--semantic-features
```

Do not use it when evaluating legacy DQN checkpoints trained without semantic features.

### `--reward-version`

Selects the reward function:

```text
legacy
shaped_v1
shaped_v2
```

`legacy` preserves compatibility with previous runs. `shaped_v1` adds denser progress, survival, gap timing, and general landing signals. `shaped_v2` shifts the reward toward outcome-based gap crossing and reduces coin/fruit chasing.

### `--save-replay-buffer`

DQN-only. Saves replay buffers next to checkpoints. This is disabled by default because replay buffers can consume many gigabytes.

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

6. Train a DQN baseline:

```powershell
python .\scripts\train_dqn.py --episodes 10000 --save-episode-freq 1000 --frame-skip 2 --model-path models\junimo_dqn --run-name fresh_10k
```

Or train a PPO baseline:

```powershell
python .\scripts\train_ppo.py --episodes 10000 --save-episode-freq 1000 --frame-skip 2 --model-path models\ppo\junimo_ppo --run-name ppo_10k
```

7. Plot:

```powershell
python .\scripts\plot_training.py .\logs\fresh_10k\monitor.csv
python .\scripts\plot_training.py .\logs\ppo\ppo_10k\monitor.csv
```

8. Evaluate checkpoints:

```powershell
python .\scripts\evaluate_models.py ".\logs\fresh_10k\checkpoints\junimo_dqn_ep*.zip" --episodes 20 --out logs\fresh_10k\evaluation.csv
python .\scripts\evaluate_ppo_models.py ".\logs\ppo\ppo_10k\checkpoints\junimo_ppo_ep*.zip" --episodes 20 --frame-skip 2 --out logs\ppo\ppo_10k\evaluation.csv
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
2. DQN and PPO from scratch may require many episodes.
3. Reward shaping is still simple.
4. Macro action mode is experimental and creates checkpoints that are incompatible with binary-action checkpoints.
5. Models trained before the jump-control fix should not be trusted.

## Possible next improvements

1. Add a heuristic teacher for warm starts.
2. Evaluate semantic/shaped PPO against macro-action PPO.
3. Add a heuristic teacher for warm starts.
4. Add Recurrent PPO / PPO-LSTM.
5. Add DRQN for DQN + memory comparisons.
6. Add accelerated simulation mode in the mod.
7. Add curriculum training by starting from specific levels/themes.
8. Save gameplay snapshots or videos for debugging.
