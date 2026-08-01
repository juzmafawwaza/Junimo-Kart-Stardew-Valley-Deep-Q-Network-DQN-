# Stardew Valley Junimo Kart Reinforcement Learning

This project is a reinforcement learning experiment for Stardew Valley's Junimo Kart minigame.

Instead of using screen capture, it uses a SMAPI mod to read Junimo Kart's internal game state directly and exposes that state through a local TCP bridge. A Python Gymnasium environment then trains reinforcement learning agents to control the cart by holding or releasing jump.

The first goal is not instant mastery. The goal is to build a reliable pipeline:

1. Stardew Valley runs with SMAPI.
2. The SMAPI mod starts Junimo Kart Progress Mode.
3. Python receives internal observations, not raw pixels.
4. Python sends jump actions back to the game.
5. Optional utility scripts inspect the engineered semantic features.
6. Stable-Baselines3 trains DQN or PPO agents.
7. Training logs, plots, checkpoints, and model comparisons are saved for analysis.

## Recommended v8 baseline

The recommended baseline now uses a bounded 27-feature egocentric observation and the balanced `shaped_v5` reward. It replaces the experimental v7 multi-input model, whose flattened spatial input produced a 4.7-million-parameter PPO-LSTM network.

After closing Stardew/SMAPI, install the rebuilt bridge:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -Force
.\scripts\install_bridge_v8.ps1
```

Restart SMAPI, load a save, then train compact PPO from scratch:

```powershell
.\scripts\run_ppo_compact_v8.ps1
```

The compact PPO-LSTM comparison is available separately:

```powershell
.\scripts\run_ppo_lstm_compact_v8.ps1
```

See `docs/EXPERIMENT_COMPACT_V8_ID.md` for the observation table, reward equation, migration notes, and evaluation commands.

## Project structure

```text
src/JunimoKartRLBridge/   SMAPI mod written in C#
junimo_rl/                Python TCP client and Gymnasium environment
scripts/                  Smoke test, training, plotting, and evaluation scripts
docs/                     Detailed code walkthrough
logs/                     Training logs, ignored by Git
models/                   Saved models, ignored by Git
outputs/                  Local analysis workbooks/exports, ignored by Git
```

## Requirements

- Stardew Valley 1.6+
- SMAPI
- .NET 6 SDK
- Python 3.10+
- Windows PowerShell or Command Prompt

## Build the SMAPI mod

Close Stardew Valley before building. If the game is running, Windows may lock the mod DLL and deployment can fail.

```powershell
cd "C:\Users\VICTUS\OneDrive\Documents\Stardew Valley Reinforcement Learning"
dotnet build .\src\JunimoKartRLBridge\JunimoKartRLBridge.csproj -c Release
```

The `Pathoschild.Stardew.ModBuildConfig` package should automatically copy the mod to:

```text
C:\Program Files (x86)\Steam\steamapps\common\Stardew Valley\Mods\JunimoKartRLBridge
```

## Install Python dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[train,analysis]"
```

## Start Stardew Valley through SMAPI

PowerShell:

```powershell
& "C:\Program Files (x86)\Steam\steamapps\common\Stardew Valley\StardewModdingAPI.exe"
```

Command Prompt:

```cmd
"C:\Program Files (x86)\Steam\steamapps\common\Stardew Valley\StardewModdingAPI.exe"
```

Load a save until you are inside the farm/world. You do not need to walk to the arcade machine; the bridge can start Junimo Kart directly.

## Smoke test

Start Junimo Kart Progress Mode and print the internal state:

```powershell
python .\scripts\smoke_test.py --start
```

Expected signs:

```json
"inMinigame": true
"gameMode": 2
```

Test jump control:

```powershell
python .\scripts\smoke_test.py --start --hold 0.4
```

The cart should visibly jump. If it does not, do not start training yet.

## Inspect semantic features

To see the engineered features live while Junimo Kart is open:

```powershell
python .\scripts\inspect_semantic_features.py --interval 0.25
```

This prints values such as next gap distance, gap width, landing height, obstacle distance, pickup distance, and progress fraction.

## Training

### DQN baseline

Train for 1,000 episodes and save a model checkpoint every 100 episodes:

```powershell
python .\scripts\train_dqn.py --episodes 1000 --save-episode-freq 100 --frame-skip 2 --model-path models\junimo_dqn --run-name ep_compare_01
```

Train from scratch up to 10,000 episodes and save every 1,000 episodes:

```powershell
python .\scripts\train_dqn.py --episodes 10000 --save-episode-freq 1000 --frame-skip 2 --model-path models\junimo_dqn --run-name fresh_10k
```

Continue from a previously trained 1,000-episode model and save cumulative checkpoints at 2,000, 3,000, etc.:

```powershell
python .\scripts\train_dqn.py --load-model models\junimo_dqn.zip --episodes 9000 --episode-offset 1000 --save-episode-freq 1000 --frame-skip 2 --model-path models\junimo_dqn --run-name continue_to_10k
```

### PPO baseline

PPO uses the same live Junimo Kart environment, but writes to separate PPO folders by default:

```text
logs/ppo/<run-name>/
models/ppo/
```

Train PPO from scratch up to 10,000 episodes and save every 1,000 episodes:

```powershell
python .\scripts\train_ppo.py --episodes 10000 --save-episode-freq 1000 --frame-skip 2 --model-path models\ppo\junimo_ppo --run-name ppo_10k
```

Train PPO with engineered semantic features and the first shaped reward revision:

```powershell
python .\scripts\train_ppo.py --episodes 10000 --save-episode-freq 1000 --frame-skip 2 --semantic-features --reward-version shaped_v1 --model-path models\ppo\junimo_ppo_semantic --run-name ppo_semantic_shaped_10k
```

Recommended next experiment: train PPO with semantic features, shaped reward, and macro actions:

```powershell
python .\scripts\train_ppo.py --episodes 5000 --save-episode-freq 1000 --save-freq 0 --frame-skip 2 --semantic-features --reward-version shaped_v2 --action-mode macro --macro-action-frames 8 --model-path models\ppo\junimo_ppo_macro_v2 --run-name ppo_semantic_shaped_v2_macro_5k
```

Equivalent launcher script:

```powershell
.\scripts\run_ppo_macro_v2.ps1
```

Recommended diagnostic experiment after the early plateau: train PPO with `shaped_v3`, macro actions every 6 frames, lower entropy, and survival-first reward:

```powershell
python .\scripts\train_ppo.py --episodes 5000 --save-episode-freq 1000 --save-freq 0 --frame-skip 2 --semantic-features --reward-version shaped_v3 --action-mode macro --macro-action-frames 6 --score-reward-coef 0.0 --gap-landing-confirm-steps 2 --learning-rate 0.0003 --ent-coef 0.003 --model-path models\ppo\junimo_ppo_macro6_v3 --run-name ppo_semantic_shaped_v3_macro6_5k
```

Equivalent launcher script:

```powershell
.\scripts\run_ppo_macro6_v3.ps1
```

PPO-LSTM version of the same `shaped_v3` macro-6 experiment:

```powershell
python .\scripts\train_ppo_lstm.py --episodes 5000 --save-episode-freq 1000 --save-freq 0 --frame-skip 2 --semantic-features --reward-version shaped_v3 --action-mode macro --macro-action-frames 6 --score-reward-coef 0.0 --gap-landing-confirm-steps 2 --learning-rate 0.0001 --ent-coef 0.003 --lstm-hidden-size 128 --n-lstm-layers 1 --model-path models\ppo_lstm\junimo_ppo_lstm_macro6_v3 --run-name ppo_lstm_semantic_shaped_v3_macro6_5k
```

Equivalent launcher script:

```powershell
.\scripts\run_ppo_lstm_macro6_v3.ps1
```

Current recommended diagnostic experiment: train PPO with semantic features, temporal timing features, `shaped_v3`, and macro actions. This keeps long jump continuous while still giving the model timing sensors like jump-held duration and airborne duration. The SMAPI bridge also exposes track bounds so semantic gap detection can use track edges when available:

```powershell
python .\scripts\train_ppo.py --episodes 5000 --save-episode-freq 1000 --save-freq 0 --frame-skip 2 --semantic-features --temporal-features --reward-version shaped_v3 --action-mode macro --macro-action-frames 6 --score-reward-coef 0.0 --gap-landing-confirm-steps 2 --learning-rate 0.0003 --ent-coef 0.003 --model-path models\ppo\junimo_ppo_macro6_temporal_v4b --run-name ppo_semantic_temporal_shaped_v3_macro6_5k
```

Equivalent launcher script:

```powershell
.\scripts\run_ppo_macro_temporal_v4b.ps1
```

PPO-LSTM version of the same v4b experiment. This launcher saves every 100 episodes first, so an interrupted run still leaves checkpoints to evaluate:

```powershell
python .\scripts\train_ppo_lstm.py --episodes 1000 --save-episode-freq 100 --save-freq 0 --frame-skip 2 --semantic-features --temporal-features --reward-version shaped_v3 --action-mode macro --macro-action-frames 6 --score-reward-coef 0.0 --gap-landing-confirm-steps 2 --learning-rate 0.0001 --ent-coef 0.003 --lstm-hidden-size 128 --n-lstm-layers 1 --model-path models\ppo_lstm\junimo_ppo_lstm_macro6_temporal_v4b --run-name ppo_lstm_semantic_temporal_shaped_v3_macro6_1k_save100
```

For details, see `docs/EXPERIMENT_V4_ACTION_OBSERVATION_ID.md`.

MultiInput diagnostic experiment inspired by PokemonRedExperiments-style observation design. This uses separate coordinate, semantic, timing, recent-action-memory, and coordinate-rendered spatial-map inputs:

```powershell
python .\scripts\train_ppo.py --episodes 2000 --save-episode-freq 250 --save-freq 0 --frame-skip 2 --observation-mode multi --recent-action-history 12 --semantic-features --temporal-features --reward-version shaped_v3 --action-mode binary --score-reward-coef 0.0 --gap-landing-confirm-steps 2 --learning-rate 0.00025 --ent-coef 0.02 --model-path models\ppo\junimo_ppo_multiinput_binary_v6 --run-name ppo_multiinput_semantic_spatial_memory_binary_v6_2k_save250
```

Equivalent launcher script:

```powershell
.\scripts\run_ppo_multiinput_binary_v6.ps1
```

PPO-LSTM version:

```powershell
.\scripts\run_ppo_lstm_multiinput_binary_v6.ps1
```

For details, see `docs/EXPERIMENT_MULTIINPUT_V6_ID.md`.

PPO-LSTM uses `sb3-contrib`. If the command says `sb3_contrib` is missing, install/update the training dependencies:

```powershell
pip install -e ".[train,analysis]"
```

Macro actions give the agent four choices:

```text
0 = release jump
1 = short hold
2 = medium hold
3 = long/continue hold
```

This is useful because Junimo Kart is sensitive to how long jump is held. Models trained with `--action-mode macro` are not action-compatible with binary-action checkpoints.

PPO checkpoints do not save DQN-style replay buffers, so they should use much less disk space.

By default, PPO only prints the standard Stable-Baselines3 training tables.

If you want the extra compact episode progress line again, enable it manually:

```powershell
--progress-episode-freq 10
```

Disable it explicitly with:

```powershell
--progress-episode-freq 0
```

## Training outputs

Each run creates:

```text
logs/<algorithm>/<run-name>/monitor.csv     Episode rewards and episode lengths for PPO-style runs
logs/<algorithm>/<run-name>/hparams.txt     Hyperparameters for the run
logs/<algorithm>/<run-name>/checkpoints/    Periodic model checkpoints
logs/<algorithm>/<run-name>/tensorboard/    TensorBoard logs
models/<algorithm>/*.zip                    Final model
```

Older DQN runs may still use `logs/<run-name>/` and `models/junimo_dqn.zip`.

Example episode checkpoints:

```text
junimo_dqn_ep000100_steps6241.zip
junimo_dqn_ep001000_steps58633.zip
```

## Plot training curves

```powershell
python .\scripts\plot_training.py .\logs\ep_compare_01\monitor.csv
```

Output:

```text
logs\ep_compare_01\training_plot.png
```

## TensorBoard

```powershell
tensorboard --logdir .\logs
```

## Compare checkpoints

Make sure Stardew Valley is open through SMAPI and a save is loaded, because evaluation also runs the model inside the live game.

DQN:

```powershell
python .\scripts\evaluate_models.py ".\logs\ep_compare_01\checkpoints\junimo_dqn_ep*.zip" --episodes 20 --out logs\ep_compare_01\evaluation.csv
```

PPO:

```powershell
python .\scripts\evaluate_ppo_models.py ".\logs\ppo\ppo_10k\checkpoints\junimo_ppo_ep*.zip" --episodes 20 --frame-skip 2 --out logs\ppo\ppo_10k\evaluation.csv
```

PPO trained with semantic features must also be evaluated with semantic features:

```powershell
python .\scripts\evaluate_ppo_models.py ".\logs\ppo\ppo_semantic_shaped_10k\checkpoints\junimo_ppo_ep*.zip" --episodes 20 --frame-skip 2 --semantic-features --reward-version shaped_v1 --out logs\ppo\ppo_semantic_shaped_10k\evaluation.csv
```

PPO trained with macro actions must also be evaluated with macro actions:

```powershell
python .\scripts\evaluate_ppo_models.py ".\logs\ppo\ppo_semantic_shaped_v2_macro_5k\checkpoints\junimo_ppo_ep*.zip" --episodes 20 --frame-skip 2 --semantic-features --reward-version shaped_v2 --action-mode macro --macro-action-frames 8 --out logs\ppo\ppo_semantic_shaped_v2_macro_5k\evaluation.csv
```

PPO-LSTM checkpoints must be evaluated with the recurrent evaluator because the LSTM hidden state must be carried between steps:

```powershell
python .\scripts\evaluate_ppo_lstm_models.py ".\logs\ppo_lstm\ppo_lstm_semantic_shaped_v3_macro6_5k\checkpoints\junimo_ppo_lstm_ep*.zip" --episodes 20 --frame-skip 2 --semantic-features --reward-version shaped_v3 --action-mode macro --macro-action-frames 6 --score-reward-coef 0.0 --gap-landing-confirm-steps 2 --out logs\ppo_lstm\ppo_lstm_semantic_shaped_v3_macro6_5k\evaluation.csv
```

View the result:

```powershell
Import-Csv .\logs\ppo\ppo_10k\evaluation.csv | Format-Table
```

## How the agent learns

The default action space is intentionally small:

```text
0 = release jump
1 = hold jump
```

Jump duration is not a separate action. It emerges from repeated actions over time:

```text
1, 1, 1, 1, 0
```

means the agent held jump for four environment steps and then released it.

For timing-sensitive experiments, `--action-mode macro` exposes jump duration more directly:

```text
0 = release jump
1 = short hold
2 = medium hold
3 = long/continue hold
```

This is often more useful for Junimo Kart because jump height depends on how long the button is held.

The DQN learns a function:

```text
Q(state, action)
```

which estimates how valuable each action is from the current game state.

PPO learns a policy directly:

```text
policy(state) -> action probabilities
```

For the default two-action setup, PPO learns whether release or hold jump is more likely to produce good future outcomes. For macro-action runs, PPO learns a probability distribution across release, short hold, medium hold, and long hold.

The reward function currently rewards:

- moving forward,
- gaining score,
- beating a level,
- completing Progress Mode,

and penalizes:

- losing lives,
- game over,
- leaving the minigame.

The agent does not see raw pixels. It receives internal game-state features such as:

- cart position,
- cart velocity,
- grounded/jumping flags,
- track tiles ahead,
- track type,
- obstacle/pickup entities ahead,
- score,
- lives,
- current level/theme.

Optional semantic features can be enabled with `--semantic-features`. These append engineered features derived from the existing track/entity snapshots:

- next track distance and height,
- whether the next visible track has an obstacle,
- next gap presence,
- next gap start distance,
- next gap width,
- landing height,
- landing height delta,
- next obstacle distance/height,
- next pickup distance/height,
- distance to finish,
- progress fraction.

Gap detection is heuristic: Python infers a gap when the distance between two forward track pieces is larger than the configured threshold.

For `shaped_v2`, the gap-landing reward is stricter: a tracked gap attempt starts only when the inferred gap is at least `56px` wide and its start is within `180px` ahead of the cart.

The legacy reward is:

```text
0.001 * delta_x
+ 0.01 * delta_score
+ 50 * level_delta
+ 10 * life_delta
+ 250 if completed
- 100 if game over starts
- 1 if outside minigame
```

The optional `shaped_v1` reward keeps the same intent but makes survival/progress signals denser:

```text
0.003 * delta_x
+ 0.02 * delta_score
+ 100 * level_delta
+ 25 * life_delta
+ 0.02 per alive gameplay step
+ small bonus for jumping near a detected gap
- small penalty for not jumping near a close gap
- small penalty for unnecessary grounded jump holds
+ small bonus for safe landing
+ 500 if completed
- 80 if game over starts
- 2 if outside minigame
```

The optional `shaped_v2` reward is more outcome-based:

```text
0.004 * delta_x
+ 0.005 * delta_score
+ 120 * level_delta
+ 25 * life_delta
+ 0.03 per alive gameplay step
+ bonus only when the cart lands safely after a tracked gap
- small penalty for not holding jump when a real gap is very close
- small penalty for unnecessary grounded jump holds
+ 600 if completed
- 80 if game over starts
- extra small penalty if a tracked gap attempt ends in game over
- 2 if outside minigame
```

`shaped_v2` intentionally reduces the score/coin/fruit incentive so the agent focuses more on survival and gap crossing.

The optional `shaped_v3` reward is a survival-first diagnostic reward for the current plateau:

```text
0.006 * delta_x
+ 0.0 * delta_score by default
+ 150 * level_delta
+ 30 * life_delta
+ 0.035 per alive gameplay step
+ tiny encouragement for grounded jump near a valid gap/obstacle
+ gap landing reward only after the landing survives extra confirmation steps
- stronger penalty for unnecessary grounded jump holds
+ 700 if completed
- 80 if game over starts
- extra small penalty if a tracked gap attempt ends in game over
- 2 if outside minigame
```

`shaped_v3` intentionally disables coin/fruit score reward by default with `--score-reward-coef 0.0`, because the current goal is clearing Progress Mode, not maximizing score. You can reintroduce score later after survival improves.

New training runs also write extra telemetry columns to `monitor.csv`, including action counts, gap attempts, gap landings, gap failures, death-near-gap, death-near-obstacle, pickup events, score delta, and max episode x-position. These columns make it easier to diagnose whether a model is dying because it never jumps, jumps too often, fails gaps, or hits obstacles.

Quickly summarize a monitor file:

```powershell
python .\scripts\summarize_monitor.py .\logs\ppo\ppo_semantic_shaped_v3_macro6_5k\monitor.csv --window 100
```

Models trained with `--semantic-features` are not observation-compatible with legacy models. For fair paper comparisons, compare game metrics such as score, levels beaten, and completion rate, or rerun every algorithm under the same observation/reward settings.

## Important notes

- Training is real-time because it runs inside the actual Stardew Valley game loop.
- Do not manually press Space during training; it can corrupt the training data.
- Disable or force off "pause when unfocused" so the game keeps running while the window is not active.
- Do not minimize the game window completely if it causes rendering or update throttling.
- Models trained before the jump-control fix should not be trusted.

## Ethics / achievement note

This is intended for single-player experimentation and learning. SMAPI can coexist with Steam achievements, but this is still automation/modding. Avoid using it for leaderboard manipulation or anything that affects other players.

## More documentation

See the detailed walkthrough:

[docs/CODE_WALKTHROUGH.md](docs/CODE_WALKTHROUGH.md)

For a learning-focused Indonesian explanation of every major file and code flow, see:

[docs/CODE_EXPLANATION_ID.md](docs/CODE_EXPLANATION_ID.md)

For the PPO macro 6 + `shaped_v3` diagnostic experiment notes, see:

[docs/EXPERIMENT_SHAPED_V3_ID.md](docs/EXPERIMENT_SHAPED_V3_ID.md)

For the PPO-LSTM version of the `shaped_v3` experiment, see:

[docs/EXPERIMENT_PPO_LSTM_ID.md](docs/EXPERIMENT_PPO_LSTM_ID.md)

For the deterministic baseline and automated calibration logger, see:

[deterministic/README.md](deterministic/README.md)

## Experimental `shaped_v6` anti-spam continuation

The compact v8 observation remains unchanged at 27 features, but the optional `shaped_v6` reward adds a tiny cost for each real grounded jump start and a bounded quadratic penalty when a gap death occurs far from the full landing-track interval. The original `shaped_v5` behavior remains available for reproducibility.

Continue from the selected episode-1,500 PPO checkpoint:

```powershell
.\scripts\run_ppo_compact_shaped_v6.ps1
```

Detailed Indonesian design notes, formulas, baseline evaluation, telemetry fields, and paper ablation guidance:

[docs/EXPERIMENT_SHAPED_V6_ID.md](docs/EXPERIMENT_SHAPED_V6_ID.md)

## Anchored gap detector experiment

Multi-level Junimo Kart layouts can contain several rails at similar x-coordinates. The legacy detector scans all x-sorted adjacent rails and may therefore select a gap that does not belong to the cart's current path. The optional `anchored` detector starts from the grounded supporting rail, follows its connected rail run, selects the first plausible landing run, and keeps that absolute takeoff/landing pair stable while the cart is airborne.

The legacy detector remains the default for reproducibility. The new experiment explicitly enables the improved detector and continues from the selected episode-1,500 compact PPO checkpoint:

```powershell
.\scripts\run_ppo_compact_anchored_v9.ps1
```

Evaluate matching checkpoints with:

```powershell
.\scripts\evaluate_ppo_compact_anchored_v9.ps1 -Episodes 20
```

The compact observation remains 27-dimensional and the action remains binary release/hold. No bridge reinstall is required. Detailed Indonesian implementation notes and test cases are available in [docs/EXPERIMENT_ANCHORED_GAP_V9_ID.md](docs/EXPERIMENT_ANCHORED_GAP_V9_ID.md).

## Anti-long-jump v10 experiment

Early anchored-v9 telemetry showed that the inherited episode-1,500 policy remained low-entropy and continued holding jump for long airborne periods. `shaped_v6` charged only the jump-start edge, so short and long holds could have the same action cost.

The separate `shaped_v7` reward keeps v6 unchanged and adds a small airborne hold-duration cost after four free hold decisions. The recommended v10 launcher starts a fresh PPO policy, uses the anchored detector, keeps the 27-feature compact observation and binary release/hold action, and raises the entropy coefficient to encourage exploration of different release timings:

```powershell
.\scripts\run_ppo_compact_anchored_v10.ps1
```

Evaluate matching checkpoints with:

```powershell
.\scripts\evaluate_ppo_compact_anchored_v10.ps1 -Episodes 20
```

See [docs/EXPERIMENT_ANTI_LONG_JUMP_V10_ID.md](docs/EXPERIMENT_ANTI_LONG_JUMP_V10_ID.md) for the diagnosis, reward formula, telemetry, hyperparameters, and warm-start ablation command.
