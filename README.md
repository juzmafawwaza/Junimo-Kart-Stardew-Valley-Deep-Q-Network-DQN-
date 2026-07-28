# Stardew Valley Junimo Kart Deep Q-Network (DQN)

This project is a reinforcement learning experiment for Stardew Valley's Junimo Kart minigame.

Instead of using screen capture, it uses a SMAPI mod to read Junimo Kart's internal game state directly and exposes that state through a local TCP bridge. A Python Gymnasium environment then trains a DQN agent to control the cart by holding or releasing jump.

The first goal is not instant mastery. The goal is to build a reliable pipeline:

1. Stardew Valley runs with SMAPI.
2. The SMAPI mod starts Junimo Kart Progress Mode.
3. Python receives internal observations, not raw pixels.
4. Python sends jump actions back to the game.
5. Stable-Baselines3 trains a DQN agent.
6. Training logs, plots, checkpoints, and model comparisons are saved for analysis.

## Project structure

```text
src/JunimoKartRLBridge/   SMAPI mod written in C#
junimo_rl/                Python TCP client and Gymnasium environment
scripts/                  Smoke test, training, plotting, and evaluation scripts
docs/                     Detailed code walkthrough
logs/                     Training logs, ignored by Git
models/                   Saved models, ignored by Git
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

## Training

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

## Training outputs

Each run creates:

```text
logs/<run-name>/monitor.csv          Episode rewards and episode lengths
logs/<run-name>/hparams.txt          Hyperparameters for the run
logs/<run-name>/checkpoints/         Periodic model checkpoints
logs/<run-name>/tensorboard/         TensorBoard logs
models/junimo_dqn.zip                Final model
```

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

```powershell
python .\scripts\evaluate_models.py ".\logs\ep_compare_01\checkpoints\junimo_dqn_ep*.zip" --episodes 20 --out logs\ep_compare_01\evaluation.csv
```

View the result:

```powershell
Import-Csv .\logs\ep_compare_01\evaluation.csv | Format-Table
```

## How the agent learns

The action space is intentionally small:

```text
0 = release jump
1 = hold jump
```

Jump duration is not a separate action. It emerges from repeated actions over time:

```text
1, 1, 1, 1, 0
```

means the agent held jump for four environment steps and then released it.

The DQN learns a function:

```text
Q(state, action)
```

which estimates how valuable each action is from the current game state.

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
