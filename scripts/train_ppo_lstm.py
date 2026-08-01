from __future__ import annotations

import argparse
import sys
from collections import deque
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from sb3_contrib import RecurrentPPO
except ImportError as exc:  # pragma: no cover - only used when dependency is missing
    raise SystemExit(
        "PPO-LSTM requires sb3-contrib.\n"
        "Install it with one of these commands:\n"
        "  pip install sb3-contrib\n"
        "  pip install -e \".[train,analysis]\""
    ) from exc

from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, StopTrainingOnMaxEpisodes
from stable_baselines3.common.monitor import Monitor

from junimo_rl import JunimoKartEnv, TELEMETRY_INFO_KEYS
from junimo_rl.state_trace import StateTraceCallback


class EpisodeCheckpointCallback(BaseCallback):
    def __init__(
        self,
        save_freq_episodes: int,
        save_path: Path,
        name_prefix: str = "junimo_ppo_lstm",
        episode_offset: int = 0,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.save_freq_episodes = save_freq_episodes
        self.save_path = save_path
        self.episode_count = max(episode_offset, 0)
        self.name_prefix = name_prefix
        self.next_checkpoint_episode = (
            (self.episode_count // save_freq_episodes) + 1
        ) * save_freq_episodes

    def _init_callback(self) -> None:
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is None:
            return True

        finished_episodes = int(sum(bool(done) for done in dones))
        if finished_episodes <= 0:
            return True

        self.episode_count += finished_episodes
        while self.episode_count >= self.next_checkpoint_episode:
            stem = f"{self.name_prefix}_ep{self.next_checkpoint_episode:06d}_steps{self.num_timesteps}"
            model_file = self.save_path / f"{stem}.zip"
            self.model.save(model_file)

            if self.verbose > 0:
                print(f"Saved episode checkpoint: {model_file}")

            self.next_checkpoint_episode += self.save_freq_episodes

        return True


class EpisodeProgressCallback(BaseCallback):
    def __init__(
        self,
        log_freq_episodes: int,
        episode_offset: int = 0,
        window: int = 20,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.log_freq_episodes = log_freq_episodes
        self.episode_count = max(episode_offset, 0)
        self.next_log_episode = (
            (self.episode_count // log_freq_episodes) + 1
        ) * log_freq_episodes
        self.recent_rewards: deque[float] = deque(maxlen=max(window, 1))
        self.recent_lengths: deque[float] = deque(maxlen=max(window, 1))

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        infos = self.locals.get("infos") or []
        if dones is None:
            return True

        for done, info in zip(dones, infos):
            if not bool(done):
                continue

            self.episode_count += 1
            episode_info = info.get("episode") if isinstance(info, dict) else None
            if episode_info:
                self.recent_rewards.append(float(episode_info.get("r", 0.0)))
                self.recent_lengths.append(float(episode_info.get("l", 0.0)))

        while self.episode_count >= self.next_log_episode:
            if self.verbose > 0:
                reward_mean = sum(self.recent_rewards) / len(self.recent_rewards) if self.recent_rewards else 0.0
                length_mean = sum(self.recent_lengths) / len(self.recent_lengths) if self.recent_lengths else 0.0
                print(
                    "PPO-LSTM episode progress | "
                    f"episodes={self.episode_count} | "
                    f"total_timesteps={self.num_timesteps} | "
                    f"recent_reward_mean={reward_mean:.3f} | "
                    f"recent_length_mean={length_mean:.1f}"
                )

            self.next_log_episode += self.log_freq_episodes

        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a RecurrentPPO/PPO-LSTM agent against Junimo Kart through the SMAPI bridge.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--episodes", type=int, default=None, help="Optional episode target; training stops after this many completed episodes.")
    parser.add_argument("--episode-offset", type=int, default=0, help="Episode number to add to checkpoint names when continuing a previous run.")
    parser.add_argument("--model-path", default="models/ppo_lstm/junimo_ppo_lstm")
    parser.add_argument("--load-model", default=None, help="Optional existing PPO-LSTM model .zip path to continue training.")
    parser.add_argument("--log-dir", default="logs/ppo_lstm")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--save-freq", type=int, default=10_000, help="Save timestep checkpoints every N environment steps. Set 0 to disable.")
    parser.add_argument("--save-episode-freq", type=int, default=1000, help="Save a comparable model checkpoint every N completed episodes. Set 0 to disable.")
    parser.add_argument("--progress-episode-freq", type=int, default=0, help="Print a compact progress line every N completed episodes. Default 0 disables it so only SB3 tables are shown.")
    parser.add_argument("--progress-window", type=int, default=20, help="Number of recent episodes used for compact progress means.")
    parser.add_argument("--trace-state-print-freq", type=int, default=0, help="Print one live state trace line every N environment steps. Use 1 for every movement; 0 disables printing.")
    parser.add_argument("--trace-state-format", default="full", choices=["full", "simple"], help="Printed state trace format. simple prints Flappy-Bird-style x/y/reward/score/generation lines.")
    parser.add_argument("--trace-state-simple-raw", action="store_true", help="For --trace-state-format simple, print raw target dx/dy instead of discretized bins.")
    parser.add_argument("--trace-state-simple-action", action="store_true", help="For --trace-state-format simple, also print action and hold fields.")
    parser.add_argument("--trace-state-csv", default=None, help="Optional CSV path for per-step state traces. Useful for later analysis.")
    parser.add_argument("--trace-state-csv-freq", type=int, default=1, help="Write one state trace CSV row every N environment steps.")
    parser.add_argument("--trace-state-max-rows", type=int, default=0, help="Maximum state trace rows to print/write. Use 0 for unlimited.")
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--observation-mode", default="flat", choices=["flat", "multi", "compact"], help="Observation format. compact is the recommended 27-feature egocentric v8 state.")
    parser.add_argument("--gap-detection-mode", default="legacy", choices=["legacy", "anchored"], help="legacy scans all x-sorted tracks; anchored follows the grounded cart's connected rail and persists the landing target while airborne.")
    parser.add_argument("--recent-action-history", type=int, default=12, help="Number of recent actions encoded in multi-input observations.")
    parser.add_argument("--semantic-features", action="store_true", help="Append engineered gap/landing/obstacle/pickup features to the observation vector.")
    parser.add_argument("--temporal-features", action="store_true", help="Append timing features such as jump-held duration, airborne duration, grounded duration, and previous action.")
    parser.add_argument("--reward-version", default="legacy", choices=["legacy", "shaped_v1", "shaped_v2", "shaped_v3", "shaped_v4", "shaped_v5", "shaped_v6", "shaped_v7"], help="Reward function to use. shaped_v7 adds an airborne hold-duration cost to shaped_v6.")
    parser.add_argument("--action-mode", default="binary", choices=["binary", "macro", "tap_macro"], help="Action representation. Use tap_macro to force a release tail after each jump macro.")
    parser.add_argument("--macro-action-frames", type=int, default=8, help="Total game frames controlled by each macro action when --action-mode macro/tap_macro is used.")
    parser.add_argument("--macro-release-frames", type=int, default=1, help="For --action-mode tap_macro, force this many release frames at the end of every jump macro.")
    parser.add_argument("--score-reward-coef", type=float, default=None, help="Override score reward coefficient. Use 0.0 to ignore coin/fruit score.")
    parser.add_argument("--coin-reward-coef", type=float, default=None, help="Reward coefficient for detected coin pickups. When set, split pickup reward is used instead of generic score reward.")
    parser.add_argument("--fruit-reward-coef", type=float, default=None, help="Reward coefficient for detected fruit pickups. Use a higher value than coin if fruit should matter more.")
    parser.add_argument("--fruit-score-threshold", type=float, default=100.0, help="Fallback: unknown pickup score_delta at/above this value is treated as fruit.")
    parser.add_argument("--gap-landing-confirm-steps", type=int, default=2, help="For shaped_v3, require this many extra env steps alive after landing before paying the gap landing reward.")
    parser.add_argument("--gap-landing-base-reward", type=float, default=8.0, help="Base reward paid for a confirmed gap landing in shaped_v4.")
    parser.add_argument("--gap-landing-width-coef", type=float, default=0.04, help="Extra gap landing reward per pixel of gap width in shaped_v4.")
    parser.add_argument("--progress-reward-coef", type=float, default=0.01, help="shaped_v5 reward per newly reached world-x pixel.")
    parser.add_argument("--death-penalty", type=float, default=5.0, help="Single terminal death penalty used by shaped_v5.")
    parser.add_argument("--level-complete-reward", type=float, default=50.0, help="Reward per completed level in shaped_v5.")
    parser.add_argument("--game-complete-reward", type=float, default=200.0, help="Additional full-game completion reward in shaped_v5.")
    parser.add_argument("--coin-reward-value", type=float, default=0.2, help="Fixed reward for a confirmed coin entity collection in shaped_v5.")
    parser.add_argument("--fruit-reward-value", type=float, default=2.0, help="Fixed reward for a confirmed fruit entity collection in shaped_v5.")
    parser.add_argument("--jump-start-penalty", type=float, default=0.02, help="shaped_v6 cost charged once for a real grounded jump start.")
    parser.add_argument("--gap-miss-penalty-coef", type=float, default=2.0, help="shaped_v6 maximum extra quadratic death penalty for missing the landing interval.")
    parser.add_argument("--airborne-hold-free-steps", type=int, default=4, help="shaped_v7 airborne hold decisions allowed before duration cost starts.")
    parser.add_argument("--airborne-hold-penalty", type=float, default=0.02, help="shaped_v7 cost for each airborne hold decision after the free steps.")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-steps", type=int, default=1024, help="Number of rollout steps before each PPO-LSTM update.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.003, help="Entropy bonus. Higher values encourage more exploration.")
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--lstm-hidden-size", type=int, default=128)
    parser.add_argument("--n-lstm-layers", type=int, default=1)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or datetime.now().strftime("junimo_ppo_lstm_%Y%m%d_%H%M%S")
    run_dir = Path(args.log_dir) / run_name
    checkpoint_dir = run_dir / "checkpoints"
    tensorboard_dir = run_dir / "tensorboard"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(
        JunimoKartEnv(
            host=args.host,
            port=args.port,
            frame_skip=args.frame_skip,
            observation_mode=args.observation_mode,
            gap_detection_mode=args.gap_detection_mode,
            use_semantic_features=args.semantic_features,
            use_temporal_features=args.temporal_features,
            recent_action_history=args.recent_action_history,
            reward_version=args.reward_version,
            action_mode=args.action_mode,
            macro_action_frames=args.macro_action_frames,
            macro_release_frames=args.macro_release_frames,
            score_reward_coef=args.score_reward_coef,
            coin_reward_coef=args.coin_reward_coef,
            fruit_reward_coef=args.fruit_reward_coef,
            fruit_score_threshold=args.fruit_score_threshold,
            gap_landing_confirm_steps=args.gap_landing_confirm_steps,
            gap_landing_base_reward=args.gap_landing_base_reward,
            gap_landing_width_coef=args.gap_landing_width_coef,
            progress_reward_coef=args.progress_reward_coef,
            death_penalty=args.death_penalty,
            level_complete_reward=args.level_complete_reward,
            game_complete_reward=args.game_complete_reward,
            coin_reward_value=args.coin_reward_value,
            fruit_reward_value=args.fruit_reward_value,
            jump_start_penalty=args.jump_start_penalty,
            gap_miss_penalty_coef=args.gap_miss_penalty_coef,
            airborne_hold_free_steps=args.airborne_hold_free_steps,
            airborne_hold_penalty=args.airborne_hold_penalty,
        ),
        filename=str(run_dir / "monitor.csv"),
        info_keywords=TELEMETRY_INFO_KEYS,
    )

    policy_kwargs = {
        "lstm_hidden_size": args.lstm_hidden_size,
        "n_lstm_layers": args.n_lstm_layers,
    }
    policy = "MultiInputLstmPolicy" if args.observation_mode == "multi" else "MlpLstmPolicy"

    hparams = {
        "algorithm": "PPO-LSTM",
        "implementation": "sb3_contrib.RecurrentPPO",
        "policy": policy,
        "host": args.host,
        "port": args.port,
        "timesteps": args.timesteps,
        "episodes": args.episodes,
        "episode_offset": args.episode_offset,
        "model_path": str(model_path),
        "load_model": args.load_model,
        "log_dir": str(run_dir),
        "frame_skip": args.frame_skip,
        "observation_mode": args.observation_mode,
        "gap_detection_mode": args.gap_detection_mode,
        "recent_action_history": args.recent_action_history,
        "semantic_features": args.semantic_features,
        "temporal_features": args.temporal_features,
        "reward_version": args.reward_version,
        "action_mode": args.action_mode,
        "macro_action_frames": args.macro_action_frames,
        "macro_release_frames": args.macro_release_frames,
        "score_reward_coef": args.score_reward_coef,
        "coin_reward_coef": args.coin_reward_coef,
        "fruit_reward_coef": args.fruit_reward_coef,
        "fruit_score_threshold": args.fruit_score_threshold,
        "gap_landing_confirm_steps": args.gap_landing_confirm_steps,
        "gap_landing_base_reward": args.gap_landing_base_reward,
        "gap_landing_width_coef": args.gap_landing_width_coef,
        "progress_reward_coef": args.progress_reward_coef,
        "death_penalty": args.death_penalty,
        "level_complete_reward": args.level_complete_reward,
        "game_complete_reward": args.game_complete_reward,
        "coin_reward_value": args.coin_reward_value,
        "fruit_reward_value": args.fruit_reward_value,
        "jump_start_penalty": args.jump_start_penalty,
        "gap_miss_penalty_coef": args.gap_miss_penalty_coef,
        "airborne_hold_free_steps": args.airborne_hold_free_steps,
        "airborne_hold_penalty": args.airborne_hold_penalty,
        "monitor_info_keywords": ",".join(TELEMETRY_INFO_KEYS),
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "clip_range": args.clip_range,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "max_grad_norm": args.max_grad_norm,
        "lstm_hidden_size": args.lstm_hidden_size,
        "n_lstm_layers": args.n_lstm_layers,
        "save_freq": args.save_freq,
        "save_episode_freq": args.save_episode_freq,
        "progress_episode_freq": args.progress_episode_freq,
        "progress_window": args.progress_window,
        "trace_state_print_freq": args.trace_state_print_freq,
        "trace_state_format": args.trace_state_format,
        "trace_state_simple_raw": args.trace_state_simple_raw,
        "trace_state_simple_action": args.trace_state_simple_action,
        "trace_state_csv": args.trace_state_csv,
        "trace_state_csv_freq": args.trace_state_csv_freq,
        "trace_state_max_rows": args.trace_state_max_rows,
    }
    (run_dir / "hparams.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in hparams.items()) + "\n",
        encoding="utf-8",
    )

    if args.load_model:
        custom_objects = {
            "learning_rate": args.learning_rate,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "clip_range": args.clip_range,
            "ent_coef": args.ent_coef,
            "vf_coef": args.vf_coef,
            "max_grad_norm": args.max_grad_norm,
        }
        model = RecurrentPPO.load(
            args.load_model,
            env=env,
            tensorboard_log=str(tensorboard_dir),
            verbose=1,
            custom_objects=custom_objects,
        )
    else:
        model = RecurrentPPO(
            policy,
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            max_grad_norm=args.max_grad_norm,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(tensorboard_dir),
            verbose=1,
        )

    callbacks: list[BaseCallback] = []
    if args.progress_episode_freq > 0:
        callbacks.append(
            EpisodeProgressCallback(
                log_freq_episodes=args.progress_episode_freq,
                episode_offset=args.episode_offset,
                window=args.progress_window,
                verbose=1,
            )
        )
    if args.trace_state_print_freq > 0 or args.trace_state_csv:
        callbacks.append(
            StateTraceCallback(
                action_mode=args.action_mode,
                line_format=args.trace_state_format,
                simple_binned=not args.trace_state_simple_raw,
                simple_show_action=args.trace_state_simple_action,
                print_freq=args.trace_state_print_freq,
                csv_path=Path(args.trace_state_csv) if args.trace_state_csv else None,
                csv_freq=args.trace_state_csv_freq,
                max_rows=args.trace_state_max_rows,
                verbose=1,
            )
        )
    if args.save_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=args.save_freq,
                save_path=str(checkpoint_dir),
                name_prefix="junimo_ppo_lstm_steps",
                save_replay_buffer=False,
                save_vecnormalize=True,
            )
        )
    if args.save_episode_freq > 0:
        callbacks.append(
            EpisodeCheckpointCallback(
                save_freq_episodes=args.save_episode_freq,
                save_path=checkpoint_dir,
                name_prefix="junimo_ppo_lstm",
                episode_offset=args.episode_offset,
                verbose=1,
            )
        )
    if args.episodes is not None:
        callbacks.append(StopTrainingOnMaxEpisodes(max_episodes=args.episodes, verbose=1))

    print(f"Run dir: {run_dir}")
    print(f"Monitor CSV: {run_dir / 'monitor.csv'}")
    print(f"TensorBoard dir: {tensorboard_dir}")
    print(f"Checkpoints: {checkpoint_dir}")
    print("PPO-LSTM checkpoints do not save replay buffers, so they should use far less disk space than DQN.")

    interrupted = False
    try:
        model.learn(
            total_timesteps=args.timesteps if args.episodes is None else 2_147_483_647,
            callback=CallbackList(callbacks) if callbacks else None,
            tb_log_name="ppo_lstm",
            reset_num_timesteps=args.load_model is None,
        )
    except KeyboardInterrupt:
        interrupted = True
        print("Training interrupted by Ctrl+C. Saving current PPO-LSTM model before exit...")
    finally:
        model.save(model_path)
        env.close()

    if interrupted:
        print(f"Saved interrupted PPO-LSTM model to: {model_path}")
    else:
        print(f"Saved final PPO-LSTM model to: {model_path}")


if __name__ == "__main__":
    main()
