from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, StopTrainingOnMaxEpisodes
from stable_baselines3.common.monitor import Monitor

from junimo_rl import JunimoKartEnv, TELEMETRY_INFO_KEYS


class EpisodeCheckpointCallback(BaseCallback):
    def __init__(
        self,
        save_freq_episodes: int,
        save_path: Path,
        name_prefix: str = "junimo_dqn",
        episode_offset: int = 0,
        save_replay_buffer: bool = True,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.save_freq_episodes = save_freq_episodes
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.episode_count = max(episode_offset, 0)
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

            if self.save_replay_buffer and hasattr(self.model, "save_replay_buffer"):
                self.model.save_replay_buffer(self.save_path / f"{stem}_replay_buffer.pkl")

            if self.verbose > 0:
                print(f"Saved episode checkpoint: {model_file}")

            self.next_checkpoint_episode += self.save_freq_episodes

        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DQN agent against Junimo Kart through the SMAPI bridge.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--timesteps", type=int, default=100_000)
    parser.add_argument("--episodes", type=int, default=None, help="Optional episode target; training stops after this many completed episodes.")
    parser.add_argument("--episode-offset", type=int, default=0, help="Episode number to add to checkpoint names when continuing a previous run.")
    parser.add_argument("--model-path", default="models/junimo_dqn")
    parser.add_argument("--load-model", default=None, help="Optional existing model .zip path to continue training.")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--save-freq", type=int, default=5_000)
    parser.add_argument("--save-episode-freq", type=int, default=100, help="Save a comparable model checkpoint every N completed episodes. Set 0 to disable.")
    parser.add_argument("--save-replay-buffer", action="store_true", help="Also save DQN replay buffers. This can use a lot of disk space.")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--semantic-features", action="store_true", help="Append engineered gap/landing/obstacle/pickup features to the observation vector. Models trained without this flag are not compatible with it.")
    parser.add_argument("--temporal-features", action="store_true", help="Append timing features such as jump-held duration, airborne duration, grounded duration, and previous action.")
    parser.add_argument("--reward-version", default="legacy", choices=["legacy", "shaped_v1", "shaped_v2", "shaped_v3", "shaped_v4"], help="Reward function to use.")
    parser.add_argument("--action-mode", default="binary", choices=["binary", "macro", "tap_macro"], help="Action representation. binary has release/hold; macro adds short/medium/long hold; tap_macro forces a release tail after each jump macro.")
    parser.add_argument("--macro-action-frames", type=int, default=8, help="Total game frames controlled by each macro action when --action-mode macro/tap_macro is used.")
    parser.add_argument("--macro-release-frames", type=int, default=1, help="For --action-mode tap_macro, force this many release frames at the end of every jump macro.")
    parser.add_argument("--score-reward-coef", type=float, default=None, help="Override score reward coefficient. Leave unset for the reward-version default; use 0.0 to ignore coin/fruit score.")
    parser.add_argument("--coin-reward-coef", type=float, default=None, help="Reward coefficient for detected coin pickups. When set, split pickup reward is used instead of generic score reward.")
    parser.add_argument("--fruit-reward-coef", type=float, default=None, help="Reward coefficient for detected fruit pickups. Use a higher value than coin if fruit should matter more.")
    parser.add_argument("--fruit-score-threshold", type=float, default=100.0, help="Fallback: unknown pickup score_delta at/above this value is treated as fruit.")
    parser.add_argument("--gap-landing-confirm-steps", type=int, default=2, help="For shaped_v3, require this many extra env steps alive after landing before paying the gap landing reward.")
    parser.add_argument("--gap-landing-base-reward", type=float, default=8.0, help="Base reward paid for a confirmed gap landing in shaped_v4.")
    parser.add_argument("--gap-landing-width-coef", type=float, default=0.04, help="Extra gap landing reward per pixel of gap width in shaped_v4.")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--buffer-size", type=int, default=50_000)
    parser.add_argument("--learning-starts", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--target-update-interval", type=int, default=2_000)
    parser.add_argument("--exploration-fraction", type=float, default=0.25)
    parser.add_argument("--exploration-final-eps", type=float, default=0.05)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    run_name = args.run_name or datetime.now().strftime("junimo_dqn_%Y%m%d_%H%M%S")
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
            use_semantic_features=args.semantic_features,
            use_temporal_features=args.temporal_features,
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
        ),
        filename=str(run_dir / "monitor.csv"),
        info_keywords=TELEMETRY_INFO_KEYS,
    )

    hparams = {
        "host": args.host,
        "port": args.port,
        "timesteps": args.timesteps,
        "episodes": args.episodes,
        "episode_offset": args.episode_offset,
        "model_path": str(model_path),
        "load_model": args.load_model,
        "frame_skip": args.frame_skip,
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
        "monitor_info_keywords": ",".join(TELEMETRY_INFO_KEYS),
        "learning_rate": args.learning_rate,
        "buffer_size": args.buffer_size,
        "learning_starts": args.learning_starts,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "train_freq": args.train_freq,
        "target_update_interval": args.target_update_interval,
        "exploration_fraction": args.exploration_fraction,
        "exploration_final_eps": args.exploration_final_eps,
        "save_freq": args.save_freq,
        "save_episode_freq": args.save_episode_freq,
        "save_replay_buffer": args.save_replay_buffer,
    }
    (run_dir / "hparams.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in hparams.items()) + "\n",
        encoding="utf-8",
    )

    if args.load_model:
        model = DQN.load(
            args.load_model,
            env=env,
            tensorboard_log=str(tensorboard_dir),
            verbose=1,
        )
    else:
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            gamma=args.gamma,
            train_freq=args.train_freq,
            target_update_interval=args.target_update_interval,
            exploration_fraction=args.exploration_fraction,
            exploration_final_eps=args.exploration_final_eps,
            tensorboard_log=str(tensorboard_dir),
            verbose=1,
        )

    callbacks: list[BaseCallback] = []
    if args.save_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=args.save_freq,
                save_path=str(checkpoint_dir),
                name_prefix="junimo_dqn_steps",
                save_replay_buffer=args.save_replay_buffer,
                save_vecnormalize=True,
            )
        )
    if args.save_episode_freq > 0:
        callbacks.append(
            EpisodeCheckpointCallback(
                save_freq_episodes=args.save_episode_freq,
                save_path=checkpoint_dir,
                name_prefix="junimo_dqn",
                episode_offset=args.episode_offset,
                save_replay_buffer=args.save_replay_buffer,
                verbose=1,
            )
        )
    if args.episodes is not None:
        callbacks.append(StopTrainingOnMaxEpisodes(max_episodes=args.episodes, verbose=1))

    print(f"Run dir: {run_dir}")
    print(f"Monitor CSV: {run_dir / 'monitor.csv'}")
    print(f"TensorBoard dir: {tensorboard_dir}")
    print(f"Checkpoints: {checkpoint_dir}")

    model.learn(
        total_timesteps=args.timesteps if args.episodes is None else 2_147_483_647,
        callback=CallbackList(callbacks) if callbacks else None,
        tb_log_name="dqn",
        reset_num_timesteps=args.load_model is None,
    )
    model.save(model_path)
    env.close()
    print(f"Saved final model to: {model_path}")


if __name__ == "__main__":
    main()
