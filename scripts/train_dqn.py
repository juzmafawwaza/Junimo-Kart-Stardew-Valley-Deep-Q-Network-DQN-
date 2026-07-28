from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, StopTrainingOnMaxEpisodes
from stable_baselines3.common.monitor import Monitor

from junimo_rl import JunimoKartEnv


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
    parser.add_argument("--frame-skip", type=int, default=4)
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
        JunimoKartEnv(host=args.host, port=args.port, frame_skip=args.frame_skip),
        filename=str(run_dir / "monitor.csv"),
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
                save_replay_buffer=True,
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
                save_replay_buffer=True,
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
