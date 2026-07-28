from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stable_baselines3 import DQN

from junimo_rl import JunimoKartEnv


def evaluate_model(model_path: Path, episodes: int, host: str, port: int, frame_skip: int) -> dict[str, float | str]:
    env = JunimoKartEnv(host=host, port=port, frame_skip=frame_skip)
    model = DQN.load(model_path, env=env)

    rewards: list[float] = []
    lengths: list[int] = []
    completed = 0
    max_levels = 0

    for _episode in range(episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        final_snapshot = info.get("snapshot", {})

        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += float(reward)
            episode_length += 1
            done = bool(terminated or truncated)
            final_snapshot = info.get("snapshot", final_snapshot)

        rewards.append(episode_reward)
        lengths.append(episode_length)
        if final_snapshot.get("completed"):
            completed += 1
        max_levels = max(max_levels, int(final_snapshot.get("levelsBeat") or 0))

    env.close()

    return {
        "model": str(model_path),
        "episodes": episodes,
        "mean_reward": sum(rewards) / len(rewards),
        "mean_length": sum(lengths) / len(lengths),
        "completion_rate": completed / episodes,
        "max_levels_beat": max_levels,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one or more Junimo Kart DQN checkpoints deterministically.")
    parser.add_argument("models", nargs="+", help="Model checkpoint .zip files.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--out", default="logs/evaluation.csv")
    args = parser.parse_args()

    model_paths: list[Path] = []
    for pattern in args.models:
        matches = [Path(match) for match in glob.glob(pattern)]
        model_paths.extend(matches or [Path(pattern)])

    rows = []
    for model in model_paths:
        try:
            rows.append(evaluate_model(model, args.episodes, args.host, args.port, args.frame_skip))
        except ConnectionError as exc:
            raise SystemExit(
                f"Evaluation stopped because the bridge disconnected while evaluating {model}.\n"
                f"{exc}\n\n"
                "Buka Stardew lewat SMAPI, load save sampai masuk farm/world, lalu jalankan command evaluate lagi."
            ) from exc

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)
    print(f"Saved evaluation CSV to: {out}")


if __name__ == "__main__":
    main()
