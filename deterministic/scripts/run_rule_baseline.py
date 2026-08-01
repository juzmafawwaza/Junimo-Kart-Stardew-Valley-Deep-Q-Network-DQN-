from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from junimo_det.controller import RuleBasedController, RuleConfig
from junimo_rl import JunimoKartEnv


FIELDNAMES = [
    "episode",
    "total_reward",
    "length",
    "score",
    "levels_beat",
    "completed",
    "game_over",
    "jump_hold_ratio",
    "final_player_x",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a simple deterministic Junimo Kart baseline.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--reward-version", default="shaped_v2", choices=["legacy", "shaped_v1", "shaped_v2"])
    parser.add_argument("--action-mode", default="macro", choices=["binary", "macro"])
    parser.add_argument("--macro-action-frames", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--gap-trigger-dx", type=float, default=25.0)
    parser.add_argument("--obstacle-trigger-dx", type=float, default=95.0)
    parser.add_argument("--out", default="outputs/deterministic/rule_baseline.csv")
    args = parser.parse_args()

    config = RuleConfig(
        action_mode=args.action_mode,
        gap_trigger_dx=args.gap_trigger_dx,
        obstacle_trigger_dx=args.obstacle_trigger_dx,
    )
    controller = RuleBasedController(config)
    env = JunimoKartEnv(
        host=args.host,
        port=args.port,
        frame_skip=args.frame_skip,
        use_semantic_features=True,
        reward_version=args.reward_version,
        action_mode=args.action_mode,
        macro_action_frames=args.macro_action_frames,
    )

    rows: list[dict[str, float | int]] = []
    try:
        for episode in range(1, max(args.episodes, 1) + 1):
            _obs, info = env.reset()
            done = False
            total_reward = 0.0
            length = 0
            jump_actions = 0
            final_snapshot = info.get("snapshot", {})

            while not done and length < max(args.max_steps, 1):
                snapshot = info.get("snapshot", final_snapshot)
                action = controller.decide(snapshot)
                _obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                length += 1
                jump_actions += int(action == 1 if args.action_mode == "binary" else action > 0)
                done = bool(terminated or truncated)
                final_snapshot = info.get("snapshot", final_snapshot)

            player = final_snapshot.get("player") or {}
            position = player.get("position") or {}
            row = {
                "episode": episode,
                "total_reward": total_reward,
                "length": length,
                "score": int(final_snapshot.get("score") or 0),
                "levels_beat": int(final_snapshot.get("levelsBeat") or 0),
                "completed": int(bool(final_snapshot.get("completed"))),
                "game_over": int(bool(final_snapshot.get("gameOver"))),
                "jump_hold_ratio": jump_actions / length if length else 0.0,
                "final_player_x": float(position.get("x") or 0.0),
            }
            rows.append(row)
            print(row)
    finally:
        env.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved rule baseline CSV to: {out}")


if __name__ == "__main__":
    main()
