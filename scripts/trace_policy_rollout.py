from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from junimo_rl import JunimoKartEnv
from junimo_rl.env import semantic_feature_snapshot


SEMANTIC_TRACE_FIELDS = [
    "next_track_dx",
    "next_track_y",
    "next_track_type_id",
    "next_track_has_obstacle",
    "next_gap_present",
    "next_gap_start_dx",
    "next_gap_width",
    "landing_y",
    "landing_delta_y",
    "next_obstacle_present",
    "next_obstacle_dx",
    "next_obstacle_y",
    "next_pickup_present",
    "next_pickup_dx",
    "next_pickup_y",
    "distance_to_finish",
    "progress_fraction",
]

CSV_FIELDS = [
    "episode",
    "step",
    "action",
    "action_holds_jump",
    "reward",
    "episode_reward",
    "terminated",
    "truncated",
    "in_minigame",
    "game_mode",
    "game_state",
    "game_over",
    "completed",
    "score",
    "lives_left",
    "levels_beat",
    "jump_held",
    "player_x",
    "player_y",
    "velocity_x",
    "velocity_y",
    "grounded",
    "jumping",
    *SEMANTIC_TRACE_FIELDS,
]


def _num(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _flag(value: Any) -> int:
    return 1 if bool(value) else 0


def action_holds_jump(action: int, action_mode: str) -> int:
    if action_mode == "binary":
        return 1 if action == 1 else 0
    return 1 if action > 0 else 0


def snapshot_row(
    snapshot: dict[str, Any],
    episode: int,
    step: int,
    action: int,
    reward: float,
    episode_reward: float,
    terminated: bool,
    truncated: bool,
    action_mode: str,
    semantic_features: dict[str, float] | None = None,
) -> dict[str, float | int]:
    player = snapshot.get("player") or {}
    position = player.get("position") or {}
    velocity = player.get("velocity") or {}
    semantic = semantic_features or semantic_feature_snapshot(snapshot)

    row: dict[str, float | int] = {
        "episode": episode,
        "step": step,
        "action": action,
        "action_holds_jump": action_holds_jump(action, action_mode),
        "reward": reward,
        "episode_reward": episode_reward,
        "terminated": _flag(terminated),
        "truncated": _flag(truncated),
        "in_minigame": _flag(snapshot.get("inMinigame")),
        "game_mode": _num(snapshot.get("gameMode")),
        "game_state": _num(snapshot.get("gameState")),
        "game_over": _flag(snapshot.get("gameOver")),
        "completed": _flag(snapshot.get("completed")),
        "score": _num(snapshot.get("score")),
        "lives_left": _num(snapshot.get("livesLeft")),
        "levels_beat": _num(snapshot.get("levelsBeat")),
        "jump_held": _flag(snapshot.get("jumpHeld")),
        "player_x": _num(position.get("x")),
        "player_y": _num(position.get("y")),
        "velocity_x": _num(velocity.get("x")),
        "velocity_y": _num(velocity.get("y")),
        "grounded": _flag(player.get("grounded")),
        "jumping": _flag(player.get("jumping")),
    }
    for field in SEMANTIC_TRACE_FIELDS:
        row[field] = semantic[field]
    return row


def load_policy(algorithm: str, model_path: Path, env: JunimoKartEnv):
    if algorithm == "dqn":
        from stable_baselines3 import DQN

        return DQN.load(model_path, env=env)
    if algorithm == "ppo":
        from stable_baselines3 import PPO

        return PPO.load(model_path, env=env)
    if algorithm == "ppo_lstm":
        try:
            from sb3_contrib import RecurrentPPO
        except ImportError as exc:  # pragma: no cover - only used when dependency is missing
            raise SystemExit(
                "PPO-LSTM trace requires sb3-contrib.\n"
                "Install it with: pip install sb3-contrib"
            ) from exc

        return RecurrentPPO.load(model_path, env=env)
    raise ValueError(f"Unknown algorithm: {algorithm!r}")


def predict_action(
    model,
    algorithm: str,
    obs: np.ndarray,
    lstm_states,
    episode_starts: np.ndarray,
    deterministic: bool,
):
    if algorithm == "ppo_lstm":
        action, next_lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        return int(np.asarray(action).item()), next_lstm_states

    action, _state = model.predict(obs, deterministic=deterministic)
    return int(np.asarray(action).item()), lstm_states


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a trained policy in Junimo Kart and write a per-step CSV trace for debugging."
    )
    parser.add_argument("--algorithm", choices=["dqn", "ppo", "ppo_lstm"], required=True)
    parser.add_argument("--model", required=True, help="Model checkpoint .zip path.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps-per-episode", type=int, default=0, help="Optional safety cap. Use 0 for no cap.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--observation-mode", default="flat", choices=["flat", "multi", "compact"], help="Observation format used by the model. Must match training.")
    parser.add_argument("--gap-detection-mode", default="legacy", choices=["legacy", "anchored"], help="Gap detector used by the model experiment.")
    parser.add_argument("--recent-action-history", type=int, default=12, help="Recent action history length used by multi-input models.")
    parser.add_argument("--semantic-features", action="store_true", help="Use semantic observation features. Must match training.")
    parser.add_argument("--temporal-features", action="store_true", help="Use temporal observation features. Must match training.")
    parser.add_argument("--reward-version", default="legacy", choices=["legacy", "shaped_v1", "shaped_v2", "shaped_v3", "shaped_v4", "shaped_v5", "shaped_v6", "shaped_v7"])
    parser.add_argument("--action-mode", default="binary", choices=["binary", "macro", "tap_macro"])
    parser.add_argument("--macro-action-frames", type=int, default=8)
    parser.add_argument("--macro-release-frames", type=int, default=1)
    parser.add_argument("--score-reward-coef", type=float, default=None)
    parser.add_argument("--coin-reward-coef", type=float, default=None)
    parser.add_argument("--fruit-reward-coef", type=float, default=None)
    parser.add_argument("--fruit-score-threshold", type=float, default=100.0)
    parser.add_argument("--gap-landing-confirm-steps", type=int, default=2)
    parser.add_argument("--gap-landing-base-reward", type=float, default=8.0)
    parser.add_argument("--gap-landing-width-coef", type=float, default=0.04)
    parser.add_argument("--progress-reward-coef", type=float, default=0.01)
    parser.add_argument("--death-penalty", type=float, default=5.0)
    parser.add_argument("--level-complete-reward", type=float, default=50.0)
    parser.add_argument("--game-complete-reward", type=float, default=200.0)
    parser.add_argument("--coin-reward-value", type=float, default=0.2)
    parser.add_argument("--fruit-reward-value", type=float, default=2.0)
    parser.add_argument("--jump-start-penalty", type=float, default=0.02)
    parser.add_argument("--gap-miss-penalty-coef", type=float, default=2.0)
    parser.add_argument("--airborne-hold-free-steps", type=int, default=4)
    parser.add_argument("--airborne-hold-penalty", type=float, default=0.02)
    parser.add_argument("--stochastic", action="store_true", help="Sample actions from the policy instead of deterministic evaluation.")
    parser.add_argument("--out", default="logs/policy_trace.csv")
    args = parser.parse_args()

    env = JunimoKartEnv(
        host=args.host,
        port=args.port,
        frame_skip=args.frame_skip,
        observation_mode=args.observation_mode,
        gap_detection_mode=args.gap_detection_mode,
        recent_action_history=args.recent_action_history,
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
    )
    model = load_policy(args.algorithm, Path(args.model), env)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0

    try:
        with out.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
            writer.writeheader()

            for episode in range(1, args.episodes + 1):
                obs, info = env.reset()
                done = False
                step = 0
                episode_reward = 0.0
                lstm_states = None
                episode_starts = np.ones((1,), dtype=bool)

                while not done:
                    action, lstm_states = predict_action(
                        model,
                        args.algorithm,
                        obs,
                        lstm_states,
                        episode_starts,
                        deterministic=not args.stochastic,
                    )
                    obs, reward, terminated, truncated, info = env.step(action)
                    step += 1
                    episode_reward += float(reward)
                    done = bool(terminated or truncated)
                    episode_starts = np.array([done], dtype=bool)

                    snapshot = info.get("snapshot", {})
                    writer.writerow(
                        snapshot_row(
                            snapshot=snapshot,
                            episode=episode,
                            step=step,
                            action=action,
                            reward=float(reward),
                            episode_reward=episode_reward,
                            terminated=bool(terminated),
                            truncated=bool(truncated),
                            action_mode=args.action_mode,
                            semantic_features=info.get("semantic"),
                        )
                    )
                    rows_written += 1

                    if args.max_steps_per_episode > 0 and step >= args.max_steps_per_episode:
                        break

                print(
                    f"Trace episode {episode}/{args.episodes} | "
                    f"steps={step} | reward={episode_reward:.3f} | rows={rows_written}"
                )
    except ConnectionError as exc:
        raise SystemExit(
            f"Trace stopped because the Junimo Kart bridge disconnected.\n{exc}\n\n"
            "Buka Stardew lewat SMAPI, load save, masuk Junimo Kart, lalu ulang command trace."
        ) from exc
    finally:
        env.close()

    print(f"Saved policy trace CSV to: {out}")


if __name__ == "__main__":
    main()
