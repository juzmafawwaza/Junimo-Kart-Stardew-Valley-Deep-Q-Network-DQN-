from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from sb3_contrib import RecurrentPPO
except ImportError as exc:  # pragma: no cover - only used when dependency is missing
    raise SystemExit(
        "PPO-LSTM evaluation requires sb3-contrib.\n"
        "Install it with one of these commands:\n"
        "  pip install sb3-contrib\n"
        "  pip install -e \".[train,analysis]\""
    ) from exc

from junimo_rl import JunimoKartEnv, TELEMETRY_INFO_KEYS


def evaluate_model(
    model_path: Path,
    episodes: int,
    host: str,
    port: int,
    frame_skip: int,
    observation_mode: str,
    gap_detection_mode: str,
    recent_action_history: int,
    use_semantic_features: bool,
    use_temporal_features: bool,
    reward_version: str,
    action_mode: str,
    macro_action_frames: int,
    macro_release_frames: int,
    score_reward_coef: float | None,
    coin_reward_coef: float | None,
    fruit_reward_coef: float | None,
    fruit_score_threshold: float,
    gap_landing_confirm_steps: int,
    gap_landing_base_reward: float,
    gap_landing_width_coef: float,
    progress_reward_coef: float,
    death_penalty: float,
    level_complete_reward: float,
    game_complete_reward: float,
    coin_reward_value: float,
    fruit_reward_value: float,
    jump_start_penalty: float,
    gap_miss_penalty_coef: float,
    airborne_hold_free_steps: int,
    airborne_hold_penalty: float,
    max_steps_per_episode: int,
    deterministic: bool,
) -> dict[str, float | str | None]:
    env = JunimoKartEnv(
        host=host,
        port=port,
        frame_skip=frame_skip,
        observation_mode=observation_mode,
        gap_detection_mode=gap_detection_mode,
        recent_action_history=recent_action_history,
        use_semantic_features=use_semantic_features,
        use_temporal_features=use_temporal_features,
        reward_version=reward_version,
        action_mode=action_mode,
        macro_action_frames=macro_action_frames,
        macro_release_frames=macro_release_frames,
        score_reward_coef=score_reward_coef,
        coin_reward_coef=coin_reward_coef,
        fruit_reward_coef=fruit_reward_coef,
        fruit_score_threshold=fruit_score_threshold,
        gap_landing_confirm_steps=gap_landing_confirm_steps,
        gap_landing_base_reward=gap_landing_base_reward,
        gap_landing_width_coef=gap_landing_width_coef,
        progress_reward_coef=progress_reward_coef,
        death_penalty=death_penalty,
        level_complete_reward=level_complete_reward,
        game_complete_reward=game_complete_reward,
        coin_reward_value=coin_reward_value,
        fruit_reward_value=fruit_reward_value,
        jump_start_penalty=jump_start_penalty,
        gap_miss_penalty_coef=gap_miss_penalty_coef,
        airborne_hold_free_steps=airborne_hold_free_steps,
        airborne_hold_penalty=airborne_hold_penalty,
    )
    model = RecurrentPPO.load(model_path, env=env)

    rewards: list[float] = []
    lengths: list[int] = []
    scores: list[int] = []
    levels: list[int] = []
    completed = 0
    gameovers = 0
    total_jump_actions = 0
    total_actions = 0
    telemetry_totals = {key: 0.0 for key in TELEMETRY_INFO_KEYS}
    capped_episodes = 0

    for _episode in range(episodes):
        obs, info = env.reset()
        done = False
        episode_reward = 0.0
        episode_length = 0
        final_snapshot = info.get("snapshot", {})
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)

        while not done:
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=deterministic,
            )
            action_int = int(action)
            obs, reward, terminated, truncated, info = env.step(action_int)
            episode_reward += float(reward)
            episode_length += 1
            total_actions += 1
            total_jump_actions += 1 if (action_int == 1 if action_mode == "binary" else action_int > 0) else 0
            done = bool(terminated or truncated)
            if max_steps_per_episode > 0 and episode_length >= max_steps_per_episode:
                capped_episodes += 1
                done = True
            episode_starts = np.array([done], dtype=bool)
            final_snapshot = info.get("snapshot", final_snapshot)

        rewards.append(episode_reward)
        lengths.append(episode_length)
        scores.append(int(final_snapshot.get("score") or 0))
        levels.append(int(final_snapshot.get("levelsBeat") or 0))
        if final_snapshot.get("completed"):
            completed += 1
        if final_snapshot.get("gameOver"):
            gameovers += 1
        for key in TELEMETRY_INFO_KEYS:
            telemetry_totals[key] += float(info.get(key, 0.0))

    env.close()
    telemetry_action_total = sum(telemetry_totals[f"action_{idx}_count"] for idx in range(4))
    gap_attempts = telemetry_totals["gap_attempts"]

    row = {
        "algorithm": "PPO-LSTM",
        "deterministic": deterministic,
        "model": str(model_path),
        "observation_mode": observation_mode,
        "gap_detection_mode": gap_detection_mode,
        "recent_action_history": recent_action_history,
        "semantic_features": use_semantic_features,
        "temporal_features": use_temporal_features,
        "reward_version": reward_version,
        "action_mode": action_mode,
        "macro_action_frames": macro_action_frames,
        "macro_release_frames": macro_release_frames,
        "score_reward_coef": score_reward_coef,
        "coin_reward_coef": coin_reward_coef,
        "fruit_reward_coef": fruit_reward_coef,
        "fruit_score_threshold": fruit_score_threshold,
        "gap_landing_confirm_steps": gap_landing_confirm_steps,
        "gap_landing_base_reward": gap_landing_base_reward,
        "gap_landing_width_coef": gap_landing_width_coef,
        "progress_reward_coef": progress_reward_coef,
        "death_penalty": death_penalty,
        "level_complete_reward": level_complete_reward,
        "game_complete_reward": game_complete_reward,
        "coin_reward_value": coin_reward_value,
        "fruit_reward_value": fruit_reward_value,
        "jump_start_penalty": jump_start_penalty,
        "gap_miss_penalty_coef": gap_miss_penalty_coef,
        "airborne_hold_free_steps": airborne_hold_free_steps,
        "airborne_hold_penalty": airborne_hold_penalty,
        "max_steps_per_episode": max_steps_per_episode,
        "episodes": episodes,
        "capped_episode_rate": capped_episodes / episodes,
        "mean_reward": sum(rewards) / len(rewards),
        "mean_length": sum(lengths) / len(lengths),
        "mean_score": sum(scores) / len(scores),
        "max_score": max(scores),
        "mean_levels_beat": sum(levels) / len(levels),
        "max_levels_beat": max(levels),
        "completion_rate": completed / episodes,
        "gameover_rate": gameovers / episodes,
        "jump_hold_ratio": total_jump_actions / total_actions if total_actions else 0.0,
        "mean_gap_attempts": gap_attempts / episodes,
        "mean_gap_landings": telemetry_totals["gap_landings"] / episodes,
        "gap_landing_rate": telemetry_totals["gap_landings"] / gap_attempts if gap_attempts else 0.0,
        "mean_gap_failures": telemetry_totals["gap_failures"] / episodes,
        "death_near_gap_rate": telemetry_totals["death_near_gap"] / episodes,
        "death_near_obstacle_rate": telemetry_totals["death_near_obstacle"] / episodes,
        "mean_pickup_events": telemetry_totals["pickup_events"] / episodes,
        "mean_coin_events": telemetry_totals["coin_events"] / episodes,
        "mean_fruit_events": telemetry_totals["fruit_events"] / episodes,
        "mean_coin_reward_total": telemetry_totals["coin_reward_total"] / episodes,
        "mean_fruit_reward_total": telemetry_totals["fruit_reward_total"] / episodes,
        "mean_score_delta_total": telemetry_totals["score_delta_total"] / episodes,
        "mean_max_episode_x": telemetry_totals["max_episode_x"] / episodes,
        "mean_jump_starts": telemetry_totals["jump_start_events"] / episodes,
        "jump_start_near_gap_rate": telemetry_totals["jump_start_near_gap_events"] / telemetry_totals["jump_start_events"] if telemetry_totals["jump_start_events"] else 0.0,
        "jump_start_without_near_gap_rate": telemetry_totals["jump_start_without_near_gap_events"] / telemetry_totals["jump_start_events"] if telemetry_totals["jump_start_events"] else 0.0,
        "mean_jump_start_penalty_total": telemetry_totals["jump_start_penalty_total"] / episodes,
        "mean_gap_miss_deaths": telemetry_totals["gap_miss_deaths"] / episodes,
        "mean_gap_miss_distance": telemetry_totals["gap_miss_distance_total"] / telemetry_totals["gap_miss_deaths"] if telemetry_totals["gap_miss_deaths"] else 0.0,
        "mean_gap_miss_ratio": telemetry_totals["gap_miss_ratio_total"] / telemetry_totals["gap_miss_deaths"] if telemetry_totals["gap_miss_deaths"] else 0.0,
        "mean_gap_miss_penalty_total": telemetry_totals["gap_miss_penalty_total"] / episodes,
        "mean_airborne_hold_penalty_steps": telemetry_totals["airborne_hold_penalty_steps"] / episodes,
        "mean_airborne_hold_penalty_total": telemetry_totals["airborne_hold_penalty_total"] / episodes,
        "mean_max_jump_hold_steps": telemetry_totals["max_jump_hold_steps"] / episodes,
    }
    for idx in range(4):
        action_count = telemetry_totals[f"action_{idx}_count"]
        row[f"action_{idx}_ratio"] = action_count / telemetry_action_total if telemetry_action_total else 0.0
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one or more Junimo Kart PPO-LSTM checkpoints deterministically.")
    parser.add_argument("models", nargs="+", help="PPO-LSTM model checkpoint .zip files.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--frame-skip", type=int, default=2)
    parser.add_argument("--observation-mode", default="flat", choices=["flat", "multi", "compact"], help="Observation format used by the model. Required to match training.")
    parser.add_argument("--gap-detection-mode", default="legacy", choices=["legacy", "anchored"], help="Gap detector used during training. Required to match the model experiment.")
    parser.add_argument("--recent-action-history", type=int, default=12, help="Recent action history length used by multi-input models.")
    parser.add_argument("--semantic-features", action="store_true", help="Use the semantic observation vector. Required for models trained with --semantic-features.")
    parser.add_argument("--temporal-features", action="store_true", help="Use timing features. Required for models trained with --temporal-features.")
    parser.add_argument("--reward-version", default="legacy", choices=["legacy", "shaped_v1", "shaped_v2", "shaped_v3", "shaped_v4", "shaped_v5", "shaped_v6", "shaped_v7"], help="Reward function to use during evaluation.")
    parser.add_argument("--action-mode", default="binary", choices=["binary", "macro", "tap_macro"], help="Action representation used by the model. Required to match training.")
    parser.add_argument("--macro-action-frames", type=int, default=8, help="Macro action frame window used by models trained with --action-mode macro/tap_macro.")
    parser.add_argument("--macro-release-frames", type=int, default=1, help="Release frames used by models trained with --action-mode tap_macro.")
    parser.add_argument("--score-reward-coef", type=float, default=None, help="Score reward coefficient used by the evaluation environment.")
    parser.add_argument("--coin-reward-coef", type=float, default=None, help="Coin pickup reward coefficient used by the evaluation environment.")
    parser.add_argument("--fruit-reward-coef", type=float, default=None, help="Fruit pickup reward coefficient used by the evaluation environment.")
    parser.add_argument("--fruit-score-threshold", type=float, default=100.0, help="Fallback: unknown pickup score_delta at/above this value is treated as fruit.")
    parser.add_argument("--gap-landing-confirm-steps", type=int, default=2, help="Gap landing confirmation steps used by shaped_v3/shaped_v4 evaluation.")
    parser.add_argument("--gap-landing-base-reward", type=float, default=8.0, help="Base reward paid for a confirmed gap landing in shaped_v4.")
    parser.add_argument("--gap-landing-width-coef", type=float, default=0.04, help="Extra gap landing reward per pixel of gap width in shaped_v4.")
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
    parser.add_argument("--max-steps-per-episode", type=int, default=300, help="Safety cap for live-game evaluation. Set 0 to disable.")
    parser.add_argument("--stochastic", action="store_true", help="Sample policy actions instead of always taking the highest-probability action.")
    parser.add_argument("--out", default="logs/ppo_lstm/evaluation.csv")
    args = parser.parse_args()

    model_paths: list[Path] = []
    for pattern in args.models:
        matches = [Path(match) for match in glob.glob(pattern)]
        if matches:
            model_paths.extend(matches)
            continue
        if any(char in pattern for char in "*?[]"):
            raise SystemExit(
                f"No PPO-LSTM checkpoint matched pattern: {pattern}\n"
                "Cek nama folder run dan pastikan folder checkpoints berisi file .zip."
            )
        model_paths.append(Path(pattern))

    rows = []
    for model in model_paths:
        try:
            print(f"Evaluating PPO-LSTM checkpoint: {model}", flush=True)
            rows.append(
                evaluate_model(
                    model,
                    args.episodes,
                    args.host,
                    args.port,
                    args.frame_skip,
                    args.observation_mode,
                    args.gap_detection_mode,
                    args.recent_action_history,
                    args.semantic_features,
                    args.temporal_features,
                    args.reward_version,
                    args.action_mode,
                    args.macro_action_frames,
                    args.macro_release_frames,
                    args.score_reward_coef,
                    args.coin_reward_coef,
                    args.fruit_reward_coef,
                    args.fruit_score_threshold,
                    args.gap_landing_confirm_steps,
                    args.gap_landing_base_reward,
                    args.gap_landing_width_coef,
                    args.progress_reward_coef,
                    args.death_penalty,
                    args.level_complete_reward,
                    args.game_complete_reward,
                    args.coin_reward_value,
                    args.fruit_reward_value,
                    args.jump_start_penalty,
                    args.gap_miss_penalty_coef,
                    args.airborne_hold_free_steps,
                    args.airborne_hold_penalty,
                    args.max_steps_per_episode,
                    not args.stochastic,
                )
            )
            print(rows[-1], flush=True)
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

    print(f"Saved PPO-LSTM evaluation CSV to: {out}")


if __name__ == "__main__":
    main()
