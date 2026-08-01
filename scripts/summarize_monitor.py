from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean


ACTION_KEYS = [f"action_{idx}_count" for idx in range(4)]


def read_monitor_csv(path: Path) -> list[dict[str, str]]:
    lines: list[str] = []
    with path.open(newline="", encoding="utf-8") as file:
        for line in file:
            if not line.startswith("#"):
                lines.append(line)
    return list(csv.DictReader(lines))


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return 0.0
    return float(value)


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def summarize(rows: list[dict[str, str]], window: int) -> dict[str, float]:
    selected = rows[-window:] if window > 0 else rows
    if not selected:
        return {}

    action_totals = {key: sum(number(row, key) for row in selected) for key in ACTION_KEYS}
    total_actions = sum(action_totals.values())
    gap_attempts = sum(number(row, "gap_attempts") for row in selected)
    gap_landings = sum(number(row, "gap_landings") for row in selected)
    jump_starts = sum(number(row, "jump_start_events") for row in selected)
    no_gap_jump_starts = sum(number(row, "jump_start_without_near_gap_events") for row in selected)
    gap_takeoffs = sum(number(row, "gap_takeoff_events") for row in selected)
    edge_landings = sum(number(row, "edge_qualified_landings") for row in selected)

    summary = {
        "episodes": float(len(selected)),
        "mean_reward": mean(number(row, "r") for row in selected),
        "mean_length": mean(number(row, "l") for row in selected),
        "best_length": max(number(row, "l") for row in selected),
        "mean_gap_attempts": gap_attempts / len(selected),
        "mean_gap_landings": gap_landings / len(selected),
        "gap_landing_rate": ratio(gap_landings, gap_attempts),
        "mean_gap_failures": mean(number(row, "gap_failures") for row in selected),
        "death_near_gap_rate": mean(number(row, "death_near_gap") for row in selected),
        "death_near_obstacle_rate": mean(number(row, "death_near_obstacle") for row in selected),
        "mean_pickup_events": mean(number(row, "pickup_events") for row in selected),
        "mean_coin_events": mean(number(row, "coin_events") for row in selected),
        "mean_fruit_events": mean(number(row, "fruit_events") for row in selected),
        "mean_coin_reward_total": mean(number(row, "coin_reward_total") for row in selected),
        "mean_fruit_reward_total": mean(number(row, "fruit_reward_total") for row in selected),
        "mean_score_delta_total": mean(number(row, "score_delta_total") for row in selected),
        "mean_max_episode_x": mean(number(row, "max_episode_x") for row in selected),
        "mean_jump_starts": jump_starts / len(selected),
        "jump_start_without_near_gap_rate": ratio(no_gap_jump_starts, jump_starts),
        "mean_airborne_hold_penalty_steps": mean(number(row, "airborne_hold_penalty_steps") for row in selected),
        "mean_airborne_hold_penalty_total": mean(number(row, "airborne_hold_penalty_total") for row in selected),
        "mean_max_jump_hold_steps": mean(number(row, "max_jump_hold_steps") for row in selected),
        "mean_grounded_progress_bonus_total": mean(number(row, "grounded_progress_bonus_total") for row in selected),
        "mean_unnecessary_jump_events": mean(number(row, "unnecessary_jump_events") for row in selected),
        "mean_unnecessary_jump_penalty_total": mean(number(row, "unnecessary_jump_penalty_total") for row in selected),
        "mean_non_gap_airborne_steps": mean(number(row, "non_gap_airborne_steps") for row in selected),
        "mean_non_gap_airborne_penalty_total": mean(number(row, "non_gap_airborne_penalty_total") for row in selected),
        "mean_takeoff_tip_quality": ratio(sum(number(row, "takeoff_tip_quality_total") for row in selected), gap_takeoffs),
        "mean_successful_takeoff_tip_quality": ratio(sum(number(row, "successful_takeoff_tip_quality_total") for row in selected), edge_landings),
        "mean_landing_tip_quality": ratio(sum(number(row, "landing_tip_quality_total") for row in selected), edge_landings),
        "mean_edge_technique_reward_total": mean(number(row, "edge_technique_reward_total") for row in selected),
        "mean_takeoff_target_distance": ratio(sum(number(row, "takeoff_target_distance_total") for row in selected), gap_takeoffs),
        "mean_successful_takeoff_distance": ratio(sum(number(row, "successful_takeoff_tip_distance_total") for row in selected), edge_landings),
        "mean_successful_takeoff_target_distance": ratio(sum(number(row, "successful_takeoff_target_distance_total") for row in selected), edge_landings),
        "mean_gap_inaction_steps": mean(number(row, "gap_inaction_steps") for row in selected),
        "mean_gap_inaction_penalty_total": mean(number(row, "gap_inaction_penalty_total") for row in selected),
    }
    for key, total in action_totals.items():
        summary[f"{key.replace('_count', '')}_ratio"] = ratio(total, total_actions)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Junimo Kart monitor.csv reward, length, and telemetry.")
    parser.add_argument("monitor_csv", type=Path)
    parser.add_argument("--window", type=int, default=100, help="Number of latest episodes to summarize. Use 0 for all rows.")
    args = parser.parse_args()

    rows = read_monitor_csv(args.monitor_csv)
    summary = summarize(rows, args.window)
    if not summary:
        raise SystemExit(f"No rows found in {args.monitor_csv}")

    print(f"Monitor: {args.monitor_csv}")
    print(f"Rows in file: {len(rows)}")
    print(f"Rows summarized: {int(summary['episodes'])}")
    for key, value in summary.items():
        if key == "episodes":
            continue
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
