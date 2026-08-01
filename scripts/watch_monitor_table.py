from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from statistics import mean


ACTION_KEYS = [f"action_{idx}_count" for idx in range(4)]
DEFAULT_COLUMNS = [
    "iter",
    "episodes",
    "n",
    "rew_mean",
    "len_mean",
    "best_len",
    "max_x_mean",
    "grounded_pct",
    "jump_held_pct",
    "jump_starts_ep",
    "no_gap_jump_pct",
    "max_hold_steps",
    "hold_cost_ep",
    "ground_bonus_ep",
    "unneeded_jump_ep",
    "non_gap_air_ep",
    "takeoff_tip_q",
    "landing_tip_q",
    "tip_reward_ep",
    "takeoff_target",
    "success_takeoff",
    "gap_wait_ep",
    "gap_visible_pct",
    "gap_near_pct",
    "gap_dx_mean",
    "gap_w_mean",
    "landing_dy_mean",
    "obs_near_pct",
    "pickup_visible_pct",
    "gap_att_ep",
    "gap_land_rate",
    "death_gap_rate",
    "death_obs_rate",
    "pickup_ep",
    "coin_ep",
    "fruit_ep",
    "coin_rew",
    "fruit_rew",
    "score_mean",
    "a0_pct",
    "a1_pct",
    "a2_pct",
    "a3_pct",
    "final_gap_dx",
    "final_gap_w",
    "final_obs_dx",
]


def read_monitor_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []

    lines: list[str] = []
    with path.open(newline="", encoding="utf-8") as file:
        for line in file:
            if line.startswith("#"):
                continue
            if not line.endswith("\n"):
                continue
            lines.append(line)

    if len(lines) <= 1:
        return []

    return list(csv.DictReader(lines))


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def find_latest_monitor(root: Path) -> Path | None:
    candidates = sorted(
        root.glob("**/monitor.csv"),
        key=lambda path: path.stat().st_mtime if path.exists() else 0.0,
        reverse=True,
    )
    return candidates[0] if candidates else None


def summarize_window(rows: list[dict[str, str]], start_idx: int, end_idx: int, iteration: int) -> dict[str, float | str]:
    selected = rows[start_idx:end_idx]
    action_totals = {key: sum(number(row, key) for row in selected) for key in ACTION_KEYS}
    total_actions = sum(action_totals.values())
    gap_attempts = sum(number(row, "gap_attempts") for row in selected)
    gap_landings = sum(number(row, "gap_landings") for row in selected)
    jump_starts = sum(number(row, "jump_start_events") for row in selected)
    gap_takeoffs = sum(number(row, "gap_takeoff_events") for row in selected)
    edge_landings = sum(number(row, "edge_qualified_landings") for row in selected)
    state_samples = sum(number(row, "state_samples") for row in selected)
    gap_visible_steps = sum(number(row, "gap_visible_steps") for row in selected)
    obstacle_visible_steps = sum(number(row, "obstacle_visible_steps") for row in selected)
    pickup_visible_steps = sum(number(row, "pickup_visible_steps") for row in selected)

    summary: dict[str, float | str] = {
        "iter": iteration,
        "episodes": f"{start_idx + 1}-{end_idx}",
        "n": len(selected),
        "rew_mean": mean(number(row, "r") for row in selected),
        "len_mean": mean(number(row, "l") for row in selected),
        "best_len": max(number(row, "l") for row in selected),
        "max_x_mean": mean(number(row, "max_episode_x") for row in selected),
        "gap_att_ep": gap_attempts / len(selected),
        "gap_land_rate": ratio(gap_landings, gap_attempts),
        "death_gap_rate": mean(number(row, "death_near_gap") for row in selected),
        "death_obs_rate": mean(number(row, "death_near_obstacle") for row in selected),
        "pickup_ep": mean(number(row, "pickup_events") for row in selected),
        "coin_ep": mean(number(row, "coin_events") for row in selected),
        "fruit_ep": mean(number(row, "fruit_events") for row in selected),
        "coin_rew": mean(number(row, "coin_reward_total") for row in selected),
        "fruit_rew": mean(number(row, "fruit_reward_total") for row in selected),
        "score_mean": mean(number(row, "score_delta_total") for row in selected),
        "jump_starts_ep": jump_starts / len(selected),
        "no_gap_jump_pct": ratio(
            sum(number(row, "jump_start_without_near_gap_events") for row in selected),
            jump_starts,
        ) * 100.0,
        "max_hold_steps": mean(number(row, "max_jump_hold_steps") for row in selected),
        "hold_cost_ep": mean(number(row, "airborne_hold_penalty_total") for row in selected),
        "ground_bonus_ep": mean(number(row, "grounded_progress_bonus_total") for row in selected),
        "unneeded_jump_ep": mean(number(row, "unnecessary_jump_events") for row in selected),
        "non_gap_air_ep": mean(number(row, "non_gap_airborne_steps") for row in selected),
        "takeoff_tip_q": ratio(
            sum(number(row, "takeoff_tip_quality_total") for row in selected),
            gap_takeoffs,
        ),
        "landing_tip_q": ratio(
            sum(number(row, "landing_tip_quality_total") for row in selected),
            edge_landings,
        ),
        "tip_reward_ep": mean(number(row, "edge_technique_reward_total") for row in selected),
        "takeoff_target": ratio(
            sum(number(row, "takeoff_target_distance_total") for row in selected),
            gap_takeoffs,
        ),
        "success_takeoff": ratio(
            sum(number(row, "successful_takeoff_tip_distance_total") for row in selected),
            edge_landings,
        ),
        "gap_wait_ep": mean(number(row, "gap_inaction_steps") for row in selected),
        "final_gap_dx": mean(number(row, "final_gap_start_dx") for row in selected),
        "final_gap_w": mean(number(row, "final_gap_width") for row in selected),
        "final_obs_dx": mean(number(row, "final_obstacle_dx") for row in selected),
    }

    if state_samples > 0:
        summary.update(
            {
                "grounded_pct": ratio(sum(number(row, "grounded_steps_total") for row in selected), state_samples) * 100.0,
                "jump_held_pct": ratio(sum(number(row, "jump_held_steps_total") for row in selected), state_samples) * 100.0,
                "gap_visible_pct": ratio(gap_visible_steps, state_samples) * 100.0,
                "gap_near_pct": ratio(sum(number(row, "gap_near_steps") for row in selected), state_samples) * 100.0,
                "gap_dx_mean": ratio(sum(number(row, "sum_gap_start_dx") for row in selected), gap_visible_steps),
                "gap_w_mean": ratio(sum(number(row, "sum_gap_width") for row in selected), gap_visible_steps),
                "landing_dy_mean": ratio(sum(number(row, "sum_landing_delta_y") for row in selected), gap_visible_steps),
                "obs_near_pct": ratio(sum(number(row, "obstacle_near_steps") for row in selected), state_samples) * 100.0,
                "pickup_visible_pct": ratio(pickup_visible_steps, state_samples) * 100.0,
            }
        )
    else:
        summary.update(
            {
                "grounded_pct": "-",
                "jump_held_pct": "-",
                "gap_visible_pct": "-",
                "gap_near_pct": "-",
                "gap_dx_mean": "-",
                "gap_w_mean": "-",
                "landing_dy_mean": "-",
                "obs_near_pct": "-",
                "pickup_visible_pct": "-",
            }
        )

    for idx, key in enumerate(ACTION_KEYS):
        summary[f"a{idx}_pct"] = ratio(action_totals[key], total_actions) * 100.0

    return summary


def build_rows(rows: list[dict[str, str]], every_episodes: int, history: int, include_partial: bool) -> list[dict[str, float | str]]:
    every = max(every_episodes, 1)
    total = len(rows)
    if total == 0:
        return []

    window_count = total // every
    if include_partial and total % every:
        window_count += 1
    if window_count == 0 and include_partial:
        return [summarize_window(rows, 0, total, 1)]
    if window_count == 0:
        return []

    first_window = max(0, window_count - max(history, 1))
    table_rows: list[dict[str, float | str]] = []
    for window_idx in range(first_window, window_count):
        start_idx = window_idx * every
        end_idx = min(start_idx + every, total)
        if start_idx >= end_idx:
            continue
        table_rows.append(summarize_window(rows, start_idx, end_idx, window_idx + 1))
    return table_rows


def format_value(value: float | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def render_table(table_rows: list[dict[str, float | str]], columns: list[str]) -> str:
    if not table_rows:
        return "No completed monitor rows yet."

    rendered_rows = [
        [format_value(row.get(column, "")) for column in columns]
        for row in table_rows
    ]
    widths = [
        max(len(column), *(len(row[idx]) for row in rendered_rows))
        for idx, column in enumerate(columns)
    ]

    header = " | ".join(column.ljust(widths[idx]) for idx, column in enumerate(columns))
    separator = "-+-".join("-" * width for width in widths)
    body = "\n".join(
        " | ".join(value.rjust(widths[idx]) for idx, value in enumerate(row))
        for row in rendered_rows
    )
    return f"{header}\n{separator}\n{body}"


def print_snapshot(path: Path, rows: list[dict[str, str]], every_episodes: int, history: int, include_partial: bool) -> None:
    table_rows = build_rows(rows, every_episodes, history, include_partial)
    print()
    print(f"Monitor: {path}")
    print(f"Total episodes logged: {len(rows)}")
    print(f"Table window: {every_episodes} episodes per iteration")
    print(render_table(table_rows, DEFAULT_COLUMNS))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Watch a Junimo Kart Stable-Baselines monitor.csv and print a compact state/telemetry table "
            "whenever a new episode window is available."
        )
    )
    parser.add_argument("monitor_csv", nargs="?", type=Path, help="Path to monitor.csv. Omit with --latest to use the newest monitor under logs/.")
    parser.add_argument("--latest", action="store_true", help="Use the newest logs/**/monitor.csv.")
    parser.add_argument("--logs-root", type=Path, default=Path("logs"), help="Root folder used by --latest.")
    parser.add_argument("--watch", action="store_true", help="Keep watching and print again when a new window appears.")
    parser.add_argument("--refresh-seconds", type=float, default=10.0, help="Polling interval for --watch.")
    parser.add_argument("--every-episodes", type=int, default=100, help="One table iteration summarizes this many completed episodes.")
    parser.add_argument("--history", type=int, default=10, help="Number of latest table iterations to display.")
    parser.add_argument("--no-partial", action="store_true", help="Only display full --every-episodes windows.")
    args = parser.parse_args()

    monitor_path = args.monitor_csv
    if args.latest:
        monitor_path = find_latest_monitor(args.logs_root)
    if monitor_path is None:
        raise SystemExit("No monitor.csv path provided. Use --latest or pass a monitor.csv path.")

    refresh_seconds = max(args.refresh_seconds, 1.0)
    last_print_key: tuple[int, int] | None = None
    include_partial = not args.no_partial

    while True:
        if args.latest:
            latest = find_latest_monitor(args.logs_root)
            if latest is not None:
                monitor_path = latest

        rows = read_monitor_csv(monitor_path)
        every = max(args.every_episodes, 1)
        completed_windows = len(rows) // every
        partial_marker = len(rows) % every if include_partial else 0
        print_key = (completed_windows, partial_marker)

        if rows and print_key != last_print_key:
            print_snapshot(monitor_path, rows, every, args.history, include_partial)
            last_print_key = print_key
        elif not rows and last_print_key is None:
            print(f"Waiting for completed episodes in {monitor_path}...")
            last_print_key = (0, 0)

        if not args.watch:
            break

        time.sleep(refresh_seconds)


if __name__ == "__main__":
    main()
