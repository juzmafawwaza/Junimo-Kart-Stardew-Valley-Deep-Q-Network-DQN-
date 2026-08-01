from __future__ import annotations

import argparse
import csv
import glob
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


FIELDNAMES = [
    "hold_frames",
    "trials",
    "landed_trials",
    "gameover_trials",
    "mean_airtime_frames",
    "mean_horizontal_distance",
    "mean_peak_height",
    "mean_velocity_x_per_frame",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize jump calibration CSV files.")
    parser.add_argument("csv_files", nargs="+", help="Calibration CSV paths or glob patterns.")
    parser.add_argument("--out", default=None, help="Optional output CSV for the summary.")
    args = parser.parse_args()

    paths = _expand_paths(args.csv_files)
    if not paths:
        raise SystemExit("No calibration CSV files matched.")

    grouped: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for path in paths:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            for row in reader:
                key = (int(float(row["hold_frames"])), int(float(row["trial_id"])))
                grouped[key].append(row)

    summaries_by_hold: dict[int, list[dict[str, float]]] = defaultdict(list)
    for (hold_frames, _trial_id), rows in grouped.items():
        summary = summarize_trial(rows)
        if summary is not None:
            summaries_by_hold[hold_frames].append(summary)

    output_rows = []
    for hold_frames in sorted(summaries_by_hold):
        summaries = summaries_by_hold[hold_frames]
        successful_summaries = [summary for summary in summaries if summary["landed"]]
        metric_summaries = successful_summaries or summaries
        output_rows.append(
            {
                "hold_frames": hold_frames,
                "trials": len(summaries),
                "landed_trials": len(successful_summaries),
                "gameover_trials": sum(1 for summary in summaries if summary["game_over"]),
                "mean_airtime_frames": _mean(summary["airtime_frames"] for summary in metric_summaries),
                "mean_horizontal_distance": _mean(summary["horizontal_distance"] for summary in metric_summaries),
                "mean_peak_height": _mean(summary["peak_height"] for summary in metric_summaries),
                "mean_velocity_x_per_frame": _mean(summary["velocity_x_per_frame"] for summary in metric_summaries),
            }
        )

    for row in output_rows:
        print(row)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(output_rows)
        print(f"Saved summary CSV to: {out}")


def summarize_trial(rows: list[dict[str, str]]) -> dict[str, float] | None:
    rows = sorted(rows, key=lambda row: int(float(row["frame_index"])))
    active_rows = [row for row in rows if int(float(row["frame_index"])) >= 0]
    if not active_rows:
        return None

    game_over = any(_truthy(row["game_over"]) for row in active_rows)
    usable_rows = []
    for row in active_rows:
        if _truthy(row["game_over"]):
            break
        if float(row["player_y"]) <= -500.0:
            break
        usable_rows.append(row)

    if not usable_rows:
        return None

    start = usable_rows[0]
    airborne_rows = [row for row in usable_rows if _truthy(row["jumping"]) or not _truthy(row["grounded"])]
    if not airborne_rows:
        return None

    landing = None
    seen_airborne = False
    for row in usable_rows:
        airborne = _truthy(row["jumping"]) or not _truthy(row["grounded"])
        if airborne:
            seen_airborne = True
        elif seen_airborne and _truthy(row["grounded"]):
            landing = row
            break

    end = landing or active_rows[-1]
    start_x = float(start["player_x"])
    start_y = float(start["player_y"])
    end_x = float(end["player_x"])
    min_y = min(float(row["player_y"]) for row in airborne_rows)
    first_air = int(float(airborne_rows[0]["frame_index"]))
    last_air = int(float(airborne_rows[-1]["frame_index"]))

    elapsed_frames = max(int(float(end["frame_index"])) - int(float(start["frame_index"])), 1)
    return {
        "landed": 1.0 if landing is not None else 0.0,
        "game_over": 1.0 if game_over else 0.0,
        "airtime_frames": float(max(last_air - first_air + 1, 0)),
        "horizontal_distance": end_x - start_x,
        "peak_height": start_y - min_y,
        "velocity_x_per_frame": (end_x - start_x) / elapsed_frames,
    }


def _expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(match) for match in glob.glob(pattern)]
        paths.extend(matches or [Path(pattern)])
    return [path for path in paths if path.exists()]


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _mean(values: object) -> float:
    materialized = list(values)
    return float(statistics.mean(materialized)) if materialized else 0.0


if __name__ == "__main__":
    main()
