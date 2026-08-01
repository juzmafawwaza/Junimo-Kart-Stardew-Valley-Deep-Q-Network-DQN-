from __future__ import annotations

import argparse
import csv
import glob
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize per-frame deterministic controller runs.")
    parser.add_argument("csv_files", nargs="+", help="Rule live CSV paths or glob patterns.")
    args = parser.parse_args()

    rows = []
    for path in _expand_paths(args.csv_files):
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            rows.extend(list(reader))

    if not rows:
        raise SystemExit("No rule live rows found.")

    summary = {
        "episodes": len(rows),
        "mean_frames": _mean(row["frames"] for row in rows),
        "max_frames": max(float(row["frames"]) for row in rows),
        "mean_score": _mean(row["score"] for row in rows),
        "max_score": max(float(row["score"]) for row in rows),
        "mean_final_player_x": _mean(row["final_player_x"] for row in rows),
        "max_final_player_x": max(float(row["final_player_x"]) for row in rows),
        "mean_gap_jump_count": _mean(row["gap_jump_count"] for row in rows),
        "max_gap_jump_count": max(float(row["gap_jump_count"]) for row in rows),
        "mean_track_step_jump_count": _mean(row.get("track_step_jump_count", 0) for row in rows),
        "max_track_step_jump_count": max(float(row.get("track_step_jump_count", 0)) for row in rows),
        "gameover_rate": _rate(rows, "game_over"),
        "completion_rate": _rate(rows, "completed"),
    }
    print(summary)


def _expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(match) for match in glob.glob(pattern)]
        paths.extend(matches or [Path(pattern)])
    return [path for path in paths if path.exists()]


def _mean(values: object) -> float:
    materialized = [float(value) for value in values]
    return float(statistics.mean(materialized)) if materialized else 0.0


def _rate(rows: list[dict[str, str]], field: str) -> float:
    return sum(1 for row in rows if row[field].strip() in {"1", "true", "True"}) / len(rows) if rows else 0.0


if __name__ == "__main__":
    main()
