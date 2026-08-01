from __future__ import annotations

import argparse
import csv
import glob
import re
import statistics
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank deterministic live controller CSV outputs.")
    parser.add_argument("csv_files", nargs="+", help="Rule live CSV paths or glob patterns.")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--out", default=None, help="Optional CSV summary output path.")
    args = parser.parse_args()

    summaries = [summarize_file(path) for path in _expand_paths(args.csv_files)]
    summaries = [summary for summary in summaries if summary is not None]
    if not summaries:
        raise SystemExit("No rule live rows found.")

    summaries.sort(
        key=lambda row: (
            row["completion_rate"],
            row["mean_final_player_x"],
            row["p75_final_player_x"],
            row["max_final_player_x"],
            row["mean_score"],
        ),
        reverse=True,
    )

    fieldnames = [
        "rank",
        "file",
        "episodes",
        "mean_final_player_x",
        "median_final_player_x",
        "p75_final_player_x",
        "max_final_player_x",
        "mean_score",
        "max_score",
        "mean_frames",
        "mean_gap_jump_count",
        "mean_track_step_jump_count",
        "gameover_rate",
        "completion_rate",
        "track_step_trigger",
        "track_step_hold",
        "track_step_y_delta",
    ]

    for index, summary in enumerate(summaries, start=1):
        summary["rank"] = index

    writer = csv.DictWriter(sys.stdout, fieldnames=fieldnames, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    writer.writerows(summaries[: max(int(args.top), 1)])

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as file:
            out_writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
            out_writer.writeheader()
            out_writer.writerows(summaries)
        print(f"Saved rank CSV to: {out}", file=sys.stderr)


def summarize_file(path: Path) -> dict[str, float | int | str] | None:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        return None

    final_x = [_float(row["final_player_x"]) for row in rows]
    scores = [_float(row["score"]) for row in rows]
    frames = [_float(row["frames"]) for row in rows]
    gap_jumps = [_float(row["gap_jump_count"]) for row in rows]
    track_step_jumps = [_float(row.get("track_step_jump_count", 0.0)) for row in rows]
    tags = _parse_grid_tags(path.name)

    return {
        "file": str(path),
        "episodes": len(rows),
        "mean_final_player_x": round(statistics.mean(final_x), 3),
        "median_final_player_x": round(statistics.median(final_x), 3),
        "p75_final_player_x": round(_percentile(final_x, 0.75), 3),
        "max_final_player_x": round(max(final_x), 3),
        "mean_score": round(statistics.mean(scores), 3),
        "max_score": round(max(scores), 3),
        "mean_frames": round(statistics.mean(frames), 3),
        "mean_gap_jump_count": round(statistics.mean(gap_jumps), 3),
        "mean_track_step_jump_count": round(statistics.mean(track_step_jumps), 3),
        "gameover_rate": round(_rate(rows, "game_over"), 3),
        "completion_rate": round(_rate(rows, "completed"), 3),
        **tags,
    }


def _expand_paths(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(match) for match in glob.glob(pattern)]
        paths.extend(matches or [Path(pattern)])
    return [path for path in paths if path.exists()]


def _parse_grid_tags(name: str) -> dict[str, str]:
    tags = {
        "track_step_trigger": "",
        "track_step_hold": "",
        "track_step_y_delta": "",
    }
    match = re.search(r"_ts_t(?P<trigger>\d+(?:\.\d+)?)_h(?P<hold>\d+)_yd(?P<yd>\d+(?:\.\d+)?)", name)
    if match:
        tags["track_step_trigger"] = match.group("trigger")
        tags["track_step_hold"] = match.group("hold")
        tags["track_step_y_delta"] = match.group("yd")
    return tags


def _float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(max(int(round((len(ordered) - 1) * quantile)), 0), len(ordered) - 1)
    return ordered[index]


def _rate(rows: list[dict[str, str]], field: str) -> float:
    return sum(1 for row in rows if row.get(field, "").strip() in {"1", "true", "True"}) / len(rows)


if __name__ == "__main__":
    main()
