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
    "trigger_dx",
    "hold_frames",
    "trials",
    "triggered_rate",
    "crossed_rate",
    "landed_rate",
    "survived_rate",
    "gameover_rate",
    "mean_required_jump_distance",
    "mean_final_shortfall",
    "mean_trigger_gap_start_dx",
    "mean_trigger_gap_width",
    "mean_trigger_landing_delta_y",
    "mean_final_player_x",
    "mean_final_score",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize gap timing calibration CSV files.")
    parser.add_argument("csv_files", nargs="+", help="Gap timing CSV paths or glob patterns.")
    parser.add_argument("--out", default=None, help="Optional output CSV for the summary.")
    parser.add_argument("--top", type=int, default=0, help="Print only the top N rows. Use 0 for all rows.")
    args = parser.parse_args()

    paths = _expand_paths(args.csv_files)
    if not paths:
        raise SystemExit("No gap timing CSV files matched.")

    groups: dict[tuple[float, int], list[dict[str, str]]] = defaultdict(list)
    for path in paths:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if not _has_required_fields(reader.fieldnames or []):
                print(f"Skipping non-raw gap timing CSV: {path}")
                continue
            for row in reader:
                key = (float(row["trigger_dx"]), int(float(row["hold_frames"])))
                groups[key].append(row)

    if not groups:
        raise SystemExit("No raw gap timing rows found.")

    output_rows = []
    for (trigger_dx, hold_frames), rows in sorted(groups.items()):
        output_rows.append(
            {
                "trigger_dx": trigger_dx,
                "hold_frames": hold_frames,
                "trials": len(rows),
                "triggered_rate": _rate(rows, "triggered"),
                "crossed_rate": _rate(rows, "crossed_gap"),
                "landed_rate": _rate(rows, "landed_after_gap"),
                "survived_rate": _rate(rows, "survived_until_margin"),
                "gameover_rate": _rate(rows, "game_over"),
                "mean_required_jump_distance": _mean(
                    float(row["trigger_gap_start_dx"]) + float(row["trigger_gap_width"])
                    for row in rows
                    if _truthy(row["triggered"])
                ),
                "mean_final_shortfall": _mean(
                    float(row["gap_end_x"]) - float(row["final_player_x"])
                    for row in rows
                    if _truthy(row["triggered"])
                ),
                "mean_trigger_gap_start_dx": _mean(row["trigger_gap_start_dx"] for row in rows if _truthy(row["triggered"])),
                "mean_trigger_gap_width": _mean(row["trigger_gap_width"] for row in rows if _truthy(row["triggered"])),
                "mean_trigger_landing_delta_y": _mean(row["trigger_landing_delta_y"] for row in rows if _truthy(row["triggered"])),
                "mean_final_player_x": _mean(row["final_player_x"] for row in rows),
                "mean_final_score": _mean(row["final_score"] for row in rows),
            }
        )

    ranked_rows = sorted(
        output_rows,
        key=lambda row: (
            -float(row["survived_rate"]),
            -float(row["landed_rate"]),
            float(row["gameover_rate"]),
            -float(row["mean_final_player_x"]),
        ),
    )
    visible_rows = ranked_rows if args.top <= 0 else ranked_rows[: args.top]

    for row in visible_rows:
        print(row)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(ranked_rows)
        print(f"Saved gap timing summary CSV to: {out}")


def _expand_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = [Path(match) for match in glob.glob(pattern)]
        paths.extend(matches or [Path(pattern)])
    return [path for path in paths if path.exists()]


def _has_required_fields(fieldnames: list[str]) -> bool:
    required = {
        "trigger_dx",
        "hold_frames",
        "triggered",
        "crossed_gap",
        "landed_after_gap",
        "survived_until_margin",
        "game_over",
    }
    return required.issubset(set(fieldnames))


def _rate(rows: list[dict[str, str]], field: str) -> float:
    return sum(1 for row in rows if _truthy(row[field])) / len(rows) if rows else 0.0


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def _mean(values: object) -> float:
    materialized = [float(value) for value in values]
    return float(statistics.mean(materialized)) if materialized else 0.0


if __name__ == "__main__":
    main()
