from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


def read_monitor_csv(path: Path) -> tuple[list[int], list[float], list[float]]:
    episodes: list[int] = []
    rewards: list[float] = []
    lengths: list[float] = []

    with path.open("r", encoding="utf-8") as file:
        filtered_lines = (line for line in file if not line.startswith("#"))
        reader = csv.DictReader(filtered_lines)
        for index, row in enumerate(reader, start=1):
            episodes.append(index)
            rewards.append(float(row["r"]))
            lengths.append(float(row["l"]))

    return episodes, rewards, lengths


def rolling_mean(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values

    output: list[float] = []
    running_sum = 0.0
    queue: list[float] = []
    for value in values:
        running_sum += value
        queue.append(value)
        if len(queue) > window:
            running_sum -= queue.pop(0)
        output.append(running_sum / len(queue))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Junimo Kart training curves from a Stable-Baselines monitor.csv file.")
    parser.add_argument("monitor_csv", help="Path to monitor.csv, e.g. logs/junimo_dqn_.../monitor.csv")
    parser.add_argument("--window", type=int, default=20, help="Rolling mean window in episodes.")
    parser.add_argument("--out", default=None, help="Optional output PNG path.")
    args = parser.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib belum terinstall. Jalankan: pip install -e .[analysis]", file=sys.stderr)
        raise

    monitor_csv = Path(args.monitor_csv)
    episodes, rewards, lengths = read_monitor_csv(monitor_csv)
    if not episodes:
        raise SystemExit(f"Tidak ada episode di {monitor_csv}. Training mungkin belum menyelesaikan satu episode.")

    smoothed_rewards = rolling_mean(rewards, args.window)
    smoothed_lengths = rolling_mean(lengths, args.window)

    figure, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(episodes, rewards, alpha=0.25, label="episode reward")
    axes[0].plot(episodes, smoothed_rewards, label=f"reward rolling mean ({args.window})")
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(episodes, lengths, alpha=0.25, label="episode length")
    axes[1].plot(episodes, smoothed_lengths, label=f"length rolling mean ({args.window})")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Steps")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    figure.suptitle(f"Junimo Kart training: {monitor_csv.parent.name}")
    figure.tight_layout()

    output = Path(args.out) if args.out else monitor_csv.with_name("training_plot.png")
    figure.savefig(output, dpi=160)
    print(f"Saved plot to: {output}")


if __name__ == "__main__":
    main()
