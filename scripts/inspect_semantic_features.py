from __future__ import annotations

import argparse
import time

from junimo_rl.client import JunimoKartBridgeClient
from junimo_rl.env import GapGeometryTracker


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print live engineered semantic features from the Junimo Kart bridge."
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--interval", type=float, default=0.25, help="Seconds between printed snapshots.")
    parser.add_argument("--rows", type=int, default=0, help="Number of rows to print. Use 0 to run until Ctrl+C.")
    parser.add_argument("--gap-detection-mode", default="anchored", choices=["legacy", "anchored"])
    args = parser.parse_args()

    client = JunimoKartBridgeClient(host=args.host, port=args.port)
    gap_tracker = GapGeometryTracker(args.gap_detection_mode)
    printed = 0

    try:
        while args.rows <= 0 or printed < args.rows:
            snapshot = client.state()
            player = snapshot.get("player") or {}
            position = player.get("position") or {}
            features = gap_tracker.update(snapshot)

            print(
                " | ".join(
                    [
                        f"x={float(position.get('x') or 0.0):8.1f}",
                        f"grounded={bool(player.get('grounded'))!s:5}",
                        f"gap={bool(features['next_gap_present'])!s:5}",
                        f"gap_dx={features['next_gap_start_dx']:7.1f}",
                        f"gap_width={features['next_gap_width']:6.1f}",
                        f"landing_end_dx={features['landing_end_dx']:7.1f}",
                        f"landing_y={features['landing_y']:7.1f}",
                        f"landing_dy={features['landing_delta_y']:7.1f}",
                        f"obstacle_dx={features['next_obstacle_dx']:7.1f}",
                        f"pickup_dx={features['next_pickup_dx']:7.1f}",
                        f"progress={features['progress_fraction']:.3f}",
                    ]
                )
            )

            printed += 1
            time.sleep(max(args.interval, 0.01))
    except KeyboardInterrupt:
        pass
    finally:
        client.close()


if __name__ == "__main__":
    main()
