from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from junimo_det.calibration import collect_jump_calibration, parse_frame_list, timestamped_output_path
from junimo_rl.client import JunimoKartBridgeClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect per-frame jump calibration traces from Junimo Kart.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--mode", default="progress", choices=["progress", "endless", "infinite"])
    parser.add_argument("--hold-frames", default="1,2,4,6,8,10,12,16,20,24,30")
    parser.add_argument("--trials-per-hold", type=int, default=3)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--pre-roll-frames", type=int, default=8)
    parser.add_argument("--max-trial-frames", type=int, default=180)
    parser.add_argument("--post-landing-frames", type=int, default=18)
    parser.add_argument("--settle", type=float, default=0.25, help="Seconds to wait after starting each trial.")
    parser.add_argument("--no-start-each-trial", action="store_true", help="Do not restart Junimo Kart between trials.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    output_root = Path("outputs") / "deterministic"
    out_path = Path(args.out) if args.out else timestamped_output_path(output_root, "jump_calibration")
    hold_frames = parse_frame_list(args.hold_frames)

    with JunimoKartBridgeClient(host=args.host, port=args.port) as client:
        out = collect_jump_calibration(
            client,
            out_path=out_path,
            hold_frames_values=hold_frames,
            trials_per_hold=max(args.trials_per_hold, 1),
            fps=args.fps,
            mode=args.mode,
            start_each_trial=not args.no_start_each_trial,
            settle_s=args.settle,
            pre_roll_frames=max(args.pre_roll_frames, 0),
            max_trial_frames=max(args.max_trial_frames, 1),
            post_landing_frames=max(args.post_landing_frames, 0),
        )

    print(f"Saved jump calibration CSV to: {out}")


if __name__ == "__main__":
    main()
