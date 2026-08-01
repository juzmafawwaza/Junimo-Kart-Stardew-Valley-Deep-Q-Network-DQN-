from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from junimo_det.calibration import (
    collect_gap_timing_calibration,
    parse_float_list,
    parse_frame_list,
    timestamped_output_path,
)
from junimo_rl.client import JunimoKartBridgeClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate when to jump before Junimo Kart gaps.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--mode", default="progress", choices=["progress", "endless", "infinite"])
    parser.add_argument("--trigger-dx", default="10,20,30,40,50,60,80")
    parser.add_argument("--hold-frames", default="4,6,8,10,12,16")
    parser.add_argument("--trials-per-combo", type=int, default=2)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--settle", type=float, default=0.25, help="Seconds to wait after starting each trial.")
    parser.add_argument("--min-gap-width", type=float, default=56.0)
    parser.add_argument("--detect-timeout-frames", type=int, default=240)
    parser.add_argument("--max-trial-frames", type=int, default=420)
    parser.add_argument("--post-cross-frames", type=int, default=24)
    parser.add_argument("--trigger-while-airborne", action="store_true")
    parser.add_argument("--trace", action="store_true", help="Also save a per-frame trace CSV.")
    parser.add_argument("--out", default=None)
    parser.add_argument("--trace-out", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    output_root = Path("outputs") / "deterministic"
    out_path = Path(args.out) if args.out else timestamped_output_path(output_root, "gap_timing")
    trace_path = None
    if args.trace or args.trace_out:
        trace_path = Path(args.trace_out) if args.trace_out else timestamped_output_path(output_root, "gap_timing_trace")

    trigger_dx_values = parse_float_list(args.trigger_dx)
    hold_frames_values = parse_frame_list(args.hold_frames)

    with JunimoKartBridgeClient(host=args.host, port=args.port) as client:
        out = collect_gap_timing_calibration(
            client,
            out_path=out_path,
            trigger_dx_values=trigger_dx_values,
            hold_frames_values=hold_frames_values,
            trials_per_combo=max(args.trials_per_combo, 1),
            fps=args.fps,
            mode=args.mode,
            settle_s=args.settle,
            min_gap_width=args.min_gap_width,
            trigger_only_when_grounded=not args.trigger_while_airborne,
            detect_timeout_frames=max(args.detect_timeout_frames, 1),
            max_trial_frames=max(args.max_trial_frames, 1),
            post_cross_frames=max(args.post_cross_frames, 0),
            trace_path=trace_path,
            progress=not args.quiet,
        )

    print(f"Saved gap timing CSV to: {out}")
    if trace_path is not None:
        print(f"Saved gap timing trace CSV to: {trace_path}")


if __name__ == "__main__":
    main()
