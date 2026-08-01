from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from junimo_det.calibration import wait_for_gameplay
from junimo_rl.client import JunimoKartBridgeClient
from junimo_rl.env import semantic_feature_snapshot


FIELDNAMES = [
    "episode",
    "frames",
    "score",
    "levels_beat",
    "completed",
    "game_over",
    "final_player_x",
    "final_player_y",
    "jump_frames",
    "gap_jump_count",
    "track_step_jump_count",
    "obstacle_jump_count",
    "mean_gap_trigger_dx",
    "mean_gap_width",
    "mean_landing_delta_y",
]

TRACE_FIELDNAMES = [
    "episode",
    "frame",
    "player_x",
    "player_y",
    "velocity_x",
    "velocity_y",
    "grounded",
    "jumping",
    "jump_commanded",
    "hold_remaining_after_decision",
    "gap_jump_count",
    "track_step_jump_count",
    "obstacle_jump_count",
    "next_gap_present",
    "next_gap_start_dx",
    "next_gap_width",
    "landing_delta_y",
    "plan_trigger_dx",
    "plan_hold_frames",
    "next_obstacle_present",
    "next_obstacle_dx",
    "game_over",
    "score",
]


@dataclass(slots=True)
class GapPlan:
    trigger_dx: float
    hold_frames: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a per-frame deterministic Junimo Kart controller.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--mode", default="progress", choices=["progress", "endless", "infinite"])
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--settle", type=float, default=0.25)
    parser.add_argument("--max-frames", type=int, default=7200)
    parser.add_argument("--min-gap-width", type=float, default=56.0)
    parser.add_argument("--down-trigger-dx", type=float, default=12.0)
    parser.add_argument("--down-hold-frames", type=int, default=10)
    parser.add_argument("--normal-trigger-dx", type=float, default=22.0)
    parser.add_argument("--normal-hold-frames", type=int, default=14)
    parser.add_argument("--up-trigger-dx", type=float, default=22.0)
    parser.add_argument("--up-hold-frames", type=int, default=16)
    parser.add_argument("--wide-trigger-dx", type=float, default=22.0)
    parser.add_argument("--wide-hold-frames", type=int, default=16)
    parser.add_argument("--track-step-trigger-dx", type=float, default=36.0)
    parser.add_argument("--track-step-y-delta", type=float, default=12.0)
    parser.add_argument("--track-step-hold-frames", type=int, default=16)
    parser.add_argument("--obstacle-trigger-dx", type=float, default=24.0)
    parser.add_argument("--obstacle-hold-frames", type=int, default=8)
    parser.add_argument("--post-trigger-lockout", type=float, default=16.0)
    parser.add_argument("--out", default="outputs/deterministic/rule_live.csv")
    parser.add_argument("--trace-out", default=None)
    parser.add_argument("--quiet", action="store_true", help="Only print the final CSV path.")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    with JunimoKartBridgeClient(host=args.host, port=args.port) as client:
        trace_file = Path(args.trace_out).open("w", newline="", encoding="utf-8") if args.trace_out else None
        try:
            trace_writer = csv.DictWriter(trace_file, fieldnames=TRACE_FIELDNAMES) if trace_file else None
            if trace_writer is not None:
                trace_writer.writeheader()

            for episode in range(1, max(args.episodes, 1) + 1):
                row = run_episode(client, args=args, episode=episode, trace_writer=trace_writer)
                rows.append(row)
                if not args.quiet:
                    print(row)
        finally:
            if trace_file is not None:
                trace_file.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved live rule CSV to: {out}")


def run_episode(
    client: JunimoKartBridgeClient,
    *,
    args: argparse.Namespace,
    episode: int,
    trace_writer: csv.DictWriter | None = None,
) -> dict[str, Any]:
    frame_s = 1.0 / max(float(args.fps), 1.0)
    client.start(args.mode)
    time.sleep(max(float(args.settle), 0.0))
    snapshot = wait_for_gameplay(client)

    hold_remaining = 0
    handled_gap_end_x = -1.0
    jump_frames = 0
    gap_jump_count = 0
    track_step_jump_count = 0
    obstacle_jump_count = 0
    trigger_gap_dx_values: list[float] = []
    trigger_gap_width_values: list[float] = []
    landing_delta_values: list[float] = []

    frame = 0
    for frame in range(max(int(args.max_frames), 1)):
        player = snapshot.get("player") or {}
        position = player.get("position") or {}
        player_x = _num(position.get("x"))
        grounded = bool(player.get("grounded"))
        features = semantic_feature_snapshot(snapshot)
        gap_plan = choose_gap_plan(features, args=args)

        if handled_gap_end_x > 0 and player_x > handled_gap_end_x + float(args.post_trigger_lockout):
            handled_gap_end_x = -1.0

        jump = False
        if hold_remaining > 0:
            jump = True
            hold_remaining -= 1
        elif grounded:
            gap_start_x = player_x + features["next_gap_start_dx"]
            gap_is_new = handled_gap_end_x < 0 or gap_start_x > handled_gap_end_x + float(args.post_trigger_lockout)
            track_step_plan = choose_track_step_plan(snapshot, args=args)
            if track_step_plan is not None:
                jump = True
                hold_remaining = max(track_step_plan.hold_frames - 1, 0)
                track_step_jump_count += 1
            elif gap_plan and gap_is_new and features["next_gap_start_dx"] <= gap_plan.trigger_dx:
                jump = True
                hold_remaining = max(gap_plan.hold_frames - 1, 0)
                handled_gap_end_x = gap_start_x + features["next_gap_width"]
                gap_jump_count += 1
                trigger_gap_dx_values.append(features["next_gap_start_dx"])
                trigger_gap_width_values.append(features["next_gap_width"])
                landing_delta_values.append(features["landing_delta_y"])
            elif (
                features["next_obstacle_present"]
                and 0.0 <= features["next_obstacle_dx"] <= float(args.obstacle_trigger_dx)
            ):
                jump = True
                hold_remaining = max(int(args.obstacle_hold_frames) - 1, 0)
                obstacle_jump_count += 1

        if trace_writer is not None:
            velocity = player.get("velocity") or {}
            trace_writer.writerow(
                {
                    "episode": episode,
                    "frame": frame,
                    "player_x": player_x,
                    "player_y": _num(position.get("y")),
                    "velocity_x": _num(velocity.get("x")),
                    "velocity_y": _num(velocity.get("y")),
                    "grounded": int(grounded),
                    "jumping": int(bool(player.get("jumping"))),
                    "jump_commanded": int(jump),
                    "hold_remaining_after_decision": hold_remaining,
                    "gap_jump_count": gap_jump_count,
                    "track_step_jump_count": track_step_jump_count,
                    "obstacle_jump_count": obstacle_jump_count,
                    "next_gap_present": int(bool(features["next_gap_present"])),
                    "next_gap_start_dx": features["next_gap_start_dx"],
                    "next_gap_width": features["next_gap_width"],
                    "landing_delta_y": features["landing_delta_y"],
                    "plan_trigger_dx": gap_plan.trigger_dx if gap_plan else 0.0,
                    "plan_hold_frames": gap_plan.hold_frames if gap_plan else 0,
                    "next_obstacle_present": int(bool(features["next_obstacle_present"])),
                    "next_obstacle_dx": features["next_obstacle_dx"],
                    "game_over": int(bool(snapshot.get("gameOver"))),
                    "score": int(snapshot.get("score") or 0),
                }
            )

        client.action(jump)
        if jump:
            jump_frames += 1
        time.sleep(frame_s)
        snapshot = client.state()

        if snapshot.get("completed") or snapshot.get("gameOver") or not snapshot.get("inMinigame"):
            break

    client.action(False)
    final_player = snapshot.get("player") or {}
    final_position = final_player.get("position") or {}
    return {
        "episode": episode,
        "frames": frame + 1,
        "score": int(snapshot.get("score") or 0),
        "levels_beat": int(snapshot.get("levelsBeat") or 0),
        "completed": int(bool(snapshot.get("completed"))),
        "game_over": int(bool(snapshot.get("gameOver"))),
        "final_player_x": _num(final_position.get("x")),
        "final_player_y": _num(final_position.get("y")),
        "jump_frames": jump_frames,
        "gap_jump_count": gap_jump_count,
        "track_step_jump_count": track_step_jump_count,
        "obstacle_jump_count": obstacle_jump_count,
        "mean_gap_trigger_dx": _mean(trigger_gap_dx_values),
        "mean_gap_width": _mean(trigger_gap_width_values),
        "mean_landing_delta_y": _mean(landing_delta_values),
    }


def choose_gap_plan(features: dict[str, float], *, args: argparse.Namespace) -> GapPlan | None:
    if not features["next_gap_present"]:
        return None

    gap_width = features["next_gap_width"]
    landing_delta_y = features["landing_delta_y"]
    if gap_width < float(args.min_gap_width):
        return None

    if landing_delta_y <= -24.0 and gap_width <= 88.0:
        return GapPlan(trigger_dx=float(args.down_trigger_dx), hold_frames=int(args.down_hold_frames))
    if landing_delta_y <= -16.0:
        return GapPlan(trigger_dx=float(args.down_trigger_dx), hold_frames=int(args.down_hold_frames))
    if gap_width >= 104.0 or landing_delta_y >= 16.0:
        return GapPlan(trigger_dx=float(args.wide_trigger_dx), hold_frames=int(args.wide_hold_frames))
    if landing_delta_y > 0.0:
        return GapPlan(trigger_dx=float(args.up_trigger_dx), hold_frames=int(args.up_hold_frames))
    return GapPlan(trigger_dx=float(args.normal_trigger_dx), hold_frames=int(args.normal_hold_frames))


def choose_track_step_plan(snapshot: dict[str, Any], *, args: argparse.Namespace) -> GapPlan | None:
    player = snapshot.get("player") or {}
    position = player.get("position") or {}
    player_y = _num(position.get("y"))
    tracks = [track for track in (snapshot.get("tracksAhead") or []) if isinstance(track, dict)]
    forward_tracks = sorted(
        (track for track in tracks if _num(track.get("dx")) >= 0.0),
        key=lambda track: (_num(track.get("dx")), _num(track.get("y"))),
    )

    for track in forward_tracks:
        dx = _num(track.get("dx"))
        y_delta_up = player_y - _num(track.get("y"))
        if dx > float(args.track_step_trigger_dx):
            break
        if y_delta_up >= float(args.track_step_y_delta):
            return GapPlan(trigger_dx=dx, hold_frames=int(args.track_step_hold_frames))

    return None


def _num(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


if __name__ == "__main__":
    main()
