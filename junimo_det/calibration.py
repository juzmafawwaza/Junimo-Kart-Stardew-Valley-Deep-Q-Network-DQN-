from __future__ import annotations

import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from junimo_rl.client import JunimoKartBridgeClient
from junimo_rl.env import semantic_feature_snapshot


CALIBRATION_FIELDNAMES = [
    "trial_id",
    "trial_repeat",
    "hold_frames",
    "frame_index",
    "phase",
    "elapsed_s",
    "jump_commanded",
    "in_minigame",
    "game_mode",
    "game_state",
    "game_over",
    "score",
    "lives_left",
    "levels_beat",
    "player_x",
    "player_y",
    "velocity_x",
    "velocity_y",
    "grounded",
    "jumping",
    "next_gap_present",
    "next_gap_start_dx",
    "next_gap_width",
    "landing_y",
    "landing_delta_y",
    "next_obstacle_present",
    "next_obstacle_dx",
    "next_obstacle_y",
    "next_pickup_present",
    "next_pickup_dx",
    "next_pickup_y",
    "distance_to_finish",
    "progress_fraction",
]

GAP_TIMING_FIELDNAMES = [
    "trial_id",
    "trial_repeat",
    "trigger_dx",
    "hold_frames",
    "triggered",
    "trigger_frame",
    "trigger_gap_start_dx",
    "trigger_gap_width",
    "trigger_landing_y",
    "trigger_landing_delta_y",
    "trigger_player_x",
    "trigger_player_y",
    "trigger_velocity_x",
    "trigger_velocity_y",
    "gap_start_x",
    "gap_end_x",
    "crossed_gap",
    "landed_after_gap",
    "survived_until_margin",
    "game_over",
    "frames_elapsed",
    "jump_frames_sent",
    "final_score",
    "final_lives_left",
    "final_levels_beat",
    "final_player_x",
    "final_player_y",
    "final_grounded",
    "reason",
]

GAP_TRACE_FIELDNAMES = CALIBRATION_FIELDNAMES + [
    "trigger_dx",
    "triggered",
    "trigger_frame",
    "gap_start_x",
    "gap_end_x",
    "crossed_gap",
    "landed_after_gap",
]


def parse_frame_list(raw: str) -> list[int]:
    frames: list[int] = []
    for part in raw.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        value = int(stripped)
        if value < 0:
            raise ValueError(f"Frame counts must be >= 0, got {value}.")
        frames.append(value)

    if not frames:
        raise ValueError("At least one hold frame count is required.")

    return frames


def parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for part in raw.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        value = float(stripped)
        if value < 0:
            raise ValueError(f"Values must be >= 0, got {value}.")
        values.append(value)

    if not values:
        raise ValueError("At least one value is required.")

    return values


def timestamped_output_path(root: Path, stem: str, suffix: str = ".csv") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return root / f"{stem}_{timestamp}{suffix}"


def wait_for_gameplay(
    client: JunimoKartBridgeClient,
    *,
    timeout_s: float = 5.0,
    poll_s: float = 0.05,
) -> dict[str, Any]:
    deadline = time.time() + timeout_s
    snapshot = client.state()
    while time.time() < deadline:
        if _ready_for_gameplay(snapshot):
            return snapshot
        if snapshot.get("inMinigame") and snapshot.get("gameMode") == 2:
            try:
                client.advance()
            except RuntimeError:
                pass
        time.sleep(max(poll_s, 0.01))
        snapshot = client.state()

    return snapshot


def snapshot_to_calibration_row(
    snapshot: dict[str, Any],
    *,
    trial_id: int,
    trial_repeat: int,
    hold_frames: int,
    frame_index: int,
    phase: str,
    elapsed_s: float,
    jump_commanded: bool,
) -> dict[str, Any]:
    player = snapshot.get("player") or {}
    position = player.get("position") or {}
    velocity = player.get("velocity") or {}
    semantic = semantic_feature_snapshot(snapshot)

    return {
        "trial_id": trial_id,
        "trial_repeat": trial_repeat,
        "hold_frames": hold_frames,
        "frame_index": frame_index,
        "phase": phase,
        "elapsed_s": elapsed_s,
        "jump_commanded": int(jump_commanded),
        "in_minigame": int(bool(snapshot.get("inMinigame"))),
        "game_mode": _num(snapshot.get("gameMode")),
        "game_state": _num(snapshot.get("gameState")),
        "game_over": int(bool(snapshot.get("gameOver"))),
        "score": _num(snapshot.get("score")),
        "lives_left": _num(snapshot.get("livesLeft")),
        "levels_beat": _num(snapshot.get("levelsBeat")),
        "player_x": _num(position.get("x")),
        "player_y": _num(position.get("y")),
        "velocity_x": _num(velocity.get("x")),
        "velocity_y": _num(velocity.get("y")),
        "grounded": int(bool(player.get("grounded"))),
        "jumping": int(bool(player.get("jumping"))),
        "next_gap_present": int(bool(semantic["next_gap_present"])),
        "next_gap_start_dx": semantic["next_gap_start_dx"],
        "next_gap_width": semantic["next_gap_width"],
        "landing_y": semantic["landing_y"],
        "landing_delta_y": semantic["landing_delta_y"],
        "next_obstacle_present": int(bool(semantic["next_obstacle_present"])),
        "next_obstacle_dx": semantic["next_obstacle_dx"],
        "next_obstacle_y": semantic["next_obstacle_y"],
        "next_pickup_present": int(bool(semantic["next_pickup_present"])),
        "next_pickup_dx": semantic["next_pickup_dx"],
        "next_pickup_y": semantic["next_pickup_y"],
        "distance_to_finish": semantic["distance_to_finish"],
        "progress_fraction": semantic["progress_fraction"],
    }


def collect_jump_calibration(
    client: JunimoKartBridgeClient,
    *,
    out_path: Path,
    hold_frames_values: Iterable[int],
    trials_per_hold: int = 3,
    fps: float = 60.0,
    mode: str = "progress",
    start_each_trial: bool = True,
    settle_s: float = 0.25,
    pre_roll_frames: int = 8,
    max_trial_frames: int = 180,
    post_landing_frames: int = 18,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame_s = 1.0 / max(fps, 1.0)

    with out_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CALIBRATION_FIELDNAMES)
        writer.writeheader()

        trial_id = 0
        for hold_frames in hold_frames_values:
            for repeat in range(1, trials_per_hold + 1):
                trial_id += 1
                if start_each_trial:
                    client.start(mode)
                    time.sleep(max(settle_s, 0.0))
                    wait_for_gameplay(client)

                client.action(False)
                time.sleep(frame_s)

                seen_airborne = False
                landing_frames_left: int | None = None
                started_at = time.perf_counter()

                for frame_index in range(-pre_roll_frames, max_trial_frames):
                    jump = 0 <= frame_index < hold_frames
                    if frame_index < 0:
                        phase = "pre_roll"
                    elif jump:
                        phase = "hold"
                    elif landing_frames_left is not None:
                        phase = "post_landing"
                    else:
                        phase = "release"

                    client.action(jump)
                    time.sleep(frame_s)
                    snapshot = client.state()

                    writer.writerow(
                        snapshot_to_calibration_row(
                            snapshot,
                            trial_id=trial_id,
                            trial_repeat=repeat,
                            hold_frames=hold_frames,
                            frame_index=frame_index,
                            phase=phase,
                            elapsed_s=time.perf_counter() - started_at,
                            jump_commanded=jump,
                        )
                    )

                    player = snapshot.get("player") or {}
                    grounded = bool(player.get("grounded"))
                    jumping = bool(player.get("jumping"))
                    if not grounded or jumping:
                        seen_airborne = True

                    if seen_airborne and grounded and frame_index >= hold_frames:
                        if landing_frames_left is None:
                            landing_frames_left = max(post_landing_frames, 0)
                        landing_frames_left -= 1
                        if landing_frames_left <= 0:
                            break

                    if snapshot.get("gameOver") or not snapshot.get("inMinigame"):
                        break

                client.action(False)

    return out_path


def collect_gap_timing_calibration(
    client: JunimoKartBridgeClient,
    *,
    out_path: Path,
    trigger_dx_values: Iterable[float],
    hold_frames_values: Iterable[int],
    trials_per_combo: int = 2,
    fps: float = 60.0,
    mode: str = "progress",
    settle_s: float = 0.25,
    min_gap_width: float = 56.0,
    trigger_only_when_grounded: bool = True,
    detect_timeout_frames: int = 240,
    max_trial_frames: int = 420,
    post_cross_frames: int = 24,
    trace_path: Path | None = None,
    progress: bool = False,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)

    frame_s = 1.0 / max(fps, 1.0)
    trigger_values = list(trigger_dx_values)
    hold_values = list(hold_frames_values)

    with out_path.open("w", newline="", encoding="utf-8") as summary_file:
        summary_writer = csv.DictWriter(summary_file, fieldnames=GAP_TIMING_FIELDNAMES)
        summary_writer.writeheader()

        trace_file = trace_path.open("w", newline="", encoding="utf-8") if trace_path is not None else None
        try:
            trace_writer = csv.DictWriter(trace_file, fieldnames=GAP_TRACE_FIELDNAMES) if trace_file else None
            if trace_writer is not None:
                trace_writer.writeheader()

            trial_id = 0
            for trigger_dx in trigger_values:
                for hold_frames in hold_values:
                    for repeat in range(1, max(trials_per_combo, 1) + 1):
                        trial_id += 1
                        row = _run_gap_timing_trial(
                            client,
                            trial_id=trial_id,
                            trial_repeat=repeat,
                            trigger_dx=float(trigger_dx),
                            hold_frames=int(hold_frames),
                            fps=fps,
                            frame_s=frame_s,
                            mode=mode,
                            settle_s=settle_s,
                            min_gap_width=min_gap_width,
                            trigger_only_when_grounded=trigger_only_when_grounded,
                            detect_timeout_frames=detect_timeout_frames,
                            max_trial_frames=max_trial_frames,
                            post_cross_frames=post_cross_frames,
                            trace_writer=trace_writer,
                        )
                        summary_writer.writerow(row)
                        summary_file.flush()

                        if progress:
                            print(row)
        finally:
            if trace_file is not None:
                trace_file.close()

    return out_path


def _run_gap_timing_trial(
    client: JunimoKartBridgeClient,
    *,
    trial_id: int,
    trial_repeat: int,
    trigger_dx: float,
    hold_frames: int,
    fps: float,
    frame_s: float,
    mode: str,
    settle_s: float,
    min_gap_width: float,
    trigger_only_when_grounded: bool,
    detect_timeout_frames: int,
    max_trial_frames: int,
    post_cross_frames: int,
    trace_writer: csv.DictWriter | None,
) -> dict[str, Any]:
    client.start(mode)
    time.sleep(max(settle_s, 0.0))
    snapshot = wait_for_gameplay(client)
    client.action(False)
    time.sleep(frame_s)
    snapshot = client.state()

    triggered = False
    crossed_gap = False
    landed_after_gap = False
    survived_until_margin = False
    trigger_frame = -1
    gap_start_x = 0.0
    gap_end_x = 0.0
    jump_frames_sent = 0
    post_cross_remaining: int | None = None
    reason = "max_frames"
    trigger_data = {
        "trigger_gap_start_dx": 0.0,
        "trigger_gap_width": 0.0,
        "trigger_landing_y": 0.0,
        "trigger_landing_delta_y": 0.0,
        "trigger_player_x": 0.0,
        "trigger_player_y": 0.0,
        "trigger_velocity_x": 0.0,
        "trigger_velocity_y": 0.0,
    }
    started_at = time.perf_counter()

    for frame_index in range(max(max_trial_frames, 1)):
        player = snapshot.get("player") or {}
        position = player.get("position") or {}
        velocity = player.get("velocity") or {}
        player_x = _num(position.get("x"))
        grounded = bool(player.get("grounded"))
        semantic = semantic_feature_snapshot(snapshot)

        if not triggered:
            can_trigger = bool(
                semantic["next_gap_present"]
                and semantic["next_gap_width"] >= min_gap_width
                and 0.0 <= semantic["next_gap_start_dx"] <= trigger_dx
                and (grounded or not trigger_only_when_grounded)
            )
            if can_trigger:
                triggered = True
                trigger_frame = frame_index
                gap_start_x = player_x + semantic["next_gap_start_dx"]
                gap_end_x = gap_start_x + semantic["next_gap_width"]
                trigger_data = {
                    "trigger_gap_start_dx": semantic["next_gap_start_dx"],
                    "trigger_gap_width": semantic["next_gap_width"],
                    "trigger_landing_y": semantic["landing_y"],
                    "trigger_landing_delta_y": semantic["landing_delta_y"],
                    "trigger_player_x": player_x,
                    "trigger_player_y": _num(position.get("y")),
                    "trigger_velocity_x": _num(velocity.get("x")),
                    "trigger_velocity_y": _num(velocity.get("y")),
                }
            elif frame_index >= detect_timeout_frames:
                reason = "detect_timeout"
                _write_gap_trace_row(
                    trace_writer,
                    snapshot,
                    trial_id=trial_id,
                    trial_repeat=trial_repeat,
                    trigger_dx=trigger_dx,
                    hold_frames=hold_frames,
                    frame_index=frame_index,
                    phase="detect_timeout",
                    elapsed_s=time.perf_counter() - started_at,
                    jump_commanded=False,
                    triggered=triggered,
                    trigger_frame=trigger_frame,
                    gap_start_x=gap_start_x,
                    gap_end_x=gap_end_x,
                    crossed_gap=crossed_gap,
                    landed_after_gap=landed_after_gap,
                )
                break

        jump = bool(triggered and 0 <= frame_index - trigger_frame < hold_frames)
        if jump:
            jump_frames_sent += 1

        if not triggered:
            phase = "approach"
        elif jump:
            phase = "hold"
        elif landed_after_gap:
            phase = "post_cross"
        else:
            phase = "release"

        client.action(jump)
        time.sleep(frame_s)
        snapshot = client.state()

        new_player = snapshot.get("player") or {}
        new_position = new_player.get("position") or {}
        new_x = _num(new_position.get("x"))
        if triggered and not snapshot.get("gameOver"):
            if new_x >= gap_end_x:
                crossed_gap = True
            if crossed_gap and bool(new_player.get("grounded")):
                landed_after_gap = True
                if post_cross_remaining is None:
                    post_cross_remaining = max(post_cross_frames, 0)

        _write_gap_trace_row(
            trace_writer,
            snapshot,
            trial_id=trial_id,
            trial_repeat=trial_repeat,
            trigger_dx=trigger_dx,
            hold_frames=hold_frames,
            frame_index=frame_index,
            phase=phase,
            elapsed_s=time.perf_counter() - started_at,
            jump_commanded=jump,
            triggered=triggered,
            trigger_frame=trigger_frame,
            gap_start_x=gap_start_x,
            gap_end_x=gap_end_x,
            crossed_gap=crossed_gap,
            landed_after_gap=landed_after_gap,
        )

        if snapshot.get("gameOver"):
            reason = "game_over"
            break
        if not snapshot.get("inMinigame"):
            reason = "left_minigame"
            break
        if post_cross_remaining is not None:
            post_cross_remaining -= 1
            if post_cross_remaining <= 0:
                survived_until_margin = True
                reason = "survived_margin"
                break

    client.action(False)

    final_player = snapshot.get("player") or {}
    final_position = final_player.get("position") or {}
    return {
        "trial_id": trial_id,
        "trial_repeat": trial_repeat,
        "trigger_dx": trigger_dx,
        "hold_frames": hold_frames,
        "triggered": int(triggered),
        "trigger_frame": trigger_frame,
        **trigger_data,
        "gap_start_x": gap_start_x,
        "gap_end_x": gap_end_x,
        "crossed_gap": int(crossed_gap),
        "landed_after_gap": int(landed_after_gap),
        "survived_until_margin": int(survived_until_margin),
        "game_over": int(bool(snapshot.get("gameOver"))),
        "frames_elapsed": frame_index + 1,
        "jump_frames_sent": jump_frames_sent,
        "final_score": _num(snapshot.get("score")),
        "final_lives_left": _num(snapshot.get("livesLeft")),
        "final_levels_beat": _num(snapshot.get("levelsBeat")),
        "final_player_x": _num(final_position.get("x")),
        "final_player_y": _num(final_position.get("y")),
        "final_grounded": int(bool(final_player.get("grounded"))),
        "reason": reason,
    }


def _write_gap_trace_row(
    writer: csv.DictWriter | None,
    snapshot: dict[str, Any],
    *,
    trial_id: int,
    trial_repeat: int,
    trigger_dx: float,
    hold_frames: int,
    frame_index: int,
    phase: str,
    elapsed_s: float,
    jump_commanded: bool,
    triggered: bool,
    trigger_frame: int,
    gap_start_x: float,
    gap_end_x: float,
    crossed_gap: bool,
    landed_after_gap: bool,
) -> None:
    if writer is None:
        return

    row = snapshot_to_calibration_row(
        snapshot,
        trial_id=trial_id,
        trial_repeat=trial_repeat,
        hold_frames=hold_frames,
        frame_index=frame_index,
        phase=phase,
        elapsed_s=elapsed_s,
        jump_commanded=jump_commanded,
    )
    row.update(
        {
            "trigger_dx": trigger_dx,
            "triggered": int(triggered),
            "trigger_frame": trigger_frame,
            "gap_start_x": gap_start_x,
            "gap_end_x": gap_end_x,
            "crossed_gap": int(crossed_gap),
            "landed_after_gap": int(landed_after_gap),
        }
    )
    writer.writerow(row)


def _ready_for_gameplay(snapshot: dict[str, Any]) -> bool:
    return bool(
        snapshot.get("inMinigame")
        and snapshot.get("gameState") == 1
        and snapshot.get("player")
    )


def _num(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
