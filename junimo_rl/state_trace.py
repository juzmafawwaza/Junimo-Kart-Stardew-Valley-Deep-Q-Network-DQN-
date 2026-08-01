from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from stable_baselines3.common.callbacks import BaseCallback

from .env import semantic_feature_snapshot


TRACE_FIELDS = [
    "timestep",
    "episode",
    "episode_step",
    "action",
    "action_holds_jump",
    "step_reward",
    "reward",
    "event",
    "progress_reward",
    "pickup_reward",
    "gap_reward",
    "level_reward",
    "completion_reward",
    "death_reward",
    "jump_start_reward",
    "gap_miss_reward",
    "gap_miss_distance",
    "gap_miss_ratio",
    "airborne_hold_reward",
    "grounded_progress_reward",
    "unnecessary_jump_reward",
    "non_gap_airborne_reward",
    "gap_tip_technique_reward",
    "gap_inaction_reward",
    "jump_hold_steps",
    "done",
    "x",
    "y",
    "vx",
    "vy",
    "grounded",
    "jump_held",
    "jumping",
    "jump_ready",
    "current_track_end_dx",
    "gap_present",
    "gap_dx",
    "gap_width",
    "gap_end_dx",
    "landing_y",
    "landing_dy",
    "obstacle_present",
    "obstacle_dx",
    "obstacle_y",
    "pickup_present",
    "pickup_dx",
    "pickup_y",
    "progress",
    "score",
    "lives_left",
    "levels_beat",
]


def _num(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _flag(value: Any) -> int:
    return 1 if bool(value) else 0


def _indexed(value: Any, index: int, default: Any = None) -> Any:
    if value is None:
        return default
    try:
        return value[index]
    except (IndexError, KeyError, TypeError):
        return value if index == 0 else default


def action_holds_jump(action: int, action_mode: str) -> int:
    if action_mode == "binary":
        return 1 if action == 1 else 0
    return 1 if action > 0 else 0


def build_state_trace_row(
    snapshot: dict[str, Any],
    *,
    timestep: int,
    episode: int,
    episode_step: int,
    action: int,
    action_mode: str,
    step_reward: float,
    episode_reward: float,
    event: str,
    reward_components: dict[str, Any] | None,
    done: bool,
    semantic_features: dict[str, float] | None = None,
) -> dict[str, float | int | str]:
    player = snapshot.get("player") or {}
    position = player.get("position") or {}
    velocity = player.get("velocity") or {}
    semantic = semantic_features or semantic_feature_snapshot(snapshot)
    components = reward_components or {}

    return {
        "timestep": int(timestep),
        "episode": int(episode),
        "episode_step": int(episode_step),
        "action": int(action),
        "action_holds_jump": action_holds_jump(action, action_mode),
        "step_reward": float(step_reward),
        "reward": float(episode_reward),
        "event": str(event),
        "progress_reward": _num(components.get("progress")),
        "pickup_reward": _num(components.get("pickup")),
        "gap_reward": _num(components.get("gap")),
        "level_reward": _num(components.get("level")),
        "completion_reward": _num(components.get("completion")),
        "death_reward": _num(components.get("death")),
        "jump_start_reward": _num(components.get("jump_start")),
        "gap_miss_reward": _num(components.get("gap_miss")),
        "gap_miss_distance": _num(components.get("gap_miss_distance")),
        "gap_miss_ratio": _num(components.get("gap_miss_ratio")),
        "airborne_hold_reward": _num(components.get("airborne_hold")),
        "grounded_progress_reward": _num(components.get("grounded_progress")),
        "unnecessary_jump_reward": _num(components.get("unnecessary_jump")),
        "non_gap_airborne_reward": _num(components.get("non_gap_airborne")),
        "gap_tip_technique_reward": _num(components.get("gap_tip_technique")),
        "gap_inaction_reward": _num(components.get("gap_inaction")),
        "jump_hold_steps": _num(components.get("jump_hold_steps")),
        "done": _flag(done),
        "x": _num(position.get("x")),
        "y": _num(position.get("y")),
        "vx": _num(velocity.get("x")),
        "vy": _num(velocity.get("y")),
        "grounded": _flag(player.get("grounded")),
        "jump_held": _flag(snapshot.get("jumpHeld")),
        "jumping": _flag(player.get("jumping")),
        "jump_ready": _flag(player.get("jumpReady", player.get("grounded") and not snapshot.get("jumpHeld"))),
        "current_track_end_dx": semantic["current_track_end_dx"],
        "gap_present": _flag(semantic["next_gap_present"]),
        "gap_dx": semantic["next_gap_start_dx"],
        "gap_width": semantic["next_gap_width"],
        "gap_end_dx": semantic["next_gap_end_dx"],
        "landing_y": semantic["landing_y"],
        "landing_dy": semantic["landing_delta_y"],
        "obstacle_present": _flag(semantic["next_obstacle_present"]),
        "obstacle_dx": semantic["next_obstacle_dx"],
        "obstacle_y": semantic["next_obstacle_y"],
        "pickup_present": _flag(semantic["next_pickup_present"]),
        "pickup_dx": semantic["next_pickup_dx"],
        "pickup_y": semantic["next_pickup_y"],
        "progress": semantic["progress_fraction"],
        "score": _num(snapshot.get("score")),
        "lives_left": _num(snapshot.get("livesLeft")),
        "levels_beat": _num(snapshot.get("levelsBeat")),
    }


def format_state_trace_line(row: dict[str, float | int]) -> str:
    return " | ".join(
        [
            f"step={int(row['timestep']):7d}",
            f"ep={int(row['episode']):5d}",
            f"ep_step={int(row['episode_step']):4d}",
            f"a={int(row['action'])}",
            f"hold={bool(row['action_holds_jump'])!s:5}",
            f"r={float(row['step_reward']):7.3f}",
            f"ep_r={float(row['reward']):8.3f}",
            f"event={row['event']}",
            f"parts=p{float(row['progress_reward']):.2f}/g{float(row['gap_reward']):.2f}/tip{float(row['gap_tip_technique_reward']):.2f}/d{float(row['death_reward']):.2f}/j{float(row['jump_start_reward']):.3f}/u{float(row['unnecessary_jump_reward']):.3f}/wait{float(row['gap_inaction_reward']):.3f}/a{float(row['non_gap_airborne_reward']):.3f}/gr{float(row['grounded_progress_reward']):.3f}/h{float(row['airborne_hold_reward']):.3f}/m{float(row['gap_miss_reward']):.2f}",
            f"x={float(row['x']):8.1f}",
            f"y={float(row['y']):7.1f}",
            f"vy={float(row['vy']):7.2f}",
            f"grounded={bool(row['grounded'])!s:5}",
            f"jumpHeld={bool(row['jump_held'])!s:5}",
            f"jumpReady={bool(row['jump_ready'])!s:5}",
            f"gap={bool(row['gap_present'])!s:5}",
            f"gap_dx={float(row['gap_dx']):7.1f}",
            f"gap_width={float(row['gap_width']):6.1f}",
            f"gap_end={float(row['gap_end_dx']):7.1f}",
            f"landing_y={float(row['landing_y']):7.1f}",
            f"landing_dy={float(row['landing_dy']):7.1f}",
            f"obstacle_dx={float(row['obstacle_dx']):7.1f}",
            f"pickup_dx={float(row['pickup_dx']):7.1f}",
            f"score={float(row['score']):6.1f}",
        ]
    )


def _simple_target_dx(row: dict[str, float | int]) -> float:
    if int(row["gap_present"]):
        return float(row["gap_dx"])
    if int(row["obstacle_present"]):
        return float(row["obstacle_dx"])
    if int(row["pickup_present"]):
        return float(row["pickup_dx"])
    return 0.0


def _simple_target_dy(row: dict[str, float | int]) -> float:
    if int(row["gap_present"]):
        return float(row["landing_dy"])
    if int(row["obstacle_present"]):
        return float(row["obstacle_y"]) - float(row["y"])
    if int(row["pickup_present"]):
        return float(row["pickup_y"]) - float(row["y"])
    return 0.0


def _format_compact_number(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.3f}".rstrip("0").rstrip(".")


def format_state_trace_simple_line(
    row: dict[str, float | int],
    *,
    binned: bool = True,
    show_action: bool = False,
) -> str:
    target_dx = _simple_target_dx(row)
    target_dy = _simple_target_dy(row)
    if binned:
        x_value: float | int = int(max(target_dx, 0.0) // 40.0)
        y_value = int(round(target_dy / 16.0))
    else:
        x_value = round(target_dx, 1)
        y_value = round(target_dy, 1)

    parts = [
        f"x: {x_value}",
        f"y: {y_value}",
    ]
    if show_action:
        parts.extend(
            [
                f"action: {int(row['action'])}",
                f"hold: {int(row['action_holds_jump'])}",
            ]
        )
    parts.extend(
        [
            f"reward: {_format_compact_number(float(row['reward']))}",
            f"event: {row['event']}",
            f"score: {_format_compact_number(float(row['score']))}",
            f"generation: {int(row['episode'])}",
        ]
    )
    return ", ".join(parts)


class StateTraceCallback(BaseCallback):
    def __init__(
        self,
        *,
        action_mode: str,
        line_format: str = "full",
        simple_binned: bool = True,
        simple_show_action: bool = False,
        print_freq: int = 0,
        csv_path: Path | None = None,
        csv_freq: int = 1,
        max_rows: int = 0,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        self.action_mode = action_mode
        self.line_format = line_format
        self.simple_binned = simple_binned
        self.simple_show_action = simple_show_action
        self.print_freq = max(int(print_freq), 0)
        self.csv_path = csv_path
        self.csv_freq = max(int(csv_freq), 1)
        self.max_rows = max(int(max_rows), 0)
        self.episode = 1
        self.episode_step = 0
        self.rows_seen = 0
        self.rows_written = 0
        self.episode_reward = 0.0
        self._csv_file = None
        self._writer: csv.DictWriter | None = None

    def _init_callback(self) -> None:
        if self.csv_path is None:
            return

        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._csv_file, fieldnames=TRACE_FIELDS)
        self._writer.writeheader()
        self._csv_file.flush()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos") or []
        actions = self.locals.get("actions")
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")

        for index, info in enumerate(infos):
            snapshot = info.get("snapshot") if isinstance(info, dict) else None
            if not snapshot:
                continue

            self.episode_step += 1
            self.rows_seen += 1

            action = int(_indexed(actions, index, 0))
            reward = float(_indexed(rewards, index, 0.0))
            self.episode_reward += reward
            event = str(info.get("reward_event") or "none")
            done = bool(_indexed(dones, index, False))
            row = build_state_trace_row(
                snapshot,
                timestep=self.num_timesteps,
                episode=self.episode,
                episode_step=self.episode_step,
                action=action,
                action_mode=self.action_mode,
                step_reward=reward,
                episode_reward=self.episode_reward,
                event=event,
                reward_components=info.get("reward_components") if isinstance(info, dict) else None,
                done=done,
                semantic_features=info.get("semantic") if isinstance(info, dict) else None,
            )

            trace_enabled = self.max_rows <= 0 or self.rows_seen <= self.max_rows
            if trace_enabled and self.print_freq > 0 and self.rows_seen % self.print_freq == 0:
                if self.line_format == "simple":
                    print(
                        format_state_trace_simple_line(
                            row,
                            binned=self.simple_binned,
                            show_action=self.simple_show_action,
                        )
                    )
                else:
                    print(format_state_trace_line(row))

            if trace_enabled and self._writer is not None and self.rows_seen % self.csv_freq == 0:
                self._writer.writerow(row)
                self.rows_written += 1
                if self._csv_file is not None:
                    self._csv_file.flush()

            if done:
                self.episode += 1
                self.episode_step = 0
                self.episode_reward = 0.0

        return True

    def _on_training_end(self) -> None:
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None
