from __future__ import annotations

import hashlib
import time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .client import JunimoKartBridgeClient


MAX_TRACKS = 24
MAX_ENTITIES = 24
TRACK_FEATURES = 5
ENTITY_FEATURES = 7
BASE_FEATURES = 18
SEMANTIC_FEATURES = 17
TEMPORAL_FEATURES = 5
COMPACT_FEATURES = 27
LEGACY_OBSERVATION_SIZE = BASE_FEATURES + MAX_TRACKS * TRACK_FEATURES + MAX_ENTITIES * ENTITY_FEATURES
OBSERVATION_SIZE = LEGACY_OBSERVATION_SIZE
SEMANTIC_OBSERVATION_SIZE = LEGACY_OBSERVATION_SIZE + SEMANTIC_FEATURES
SEMANTIC_TEMPORAL_OBSERVATION_SIZE = SEMANTIC_OBSERVATION_SIZE + TEMPORAL_FEATURES
OBSERVATION_MODES = {"flat", "multi", "compact"}
DEFAULT_RECENT_ACTION_HISTORY = 12
SPATIAL_MAP_CHANNELS = 4
SPATIAL_MAP_HEIGHT = 16
SPATIAL_MAP_WIDTH = 64
SPATIAL_MAP_X_BEHIND = 96.0
SPATIAL_MAP_X_AHEAD = 640.0
SPATIAL_MAP_Y_MIN = 0.0
SPATIAL_MAP_Y_MAX = 320.0
GAP_PIXEL_THRESHOLD = 36.0
GAP_LANDING_REWARD_MIN_WIDTH = 56.0
GAP_LANDING_REWARD_ACTIVATION_DX = 180.0
GAP_LANDING_REWARD_EXPIRE_PIXELS = 320.0
GAP_DETECTION_MODES = {"legacy", "anchored"}
TRACK_SUPPORT_TOLERANCE = 10.0
TRACK_CHAIN_MAX_DY = 48.0
LANDING_MAX_DY = 112.0
DEFAULT_GAP_LANDING_CONFIRM_STEPS = 2
DEFAULT_GAP_LANDING_BASE_REWARD = 8.0
DEFAULT_GAP_LANDING_WIDTH_COEF = 0.04
ACTION_MODES = {"binary", "macro", "tap_macro"}
REWARD_VERSIONS = {
    "legacy",
    "shaped_v1",
    "shaped_v2",
    "shaped_v3",
    "shaped_v4",
    "shaped_v5",
    "shaped_v6",
    "shaped_v7",
    "shaped_v8",
    "shaped_v9",
}
MACRO_ACTIONS = 4
DEFAULT_MACRO_ACTION_FRAMES = 8
TELEMETRY_INFO_KEYS = (
    "action_0_count",
    "action_1_count",
    "action_2_count",
    "action_3_count",
    "gap_attempts",
    "gap_landings",
    "gap_failures",
    "gap_deaths",
    "death_near_gap",
    "death_near_obstacle",
    "pickup_events",
    "coin_events",
    "fruit_events",
    "unknown_pickup_events",
    "score_delta_total",
    "coin_reward_total",
    "fruit_reward_total",
    "progress_reward_total",
    "gap_landing_reward_total",
    "death_penalty_total",
    "max_episode_x",
    "state_samples",
    "gap_visible_steps",
    "gap_near_steps",
    "obstacle_visible_steps",
    "obstacle_near_steps",
    "pickup_visible_steps",
    "grounded_steps_total",
    "jump_held_steps_total",
    "jump_start_events",
    "jump_start_near_gap_events",
    "jump_start_without_near_gap_events",
    "jump_start_penalty_total",
    "gap_miss_deaths",
    "gap_miss_distance_total",
    "gap_miss_ratio_total",
    "gap_miss_penalty_total",
    "airborne_hold_penalty_steps",
    "airborne_hold_penalty_total",
    "max_jump_hold_steps",
    "grounded_progress_bonus_total",
    "unnecessary_jump_events",
    "unnecessary_jump_penalty_total",
    "non_gap_airborne_steps",
    "non_gap_airborne_penalty_total",
    "gap_takeoff_events",
    "takeoff_tip_distance_total",
    "takeoff_tip_quality_total",
    "edge_qualified_landings",
    "successful_takeoff_tip_quality_total",
    "landing_tip_depth_total",
    "landing_tip_quality_total",
    "edge_technique_reward_total",
    "takeoff_target_distance_total",
    "successful_takeoff_tip_distance_total",
    "successful_takeoff_target_distance_total",
    "gap_inaction_steps",
    "gap_inaction_penalty_total",
    "sum_gap_start_dx",
    "sum_gap_width",
    "sum_landing_delta_y",
    "sum_obstacle_dx",
    "sum_pickup_dx",
    "final_gap_start_dx",
    "final_gap_width",
    "final_obstacle_dx",
)


def _num(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _bool(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


def _stable_unit(label: str | None) -> float:
    if not label:
        return 0.0
    digest = hashlib.blake2b(label.encode("utf-8"), digest_size=2).digest()
    return int.from_bytes(digest, "big") / 65535.0


def _clip_scale(value: Any, scale: float, low: float = -1.0, high: float = 1.0) -> float:
    """Scale a numeric feature and keep it inside a stable neural-network range."""
    safe_scale = max(abs(float(scale)), 1e-6)
    return float(np.clip(_num(value) / safe_scale, low, high))


def observation_size(use_semantic_features: bool = False, use_temporal_features: bool = False) -> int:
    size = LEGACY_OBSERVATION_SIZE
    if use_semantic_features:
        size += SEMANTIC_FEATURES
    if use_temporal_features:
        size += TEMPORAL_FEATURES
    return size


def action_size(action_mode: str = "binary") -> int:
    if action_mode == "binary":
        return 2
    if action_mode in {"macro", "tap_macro"}:
        return MACRO_ACTIONS
    raise ValueError(f"Unknown action_mode: {action_mode!r}")


def multi_observation_space(action_mode: str = "binary", recent_action_history: int = DEFAULT_RECENT_ACTION_HISTORY):
    action_count = action_size(action_mode)
    history = max(int(recent_action_history), 1)
    return spaces.Dict(
        {
            "state": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(LEGACY_OBSERVATION_SIZE,),
                dtype=np.float32,
            ),
            "semantic": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(SEMANTIC_FEATURES,),
                dtype=np.float32,
            ),
            "temporal": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(TEMPORAL_FEATURES,),
                dtype=np.float32,
            ),
            "recent_actions": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(history * action_count,),
                dtype=np.float32,
            ),
            "spatial": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(SPATIAL_MAP_CHANNELS, SPATIAL_MAP_HEIGHT, SPATIAL_MAP_WIDTH),
                dtype=np.float32,
            ),
        }
    )


def _sorted_tracks(snapshot: dict[str, Any], player_x: float | None = None) -> list[dict[str, Any]]:
    tracks = [track for track in (snapshot.get("tracksAhead") or []) if isinstance(track, dict)]
    if player_x is None:
        return sorted(tracks, key=lambda track: (_num(track.get("dx")), _num(track.get("y"))))
    return sorted(tracks, key=lambda track: (_track_left_dx(track, player_x), _track_y(track)))


def _sorted_entities(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    entities = [entity for entity in (snapshot.get("entitiesAhead") or []) if isinstance(entity, dict)]
    return sorted(entities, key=lambda entity: (_num(entity.get("dx")), _num(entity.get("y"))))


def _type_contains(value: dict[str, Any], *needles: str) -> bool:
    haystack = f"{value.get('type') or ''} {value.get('obstacleType') or ''}".lower()
    return any(needle in haystack for needle in needles)


def _pickup_kind(entity: dict[str, Any]) -> str:
    if _type_contains(entity, "fruit"):
        return "fruit"
    if _type_contains(entity, "coin", "gem"):
        return "coin"
    return "unknown"


def _is_pickup(entity: dict[str, Any]) -> bool:
    return bool(entity.get("isPickup")) or _type_contains(entity, "coin", "fruit", "gem", "pickup")


def _track_bounds(track: dict[str, Any]) -> dict[str, Any]:
    bounds = track.get("bounds") or {}
    return bounds if isinstance(bounds, dict) else {}


def _entity_bounds(entity: dict[str, Any]) -> dict[str, Any]:
    bounds = entity.get("bounds") or {}
    return bounds if isinstance(bounds, dict) else {}


def _bounds_overlap(a: dict[str, Any], b: dict[str, Any], padding: float = 0.0) -> bool:
    aw = _num(a.get("width"))
    ah = _num(a.get("height"))
    bw = _num(b.get("width"))
    bh = _num(b.get("height"))
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return False

    ax0 = _num(a.get("x")) - padding
    ay0 = _num(a.get("y")) - padding
    ax1 = _num(a.get("x")) + aw + padding
    ay1 = _num(a.get("y")) + ah + padding
    bx0 = _num(b.get("x"))
    by0 = _num(b.get("y"))
    bx1 = bx0 + bw
    by1 = by0 + bh
    return ax0 <= bx1 and ax1 >= bx0 and ay0 <= by1 and ay1 >= by0


def _track_left_dx(track: dict[str, Any], player_x: float) -> float:
    bounds = _track_bounds(track)
    if _num(bounds.get("width")) > 0:
        return _num(bounds.get("x")) - player_x
    return _num(track.get("dx"))


def _track_right_dx(track: dict[str, Any], player_x: float) -> float:
    bounds = _track_bounds(track)
    width = _num(bounds.get("width"))
    if width > 0:
        return _num(bounds.get("x")) + width - player_x
    return _num(track.get("dx"))


def _track_y(track: dict[str, Any]) -> float:
    bounds = _track_bounds(track)
    if _num(bounds.get("height")) > 0:
        return _num(bounds.get("y"))
    return _num(track.get("y"))


def _supporting_track(
    snapshot: dict[str, Any],
    tracks: list[dict[str, Any]],
    player_x: float,
    player_y: float,
) -> dict[str, Any]:
    """Return the rail physically underneath a grounded cart.

    Merely sorting every rail by x is ambiguous in multi-level layouts.  A valid
    anchor must horizontally overlap the cart and is only trusted while the
    bridge reports the cart as grounded.
    """
    player = snapshot.get("player") or {}
    if not bool(player.get("grounded")):
        return {}

    candidates = [
        track
        for track in tracks
        if _track_left_dx(track, player_x) <= TRACK_SUPPORT_TOLERANCE
        and _track_right_dx(track, player_x) >= -TRACK_SUPPORT_TOLERANCE
    ]
    if not candidates:
        return {}

    player_bounds = player.get("bounds") or {}
    player_bottom = _num(player_bounds.get("y")) + _num(player_bounds.get("height"))

    def support_distance(track: dict[str, Any]) -> tuple[float, float]:
        bounds = _track_bounds(track)
        if player_bottom and _num(bounds.get("height")) > 0:
            contact_error = abs(_num(bounds.get("y")) - player_bottom)
        else:
            contact_error = abs(_track_y(track) - player_y)
        center_error = abs((_track_left_dx(track, player_x) + _track_right_dx(track, player_x)) * 0.5)
        return contact_error, center_error

    return min(candidates, key=support_distance)


def _extend_connected_track_run(
    anchor: dict[str, Any],
    tracks: list[dict[str, Any]],
    player_x: float,
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    """Follow one connected rail run forward from an anchored track piece."""
    selected = [anchor]
    selected_ids = {id(anchor)}
    frontier = anchor
    run_right_dx = _track_right_dx(anchor, player_x)

    while True:
        candidates: list[tuple[float, float, float, dict[str, Any]]] = []
        frontier_y = _track_y(frontier)
        for track in tracks:
            if id(track) in selected_ids:
                continue
            right_dx = _track_right_dx(track, player_x)
            if right_dx <= run_right_dx + 1e-6:
                continue
            left_dx = _track_left_dx(track, player_x)
            horizontal_gap = left_dx - run_right_dx
            vertical_delta = abs(_track_y(track) - frontier_y)
            if horizontal_gap > GAP_PIXEL_THRESHOLD or vertical_delta > TRACK_CHAIN_MAX_DY:
                continue
            candidates.append((max(horizontal_gap, 0.0), vertical_delta, left_dx, track))

        if not candidates:
            break

        _gap, _dy, _left, next_track = min(candidates, key=lambda item: item[:3])
        selected.append(next_track)
        selected_ids.add(id(next_track))
        frontier = next_track
        run_right_dx = max(run_right_dx, _track_right_dx(next_track, player_x))

    return selected, frontier, run_right_dx


def _anchored_gap_geometry(
    snapshot: dict[str, Any],
    tracks: list[dict[str, Any]],
    player_x: float,
    player_y: float,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Find the first reachable gap after the cart's connected supporting rail."""
    anchor = _supporting_track(snapshot, tracks, player_x, player_y)
    if not anchor:
        return {}, {}

    takeoff_tracks, takeoff_end_track, takeoff_end_dx = _extend_connected_track_run(
        anchor,
        tracks,
        player_x,
    )
    takeoff_ids = {id(track) for track in takeoff_tracks}
    takeoff_y = _track_y(takeoff_end_track)

    landing_candidates: list[tuple[float, float, float, dict[str, Any]]] = []
    for track in tracks:
        if id(track) in takeoff_ids:
            continue
        landing_start_dx = _track_left_dx(track, player_x)
        gap_width = landing_start_dx - takeoff_end_dx
        landing_delta_y = _track_y(track) - takeoff_y
        if gap_width <= GAP_PIXEL_THRESHOLD:
            continue
        if abs(landing_delta_y) > LANDING_MAX_DY:
            continue
        landing_candidates.append((landing_start_dx, abs(landing_delta_y), gap_width, track))

    if not landing_candidates:
        return anchor, {"current_track_end_dx": takeoff_end_dx}

    landing_start_dx, _abs_dy, gap_width, landing_track = min(
        landing_candidates,
        key=lambda item: item[:3],
    )
    _landing_tracks, landing_end_track, landing_end_dx = _extend_connected_track_run(
        landing_track,
        [track for track in tracks if id(track) not in takeoff_ids],
        player_x,
    )
    landing_y = _track_y(landing_track)
    return anchor, {
        "current_track_end_dx": takeoff_end_dx,
        "gap_start_dx": takeoff_end_dx,
        "gap_width": gap_width,
        "landing_y": landing_y,
        "landing_end_dx": max(landing_end_dx, _track_right_dx(landing_end_track, player_x)),
        "landing_width": max(landing_end_dx - landing_start_dx, 0.0),
        "landing_delta_y": landing_y - takeoff_y,
        "takeoff_y": takeoff_y,
    }


def _spatial_col(dx: float) -> int:
    span = SPATIAL_MAP_X_AHEAD + SPATIAL_MAP_X_BEHIND
    ratio = (dx + SPATIAL_MAP_X_BEHIND) / span
    return int(np.clip(round(ratio * (SPATIAL_MAP_WIDTH - 1)), 0, SPATIAL_MAP_WIDTH - 1))


def _spatial_row(y: float) -> int:
    span = max(SPATIAL_MAP_Y_MAX - SPATIAL_MAP_Y_MIN, 1.0)
    ratio = (y - SPATIAL_MAP_Y_MIN) / span
    return int(np.clip(round(ratio * (SPATIAL_MAP_HEIGHT - 1)), 0, SPATIAL_MAP_HEIGHT - 1))


def _paint_spatial_rect(
    grid: np.ndarray,
    channel: int,
    left_dx: float,
    right_dx: float,
    top_y: float,
    bottom_y: float,
    value: float = 1.0,
) -> None:
    if right_dx < -SPATIAL_MAP_X_BEHIND or left_dx > SPATIAL_MAP_X_AHEAD:
        return
    if bottom_y < SPATIAL_MAP_Y_MIN or top_y > SPATIAL_MAP_Y_MAX:
        return

    left_dx, right_dx = sorted((left_dx, right_dx))
    top_y, bottom_y = sorted((top_y, bottom_y))

    c0 = _spatial_col(left_dx)
    c1 = _spatial_col(right_dx)
    r0 = _spatial_row(top_y)
    r1 = _spatial_row(bottom_y)
    if c1 < c0:
        c0, c1 = c1, c0
    if r1 < r0:
        r0, r1 = r1, r0
    grid[channel, r0 : r1 + 1, c0 : c1 + 1] = np.maximum(
        grid[channel, r0 : r1 + 1, c0 : c1 + 1],
        value,
    )


def spatial_feature_map(snapshot: dict[str, Any]) -> np.ndarray:
    """Render a small coordinate-derived map: tracks, obstacles, pickups, player.

    This is not a screen capture. It is a compact spatial encoding derived from the
    bridge's internal state, so it is stable and cheap enough for live-game training.
    """
    grid = np.zeros((SPATIAL_MAP_CHANNELS, SPATIAL_MAP_HEIGHT, SPATIAL_MAP_WIDTH), dtype=np.float32)
    player = snapshot.get("player") or {}
    position = player.get("position") or {}
    player_x = _num(position.get("x"))
    player_y = _num(position.get("y"))

    for track in _sorted_tracks(snapshot, player_x=player_x):
        left_dx = _track_left_dx(track, player_x)
        right_dx = _track_right_dx(track, player_x)
        if right_dx <= left_dx:
            right_dx = left_dx + 24.0
        y = _track_y(track)
        _paint_spatial_rect(grid, 0, left_dx, right_dx, y - 4.0, y + 12.0, value=1.0)
        if bool(track.get("hasObstacle")):
            _paint_spatial_rect(grid, 1, left_dx, right_dx, y - 24.0, y + 12.0, value=1.0)

    for entity in _sorted_entities(snapshot):
        dx = _num(entity.get("dx"))
        y = _num(entity.get("y"))
        bounds = entity.get("bounds") or {}
        width = max(_num(bounds.get("width")), 12.0)
        height = max(_num(bounds.get("height")), 12.0)
        left_dx = dx - width * 0.5
        right_dx = dx + width * 0.5
        top_y = y - height * 0.5
        bottom_y = y + height * 0.5
        if bool(entity.get("isObstacle")) or _type_contains(entity, "obstacle", "rock", "slime", "bat"):
            _paint_spatial_rect(grid, 1, left_dx, right_dx, top_y, bottom_y, value=1.0)
        if _is_pickup(entity):
            _paint_spatial_rect(grid, 2, left_dx, right_dx, top_y, bottom_y, value=1.0)

    _paint_spatial_rect(grid, 3, -8.0, 8.0, player_y - 10.0, player_y + 10.0, value=1.0)
    return grid


def semantic_feature_snapshot(
    snapshot: dict[str, Any],
    gap_detection_mode: str = "legacy",
    gap_override: dict[str, float] | None = None,
) -> dict[str, float]:
    if gap_detection_mode not in GAP_DETECTION_MODES:
        raise ValueError(
            f"gap_detection_mode must be one of {sorted(GAP_DETECTION_MODES)}, "
            f"got {gap_detection_mode!r}"
        )

    player = snapshot.get("player") or {}
    position = player.get("position") or {}
    player_x = _num(position.get("x"))
    player_y = _num(position.get("y"))

    tracks = _sorted_tracks(snapshot, player_x=player_x)
    entities = _sorted_entities(snapshot)

    forward_tracks = [track for track in tracks if _track_right_dx(track, player_x) >= 0]
    next_track = forward_tracks[0] if forward_tracks else {}
    supporting_tracks = [
        track
        for track in tracks
        if _track_left_dx(track, player_x) <= 8.0 and _track_right_dx(track, player_x) >= -8.0
    ]
    if gap_detection_mode == "anchored":
        current_track, anchored_gap = _anchored_gap_geometry(
            snapshot,
            tracks,
            player_x,
            player_y,
        )
    else:
        current_track = (
            min(supporting_tracks, key=lambda track: abs(_track_y(track) - player_y))
            if supporting_tracks
            else next_track
        )
        anchored_gap = {}
    current_track_y = _track_y(current_track)
    current_track_end_dx = _track_right_dx(current_track, player_x) if current_track else 0.0

    gap_present = 0.0
    gap_start_dx = 0.0
    gap_width = 0.0
    landing_y = 0.0
    landing_end_dx = 0.0
    landing_width = 0.0
    landing_delta_y = 0.0
    takeoff_y = current_track_y
    if gap_detection_mode == "anchored" and anchored_gap:
        current_track_end_dx = _num(anchored_gap.get("current_track_end_dx"), current_track_end_dx)
    if gap_detection_mode == "anchored" and anchored_gap.get("gap_width", 0.0) > 0.0:
        gap_present = 1.0
        gap_start_dx = _num(anchored_gap.get("gap_start_dx"))
        gap_width = _num(anchored_gap.get("gap_width"))
        landing_y = _num(anchored_gap.get("landing_y"))
        landing_end_dx = _num(anchored_gap.get("landing_end_dx"))
        landing_width = _num(anchored_gap.get("landing_width"))
        landing_delta_y = _num(anchored_gap.get("landing_delta_y"))
        takeoff_y = _num(anchored_gap.get("takeoff_y"), current_track_y)
    elif gap_detection_mode == "legacy":
        for left, right in zip(tracks, tracks[1:]):
            left_dx = _track_right_dx(left, player_x)
            right_dx = _track_left_dx(right, player_x)
            width = right_dx - left_dx
            if right_dx < 0 or width <= GAP_PIXEL_THRESHOLD:
                continue

            gap_present = 1.0
            gap_start_dx = max(left_dx, 0.0)
            gap_width = width
            landing_y = _track_y(right)
            landing_end_dx = _track_right_dx(right, player_x)
            landing_width = max(landing_end_dx - right_dx, 0.0)
            landing_delta_y = _track_y(right) - _track_y(left)
            takeoff_y = _track_y(left)
            break

    if gap_detection_mode == "anchored" and gap_override:
        gap_present = 1.0
        gap_start_dx = _num(gap_override.get("start_x")) - player_x
        gap_end_x = max(_num(gap_override.get("end_x")), _num(gap_override.get("start_x")))
        gap_width = gap_end_x - _num(gap_override.get("start_x"))
        landing_y = _num(gap_override.get("landing_y"))
        landing_end_dx = max(_num(gap_override.get("landing_end_x")), gap_end_x) - player_x
        landing_width = max(landing_end_dx - (gap_end_x - player_x), 0.0)
        takeoff_y = _num(gap_override.get("takeoff_y"), current_track_y)
        landing_delta_y = landing_y - takeoff_y
        current_track_end_dx = gap_start_dx
        current_track_y = takeoff_y

    track_obstacles = [
        {
            "dx": _track_left_dx(track, player_x),
            "y": _track_y(track),
            "type": track.get("obstacleType") or "TrackObstacle",
        }
        for track in tracks
        if _track_right_dx(track, player_x) >= 0 and bool(track.get("hasObstacle"))
    ]
    entity_obstacles = [
        {
            "dx": _num(entity.get("dx")),
            "y": _num(entity.get("y")),
            "type": entity.get("type") or "EntityObstacle",
        }
        for entity in entities
        if _num(entity.get("dx")) >= 0
        and (bool(entity.get("isObstacle")) or _type_contains(entity, "obstacle", "rock", "slime", "bat"))
    ]
    obstacles = sorted(track_obstacles + entity_obstacles, key=lambda item: (item["dx"], item["y"]))
    next_obstacle = obstacles[0] if obstacles else {}

    pickups = [
        entity
        for entity in entities
        if _num(entity.get("dx")) >= 0
        and _is_pickup(entity)
    ]
    next_pickup = pickups[0] if pickups else {}

    # MineCart.distanceToTravel is not the total level length; in live traces it can
    # stay near 150 while player_x is already above 600.  Keep the legacy slots
    # bounded and use world progress only as a weak diagnostic feature.
    distance_to_finish = 0.0
    progress_fraction = float(np.clip(player_x / 10000.0, 0.0, 1.0))

    return {
        "current_track_present": 1.0 if current_track else 0.0,
        "current_track_end_dx": current_track_end_dx,
        "current_track_y": current_track_y,
        "player_track_delta_y": player_y - current_track_y if current_track or gap_override else 0.0,
        "next_track_dx": _track_left_dx(next_track, player_x),
        "next_track_y": _track_y(next_track),
        "next_track_type_id": _num(next_track.get("typeId")),
        "next_track_has_obstacle": _bool(next_track.get("hasObstacle")),
        "next_gap_present": gap_present,
        "next_gap_start_dx": gap_start_dx,
        "next_gap_width": gap_width,
        "next_gap_end_dx": gap_start_dx + gap_width if gap_present else 0.0,
        "landing_y": landing_y,
        "landing_end_dx": landing_end_dx,
        "landing_width": landing_width,
        "landing_delta_y": landing_delta_y,
        "gap_detection_anchored": 1.0 if gap_detection_mode == "anchored" else 0.0,
        "next_obstacle_present": 1.0 if next_obstacle else 0.0,
        "next_obstacle_dx": _num(next_obstacle.get("dx")),
        "next_obstacle_y": _num(next_obstacle.get("y")),
        "next_obstacle_delta_y": _num(next_obstacle.get("y")) - player_y if next_obstacle else 0.0,
        "next_pickup_present": 1.0 if next_pickup else 0.0,
        "next_pickup_dx": _num(next_pickup.get("dx")),
        "next_pickup_y": _num(next_pickup.get("y")),
        "next_pickup_delta_y": _num(next_pickup.get("y")) - player_y if next_pickup else 0.0,
        "distance_to_finish": distance_to_finish,
        "progress_fraction": progress_fraction,
    }


class GapGeometryTracker:
    """Keep one anchored takeoff/landing pair stable while the cart is airborne."""

    def __init__(self, mode: str = "legacy") -> None:
        if mode not in GAP_DETECTION_MODES:
            raise ValueError(f"Unknown gap detection mode: {mode!r}")
        self.mode = mode
        self.geometry: dict[str, float] | None = None

    def reset(self) -> None:
        self.geometry = None

    @staticmethod
    def _absolute_geometry(snapshot: dict[str, Any], semantic: dict[str, float]) -> dict[str, float]:
        player = snapshot.get("player") or {}
        position = player.get("position") or {}
        player_x = _num(position.get("x"))
        start_x = player_x + semantic["next_gap_start_dx"]
        end_x = player_x + semantic["next_gap_end_dx"]
        return {
            "start_x": start_x,
            "end_x": end_x,
            "landing_end_x": max(player_x + semantic["landing_end_dx"], end_x),
            "landing_y": semantic["landing_y"],
            "takeoff_y": semantic["landing_y"] - semantic["landing_delta_y"],
        }

    def semantic(self, snapshot: dict[str, Any]) -> dict[str, float]:
        player = snapshot.get("player") or {}
        features = semantic_feature_snapshot(snapshot, self.mode)
        should_persist = bool(
            self.mode == "anchored"
            and self.geometry is not None
            and (
                not bool(player.get("grounded"))
                or not bool(features["current_track_present"])
            )
        )
        return (
            semantic_feature_snapshot(snapshot, self.mode, self.geometry)
            if should_persist
            else features
        )

    def update(self, snapshot: dict[str, Any]) -> dict[str, float]:
        features = semantic_feature_snapshot(snapshot, self.mode)
        if self.mode != "anchored":
            return features

        if not snapshot.get("inMinigame"):
            self.geometry = None
            return features

        player = snapshot.get("player") or {}
        if bool(player.get("grounded")):
            if features["next_gap_present"]:
                self.geometry = self._absolute_geometry(snapshot, features)
                return features
            # Junimo Kart can report grounded for a brief edge frame after the
            # supporting rail is already gone.  Do not erase the target in that
            # transition; keep it until a real landing support appears.
            if self.geometry is not None and not features["current_track_present"]:
                return semantic_feature_snapshot(snapshot, self.mode, self.geometry)
            self.geometry = None
            return features

        if self.geometry is None:
            return features

        player_x = _num((player.get("position") or {}).get("x"))
        if player_x > _num(self.geometry.get("landing_end_x")) + GAP_LANDING_REWARD_EXPIRE_PIXELS:
            self.geometry = None
            return features
        return semantic_feature_snapshot(snapshot, self.mode, self.geometry)


def _semantic_values(
    snapshot: dict[str, Any],
    semantic_features: dict[str, float] | None = None,
) -> list[float]:
    features = semantic_features or semantic_feature_snapshot(snapshot)
    return [
        features["next_track_dx"] / 1000.0,
        features["next_track_y"] / 1000.0,
        features["next_track_type_id"] / 10.0,
        features["next_track_has_obstacle"],
        features["next_gap_present"],
        features["next_gap_start_dx"] / 1000.0,
        features["next_gap_width"] / 1000.0,
        features["landing_y"] / 1000.0,
        features["landing_delta_y"] / 1000.0,
        features["next_obstacle_present"],
        features["next_obstacle_dx"] / 1000.0,
        features["next_obstacle_y"] / 1000.0,
        features["next_pickup_present"],
        features["next_pickup_dx"] / 1000.0,
        features["next_pickup_y"] / 1000.0,
        features["distance_to_finish"] / 10000.0,
        features["progress_fraction"],
    ]


def compact_feature_vector(
    snapshot: dict[str, Any],
    temporal_state: dict[str, float],
    semantic_features: dict[str, float] | None = None,
) -> np.ndarray:
    """Return a small egocentric state for sample-efficient PPO/PPO-LSTM training.

    Every value is bounded to [-1, 1].  The vector deliberately excludes the
    bridge version counter, absolute score, raw entity arrays, and the flattened
    4096-cell spatial map that made the v6/v7 model unnecessarily large.
    """
    player = snapshot.get("player") or {}
    velocity = player.get("velocity") or {}
    semantic = semantic_features or semantic_feature_snapshot(snapshot)
    grounded = bool(player.get("grounded"))
    jump_held = bool(snapshot.get("jumpHeld"))
    jump_ready = bool(player.get("jumpReady", grounded and not jump_held))

    values = [
        _clip_scale(velocity.get("x"), 160.0),
        _clip_scale(velocity.get("y"), 320.0),
        _clip_scale(semantic["player_track_delta_y"], 160.0),
        _bool(grounded),
        _bool(player.get("jumping")),
        _bool(jump_held),
        _bool(jump_ready),
        semantic["current_track_present"],
        _clip_scale(semantic["current_track_end_dx"], 320.0),
        _clip_scale(player.get("currentTrackTypeId"), 10.0, 0.0, 1.0),
        semantic["next_gap_present"],
        _clip_scale(semantic["next_gap_start_dx"], 320.0),
        _clip_scale(semantic["next_gap_width"], 160.0, 0.0, 1.0),
        _clip_scale(semantic["next_gap_end_dx"], 480.0),
        _clip_scale(semantic["landing_delta_y"], 160.0),
        semantic["next_obstacle_present"],
        _clip_scale(semantic["next_obstacle_dx"], 320.0),
        _clip_scale(semantic["next_obstacle_delta_y"], 160.0),
        semantic["next_pickup_present"],
        _clip_scale(semantic["next_pickup_dx"], 480.0),
        _clip_scale(semantic["next_pickup_delta_y"], 160.0),
        _clip_scale(temporal_state.get("jump_held_steps"), 30.0, 0.0, 1.0),
        _clip_scale(temporal_state.get("airborne_steps"), 30.0, 0.0, 1.0),
        _clip_scale(temporal_state.get("grounded_steps"), 30.0, 0.0, 1.0),
        _bool(temporal_state.get("last_action_holds_jump")),
        _clip_scale(snapshot.get("currentTheme"), 10.0, 0.0, 1.0),
        _clip_scale(snapshot.get("levelsBeat"), 6.0, 0.0, 1.0),
    ]
    array = np.asarray(values, dtype=np.float32)
    if array.shape != (COMPACT_FEATURES,):
        raise RuntimeError(f"Compact observation size mismatch: got {array.shape}, expected {(COMPACT_FEATURES,)}")
    return array


def snapshot_to_vector(
    snapshot: dict[str, Any],
    use_semantic_features: bool = False,
    temporal_values: list[float] | None = None,
    semantic_snapshot: dict[str, float] | None = None,
) -> np.ndarray:
    player = snapshot.get("player") or {}
    velocity = player.get("velocity") or {}
    position = player.get("position") or {}

    values: list[float] = [
        _bool(snapshot.get("inMinigame")),
        0.0,  # protocol version/tick is transport metadata, not gameplay state
        _num(snapshot.get("score")) / 10000.0,
        _num(snapshot.get("livesLeft")) / 10.0,
        _num(snapshot.get("levelsBeat")) / 6.0,
        _num(snapshot.get("gameMode")) / 10.0,
        _num(snapshot.get("currentTheme")) / 10.0,
        _num(snapshot.get("gameState")) / 10.0,
        _bool(snapshot.get("gameOver")),
        _bool(snapshot.get("reachedFinish")),
        _bool(snapshot.get("completed")),
        _bool(snapshot.get("jumpHeld")),
        _num(position.get("x")) / 10000.0,
        _num(position.get("y")) / 1000.0,
        _num(velocity.get("x")) / 1000.0,
        _num(velocity.get("y")) / 1000.0,
        _bool(player.get("grounded")),
        _bool(player.get("jumping")),
    ]

    tracks = list(snapshot.get("tracksAhead") or [])[:MAX_TRACKS]
    for track in tracks:
        values.extend(
            [
                _num(track.get("dx")) / 1000.0,
                _num(track.get("y")) / 1000.0,
                _num(track.get("typeId")) / 10.0,
                _bool(track.get("hasObstacle")),
                _stable_unit(track.get("obstacleType")),
            ]
        )
    values.extend([0.0] * ((MAX_TRACKS - len(tracks)) * TRACK_FEATURES))

    entities = list(snapshot.get("entitiesAhead") or [])[:MAX_ENTITIES]
    for entity in entities:
        bounds = entity.get("bounds") or {}
        values.extend(
            [
                _num(entity.get("dx")) / 1000.0,
                _num(entity.get("y")) / 1000.0,
                _num(bounds.get("width")) / 200.0,
                _num(bounds.get("height")) / 200.0,
                _stable_unit(entity.get("type")),
                _bool(entity.get("isObstacle")),
                _bool(entity.get("isPickup")),
            ]
        )
    values.extend([0.0] * ((MAX_ENTITIES - len(entities)) * ENTITY_FEATURES))

    if use_semantic_features:
        values.extend(_semantic_values(snapshot, semantic_snapshot))

    if temporal_values is not None:
        if len(temporal_values) != TEMPORAL_FEATURES:
            raise ValueError(
                f"temporal_values must contain {TEMPORAL_FEATURES} values, got {len(temporal_values)}"
            )
        values.extend(temporal_values)

    array = np.asarray(values, dtype=np.float32)
    expected_shape = (observation_size(use_semantic_features, temporal_values is not None),)
    if array.shape != expected_shape:
        raise RuntimeError(f"Observation size mismatch: got {array.shape}, expected {expected_shape}")
    return array


class JunimoKartEnv(gym.Env[Any, int]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        mode: str = "progress",
        frame_skip: int = 4,
        fps: float = 60.0,
        start_on_reset: bool = True,
        observation_mode: str = "flat",
        gap_detection_mode: str = "legacy",
        use_semantic_features: bool = False,
        use_temporal_features: bool = False,
        recent_action_history: int = DEFAULT_RECENT_ACTION_HISTORY,
        reward_version: str = "legacy",
        action_mode: str = "binary",
        macro_action_frames: int = DEFAULT_MACRO_ACTION_FRAMES,
        macro_release_frames: int = 1,
        score_reward_coef: float | None = None,
        coin_reward_coef: float | None = None,
        fruit_reward_coef: float | None = None,
        fruit_score_threshold: float = 100.0,
        gap_landing_confirm_steps: int = DEFAULT_GAP_LANDING_CONFIRM_STEPS,
        gap_landing_base_reward: float = DEFAULT_GAP_LANDING_BASE_REWARD,
        gap_landing_width_coef: float = DEFAULT_GAP_LANDING_WIDTH_COEF,
        progress_reward_coef: float = 0.01,
        death_penalty: float = 5.0,
        level_complete_reward: float = 50.0,
        game_complete_reward: float = 200.0,
        coin_reward_value: float = 0.2,
        fruit_reward_value: float = 2.0,
        jump_start_penalty: float = 0.02,
        gap_miss_penalty_coef: float = 2.0,
        airborne_hold_free_steps: int = 4,
        airborne_hold_penalty: float = 0.02,
        grounded_progress_bonus_coef: float = 0.005,
        unnecessary_jump_penalty: float = 0.15,
        non_gap_airborne_penalty: float = 0.01,
        gap_tip_technique_reward: float = 1.5,
        takeoff_tip_target_distance: float = 12.0,
        takeoff_tip_tolerance: float = 48.0,
        landing_tip_target_depth: float = 16.0,
        landing_tip_tolerance: float = 64.0,
        takeoff_target_width_coef: float = 0.65,
        takeoff_target_uphill_coef: float = 0.25,
        takeoff_target_downhill_coef: float = 0.10,
        takeoff_target_min_distance: float = 32.0,
        takeoff_target_max_distance: float = 112.0,
        takeoff_dynamic_tolerance: float = 64.0,
        gap_inaction_margin: float = 24.0,
        gap_inaction_penalty: float = 0.05,
    ) -> None:
        super().__init__()
        if observation_mode not in OBSERVATION_MODES:
            raise ValueError(f"observation_mode must be one of {sorted(OBSERVATION_MODES)}, got {observation_mode!r}")
        if action_mode not in ACTION_MODES:
            raise ValueError(f"action_mode must be one of {sorted(ACTION_MODES)}, got {action_mode!r}")
        if gap_detection_mode not in GAP_DETECTION_MODES:
            raise ValueError(
                f"gap_detection_mode must be one of {sorted(GAP_DETECTION_MODES)}, "
                f"got {gap_detection_mode!r}"
            )
        if reward_version not in REWARD_VERSIONS:
            raise ValueError(f"reward_version must be one of {sorted(REWARD_VERSIONS)}, got {reward_version!r}")
        self.client = JunimoKartBridgeClient(host=host, port=port)
        self.mode = mode
        self.frame_skip = frame_skip
        self.fps = fps
        self.start_on_reset = start_on_reset
        self.observation_mode = observation_mode
        self.gap_detection_mode = gap_detection_mode
        self.use_semantic_features = use_semantic_features
        self.use_temporal_features = use_temporal_features
        self.recent_action_history = max(int(recent_action_history), 1)
        self.reward_version = reward_version
        self.action_mode = action_mode
        self.macro_action_frames = max(int(macro_action_frames), 1)
        self.macro_release_frames = max(int(macro_release_frames), 0)
        self.score_reward_coef = score_reward_coef
        self.coin_reward_coef = coin_reward_coef
        self.fruit_reward_coef = fruit_reward_coef
        self.fruit_score_threshold = max(float(fruit_score_threshold), 0.0)
        self.gap_landing_confirm_steps = max(int(gap_landing_confirm_steps), 0)
        self.gap_landing_base_reward = float(gap_landing_base_reward)
        self.gap_landing_width_coef = float(gap_landing_width_coef)
        self.progress_reward_coef = max(float(progress_reward_coef), 0.0)
        self.death_penalty = max(float(death_penalty), 0.0)
        self.level_complete_reward = float(level_complete_reward)
        self.game_complete_reward = float(game_complete_reward)
        self.coin_reward_value = float(coin_reward_value)
        self.fruit_reward_value = float(fruit_reward_value)
        self.jump_start_penalty = max(float(jump_start_penalty), 0.0)
        self.gap_miss_penalty_coef = max(float(gap_miss_penalty_coef), 0.0)
        self.airborne_hold_free_steps = max(int(airborne_hold_free_steps), 0)
        self.airborne_hold_penalty = max(float(airborne_hold_penalty), 0.0)
        self.grounded_progress_bonus_coef = max(float(grounded_progress_bonus_coef), 0.0)
        self.unnecessary_jump_penalty = max(float(unnecessary_jump_penalty), 0.0)
        self.non_gap_airborne_penalty = max(float(non_gap_airborne_penalty), 0.0)
        self.gap_tip_technique_reward = max(float(gap_tip_technique_reward), 0.0)
        self.takeoff_tip_target_distance = float(takeoff_tip_target_distance)
        self.takeoff_tip_tolerance = max(float(takeoff_tip_tolerance), 1e-6)
        self.landing_tip_target_depth = float(landing_tip_target_depth)
        self.landing_tip_tolerance = max(float(landing_tip_tolerance), 1e-6)
        self.takeoff_target_width_coef = max(float(takeoff_target_width_coef), 0.0)
        self.takeoff_target_uphill_coef = max(float(takeoff_target_uphill_coef), 0.0)
        self.takeoff_target_downhill_coef = max(float(takeoff_target_downhill_coef), 0.0)
        self.takeoff_target_min_distance = max(float(takeoff_target_min_distance), 0.0)
        self.takeoff_target_max_distance = max(
            float(takeoff_target_max_distance),
            self.takeoff_target_min_distance,
        )
        self.takeoff_dynamic_tolerance = max(float(takeoff_dynamic_tolerance), 1e-6)
        self.gap_inaction_margin = max(float(gap_inaction_margin), 0.0)
        self.gap_inaction_penalty = max(float(gap_inaction_penalty), 0.0)
        self.action_space = spaces.Discrete(action_size(self.action_mode))
        if self.observation_mode == "multi":
            self.observation_space = multi_observation_space(self.action_mode, self.recent_action_history)
        elif self.observation_mode == "compact":
            self.observation_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(COMPACT_FEATURES,),
                dtype=np.float32,
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(observation_size(self.use_semantic_features, self.use_temporal_features),),
                dtype=np.float32,
            )
        self._last_snapshot: dict[str, Any] | None = None
        self._gap_geometry_tracker = GapGeometryTracker(self.gap_detection_mode)
        self._active_gap_attempt: dict[str, Any] | None = None
        self._episode_stats = self._new_episode_stats()
        self._temporal_state = self._new_temporal_state()
        self._recent_actions = self._new_recent_actions()
        self._last_reward_event = "none"
        self._last_reward_components: dict[str, float] = {}
        self._reward_progress_x = 0.0

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        self._gap_geometry_tracker.reset()
        self._active_gap_attempt = None
        self._episode_stats = self._new_episode_stats()
        self._temporal_state = self._new_temporal_state()
        self._recent_actions = self._new_recent_actions()
        self._last_reward_event = "reset"
        self._last_reward_components = {}
        if self.start_on_reset:
            self.client.start(options.get("mode", self.mode) if options else self.mode)
            deadline = time.time() + 5.0
            snapshot = self.client.state()
            while (not self._ready_for_agent(snapshot)) and time.time() < deadline:
                if snapshot.get("inMinigame") and snapshot.get("gameMode") == 2:
                    try:
                        self.client.advance()
                    except RuntimeError:
                        pass
                time.sleep(0.05)
                snapshot = self.client.state()
        else:
            snapshot = self.client.state()
        self._last_snapshot = snapshot
        self._gap_geometry_tracker.update(snapshot)
        self._reward_progress_x = self._player_x(snapshot)
        self._update_temporal_state(snapshot, action=0)
        return self._observation(snapshot), {
            "snapshot": snapshot,
            "semantic": self._semantic_snapshot(snapshot),
        }

    def step(self, action: int):
        old = self._last_snapshot or self.client.state()
        action_int = int(action)
        self._record_action(action_int)
        self._apply_action(action_int)
        new = self.client.state()
        self._last_snapshot = new

        self._record_transition_stats(old, new)
        self._last_reward_event = "progress"
        self._last_reward_components = {}
        reward = self._reward(old, new, action_int)
        semantic = self._gap_geometry_tracker.update(new)
        terminated = bool(new.get("completed", False))
        truncated = bool(new.get("gameOver", False) and not terminated)
        self._update_temporal_state(new, action_int)
        self._record_recent_action(action_int)
        info = {
            "snapshot": new,
            "semantic": semantic,
            "reward_event": self._last_reward_event,
            "reward_components": dict(self._last_reward_components),
        }
        info.update(self._episode_info(old))
        return self._observation(new), reward, terminated, truncated, info

    def close(self) -> None:
        self.client.close()

    def _new_episode_stats(self) -> dict[str, float]:
        return {key: 0.0 for key in TELEMETRY_INFO_KEYS}

    def _new_temporal_state(self) -> dict[str, float]:
        return {
            "jump_held_steps": 0.0,
            "airborne_steps": 0.0,
            "grounded_steps": 0.0,
            "last_action": 0.0,
            "last_action_holds_jump": 0.0,
        }

    def _new_recent_actions(self) -> list[int]:
        return [-1] * self.recent_action_history

    def _record_recent_action(self, action: int) -> None:
        self._recent_actions = [*self._recent_actions[1:], int(action)]

    def _recent_action_values(self) -> np.ndarray:
        action_count = self.action_space.n
        values = np.zeros((self.recent_action_history, action_count), dtype=np.float32)
        for row, action in enumerate(self._recent_actions):
            if 0 <= action < action_count:
                values[row, action] = 1.0
        return values.reshape(-1)

    def _update_temporal_state(self, snapshot: dict[str, Any], action: int) -> None:
        player = snapshot.get("player") or {}
        if bool(snapshot.get("jumpHeld")):
            self._temporal_state["jump_held_steps"] += 1.0
        else:
            self._temporal_state["jump_held_steps"] = 0.0

        if bool(player.get("grounded")):
            self._temporal_state["grounded_steps"] += 1.0
            self._temporal_state["airborne_steps"] = 0.0
        else:
            self._temporal_state["airborne_steps"] += 1.0
            self._temporal_state["grounded_steps"] = 0.0

        self._temporal_state["last_action"] = float(action)
        self._temporal_state["last_action_holds_jump"] = 1.0 if self._action_holds_jump(action) else 0.0

    def _temporal_values(self) -> list[float]:
        max_action = max(self.action_space.n - 1, 1)
        return [
            min(self._temporal_state["jump_held_steps"], 60.0) / 60.0,
            min(self._temporal_state["airborne_steps"], 60.0) / 60.0,
            min(self._temporal_state["grounded_steps"], 60.0) / 60.0,
            self._temporal_state["last_action"] / max_action,
            self._temporal_state["last_action_holds_jump"],
        ]

    def _observation(self, snapshot: dict[str, Any]) -> Any:
        semantic = self._semantic_snapshot(snapshot)
        if self.observation_mode == "compact":
            return compact_feature_vector(snapshot, self._temporal_state, semantic)
        if self.observation_mode == "multi":
            return {
                "state": snapshot_to_vector(snapshot, use_semantic_features=False),
                "semantic": np.asarray(_semantic_values(snapshot, semantic), dtype=np.float32),
                "temporal": np.asarray(self._temporal_values(), dtype=np.float32),
                "recent_actions": self._recent_action_values(),
                "spatial": spatial_feature_map(snapshot),
            }
        temporal_values = self._temporal_values() if self.use_temporal_features else None
        return snapshot_to_vector(
            snapshot,
            self.use_semantic_features,
            temporal_values,
            semantic,
        )

    def _semantic_snapshot(self, snapshot: dict[str, Any]) -> dict[str, float]:
        return self._gap_geometry_tracker.semantic(snapshot)

    def _episode_info(self, final_reference: dict[str, Any]) -> dict[str, float]:
        semantic = self._semantic_snapshot(final_reference)
        self._episode_stats["final_gap_start_dx"] = semantic["next_gap_start_dx"]
        self._episode_stats["final_gap_width"] = semantic["next_gap_width"]
        self._episode_stats["final_obstacle_dx"] = semantic["next_obstacle_dx"]
        return dict(self._episode_stats)

    def _increment_stat(self, key: str, amount: float = 1.0) -> None:
        self._episode_stats[key] = self._episode_stats.get(key, 0.0) + amount

    def _set_reward_event(self, event: str) -> None:
        self._last_reward_event = event

    def _record_action(self, action: int) -> None:
        key = f"action_{action}_count"
        if key in self._episode_stats:
            self._increment_stat(key)

    def _record_transition_stats(self, old: dict[str, Any], new: dict[str, Any]) -> None:
        old_player = old.get("player") or {}
        new_player = new.get("player") or {}
        old_pos = old_player.get("position") or {}
        new_pos = new_player.get("position") or {}

        score_delta = _num(new.get("score")) - _num(old.get("score"))
        if self._collected_pickup(old, new) is not None:
            self._increment_stat("pickup_events")
        self._increment_stat("score_delta_total", score_delta)

        new_x = _num(new_pos.get("x"))
        self._episode_stats["max_episode_x"] = max(self._episode_stats["max_episode_x"], new_x)
        self._record_state_stats(new)

        game_over_started = bool(new.get("gameOver") and not old.get("gameOver"))
        if not game_over_started:
            return

        semantic = self._semantic_snapshot(old)
        near_gap = bool(
            semantic["next_gap_present"]
            and semantic["next_gap_width"] >= GAP_LANDING_REWARD_MIN_WIDTH
            and 0.0 <= semantic["next_gap_start_dx"] <= 160.0
        )
        near_obstacle = bool(
            semantic["next_obstacle_present"]
            and 0.0 <= semantic["next_obstacle_dx"] <= 140.0
        )
        if near_gap:
            self._episode_stats["death_near_gap"] = 1.0
        if near_obstacle:
            self._episode_stats["death_near_obstacle"] = 1.0

    def _record_state_stats(self, snapshot: dict[str, Any]) -> None:
        semantic = self._semantic_snapshot(snapshot)
        player = snapshot.get("player") or {}

        self._increment_stat("state_samples")

        if semantic["next_gap_present"]:
            self._increment_stat("gap_visible_steps")
            self._increment_stat("sum_gap_start_dx", semantic["next_gap_start_dx"])
            self._increment_stat("sum_gap_width", semantic["next_gap_width"])
            self._increment_stat("sum_landing_delta_y", semantic["landing_delta_y"])
            if (
                semantic["next_gap_width"] >= GAP_LANDING_REWARD_MIN_WIDTH
                and 0.0 <= semantic["next_gap_start_dx"] <= GAP_LANDING_REWARD_ACTIVATION_DX
            ):
                self._increment_stat("gap_near_steps")

        if semantic["next_obstacle_present"]:
            self._increment_stat("obstacle_visible_steps")
            self._increment_stat("sum_obstacle_dx", semantic["next_obstacle_dx"])
            if 0.0 <= semantic["next_obstacle_dx"] <= 120.0:
                self._increment_stat("obstacle_near_steps")

        if semantic["next_pickup_present"]:
            self._increment_stat("pickup_visible_steps")
            self._increment_stat("sum_pickup_dx", semantic["next_pickup_dx"])

        if bool(player.get("grounded")):
            self._increment_stat("grounded_steps_total")
        if bool(snapshot.get("jumpHeld")):
            self._increment_stat("jump_held_steps_total")

    def _score_reward(self, score_delta: float, default_coef: float) -> float:
        coef = default_coef if self.score_reward_coef is None else self.score_reward_coef
        return coef * score_delta

    def _uses_split_pickup_reward(self) -> bool:
        return self.coin_reward_coef is not None or self.fruit_reward_coef is not None

    @staticmethod
    def _entity_identity(entity: dict[str, Any]) -> str | None:
        entity_id = entity.get("id")
        if entity_id is None:
            return None
        return str(entity_id)

    def _collected_pickup(
        self,
        old: dict[str, Any],
        new: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Detect a pickup that disappeared while it was close to the player.

        Score cannot be used as the pickup signal because Junimo Kart also raises
        score while travelling.  Bridge v8 gives every live entity a stable runtime
        id, so a nearby pickup present in ``old`` and absent in ``new`` is a much
        stronger collection event.
        """
        new_ids = {
            identity
            for entity in (new.get("entitiesAhead") or [])
            if isinstance(entity, dict)
            for identity in [self._entity_identity(entity)]
            if identity is not None and _is_pickup(entity)
        }
        old_player = old.get("player") or {}
        old_player_pos = old_player.get("position") or {}
        old_player_bounds = old_player.get("bounds") or {}
        player_y = _num(old_player_pos.get("y"))

        candidates: list[tuple[float, dict[str, Any]]] = []
        for entity in old.get("entitiesAhead") or []:
            if not isinstance(entity, dict) or not _is_pickup(entity):
                continue
            identity = self._entity_identity(entity)
            if identity is None or identity in new_ids:
                continue
            if entity.get("visible") is False or entity.get("enabled") is False:
                continue

            dx = _num(entity.get("dx"))
            dy = abs(_num(entity.get("y")) - player_y)
            overlaps = _bounds_overlap(old_player_bounds, _entity_bounds(entity), padding=16.0)
            close = -40.0 <= dx <= 56.0 and dy <= 72.0
            if not overlaps and not close:
                continue
            candidates.append((abs(dx) + 0.25 * dy, entity))

        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def _nearby_collected_pickup(self, snapshot: dict[str, Any]) -> dict[str, Any] | None:
        player = snapshot.get("player") or {}
        player_pos = player.get("position") or {}
        player_bounds = player.get("bounds") or {}
        player_y = _num(player_pos.get("y"))

        candidates: list[tuple[float, dict[str, Any]]] = []
        for entity in snapshot.get("entitiesAhead") or []:
            if not isinstance(entity, dict) or not _is_pickup(entity):
                continue
            if entity.get("visible") is False or entity.get("enabled") is False:
                continue

            entity_bounds = _entity_bounds(entity)
            dx = _num(entity.get("dx"))
            dy = abs(_num(entity.get("y")) - player_y)
            overlaps = _bounds_overlap(player_bounds, entity_bounds, padding=24.0)
            close = -48.0 <= dx <= 72.0 and dy <= 96.0
            if not overlaps and not close:
                continue

            priority = 0.0 if overlaps else 1000.0
            candidates.append((priority + abs(dx) + 0.25 * dy, entity))

        if not candidates:
            return None
        return sorted(candidates, key=lambda item: item[0])[0][1]

    def _pickup_score_reward(
        self,
        old: dict[str, Any],
        new: dict[str, Any],
        score_delta: float,
        default_coef: float,
    ) -> float:
        if score_delta <= 0.0:
            return 0.0

        if not self._uses_split_pickup_reward():
            return self._score_reward(score_delta, default_coef=default_coef)

        pickup = self._collected_pickup(old, new)
        if pickup is None:
            return 0.0
        kind = _pickup_kind(pickup)
        if kind == "unknown" and score_delta >= self.fruit_score_threshold:
            kind = "fruit"

        if kind == "fruit":
            coef = self.fruit_reward_coef if self.fruit_reward_coef is not None else self.score_reward_coef
            reward = (coef or 0.0) * score_delta
            self._increment_stat("fruit_events")
            self._increment_stat("fruit_reward_total", reward)
            if reward != 0.0:
                self._set_reward_event("fruit")
            return reward

        if kind == "coin":
            coef = self.coin_reward_coef if self.coin_reward_coef is not None else self.score_reward_coef
            reward = (coef or 0.0) * score_delta
            self._increment_stat("coin_events")
            self._increment_stat("coin_reward_total", reward)
            if reward != 0.0:
                self._set_reward_event("coin")
            return reward

        self._increment_stat("unknown_pickup_events")
        return 0.0

    def _pickup_fixed_reward(
        self,
        old: dict[str, Any],
        new: dict[str, Any],
        score_delta: float,
    ) -> float:
        pickup = self._collected_pickup(old, new)
        if pickup is None:
            return 0.0

        kind = _pickup_kind(pickup)
        if kind == "unknown" and score_delta >= self.fruit_score_threshold:
            kind = "fruit"

        if kind == "coin":
            reward = self.coin_reward_value
            self._increment_stat("coin_events")
            self._increment_stat("coin_reward_total", reward)
            self._set_reward_event("coin")
            return reward
        if kind == "fruit":
            reward = self.fruit_reward_value
            self._increment_stat("fruit_events")
            self._increment_stat("fruit_reward_total", reward)
            self._set_reward_event("fruit")
            return reward

        self._increment_stat("unknown_pickup_events")
        return 0.0

    def _ready_for_agent(self, snapshot: dict[str, Any]) -> bool:
        return bool(
            snapshot.get("inMinigame")
            and snapshot.get("gameMode") == 2
            and snapshot.get("gameState") == 1
            and snapshot.get("player")
        )

    def _sleep_frames(self, frames: int) -> None:
        time.sleep(max(int(frames), 1) / self.fps)

    def _apply_action(self, action: int) -> None:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r} for action_mode={self.action_mode!r}")

        if self.action_mode == "binary":
            self.client.action(action == 1)
            self._sleep_frames(self.frame_skip)
            return

        total_frames = self.macro_action_frames
        hold_frames = self._macro_hold_frames(action)
        if self.action_mode == "tap_macro" and action > 0 and total_frames > 1:
            forced_release_frames = min(max(self.macro_release_frames, 1), total_frames - 1)
            hold_frames = min(hold_frames, total_frames - forced_release_frames)

        if hold_frames > 0:
            self.client.action(True)
            self._sleep_frames(min(hold_frames, total_frames))

        release_frames = total_frames - hold_frames
        if release_frames > 0:
            self.client.action(False)
            self._sleep_frames(release_frames)

    def _macro_hold_frames(self, action: int) -> int:
        total_frames = self.macro_action_frames
        if action == 0:
            return 0
        if action == 1:
            return max(1, round(total_frames * 0.25))
        if action == 2:
            return max(1, round(total_frames * 0.50))
        if action == 3:
            return total_frames
        raise ValueError(f"Invalid macro action: {action!r}")

    def _action_holds_jump(self, action: int) -> bool:
        if self.action_mode == "binary":
            return action == 1
        return action > 0

    def _jump_started(self, snapshot: dict[str, Any], action: int) -> bool:
        """True only for a real grounded release-to-hold jump edge.

        Airborne re-presses are ignored by bridge v8, so charging them would teach
        against an action that never created a jump in the game.
        """
        player = snapshot.get("player") or {}
        grounded = bool(player.get("grounded"))
        jump_ready = bool(player.get("jumpReady", grounded and not snapshot.get("jumpHeld")))
        previous_holds = bool(self._temporal_state.get("last_action_holds_jump"))
        return self._action_holds_jump(action) and not previous_holds and grounded and jump_ready

    def _record_jump_start_diagnostics(self, snapshot: dict[str, Any]) -> None:
        self._increment_stat("jump_start_events")
        semantic = self._semantic_snapshot(snapshot)
        near_gap = bool(
            semantic["next_gap_present"]
            and semantic["next_gap_width"] >= GAP_LANDING_REWARD_MIN_WIDTH
            and 0.0 <= semantic["next_gap_start_dx"] <= GAP_LANDING_REWARD_ACTIVATION_DX
        )
        if near_gap:
            self._increment_stat("jump_start_near_gap_events")
        else:
            self._increment_stat("jump_start_without_near_gap_events")

    def _player_x(self, snapshot: dict[str, Any]) -> float:
        player = snapshot.get("player") or {}
        position = player.get("position") or {}
        return _num(position.get("x"))

    def _player_front_x(self, snapshot: dict[str, Any]) -> float:
        """Return the cart's forward collision edge in world coordinates."""
        player = snapshot.get("player") or {}
        bounds = player.get("bounds") or {}
        width = _num(bounds.get("width"))
        if width > 0.0:
            return _num(bounds.get("x")) + width
        return self._player_x(snapshot) + 8.0

    @staticmethod
    def _tip_quality(value: float, target: float, tolerance: float) -> float:
        """Triangular quality score: 1 at target, fading smoothly to 0."""
        return float(np.clip(1.0 - abs(value - target) / max(tolerance, 1e-6), 0.0, 1.0))

    def _dynamic_takeoff_target(self, active_gap: dict[str, Any]) -> float:
        """Estimate a broad takeoff target from the required flight geometry.

        Wider gaps and higher landing rails need earlier takeoff.  Lower landing
        rails allow a slightly later takeoff.  This is a shaping target, not a
        hard action rule, and its bonus is still paid only after success.
        """
        width = max(_num(active_gap.get("width")), GAP_LANDING_REWARD_MIN_WIDTH)
        landing_delta_y = _num(active_gap.get("landing_delta_y"))
        uphill = max(-landing_delta_y, 0.0)
        downhill = max(landing_delta_y, 0.0)
        target = (
            self.takeoff_target_width_coef * width
            + self.takeoff_target_uphill_coef * uphill
            - self.takeoff_target_downhill_coef * downhill
        )
        return float(
            np.clip(
                target,
                self.takeoff_target_min_distance,
                self.takeoff_target_max_distance,
            )
        )

    def _takeoff_target(self, active_gap: dict[str, Any]) -> tuple[float, float]:
        if self.reward_version == "shaped_v9":
            return self._dynamic_takeoff_target(active_gap), self.takeoff_dynamic_tolerance
        return self.takeoff_tip_target_distance, self.takeoff_tip_tolerance

    def _gap_inaction_threshold(self, active_gap: dict[str, Any]) -> float:
        target, _tolerance = self._takeoff_target(active_gap)
        # Gap dx is measured from player center while takeoff distance is measured
        # from the cart's forward edge (approximately 8 px ahead).
        return target + 8.0 + self.gap_inaction_margin

    def _record_gap_takeoff(self, snapshot: dict[str, Any]) -> float | None:
        """Record the latest real takeoff associated with the active gap.

        The value is diagnostic only at takeoff time.  No technique reward is
        paid until this same gap is landed successfully.
        """
        self._maybe_start_gap_attempt(snapshot)
        active_gap = self._active_gap_attempt
        if active_gap is None:
            return None

        distance_to_tip = _num(active_gap.get("start_x")) - self._player_front_x(snapshot)
        takeoff_target, takeoff_tolerance = self._takeoff_target(active_gap)
        quality = self._tip_quality(
            distance_to_tip,
            takeoff_target,
            takeoff_tolerance,
        )
        active_gap["takeoff_recorded"] = True
        active_gap["takeoff_tip_distance"] = distance_to_tip
        active_gap["takeoff_target_distance"] = takeoff_target
        active_gap["takeoff_tip_quality"] = quality
        self._increment_stat("gap_takeoff_events")
        self._increment_stat("takeoff_tip_distance_total", distance_to_tip)
        self._increment_stat("takeoff_target_distance_total", takeoff_target)
        self._increment_stat("takeoff_tip_quality_total", quality)
        return quality

    def _prepare_gap_tip_technique(self, active_gap: dict[str, Any], landing: dict[str, Any]) -> None:
        """Store a bounded takeoff+landing quality bonus on a successful gap.

        A geometric mean makes the bonus conjunctive: both the takeoff and the
        landing must be reasonably close to their safe tip targets.  Merely
        jumping at the edge without landing cannot earn this reward.
        """
        landing_depth = self._player_x(landing) - _num(active_gap.get("end_x"))
        landing_quality = self._tip_quality(
            landing_depth,
            self.landing_tip_target_depth,
            self.landing_tip_tolerance,
        )
        takeoff_quality = (
            _num(active_gap.get("takeoff_tip_quality"))
            if bool(active_gap.get("takeoff_recorded"))
            else 0.0
        )
        technique_quality = float(np.sqrt(max(takeoff_quality * landing_quality, 0.0)))
        active_gap["landing_tip_depth"] = landing_depth
        active_gap["landing_tip_quality"] = landing_quality
        active_gap["edge_technique_reward"] = self.gap_tip_technique_reward * technique_quality

    def _record_gap_tip_success(self, active_gap: dict[str, Any]) -> None:
        """Emit technique telemetry only after the landing is confirmed."""
        edge_reward = _num(active_gap.get("edge_technique_reward"))
        self._last_reward_components["gap_tip_technique"] = edge_reward
        if self.reward_version not in {"shaped_v8", "shaped_v9"} or not bool(active_gap.get("takeoff_recorded")):
            return

        self._increment_stat("edge_qualified_landings")
        self._increment_stat(
            "successful_takeoff_tip_quality_total",
            _num(active_gap.get("takeoff_tip_quality")),
        )
        self._increment_stat(
            "successful_takeoff_tip_distance_total",
            _num(active_gap.get("takeoff_tip_distance")),
        )
        self._increment_stat(
            "successful_takeoff_target_distance_total",
            _num(active_gap.get("takeoff_target_distance")),
        )
        self._increment_stat("landing_tip_depth_total", _num(active_gap.get("landing_tip_depth")))
        self._increment_stat("landing_tip_quality_total", _num(active_gap.get("landing_tip_quality")))
        self._increment_stat("edge_technique_reward_total", edge_reward)

    def _pay_gap_landing(self, active_gap: dict[str, Any]) -> float:
        reward_value = _num(active_gap.get("reward_value"))
        self._increment_stat("gap_landings")
        self._record_gap_tip_success(active_gap)
        self._active_gap_attempt = None
        self._set_reward_event("gap_landing")
        return reward_value

    def _maybe_start_gap_attempt(self, snapshot: dict[str, Any]) -> None:
        if self._active_gap_attempt is not None:
            return

        semantic = self._semantic_snapshot(snapshot)
        gap_width = semantic["next_gap_width"]
        gap_start_dx = semantic["next_gap_start_dx"]
        if not semantic["next_gap_present"]:
            return
        if gap_width < GAP_LANDING_REWARD_MIN_WIDTH:
            return
        if not (0.0 <= gap_start_dx <= GAP_LANDING_REWARD_ACTIVATION_DX):
            return

        player_x = self._player_x(snapshot)
        gap_start_x = player_x + gap_start_dx
        self._increment_stat("gap_attempts")
        self._active_gap_attempt = {
            "start_x": gap_start_x,
            "end_x": gap_start_x + gap_width,
            "landing_end_x": max(
                player_x + semantic.get("landing_end_dx", gap_start_dx + gap_width),
                gap_start_x + gap_width,
            ),
            "width": gap_width,
            "landing_y": semantic["landing_y"],
            "landing_delta_y": semantic["landing_delta_y"],
            "furthest_x": player_x,
            "landed": False,
            "confirm_steps_remaining": 0,
            "reward_value": 0.0,
            "takeoff_recorded": False,
            "takeoff_tip_distance": 0.0,
            "takeoff_target_distance": 0.0,
            "takeoff_tip_quality": 0.0,
            "landing_tip_depth": 0.0,
            "landing_tip_quality": 0.0,
            "edge_technique_reward": 0.0,
        }

    def _gap_landing_reward(
        self,
        old: dict[str, Any],
        new: dict[str, Any],
        confirm_steps: int = 0,
        base_reward: float = 3.0,
        width_reward_coef: float = 0.02,
        death_reward: float = -5.0,
    ) -> float:
        self._maybe_start_gap_attempt(old)
        active_gap = self._active_gap_attempt
        if active_gap is None:
            return 0.0

        active_gap["furthest_x"] = max(
            _num(active_gap.get("furthest_x")),
            self._player_x(old),
            self._player_x(new),
        )

        game_over_started = bool(new.get("gameOver") and not old.get("gameOver"))
        if game_over_started:
            self._increment_stat("gap_failures")
            self._increment_stat("gap_deaths")
            self._active_gap_attempt = None
            self._set_reward_event("gap_death")
            return float(death_reward)

        if new.get("gameOver") or not new.get("inMinigame"):
            self._increment_stat("gap_failures")
            self._active_gap_attempt = None
            return 0.0

        if bool(active_gap.get("landed")):
            active_gap["confirm_steps_remaining"] = max(int(active_gap["confirm_steps_remaining"]) - 1, 0)
            if active_gap["confirm_steps_remaining"] <= 0:
                return self._pay_gap_landing(active_gap)
            return 0.0

        new_player = new.get("player") or {}
        new_x = self._player_x(new)
        if new_x >= active_gap["end_x"] and bool(new_player.get("grounded")):
            width_bonus = width_reward_coef * min(active_gap["width"], 120.0)
            reward_value = base_reward + width_bonus
            if self.reward_version in {"shaped_v8", "shaped_v9"}:
                self._prepare_gap_tip_technique(active_gap, new)
                reward_value += _num(active_gap.get("edge_technique_reward"))
            active_gap["reward_value"] = reward_value
            if confirm_steps <= 0:
                return self._pay_gap_landing(active_gap)

            active_gap["landed"] = True
            active_gap["confirm_steps_remaining"] = confirm_steps
            return 0.0

        if new_x > active_gap["end_x"] + GAP_LANDING_REWARD_EXPIRE_PIXELS:
            self._increment_stat("gap_failures")
            self._active_gap_attempt = None

        return 0.0

    def _gap_miss_metrics(
        self,
        old: dict[str, Any],
        new: dict[str, Any],
    ) -> tuple[float, float, bool]:
        """Return horizontal miss distance and normalized miss ratio at gap death.

        Distance is measured to the entire landing-track interval, not to one
        arbitrary point.  The ratio is normalized by gap width and capped at 1 so
        extreme failures cannot dominate the reward scale.
        """
        game_over_started = bool(new.get("gameOver") and not old.get("gameOver"))
        if not game_over_started:
            return 0.0, 0.0, False

        self._maybe_start_gap_attempt(old)
        active_gap = self._active_gap_attempt
        if active_gap is None:
            return 0.0, 0.0, False

        furthest_x = max(
            _num(active_gap.get("furthest_x")),
            self._player_x(old),
            self._player_x(new),
        )
        landing_start_x = _num(active_gap.get("end_x"))
        landing_end_x = max(_num(active_gap.get("landing_end_x"), landing_start_x), landing_start_x)
        if furthest_x < landing_start_x:
            distance = landing_start_x - furthest_x
        elif furthest_x > landing_end_x:
            distance = furthest_x - landing_end_x
        else:
            distance = 0.0

        normalizer = max(_num(active_gap.get("width")), 32.0)
        ratio = float(np.clip(distance / normalizer, 0.0, 1.0))
        return distance, ratio, True

    def _reward(self, old: dict[str, Any], new: dict[str, Any], action: int = 0) -> float:
        if self.reward_version == "legacy":
            return self._legacy_reward(old, new)
        if self.reward_version == "shaped_v1":
            return self._shaped_v1_reward(old, new, action)
        if self.reward_version == "shaped_v2":
            return self._shaped_v2_reward(old, new, action)
        if self.reward_version == "shaped_v3":
            return self._shaped_v3_reward(old, new, action)
        if self.reward_version == "shaped_v4":
            return self._shaped_v4_reward(old, new, action)
        if self.reward_version == "shaped_v5":
            return self._shaped_v5_reward(old, new, action)
        if self.reward_version == "shaped_v6":
            return self._shaped_v6_reward(old, new, action)
        if self.reward_version == "shaped_v7":
            return self._shaped_v7_reward(old, new, action)
        if self.reward_version == "shaped_v8":
            return self._shaped_v8_reward(old, new, action)
        if self.reward_version == "shaped_v9":
            return self._shaped_v9_reward(old, new, action)
        raise ValueError(f"Unknown reward_version: {self.reward_version!r}")

    def _legacy_reward(self, old: dict[str, Any], new: dict[str, Any]) -> float:
        old_player = old.get("player") or {}
        new_player = new.get("player") or {}
        old_pos = old_player.get("position") or {}
        new_pos = new_player.get("position") or {}

        dx = _num(new_pos.get("x")) - _num(old_pos.get("x"))
        score_delta = _num(new.get("score")) - _num(old.get("score"))
        level_delta = _num(new.get("levelsBeat")) - _num(old.get("levelsBeat"))
        life_delta = _num(new.get("livesLeft")) - _num(old.get("livesLeft"))

        reward = 0.001 * dx
        reward += self._pickup_score_reward(old, new, score_delta, default_coef=0.01)
        reward += 50.0 * max(level_delta, 0.0)
        reward += 10.0 * life_delta

        if new.get("completed"):
            reward += 250.0
        elif new.get("gameOver") and not old.get("gameOver"):
            reward -= 100.0

        if not new.get("inMinigame"):
            reward -= 1.0

        return float(reward)

    def _shaped_v1_reward(self, old: dict[str, Any], new: dict[str, Any], action: int = 0) -> float:
        old_player = old.get("player") or {}
        new_player = new.get("player") or {}
        old_pos = (old_player.get("position") or {})
        new_pos = (new_player.get("position") or {})

        dx = _num(new_pos.get("x")) - _num(old_pos.get("x"))
        score_delta = _num(new.get("score")) - _num(old.get("score"))
        level_delta = _num(new.get("levelsBeat")) - _num(old.get("levelsBeat"))
        life_delta = _num(new.get("livesLeft")) - _num(old.get("livesLeft"))

        reward = 0.003 * dx
        reward += self._pickup_score_reward(old, new, score_delta, default_coef=0.02)
        reward += 100.0 * max(level_delta, 0.0)
        reward += 25.0 * life_delta

        in_gameplay = bool(new.get("inMinigame") and new.get("gameState") == 1)
        game_over_started = bool(new.get("gameOver") and not old.get("gameOver"))
        if in_gameplay and not new.get("gameOver"):
            reward += 0.02

        if new.get("completed"):
            reward += 500.0
        elif game_over_started:
            reward -= 80.0

        if not new.get("inMinigame"):
            reward -= 2.0

        semantic = self._semantic_snapshot(old)
        gap_near = bool(
            semantic["next_gap_present"]
            and 0.0 <= semantic["next_gap_start_dx"] <= 120.0
        )
        obstacle_near = bool(
            semantic["next_obstacle_present"]
            and 0.0 <= semantic["next_obstacle_dx"] <= 120.0
        )
        grounded = bool(old_player.get("grounded"))
        jumping = bool(old_player.get("jumping"))
        holds_jump = self._action_holds_jump(action)

        if gap_near and grounded:
            if holds_jump:
                reward += 0.08
            elif semantic["next_gap_start_dx"] <= 80.0:
                reward -= 0.08

        if holds_jump and grounded and not gap_near and not obstacle_near:
            reward -= 0.015

        landed_safely = bool(jumping and new_player.get("grounded") and not new.get("gameOver"))
        if landed_safely:
            reward += 0.20

        if dx < -5 and not level_delta:
            reward += 0.003 * dx

        return float(reward)

    def _shaped_v2_reward(self, old: dict[str, Any], new: dict[str, Any], action: int = 0) -> float:
        old_player = old.get("player") or {}
        new_player = new.get("player") or {}
        old_pos = old_player.get("position") or {}
        new_pos = new_player.get("position") or {}

        dx = _num(new_pos.get("x")) - _num(old_pos.get("x"))
        score_delta = _num(new.get("score")) - _num(old.get("score"))
        level_delta = _num(new.get("levelsBeat")) - _num(old.get("levelsBeat"))
        life_delta = _num(new.get("livesLeft")) - _num(old.get("livesLeft"))

        reward = 0.004 * dx
        reward += self._pickup_score_reward(old, new, score_delta, default_coef=0.005)
        reward += 120.0 * max(level_delta, 0.0)
        reward += 25.0 * life_delta

        in_gameplay = bool(new.get("inMinigame") and new.get("gameState") == 1)
        game_over_started = bool(new.get("gameOver") and not old.get("gameOver"))
        if in_gameplay and not new.get("gameOver"):
            reward += 0.03

        if new.get("completed"):
            reward += 600.0
        elif game_over_started:
            reward -= 80.0

        if not new.get("inMinigame"):
            reward -= 2.0

        semantic = self._semantic_snapshot(old)
        gap_near = bool(
            semantic["next_gap_present"]
            and semantic["next_gap_width"] >= GAP_LANDING_REWARD_MIN_WIDTH
            and 0.0 <= semantic["next_gap_start_dx"] <= GAP_LANDING_REWARD_ACTIVATION_DX
        )
        gap_very_near = bool(gap_near and semantic["next_gap_start_dx"] <= 80.0)
        obstacle_near = bool(
            semantic["next_obstacle_present"]
            and 0.0 <= semantic["next_obstacle_dx"] <= 120.0
        )
        grounded = bool(old_player.get("grounded"))
        holds_jump = self._action_holds_jump(action)

        if gap_very_near and grounded and not holds_jump:
            reward -= 0.03

        if holds_jump and grounded and not gap_near and not obstacle_near:
            reward -= 0.02

        reward += self._gap_landing_reward(old, new, confirm_steps=0)

        if dx < -5 and not level_delta:
            reward += 0.004 * dx

        return float(reward)

    def _shaped_v3_reward(self, old: dict[str, Any], new: dict[str, Any], action: int = 0) -> float:
        old_player = old.get("player") or {}
        new_player = new.get("player") or {}
        old_pos = old_player.get("position") or {}
        new_pos = new_player.get("position") or {}

        dx = _num(new_pos.get("x")) - _num(old_pos.get("x"))
        score_delta = _num(new.get("score")) - _num(old.get("score"))
        level_delta = _num(new.get("levelsBeat")) - _num(old.get("levelsBeat"))
        life_delta = _num(new.get("livesLeft")) - _num(old.get("livesLeft"))

        reward = 0.006 * dx
        reward += self._pickup_score_reward(old, new, score_delta, default_coef=0.0)
        reward += 150.0 * max(level_delta, 0.0)
        reward += 30.0 * life_delta

        in_gameplay = bool(new.get("inMinigame") and new.get("gameState") == 1)
        game_over_started = bool(new.get("gameOver") and not old.get("gameOver"))
        if in_gameplay and not new.get("gameOver"):
            reward += 0.035

        if new.get("completed"):
            reward += 700.0
        elif game_over_started:
            reward -= 80.0

        if not new.get("inMinigame"):
            reward -= 2.0

        semantic = self._semantic_snapshot(old)
        gap_near = bool(
            semantic["next_gap_present"]
            and semantic["next_gap_width"] >= GAP_LANDING_REWARD_MIN_WIDTH
            and 0.0 <= semantic["next_gap_start_dx"] <= GAP_LANDING_REWARD_ACTIVATION_DX
        )
        gap_approach_zone = bool(gap_near and 55.0 <= semantic["next_gap_start_dx"] <= 160.0)
        gap_very_near = bool(gap_near and semantic["next_gap_start_dx"] <= 80.0)
        obstacle_near = bool(
            semantic["next_obstacle_present"]
            and 0.0 <= semantic["next_obstacle_dx"] <= 120.0
        )
        grounded = bool(old_player.get("grounded"))
        holds_jump = self._action_holds_jump(action)

        if gap_approach_zone and grounded and holds_jump:
            reward += 0.02

        if gap_very_near and grounded and not holds_jump:
            reward -= 0.05

        if obstacle_near and grounded and holds_jump:
            reward += 0.015

        if holds_jump and grounded and not gap_near and not obstacle_near:
            reward -= 0.035

        reward += self._gap_landing_reward(
            old,
            new,
            confirm_steps=self.gap_landing_confirm_steps,
        )

        if dx < -5 and not level_delta:
            reward += 0.006 * dx

        return float(reward)

    def _shaped_v4_reward(self, old: dict[str, Any], new: dict[str, Any], action: int = 0) -> float:
        old_player = old.get("player") or {}
        new_player = new.get("player") or {}
        old_pos = old_player.get("position") or {}
        new_pos = new_player.get("position") or {}

        dx = _num(new_pos.get("x")) - _num(old_pos.get("x"))
        score_delta = _num(new.get("score")) - _num(old.get("score"))
        level_delta = _num(new.get("levelsBeat")) - _num(old.get("levelsBeat"))
        life_delta = _num(new.get("livesLeft")) - _num(old.get("livesLeft"))

        reward = 0.006 * max(dx, 0.0)
        reward += self._pickup_score_reward(old, new, score_delta, default_coef=0.0)
        reward += 150.0 * max(level_delta, 0.0)

        if life_delta < 0.0:
            reward += 30.0 * life_delta
            self._set_reward_event("life_lost")

        in_gameplay = bool(new.get("inMinigame") and new.get("gameState") == 1)
        game_over_started = bool(new.get("gameOver") and not old.get("gameOver"))
        if in_gameplay and not new.get("gameOver"):
            reward += 0.035

        if score_delta > 0.0 and self._last_reward_event == "progress" and not self._uses_split_pickup_reward():
            self._set_reward_event("score_delta")

        if new.get("completed"):
            reward += 700.0
            self._set_reward_event("completed")
        elif game_over_started:
            reward -= 80.0
            self._set_reward_event("death")

        if not new.get("inMinigame"):
            reward -= 2.0
            if self._last_reward_event == "progress":
                self._set_reward_event("not_in_minigame")

        gap_reward = self._gap_landing_reward(
            old,
            new,
            confirm_steps=self.gap_landing_confirm_steps,
            base_reward=self.gap_landing_base_reward,
            width_reward_coef=self.gap_landing_width_coef,
        )
        reward += gap_reward

        return float(reward)

    def _shaped_v5_reward(self, old: dict[str, Any], new: dict[str, Any], action: int = 0) -> float:
        """Balanced event reward for the compact v8 agents.

        Unlike v4, death is charged exactly once, forward progress uses a
        monotonic max-x potential, and coin/fruit rewards require an actual entity
        disappearance event instead of any score increase.
        """
        old_level = _num(old.get("levelsBeat"))
        new_level = _num(new.get("levelsBeat"))
        level_delta = max(new_level - old_level, 0.0)
        score_delta = _num(new.get("score")) - _num(old.get("score"))
        new_x = self._player_x(new)

        progress_delta = max(new_x - self._reward_progress_x, 0.0)
        self._reward_progress_x = max(self._reward_progress_x, new_x)
        progress_reward = self.progress_reward_coef * progress_delta
        reward = progress_reward
        self._last_reward_components["progress"] = progress_reward
        self._increment_stat("progress_reward_total", progress_reward)

        pickup_reward = self._pickup_fixed_reward(old, new, score_delta)
        reward += pickup_reward
        self._last_reward_components["pickup"] = pickup_reward

        level_reward = 0.0
        if level_delta > 0.0:
            level_reward = self.level_complete_reward * level_delta
            reward += level_reward
            self._reward_progress_x = new_x
            self._active_gap_attempt = None
            self._set_reward_event("level_complete")
        self._last_reward_components["level"] = level_reward

        game_over_started = bool(new.get("gameOver") and not old.get("gameOver"))
        completion_reward = 0.0
        death_reward = 0.0
        if new.get("completed"):
            completion_reward = self.game_complete_reward
            reward += completion_reward
            self._set_reward_event("completed")
        elif game_over_started:
            death_reward = -self.death_penalty
            reward += death_reward
            self._increment_stat("death_penalty_total", -self.death_penalty)
            self._set_reward_event("death")
        self._last_reward_components["completion"] = completion_reward
        self._last_reward_components["death"] = death_reward

        gap_reward = 0.0
        if level_delta <= 0.0:
            gap_reward = self._gap_landing_reward(
                old,
                new,
                confirm_steps=self.gap_landing_confirm_steps,
                base_reward=self.gap_landing_base_reward,
                width_reward_coef=self.gap_landing_width_coef,
                death_reward=0.0,
            )
        reward += gap_reward
        self._last_reward_components["gap"] = gap_reward
        if gap_reward > 0.0:
            self._increment_stat("gap_landing_reward_total", gap_reward)

        return float(reward)

    def _shaped_v6_reward(self, old: dict[str, Any], new: dict[str, Any], action: int = 0) -> float:
        """v5 plus conservative anti-spam and distance-aware failure shaping.

        The jump-start cost is independent of gap geometry, so the policy still
        decides when jumping is worth its tiny energy cost.  Gap miss shaping keeps
        the v5 base death penalty and only adds a bounded quadratic penalty for
        dying far from the complete landing-track interval.
        """
        jump_started = self._jump_started(old, action)
        miss_distance, miss_ratio, gap_miss_death = self._gap_miss_metrics(old, new)

        reward = self._shaped_v5_reward(old, new, action)

        jump_start_reward = 0.0
        if jump_started:
            jump_start_reward = -self.jump_start_penalty
            reward += jump_start_reward
            self._record_jump_start_diagnostics(old)
            self._increment_stat("jump_start_penalty_total", jump_start_reward)
        self._last_reward_components["jump_start"] = jump_start_reward

        gap_miss_reward = 0.0
        if gap_miss_death:
            gap_miss_reward = -self.gap_miss_penalty_coef * (miss_ratio**2)
            reward += gap_miss_reward
            self._increment_stat("gap_miss_deaths")
            self._increment_stat("gap_miss_distance_total", miss_distance)
            self._increment_stat("gap_miss_ratio_total", miss_ratio)
            self._increment_stat("gap_miss_penalty_total", gap_miss_reward)
        self._last_reward_components["gap_miss"] = gap_miss_reward
        self._last_reward_components["gap_miss_distance"] = miss_distance
        self._last_reward_components["gap_miss_ratio"] = miss_ratio

        return float(reward)

    def _shaped_v7_reward(self, old: dict[str, Any], new: dict[str, Any], action: int = 0) -> float:
        """v6 plus a small cost for unnecessarily long airborne jump holds.

        v6 charges only the release-to-hold edge. Once a jump has started, holding
        forever has no extra cost. v7 leaves the first few hold decisions free,
        then charges every additional airborne hold step. Survival and landing
        rewards can still justify a genuinely necessary long jump.
        """
        reward = self._shaped_v6_reward(old, new, action)
        old_player = old.get("player") or {}
        holds_jump = self._action_holds_jump(action)
        continued_airborne_hold = bool(not old_player.get("grounded") and holds_jump)
        hold_steps = int(self._temporal_state.get("jump_held_steps", 0.0)) + (1 if holds_jump else 0)

        if holds_jump:
            self._episode_stats["max_jump_hold_steps"] = max(
                self._episode_stats.get("max_jump_hold_steps", 0.0),
                float(hold_steps),
            )

        airborne_hold_reward = 0.0
        if continued_airborne_hold and hold_steps > self.airborne_hold_free_steps:
            airborne_hold_reward = -self.airborne_hold_penalty
            reward += airborne_hold_reward
            self._increment_stat("airborne_hold_penalty_steps")
            self._increment_stat("airborne_hold_penalty_total", airborne_hold_reward)

        self._last_reward_components["airborne_hold"] = airborne_hold_reward
        self._last_reward_components["jump_hold_steps"] = float(hold_steps)
        return float(reward)

    def _shaped_v8_reward(self, old: dict[str, Any], new: dict[str, Any], action: int = 0) -> float:
        """v7 plus outcome-conditioned tip technique and anti-bunny-hop shaping.

        The edge technique bonus is deliberately *not* paid on the jump action.
        A near-tip takeoff is remembered, combined with landing depth, and paid
        only after the same gap landing is confirmed.  Small auxiliary terms make
        staying grounded on safe rail preferable to permanent airborne spam.
        """
        old_player = old.get("player") or {}
        new_player = new.get("player") or {}
        old_grounded = bool(old_player.get("grounded"))
        new_grounded = bool(new_player.get("grounded"))
        jump_started = self._jump_started(old, action)
        progress_delta = max(self._player_x(new) - self._reward_progress_x, 0.0)
        semantic = self._semantic_snapshot(old)
        obstacle_relevant = bool(
            semantic["next_obstacle_present"]
            and -80.0 <= semantic["next_obstacle_dx"] <= 120.0
        )

        takeoff_quality: float | None = None
        if jump_started:
            takeoff_quality = self._record_gap_takeoff(old)

        reward = self._shaped_v7_reward(old, new, action)

        grounded_progress_reward = 0.0
        if old_grounded and new_grounded and not new.get("gameOver"):
            grounded_progress_reward = self.grounded_progress_bonus_coef * progress_delta
            reward += grounded_progress_reward
            self._increment_stat("grounded_progress_bonus_total", grounded_progress_reward)
        self._last_reward_components["grounded_progress"] = grounded_progress_reward

        unnecessary_jump_reward = 0.0
        useful_gap_takeoff = takeoff_quality is not None and takeoff_quality > 0.0
        if jump_started and not useful_gap_takeoff and not obstacle_relevant:
            unnecessary_jump_reward = -self.unnecessary_jump_penalty
            reward += unnecessary_jump_reward
            self._increment_stat("unnecessary_jump_events")
            self._increment_stat("unnecessary_jump_penalty_total", unnecessary_jump_reward)
        self._last_reward_components["unnecessary_jump"] = unnecessary_jump_reward

        active_gap = self._active_gap_attempt
        useful_gap_airtime = bool(
            active_gap is not None
            and active_gap.get("takeoff_recorded")
            and _num(active_gap.get("takeoff_tip_quality")) > 0.0
        )
        continued_airborne = not old_grounded and not new_grounded
        non_gap_airborne_reward = 0.0
        if continued_airborne and not useful_gap_airtime and not obstacle_relevant:
            non_gap_airborne_reward = -self.non_gap_airborne_penalty
            reward += non_gap_airborne_reward
            self._increment_stat("non_gap_airborne_steps")
            self._increment_stat("non_gap_airborne_penalty_total", non_gap_airborne_reward)
        self._last_reward_components["non_gap_airborne"] = non_gap_airborne_reward

        return float(reward)

    def _shaped_v9_reward(self, old: dict[str, Any], new: dict[str, Any], action: int = 0) -> float:
        """Correct v8's no-jump collapse with geometry-aware, sparse guidance.

        v8 used one fixed 12 px takeoff target and rewarded grounded progress even
        inside the imminent-gap zone.  v9 estimates takeoff timing from gap width
        and landing height, removes the out-of-zone jump penalty, stops grounded
        bonus near hazards, and applies only a small inaction cost when a jump is
        becoming necessary.
        """
        old_player = old.get("player") or {}
        new_player = new.get("player") or {}
        old_grounded = bool(old_player.get("grounded"))
        new_grounded = bool(new_player.get("grounded"))
        jump_started = self._jump_started(old, action)
        progress_delta = max(self._player_x(new) - self._reward_progress_x, 0.0)
        semantic = self._semantic_snapshot(old)
        obstacle_relevant = bool(
            semantic["next_obstacle_present"]
            and -80.0 <= semantic["next_obstacle_dx"] <= 120.0
        )

        if jump_started:
            self._record_gap_takeoff(old)

        reward = self._shaped_v7_reward(old, new, action)
        active_gap = self._active_gap_attempt
        gap_dx_center = (
            _num(active_gap.get("start_x")) - self._player_x(old)
            if active_gap is not None
            else float("inf")
        )
        imminent_gap = bool(
            active_gap is not None
            and 0.0 <= gap_dx_center <= self._gap_inaction_threshold(active_gap)
        )

        grounded_progress_reward = 0.0
        safe_ground_transition = bool(
            old_grounded
            and new_grounded
            and not imminent_gap
            and not obstacle_relevant
            and not new.get("gameOver")
        )
        if safe_ground_transition:
            grounded_progress_reward = self.grounded_progress_bonus_coef * progress_delta
            reward += grounded_progress_reward
            self._increment_stat("grounded_progress_bonus_total", grounded_progress_reward)
        self._last_reward_components["grounded_progress"] = grounded_progress_reward

        gap_inaction_reward = 0.0
        jump_ready = bool(old_player.get("jumpReady", old_grounded and not old.get("jumpHeld")))
        if imminent_gap and old_grounded and jump_ready and not self._action_holds_jump(action):
            gap_inaction_reward = -self.gap_inaction_penalty
            reward += gap_inaction_reward
            self._increment_stat("gap_inaction_steps")
            self._increment_stat("gap_inaction_penalty_total", gap_inaction_reward)
        self._last_reward_components["gap_inaction"] = gap_inaction_reward

        useful_gap_airtime = bool(
            active_gap is not None
            and active_gap.get("takeoff_recorded")
        )
        continued_airborne = not old_grounded and not new_grounded
        non_gap_airborne_reward = 0.0
        if continued_airborne and not useful_gap_airtime and not obstacle_relevant:
            non_gap_airborne_reward = -self.non_gap_airborne_penalty
            reward += non_gap_airborne_reward
            self._increment_stat("non_gap_airborne_steps")
            self._increment_stat("non_gap_airborne_penalty_total", non_gap_airborne_reward)
        self._last_reward_components["non_gap_airborne"] = non_gap_airborne_reward
        self._last_reward_components["unnecessary_jump"] = 0.0

        return float(reward)
