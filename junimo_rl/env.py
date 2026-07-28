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
OBSERVATION_SIZE = BASE_FEATURES + MAX_TRACKS * TRACK_FEATURES + MAX_ENTITIES * ENTITY_FEATURES


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


def snapshot_to_vector(snapshot: dict[str, Any]) -> np.ndarray:
    player = snapshot.get("player") or {}
    velocity = player.get("velocity") or {}
    position = player.get("position") or {}

    values: list[float] = [
        _bool(snapshot.get("inMinigame")),
        _num(snapshot.get("version")),
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

    array = np.asarray(values, dtype=np.float32)
    if array.shape != (OBSERVATION_SIZE,):
        raise RuntimeError(f"Observation size mismatch: got {array.shape}, expected {(OBSERVATION_SIZE,)}")
    return array


class JunimoKartEnv(gym.Env[np.ndarray, int]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8765,
        mode: str = "progress",
        frame_skip: int = 4,
        fps: float = 60.0,
        start_on_reset: bool = True,
    ) -> None:
        super().__init__()
        self.client = JunimoKartBridgeClient(host=host, port=port)
        self.mode = mode
        self.frame_skip = frame_skip
        self.fps = fps
        self.start_on_reset = start_on_reset
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBSERVATION_SIZE,),
            dtype=np.float32,
        )
        self._last_snapshot: dict[str, Any] | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
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
        return snapshot_to_vector(snapshot), {"snapshot": snapshot}

    def step(self, action: int):
        old = self._last_snapshot or self.client.state()
        self.client.action(bool(action))
        time.sleep(max(self.frame_skip, 1) / self.fps)
        new = self.client.state()
        self._last_snapshot = new

        reward = self._reward(old, new)
        terminated = bool(new.get("completed", False))
        truncated = bool(new.get("gameOver", False) and not terminated)
        return snapshot_to_vector(new), reward, terminated, truncated, {"snapshot": new}

    def close(self) -> None:
        self.client.close()

    def _ready_for_agent(self, snapshot: dict[str, Any]) -> bool:
        return bool(
            snapshot.get("inMinigame")
            and snapshot.get("gameMode") == 2
            and snapshot.get("gameState") == 1
            and snapshot.get("player")
        )

    def _reward(self, old: dict[str, Any], new: dict[str, Any]) -> float:
        old_player = old.get("player") or {}
        new_player = new.get("player") or {}
        old_pos = old_player.get("position") or {}
        new_pos = new_player.get("position") or {}

        dx = _num(new_pos.get("x")) - _num(old_pos.get("x"))
        score_delta = _num(new.get("score")) - _num(old.get("score"))
        level_delta = _num(new.get("levelsBeat")) - _num(old.get("levelsBeat"))
        life_delta = _num(new.get("livesLeft")) - _num(old.get("livesLeft"))

        reward = 0.001 * dx
        reward += 0.01 * score_delta
        reward += 50.0 * max(level_delta, 0.0)
        reward += 10.0 * life_delta

        if new.get("completed"):
            reward += 250.0
        elif new.get("gameOver") and not old.get("gameOver"):
            reward -= 100.0

        if not new.get("inMinigame"):
            reward -= 1.0

        return float(reward)
