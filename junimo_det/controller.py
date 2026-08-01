from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from junimo_rl.env import semantic_feature_snapshot


@dataclass(slots=True)
class RuleConfig:
    action_mode: str = "macro"
    gap_trigger_dx: float = 25.0
    obstacle_trigger_dx: float = 95.0
    small_gap_width: float = 56.0
    medium_gap_width: float = 80.0
    long_gap_width: float = 88.0
    upward_landing_delta_y: float = -24.0


class RuleBasedController:
    def __init__(self, config: RuleConfig | None = None) -> None:
        self.config = config or RuleConfig()
        if self.config.action_mode not in {"binary", "macro"}:
            raise ValueError("action_mode must be 'binary' or 'macro'.")

    def decide(self, snapshot: dict[str, Any]) -> int:
        action, _reason = self.decide_with_reason(snapshot)
        return action

    def decide_with_reason(self, snapshot: dict[str, Any]) -> tuple[int, str]:
        if not snapshot.get("inMinigame") or snapshot.get("gameOver"):
            return 0, "not_gameplay"

        player = snapshot.get("player") or {}
        grounded = bool(player.get("grounded"))
        if not grounded:
            return 0, "airborne_release"

        features = semantic_feature_snapshot(snapshot)
        gap_action = self._gap_action(features)
        if gap_action > 0:
            return gap_action, "gap"

        obstacle_action = self._obstacle_action(features)
        if obstacle_action > 0:
            return obstacle_action, "obstacle"

        return 0, "cruise"

    def _gap_action(self, features: dict[str, float]) -> int:
        if not features["next_gap_present"]:
            return 0

        gap_dx = features["next_gap_start_dx"]
        gap_width = features["next_gap_width"]
        landing_delta_y = features["landing_delta_y"]
        if gap_dx < 0 or gap_dx > self.config.gap_trigger_dx:
            return 0
        if gap_width < self.config.small_gap_width:
            return 0

        if self.config.action_mode == "binary":
            return 1

        if gap_width >= self.config.long_gap_width or landing_delta_y <= self.config.upward_landing_delta_y:
            return 3
        if gap_width >= self.config.medium_gap_width:
            return 2
        return 1

    def _obstacle_action(self, features: dict[str, float]) -> int:
        if not features["next_obstacle_present"]:
            return 0

        obstacle_dx = features["next_obstacle_dx"]
        if obstacle_dx < 0 or obstacle_dx > self.config.obstacle_trigger_dx:
            return 0

        return 1 if self.config.action_mode == "binary" else 2
