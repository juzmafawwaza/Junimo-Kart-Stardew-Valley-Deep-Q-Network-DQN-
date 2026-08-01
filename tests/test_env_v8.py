from __future__ import annotations

import unittest

import numpy as np

from junimo_rl.env import (
    COMPACT_FEATURES,
    GapGeometryTracker,
    JunimoKartEnv,
    compact_feature_vector,
    semantic_feature_snapshot,
    snapshot_to_vector,
)


def track(x: float, y: float, *, player_x: float = 100.0) -> dict:
    return {
        "x": x,
        "y": y,
        "dx": x - player_x,
        "bounds": {},
        "type": "Straight",
        "typeId": 1,
        "hasObstacle": False,
    }


def snapshot(
    *,
    x: float = 100.0,
    y: float = 140.0,
    grounded: bool = True,
    game_over: bool = False,
    completed: bool = False,
    score: int = 0,
    version: int = 1,
    entities: list[dict] | None = None,
    tracks: list[dict] | None = None,
) -> dict:
    return {
        "inMinigame": True,
        "version": version,
        "score": score,
        "livesLeft": 3 if not game_over else 2,
        "levelsBeat": 0,
        "gameMode": 2,
        "currentTheme": 0,
        "gameState": 1,
        "gameOver": game_over,
        "reachedFinish": completed,
        "completed": completed,
        "jumpHeld": False,
        "distanceToTravel": 150,
        "player": {
            "position": {"x": x, "y": y},
            "velocity": {"x": 95.0, "y": 0.0},
            "bounds": {"x": x - 8, "y": y - 8, "width": 16, "height": 16},
            "grounded": grounded,
            "jumping": not grounded,
            "jumpReady": grounded,
            "currentTrackTypeId": 1,
        },
        "tracksAhead": tracks or [],
        "entitiesAhead": entities or [],
    }


class CompactObservationTests(unittest.TestCase):
    def test_compact_observation_is_small_and_bounded(self) -> None:
        state = snapshot(x=622.0, y=161.0)
        temporal = {
            "jump_held_steps": 100.0,
            "airborne_steps": 100.0,
            "grounded_steps": 100.0,
            "last_action_holds_jump": 1.0,
        }
        observation = compact_feature_vector(state, temporal)
        self.assertEqual(observation.shape, (COMPACT_FEATURES,))
        self.assertTrue(np.all(observation >= -1.0))
        self.assertTrue(np.all(observation <= 1.0))

    def test_bridge_version_does_not_change_legacy_observation(self) -> None:
        first = snapshot_to_vector(snapshot(version=1))
        later = snapshot_to_vector(snapshot(version=999_999))
        np.testing.assert_array_equal(first, later)

    def test_diagnostic_progress_is_bounded(self) -> None:
        semantic = semantic_feature_snapshot(snapshot(x=20_000.0))
        self.assertEqual(semantic["progress_fraction"], 1.0)


class AnchoredGapDetectionTests(unittest.TestCase):
    def _multi_level_tracks(self, *, player_x: float = 100.0) -> list[dict]:
        return [
            *[track(x, 160.0, player_x=player_x) for x in (96, 112, 128, 144, 160)],
            # An unrelated high platform appears first by x.  The legacy adjacent
            # scan pairs its end with the real landing and reports the wrong gap.
            *[track(x, 0.0, player_x=player_x) for x in (176, 192)],
            *[track(x, 160.0, player_x=player_x) for x in (240, 256, 272)],
        ]

    def test_anchored_detector_ignores_unreachable_upper_branch(self) -> None:
        state = snapshot(x=100.0, y=160.0, tracks=self._multi_level_tracks())

        legacy = semantic_feature_snapshot(state, "legacy")
        anchored = semantic_feature_snapshot(state, "anchored")

        self.assertEqual(legacy["next_gap_start_dx"], 92.0)
        self.assertEqual(legacy["next_gap_width"], 48.0)
        self.assertEqual(anchored["next_gap_start_dx"], 60.0)
        self.assertEqual(anchored["next_gap_width"], 80.0)
        self.assertEqual(anchored["landing_end_dx"], 172.0)
        self.assertEqual(anchored["landing_width"], 32.0)

    def test_connected_run_end_replaces_single_support_piece_end(self) -> None:
        state = snapshot(x=100.0, y=160.0, tracks=self._multi_level_tracks())
        anchored = semantic_feature_snapshot(state, "anchored")
        self.assertEqual(anchored["current_track_end_dx"], 60.0)

    def test_gap_pair_persists_in_absolute_coordinates_while_airborne(self) -> None:
        tracker = GapGeometryTracker("anchored")
        grounded = snapshot(x=100.0, y=160.0, tracks=self._multi_level_tracks())
        tracker.update(grounded)

        airborne = snapshot(x=180.0, y=100.0, grounded=False, tracks=[])
        features = tracker.update(airborne)

        self.assertEqual(features["next_gap_present"], 1.0)
        self.assertEqual(features["next_gap_start_dx"], -20.0)
        self.assertEqual(features["next_gap_end_dx"], 60.0)
        self.assertEqual(features["landing_end_dx"], 92.0)
        self.assertEqual(features["next_gap_width"], 80.0)

    def test_stale_grounded_edge_frame_does_not_clear_gap_target(self) -> None:
        tracker = GapGeometryTracker("anchored")
        grounded = snapshot(x=100.0, y=160.0, tracks=self._multi_level_tracks())
        tracker.update(grounded)

        unsupported_edge = snapshot(x=180.0, y=160.0, grounded=True, tracks=[])
        features = tracker.update(unsupported_edge)

        self.assertEqual(features["current_track_present"], 0.0)
        self.assertEqual(features["next_gap_present"], 1.0)
        self.assertEqual(features["next_gap_start_dx"], -20.0)

    def test_anchored_detector_does_not_invent_support_while_airborne(self) -> None:
        airborne = snapshot(
            x=180.0,
            y=100.0,
            grounded=False,
            tracks=self._multi_level_tracks(player_x=180.0),
        )
        features = semantic_feature_snapshot(airborne, "anchored")
        self.assertEqual(features["current_track_present"], 0.0)
        self.assertEqual(features["next_gap_present"], 0.0)


class RewardV5Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = JunimoKartEnv(
            start_on_reset=False,
            observation_mode="compact",
            reward_version="shaped_v5",
            death_penalty=5.0,
            gap_landing_base_reward=5.0,
            gap_landing_width_coef=0.015,
            coin_reward_value=0.2,
            fruit_reward_value=2.0,
        )

    def tearDown(self) -> None:
        self.env.close()

    def test_death_is_penalized_once(self) -> None:
        old = snapshot(x=100.0)
        new = snapshot(x=100.0, game_over=True)
        self.env._reward_progress_x = 100.0
        reward = self.env._shaped_v5_reward(old, new)
        self.assertAlmostEqual(reward, -5.0)

    def test_progress_only_rewards_new_max_x(self) -> None:
        self.env._reward_progress_x = 100.0
        first = self.env._shaped_v5_reward(snapshot(x=100.0), snapshot(x=110.0))
        backwards = self.env._shaped_v5_reward(snapshot(x=110.0), snapshot(x=105.0))
        self.assertAlmostEqual(first, 0.1)
        self.assertAlmostEqual(backwards, 0.0)

    def test_pickup_requires_disappearing_entity_id(self) -> None:
        coin = {
            "id": 123,
            "type": "Coin",
            "x": 100.0,
            "y": 140.0,
            "dx": 0.0,
            "visible": True,
            "enabled": True,
            "isPickup": True,
            "bounds": {"x": 94, "y": 134, "width": 12, "height": 12},
        }
        old = snapshot(score=100, entities=[coin])
        same_entity = snapshot(score=110, entities=[coin])
        collected = snapshot(score=110, entities=[])
        self.assertEqual(self.env._pickup_fixed_reward(old, same_entity, 10.0), 0.0)
        self.assertEqual(self.env._pickup_fixed_reward(old, collected, 10.0), 0.2)

    def test_confirmed_gap_landing_gets_width_bonus(self) -> None:
        self.env._active_gap_attempt = {
            "start_x": 100.0,
            "end_x": 200.0,
            "width": 100.0,
            "landing_y": 140.0,
            "landed": False,
            "confirm_steps_remaining": 0,
            "reward_value": 0.0,
        }
        reward = self.env._gap_landing_reward(
            snapshot(x=190.0, grounded=False),
            snapshot(x=210.0, grounded=True),
            confirm_steps=0,
            base_reward=5.0,
            width_reward_coef=0.015,
            death_reward=0.0,
        )
        self.assertAlmostEqual(reward, 6.5)


class RewardV6Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = JunimoKartEnv(
            start_on_reset=False,
            observation_mode="compact",
            reward_version="shaped_v6",
            progress_reward_coef=0.0,
            death_penalty=5.0,
            jump_start_penalty=0.02,
            gap_miss_penalty_coef=2.0,
        )

    def tearDown(self) -> None:
        self.env.close()

    def _set_active_gap(self, *, furthest_x: float) -> None:
        self.env._active_gap_attempt = {
            "start_x": 100.0,
            "end_x": 200.0,
            "landing_end_x": 300.0,
            "width": 100.0,
            "landing_y": 140.0,
            "furthest_x": furthest_x,
            "landed": False,
            "confirm_steps_remaining": 0,
            "reward_value": 0.0,
        }

    def test_real_jump_start_is_charged_once(self) -> None:
        old = snapshot(x=100.0, grounded=True)
        new = snapshot(x=100.0, grounded=False)
        self.env._reward_progress_x = 100.0

        first = self.env._shaped_v6_reward(old, new, action=1)
        self.env._temporal_state["last_action_holds_jump"] = 1.0
        continued_hold = self.env._shaped_v6_reward(new, new, action=1)

        self.assertAlmostEqual(first, -0.02)
        self.assertAlmostEqual(continued_hold, 0.0)
        self.assertEqual(self.env._episode_stats["jump_start_events"], 1.0)

    def test_airborne_repress_is_not_charged(self) -> None:
        airborne = snapshot(x=100.0, grounded=False)
        self.env._reward_progress_x = 100.0
        reward = self.env._shaped_v6_reward(airborne, airborne, action=1)
        self.assertAlmostEqual(reward, 0.0)
        self.assertEqual(self.env._episode_stats["jump_start_events"], 0.0)

    def test_far_gap_miss_adds_quadratic_penalty(self) -> None:
        self._set_active_gap(furthest_x=120.0)
        self.env._reward_progress_x = 120.0
        old = snapshot(x=120.0, grounded=False)
        new = snapshot(x=120.0, grounded=False, game_over=True)

        reward = self.env._shaped_v6_reward(old, new, action=0)

        self.assertAlmostEqual(reward, -6.28)
        self.assertAlmostEqual(self.env._episode_stats["gap_miss_distance_total"], 80.0)
        self.assertAlmostEqual(self.env._episode_stats["gap_miss_ratio_total"], 0.8)
        self.assertAlmostEqual(self.env._episode_stats["gap_miss_penalty_total"], -1.28)

    def test_horizontal_position_inside_landing_interval_has_no_extra_penalty(self) -> None:
        self._set_active_gap(furthest_x=220.0)
        self.env._reward_progress_x = 220.0
        old = snapshot(x=220.0, grounded=False)
        new = snapshot(x=220.0, grounded=False, game_over=True)

        reward = self.env._shaped_v6_reward(old, new, action=0)

        self.assertAlmostEqual(reward, -5.0)
        self.assertEqual(self.env._episode_stats["gap_miss_deaths"], 1.0)
        self.assertEqual(self.env._episode_stats["gap_miss_distance_total"], 0.0)


class RewardV7Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = JunimoKartEnv(
            start_on_reset=False,
            observation_mode="compact",
            reward_version="shaped_v7",
            progress_reward_coef=0.0,
            jump_start_penalty=0.05,
            airborne_hold_free_steps=4,
            airborne_hold_penalty=0.02,
        )

    def tearDown(self) -> None:
        self.env.close()

    def test_first_four_airborne_hold_steps_are_free(self) -> None:
        airborne = snapshot(x=100.0, grounded=False)
        self.env._reward_progress_x = 100.0
        self.env._temporal_state["jump_held_steps"] = 3.0

        reward = self.env._shaped_v7_reward(airborne, airborne, action=1)

        self.assertAlmostEqual(reward, 0.0)
        self.assertEqual(self.env._episode_stats["airborne_hold_penalty_steps"], 0.0)
        self.assertEqual(self.env._episode_stats["max_jump_hold_steps"], 4.0)

    def test_fifth_airborne_hold_step_is_charged(self) -> None:
        airborne = snapshot(x=100.0, grounded=False)
        self.env._reward_progress_x = 100.0
        self.env._temporal_state["jump_held_steps"] = 4.0

        reward = self.env._shaped_v7_reward(airborne, airborne, action=1)

        self.assertAlmostEqual(reward, -0.02)
        self.assertEqual(self.env._episode_stats["airborne_hold_penalty_steps"], 1.0)
        self.assertAlmostEqual(self.env._episode_stats["airborne_hold_penalty_total"], -0.02)
        self.assertEqual(self.env._episode_stats["max_jump_hold_steps"], 5.0)

    def test_release_after_long_hold_has_no_duration_cost(self) -> None:
        airborne = snapshot(x=100.0, grounded=False)
        self.env._reward_progress_x = 100.0
        self.env._temporal_state["jump_held_steps"] = 12.0

        reward = self.env._shaped_v7_reward(airborne, airborne, action=0)

        self.assertAlmostEqual(reward, 0.0)
        self.assertEqual(self.env._episode_stats["airborne_hold_penalty_steps"], 0.0)


class RewardV8Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = JunimoKartEnv(
            start_on_reset=False,
            observation_mode="compact",
            reward_version="shaped_v8",
            progress_reward_coef=0.0,
            death_penalty=5.0,
            gap_landing_confirm_steps=0,
            gap_landing_base_reward=5.0,
            gap_landing_width_coef=0.015,
            jump_start_penalty=0.05,
            airborne_hold_free_steps=4,
            airborne_hold_penalty=0.02,
            grounded_progress_bonus_coef=0.005,
            unnecessary_jump_penalty=0.15,
            non_gap_airborne_penalty=0.01,
            gap_tip_technique_reward=1.5,
            takeoff_tip_target_distance=12.0,
            takeoff_tip_tolerance=48.0,
            landing_tip_target_depth=16.0,
            landing_tip_tolerance=64.0,
        )

    def tearDown(self) -> None:
        self.env.close()

    def _set_active_gap(self) -> None:
        self.env._active_gap_attempt = {
            "start_x": 150.0,
            "end_x": 250.0,
            "landing_end_x": 350.0,
            "width": 100.0,
            "landing_y": 140.0,
            "furthest_x": 100.0,
            "landed": False,
            "confirm_steps_remaining": 0,
            "reward_value": 0.0,
            "takeoff_recorded": False,
            "takeoff_tip_distance": 0.0,
            "takeoff_tip_quality": 0.0,
            "landing_tip_depth": 0.0,
            "landing_tip_quality": 0.0,
            "edge_technique_reward": 0.0,
        }

    def test_grounded_progress_gets_small_efficiency_bonus(self) -> None:
        self.env._reward_progress_x = 100.0
        reward = self.env._shaped_v8_reward(snapshot(x=100.0), snapshot(x=110.0), action=0)

        self.assertAlmostEqual(reward, 0.05)
        self.assertAlmostEqual(self.env._episode_stats["grounded_progress_bonus_total"], 0.05)

    def test_flat_rail_jump_gets_extra_unnecessary_jump_cost(self) -> None:
        self.env._reward_progress_x = 100.0
        old = snapshot(x=100.0, grounded=True)
        new = snapshot(x=100.0, grounded=False)

        reward = self.env._shaped_v8_reward(old, new, action=1)

        self.assertAlmostEqual(reward, -0.20)
        self.assertEqual(self.env._episode_stats["unnecessary_jump_events"], 1.0)

    def test_unrelated_airtime_gets_per_step_cost(self) -> None:
        self.env._reward_progress_x = 100.0
        airborne = snapshot(x=100.0, grounded=False)

        reward = self.env._shaped_v8_reward(airborne, airborne, action=0)

        self.assertAlmostEqual(reward, -0.01)
        self.assertEqual(self.env._episode_stats["non_gap_airborne_steps"], 1.0)

    def test_tip_bonus_is_delayed_until_successful_landing(self) -> None:
        self._set_active_gap()
        self.env._reward_progress_x = 130.0
        takeoff = snapshot(x=130.0, grounded=True)
        airborne = snapshot(x=130.0, grounded=False)

        takeoff_reward = self.env._shaped_v8_reward(takeoff, airborne, action=1)

        self.assertAlmostEqual(takeoff_reward, -0.05)
        self.assertAlmostEqual(self.env._active_gap_attempt["takeoff_tip_quality"], 1.0)
        self.assertEqual(self.env._episode_stats["edge_technique_reward_total"], 0.0)

        self.env._temporal_state["last_action_holds_jump"] = 1.0
        landing_reward = self.env._shaped_v8_reward(
            snapshot(x=245.0, grounded=False),
            snapshot(x=266.0, grounded=True),
            action=0,
        )

        # Gap base 5 + width bonus 1.5 + perfect conjunctive tip bonus 1.5.
        self.assertAlmostEqual(landing_reward, 8.0)
        self.assertEqual(self.env._episode_stats["gap_landings"], 1.0)
        self.assertEqual(self.env._episode_stats["edge_qualified_landings"], 1.0)
        self.assertAlmostEqual(self.env._episode_stats["edge_technique_reward_total"], 1.5)

    def test_early_gap_takeoff_does_not_earn_tip_quality(self) -> None:
        self._set_active_gap()
        self.env._reward_progress_x = 80.0
        early = snapshot(x=80.0, grounded=True)
        airborne = snapshot(x=80.0, grounded=False)

        reward = self.env._shaped_v8_reward(early, airborne, action=1)

        self.assertAlmostEqual(self.env._active_gap_attempt["takeoff_tip_quality"], 0.0)
        self.assertAlmostEqual(reward, -0.20)
        self.assertEqual(self.env._episode_stats["unnecessary_jump_events"], 1.0)

    def test_good_takeoff_that_dies_receives_no_tip_bonus(self) -> None:
        self._set_active_gap()
        self.env._reward_progress_x = 130.0
        self.env._shaped_v8_reward(
            snapshot(x=130.0, grounded=True),
            snapshot(x=130.0, grounded=False),
            action=1,
        )
        self.env._temporal_state["last_action_holds_jump"] = 1.0

        self.env._shaped_v8_reward(
            snapshot(x=180.0, grounded=False),
            snapshot(x=180.0, grounded=False, game_over=True),
            action=0,
        )

        self.assertEqual(self.env._episode_stats["edge_qualified_landings"], 0.0)
        self.assertEqual(self.env._episode_stats["edge_technique_reward_total"], 0.0)


class RewardV9Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = JunimoKartEnv(
            start_on_reset=False,
            observation_mode="compact",
            gap_detection_mode="anchored",
            reward_version="shaped_v9",
            progress_reward_coef=0.0,
            death_penalty=6.0,
            gap_landing_confirm_steps=0,
            gap_landing_base_reward=6.0,
            gap_landing_width_coef=0.02,
            jump_start_penalty=0.03,
            airborne_hold_free_steps=6,
            airborne_hold_penalty=0.01,
            grounded_progress_bonus_coef=0.002,
            non_gap_airborne_penalty=0.005,
            gap_tip_technique_reward=0.5,
            takeoff_target_width_coef=0.65,
            takeoff_target_uphill_coef=0.25,
            takeoff_target_downhill_coef=0.10,
            takeoff_target_min_distance=32.0,
            takeoff_target_max_distance=112.0,
            takeoff_dynamic_tolerance=64.0,
            gap_inaction_margin=24.0,
            gap_inaction_penalty=0.05,
        )

    def tearDown(self) -> None:
        self.env.close()

    @staticmethod
    def _active_gap(*, landing_delta_y: float = 0.0) -> dict:
        return {
            "start_x": 150.0,
            "end_x": 250.0,
            "landing_end_x": 350.0,
            "width": 100.0,
            "landing_y": 140.0,
            "landing_delta_y": landing_delta_y,
            "furthest_x": 100.0,
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

    def test_dynamic_takeoff_target_scales_with_width_and_height(self) -> None:
        level = self.env._dynamic_takeoff_target({"width": 96.0, "landing_delta_y": 0.0})
        uphill = self.env._dynamic_takeoff_target({"width": 96.0, "landing_delta_y": -32.0})
        downhill = self.env._dynamic_takeoff_target({"width": 96.0, "landing_delta_y": 32.0})

        self.assertAlmostEqual(level, 62.4)
        self.assertAlmostEqual(uphill, 70.4)
        self.assertAlmostEqual(downhill, 59.2)

    def test_dynamic_target_is_clamped(self) -> None:
        self.assertEqual(
            self.env._dynamic_takeoff_target({"width": 56.0, "landing_delta_y": 200.0}),
            32.0,
        )
        self.assertEqual(
            self.env._dynamic_takeoff_target({"width": 200.0, "landing_delta_y": -200.0}),
            112.0,
        )

    def test_imminent_gap_release_has_no_ground_bonus_and_small_inaction_cost(self) -> None:
        tracks = [
            *[track(x, 160.0, player_x=100.0) for x in (96, 112, 128, 144, 160)],
            *[track(x, 160.0, player_x=100.0) for x in (240, 256, 272)],
        ]
        old = snapshot(x=100.0, y=160.0, grounded=True, tracks=tracks)
        new_tracks = [
            *[track(x, 160.0, player_x=110.0) for x in (96, 112, 128, 144, 160)],
            *[track(x, 160.0, player_x=110.0) for x in (240, 256, 272)],
        ]
        new = snapshot(x=110.0, y=160.0, grounded=True, tracks=new_tracks)
        self.env._gap_geometry_tracker.update(old)
        self.env._reward_progress_x = 100.0

        reward = self.env._shaped_v9_reward(old, new, action=0)

        self.assertAlmostEqual(reward, -0.05)
        self.assertEqual(self.env._episode_stats["grounded_progress_bonus_total"], 0.0)
        self.assertEqual(self.env._episode_stats["gap_inaction_steps"], 1.0)

    def test_safe_flat_ground_still_gets_small_bonus(self) -> None:
        self.env._reward_progress_x = 100.0
        reward = self.env._shaped_v9_reward(snapshot(x=100.0), snapshot(x=110.0), action=0)

        self.assertAlmostEqual(reward, 0.02)
        self.assertAlmostEqual(self.env._episode_stats["grounded_progress_bonus_total"], 0.02)

    def test_gap_airtime_is_valid_even_when_takeoff_quality_is_zero(self) -> None:
        active_gap = self._active_gap()
        active_gap["takeoff_recorded"] = True
        active_gap["takeoff_tip_quality"] = 0.0
        self.env._active_gap_attempt = active_gap
        self.env._reward_progress_x = 120.0
        airborne = snapshot(x=120.0, grounded=False)

        reward = self.env._shaped_v9_reward(airborne, airborne, action=0)

        self.assertAlmostEqual(reward, 0.0)
        self.assertEqual(self.env._episode_stats["non_gap_airborne_steps"], 0.0)

    def test_dynamic_takeoff_and_landing_pay_small_confirmed_technique_bonus(self) -> None:
        self.env._active_gap_attempt = self._active_gap()
        self.env._reward_progress_x = 77.0

        takeoff_reward = self.env._shaped_v9_reward(
            snapshot(x=77.0, grounded=True),
            snapshot(x=77.0, grounded=False),
            action=1,
        )

        self.assertAlmostEqual(takeoff_reward, -0.03)
        self.assertAlmostEqual(self.env._active_gap_attempt["takeoff_target_distance"], 65.0)
        self.assertAlmostEqual(self.env._active_gap_attempt["takeoff_tip_distance"], 65.0)
        self.assertAlmostEqual(self.env._active_gap_attempt["takeoff_tip_quality"], 1.0)

        self.env._temporal_state["last_action_holds_jump"] = 1.0
        landing_reward = self.env._shaped_v9_reward(
            snapshot(x=245.0, grounded=False),
            snapshot(x=266.0, grounded=True),
            action=0,
        )

        # Gap base 6 + width bonus 2 + smaller perfect technique bonus 0.5.
        self.assertAlmostEqual(landing_reward, 8.5)
        self.assertAlmostEqual(self.env._episode_stats["edge_technique_reward_total"], 0.5)
        self.assertAlmostEqual(self.env._episode_stats["successful_takeoff_tip_distance_total"], 65.0)
        self.assertAlmostEqual(self.env._episode_stats["successful_takeoff_target_distance_total"], 65.0)


if __name__ == "__main__":
    unittest.main()
