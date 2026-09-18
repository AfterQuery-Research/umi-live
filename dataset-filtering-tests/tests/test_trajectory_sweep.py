import json
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
from dataset_filtering.kinematics import YAMKinematics
from dataset_filtering.models import ContactAnchor

from dataset_filtering_tests.trajectory_sweep import (
    CONFIG_PROFILES,
    _greedy_event_corrected_guess,
    _homotopy_spline_nlp,
    _secondary_from_start_spline_nlp,
    _TrackingAugmentedActiveSet,
    select_experiment_episodes,
)


def test_profiles_preserve_all_acceptance_gates() -> None:
    gate_fields = {
        "contact_position_tolerance_m",
        "contact_orientation_tolerance_rad",
        "tracking_position_tolerance_m",
        "tracking_orientation_tolerance_rad",
        "interarm_collision_margin_m",
        "self_collision_margin_m",
        "table_height_m",
        "table_clearance_m",
        "planning_velocity_scale",
        "planning_acceleration_limit_rad_s2",
    }
    defaults = CONFIG_PROFILES["dense_objective"]
    for profile in CONFIG_PROFILES.values():
        for field in fields(profile):
            if field.name in gate_fields:
                assert getattr(profile, field.name) == getattr(defaults, field.name)


def test_home_profiles_preserve_all_gates() -> None:
    reference = CONFIG_PROFILES["dense_objective"]
    for name, profile in CONFIG_PROFILES.items():
        if "dense_home_" not in name:
            continue
        assert profile.contact_position_tolerance_m == reference.contact_position_tolerance_m
        assert (
            profile.contact_orientation_tolerance_rad
            == reference.contact_orientation_tolerance_rad
        )
        assert profile.tracking_position_tolerance_m == reference.tracking_position_tolerance_m
        assert (
            profile.tracking_orientation_tolerance_rad
            == reference.tracking_orientation_tolerance_rad
        )
        assert profile.interarm_collision_margin_m == reference.interarm_collision_margin_m
        assert profile.self_collision_margin_m == reference.self_collision_margin_m


def test_greedy_initializer_retains_motion_and_matches_event_branch() -> None:
    frames = 11
    motion = np.zeros((frames, 6), dtype=np.float64)
    motion[:, 0] = 0.2 * np.sin(np.linspace(0, 2 * np.pi, frames))
    greedy = SimpleNamespace(
        arms=(SimpleNamespace(playback_q=motion), SimpleNamespace(playback_q=motion.copy()))
    )
    event_q = motion[5].copy()
    event_q[1] = 0.25
    anchor = ContactAnchor(
        arm=0,
        frame=5,
        feasible=True,
        q=event_q,
        position_error_m=0.0,
        orientation_error_rad=0.0,
        candidates=1,
    )

    result = _greedy_event_corrected_guess(greedy, [anchor], frames, YAMKinematics())

    np.testing.assert_allclose(result[5, :6], event_q)
    assert np.ptp(result[:, 0]) > 0.3
    np.testing.assert_allclose(result[:, 6:], motion)


def test_select_experiment_episodes_by_diagnosed_failure(tmp_path) -> None:
    report = {
        "per_episode": [
            {
                "episode_index": 1,
                "profiles": {
                    "dense_objective": {
                        "continuously_replayable": False,
                        "validation": {
                            "passes_tracking_accuracy": False,
                            "passes_interarm_clearance": True,
                            "passes_self_collision_clearance": True,
                        },
                    }
                },
            },
            {
                "episode_index": 2,
                "profiles": {
                    "dense_objective": {
                        "continuously_replayable": False,
                        "validation": {
                            "passes_tracking_accuracy": True,
                            "passes_interarm_clearance": False,
                            "passes_self_collision_clearance": True,
                        },
                    }
                },
            },
        ]
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    assert select_experiment_episodes(
        path, profile="dense_objective", criterion="tracking-fail"
    ) == [1]
    assert select_experiment_episodes(
        path, profile="dense_objective", criterion="tracking-pass-collision-fail"
    ) == [2]


def test_homotopy_repair_stages_margin_without_changing_final_config() -> None:
    seen = {}

    def solver(*args, **kwargs):
        seen["config"] = args[4]
        return np.zeros((1, 12)), {"success": True}

    active_set = SimpleNamespace(
        interarm=((1, 2, 3),),
        self_collision=(),
        minimum_interarm_surface_m=-0.025,
        minimum_self_surface_m=0.010,
    )
    final_config = CONFIG_PROFILES["dense_homotopy_repair"]
    _, metadata = _homotopy_spline_nlp(
        None,
        np.zeros((1, 2, 3)),
        np.zeros((1, 2, 4)),
        [],
        final_config,
        YAMKinematics(),
        (np.eye(4), np.eye(4)),
        np.zeros((1, 12)),
        30.0,
        active_set,
        enable_secondary_objective=True,
        production_solver=solver,
    )

    assert np.isclose(seen["config"].interarm_collision_margin_m, -0.01)
    assert final_config.interarm_collision_margin_m == 0.005
    assert metadata["homotopy_constraint_margins_mm"]["final_required_interarm"] == 5.0


def test_secondary_from_start_unlocks_joint_branch_but_keeps_cartesian_events() -> None:
    seen = {}

    def solver(*args, **kwargs):
        seen.update(kwargs)
        return np.zeros((1, 12)), {"success": True}

    _, metadata = _secondary_from_start_spline_nlp(
        production_solver=solver,
        allow_contact_branch_adjustment=False,
        enable_secondary_objective=False,
    )

    assert seen["allow_contact_branch_adjustment"] is True
    assert seen["enable_secondary_objective"] is True
    assert metadata["secondary_objective_from_initial_solve"] is True


def test_tracking_active_set_contains_every_frame_and_keeps_collisions() -> None:
    active = SimpleNamespace(
        frames={3},
        interarm=((3, 1, 2),),
        self_collision=(),
        minimum_interarm_surface_m=0.0,
        minimum_self_surface_m=1.0,
    )

    augmented = _TrackingAugmentedActiveSet(active, 5)

    assert augmented.frames == {0, 1, 2, 3, 4}
    assert augmented.interarm == active.interarm
