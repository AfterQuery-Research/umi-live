from __future__ import annotations

import json

import numpy as np
from scipy.spatial.transform import Rotation

from dataset_filtering.kinematics import YAMKinematics
from dataset_filtering.models import PoseEpisode
from dataset_filtering.reachability import trim_pose_episode_end
from dataset_filtering.retiming import time_parameterize, trajectory_dynamics
from dataset_filtering.trajectory_optimization import (
    TrajectoryOptimizationConfig,
    _collect_active_collision_constraints,
    _symbolic_fk,
    apply_failed_tail_freeze,
    interarm_surface_clearance_profile_m,
    optimize_pose_episode,
)


def test_symbolic_fk_matches_numeric_yam_model() -> None:
    q = np.asarray([0.2, 1.0, 0.8, 0.15, -0.2, 0.3])
    numeric = YAMKinematics().forward(q)
    position, rotation, points = _symbolic_fk()(q)

    np.testing.assert_allclose(np.asarray(position).reshape(3), numeric.position, atol=1e-9)
    np.testing.assert_allclose(np.asarray(rotation), numeric.rotation, atol=1e-9)
    np.testing.assert_allclose(np.asarray(points).T, numeric.points, atol=1e-9)


def test_collision_active_set_uses_validation_resolution_samples() -> None:
    frames = 4
    links = np.zeros((frames, 2, 8, 3), dtype=np.float64)
    links[:, :, :, 0] = np.arange(8)[None, None, :] * 0.1
    links[:, 1, :, 1] = 0.07

    active = _collect_active_collision_constraints(
        links,
        TrajectoryOptimizationConfig(),
    )

    assert len(active.interarm) == frames
    assert active.frames == set(range(frames))
    assert active.minimum_interarm_surface_m < 0.0
    assert not active.self_collision

    links[:, 1, :, 1] = 0.2
    separated = _collect_active_collision_constraints(
        links,
        TrajectoryOptimizationConfig(),
    )
    assert not separated.interarm
    assert separated.minimum_interarm_surface_m > 0.005

    clearance = interarm_surface_clearance_profile_m(links)
    assert clearance.shape == (frames,)
    np.testing.assert_allclose(clearance, separated.minimum_interarm_surface_m)


def test_uniform_retiming_enforces_velocity_and_acceleration() -> None:
    timestamps = np.arange(61) / 30
    phase = np.linspace(0, np.pi, len(timestamps))
    q = np.column_stack([0.8 * np.sin(phase + joint / 8) for joint in range(12)])
    velocity_limits = np.full(12, 0.3)
    acceleration_limits = np.full(12, 0.5)

    timing = time_parameterize(
        q,
        timestamps,
        velocity_limits,
        acceleration_limits,
        prefer_toppra=False,
    )
    dynamics = trajectory_dynamics(q, timing.timestamps)

    assert timing.method == "uniform_time_scale_fallback"
    assert np.max(np.abs(dynamics["velocity_rad_s"]) / velocity_limits) <= 1.001
    assert np.max(np.abs(dynamics["acceleration_rad_s2"]) / acceleration_limits) <= 1.001


def test_contact_anchored_optimizer_is_continuous_and_accurate() -> None:
    frames = 61
    timestamps = np.arange(frames) / 30
    positions = np.zeros((frames, 2, 3))
    positions[:, :, 2] = 0.01 * np.sin(np.linspace(0, np.pi, frames))[:, None]
    quaternions = np.tile(Rotation.identity().as_quat(), (frames, 2, 1))
    grippers = np.full((frames, 2), 80.0)
    grippers[30:, 0] = 30.0
    grippers[35:, 1] = 25.0
    episode = PoseEpisode(
        998,
        30.0,
        timestamps,
        positions,
        quaternions,
        grippers,
        "synthetic",
        "local",
    )
    config = TrajectoryOptimizationConfig(
        control_interval_s=0.15,
        objective_stride=3,
        collision_stride=12,
        contact_multistart_seeds=4,
        solver_max_iterations=150,
        prefer_toppra=False,
    )

    result = optimize_pose_episode(episode, config=config)
    maximum_joint_step = float(np.max(np.abs(np.diff(result.q, axis=0))))

    assert result.metadata["solver"]["success"]
    assert result.metadata["continuously_replayable"]
    assert result.metadata["validation"]["contact_position_error_max_mm"] <= 1.0
    assert result.metadata["validation"]["contact_orientation_error_max_deg"] <= 2.0
    assert result.metadata["validation"]["passes_velocity_limits"]
    assert result.metadata["validation"]["passes_acceleration_limits"]
    assert result.metadata["validation"]["passes_interarm_clearance"]
    assert result.metadata["validation"]["passes_self_collision_clearance"]
    assert result.metadata["validation"]["passes_tracking_accuracy"]
    assert result.metadata["ik_trajectory_solvable"]
    assert result.metadata["camera_alignment"]["status"] == ("uncalibrated_proxy_do_not_filter")
    np.testing.assert_allclose(result.video_alignment_weight, 1.0)
    json.dumps(result.metadata)
    assert np.max(np.diff(result.source_timestamps) / np.diff(result.robot_timestamps)) <= 1.001
    assert maximum_joint_step < np.deg2rad(3.0)


def test_failed_tail_freeze_is_conditional_and_independent_per_arm() -> None:
    frames = 40
    fps = 10.0
    positions = np.zeros((frames, 2, 3))
    positions[:, 0, 0] = np.arange(frames)
    positions[:, 1, 1] = 2 * np.arange(frames)
    quaternions = Rotation.from_rotvec(
        np.column_stack((np.zeros(frames * 2), np.zeros(frames * 2), np.arange(frames * 2) / 100))
    ).as_quat().reshape(frames, 2, 4)
    grippers = np.column_stack((np.arange(frames), -np.arange(frames)))
    episode = PoseEpisode(
        997,
        fps,
        np.arange(frames) / fps,
        positions,
        quaternions,
        grippers,
        "synthetic",
        "local",
    )
    frame_valid = np.ones((frames, 2), dtype=bool)
    frame_valid[22, 0] = False
    processed, metadata = apply_failed_tail_freeze(
        episode,
        (np.asarray([10]), np.asarray([18])),
        frame_valid,
        grace_period_s=0.5,
    )

    assert metadata["applied_any"]
    assert metadata["arms"]["umi1_left"]["cutoff_frame"] == 15
    assert metadata["arms"]["umi1_left"]["applied"]
    assert not metadata["arms"]["umi2_right"]["applied"]
    np.testing.assert_array_equal(
        processed.positions[15:, 0],
        np.repeat(episode.positions[15, 0][None, :], frames - 15, axis=0),
    )
    np.testing.assert_array_equal(
        processed.quaternions[15:, 0],
        np.repeat(episode.quaternions[15, 0][None, :], frames - 15, axis=0),
    )
    np.testing.assert_array_equal(processed.positions[:, 1], episode.positions[:, 1])
    np.testing.assert_array_equal(processed.quaternions[:, 1], episode.quaternions[:, 1])
    np.testing.assert_array_equal(processed.grippers, episode.grippers)
    np.testing.assert_array_equal(episode.positions, positions)


def test_episode_without_last_event_is_not_tail_frozen() -> None:
    frames = 20
    episode = PoseEpisode(
        996,
        30.0,
        np.arange(frames) / 30,
        np.zeros((frames, 2, 3)),
        np.tile(Rotation.identity().as_quat(), (frames, 2, 1)),
        None,
        "synthetic",
        "local",
    )
    frame_valid = np.zeros((frames, 2), dtype=bool)

    processed, metadata = apply_failed_tail_freeze(
        episode,
        (np.zeros(0, dtype=int), np.zeros(0, dtype=int)),
        frame_valid,
    )

    assert not metadata["applied_any"]
    assert metadata["arms"]["umi1_left"]["cutoff_frame"] is None
    assert metadata["arms"]["umi2_right"]["cutoff_frame"] is None
    np.testing.assert_array_equal(processed.positions, episode.positions)


def test_large_non_event_tracking_error_is_a_hard_replayability_failure() -> None:
    frames = 31
    timestamps = np.arange(frames) / 30
    positions = np.zeros((frames, 2, 3))
    positions[20:, :, 2] = 1.5
    episode = PoseEpisode(
        995,
        30.0,
        timestamps,
        positions,
        np.tile(Rotation.identity().as_quat(), (frames, 2, 1)),
        None,
        "synthetic",
        "local",
    )
    config = TrajectoryOptimizationConfig(
        control_interval_s=0.15,
        objective_stride=3,
        collision_stride=12,
        contact_multistart_seeds=2,
        solver_max_iterations=80,
        prefer_toppra=False,
    )

    result = optimize_pose_episode(episode, config=config)

    assert not result.metadata["validation"]["passes_tracking_accuracy"]
    assert not result.metadata["ik_trajectory_solvable"]
    assert not result.metadata["continuously_replayable"]


def test_trim_pose_episode_end_is_read_only() -> None:
    frames = 61
    episode = PoseEpisode(
        999,
        30.0,
        np.arange(frames) / 30,
        np.zeros((frames, 2, 3)),
        np.tile(Rotation.identity().as_quat(), (frames, 2, 1)),
        np.zeros((frames, 2)),
        "synthetic",
        "local",
    )
    original_timestamps = episode.timestamps.copy()
    original_positions = episode.positions.copy()

    trimmed = trim_pose_episode_end(episode, 1.0)

    assert len(trimmed.timestamps) == 31
    np.testing.assert_array_equal(episode.timestamps, original_timestamps)
    np.testing.assert_array_equal(episode.positions, original_positions)
    assert not np.shares_memory(trimmed.positions, episode.positions)
