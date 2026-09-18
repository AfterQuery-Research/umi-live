from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from dataset_filtering.export_dataset import optimized_world_poses_to_umi_state
from dataset_filtering.kinematics import YAMKinematics
from dataset_filtering.models import PoseEpisode
from dataset_filtering.retarget import (
    DEFAULT_PAIR_SEPARATION_M,
    UMI_TO_ROBOT_BASIS,
    map_umi_targets_to_yam_world,
    retarget_pose_episode,
    yam_base_transforms,
)


def test_left_right_base_convention() -> None:
    left, right = yam_base_transforms()
    assert left[1, 3] > right[1, 3]
    assert np.isclose(left[1, 3] - right[1, 3], DEFAULT_PAIR_SEPARATION_M)


def test_umi_rotation_axes_map_to_the_same_physical_grasp_axes() -> None:
    angle = np.deg2rad(10.0)
    source_rotvecs = np.asarray(
        [[0.0, 0.0, 0.0], [angle, 0.0, 0.0], [0.0, angle, 0.0], [0.0, 0.0, angle]]
    )
    quaternions = Rotation.from_rotvec(source_rotvecs).as_quat()
    episode = PoseEpisode(
        0,
        30.0,
        np.arange(4) / 30.0,
        np.zeros((4, 2, 3)),
        np.repeat(quaternions[:, None, :], 2, axis=1),
        None,
        "test",
        "synthetic",
    )

    _, targets = map_umi_targets_to_yam_world(episode)
    target_rotations = Rotation.from_quat(targets[:, 0])
    world_deltas = target_rotations * target_rotations[0].inv()
    np.testing.assert_allclose(
        world_deltas.as_rotvec(),
        source_rotvecs @ UMI_TO_ROBOT_BASIS.T,
        atol=1e-10,
    )


def test_umi_to_yam_pose_mapping_round_trips_through_export() -> None:
    positions = np.asarray(
        [
            [[0.1, -0.2, 0.3], [-0.3, 0.2, 0.1]],
            [[0.12, -0.19, 0.34], [-0.32, 0.24, 0.11]],
            [[0.15, -0.25, 0.31], [-0.28, 0.18, 0.16]],
        ]
    )
    rotvecs = np.asarray(
        [
            [[0.02, -0.03, 0.01], [-0.01, 0.04, -0.02]],
            [[0.08, -0.01, 0.03], [-0.04, 0.06, 0.01]],
            [[0.04, -0.09, 0.02], [0.02, 0.01, -0.08]],
        ]
    )
    quaternions = Rotation.from_rotvec(rotvecs.reshape(-1, 3)).as_quat().reshape(3, 2, 4)
    episode = PoseEpisode(
        0,
        30.0,
        np.arange(3) / 30.0,
        positions,
        quaternions,
        None,
        "test",
        "synthetic",
    )

    target_positions, target_quaternions = map_umi_targets_to_yam_world(episode)
    state = optimized_world_poses_to_umi_state(
        target_positions,
        target_quaternions,
        positions,
        quaternions,
    )
    expected = np.column_stack(
        (positions[:, 0], rotvecs[:, 0], positions[:, 1], rotvecs[:, 1])
    )
    np.testing.assert_allclose(state, expected, atol=2e-7)


def test_aligned_shared_mapping_preserves_initial_relative_pose_and_centers_pair() -> None:
    positions = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.08, 0.58, -0.03]],
            [[0.02, -0.01, 0.04], [0.10, 0.56, 0.02]],
        ]
    )
    rotvecs = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.10, -0.16, 0.07]],
            [[0.04, -0.02, 0.03], [0.13, -0.12, 0.01]],
        ]
    )
    quaternions = Rotation.from_rotvec(rotvecs.reshape(-1, 3)).as_quat().reshape(2, 2, 4)
    episode = PoseEpisode(
        0,
        30.0,
        np.arange(2) / 30.0,
        positions,
        quaternions,
        None,
        "test",
        "synthetic",
    )

    target_positions, target_quaternions = map_umi_targets_to_yam_world(
        episode, aligned_shared_frame=True
    )
    kin = YAMKinematics()
    bases = yam_base_transforms()
    home_position = kin.forward(kin.home).position
    home_positions = np.stack(
        [base[:3, :3] @ home_position + base[:3, 3] for base in bases]
    )
    np.testing.assert_allclose(
        np.mean(target_positions[0], axis=0), np.mean(home_positions, axis=0), atol=1e-12
    )
    expected_relative = (positions[0, 1] - positions[0, 0]) @ UMI_TO_ROBOT_BASIS.T
    np.testing.assert_allclose(
        target_positions[0, 1] - target_positions[0, 0], expected_relative, atol=1e-12
    )
    np.testing.assert_allclose(
        np.linalg.norm(target_positions[0, 1] - target_positions[0, 0]),
        np.linalg.norm(positions[0, 1] - positions[0, 0]),
        atol=1e-12,
    )

    source_relative = Rotation.from_quat(quaternions[0, 1]) * Rotation.from_quat(
        quaternions[0, 0]
    ).inv()
    target_relative = Rotation.from_quat(target_quaternions[0, 1]) * Rotation.from_quat(
        target_quaternions[0, 0]
    ).inv()
    expected_matrix = (
        UMI_TO_ROBOT_BASIS @ source_relative.as_matrix() @ UMI_TO_ROBOT_BASIS.T
    )
    np.testing.assert_allclose(target_relative.as_matrix(), expected_matrix, atol=1e-12)


def test_aligned_shared_mapping_round_trips_through_export() -> None:
    positions = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.08, 0.58, -0.03]],
            [[0.02, -0.01, 0.04], [0.10, 0.56, 0.02]],
            [[0.03, -0.04, 0.06], [0.14, 0.51, 0.07]],
        ]
    )
    rotvecs = np.asarray(
        [
            [[0.0, 0.0, 0.0], [0.10, -0.16, 0.07]],
            [[0.04, -0.02, 0.03], [0.13, -0.12, 0.01]],
            [[0.07, -0.06, 0.01], [0.18, -0.08, -0.04]],
        ]
    )
    quaternions = Rotation.from_rotvec(rotvecs.reshape(-1, 3)).as_quat().reshape(3, 2, 4)
    episode = PoseEpisode(
        0,
        30.0,
        np.arange(3) / 30.0,
        positions,
        quaternions,
        None,
        "test",
        "synthetic",
    )

    target_positions, target_quaternions = map_umi_targets_to_yam_world(
        episode, aligned_shared_frame=True
    )
    state = optimized_world_poses_to_umi_state(
        target_positions,
        target_quaternions,
        positions,
        quaternions,
        aligned_shared_frame=True,
    )
    expected = np.column_stack(
        (positions[:, 0], rotvecs[:, 0], positions[:, 1], rotvecs[:, 1])
    )
    np.testing.assert_allclose(state, expected, atol=2e-7)


def test_unreachable_target_holds_last_valid_then_resumes() -> None:
    n = 12
    t = np.arange(n) / 30
    positions = np.zeros((n, 2, 3))
    positions[4:7, :, 2] = 2.0  # UMI +Z maps to robot +X, well beyond YAM reach.
    quaternions = np.tile(Rotation.identity().as_quat(), (n, 2, 1))
    episode = PoseEpisode(0, 30.0, t, positions, quaternions, None, "test", "synthetic")

    result = retarget_pose_episode(episode, max_iterations=35)

    for arm in result.arms:
        assert np.all(arm.reachable[:4])
        assert not np.any(arm.reachable[4:7])
        np.testing.assert_allclose(
            arm.playback_q[4:7], np.repeat(arm.playback_q[3][None, :], 3, axis=0)
        )
        assert np.all(arm.reachable[7:])
