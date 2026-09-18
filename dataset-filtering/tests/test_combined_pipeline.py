from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from dataset_filtering.combined_pipeline import (
    joint_observation_state,
    normalize_gripper_widths,
    verify_zeroing_invariance,
    zero_pose_origins,
)
from dataset_filtering.models import PoseEpisode


def _episode() -> PoseEpisode:
    frames = 31
    timestamps = np.arange(frames) / 30
    positions = np.zeros((frames, 2, 3), dtype=np.float64)
    positions[:, 0] = np.column_stack(
        (0.4 + 0.01 * timestamps, -0.2 + 0.02 * timestamps, 1.2 + 0.03 * timestamps)
    )
    positions[:, 1] = np.column_stack(
        (-0.7 + 0.02 * timestamps, 0.8 - 0.01 * timestamps, 2.1 + 0.01 * timestamps)
    )
    rotations = np.empty((frames, 2, 4), dtype=np.float64)
    rotations[:, 0] = Rotation.from_euler(
        "xyz", np.column_stack((0.2 + 0.01 * timestamps, -0.3 + 0.02 * timestamps, timestamps))
    ).as_quat()
    rotations[:, 1] = Rotation.from_euler(
        "xyz", np.column_stack((-0.4 + 0.02 * timestamps, 0.1 - 0.01 * timestamps, -timestamps))
    ).as_quat()
    grippers = np.column_stack((80 + 20 * timestamps, 100 - 10 * timestamps))
    return PoseEpisode(
        episode_index=7,
        fps=30.0,
        timestamps=timestamps,
        positions=positions,
        quaternions=rotations,
        grippers=grippers,
        task="test",
        source="test",
    )


def test_per_arm_zeroing_preserves_yam_targets_before_and_after_smoothing() -> None:
    episode = _episode()
    zeroed = zero_pose_origins(episode)
    np.testing.assert_allclose(zeroed.positions[0], 0.0, atol=0.0)
    np.testing.assert_allclose(zeroed.quaternions[0, :, :3], 0.0, atol=0.0)
    np.testing.assert_allclose(zeroed.quaternions[0, :, 3], 1.0, atol=0.0)

    invariance = verify_zeroing_invariance(episode)
    assert invariance["raw_position_delta_max_m"] < 1e-12
    assert invariance["raw_rotation_delta_max_rad"] < 1e-12
    assert invariance["smoothed_position_delta_max_m"] < 1e-12
    assert invariance["smoothed_rotation_delta_max_rad"] < 1e-12


def test_joint_observation_matches_yam_14d_order_and_normalizes_grippers() -> None:
    q = np.arange(24, dtype=np.float64).reshape(2, 2, 6) / 10
    widths = np.asarray([[40.376321, 40.128507], [122.999636, 122.685643]])
    grippers = normalize_gripper_widths(widths)
    state = joint_observation_state(q, grippers)

    assert state.shape == (2, 14)
    np.testing.assert_allclose(state[0], [*q[0, 0], 0.0, *q[0, 1], 0.0])
    np.testing.assert_allclose(state[1], [*q[1, 0], 1.0, *q[1, 1], 1.0])


def test_normalized_grippers_are_not_normalized_a_second_time() -> None:
    source = np.asarray([[0.0, 0.25], [0.75, 1.0]], dtype=np.float64)
    np.testing.assert_allclose(normalize_gripper_widths(source), source)
