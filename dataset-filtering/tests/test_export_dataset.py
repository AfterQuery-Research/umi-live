from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from dataset_filtering.export_dataset import (
    _aggregate_stats,
    _feature_stats,
    optimized_world_poses_to_umi_state,
)
from dataset_filtering.models import PoseEpisode
from dataset_filtering.retarget import map_umi_targets_to_yam_world


def test_optimized_world_pose_inverse_preserves_umi_schema() -> None:
    positions = np.asarray(
        [
            [[0.1, -0.2, 0.3], [-0.1, 0.05, 0.2]],
            [[0.11, -0.18, 0.33], [-0.08, 0.01, 0.25]],
            [[0.13, -0.16, 0.37], [-0.05, -0.02, 0.29]],
        ]
    )
    rotvecs = np.asarray(
        [
            [[0.1, -0.2, 0.05], [-0.1, 0.03, 0.12]],
            [[0.14, -0.18, 0.02], [-0.08, 0.06, 0.15]],
            [[0.18, -0.15, -0.01], [-0.04, 0.09, 0.18]],
        ]
    )
    quaternions = Rotation.from_rotvec(rotvecs.reshape(-1, 3)).as_quat().reshape(3, 2, 4)
    episode = PoseEpisode(
        episode_index=0,
        fps=30.0,
        timestamps=np.arange(3) / 30,
        positions=positions,
        quaternions=quaternions,
        grippers=np.zeros((3, 2)),
        task="test",
        source="test",
    )
    world_positions, world_quaternions = map_umi_targets_to_yam_world(episode)
    state = optimized_world_poses_to_umi_state(
        world_positions, world_quaternions, positions, quaternions
    )
    expected = np.column_stack((positions[:, 0], rotvecs[:, 0], positions[:, 1], rotvecs[:, 1]))
    np.testing.assert_allclose(state, expected, atol=2e-7)


def test_stats_aggregation_matches_concatenated_mean_and_std() -> None:
    left = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    right = np.asarray([[5.0, 8.0], [7.0, 10.0], [9.0, 12.0]])
    aggregate = _aggregate_stats(
        [
            {"observation.state": _feature_stats(left)},
            {"observation.state": _feature_stats(right)},
        ]
    )["observation.state"]
    combined = np.concatenate((left, right))
    np.testing.assert_allclose(aggregate["mean"], np.mean(combined, axis=0))
    np.testing.assert_allclose(aggregate["std"], np.std(combined, axis=0))
    assert aggregate["count"].tolist() == [5]
