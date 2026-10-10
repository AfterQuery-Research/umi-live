from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from dataset_filtering.hf_dataset import (
    DatasetSchemaError,
    LeRobotSource,
    as_pose_episode,
    classify_schema,
)
from dataset_filtering.models import EpisodeKind


def test_classify_supported_schemas() -> None:
    umi_names = tuple(
        f"umi{arm}_{axis}" for arm in (1, 2) for axis in ("x", "y", "z", "rx", "ry", "rz")
    )
    assert classify_schema({"robot_type": None}, 12, umi_names) is EpisodeKind.UMI_POSE
    calibrated_umi_names = (*umi_names[:6], "umi1_gripper", *umi_names[6:], "umi2_gripper")
    assert (
        classify_schema({"robot_type": None}, 14, calibrated_umi_names)
        is EpisodeKind.UMI_POSE
    )
    joint_names = tuple(f"left_joint_{index}.pos" for index in range(14))
    assert (
        classify_schema({"robot_type": "bi_yam_follower"}, 14, joint_names) is EpisodeKind.YAM_JOINT
    )
    with pytest.raises(DatasetSchemaError):
        classify_schema({"robot_type": "mystery"}, 9, ())


def test_calibrated_umi_state_extracts_interleaved_poses_and_grippers(tmp_path) -> None:
    root = tmp_path / "dataset"
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    pose_names = tuple(
        f"umi{arm}_{axis}" for arm in (1, 2) for axis in ("x", "y", "z", "rx", "ry", "rz")
    )
    names = (*pose_names[:6], "umi1_gripper", *pose_names[6:], "umi2_gripper")
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "total_episodes": 1,
                "features": {"observation.state": {"shape": [14], "names": names}},
            }
        )
    )
    pd.DataFrame(
        {
            "episode_index": [0],
            "tasks": [["fold shirt"]],
            "length": [2],
            "data/chunk_index": [0],
            "data/file_index": [0],
        }
    ).to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    state = np.asarray(
        [
            [0, 1, 2, 0.1, 0.2, 0.3, 0.25, 10, 11, 12, 0.4, 0.5, 0.6, 0.75],
            [3, 4, 5, 0.2, 0.3, 0.4, 0.50, 13, 14, 15, 0.5, 0.6, 0.7, 1.00],
        ],
        dtype=np.float32,
    )
    pd.DataFrame(
        {
            "observation.state": list(state),
            "action": list(np.zeros((2, 12), dtype=np.float32)),
            "timestamp": [0.0, 1 / 30],
            "frame_index": [0, 1],
            "episode_index": [0, 0],
        }
    ).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")

    episode = LeRobotSource(str(root)).load_episode(0)
    pose = as_pose_episode(episode)

    assert episode.kind is EpisodeKind.UMI_POSE
    np.testing.assert_allclose(pose.positions[:, 0], state[:, 0:3])
    np.testing.assert_allclose(pose.positions[:, 1], state[:, 7:10])
    np.testing.assert_allclose(episode.grippers, state[:, [6, 13]])


def test_local_lerobot_episode_is_read_without_mutation(tmp_path) -> None:
    root = tmp_path / "dataset"
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    info = {
        "fps": 30,
        "total_episodes": 1,
        "features": {
            "observation.state": {
                "shape": [12],
                "names": [
                    f"umi{arm}_{axis}"
                    for arm in (1, 2)
                    for axis in ("x", "y", "z", "rx", "ry", "rz")
                ],
            }
        },
    }
    (root / "meta" / "info.json").write_text(json.dumps(info))
    pd.DataFrame(
        {
            "episode_index": [0],
            "tasks": [["test"]],
            "length": [4],
            "data/chunk_index": [0],
            "data/file_index": [0],
        }
    ).to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    states = [np.arange(12, dtype=np.float32) * frame / 100 for frame in range(4)]
    frame = pd.DataFrame(
        {
            "observation.state": states,
            "action": states,
            "timestamp": np.arange(4) / 30,
            "frame_index": np.arange(4),
            "episode_index": np.zeros(4, dtype=int),
            "observation.gripper_width.umi1": [[100.0]] * 4,
            "observation.gripper_width.umi2": [[90.0]] * 4,
        }
    )
    frame.to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    before = sorted(str(path.relative_to(root)) for path in root.rglob("*"))

    episode = LeRobotSource(str(root)).load_episode(0)
    pose = as_pose_episode(episode)

    assert episode.kind is EpisodeKind.UMI_POSE
    assert pose.positions.shape == (4, 2, 3)
    assert episode.grippers.tolist() == [[100.0, 90.0]] * 4
    assert before == sorted(str(path.relative_to(root)) for path in root.rglob("*"))


def test_complete_remote_metadata_cache_does_not_require_network(tmp_path) -> None:
    cache = tmp_path / "cache"
    revision = cache / "owner--dataset" / "main"
    episodes_root = revision / "meta" / "episodes" / "chunk-000"
    episodes_root.mkdir(parents=True)
    (revision / "meta" / "info.json").write_text(
        json.dumps({"total_episodes": 2}),
        encoding="utf-8",
    )
    for index in range(2):
        pd.DataFrame({"episode_index": [index]}).to_parquet(
            episodes_root / f"file-{index:03d}.parquet"
        )

    source = LeRobotSource("owner/dataset", cache_dir=cache)

    assert source._episode_metadata_paths() == [
        "meta/episodes/chunk-000/file-000.parquet",
        "meta/episodes/chunk-000/file-001.parquet",
    ]
