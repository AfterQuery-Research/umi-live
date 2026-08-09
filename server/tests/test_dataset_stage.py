"""Dataset stage: append-ledger idempotence + align/glim trajectory selection
(no lerobot/cv2 needed here)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from umi_flow_core import config as core_config
from umi_flow_server.dataset_stage import already_appended, mark_appended, pose_track_for_umi
from umi_flow_server.trajectory import PoseTrack, write_tum


def test_append_ledger_records_and_detects_episodes(tmp_path: Path):
    dataset_dir = tmp_path / "org" / "umi-dataset"  # repo_id can be nested

    assert not already_appended(dataset_dir, "episode_1")
    mark_appended(dataset_dir, "episode_1")
    assert already_appended(dataset_dir, "episode_1")
    assert not already_appended(dataset_dir, "episode_2")

    mark_appended(dataset_dir, "episode_2")
    assert already_appended(dataset_dir, "episode_1")
    assert already_appended(dataset_dir, "episode_2")

    ledger = (dataset_dir / ".umi_appended_episodes.txt").read_text().split()
    assert ledger == ["episode_1", "episode_2"]


def test_append_ledger_is_substring_safe(tmp_path: Path):
    # "ep" must not match "episode_10"; membership is per-line, not substring.
    dataset_dir = tmp_path / "ds"
    mark_appended(dataset_dir, "episode_10")
    assert already_appended(dataset_dir, "episode_10")
    assert not already_appended(dataset_dir, "episode_1")
    assert not already_appended(dataset_dir, "ep")


def _write_tum(path: Path, x: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_tum(
        path,
        PoseTrack(
            t=np.array([0.0, 1.0]),
            pos=np.array([[x, 0.0, 0.0], [x, 0.0, 0.0]]),
            quat=np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
        ),
    )


def _lidar_rig() -> core_config.RigConfig:
    return core_config.parse_rig_config(
        {
            "version": 1,
            "task": "t",
            "umis": {
                "umi1": {
                    "camera": {
                        "device": "/d",
                        "width": 64,
                        "height": 48,
                        "fps": 30,
                        "calib": "c.npz",
                    },
                    "lidar": {"iface": "eth0"},
                    "gripper": {"tag_ids": [2, 3]},
                }
            },
        }
    )


def test_pose_track_prefers_align_falls_back_to_glim(tmp_path: Path):
    # No pcap on disk => clock-skew realignment is a no-op, so the trajectory is read verbatim.
    umi = _lidar_rig().umis[0]
    m = {"sensors": {"umi1/lidar": {"kind": "lidar"}}}
    derived = tmp_path / "derived_ep"
    _write_tum(derived / "umi1" / "glim" / "trajectory.tum", 1.0)

    # No align output yet -> falls back to the per-lidar GLIM trajectory (non-joint path).
    track, basis = pose_track_for_umi(tmp_path, derived, m, umi)
    assert track.pos[0][0] == 1.0
    assert basis is not None  # lidar body-convention basis applied downstream

    # Align output present -> preferred (common-frame trajectory).
    _write_tum(derived / "align" / "umi1" / "trajectory.tum", 2.0)
    track2, _ = pose_track_for_umi(tmp_path, derived, m, umi)
    assert track2.pos[0][0] == 2.0
