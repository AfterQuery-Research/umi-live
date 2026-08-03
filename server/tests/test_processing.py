from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from robot_umi_server.processing import (
    DEFAULT_MAX_ALIGNMENT_GAP_NS,
    LidarSample,
    append_lerobot_episode,
    episode_to_records,
    latest_past_lidar_index,
    lidar_slam_state,
    load_lidar,
    open_dataset,
)
from tests.helpers import make_episode


def make_lidar(times_ns: list[int]) -> list[LidarSample]:
    return [
        LidarSample(time_monotonic_ns=t, ranges=np.full(360, 1.0, dtype=np.float32))
        for t in times_ns
    ]


def test_episode_to_records_aligns_video_and_lidar(tmp_path: Path):
    episode = make_episode(tmp_path)

    records = episode_to_records(episode)

    assert len(records) == 3
    assert records[0].image.shape == (48, 64, 3)
    assert records[0].state.shape == (6,)
    assert records[0].action.shape == (6,)
    assert records[0].timestamp_monotonic_ns == 1_000_000_000
    assert records[1].timestamp_monotonic_ns == 1_033_333_333
    assert np.isfinite(records[0].state).all()


def test_episode_to_records_rejects_missing_sidecar(tmp_path: Path):
    episode = make_episode(tmp_path, missing_sidecar=True)

    with pytest.raises(ValueError, match="camera_frames"):
        episode_to_records(episode)


def test_episode_to_records_rejects_sidecar_count_mismatch(tmp_path: Path):
    episode = make_episode(tmp_path, sidecar_count=2)

    with pytest.raises(ValueError, match="count"):
        episode_to_records(episode)


def test_episode_to_records_rejects_non_monotonic_sidecar(tmp_path: Path):
    episode = make_episode(tmp_path, non_monotonic_sidecar=True)

    with pytest.raises(ValueError, match="monotonic"):
        episode_to_records(episode)


def test_episode_to_records_rejects_non_contiguous_sidecar_index(tmp_path: Path):
    episode = make_episode(tmp_path, sidecar_index_gap=True)

    with pytest.raises(ValueError, match="contiguous"):
        episode_to_records(episode)


def test_episode_to_records_rejects_wrong_timestamp_clock(tmp_path: Path):
    episode = make_episode(tmp_path, timestamp_clock="realtime")

    with pytest.raises(ValueError, match="clock"):
        episode_to_records(episode)


def test_episode_to_records_rejects_wrong_timestamp_source(tmp_path: Path):
    episode = make_episode(tmp_path, timestamp_source="unknown")

    with pytest.raises(ValueError, match="source"):
        episode_to_records(episode)


def test_episode_to_records_rejects_ineligible_camera_stats(tmp_path: Path):
    episode = make_episode(tmp_path, camera_dataset_eligible=False)

    with pytest.raises(ValueError, match="not dataset eligible"):
        episode_to_records(episode)


def test_episode_to_records_rejects_empty_lidar_stream(tmp_path: Path):
    episode = make_episode(tmp_path, lidar_samples=0)

    with pytest.raises(ValueError, match="lidar"):
        episode_to_records(episode)


def test_episode_to_records_clamps_frame_just_before_first_lidar_sample(
    tmp_path: Path,
):
    # First lidar sample lands 10 ms after the first camera frame (sensor
    # warm-up). That is within tolerance, so the frame clamps to lidar[0].
    episode = make_episode(tmp_path, lidar_start_ns=1_010_000_000)

    records = episode_to_records(episode)

    assert len(records) == 3  # within tolerance: no staleness rejection


def test_alignment_clamps_within_tolerance():
    lidar = make_lidar([1_010_000_000, 1_110_000_000])
    # Frame 10 ms before the first sample clamps to index 0.
    assert latest_past_lidar_index(lidar, 0, 1_000_000_000) == 0


def test_alignment_rejects_frame_far_before_first_sample():
    lidar = make_lidar([3_000_000_000, 3_100_000_000])
    with pytest.raises(ValueError, match="precedes the lidar stream"):
        latest_past_lidar_index(lidar, 0, 1_000_000_000)


def test_alignment_rejects_stale_lidar_when_stream_dies():
    # Lidar stops after two samples; a camera frame 2 s later has only a stale
    # sample to align to, which must be rejected rather than silently frozen.
    lidar = make_lidar([1_000_000_000, 1_100_000_000])
    with pytest.raises(ValueError, match="stale"):
        latest_past_lidar_index(lidar, 0, 3_200_000_000)


def test_alignment_accepts_sample_within_gap():
    lidar = make_lidar([1_000_000_000, 1_100_000_000])
    idx = latest_past_lidar_index(
        lidar, 0, 1_100_000_000 + DEFAULT_MAX_ALIGNMENT_GAP_NS
    )
    assert idx == 1


def test_episode_to_records_rejects_non_monotonic_lidar(tmp_path: Path):
    episode = make_episode(tmp_path, non_monotonic_lidar=True)

    with pytest.raises(ValueError, match="lidar timestamps"):
        episode_to_records(episode)


def test_episode_to_records_with_3d_point_clouds(tmp_path: Path):
    episode = make_episode(tmp_path, lidar_format="points", lidar_points=240)

    records = episode_to_records(episode)

    assert len(records) == 3
    assert records[0].state.shape == (6,)
    assert records[0].action.shape == (6,)
    # TODO(SLAM): states are a zero-pose placeholder until SLAM lands.
    assert np.array_equal(np.stack([r.state for r in records]), np.zeros((3, 6)))


def test_3d_empty_frames_are_skipped_at_load(tmp_path: Path):
    episode = make_episode(tmp_path, lidar_format="points", lidar_points=60)
    index_path = episode / "mock_lidar.jsonl"
    lines = index_path.read_text().splitlines()
    empty_frame = json.loads(lines[1])
    empty_frame.update(
        {"point_count": 0, "length": 0, "time_monotonic_ns": 1_050_000_000}
    )
    lines.insert(1, json.dumps(empty_frame))
    index_path.write_text("\n".join(lines) + "\n")

    samples = load_lidar(index_path)

    assert len(samples) == 5  # the all-packets-lost frame carries no state
    assert all(sample.point_count > 0 for sample in samples)


def test_3d_missing_points_file_rejected(tmp_path: Path):
    episode = make_episode(tmp_path, lidar_format="points", lidar_points=60)
    (episode / "mock_lidar_points.bin").unlink()

    with pytest.raises(ValueError, match="missing mock_lidar_points.bin"):
        episode_to_records(episode)


def test_3d_index_beyond_points_file_rejected(tmp_path: Path):
    episode = make_episode(tmp_path, lidar_format="points", lidar_points=60)
    points_path = episode / "mock_lidar_points.bin"
    points_path.write_bytes(points_path.read_bytes()[:-16])

    with pytest.raises(ValueError, match="beyond the points file"):
        episode_to_records(episode)


def test_3d_index_length_mismatch_rejected(tmp_path: Path):
    episode = make_episode(tmp_path, lidar_format="points", lidar_points=60)
    index_path = episode / "mock_lidar.jsonl"
    lines = index_path.read_text().splitlines()
    bad = json.loads(lines[0])
    bad["length"] = bad["length"] - 8
    lines[0] = json.dumps(bad)
    index_path.write_text("\n".join(lines) + "\n")

    with pytest.raises(ValueError, match="length does not match point count"):
        episode_to_records(episode)


def test_lidar_slam_state_is_zero_pose_placeholder():
    # TODO(SLAM): replace with real pose assertions once SLAM lands.
    lidar = [LidarSample(time_monotonic_ns=1, point_count=10)]
    state = lidar_slam_state(lidar, 0)
    assert state.shape == (6,)
    assert state.dtype == np.float32
    assert not state.any()


def test_lerobot_append_writes_dataset_layout_from_3d_episode(tmp_path: Path):
    episode = make_episode(tmp_path / "raw", lidar_format="points", lidar_points=240)
    records = episode_to_records(episode)

    dataset = append_lerobot_episode(
        records,
        repo_id="test/umi3d",
        task="test task",
        root=tmp_path / "lerobot",
    )

    root = tmp_path / "lerobot" / "test" / "umi3d"
    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["total_episodes"] == dataset.meta.total_episodes == 1
    # Pose-shaped placeholder schema that SLAM will fill.
    assert info["features"]["observation.state"]["shape"] == [6]
    assert info["features"]["observation.state"]["names"] == [
        "x",
        "y",
        "z",
        "roll",
        "pitch",
        "yaw",
    ]


def test_episode_to_records_tolerates_blank_sidecar_line(tmp_path: Path):
    # A stray blank line among contiguous records must not desync the
    # frame_index contiguity check.
    episode = make_episode(tmp_path, sidecar_blank_line=True)

    records = episode_to_records(episode)

    assert len(records) == 3


def test_lerobot_append_writes_dataset_layout(tmp_path: Path):
    episode = make_episode(tmp_path / "raw")
    records = episode_to_records(episode)

    dataset = append_lerobot_episode(
        records,
        repo_id="test/umi",
        task="test task",
        root=tmp_path / "lerobot",
    )

    root = tmp_path / "lerobot" / "test" / "umi"
    assert (root / "meta" / "info.json").exists()
    assert list((root / "data").rglob("*.parquet"))
    assert list((root / "videos").rglob("*.mp4"))
    info = json.loads((root / "meta" / "info.json").read_text())
    assert info["total_episodes"] == dataset.meta.total_episodes == 1


def test_open_dataset_rejects_fps_change_on_resume(tmp_path: Path):
    episode = make_episode(tmp_path / "raw")
    records = episode_to_records(episode)
    append_lerobot_episode(
        records, repo_id="test/umi", task="t", root=tmp_path / "lerobot", fps=30
    )

    with pytest.raises(ValueError, match="does not match existing dataset fps"):
        open_dataset(
            records[0].image,
            repo_id="test/umi",
            root=tmp_path / "lerobot",
            fps=15,
            streaming_encoding=True,
        )
