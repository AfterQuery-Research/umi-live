from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from robot_umi_device.lidar import LidarError, LidarRecorder
from robot_umi_device.mock_lidar_sdk import POINT_BYTES


@pytest.fixture
def recorder():
    lidar = LidarRecorder()  # Mid-360 profile: 20k pts/frame at 10 Hz
    try:
        yield lidar
    finally:
        lidar.close()


def read_index(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_episode_writes_frame_index_and_points(recorder: LidarRecorder, tmp_path: Path):
    info = recorder.connect()
    assert info.model == "MOCK-MID360"
    assert info.points_per_second == 200_000

    index_path = tmp_path / "mock_lidar.jsonl"
    points_path = tmp_path / "mock_lidar_points.bin"
    recorder.begin_episode(index_path, points_path)
    time.sleep(1.0)
    stats = recorder.end_episode()

    index = read_index(index_path)
    assert len(index) == stats.frame_count
    assert stats.frame_count >= 8  # ~10 expected over ~1s at 10 Hz
    assert stats.model == "MOCK-MID360"
    assert stats.max_gap_ms < 300.0
    assert stats.points_total == sum(row["point_count"] for row in index)
    # Nominal operation: nearly all points delivered (an occasional packet
    # drop under IO stalls is honest accounting, not failure).
    assert stats.points_total >= int(0.95 * stats.frame_count * 20_000)
    assert sum(1 for row in index if row["point_count"] == 20_000) >= len(index) // 2

    first = index[0]
    for key in (
        "sensor",
        "model",
        "seq",
        "time_monotonic_ns",
        "time_unix_ns",
        "point_count",
        "offset",
        "length",
        "dropped_packets_total",
    ):
        assert key in first
    assert first["sensor"] == "mock_lidar"
    assert first["offset"] == 0
    assert first["length"] == first["point_count"] * POINT_BYTES

    # Index offsets/lengths exactly tile the points file.
    expected_size = sum(row["length"] for row in index)
    assert points_path.stat().st_size == expected_size
    for previous, current in zip(index, index[1:]):
        assert current["offset"] == previous["offset"] + previous["length"]

    timestamps = [row["time_monotonic_ns"] for row in index]
    assert all(b > a for a, b in zip(timestamps, timestamps[1:]))


def test_back_to_back_episodes_reuse_the_connection(
    recorder: LidarRecorder, tmp_path: Path
):
    recorder.connect()
    for name in ("ep1", "ep2"):
        recorder.begin_episode(tmp_path / f"{name}.jsonl", tmp_path / f"{name}.bin")
        time.sleep(0.6)
        stats = recorder.end_episode()
        assert stats.frame_count >= 4

    ep1_last = read_index(tmp_path / "ep1.jsonl")[-1]["time_monotonic_ns"]
    ep2_first = read_index(tmp_path / "ep2.jsonl")[0]["time_monotonic_ns"]
    assert ep2_first > ep1_last  # frames stay fresh across episodes


def test_device_death_mid_episode_surfaces_error(
    recorder: LidarRecorder, tmp_path: Path
):
    recorder.connect()
    recorder.begin_episode(tmp_path / "mock_lidar.jsonl", tmp_path / "points.bin")
    time.sleep(0.5)

    assert recorder.sdk._process is not None
    recorder.sdk._process.kill()
    recorder.sdk._process.wait(timeout=5)

    deadline = time.monotonic() + 5
    while recorder.poll_error() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert recorder.poll_error() is not None

    with pytest.raises(LidarError):
        recorder.end_episode()
    # Frames captured before the failure are preserved on disk.
    assert read_index(tmp_path / "mock_lidar.jsonl")


def test_begin_requires_connection(recorder: LidarRecorder, tmp_path: Path):
    with pytest.raises(LidarError, match="not connected"):
        recorder.begin_episode(tmp_path / "i.jsonl", tmp_path / "p.bin")


def test_begin_twice_rejected(recorder: LidarRecorder, tmp_path: Path):
    recorder.connect()
    recorder.begin_episode(tmp_path / "i.jsonl", tmp_path / "p.bin")
    try:
        with pytest.raises(LidarError, match="already active"):
            recorder.begin_episode(tmp_path / "i2.jsonl", tmp_path / "p2.bin")
    finally:
        recorder.cancel_episode()


def test_end_without_begin_raises(recorder: LidarRecorder):
    recorder.connect()
    with pytest.raises(LidarError, match="no active"):
        recorder.end_episode()


def test_end_after_cancel_never_returns_stale_stats(
    recorder: LidarRecorder, tmp_path: Path
):
    recorder.connect()
    recorder.begin_episode(tmp_path / "a.jsonl", tmp_path / "a.bin")
    time.sleep(0.3)
    recorder.cancel_episode()
    with pytest.raises(LidarError, match="no active"):
        recorder.end_episode()


def test_instant_stop_still_captures_a_frame(recorder: LidarRecorder, tmp_path: Path):
    # begin_episode waits for the first frame, so an operator stopping
    # immediately still gets a valid single-frame episode.
    recorder.connect()
    recorder.begin_episode(tmp_path / "i.jsonl", tmp_path / "p.bin")
    stats = recorder.end_episode()
    assert stats.frame_count >= 1
    assert stats.points_total > 0


def test_ring_buffer_absorbs_disk_stalls_without_losing_frames(
    recorder: LidarRecorder, tmp_path: Path, monkeypatch
):
    # Simulate an SD card that freezes for 1.5s twice: the writer thread
    # blocks, but the reader keeps draining the socket into the RAM buffer,
    # so the on-disk index has no gaps and nothing is dropped.
    recorder.connect()
    original_write = LidarRecorder._write_frame
    stalls = iter([1.5, 1.5])

    def stalling_write(self, frame):
        delay = next(stalls, 0.0)
        if delay:
            time.sleep(delay)
        original_write(self, frame)

    monkeypatch.setattr(LidarRecorder, "_write_frame", stalling_write)
    recorder.begin_episode(tmp_path / "i.jsonl", tmp_path / "p.bin")
    time.sleep(4.0)
    stats = recorder.end_episode()

    assert stats.host_dropped_frames == 0
    index = read_index(tmp_path / "i.jsonl")
    timestamps = [row["time_monotonic_ns"] for row in index]
    gaps_ms = [(b - a) / 1e6 for a, b in zip(timestamps, timestamps[1:])]
    assert max(gaps_ms) < 300.0  # buffered through the stall, no index gap
    assert stats.frame_count >= 30  # ~40 expected over ~4s at 10 Hz


def test_ring_buffer_overflow_drops_honestly(
    recorder: LidarRecorder, tmp_path: Path, monkeypatch
):
    # A stall longer than the buffer (64 frames = 6.4s at 10 Hz) must drop
    # frames with honest accounting instead of blocking or crashing.
    recorder.connect()
    original_write = LidarRecorder._write_frame
    stalled_once = []

    def long_stall_write(self, frame):
        if not stalled_once:
            stalled_once.append(True)
            time.sleep(8.5)
        original_write(self, frame)

    monkeypatch.setattr(LidarRecorder, "_write_frame", long_stall_write)
    recorder.begin_episode(tmp_path / "i.jsonl", tmp_path / "p.bin")
    time.sleep(9.5)
    stats = recorder.end_episode()

    assert stats.host_dropped_frames > 0  # overflow counted, not hidden
    assert stats.frame_count >= 60  # buffered frames were preserved


def test_writer_failure_mid_episode_surfaces_error(
    recorder: LidarRecorder, tmp_path: Path, monkeypatch
):
    recorder.connect()
    recorder.begin_episode(tmp_path / "i.jsonl", tmp_path / "p.bin")
    time.sleep(0.3)

    def exploding_write(self, frame):
        raise OSError("No space left on device")

    monkeypatch.setattr(LidarRecorder, "_write_frame", exploding_write)
    deadline = time.monotonic() + 5
    while recorder.poll_error() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert "No space left" in (recorder.poll_error() or "")
    with pytest.raises(LidarError, match="No space left"):
        recorder.end_episode()


def test_cancel_episode_is_silent_and_reusable(recorder: LidarRecorder, tmp_path: Path):
    recorder.connect()
    recorder.begin_episode(tmp_path / "c.jsonl", tmp_path / "cp.bin")
    time.sleep(0.3)
    recorder.cancel_episode()

    recorder.begin_episode(tmp_path / "n.jsonl", tmp_path / "np.bin")
    time.sleep(0.5)
    stats = recorder.end_episode()
    assert stats.frame_count >= 3
