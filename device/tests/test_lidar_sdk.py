from __future__ import annotations

import math
import struct
import time

import pytest

from robot_umi_device.mock_lidar_sdk import (
    AIRY_PROFILE,
    MID360_PROFILE,
    POINT_BYTES,
    MockLidar,
    MockLidarError,
)


@pytest.fixture
def lidar():
    device = MockLidar()  # Mid-360 profile
    try:
        yield device
    finally:
        device.disconnect()


def unpack_points(points: bytes) -> list[tuple[float, float, float, float]]:
    count = len(points) // POINT_BYTES
    return [struct.unpack_from("<4f", points, i * POINT_BYTES) for i in range(count)]


def collect_frames(device: MockLidar, n: int, timeout: float = 10.0):
    frames = []
    deadline = time.monotonic() + timeout
    while len(frames) < n and time.monotonic() < deadline:
        frame = device.read_frame(timeout=1.0)
        if frame is not None:
            frames.append(frame)
    return frames


def test_mid360_profile_matches_real_sensor_specs():
    # Livox Mid-360: 200k pts/s first return at 10 Hz, 360x59 deg FOV
    # (-7..+52), 0.1 m blind zone, 70 m max range.
    assert MID360_PROFILE.frame_rate_hz == 10
    assert MID360_PROFILE.points_per_frame == 20_000
    assert MID360_PROFILE.points_per_second == 200_000
    assert MID360_PROFILE.horizontal_fov_deg == 360.0
    assert MID360_PROFILE.vertical_fov_min_deg == -7.0
    assert MID360_PROFILE.vertical_fov_max_deg == 52.0
    assert MID360_PROFILE.range_min_m == 0.1
    assert MID360_PROFILE.range_max_m == 70.0


def test_airy_profile_matches_real_sensor_specs():
    # RoboSense Airy: ~860k pts/s single return at 10 Hz, 360x90 deg
    # hemispherical FOV, 60 m max range.
    assert AIRY_PROFILE.frame_rate_hz == 10
    assert AIRY_PROFILE.points_per_frame == 86_000
    assert AIRY_PROFILE.points_per_second == 860_000
    assert AIRY_PROFILE.vertical_fov_max_deg == 90.0
    assert AIRY_PROFILE.range_max_m == 60.0


def test_connect_returns_device_info(lidar: MockLidar):
    info = lidar.connect()
    assert info.model == "MOCK-MID360"
    assert info.frame_rate_hz == 10
    assert info.points_per_frame == 20_000
    assert info.points_per_second == 200_000
    assert info.serial.startswith("MOCK-MID360-")
    assert lidar.alive()


def test_frames_carry_full_point_clouds_at_frame_rate(lidar: MockLidar):
    lidar.connect()
    lidar.start_scanning()
    frames = collect_frames(lidar, 5)
    lidar.stop_scanning()

    assert len(frames) == 5
    for frame in frames:
        # No packets lost with an attentive consumer: full 20k-point frames.
        assert frame.point_count == 20_000
        assert len(frame.points) == 20_000 * POINT_BYTES
    timestamps = [frame.time_monotonic_ns for frame in frames]
    assert all(b > a for a, b in zip(timestamps, timestamps[1:]))
    gaps_ms = [(b - a) / 1e6 for a, b in zip(timestamps, timestamps[1:])]
    for gap in gaps_ms:
        assert 50 < gap < 250  # nominal 100 ms at 10 Hz
    sequences = [frame.seq for frame in frames]
    assert sequences == sorted(sequences)
    assert sum(frame.dropped_packets_total for frame in frames[:1]) == 0


def test_points_respect_fov_and_range_specs(lidar: MockLidar):
    info = lidar.connect()
    lidar.start_scanning()
    frames = collect_frames(lidar, 2)
    lidar.stop_scanning()

    v_min = math.radians(info.vertical_fov_min_deg) - 1e-6
    v_max = math.radians(info.vertical_fov_max_deg) + 1e-6
    for frame in frames:
        for x, y, z, intensity in unpack_points(frame.points)[::500]:
            distance = math.sqrt(x * x + y * y + z * z)
            assert info.range_min_m - 1e-3 <= distance <= info.range_max_m + 1e-3
            elevation = math.asin(max(-1.0, min(1.0, z / distance)))
            assert v_min <= elevation <= v_max
            assert 0.0 <= intensity <= 255.0


def test_azimuth_coverage_is_full_circle(lidar: MockLidar):
    lidar.connect()
    lidar.start_scanning()
    frames = collect_frames(lidar, 1)
    lidar.stop_scanning()

    seen = set()
    for x, y, _z, _i in unpack_points(frames[0].points)[::50]:
        azimuth = math.atan2(y, x)  # -pi..pi
        seen.add(int((azimuth + math.pi) / (2 * math.pi) * 12) % 12)
    assert len(seen) == 12  # every 30-degree sector hit within one frame


def test_no_frames_before_start_or_after_stop(lidar: MockLidar):
    lidar.connect()
    assert lidar.read_frame(timeout=0.15) is None  # not started yet

    lidar.start_scanning()
    assert collect_frames(lidar, 1)
    lidar.stop_scanning()
    time.sleep(0.1)
    while lidar.read_frame(timeout=0.05) is not None:
        pass  # drain in-flight packets
    assert lidar.read_frame(timeout=0.15) is None

    # scanning restarts cleanly (multiple episodes per session)
    lidar.start_scanning()
    assert collect_frames(lidar, 1)


def test_read_frame_raises_when_device_process_dies(lidar: MockLidar):
    lidar.connect()
    lidar.start_scanning()
    assert collect_frames(lidar, 1)

    assert lidar._process is not None
    lidar._process.kill()
    lidar._process.wait(timeout=5)

    with pytest.raises(MockLidarError, match="died"):
        for _ in range(1000):
            lidar.read_frame(timeout=0.1)


def test_slow_consumer_drops_packets_and_yields_partial_frames():
    # Airy profile: 54 packets x ~26 KB per frame vastly exceeds unprivileged
    # socket buffers, so a stalled consumer must lose packets (counted) and
    # then receive partial frames - never block or kill the device.
    device = MockLidar(profile=AIRY_PROFILE)
    try:
        device.connect()
        device.start_scanning()
        assert collect_frames(device, 1, timeout=5.0)
        time.sleep(1.5)  # stop consuming; buffers fill and the device drops
        assert device.alive()
        deadline = time.monotonic() + 10
        saw_drops = 0
        saw_partial = False
        while time.monotonic() < deadline:
            frame = device.read_frame(timeout=0.5)
            if frame is None:
                continue
            saw_drops = max(saw_drops, frame.dropped_packets_total)
            if 0 < frame.point_count < AIRY_PROFILE.points_per_frame:
                saw_partial = True
            if saw_drops and saw_partial:
                break
        assert saw_drops > 0
        assert saw_partial
        assert device.alive()
    finally:
        device.disconnect()


def test_disconnect_terminates_device_process(lidar: MockLidar):
    lidar.connect()
    process = lidar._process
    assert process is not None
    lidar.disconnect()
    assert process.poll() is not None
    assert not lidar.alive()


def test_operations_require_connection(lidar: MockLidar):
    with pytest.raises(MockLidarError, match="not connected"):
        lidar.start_scanning()
    lidar.connect()
    with pytest.raises(MockLidarError, match="already connected"):
        lidar.connect()
