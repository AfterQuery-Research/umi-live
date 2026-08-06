"""Fixed lidar episode lifecycle, hermetic snapshot, manifest v2, checksums."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from robot_umi_core import config as core_config
from robot_umi_core import layout, manifest
from robot_umi_device.episode import EpisodeRecorder, write_checksums
from robot_umi_device.mock_lidar import MockLidarPcapRecorder
from robot_umi_device.pcap import read_udp_packets
from robot_umi_device.sensors import SensorStats


class FakeCameraRecorder:
    """Sensor-seam fake standing in for the Rust camera (no hardware in CI)."""

    def __init__(self, sensor_key: str, fail: bool = False):
        self.sensor_key = sensor_key
        self.fail = fail

    def begin_episode(self, episode_dir: Path) -> None:
        files = layout.sensor_files(self.sensor_key)
        (episode_dir / files["video"]).write_bytes(b"AVI!")
        (episode_dir / files["frames"]).write_text('{"frame_index":0}\n')
        (episode_dir / files["stats"]).write_text("{}")

    def end_episode(self) -> SensorStats:
        if self.fail:
            return SensorStats(error="camera exploded")
        return SensorStats(stats={"camera_frame_count": 1})

    def cancel_episode(self) -> None: ...

    def poll_error(self) -> str | None:
        return None


@pytest.fixture
def rig_and_recorder(tmp_path):
    config_root = tmp_path / "repo"
    (config_root / "config" / "calib").mkdir(parents=True)
    calib = config_root / "config" / "calib" / "umi1_calib.npz"
    calib.write_bytes(b"npzblob")
    raw = {
        "version": 1,
        "task": "test",
        "umis": {
            "umi1": {
                "camera": {
                    "device": "/dev/null",
                    "width": 800,
                    "height": 600,
                    "fps": 30,
                    "calib": "config/calib/umi1_calib.npz",
                },
                "lidar": {"iface": "eth0"},
                "gripper": {"tag_ids": [0, 1]},
            }
        },
    }
    config_path = config_root / "config" / "umi.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    rig = core_config.load_rig_config(config_path)

    sensors = {
        "umi1/camera": FakeCameraRecorder("umi1/camera"),
        "umi1/lidar": MockLidarPcapRecorder("umi1/lidar"),
    }
    recorder = EpisodeRecorder(
        rig,
        config_path,
        tmp_path / "spool",
        sensors,
        config_root=config_root,
        ptp_probe=lambda: {
            "eth0": manifest.PtpStatus(
                locked=True, timestamping_mode="hardware", residual_offset_ns=1000
            )
        },
    )
    return rig, recorder, sensors


def test_full_episode_lifecycle(rig_and_recorder):
    _rig, recorder, _sensors = rig_and_recorder
    active = recorder.start(session_id="sess_1", role="umi1")
    assert recorder.active is not None
    final = recorder.stop()
    assert final.exists() and final.parent.name == "complete"

    m = manifest.read_manifest(final / layout.MANIFEST_FILE)
    assert m["manifest_version"] == 3
    assert "mode" not in m
    assert m["dataset_eligible"] is True
    assert set(m["sensors"]) == {"umi1/camera", "umi1/lidar"}
    assert m["ptp"]["eth0"]["locked"] is True
    assert m["sensors"]["umi1/lidar"]["stats"]["packets_captured"] > 0
    assert (final / layout.COMPLETE_FILE).exists()

    # Hermetic snapshot: resolved YAML + referenced calib blob travel in the episode.
    snap = final / layout.CONFIG_SNAPSHOT_DIR
    assert (snap / layout.CONFIG_SNAPSHOT_YAML).exists()
    assert (snap / "config" / "calib" / "umi1_calib.npz").read_bytes() == b"npzblob"
    assert m["config_sha256"] == layout.sha256_file(snap / layout.CONFIG_SNAPSHOT_YAML)

    # The lidar pcap contains RoboSense-shaped packets.
    pcap = final / "umi1_lidar.pcap"
    pkts = list(read_udp_packets(pcap))
    assert any(p.dst_port == 6699 for p in pkts)

    # Checksums are recursive (snapshot blobs are covered) and correct.
    checksums = (final / layout.CHECKSUMS_FILE).read_text()
    assert "config_snapshot/config/calib/umi1_calib.npz" in checksums
    assert active.episode_id == final.name


def test_fixed_capture_plan(rig_and_recorder):
    _rig, recorder, _sensors = rig_and_recorder
    active = recorder.start(session_id="sess_1", role="umi1")
    keys = {s.sensor_key for s in active.sensors}
    assert keys == {"umi1/camera", "umi1/lidar"}
    recorder.stop()


def test_armed_sensor_error_marks_ineligible(tmp_path):
    config_root = tmp_path / "repo"
    (config_root / "config").mkdir(parents=True)
    (config_root / "config" / "c.npz").write_bytes(b"x")
    raw = {
        "version": 1,
        "umis": {
            "umi1": {
                "camera": {
                    "device": "/dev/null",
                    "width": 800,
                    "height": 600,
                    "fps": 30,
                    "calib": "config/c.npz",
                },
                "lidar": {"iface": "eth0"},
                "gripper": {"tag_ids": [0, 1]},
            }
        },
    }
    config_path = config_root / "config" / "umi.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    rig = core_config.load_rig_config(config_path)
    sensors = {
        "umi1/camera": FakeCameraRecorder("umi1/camera", fail=True),
        "umi1/lidar": MockLidarPcapRecorder("umi1/lidar"),
    }
    recorder = EpisodeRecorder(
        rig, config_path, tmp_path / "spool", sensors, config_root=config_root
    )
    recorder.start(session_id="sess_1", role="umi1")
    final = recorder.stop()
    m = json.loads((final / layout.MANIFEST_FILE).read_text())
    assert m["dataset_eligible"] is False
    assert m["sensors"]["umi1/camera"]["error"] == "camera exploded"
    # Data preserved despite the failure.
    assert (final / "umi1_lidar.pcap").exists()


def test_cancel_discards_episode(rig_and_recorder):
    _rig, recorder, _sensors = rig_and_recorder
    active = recorder.start(session_id="sess_1", role="umi1")
    path = active.path
    recorder.cancel()
    assert not path.exists()
    assert recorder.active is None


def test_write_checksums_recursive(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("b")
    hashes = write_checksums(tmp_path)
    assert set(hashes) == {"a.txt", "sub/b.txt"}
