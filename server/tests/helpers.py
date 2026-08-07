"""Test helpers: build REAL v2 episodes using the actual device code.

The episode fixture is produced by ``robot_umi_device.episode.EpisodeRecorder``
with a cv2-backed fake camera (real MJPEG AVI + frames jsonl) and the mock lidar
(synthetic RoboSense-shaped pcap) — so server tests exercise the same bytes the
Pi produces.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path

import cv2
import numpy as np
import yaml
from robot_umi_core import config as core_config
from robot_umi_core import layout, manifest
from robot_umi_device.episode import EpisodeRecorder
from robot_umi_device.mock_lidar import MockLidarPcapRecorder
from robot_umi_device.sensors import SensorStats

FRAME_PERIOD_NS = 33_333_333


class CvCameraRecorder:
    """Sensor-seam camera fake: real MJPEG AVI + v2 frames jsonl (unix-ns stamps)."""

    def __init__(
        self,
        sensor_key: str,
        frames: int = 6,
        size: tuple[int, int] = (64, 48),
        start_unix_ns: int = 1_752_600_000_000_000_000,
    ):
        self.sensor_key = sensor_key
        self.frames = frames
        self.size = size
        self.start_unix_ns = start_unix_ns

    def begin_episode(self, episode_dir: Path) -> None:
        files = layout.sensor_files(self.sensor_key)
        writer = cv2.VideoWriter(
            str(episode_dir / files["video"]),
            cv2.VideoWriter.fourcc(*"MJPG"),
            30.0,
            self.size,
        )
        rows = []
        try:
            for i in range(self.frames):
                image = np.full((self.size[1], self.size[0], 3), i * 40, dtype=np.uint8)
                writer.write(image)
                rows.append(
                    {
                        "frame_index": i,
                        "v4l2_sequence": i,
                        "time_monotonic_ns": 1_000_000_000 + i * FRAME_PERIOD_NS,
                        "time_unix_ns": self.start_unix_ns + i * FRAME_PERIOD_NS,
                        "timestamp_clock": "monotonic",
                        "timestamp_source": "start_of_exposure",
                    }
                )
        finally:
            writer.release()
        (episode_dir / files["frames"]).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        (episode_dir / files["stats"]).write_text("{}")

    def end_episode(self) -> SensorStats:
        return SensorStats(stats={"camera_frame_count": self.frames})

    def cancel_episode(self) -> None: ...

    def poll_error(self) -> str | None:
        return None


def write_client_config(config_root: Path, umi_name: str) -> Path:
    (config_root / "config" / "calib").mkdir(parents=True, exist_ok=True)
    index = 1 if umi_name == "umi1" else 2
    calib = config_root / "config" / "calib" / f"{umi_name}_calib.npz"
    calib.write_bytes(b"fake-npz-" + str(index).encode())
    raw = {
        "version": 1,
        "task": "test task",
        "umis": {
            umi_name: {
                "camera": {
                    "device": f"/dev/fake{index}",
                    "width": 64,
                    "height": 48,
                    "fps": 30,
                    "calib": f"config/calib/{umi_name}_calib.npz",
                },
                "lidar": {"iface": "eth0", "host_ip": "192.168.1.102"},
                "gripper": {"tag_ids": [0, 1] if umi_name == "umi1" else [2, 3]},
            }
        },
    }
    config_path = config_root / "config" / "umi.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    return config_path


def make_arm_episode(
    tmp_path: Path,
    umi_name: str,
    *,
    frames: int = 6,
    ptp_locked: bool = True,
    session_id: str = "session_test",
) -> Path:
    """Record one arm transport episode through the real device recorder."""
    config_root = tmp_path / f"repo_{umi_name}"
    config_path = write_client_config(config_root, umi_name)
    rig = core_config.load_rig_config(config_path)

    sensors = {}
    for u in rig.umis:
        cam_key = layout.sensor_key(u.name, layout.KIND_CAMERA)
        sensors[cam_key] = CvCameraRecorder(cam_key, frames=frames)
        lidar_key = layout.sensor_key(u.name, layout.KIND_LIDAR)
        sensors[lidar_key] = MockLidarPcapRecorder(lidar_key)

    recorder = EpisodeRecorder(
        rig,
        config_path,
        tmp_path / f"spool_{umi_name}",
        sensors,
        config_root=config_root,
        ptp_probe=lambda: {
            u.lidar.iface: manifest.PtpStatus(
                locked=ptp_locked,
                timestamping_mode="hardware",
                residual_offset_ns=1_000,
            )
            for u in rig.umis
            if u.lidar is not None
        },
        clock_sync_probe=lambda: manifest.ClockSyncStatus(
            synced=True, ref="system-chrony", offset_to_ref_ns=1_000
        ),
    )
    recorder.start(session_id=session_id, role=umi_name)
    return recorder.stop()


def make_v2_episode(tmp_path: Path, frames: int = 6, ptp_locked: bool = True) -> Path:
    """Compatibility name for an ingest-ready leader arm episode."""
    return make_arm_episode(tmp_path, "umi1", frames=frames, ptp_locked=ptp_locked)


def make_paired_episode(tmp_path: Path, frames: int = 6) -> Path:
    """Build the only processable input: a merged umi1+umi2 episode."""
    from robot_umi_server.pair_stage import merge_pair

    leader = make_arm_episode(tmp_path, "umi1", frames=frames)
    follower = make_arm_episode(tmp_path, "umi2", frames=frames)
    paired = tmp_path / "paired_session_test"
    merge_pair(leader, follower, paired, "session_test")
    return paired


def make_episode_tar(episode_dir: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = out_dir / f"{episode_dir.name}.tar"
    with tarfile.open(tar_path, "w") as tar:
        tar.add(episode_dir, arcname=episode_dir.name)
    return tar_path


def frame_unix_times(episode_dir: Path, umi: str = "umi1") -> list[int]:
    files = layout.sensor_files(layout.sensor_key(umi, layout.KIND_CAMERA))
    return [
        int(json.loads(line)["time_unix_ns"])
        for line in (episode_dir / files["frames"]).read_text().splitlines()
        if line.strip()
    ]
