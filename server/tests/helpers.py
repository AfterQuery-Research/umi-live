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
        (episode_dir / files["frames"]).write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n"
        )
        (episode_dir / files["stats"]).write_text("{}")

    def end_episode(self) -> SensorStats:
        return SensorStats(stats={"camera_frame_count": self.frames})

    def cancel_episode(self) -> None: ...

    def poll_error(self) -> str | None:
        return None


def write_rig_config(config_root: Path, n_umis: int = 1, with_gripper: bool = False) -> Path:
    (config_root / "config" / "calib").mkdir(parents=True, exist_ok=True)
    umis = {}
    for i in range(1, n_umis + 1):
        calib = config_root / "config" / "calib" / f"umi{i}_calib.npz"
        calib.write_bytes(b"fake-npz-" + str(i).encode())
        umi: dict = {
            "camera": {
                "device": f"/dev/fake{i}",
                "width": 64,
                "height": 48,
                "fps": 30,
                "calib": f"config/calib/umi{i}_calib.npz",
            },
            "lidar": {"iface": f"eth{i - 1}", "host_ip": f"192.168.{i}.102"},
        }
        if with_gripper:
            umi["gripper"] = {"tag_ids": [2 * (i - 1), 2 * i - 1]}
        umis[f"umi{i}"] = umi
    raw = {"version": 1, "motion_source": "lidar", "task": "test task", "umis": umis}
    config_path = config_root / "config" / "umi.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    return config_path


def make_v2_episode(
    tmp_path: Path, n_umis: int = 1, frames: int = 6, ptp_locked: bool = True
) -> Path:
    """Record one complete episode through the real device recorder."""
    config_root = tmp_path / "repo"
    config_path = write_rig_config(config_root, n_umis=n_umis)
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
        tmp_path / "spool",
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
    )
    recorder.start(core_config.MODE_LIDAR)
    return recorder.stop()


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
