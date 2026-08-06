"""The one sensor seam: every sensor records through the same Protocol.

``EpisodeRecorder`` orchestrates the client's lidar and camera recorders.

Adapters:
- :class:`LidarPcapRecorder` — one tcpdump per lidar interface -> ``<key>.pcap``
- :class:`CameraRecorder`    — wraps the Rust V4L2 recorder (``CameraProcess``)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from robot_umi_core import layout

from robot_umi_device.camera import CameraProcess
from robot_umi_device.pcap import PcapCapture


@dataclass
class SensorStats:
    """What a sensor reports at end_episode: kind-specific counters + optional error."""

    stats: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class SensorRecorder(Protocol):
    """Per-episode lifecycle every sensor implements. ``sensor_key`` names the stream."""

    sensor_key: str

    def begin_episode(self, episode_dir: Path) -> None: ...
    def end_episode(self) -> SensorStats: ...
    def cancel_episode(self) -> None: ...
    def poll_error(self) -> str | None: ...


# --- lidar ---------------------------------------------------------------------------


class LidarPcapRecorder:
    """Raw lidar capture: verbatim wire packets to a standard pcap, one per lidar.

    Passive by design — an unplugged lidar is a packet gap, not an error here. The
    only errors are capture-process failures (tcpdump died). Kernel drop counts land
    in stats so losses are visible.
    """

    def __init__(self, sensor_key: str, iface: str, bpf: str):
        self.sensor_key = sensor_key
        self._capture = PcapCapture(iface, bpf)
        self._pcap_path: Path | None = None

    def begin_episode(self, episode_dir: Path) -> None:
        files = layout.sensor_files(self.sensor_key)
        self._pcap_path = episode_dir / files["pcap"]
        self._capture.start(self._pcap_path)

    def end_episode(self) -> SensorStats:
        stats = self._capture.stop()
        error = None
        if self._pcap_path is None or not self._pcap_path.exists():
            error = "no pcap file was written"
        elif stats.get("packets_captured", 0) == 0:
            error = "no lidar packets captured"
        return SensorStats(stats=stats, error=error)

    def cancel_episode(self) -> None:
        self._capture.cancel()

    def poll_error(self) -> str | None:
        return self._capture.poll_error()


# --- camera --------------------------------------------------------------------------


class CameraRecorder:
    """Adapts the Rust ``CameraProcess`` to the sensor seam; derives its own paths."""

    def __init__(self, sensor_key: str, camera: CameraProcess):
        self.sensor_key = sensor_key
        self.camera = camera
        self.start_monotonic_ns: int | None = None
        self.stop_monotonic_ns: int | None = None

    def prepare(self) -> None:
        self.camera.prepare()

    def begin_episode(self, episode_dir: Path) -> None:
        files = layout.sensor_files(self.sensor_key)
        self.start_monotonic_ns = self.camera.start(
            episode_dir / files["video"],
            episode_dir / files["frames"],
            episode_dir / files["stats"],
        )
        self.stop_monotonic_ns = None

    def end_episode(self) -> SensorStats:
        self.stop_monotonic_ns = self.camera.stop()
        stats = self.camera.stats()
        error = str(stats.get("camera_error") or "") or None
        stats["start_monotonic_ns"] = self.start_monotonic_ns
        stats["stop_monotonic_ns"] = self.stop_monotonic_ns
        return SensorStats(stats=stats, error=error)

    def cancel_episode(self) -> None:
        self.camera.cancel()

    def poll_error(self) -> str | None:
        return self.camera.poll_error()
