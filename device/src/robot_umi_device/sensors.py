"""The one sensor seam: every sensor records through the same Protocol.

``EpisodeRecorder`` orchestrates a flat list of :class:`SensorRecorder`s built from
the rig config + the episode's mode (the enable mask). N UMIs x M sensors is config
data, not code branches.

Adapters:
- :class:`LidarPcapRecorder` — one tcpdump per lidar interface -> ``<key>.pcap``
- :class:`CameraRecorder`    — wraps the Rust V4L2 recorder (``CameraProcess``)
- :class:`ViveRecorder`      — drains the session-long ``NetworkPoseSensor`` into
  ``<key>.jsonl`` (this UMI's tracker only), raw samples on the kernel/PTP axis
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from robot_umi_core import layout

from robot_umi_device.camera import CameraProcess
from robot_umi_device.pcap import PcapCapture
from robot_umi_device.vive import NetworkPoseSensor


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


# --- vive ----------------------------------------------------------------------------


class ViveRecorder:
    """Writes this UMI's tracker samples (by name) from the shared receiver to JSONL.

    The UDP receiver is session-long and shared across UMIs; each ViveRecorder
    filters by its arm name and appends raw samples (both clocks kept) to
    ``<key>.jsonl`` on a small writer thread.
    """

    POLL_S = 0.05

    def __init__(self, sensor_key: str, receiver: NetworkPoseSensor, arm_name: str):
        self.sensor_key = sensor_key
        self.receiver = receiver
        self.arm_name = arm_name
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._file = None
        self._count = 0
        self._error: str | None = None

    def begin_episode(self, episode_dir: Path) -> None:
        files = layout.sensor_files(self.sensor_key)
        self._file = (episode_dir / files["poses"]).open("w", encoding="utf-8")
        self._count = 0
        self._error = None
        self.receiver.read(self.arm_name)  # drop this arm's pre-episode backlog only
        self._stop.clear()
        self._thread = threading.Thread(target=self._drain_loop, daemon=True)
        self._thread.start()

    def _drain_loop(self) -> None:
        while not self._stop.is_set():
            self._drain_once()
            time.sleep(self.POLL_S)
        self._drain_once()  # final flush

    def _drain_once(self) -> None:
        if self._file is None:
            return
        try:
            for s in self.receiver.read(self.arm_name):
                self._file.write(
                    json.dumps(
                        {
                            "name": s.name,
                            "recv_unix_ns": s.recv_unix_ns,
                            "sender_ts": s.sender_ts,
                            "data": list(s.data),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                self._count += 1
            self._file.flush()
        except OSError as exc:  # disk error: surface, don't crash the thread
            self._error = f"vive writer failed: {exc}"
            self._stop.set()

    def end_episode(self) -> SensorStats:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._file is not None:
            self._file.close()
            self._file = None
        error = self._error
        if error is None and self._count == 0:
            error = "no vive samples received (tracker off / sender not running?)"
        return SensorStats(stats={"sample_count": self._count}, error=error)

    def cancel_episode(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._file is not None:
            self._file.close()
            self._file = None

    def poll_error(self) -> str | None:
        return self._error
