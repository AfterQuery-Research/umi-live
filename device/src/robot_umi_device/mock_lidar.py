"""CI mock lidar: a SensorRecorder that writes a synthetic RoboSense-shaped pcap.

Mock and hardware exercise the SAME pipeline: the mock emits MSOP/IMU-shaped UDP
payloads (correct magics, sizes, temperature field) framed as real Ethernet/IPv4/UDP
pcap records — byte-compatible with what tcpdump captures from a real Airy, and with
what the server's decode stage and the lidar monitor's parsers expect.

Rates are scaled down (CI doesn't need 2240 pkt/s) but the shape is authentic.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

from robot_umi_core import layout, ports

from robot_umi_device.lidar_monitor import synth_imu_packet, synth_msop_packet
from robot_umi_device.pcap import PcapWriter
from robot_umi_device.sensors import SensorStats

TICK_S = 0.05
MSOP_PER_TICK = 4
IMU_PER_TICK = 1


class MockLidarPcapRecorder:
    """Drop-in for LidarPcapRecorder in tests / off-Pi dev (no tcpdump, no hardware)."""

    def __init__(
        self,
        sensor_key: str,
        *,
        lidar_ip: str = ports.ROBOSENSE_DEFAULT_LIDAR_IP,
        host_ip: str = ports.ROBOSENSE_DEFAULT_HOST_IP,
        temp_c: float = 41.5,
        fail_with: str | None = None,
    ):
        self.sensor_key = sensor_key
        self.lidar_ip = lidar_ip
        self.host_ip = host_ip
        self.temp_c = temp_c
        self.fail_with = fail_with  # simulate a dead lidar (no packets)
        self._writer: PcapWriter | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._packets = 0
        self._seq = 0

    def begin_episode(self, episode_dir: Path) -> None:
        files = layout.sensor_files(self.sensor_key)
        self._packets = 0
        self._writer = PcapWriter(episode_dir / files["pcap"])
        self._stop.clear()
        if self.fail_with is None:
            self._emit_tick()  # guarantee >=1 tick even for instant stop
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.wait(TICK_S):
            self._emit_tick()

    def _emit_tick(self) -> None:
        writer = self._writer
        if writer is None:
            return
        now = time.time_ns()
        for _ in range(MSOP_PER_TICK):
            writer.write_udp(
                now,
                synth_msop_packet(temp_c=self.temp_c, seq=self._seq, unix_ns=now),
                src_ip=self.lidar_ip,
                src_port=49610,
                dst_ip=self.host_ip,
                dst_port=ports.ROBOSENSE_MSOP_PORT,
            )
            self._seq += 1
            self._packets += 1
            now += 450_000  # ~0.45 ms apart, like the real wire
        for _ in range(IMU_PER_TICK):
            writer.write_udp(
                now,
                synth_imu_packet(),
                src_ip=self.lidar_ip,
                src_port=49607,
                dst_ip=self.host_ip,
                dst_port=ports.ROBOSENSE_IMU_PORT,
            )
            self._packets += 1

    def end_episode(self) -> SensorStats:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        error = None
        if self._packets == 0:
            error = self.fail_with or "no lidar packets captured"
        return SensorStats(
            stats={"packets_captured": self._packets, "packets_dropped_kernel": 0},
            error=error,
        )

    def cancel_episode(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def poll_error(self) -> str | None:
        return None
