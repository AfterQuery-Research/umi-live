"""Synthetic lidar PCAP fixtures for tests; never installed in production."""

from __future__ import annotations

import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from umi_live_core import layout, ports
from umi_live_core.pcapio import LINKTYPE_ETHERNET, PCAP_MAGIC_NS

_ETH_HEADER = struct.Struct("!6s6sH")
_IP_HEADER = struct.Struct("!BBHHHBBH4s4s")
_UDP_HEADER = struct.Struct("!HHHH")
_MSOP_TEMP_OFFSET = 38
_MSOP_TS_OFFSET = 20

TICK_S = 0.05
MSOP_PER_TICK = 4
IMU_PER_TICK = 1


@dataclass
class SensorStats:
    """Test-side equivalent of the device sensor result value."""

    stats: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


def _ip_checksum(header: bytes) -> int:
    total = 0
    for i in range(0, len(header), 2):
        total += (header[i] << 8) | header[i + 1]
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return ~total & 0xFFFF


def _ip_to_bytes(ip: str) -> bytes:
    return bytes(int(part) for part in ip.split("."))


class PcapWriter:
    """Nanosecond classic PCAP with synthesized Ethernet/IPv4/UDP framing."""

    def __init__(self, path: Path):
        self._f = path.open("wb")
        self._f.write(struct.pack("<IHHiIII", PCAP_MAGIC_NS, 2, 4, 0, 0, 0xFFFF, LINKTYPE_ETHERNET))

    def write_udp(
        self,
        ts_unix_ns: int,
        payload: bytes,
        *,
        src_ip: str,
        src_port: int,
        dst_ip: str,
        dst_port: int,
    ) -> None:
        udp = _UDP_HEADER.pack(src_port, dst_port, 8 + len(payload), 0) + payload
        ip_len = 20 + len(udp)
        ip_wo_checksum = _IP_HEADER.pack(
            0x45, 0, ip_len, 0, 0, 64, 17, 0, _ip_to_bytes(src_ip), _ip_to_bytes(dst_ip)
        )
        ip = _IP_HEADER.pack(
            0x45,
            0,
            ip_len,
            0,
            0,
            64,
            17,
            _ip_checksum(ip_wo_checksum),
            _ip_to_bytes(src_ip),
            _ip_to_bytes(dst_ip),
        )
        frame = _ETH_HEADER.pack(b"\x02" * 6, b"\x04" * 6, 0x0800) + ip + udp
        self._f.write(
            struct.pack(
                "<IIII",
                ts_unix_ns // 1_000_000_000,
                ts_unix_ns % 1_000_000_000,
                len(frame),
                len(frame),
            )
        )
        self._f.write(frame)

    def flush(self) -> None:
        self._f.flush()

    def close(self) -> None:
        self._f.close()


def synth_msop_packet(temp_c: float = 40.0, seq: int = 0, unix_ns: int | None = None) -> bytes:
    """Build a RoboSense Airy-shaped MSOP payload for tests."""
    packet = bytearray(ports.ROBOSENSE_MSOP_LEN)
    packet[0:4] = ports.ROBOSENSE_MSOP_MAGIC
    packet[8:12] = struct.pack(">I", seq)
    if unix_ns is not None:
        seconds, micros = unix_ns // 1_000_000_000, (unix_ns % 1_000_000_000) // 1_000
        packet[_MSOP_TS_OFFSET : _MSOP_TS_OFFSET + 6] = seconds.to_bytes(6, "big")
        packet[_MSOP_TS_OFFSET + 6 : _MSOP_TS_OFFSET + 10] = micros.to_bytes(4, "big")
    value = int(round(temp_c / ports.ROBOSENSE_TEMP_RES_C))
    negative = value < 0
    value = abs(value)
    packet[_MSOP_TEMP_OFFSET] = (value & 0x1F) << 3
    packet[_MSOP_TEMP_OFFSET + 1] = ((value >> 5) & 0x7F) | (0x80 if negative else 0)
    return bytes(packet)


def synth_imu_packet() -> bytes:
    """Build a RoboSense Airy-shaped IMU payload for tests."""
    packet = bytearray(ports.ROBOSENSE_IMU_LEN)
    packet[0:4] = ports.ROBOSENSE_IMU_MAGIC
    return bytes(packet)


class MockLidarPcapRecorder:
    """Test-only drop-in replacement for the real lidar recorder."""

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
        self.fail_with = fail_with
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
            self._emit_tick()
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
            now += 450_000
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

    def end_episode(self) -> Any:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        error = None if self._packets else self.fail_with or "no lidar packets captured"
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
