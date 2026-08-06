"""Stateless lidar liveness/type/temperature monitor — ADVISORY TELEMETRY ONLY.

Feeds the LCD and manifest QC; never touches captured bytes (capture is tcpdump's
job and keeps running regardless — an unplugged lidar is just a packet gap).

How it works: one tiny-RCVBUF UDP socket per known lidar port. The kernel keeps at
most a few datagrams per socket and silently discards the rest (zero CPU); poll()
drains whatever is there, classifies the vendor by port, and peeks the RoboSense
MSOP header temperature at a fixed offset. If the parse is ever wrong the only
casualty is an LCD number — the server decodes authoritative values from the pcap.
"""

from __future__ import annotations

import socket
import struct
import time
from dataclasses import dataclass

from robot_umi_core import ports

# Advisory staleness: no packet on any lidar port for this long => absent.
STALE_S = 2.0

# RoboSense MSOP header: id[4] rsvd[4] cnt[4] cnt[4] type[2] rsvd[2] ts[10] rsvd[1]
# lidar_type[1] mode[1] rsvd[5] temp[2]  -> temp at offset 38 (rs_driver RSAIRY layout).
_MSOP_TEMP_OFFSET = 38
# RSAIRY MSOP capture time: 6B big-endian seconds + 4B big-endian microseconds at offset 20
# (see robot_umi_core.lidar_clock). Synth packets stamp it so the pcap-arrival skew is ~0.
_MSOP_TS_OFFSET = 20


def parse_robosense_temp_c(msop_payload: bytes) -> float | None:
    """rs_driver's parseTempInLe bit layout: |lsb:5|pad:3| |neg:1|msb:7|, res 0.0625 C."""
    if len(msop_payload) < _MSOP_TEMP_OFFSET + 2:
        return None
    b0, b1 = msop_payload[_MSOP_TEMP_OFFSET], msop_payload[_MSOP_TEMP_OFFSET + 1]
    lsb = b0 >> 3
    neg = b1 & 0x80
    msb = b1 & 0x7F
    value = (msb << 5) + lsb
    if neg:
        value = -value
    return value * ports.ROBOSENSE_TEMP_RES_C


@dataclass
class LidarHealth:
    present: bool
    lidar_type: str | None  # "robosense" or None
    temp_c: float | None
    last_seen_unix_ns: int | None
    packets_seen: int


class LidarMonitor:
    """Per-lidar-interface health monitor. One instance per configured lidar."""

    # Ports we listen on: RoboSense MSOP (temperature lives there) + DIFOP.
    _PORTS = (
        ports.ROBOSENSE_MSOP_PORT,
        ports.ROBOSENSE_DIFOP_PORT,
    )
    _RCVBUF = 16 * 1024  # a handful of datagrams; kernel discards the rest for free

    def __init__(self, host_ip: str = "0.0.0.0"):
        self.host_ip = host_ip
        self._socks: list[socket.socket] = []
        self._last_seen_ns: int | None = None
        self._lidar_type: str | None = None
        self._temp_c: float | None = None
        self._packets = 0

    def start(self) -> None:
        for port in self._PORTS:
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self._RCVBUF)
                except OSError:
                    pass
                s.bind((self.host_ip, port))
                s.setblocking(False)
                self._socks.append(s)
            except OSError:
                continue  # port busy (another monitor?) — advisory, skip

    def poll(self) -> LidarHealth:
        """Drain a few pending datagrams per socket and update health."""
        now_ns = time.time_ns()
        for s in self._socks:
            for _ in range(4):  # bounded drain per poll — never a hot loop
                try:
                    data, _addr = s.recvfrom(2048)
                except (BlockingIOError, OSError):
                    break
                port = s.getsockname()[1]
                kind = ports.classify_port(port)
                if kind is None:
                    continue
                self._packets += 1
                self._last_seen_ns = now_ns
                self._lidar_type = kind
                if port == ports.ROBOSENSE_MSOP_PORT and data[:4] == ports.ROBOSENSE_MSOP_MAGIC:
                    temp = parse_robosense_temp_c(data)
                    if temp is not None:
                        self._temp_c = temp
        present = (
            self._last_seen_ns is not None
            and (now_ns - self._last_seen_ns) < STALE_S * 1e9
        )
        return LidarHealth(
            present=present,
            lidar_type=self._lidar_type if present else None,
            temp_c=self._temp_c if present else None,
            last_seen_unix_ns=self._last_seen_ns,
            packets_seen=self._packets,
        )

    def close(self) -> None:
        for s in self._socks:
            s.close()
        self._socks.clear()


def synth_msop_packet(temp_c: float = 40.0, seq: int = 0, unix_ns: int | None = None) -> bytes:
    """A RoboSense-Airy-shaped MSOP packet (magic + temp + optional capture time) for
    tests/mock pcap. Pass ``unix_ns`` to stamp the MSOP timestamp so the server's
    lidar_clock pcap-arrival skew comes out ~0 (a faithful, in-sync mock lidar)."""
    pkt = bytearray(ports.ROBOSENSE_MSOP_LEN)
    pkt[0:4] = ports.ROBOSENSE_MSOP_MAGIC
    pkt[8:12] = struct.pack(">I", seq)
    if unix_ns is not None:
        sec, us = unix_ns // 1_000_000_000, (unix_ns % 1_000_000_000) // 1000
        pkt[_MSOP_TS_OFFSET : _MSOP_TS_OFFSET + 6] = sec.to_bytes(6, "big")
        pkt[_MSOP_TS_OFFSET + 6 : _MSOP_TS_OFFSET + 10] = us.to_bytes(4, "big")
    value = int(round(temp_c / ports.ROBOSENSE_TEMP_RES_C))
    neg = value < 0
    value = abs(value)
    pkt[_MSOP_TEMP_OFFSET] = (value & 0x1F) << 3
    pkt[_MSOP_TEMP_OFFSET + 1] = ((value >> 5) & 0x7F) | (0x80 if neg else 0)
    return bytes(pkt)


def synth_imu_packet() -> bytes:
    """A RoboSense-Airy-shaped IMU packet (magic + length) for tests/mock pcap."""
    pkt = bytearray(ports.ROBOSENSE_IMU_LEN)
    pkt[0:4] = ports.ROBOSENSE_IMU_MAGIC
    return bytes(pkt)
