"""Recover the lidar-clock -> host-clock skew for trajectory time alignment.

GLIM trajectories are timestamped in the LIDAR's own clock (the MSOP payload capture
time). When the lidar's PTP discipline to the host lags — e.g. right after chrony steps
the Pi's system clock — that clock can be many seconds off the host/camera clock, so the
trajectory won't overlap the camera timeline and every pose clamps to one value.

Every MSOP packet carries BOTH its lidar capture time (in the payload) AND its host
arrival time (the pcap/tcpdump stamp), so the constant skew is directly measurable:
``skew = median(host_arrival - lidar_payload)``. Adding it to the trajectory timestamps
re-expresses them on the host clock — the same clock the camera frames use — regardless of
how far the lidar's PTP has drifted. (This is the "pcap-arrival time-align" trick.)

RSAIRY MSOP timestamp (verified against RoboSense-LiDAR/rs_driver decoder_RSAIRY.hpp):
payload offset 20, 6 big-endian bytes of unix SECONDS + 4 big-endian bytes of MICROseconds.
"""

from __future__ import annotations

from pathlib import Path
from statistics import median

from robot_umi_core import ports
from robot_umi_core.pcapio import read_udp_packets

_MSOP_TS_OFFSET = 20  # RSAIRY: 6B BE seconds + 4B BE microseconds


def msop_lidar_time(payload: bytes) -> float | None:
    """The lidar capture time (unix seconds) from one RoboSense MSOP payload, or None."""
    if len(payload) < _MSOP_TS_OFFSET + 10:
        return None
    if payload[: len(ports.ROBOSENSE_MSOP_MAGIC)] != ports.ROBOSENSE_MSOP_MAGIC:
        return None
    sec = int.from_bytes(payload[_MSOP_TS_OFFSET : _MSOP_TS_OFFSET + 6], "big")
    us = int.from_bytes(payload[_MSOP_TS_OFFSET + 6 : _MSOP_TS_OFFSET + 10], "big")
    return sec + us / 1e6


def pcap_host_skew(pcap_path: Path, max_packets: int = 4000) -> float | None:
    """``median(host_arrival - lidar_payload)`` over MSOP packets (seconds), or None.

    Add the result to a lidar-clock timestamp to move it onto the host clock. Returns
    None (caller leaves timestamps untouched) if the pcap has no parseable MSOP packets —
    non-RoboSense captures, mocks, or truncated files."""
    skews: list[float] = []
    try:
        for pkt in read_udp_packets(pcap_path):
            if pkt.dst_port != ports.ROBOSENSE_MSOP_PORT:
                continue
            lt = msop_lidar_time(pkt.payload)
            if lt is None:
                continue
            skews.append(pkt.ts_unix_ns / 1e9 - lt)
            if len(skews) >= max_packets:
                break
    except (ValueError, OSError):
        return None
    if not skews:
        return None
    return median(skews)
