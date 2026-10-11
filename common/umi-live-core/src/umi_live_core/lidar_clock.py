"""Recover the LiDAR-clock -> host-clock mapping for trajectory time alignment.

GLIM trajectories are timestamped in the LIDAR's own clock (the MSOP payload capture
time). When the lidar's PTP discipline to the host lags — e.g. right after chrony steps
the Pi's system clock — that clock can be many seconds off the host/camera clock, so the
trajectory won't overlap the camera timeline and every pose clamps to one value.

Every MSOP packet carries BOTH its LiDAR capture time (in the payload) AND its host
arrival time (the PCAP stamp). Fitting their affine relationship corrects constant offset
and an unlocked sensor's frequency error. The legacy constant-skew helper remains for
callers that need a quick diagnostic, but dataset construction uses the affine mapping.

RSAIRY MSOP timestamp (verified against RoboSense-LiDAR/rs_driver decoder_RSAIRY.hpp):
payload offset 20, 6 big-endian bytes of unix SECONDS + 4 big-endian bytes of MICROseconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np

from umi_live_core import ports
from umi_live_core.pcapio import read_udp_packets

_MSOP_TS_OFFSET = 20  # RSAIRY: 6B BE seconds + 4B BE microseconds
MIN_CLOCK_SAMPLES = 32
MIN_CLOCK_SPAN_S = 1.0
MAX_CLOCK_RATE_ERROR = 0.2
MAX_CLOCK_RESIDUAL_RANGE_S = 0.05


@dataclass(frozen=True)
class LidarClockMapping:
    """Affine conversion from the Airy's packet clock to the host/camera clock."""

    lidar_origin_s: float
    host_origin_s: float
    rate: float
    sample_count: int
    lidar_span_s: float
    residual_p01_s: float
    residual_p99_s: float

    def to_host(self, lidar_time_s: float | np.ndarray) -> float | np.ndarray:
        values = np.asarray(lidar_time_s, dtype=np.float64)
        mapped = self.host_origin_s + self.rate * (values - self.lidar_origin_s)
        return float(mapped) if values.ndim == 0 else mapped


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


def pcap_host_clock_mapping(
    pcap_path: Path,
    *,
    max_samples: int = 20_000,
) -> LidarClockMapping | None:
    """Fit ``host = origin + rate * (lidar - origin)`` from one raw PCAP.

    PCAP record timestamps are on the host/camera clock and every Airy MSOP packet
    carries the corresponding LiDAR clock reading. An affine mapping corrects both
    offset and frequency error. Evenly sampling the complete capture (rather than only
    its beginning) makes clock-rate estimation observable while keeping memory bounded.

    A nonlinear residual indicates a clock step or a lock-state transition. Those takes
    are rejected instead of silently applying one invalid mapping across the episode.
    """
    pairs: list[tuple[float, float]] = []
    try:
        for packet in read_udp_packets(pcap_path):
            if packet.dst_port != ports.ROBOSENSE_MSOP_PORT:
                continue
            lidar_time = msop_lidar_time(packet.payload)
            if lidar_time is not None:
                pairs.append((lidar_time, packet.ts_unix_ns / 1e9))
    except (ValueError, OSError):
        return None
    if len(pairs) < MIN_CLOCK_SAMPLES:
        return None

    samples = np.asarray(pairs, dtype=np.float64)
    if len(samples) > max_samples:
        indices = np.linspace(0, len(samples) - 1, max_samples, dtype=np.int64)
        samples = samples[indices]
    lidar = samples[:, 0]
    host = samples[:, 1]
    lidar_origin = float(np.median(lidar))
    host_center = float(np.median(host))
    centered_lidar = lidar - lidar_origin
    centered_host = host - host_center
    denominator = float(np.sum(centered_lidar * centered_lidar))
    lidar_span = float(np.ptp(lidar))
    if lidar_span < MIN_CLOCK_SPAN_S or denominator <= 0:
        return None

    rate = float(np.sum(centered_lidar * centered_host) / denominator)
    if abs(rate - 1.0) > MAX_CLOCK_RATE_ERROR:
        raise ValueError(f"implausible LiDAR clock rate {rate:.9f}")
    host_origin = float(np.median(host - rate * centered_lidar))
    residual = host - (host_origin + rate * centered_lidar)
    residual_p01, residual_p99 = np.quantile(residual, [0.01, 0.99]).tolist()
    residual_range = float(residual_p99 - residual_p01)
    if residual_range > MAX_CLOCK_RESIDUAL_RANGE_S:
        raise ValueError(
            "LiDAR clock is not affine over the episode "
            f"(p01-p99 residual range {residual_range * 1000:.1f} ms)"
        )
    return LidarClockMapping(
        lidar_origin_s=lidar_origin,
        host_origin_s=host_origin,
        rate=rate,
        sample_count=len(samples),
        lidar_span_s=lidar_span,
        residual_p01_s=float(residual_p01),
        residual_p99_s=float(residual_p99),
    )
