"""RoboSense Airy DIFOP factory IMU->lidar extrinsic extraction.

Each Airy carries a per-UNIT factory calibration of the transform between its built-in
IMU frame and its lidar frame, broadcast (~1 Hz) in its DIFOP packet. GLIM needs this as
``T_lidar_imu`` — feed it the wrong unit's value and the IMU prior fights scan matching
(~20 cm position kicks per 90 deg turn were observed). For a multi-lidar rig each lidar
must get its OWN value, so we read it from each episode's own pcap here.

Wire layout (verified against RoboSense-LiDAR/rs_driver ``decoder_RSAIRY.hpp``,
``RSAIRYDifopPkt``; the 1248-byte struct's offsets self-check to exactly 1248):
  - offset 0    : 8-byte magic ``A5 FF 00 5A 11 11 55 55`` (ports.ROBOSENSE_DIFOP_MAGIC)
  - offset 1092 : ``qx, qy, qz, qw`` as big-endian IEEE-754 float32 (16 B)
  - offset 1108 : ``x, y, z`` translation in METERS, big-endian float32 (12 B)
rs_driver reads these straight into ``DeviceInfo`` with no matrix math, so they already
ARE GLIM's TUM ``T_lidar_imu = [x, y, z, qx, qy, qz, qw]`` — no conversion needed. (The
separate pitch/yaw/roll fields at 1124+ are a DEGREE point-cloud transform, NOT the IMU
extrinsic — ignored here.)
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

from robot_umi_core import ports
from robot_umi_core.pcapio import read_udp_packets

# The hardcoded fallback (a known-good unit's extrinsic), used when a pcap carries no
# parseable DIFOP (old captures, mock data) and no config override is set. This is the
# single source of truth for that literal and the GLIM container configuration
# config_sensors.json ``T_lidar_imu`` must match it.
DEFAULT_AIRY_T_LIDAR_IMU: tuple[float, ...] = (
    0.00425, 0.00418, -0.00446, 0.707999, -0.706198, 0.003362, 0.003284,
)

_QUAT_OFFSET = 1092  # qx, qy, qz, qw (BE float32)
_TRANS_OFFSET = 1108  # x, y, z (BE float32, meters)


def parse_difop_extrinsic(payload: bytes) -> tuple[float, ...] | None:
    """Parse one DIFOP payload's extrinsic into GLIM TUM ``[x,y,z,qx,qy,qz,qw]``.

    Returns None (never raises) if the payload is too short, isn't a DIFOP packet, or
    carries a degenerate (zero-norm) quaternion — so callers can fall back cleanly."""
    if len(payload) < ports.ROBOSENSE_DIFOP_LEN:
        return None
    if payload[: len(ports.ROBOSENSE_DIFOP_MAGIC)] != ports.ROBOSENSE_DIFOP_MAGIC:
        return None
    qx, qy, qz, qw = struct.unpack_from(">4f", payload, _QUAT_OFFSET)
    x, y, z = struct.unpack_from(">3f", payload, _TRANS_OFFSET)
    vals = (x, y, z, qx, qy, qz, qw)
    if not all(math.isfinite(v) for v in vals):
        return None
    if math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) < 1e-6:
        return None  # never-configured / zeroed DIFOP
    return vals


def extract_t_lidar_imu(pcap_path: Path) -> tuple[float, ...] | None:
    """The first parseable RoboSense DIFOP extrinsic in ``pcap_path``, or None.

    None means "no usable DIFOP" (mock, truncated, corrupt, or unreadable pcaps) —
    the resolver then falls back to a config override or the hardcoded default. File-level
    errors (bad pcap magic, unsupported linktype, permission denied) degrade to None here,
    not an exception, so a bad capture can never block the SLAM stage."""
    try:
        for pkt in read_udp_packets(pcap_path):
            if pkt.dst_port != ports.ROBOSENSE_DIFOP_PORT:
                continue
            ext = parse_difop_extrinsic(pkt.payload)
            if ext is not None:
                return ext
    except (ValueError, OSError):
        return None
    return None


def synth_difop_packet(
    t_lidar_imu: tuple[float, ...] = DEFAULT_AIRY_T_LIDAR_IMU,
) -> bytes:
    """A RoboSense-Airy-shaped DIFOP payload carrying ``t_lidar_imu`` (for tests/mock).

    The inverse of :func:`parse_difop_extrinsic`: values round-trip to float32 precision."""
    if len(t_lidar_imu) != 7:
        raise ValueError("t_lidar_imu must be [x,y,z,qx,qy,qz,qw]")
    x, y, z, qx, qy, qz, qw = t_lidar_imu
    pkt = bytearray(ports.ROBOSENSE_DIFOP_LEN)
    pkt[: len(ports.ROBOSENSE_DIFOP_MAGIC)] = ports.ROBOSENSE_DIFOP_MAGIC
    struct.pack_into(">4f", pkt, _QUAT_OFFSET, qx, qy, qz, qw)
    struct.pack_into(">3f", pkt, _TRANS_OFFSET, x, y, z)
    return bytes(pkt)
