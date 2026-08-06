"""Lidar wire constants: UDP ports, packet sizes, magic bytes.

Used by the device (BPF capture filter, port-activity monitor, DIFOP temperature peek)
and by the server (offline type-by-port backend selection when decoding pcaps).

Values verified on hardware (RoboSense Airy bring-up, 2026-07) and against vendor docs.
"""

from __future__ import annotations

from dataclasses import dataclass

# --- RoboSense (Airy defaults; per-unit reconfigurable via the web UI / DIFOP) -------
ROBOSENSE_MSOP_PORT = 6699  # point cloud, 1248 B packets, ~2240 pkt/s (Airy 96-beam)
ROBOSENSE_DIFOP_PORT = 7788  # device info (temperature, rpm, calibration), ~1 Hz
ROBOSENSE_IMU_PORT = 6688  # built-in IMU, 51 B packets, 200 Hz
ROBOSENSE_MSOP_LEN = 1248
ROBOSENSE_DIFOP_LEN = 1248
ROBOSENSE_IMU_LEN = 51
ROBOSENSE_IMU_MAGIC = bytes.fromhex("aa555a05")
ROBOSENSE_MSOP_MAGIC = bytes.fromhex("55aa055a")
ROBOSENSE_DIFOP_MAGIC = bytes.fromhex("a5ff005a11115555")
# NOTE (advisory telemetry only): rs_driver parses MSOP temperature from the header
# with resolution 0.0625 C. The device peeks it for the LCD; the decoder re-derives it.
ROBOSENSE_TEMP_RES_C = 0.0625

# Factory network defaults (per-unit reprovision needed for 2 lidars on one Pi).
ROBOSENSE_DEFAULT_LIDAR_IP = "192.168.1.200"
ROBOSENSE_DEFAULT_HOST_IP = "192.168.1.102"

LIDAR_TYPE_ROBOSENSE = "robosense"


@dataclass(frozen=True)
class PortProfile:
    """The UDP ports one lidar unit occupies on its host interface."""

    lidar_type: str
    ports: tuple[int, ...]


ROBOSENSE_PROFILE = PortProfile(
    LIDAR_TYPE_ROBOSENSE,
    (ROBOSENSE_MSOP_PORT, ROBOSENSE_DIFOP_PORT, ROBOSENSE_IMU_PORT),
)
def classify_port(dst_port: int) -> str | None:
    """Lidar type implied by a destination UDP port, or None if unrecognized."""
    if dst_port in ROBOSENSE_PROFILE.ports:
        return LIDAR_TYPE_ROBOSENSE
    return None


def bpf_filter(host_ip: str) -> str:
    """The RoboSense Airy capture filter for one lidar interface."""
    rs = " or ".join(f"port {p}" for p in ROBOSENSE_PROFILE.ports)
    return f"udp and dst host {host_ip} and ({rs})"
