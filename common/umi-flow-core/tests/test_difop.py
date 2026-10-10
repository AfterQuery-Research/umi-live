"""RoboSense Airy DIFOP extrinsic parser tests.

The acceptance test pins the byte layout against an INDEPENDENT source: the exact
big-endian float32 wire bytes derived from the rs_driver ``RSAIRYDifopPkt`` struct for a
known-good unit. If the parser reproduces the known-good ``T_lidar_imu`` from those raw
bytes, the offset (1092/1108) + endianness are correct.
"""

from __future__ import annotations

import numpy as np
import pytest
from umi_flow_core import difop, ports

# Known-good unit's extrinsic, GLIM TUM [x, y, z, qx, qy, qz, qw].
KNOWN = difop.DEFAULT_AIRY_T_LIDAR_IMU

# The same value as it appears on the wire (rs_driver: qx,qy,qz,qw at 1092; x,y,z at 1108;
# each big-endian IEEE-754 float32). Bytes from decoding the known-good value to float32.
_WIRE_QUAT = bytes.fromhex("3f353f6cbf34c9643b5c55003b573861")  # qx qy qz qw
_WIRE_TRANS = bytes.fromhex("3b8b43963b88f862bb922531")  # x y z


def _known_wire_payload() -> bytes:
    pkt = bytearray(ports.ROBOSENSE_DIFOP_LEN)
    pkt[: len(ports.ROBOSENSE_DIFOP_MAGIC)] = ports.ROBOSENSE_DIFOP_MAGIC
    pkt[1092 : 1092 + 16] = _WIRE_QUAT
    pkt[1108 : 1108 + 12] = _WIRE_TRANS
    return bytes(pkt)


def test_difop_acceptance_wire_bytes():
    # Independent-source acceptance: raw rs_driver-layout bytes -> the known extrinsic.
    ext = difop.parse_difop_extrinsic(_known_wire_payload())
    assert ext is not None
    assert np.allclose(ext, KNOWN, atol=1e-5)


def test_difop_synth_roundtrip():
    # synth is the inverse of parse; agreement with the wire-bytes test cross-confirms it.
    ext = difop.parse_difop_extrinsic(difop.synth_difop_packet(KNOWN))
    assert np.allclose(ext, KNOWN, atol=1e-5)
    assert difop.synth_difop_packet(KNOWN)[1092 : 1092 + 16] == _WIRE_QUAT
    assert difop.synth_difop_packet(KNOWN)[1108 : 1108 + 12] == _WIRE_TRANS


def test_difop_synth_roundtrip_arbitrary():
    ext_in = (0.01, -0.02, 0.03, 0.0, 0.0, 0.0, 1.0)  # identity rotation
    ext = difop.parse_difop_extrinsic(difop.synth_difop_packet(ext_in))
    assert np.allclose(ext, ext_in, atol=1e-6)


def test_difop_rejects_non_difop():
    assert difop.parse_difop_extrinsic(b"\x00" * ports.ROBOSENSE_DIFOP_LEN) is None  # no magic
    assert difop.parse_difop_extrinsic(b"") is None  # too short
    # right magic + length but a zeroed (degenerate) quaternion.
    zeroed = bytearray(ports.ROBOSENSE_DIFOP_LEN)
    zeroed[: len(ports.ROBOSENSE_DIFOP_MAGIC)] = ports.ROBOSENSE_DIFOP_MAGIC
    assert difop.parse_difop_extrinsic(bytes(zeroed)) is None


def test_synth_difop_rejects_bad_length():
    with pytest.raises(ValueError):
        difop.synth_difop_packet((1.0, 2.0, 3.0))
