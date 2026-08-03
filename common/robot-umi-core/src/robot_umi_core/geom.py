"""Pure-numpy pose geometry shared by device (LCD live readout) and server (dataset math).

Consolidates the proven robot_umi modules: quaternion primitives (pose_zero), the
OpenVR->body conversion + multi-tracker parsing (vive), and the EE<-Vive rigid calib
(pipeline). Quaternions are ``[x, y, z, w]`` unless a name says ``wxyz``.

The device applies NONE of this to recorded data (record raw, process later) — it may
import it only for advisory display math. The server applies EE calibration from the
episode-shipped calib file.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]
# (x, y, z, roll, pitch, yaw): position in metres, orientation in degrees.
Pose6 = tuple[float, float, float, float, float, float]
# Body-frame tracker pose: (position[3], quaternion_xyzw[4]).
TrackerPose = tuple[np.ndarray, np.ndarray]

# --- quaternion primitives ----------------------------------------------------------


def quat_normalize(q: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(q))
    return q / n if n else np.array([0.0, 0.0, 0.0, 1.0])


def quat_normalize_strict(q: np.ndarray) -> np.ndarray:
    """L2-normalize, rejecting non-finite / zero-norm input."""
    if not np.isfinite(q).all():
        raise ValueError("quaternion must contain only finite values")
    norm = np.linalg.norm(q)
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("quaternion has zero or non-finite norm")
    return q / norm


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.array([v[0], v[1], v[2], 0.0])
    return quat_multiply(quat_multiply(q, qv), quat_conjugate(q))[:3]


def quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize_strict(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat_xyzw(m: np.ndarray) -> np.ndarray:
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return quat_normalize_strict(np.array([x, y, z, w], dtype=np.float64))


def quat_to_euler_deg(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    """(roll, pitch, yaw) degrees, ZYX Tait-Bryan. Display use; naive near gimbal lock."""
    n = float(np.sqrt(x * x + y * y + z * z + w * w)) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw))


def euler_deg(q: np.ndarray) -> tuple[float, float, float]:
    """ZYX roll/pitch/yaw (deg) from a normalized quat [x,y,z,w], gimbal-lock-safe.

    Near |pitch| = 90 deg the ZYX roll and yaw are degenerate; fold them into a single
    well-defined angle (roll = 0, everything in yaw) so a live readout stays stable.
    """
    x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    sinp = 2.0 * (w * y - z * x)
    if sinp > 0.99999:  # pitch ~ +90 deg
        return 0.0, 90.0, float(np.degrees(-2.0 * np.arctan2(x, w)))
    if sinp < -0.99999:  # pitch ~ -90 deg
        return 0.0, -90.0, float(np.degrees(2.0 * np.arctan2(x, w)))
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(sinp)
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return float(np.degrees(roll)), float(np.degrees(pitch)), float(np.degrees(yaw))


# UMI body convention: forward = +Z, right = +Y, up = +X (matches the Vive EE
# convention from the previous rig, so lidar- and vive-sourced states agree).
# Maps episode-relative lidar/GLIM body-frame coordinates (recalibrated on the
# Airy mount 2026-07-16 AFTER the factory IMU extrinsic fix — the original
# calibration was taken against GLIM output with a flipped IMU frame, which
# inverted up and forward: forward=-Y, right=-X, up=+Z) into the UMI
# convention:
#   X_umi = +z,  Y_umi = -x,  Z_umi = -y   (det = +1, proper rotation)
# Consistent with the measured cloud frame: all points z >= 0, ceiling at +Z,
# so up really is +z in the lidar body frame.
UMI_FROM_LIDAR_BODY = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)

# Lidar body -> gripper end-effector (TCP) lever, in the UMI convention (up, right, fwd):
# the RoboSense lidar bottom-face sits 104.79 mm above and 222.44 mm behind the TCP, so the
# EE is 104.79 mm DOWN + 222.44 mm FORWARD of the lidar. Applied as ee = pos + R @ lever so a
# pure gripper rotation pivots about the TCP, not the lidar. Same value the mac viewers use.
LIDAR_EE_LEVER_M = np.array([-104.79, 0.0, 222.44]) / 1000.0

# --- SE(3) on (position, quat_xyzw) pairs --------------------------------------------


def pose_multiply(
    p_a: np.ndarray, q_a: np.ndarray, p_b: np.ndarray, q_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compose a_T_b: (p, q) of the transform ``a . b``."""
    return p_a + quat_rotate(q_a, p_b), quat_multiply(q_a, q_b)


def pose_inverse(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q_inv = quat_conjugate(quat_normalize(q))
    return -quat_rotate(q_inv, p), q_inv


# --- OpenVR -> body-frame conversion + tracker parsing -------------------------------

# OpenVR: +X right, +Y up, -Z forward.  REP-103 body: +X forward, +Y left, +Z up.
OPENVR_TO_ROS = np.array(
    [
        [0.0, 0.0, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

# Tracker body-axis correction (180 deg about X), right-multiplied on the BODY side so
# body-relative poses downstream read forward +Z / right +Y / up +X.
R_BODY = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class VivePose:
    serial: str
    position: np.ndarray
    orientation_xyzw: np.ndarray


def openvr_pose_to_ros(pos: Any, quat_wxyz: Any) -> TrackerPose:
    """Convert an OpenVR (pos, quat_wxyz) to a body-frame (pos, quat_xyzw) pose."""
    p_vr = np.asarray(pos, dtype=np.float64)
    q_vr = np.asarray(quat_wxyz, dtype=np.float64)
    if p_vr.shape != (3,):
        raise ValueError(f"pos must have 3 values, got {p_vr.shape}")
    if q_vr.shape != (4,):
        raise ValueError(f"quat must have 4 values, got {q_vr.shape}")
    if not np.isfinite(p_vr).all():
        raise ValueError("pos must contain only finite values")
    if not np.isfinite(q_vr).all():
        raise ValueError("quat must contain only finite values")

    p_ros = OPENVR_TO_ROS @ p_vr
    r_vr = quat_wxyz_to_matrix(q_vr)
    r_ros = OPENVR_TO_ROS @ r_vr @ OPENVR_TO_ROS.T @ R_BODY
    return p_ros, matrix_to_quat_xyzw(r_ros)


def _trackers(packet: Any) -> list[Any]:
    if not isinstance(packet, dict):
        raise ValueError("packet must be a JSON object")
    trackers = packet.get("trackers", [])
    if not isinstance(trackers, list):
        raise ValueError("packet['trackers'] must be a list")
    return trackers


def _tracker_pose_fields(tracker: Any) -> tuple[str, Any, Any] | None:
    """(serial, pos, quat) for a valid, well-formed tracker; ``None`` to skip it.

    A single malformed tracker must NOT crash the whole packet — one bad entry would
    otherwise drop every good tracker in the same UDP frame.
    """
    if not isinstance(tracker, dict):
        raise ValueError("tracker entry must be a JSON object")
    if not tracker.get("valid"):
        return None
    serial, pos, quat = tracker.get("serial"), tracker.get("pos"), tracker.get("quat")
    if serial is None or pos is None or quat is None:
        return None
    return str(serial), pos, quat


def valid_tracker_serials(packet: Any) -> list[str]:
    serials = []
    for tracker in _trackers(packet):
        fields = _tracker_pose_fields(tracker)
        if fields is not None:
            serials.append(fields[0])
    return serials


def pose_from_packet(packet: Any, tracker_serial: str | None = None) -> VivePose | None:
    """First valid tracker (or the one matching ``tracker_serial``) as a body-frame pose."""
    for tracker in _trackers(packet):
        fields = _tracker_pose_fields(tracker)
        if fields is None:
            continue
        serial, pos, quat = fields
        if tracker_serial is not None and serial != tracker_serial:
            continue
        position, orientation = openvr_pose_to_ros(pos, quat)
        return VivePose(serial=serial, position=position, orientation_xyzw=orientation)
    return None


def parse_trackers(packet: Any) -> dict[str, TrackerPose]:
    """Every valid tracker in the packet as {serial: (pos, quat_xyzw)} in the body frame."""
    poses: dict[str, TrackerPose] = {}
    for tracker in _trackers(packet):
        fields = _tracker_pose_fields(tracker)
        if fields is None:
            continue
        serial, pos, quat = fields
        poses[serial] = openvr_pose_to_ros(pos, quat)
    return poses


# --- live tare (display only; recorded data is never zeroed on the device) ----------


class PoseZeroer:
    """Tare the live pose to a captured origin (SE(3)); 6-DoF relative output."""

    def __init__(self) -> None:
        self._ref: tuple[np.ndarray, np.ndarray] | None = None
        self._request = False

    def request_zero(self) -> None:
        self._request = True

    def relative(self, pos: Vec3, quat: Quat) -> Pose6:
        p = np.asarray(pos, dtype=np.float64)
        q = quat_normalize(np.asarray(quat, dtype=np.float64))
        if self._request:
            self._ref = (p, q)
            self._request = False
        if self._ref is None:
            roll, pitch, yaw = quat_to_euler_deg(q[0], q[1], q[2], q[3])
            return float(p[0]), float(p[1]), float(p[2]), roll, pitch, yaw
        p0, q0_inv = self._ref[0], quat_conjugate(self._ref[1])
        rel_p = quat_rotate(q0_inv, p - p0)
        rel_q = quat_multiply(q0_inv, q)
        roll, pitch, yaw = quat_to_euler_deg(rel_q[0], rel_q[1], rel_q[2], rel_q[3])
        return float(rel_p[0]), float(rel_p[1]), float(rel_p[2]), roll, pitch, yaw


class MultiViveZeroer:
    """Tare every UMI to ONE shared reference frame captured from a chosen UMI.

    After ``request_zero(serial)``, the next :meth:`relative` snapshots that UMI's pose
    as origin ``F0`` and thereafter returns every UMI's 6-DoF pose as ``F0^-1 . pose``.
    Before the first zero, absolute poses are returned.
    """

    def __init__(self) -> None:
        self._frame: tuple[np.ndarray, np.ndarray] | None = None
        self._ref_serial: str | None = None
        self._request: str | None = None

    @property
    def reference(self) -> str | None:
        return self._ref_serial

    def request_zero(self, serial: str) -> None:
        self._request = serial

    def relative(self, poses: dict[str, TrackerPose]) -> dict[str, Pose6]:
        if self._request is not None and self._request in poses:
            p0, q0 = poses[self._request]
            self._frame = (
                np.asarray(p0, dtype=np.float64),
                quat_normalize(np.asarray(q0)),
            )
            self._ref_serial = self._request
            self._request = None

        out: dict[str, Pose6] = {}
        if self._frame is None:
            for serial, (p, q) in poses.items():
                out[serial] = _to_pose6(
                    np.asarray(p, dtype=np.float64), quat_normalize(np.asarray(q))
                )
            return out

        p0, q0_inv = self._frame[0], quat_conjugate(self._frame[1])
        for serial, (p, q) in poses.items():
            rel_p = quat_rotate(q0_inv, np.asarray(p, dtype=np.float64) - p0)
            rel_q = quat_multiply(q0_inv, quat_normalize(np.asarray(q)))
            out[serial] = _to_pose6(rel_p, rel_q)
        return out


def _to_pose6(p: np.ndarray, q: np.ndarray) -> Pose6:
    roll, pitch, yaw = euler_deg(q)
    return float(p[0]), float(p[1]), float(p[2]), roll, pitch, yaw


# --- EE <- Vive rigid calibration ----------------------------------------------------


def load_ee_calib(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """(translation_m, quat_xyzw) of T(ee <- vive) from a per-UMI JSON calib file.

    Strict: raises on a missing or malformed file. Hermetic episodes ship the calib
    they were recorded with, so a silent fallback would hide a real provisioning bug.
    """
    d = json.loads(Path(path).read_text())
    t = np.asarray(d["translation_mm"], dtype=np.float64) / 1000.0
    q = np.asarray(d["quat_xyzw"], dtype=np.float64)
    if t.shape != (3,) or q.shape != (4,):
        raise ValueError(f"malformed EE calib in {path}: shapes {t.shape}, {q.shape}")
    return t, q


def end_effector_poses_from_vive(
    vive_poses: np.ndarray, ee_from_vive_t: np.ndarray, ee_from_vive_q: np.ndarray
) -> np.ndarray:
    """Convert world_T_vive pose rows [x,y,z,qx,qy,qz,qw] to world_T_ee rows."""
    ee_poses = np.zeros_like(vive_poses, dtype=np.float64)
    if len(vive_poses) == 0:
        return ee_poses
    vive_from_ee_p, vive_from_ee_q = pose_inverse(ee_from_vive_t, ee_from_vive_q)
    for i, pose in enumerate(vive_poses):
        p, q = pose[:3], quat_normalize(pose[3:7])
        ee_p, ee_q = pose_multiply(p, q, vive_from_ee_p, vive_from_ee_q)
        ee_poses[i, :3] = ee_p
        ee_poses[i, 3:7] = quat_normalize(ee_q)
    return ee_poses
