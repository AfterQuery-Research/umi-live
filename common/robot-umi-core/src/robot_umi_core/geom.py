"""Pure-numpy pose geometry shared by GLIM processing stages."""

from __future__ import annotations

import numpy as np


def quat_normalize(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    return q / norm if norm else np.array([0.0, 0.0, 0.0, 1.0])


def quat_normalize_strict(q: np.ndarray) -> np.ndarray:
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


# Lidar body coordinates -> UMI coordinates: up=+X, right=+Y, forward=+Z.
UMI_FROM_LIDAR_BODY = np.array(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)

# Lidar body -> gripper end-effector lever in UMI coordinates (metres).
LIDAR_EE_LEVER_M = np.array([-104.79, 0.0, 222.44]) / 1000.0


def pose_multiply(
    p_a: np.ndarray, q_a: np.ndarray, p_b: np.ndarray, q_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compose two SE(3) poses represented as position and quaternion."""
    return p_a + quat_rotate(q_a, p_b), quat_multiply(q_a, q_b)


def pose_inverse(p: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q_inv = quat_conjugate(quat_normalize(q))
    return -quat_rotate(q_inv, p), q_inv
