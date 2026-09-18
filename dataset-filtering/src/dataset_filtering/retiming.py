from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TimeParameterization:
    timestamps: np.ndarray
    method: str
    metadata: dict[str, Any]


def time_parameterize(
    q: np.ndarray,
    source_timestamps: np.ndarray,
    velocity_limits_rad_s: np.ndarray,
    acceleration_limits_rad_s2: np.ndarray,
    *,
    prefer_toppra: bool = True,
) -> TimeParameterization:
    """Preserve source timing when feasible, otherwise compute a safe monotonic warp.

    TOPP-RA is optional because its published wheels do not support macOS ARM. The
    fallback is a conservative uniform time dilation derived from the measured peak
    velocity and acceleration; it changes timing but never the optimized geometric path.
    """
    q = np.asarray(q, dtype=np.float64)
    source = np.asarray(source_timestamps, dtype=np.float64)
    velocity_limits = np.asarray(velocity_limits_rad_s, dtype=np.float64).reshape(-1)
    acceleration_limits = np.asarray(acceleration_limits_rad_s2, dtype=np.float64).reshape(-1)
    if q.ndim != 2 or q.shape[0] != len(source):
        raise ValueError("q must have shape (N, D) on the source timestamp timeline")
    if q.shape[1] != len(velocity_limits) or q.shape[1] != len(acceleration_limits):
        raise ValueError("one velocity and acceleration limit is required per joint")
    if len(source) < 2 or np.any(np.diff(source) <= 0):
        raise ValueError("source timestamps must be strictly increasing")

    diagnostics = trajectory_dynamics(q, source)
    velocity_ratio = float(np.max(np.abs(diagnostics["velocity_rad_s"]) / velocity_limits[None, :]))
    acceleration_ratio = float(
        np.max(np.abs(diagnostics["acceleration_rad_s2"]) / acceleration_limits[None, :])
    )
    if velocity_ratio <= 1.0 + 1e-9 and acceleration_ratio <= 1.0 + 1e-9:
        return TimeParameterization(
            timestamps=source.copy(),
            method="original_timing",
            metadata={
                "time_scale": 1.0,
                "peak_velocity_limit_ratio_before": velocity_ratio,
                "peak_acceleration_limit_ratio_before": acceleration_ratio,
                "source_duration_s": float(source[-1] - source[0]),
                "robot_duration_s": float(source[-1] - source[0]),
            },
        )

    if prefer_toppra:
        try:
            return _time_parameterize_with_toppra(q, source, velocity_limits, acceleration_limits)
        except (ImportError, RuntimeError, ValueError) as exc:
            fallback_reason = f"{type(exc).__name__}: {exc}"
    else:
        fallback_reason = "TOPP-RA disabled"

    scale = float(max(1.0, velocity_ratio, np.sqrt(acceleration_ratio)) * 1.001)
    robot = source[0] + (source - source[0]) * scale
    after = trajectory_dynamics(q, robot)
    return TimeParameterization(
        timestamps=robot,
        method="uniform_time_scale_fallback",
        metadata={
            "time_scale": scale,
            "peak_velocity_limit_ratio_before": velocity_ratio,
            "peak_acceleration_limit_ratio_before": acceleration_ratio,
            "peak_velocity_limit_ratio_after": float(
                np.max(np.abs(after["velocity_rad_s"]) / velocity_limits[None, :])
            ),
            "peak_acceleration_limit_ratio_after": float(
                np.max(np.abs(after["acceleration_rad_s2"]) / acceleration_limits[None, :])
            ),
            "source_duration_s": float(source[-1] - source[0]),
            "robot_duration_s": float(robot[-1] - robot[0]),
            "fallback_reason": fallback_reason,
        },
    )


def trajectory_dynamics(q: np.ndarray, timestamps: np.ndarray) -> dict[str, np.ndarray]:
    q = np.asarray(q, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    edge_order = 2 if len(timestamps) >= 3 else 1
    velocity = np.gradient(q, timestamps, axis=0, edge_order=edge_order)
    acceleration = np.gradient(velocity, timestamps, axis=0, edge_order=edge_order)
    jerk = np.gradient(acceleration, timestamps, axis=0, edge_order=edge_order)
    return {
        "velocity_rad_s": velocity,
        "acceleration_rad_s2": acceleration,
        "jerk_rad_s3": jerk,
    }


def _time_parameterize_with_toppra(
    q: np.ndarray,
    source: np.ndarray,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
) -> TimeParameterization:
    """Use TOPP-RA when installed; reconstruct time at each original path sample."""
    try:
        import toppra as ta
        import toppra.algorithm as algo
        import toppra.constraint as constraint
    except ImportError as exc:  # pragma: no cover - platform-dependent optional package
        raise ImportError("install the 'retiming' extra on Linux for TOPP-RA") from exc

    progress = (source - source[0]) / (source[-1] - source[0])
    # Add source elapsed time as a virtual path coordinate. Its unit velocity
    # limit prevents the retimer from advancing through the demonstration faster
    # than the recording. TOPP-RA may introduce local slowdowns, but it cannot
    # turn a 40 s demonstration into a time-optimal 18 s robot motion.
    source_elapsed = source - source[0]
    augmented_path = np.column_stack((q, source_elapsed))
    augmented_velocity_limits = np.concatenate((velocity_limits, [1.0]))
    augmented_acceleration_limits = np.concatenate((acceleration_limits, [1e6]))
    path = ta.SplineInterpolator(progress, augmented_path)
    constraints = [
        constraint.JointVelocityConstraint(augmented_velocity_limits),
        constraint.JointAccelerationConstraint(augmented_acceleration_limits),
    ]
    instance = algo.TOPPRA(constraints, path, gridpoints=progress, solver_wrapper="seidel")
    try:
        parameterization = instance.compute_parameterization(0.0, 0.0)
    except Exception as exc:  # pragma: no cover - native solver failure
        raise RuntimeError(f"TOPP-RA could not parameterize this path: {exc}") from exc
    if parameterization is None or len(parameterization) < 2:
        raise RuntimeError("TOPP-RA returned no path parameterization")
    # TOPP-RA 0.6 returns (path_acceleration, path_speed, auxiliary). Older
    # releases returned a fourth diagnostic array when requested. Only the
    # path-speed profile is needed to recover timestamps at the source samples.
    speed_raw = parameterization[1]
    if speed_raw is None:
        raise RuntimeError("TOPP-RA could not find a feasible path-speed profile")
    speed = np.asarray(speed_raw, dtype=np.float64)
    if len(speed) != len(progress) or not np.all(np.isfinite(speed)):
        raise RuntimeError("TOPP-RA returned an invalid path-speed profile")
    ds = np.diff(progress)
    average_speed = 0.5 * (speed[:-1] + speed[1:])
    if np.any(average_speed <= 1e-12):
        # Rest-to-rest profiles have zero endpoint speeds; use the constant-acceleration
        # identity dt = 2 ds / (sd_i + sd_{i+1}), which remains finite unless an entire
        # segment has zero speed.
        if np.any((speed[:-1] + speed[1:]) <= 1e-12):
            raise RuntimeError("TOPP-RA returned a zero-speed path segment")
    dt = 2.0 * ds / np.maximum(speed[:-1] + speed[1:], 1e-12)
    robot = source[0] + np.concatenate(([0.0], np.cumsum(dt)))
    diagnostics = trajectory_dynamics(q, robot)
    raw_velocity_ratio = float(
        np.max(np.abs(diagnostics["velocity_rad_s"]) / velocity_limits[None, :])
    )
    raw_acceleration_ratio = float(
        np.max(np.abs(diagnostics["acceleration_rad_s2"]) / acceleration_limits[None, :])
    )
    # TOPP-RA constrains its continuous interpolating spline. Validation operates
    # on the exported 30 Hz samples, whose finite-difference acceleration can be
    # a few tenths of a percent higher. Apply a tiny uniform safety dilation so
    # the artifact that will actually be replayed also passes sampled limits.
    sampled_safety_scale = float(
        max(1.0, raw_velocity_ratio, np.sqrt(raw_acceleration_ratio)) * 1.001
    )
    robot = source[0] + (robot - source[0]) * sampled_safety_scale
    diagnostics = trajectory_dynamics(q, robot)
    source_clock_rate = np.diff(source) / np.diff(robot)
    return TimeParameterization(
        timestamps=robot,
        method="toppra",
        metadata={
            "time_scale": float((robot[-1] - robot[0]) / (source[-1] - source[0])),
            "sampled_safety_scale": sampled_safety_scale,
            "toppra_peak_velocity_limit_ratio_raw": raw_velocity_ratio,
            "toppra_peak_acceleration_limit_ratio_raw": raw_acceleration_ratio,
            "maximum_source_clock_rate": float(np.max(source_clock_rate)),
            "source_clock_rate_limit": 1.0,
            "peak_velocity_limit_ratio_after": float(
                np.max(np.abs(diagnostics["velocity_rad_s"]) / velocity_limits[None, :])
            ),
            "peak_acceleration_limit_ratio_after": float(
                np.max(np.abs(diagnostics["acceleration_rad_s2"]) / acceleration_limits[None, :])
            ),
            "source_duration_s": float(source[-1] - source[0]),
            "robot_duration_s": float(robot[-1] - robot[0]),
        },
    )
