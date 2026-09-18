from __future__ import annotations

from typing import Any

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.spatial.transform import Rotation

from dataset_filtering.models import Episode, EpisodeKind, PoseEpisode


def robust_rms(values: np.ndarray, axis: int | tuple[int, ...] | None = None) -> np.ndarray:
    """RMS after clipping large transients with a median/MAD envelope."""
    values = np.asarray(values, dtype=np.float64)
    median = np.median(values, axis=axis, keepdims=True)
    absolute = np.abs(values - median)
    mad = np.median(absolute, axis=axis, keepdims=True)
    clipped = np.clip(values, median - 6.0 * (mad + 1e-12), median + 6.0 * (mad + 1e-12))
    return np.sqrt(np.mean(np.square(clipped), axis=axis))


def high_frequency_residual(values: np.ndarray, fps: float, cutoff_hz: float = 6.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 12 or fps <= 2 * cutoff_hz:
        return values - np.mean(values, axis=0, keepdims=True)
    sos = butter(3, cutoff_hz, btype="highpass", fs=fps, output="sos")
    return sosfiltfilt(sos, values, axis=0)


def rotation_step_angles(quaternions: np.ndarray) -> np.ndarray:
    rotations = Rotation.from_quat(np.asarray(quaternions).reshape(-1, 4))
    if len(rotations) < 2:
        return np.zeros(0, dtype=np.float64)
    relative = rotations[:-1].inv() * rotations[1:]
    if relative is NotImplemented:
        raise TypeError("could not compose rotations")
    return relative.magnitude()


def pose_noise_metrics(episode: PoseEpisode) -> dict[str, Any]:
    """Estimate motion noise without pretending a ground-truth trajectory exists.

    The 6 Hz high-pass residual is an operational noise estimate at 30 Hz. Step, jerk,
    and outlier metrics are reported too so real fast motions are not hidden inside one
    scalar. Results are per arm and aggregated.
    """
    fps = episode.fps
    arms: list[dict[str, float]] = []
    for arm in range(2):
        positions = episode.positions[:, arm]
        pos_step = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        pos_hf = high_frequency_residual(positions, fps)
        pos_acceleration = np.diff(positions, n=2, axis=0) * fps**2
        pos_jerk = np.diff(positions, n=3, axis=0) * fps**3

        rotations = Rotation.from_quat(episode.quaternions[:, arm])
        reference = rotations[0]
        relative = reference.inv() * rotations
        if relative is NotImplemented:
            raise TypeError("could not compose rotations")
        rotvec = relative.as_rotvec()
        rot_hf = high_frequency_residual(rotvec, fps)
        rot_step = rotation_step_angles(episode.quaternions[:, arm])
        rot_jerk = np.diff(rotvec, n=3, axis=0) * fps**3

        pos_step_median = float(np.median(pos_step)) if len(pos_step) else 0.0
        pos_step_mad = (
            float(np.median(np.abs(pos_step - pos_step_median))) if len(pos_step) else 0.0
        )
        step_threshold = pos_step_median + 8.0 * max(pos_step_mad, 1e-6)
        arms.append(
            {
                "translation_hf_rms_mm": float(1000 * robust_rms(pos_hf)),
                "translation_step_median_mm": float(1000 * pos_step_median),
                "translation_step_p95_mm": float(1000 * np.percentile(pos_step, 95))
                if len(pos_step)
                else 0.0,
                "translation_step_max_mm": float(1000 * np.max(pos_step)) if len(pos_step) else 0.0,
                "translation_step_outlier_fraction": float(np.mean(pos_step > step_threshold))
                if len(pos_step)
                else 0.0,
                "translation_acceleration_rms_m_s2": float(robust_rms(pos_acceleration)),
                "translation_jerk_rms_m_s3": float(robust_rms(pos_jerk)),
                "rotation_hf_rms_deg": float(np.degrees(robust_rms(rot_hf))),
                "rotation_step_median_deg": float(np.degrees(np.median(rot_step)))
                if len(rot_step)
                else 0.0,
                "rotation_step_p95_deg": float(np.degrees(np.percentile(rot_step, 95)))
                if len(rot_step)
                else 0.0,
                "rotation_step_max_deg": float(np.degrees(np.max(rot_step)))
                if len(rot_step)
                else 0.0,
                "rotation_jerk_rms_rad_s3": float(robust_rms(rot_jerk)),
            }
        )
    aggregate = {
        key: float(np.mean([arm[key] for arm in arms]))
        for key in arms[0]
        if key != "translation_step_outlier_fraction"
    }
    aggregate["translation_step_outlier_fraction"] = float(
        np.mean([arm["translation_step_outlier_fraction"] for arm in arms])
    )
    return {
        "episode_index": episode.episode_index,
        "frames": len(episode.timestamps),
        "duration_s": len(episode.timestamps) / episode.fps,
        "fps": episode.fps,
        "noise_definition": "zero-phase third-order high-pass residual above 6 Hz",
        "arms": {"umi1_left": arms[0], "umi2_right": arms[1]},
        "aggregate": aggregate,
    }


def vector_noise_metrics(episode: Episode) -> dict[str, Any]:
    """Fallback diagnostic for a joint-space source; never label it as LiDAR noise."""
    residual = high_frequency_residual(episode.state, episode.fps)
    steps = np.diff(episode.state, axis=0)
    jerk = np.diff(episode.state, n=3, axis=0) * episode.fps**3
    return {
        "episode_index": episode.episode_index,
        "frames": len(episode.timestamps),
        "duration_s": len(episode.timestamps) / episode.fps,
        "fps": episode.fps,
        "space": "yam_joint" if episode.kind is EpisodeKind.YAM_JOINT else "generic_vector",
        "warning": (
            "These are joint-space signal metrics, not Cartesian LiDAR pose-noise metrics."
            if episode.kind is EpisodeKind.YAM_JOINT
            else ""
        ),
        "high_frequency_rms_per_channel": robust_rms(residual, axis=0).tolist(),
        "step_p95_per_channel": np.percentile(np.abs(steps), 95, axis=0).tolist(),
        "jerk_rms_per_channel": robust_rms(jerk, axis=0).tolist(),
    }


def aggregate_pose_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("cannot aggregate an empty report list")
    keys = reports[0]["aggregate"].keys()
    values = {key: np.asarray([r["aggregate"][key] for r in reports]) for key in keys}
    return {
        "episodes": len(reports),
        "frames": int(sum(r["frames"] for r in reports)),
        "duration_s": float(sum(r["duration_s"] for r in reports)),
        "metrics": {
            key: {
                "median": float(np.median(value)),
                "p90": float(np.percentile(value, 90)),
                "mean": float(np.mean(value)),
            }
            for key, value in values.items()
        },
    }
