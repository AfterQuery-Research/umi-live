from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, savgol_filter, sosfiltfilt
from scipy.spatial.transform import Rotation

from dataset_filtering.models import PoseEpisode, SmoothingResult
from dataset_filtering.noise import high_frequency_residual, robust_rms, rotation_step_angles

DEFAULT_CANDIDATES: tuple[tuple[str, dict[str, float | int]], ...] = (
    ("savgol", {"window": 9, "polyorder": 2}),
    ("savgol", {"window": 15, "polyorder": 3}),
    ("savgol", {"window": 21, "polyorder": 3}),
    ("gaussian", {"sigma_frames": 1.0}),
    ("gaussian", {"sigma_frames": 2.0}),
    ("gaussian", {"sigma_frames": 3.0}),
    ("butterworth", {"cutoff_hz": 4.0, "order": 3}),
    ("butterworth", {"cutoff_hz": 6.0, "order": 3}),
    ("butterworth", {"cutoff_hz": 8.0, "order": 3}),
)

# Ratios below a nanometre/nanoradian-scale signal are numerical diagnostics, not
# physical smoothing evidence. A concrete floor keeps candidate ordering invariant
# across CPU architectures for exactly linear/constant synthetic paths while remaining
# many orders of magnitude below the measured UMI trajectory noise.
NUMERICAL_SIGNAL_FLOOR = 1e-9


def detect_gripper_events(
    grippers: np.ndarray | None, *, fps: float, minimum_separation_s: float = 0.18
) -> tuple[np.ndarray, np.ndarray]:
    if grippers is None or len(grippers) < 3:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    result: list[np.ndarray] = []
    minimum_separation = max(1, int(round(minimum_separation_s * fps)))
    for arm in range(2):
        width = np.asarray(grippers[:, arm], dtype=np.float64)
        derivative = np.abs(np.diff(width, prepend=width[0]))
        median = float(np.median(derivative))
        mad = float(np.median(np.abs(derivative - median)))
        width_range = float(np.percentile(width, 95) - np.percentile(width, 5))
        threshold = max(median + 8 * mad, 0.06 * width_range, 1e-6)
        candidates = np.flatnonzero(derivative >= threshold)
        selected: list[int] = []
        for index in candidates:
            if not selected or index - selected[-1] >= minimum_separation:
                selected.append(int(index))
            elif derivative[index] > derivative[selected[-1]]:
                selected[-1] = int(index)
        result.append(np.asarray(selected, dtype=np.int64))
    return result[0], result[1]


def smooth_pose_episode(
    episode: PoseEpisode,
    method: str,
    parameters: dict[str, Any],
    *,
    preserve_gripper_events: bool = True,
) -> SmoothingResult:
    event_indices = detect_gripper_events(episode.grippers, fps=episode.fps)
    positions = np.empty_like(episode.positions, dtype=np.float64)
    quaternions = np.empty_like(episode.quaternions, dtype=np.float64)
    for arm in range(2):
        positions[:, arm] = _filter_values(
            episode.positions[:, arm], episode.fps, method, parameters
        )
        rotations = Rotation.from_quat(episode.quaternions[:, arm])
        reference = rotations[0]
        relative = reference.inv() * rotations
        if relative is NotImplemented:
            raise TypeError("could not compose rotations")
        filtered_rotvec = _filter_values(relative.as_rotvec(), episode.fps, method, parameters)
        filtered_rotations = reference * Rotation.from_rotvec(filtered_rotvec)
        if filtered_rotations is NotImplemented:
            raise TypeError("could not compose rotations")
        quaternions[:, arm] = filtered_rotations.as_quat()
        if preserve_gripper_events:
            positions[:, arm], quaternions[:, arm] = _anchor_event_poses(
                positions[:, arm],
                quaternions[:, arm],
                episode.positions[:, arm],
                episode.quaternions[:, arm],
                event_indices[arm],
                radius=max(2, int(round(0.12 * episode.fps))),
            )
    metrics = smoothing_metrics(episode, positions, quaternions, event_indices)
    return SmoothingResult(
        method=method,
        parameters=dict(parameters),
        positions=positions,
        quaternions=quaternions,
        metrics=metrics,
        score=_candidate_score(metrics),
    )


def benchmark_smoothers(
    episode: PoseEpisode,
    candidates: tuple[tuple[str, dict[str, float | int]], ...] = DEFAULT_CANDIDATES,
) -> list[SmoothingResult]:
    results = [
        smooth_pose_episode(episode, method, dict(parameters)) for method, parameters in candidates
    ]
    return sorted(results, key=lambda result: result.score)


def smoothing_metrics(
    raw: PoseEpisode,
    positions: np.ndarray,
    quaternions: np.ndarray,
    event_indices: tuple[np.ndarray, np.ndarray],
) -> dict[str, float]:
    pos_distortion = np.linalg.norm(positions - raw.positions, axis=-1)
    raw_rot = Rotation.from_quat(raw.quaternions.reshape(-1, 4))
    filtered_rot = Rotation.from_quat(quaternions.reshape(-1, 4))
    rot_delta = raw_rot.inv() * filtered_rot
    if rot_delta is NotImplemented:
        raise TypeError("could not compose rotations")
    rot_distortion = rot_delta.magnitude().reshape(-1, 2)

    raw_pos_hf = high_frequency_residual(raw.positions.reshape(len(raw.timestamps), -1), raw.fps)
    filtered_pos_hf = high_frequency_residual(positions.reshape(len(raw.timestamps), -1), raw.fps)
    raw_rot_hf: list[np.ndarray] = []
    filtered_rot_hf: list[np.ndarray] = []
    direction_cosines: list[float] = []
    for arm in range(2):
        reference = Rotation.from_quat(raw.quaternions[0, arm])
        raw_relative = reference.inv() * Rotation.from_quat(raw.quaternions[:, arm])
        filtered_relative = reference.inv() * Rotation.from_quat(quaternions[:, arm])
        if raw_relative is NotImplemented or filtered_relative is NotImplemented:
            raise TypeError("could not compose rotations")
        raw_rot_hf.append(high_frequency_residual(raw_relative.as_rotvec(), raw.fps))
        filtered_rot_hf.append(high_frequency_residual(filtered_relative.as_rotvec(), raw.fps))
        # Compare direction over 200 ms, not frame-to-frame: the latter is dominated by
        # the very high-frequency noise this metric is meant to let us remove.
        lag = max(1, int(round(0.2 * raw.fps)))
        raw_velocity = raw.positions[lag:, arm] - raw.positions[:-lag, arm]
        filtered_velocity = positions[lag:, arm] - positions[:-lag, arm]
        moving = np.linalg.norm(raw_velocity, axis=1) > 1e-3
        if np.any(moving):
            numerator = np.sum(raw_velocity[moving] * filtered_velocity[moving], axis=1)
            denominator = np.linalg.norm(raw_velocity[moving], axis=1) * np.linalg.norm(
                filtered_velocity[moving], axis=1
            )
            direction_cosines.extend((numerator / np.maximum(denominator, 1e-12)).tolist())

    raw_jerk = np.diff(raw.positions, n=3, axis=0) * raw.fps**3
    filtered_jerk = np.diff(positions, n=3, axis=0) * raw.fps**3
    event_pos: list[float] = []
    event_rot: list[float] = []
    for arm in range(2):
        event_pos.extend(pos_distortion[event_indices[arm], arm].tolist())
        event_rot.extend(rot_distortion[event_indices[arm], arm].tolist())

    return {
        "translation_distortion_rms_mm": float(1000 * robust_rms(pos_distortion)),
        "translation_distortion_p95_mm": float(1000 * np.percentile(pos_distortion, 95)),
        "rotation_distortion_rms_deg": float(np.degrees(robust_rms(rot_distortion))),
        "translation_hf_remaining_ratio": float(
            robust_rms(filtered_pos_hf)
            / max(float(robust_rms(raw_pos_hf)), NUMERICAL_SIGNAL_FLOOR)
        ),
        "rotation_hf_remaining_ratio": float(
            robust_rms(np.stack(filtered_rot_hf, axis=1))
            / max(
                float(robust_rms(np.stack(raw_rot_hf, axis=1))),
                NUMERICAL_SIGNAL_FLOOR,
            )
        ),
        "translation_jerk_remaining_ratio": float(
            robust_rms(filtered_jerk)
            / max(float(robust_rms(raw_jerk)), NUMERICAL_SIGNAL_FLOOR)
        ),
        "direction_cosine_median": float(np.median(direction_cosines))
        if direction_cosines
        else 1.0,
        "event_translation_error_max_mm": float(1000 * max(event_pos, default=0.0)),
        "event_rotation_error_max_deg": float(np.degrees(max(event_rot, default=0.0))),
        "raw_translation_step_p95_mm": float(
            1000 * np.percentile(np.linalg.norm(np.diff(raw.positions, axis=0), axis=-1), 95)
        ),
        "filtered_translation_step_p95_mm": float(
            1000 * np.percentile(np.linalg.norm(np.diff(positions, axis=0), axis=-1), 95)
        ),
        "raw_rotation_step_p95_deg": float(
            np.degrees(
                np.mean(
                    [
                        np.percentile(rotation_step_angles(raw.quaternions[:, arm]), 95)
                        for arm in range(2)
                    ]
                )
            )
        ),
        "filtered_rotation_step_p95_deg": float(
            np.degrees(
                np.mean(
                    [
                        np.percentile(rotation_step_angles(quaternions[:, arm]), 95)
                        for arm in range(2)
                    ]
                )
            )
        ),
    }


def with_smoothed_pose(episode: PoseEpisode, result: SmoothingResult) -> PoseEpisode:
    return replace(episode, positions=result.positions, quaternions=result.quaternions)


def _filter_values(
    values: np.ndarray, fps: float, method: str, parameters: dict[str, Any]
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 5:
        return values.copy()
    if method == "savgol":
        requested = int(parameters.get("window", 15))
        window = min(
            requested if requested % 2 else requested + 1, len(values) - (1 - len(values) % 2)
        )
        window = max(3, window)
        polyorder = min(int(parameters.get("polyorder", 3)), window - 1)
        return savgol_filter(values, window, polyorder, axis=0, mode="interp")
    if method == "gaussian":
        return gaussian_filter1d(
            values, float(parameters.get("sigma_frames", 2.0)), axis=0, mode="nearest"
        )
    if method == "butterworth":
        cutoff = min(float(parameters.get("cutoff_hz", 6.0)), 0.45 * fps)
        order = int(parameters.get("order", 3))
        sos = butter(order, cutoff, btype="lowpass", fs=fps, output="sos")
        if len(values) <= 3 * (2 * order + 1):
            return gaussian_filter1d(values, 1.0, axis=0, mode="nearest")
        return sosfiltfilt(sos, values, axis=0)
    raise ValueError(f"unknown smoothing method {method!r}")


def _anchor_event_poses(
    filtered_positions: np.ndarray,
    filtered_quaternions: np.ndarray,
    raw_positions: np.ndarray,
    raw_quaternions: np.ndarray,
    events: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    positions = filtered_positions.copy()
    rotations = Rotation.from_quat(filtered_quaternions)
    for event in events:
        start = max(0, int(event) - radius)
        stop = min(len(positions), int(event) + radius + 1)
        indices = np.arange(start, stop)
        weights = 1.0 - np.abs(indices - int(event)) / (radius + 1)
        offset = raw_positions[event] - positions[event]
        positions[indices] += weights[:, None] * offset

        raw_event = Rotation.from_quat(raw_quaternions[event])
        filtered_event = rotations[event]
        correction = raw_event * filtered_event.inv()
        if correction is NotImplemented:
            raise TypeError("could not compose rotations")
        correction_rotvec = correction.as_rotvec()
        local_corrections = Rotation.from_rotvec(weights[:, None] * correction_rotvec)
        corrected = local_corrections * rotations[indices]
        if corrected is NotImplemented:
            raise TypeError("could not compose rotations")
        quats = rotations.as_quat()
        quats[indices] = corrected.as_quat()
        rotations = Rotation.from_quat(quats)
    return positions, rotations.as_quat()


def _candidate_score(metrics: dict[str, float]) -> float:
    """Lower is better: a Pareto score that makes every millimetre of drift costly.

    The raw high-frequency estimate in this dataset is sub-millimetre, so allowing a
    several-millimetre dead zone would reward aggressive over-smoothing. Distortion is
    therefore charged from zero while direction and gripper-event errors are guardrails.
    """
    distortion_penalty = metrics["translation_distortion_rms_mm"] / 1.0
    rotation_penalty = metrics["rotation_distortion_rms_deg"] / 0.25
    direction_penalty = max(0.0, 0.985 - metrics["direction_cosine_median"]) * 20.0
    event_penalty = (
        metrics["event_translation_error_max_mm"] + metrics["event_rotation_error_max_deg"]
    )
    return float(
        0.30 * metrics["translation_hf_remaining_ratio"]
        + 0.15 * metrics["rotation_hf_remaining_ratio"]
        + 0.15 * metrics["translation_jerk_remaining_ratio"]
        + 0.25 * distortion_penalty
        + 0.10 * rotation_penalty
        + 0.05 * direction_penalty
        + 100.0 * event_penalty
    )
