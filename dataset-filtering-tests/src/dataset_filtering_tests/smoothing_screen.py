from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from dataset_filtering.hf_dataset import LeRobotSource, as_pose_episode
from dataset_filtering.retarget import retarget_pose_episode
from dataset_filtering.smoothing import smooth_pose_episode, with_smoothed_pose

from dataset_filtering_tests.baseline import BASELINE_REPORT, load_baseline

SMOOTHING_CANDIDATES: tuple[tuple[str, dict[str, float | int]], ...] = (
    ("butterworth", {"cutoff_hz": 4.0, "order": 3}),
    ("butterworth", {"cutoff_hz": 3.0, "order": 3}),
    ("butterworth", {"cutoff_hz": 2.0, "order": 3}),
    ("butterworth", {"cutoff_hz": 1.5, "order": 3}),
    ("butterworth", {"cutoff_hz": 1.0, "order": 3}),
)


def screen_smoothing_strength(
    *,
    episode_indices: list[int],
    source_name: str = "brandonyang/dual-lidar-umi-relative",
    source_revision: str = "main",
    baseline_report: str | Path = BASELINE_REPORT,
    workers: int = 8,
) -> dict[str, Any]:
    if not episode_indices:
        raise ValueError("at least one episode is required")
    baseline = load_baseline(baseline_report)
    valid_indices = {int(row["episode_index"]) for row in baseline["per_episode"]}
    unique_indices = sorted(set(episode_indices))
    missing = sorted(set(unique_indices) - valid_indices)
    if missing:
        raise ValueError(f"episodes are missing from the baseline: {missing}")

    source = LeRobotSource(source_name, revision=source_revision)
    source.prefetch_episode_data(unique_indices)
    arguments = [
        (
            source_name,
            source.location.revision,
            str(source.cache_dir),
            episode_index,
        )
        for episode_index in unique_indices
    ]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(arguments)))) as executor:
        futures = [executor.submit(_screen_episode, item) for item in arguments]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            print(f"smoothing screen: {completed}/{len(futures)} episodes", flush=True)
    rows.sort(key=lambda row: row["episode_index"])

    aggregates: list[dict[str, Any]] = []
    for method, parameters in SMOOTHING_CANDIDATES:
        candidate_id = _candidate_id(method, parameters)
        values = [row["candidates"][candidate_id] for row in rows]
        aggregates.append(
            {
                "candidate": candidate_id,
                "episodes": len(values),
                "fidelity_guard_pass": sum(value["fidelity_guard_pass"] for value in values),
                "all_frames_both_ik_reachable": sum(
                    value["all_frames_both_ik_reachable"] for value in values
                ),
                "all_strict_events_ik_reachable": sum(
                    value["all_strict_events_ik_reachable"] for value in values
                ),
                "median_both_frame_fraction": float(
                    np.median([value["both_frame_fraction"] for value in values])
                ),
                "median_translation_distortion_p95_mm": float(
                    np.median(
                        [value["metrics"]["translation_distortion_p95_mm"] for value in values]
                    )
                ),
                "median_rotation_distortion_rms_deg": float(
                    np.median(
                        [value["metrics"]["rotation_distortion_rms_deg"] for value in values]
                    )
                ),
            }
        )
    return {
        "experiment": "event_preserving_smoothing_strength_framewise_ik_screen",
        "source": source_name,
        "baseline_report": str(Path(baseline_report).expanduser().resolve()),
        "episodes": unique_indices,
        "gate_semantics_changed": False,
        "warning": (
            "Framewise IK is a screening proxy. Any selected smoother must be rerun through "
            "the complete continuous, contact, dynamics, and collision pipeline."
        ),
        "fidelity_guard": {
            "translation_distortion_p95_mm_max": 10.0,
            "rotation_distortion_rms_deg_max": 5.0,
            "direction_cosine_median_min": 0.98,
            "event_translation_error_max_mm": 1e-6,
            "event_rotation_error_max_deg": 1e-6,
            "rationale": "preserve training/video semantics rather than tune acceptance gates",
        },
        "aggregates": aggregates,
        "per_episode": rows,
    }


def _screen_episode(arguments: tuple[str, str, str, int]) -> dict[str, Any]:
    source_name, revision, cache_dir, episode_index = arguments
    source = LeRobotSource(source_name, revision=revision, cache_dir=cache_dir)
    episode = as_pose_episode(source.load_episode(episode_index))
    candidates: dict[str, Any] = {}
    for method, parameters in SMOOTHING_CANDIDATES:
        smoothing = smooth_pose_episode(episode, method, parameters)
        filtered = with_smoothed_pose(episode, smoothing)
        result = retarget_pose_episode(
            filtered,
            position_tolerance_m=0.01,
            contact_position_tolerance_m=0.001,
            orientation_tolerance_rad=np.deg2rad(8.0),
            max_iterations=75,
        )
        both = result.arms[0].reachable & result.arms[1].reachable
        strict_events: list[np.ndarray] = []
        for arm in range(2):
            events = result.event_indices[arm]
            strict_events.append(
                (result.arms[arm].position_error_m[events] <= 0.001)
                & (result.arms[arm].orientation_error_rad[events] <= np.deg2rad(2.0))
            )
        metrics = smoothing.metrics
        fidelity_guard_pass = bool(
            metrics["translation_distortion_p95_mm"] <= 10.0
            and metrics["rotation_distortion_rms_deg"] <= 5.0
            and metrics["direction_cosine_median"] >= 0.98
            and metrics["event_translation_error_max_mm"] <= 1e-6
            and metrics["event_rotation_error_max_deg"] <= 1e-6
        )
        candidates[_candidate_id(method, parameters)] = {
            "method": method,
            "parameters": parameters,
            "score": smoothing.score,
            "metrics": metrics,
            "fidelity_guard_pass": fidelity_guard_pass,
            "both_frame_fraction": float(np.mean(both)),
            "all_frames_both_ik_reachable": bool(np.all(both)),
            "all_strict_events_ik_reachable": bool(
                all(np.all(values) for values in strict_events)
            ),
        }
    return {"episode_index": episode_index, "candidates": candidates}


def _candidate_id(method: str, parameters: dict[str, float | int]) -> str:
    values = ",".join(f"{key}={parameters[key]}" for key in sorted(parameters))
    return f"{method}:{values}"


def select_safe_smoothing_improvements(
    report_path: str | Path,
) -> dict[int, tuple[str, dict[str, Any]]]:
    import json

    report = json.loads(Path(report_path).expanduser().resolve().read_text(encoding="utf-8"))
    baseline_id = _candidate_id("butterworth", {"cutoff_hz": 4.0, "order": 3})
    selected: dict[int, tuple[str, dict[str, Any]]] = {}
    for row in report["per_episode"]:
        baseline = row["candidates"][baseline_id]
        safe = [
            candidate
            for candidate in row["candidates"].values()
            if candidate["fidelity_guard_pass"]
            and candidate["all_strict_events_ik_reachable"]
        ]
        if not safe:
            continue
        best = max(
            safe,
            key=lambda candidate: (
                candidate["all_frames_both_ik_reachable"],
                candidate["both_frame_fraction"],
                -candidate["metrics"]["translation_distortion_p95_mm"],
                -candidate["score"],
            ),
        )
        if best["both_frame_fraction"] > baseline["both_frame_fraction"] + 1e-6:
            selected[int(row["episode_index"])] = (
                best["method"],
                dict(best["parameters"]),
            )
    return selected
