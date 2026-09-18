from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from dataset_filtering.hf_dataset import LeRobotSource, as_pose_episode
from dataset_filtering.models import PoseEpisode
from dataset_filtering.retarget import retarget_pose_episode
from dataset_filtering.smoothing import benchmark_smoothers, with_smoothed_pose
from dataset_filtering.trajectory_optimization import (
    TrajectoryOptimizationConfig,
    optimize_pose_episode,
)

FAILURE_PROGRESS_BIN_COUNT = 20


def analyze_smoothed_reachability(
    source: LeRobotSource,
    *,
    episode_indices: list[int] | None = None,
    workers: int | None = None,
) -> dict[str, Any]:
    """Measure frame-weighted YAM IK reachability after per-episode auto smoothing."""
    summaries = source.episode_summaries()
    available = {summary.episode_index for summary in summaries}
    selected = sorted(available if episode_indices is None else set(episode_indices))
    missing = sorted(set(selected) - available)
    if missing:
        raise IndexError(f"episodes not found: {missing}")
    source.prefetch_episode_data(selected)

    worker_count = max(1, min(workers or max(1, (os.cpu_count() or 2) - 2), len(selected)))
    arguments = [
        (
            source.source,
            source.location.revision,
            str(source.cache_dir),
            episode_index,
        )
        for episode_index in selected
    ]
    if worker_count == 1:
        rows = [_measure_episode(argument) for argument in arguments]
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_measure_episode, argument): argument[3] for argument in arguments
            }
            for future in as_completed(futures):
                rows.append(future.result())
    rows.sort(key=lambda row: row["episode_index"])
    return _aggregate_reachability(source, rows, worker_count)


def write_reachability(report: dict[str, Any], destination: str | Path) -> Path:
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def analyze_continuous_replayability(
    source: LeRobotSource,
    *,
    episode_indices: list[int] | None = None,
    workers: int | None = None,
    progress: bool = False,
    trim_end_s: float = 0.0,
    tail_freeze_grace_s: float = 0.5,
) -> dict[str, Any]:
    """Run the complete continuous optimizer and aggregate episode/frame validity."""
    if trim_end_s < 0:
        raise ValueError("trim_end_s must be non-negative")
    if tail_freeze_grace_s < 0:
        raise ValueError("tail_freeze_grace_s must be non-negative")
    summaries = source.episode_summaries()
    available = {summary.episode_index for summary in summaries}
    selected = sorted(available if episode_indices is None else set(episode_indices))
    missing = sorted(set(selected) - available)
    if missing:
        raise IndexError(f"episodes not found: {missing}")
    source.prefetch_episode_data(selected)
    worker_count = max(
        1,
        min(workers or min(8, max(1, (os.cpu_count() or 2) - 2)), len(selected)),
    )
    arguments = [
        (
            source.source,
            source.location.revision,
            str(source.cache_dir),
            episode_index,
            trim_end_s,
            tail_freeze_grace_s,
        )
        for episode_index in selected
    ]
    if worker_count == 1:
        rows = []
        for argument in arguments:
            rows.append(_measure_continuous_episode(argument))
            if progress:
                print(
                    f"continuous replayability: {len(rows)}/{len(arguments)} episodes",
                    flush=True,
                )
    else:
        rows = []
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_measure_continuous_episode, argument): argument[3]
                for argument in arguments
            }
            for future in as_completed(futures):
                rows.append(future.result())
                if progress:
                    print(
                        f"continuous replayability: {len(rows)}/{len(arguments)} episodes",
                        flush=True,
                    )
    rows.sort(key=lambda row: row["episode_index"])
    frames = sum(row["frames"] for row in rows)
    individual_valid = sum(row["individual_valid_targets"] for row in rows)
    both_valid = sum(row["both_valid_frames"] for row in rows)
    event_total = sum(row["contact_events_total"] for row in rows)
    event_feasible = sum(row["contact_events_feasible"] for row in rows)
    validation_keys = [key for key in rows[0]["validation"] if key.startswith("passes_")]
    rejection_reasons = {
        "solver_failure": sum(not row["solver_success"] for row in rows),
        "strict_contact_failure": sum(not row["all_contacts_strictly_feasible"] for row in rows),
        **{
            key.removeprefix("passes_"): sum(not bool(row["validation"][key]) for row in rows)
            for key in validation_keys
        },
    }
    tail_arms = ("umi1_left", "umi2_right")
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": source.source,
        "revision": source.location.revision,
        "read_only": True,
        "preprocessing": {
            "trim_end_s": trim_end_s,
            "conditional_tail_freeze": {
                "enabled": True,
                "per_arm": True,
                "grace_period_s_after_last_gripper_event": tail_freeze_grace_s,
                "trigger": "any provisional continuous tracking failure after cutoff",
                "frozen_value": "smoothed 6-DoF tool pose at cutoff",
            },
            "source_dataset_modified": False,
        },
        "algorithm": "contact_anchored_bimanual_cubic_bspline_nlp_with_tail_freeze_v3",
        "classification_scope": (
            "all-frame 10 mm/8 deg tracking, strict 1 mm/2 deg gripper events, continuous "
            "replayability under the public YAM kinematic model, URDF velocity limits, "
            "configured acceleration assumption, and sampled-capsule collision proxy; "
            "exact scene collision validation remains required"
        ),
        "episodes": len(rows),
        "frames": frames,
        "duration_s": sum(row["source_duration_s"] for row in rows),
        "workers": worker_count,
        "episode_replayability": {
            "continuously_replayable": sum(row["continuously_replayable"] for row in rows),
            "ik_trajectory_solvable": sum(row["ik_trajectory_solvable"] for row in rows),
            "solver_success": sum(row["solver_success"] for row in rows),
            "all_contacts_strictly_feasible": sum(
                row["all_contacts_strictly_feasible"] for row in rows
            ),
            "all_frames_within_10mm_8deg": sum(
                row["both_valid_frames"] == row["frames"] for row in rows
            ),
            "continuously_replayable_and_all_frames_within_10mm_8deg": sum(
                row["continuously_replayable"] and row["both_valid_frames"] == row["frames"]
                for row in rows
            ),
        },
        "frame_weighted_tracking": {
            "individual_target_fraction_10mm_8deg": individual_valid / (2 * frames),
            "both_target_fraction_10mm_8deg": both_valid / frames,
        },
        "failure_timing": {
            "definition": "frame where either arm exceeds 10 mm position or 8 deg orientation",
            "normalization": "frame_index / episode_frame_count, aggregated into 5% bins",
            "bins": _aggregate_failure_timing(rows),
        },
        "tail_freeze": {
            "episodes_modified": sum(row["tail_freeze"]["applied_any"] for row in rows),
            "episodes_recovered_as_ik_solvable": sum(
                row["ik_trajectory_solvable"]
                and not row["tail_freeze"]["provisional_ik_trajectory_solvable"]
                for row in rows
            ),
            "episodes_recovered_as_continuously_replayable": sum(
                row["continuously_replayable"]
                and not row["tail_freeze"]["provisional_continuously_replayable"]
                for row in rows
            ),
            "continuously_replayable_without_modification": sum(
                row["continuously_replayable"] and not row["tail_freeze"]["applied_any"]
                for row in rows
            ),
            "arms_modified": {
                arm: sum(row["tail_freeze"]["arms"][arm]["applied"] for row in rows)
                for arm in tail_arms
            },
            "provisional_failed_bimanual_frames": sum(
                row["tail_freeze"]["provisional_failed_bimanual_frames"] for row in rows
            ),
            "final_failed_bimanual_frames": sum(
                row["tail_freeze"]["final_failed_bimanual_frames"] for row in rows
            ),
        },
        "rejection_reasons": rejection_reasons,
        "strict_contact_reachability": {
            "fraction": event_feasible / max(event_total, 1),
            "feasible": event_feasible,
            "total": event_total,
            "position_tolerance_mm": 1.0,
            "orientation_tolerance_deg": 2.0,
        },
        "validation_pass_episodes": {
            key.removeprefix("passes_"): sum(bool(row["validation"][key]) for row in rows)
            for key in validation_keys
        },
        "timing_methods": {
            method: sum(row["timing_method"] == method for row in rows)
            for method in sorted({row["timing_method"] for row in rows})
        },
        "video_alignment": {
            "extrinsics_calibrated": False,
            "metric_is_uncalibrated_tcp_proxy": True,
            "mean_episode_fraction_10mm_2deg": float(
                np.mean([row["vision_action_alignment_fraction_10mm_2deg"] for row in rows])
            ),
            "mean_episode_time_scale": float(np.mean([row["time_scale"] for row in rows])),
            "maximum_episode_time_scale": float(np.max([row["time_scale"] for row in rows])),
        },
        "per_episode": rows,
    }


def _measure_episode(arguments: tuple[str, str, str, int]) -> dict[str, Any]:
    source_name, revision, cache_dir, episode_index = arguments
    source = LeRobotSource(source_name, revision=revision, cache_dir=cache_dir)
    episode = as_pose_episode(source.load_episode(episode_index))
    smoothing = benchmark_smoothers(episode)[0]
    filtered = with_smoothed_pose(episode, smoothing)
    result = retarget_pose_episode(filtered, max_iterations=75)
    left = result.arms[0].reachable
    right = result.arms[1].reachable
    left_events, right_events = result.event_indices
    return {
        "episode_index": episode_index,
        "frames": len(episode.timestamps),
        "duration_s": len(episode.timestamps) / episode.fps,
        "smoothing": {
            "method": smoothing.method,
            "parameters": smoothing.parameters,
            "score": smoothing.score,
        },
        "reachable_frames": {
            "umi1_left": int(np.count_nonzero(left)),
            "umi2_right": int(np.count_nonzero(right)),
            "both": int(np.count_nonzero(left & right)),
            "at_least_one": int(np.count_nonzero(left | right)),
        },
        "gripper_event_targets": {
            "umi1_left_total": len(left_events),
            "umi1_left_reachable": int(np.count_nonzero(left[left_events])),
            "umi2_right_total": len(right_events),
            "umi2_right_reachable": int(np.count_nonzero(right[right_events])),
        },
        "position_error_mm": {
            "umi1_left_median": float(1000 * np.median(result.arms[0].position_error_m)),
            "umi1_left_p95": float(1000 * np.percentile(result.arms[0].position_error_m, 95)),
            "umi2_right_median": float(1000 * np.median(result.arms[1].position_error_m)),
            "umi2_right_p95": float(1000 * np.percentile(result.arms[1].position_error_m, 95)),
        },
    }


def _measure_continuous_episode(
    arguments: tuple[str, str, str, int, float, float],
) -> dict[str, Any]:
    source_name, revision, cache_dir, episode_index, trim_end_s, tail_freeze_grace_s = arguments
    source = LeRobotSource(source_name, revision=revision, cache_dir=cache_dir)
    original = as_pose_episode(source.load_episode(episode_index))
    episode = trim_pose_episode_end(original, trim_end_s)
    smoothing = benchmark_smoothers(episode)[0]
    filtered = with_smoothed_pose(episode, smoothing)
    result = optimize_pose_episode(
        filtered,
        config=TrajectoryOptimizationConfig(tail_freeze_grace_s=tail_freeze_grace_s),
    )
    metadata = result.metadata
    failure_timing = _episode_failure_timing(result)
    return {
        "episode_index": episode_index,
        "source_original_frames": len(original.timestamps),
        "trimmed_end_frames": len(original.timestamps) - len(episode.timestamps),
        "frames": len(episode.timestamps),
        "source_duration_s": len(episode.timestamps) / episode.fps,
        "robot_duration_s": float(result.robot_timestamps[-1] - result.robot_timestamps[0]),
        "smoothing": {
            "method": smoothing.method,
            "parameters": smoothing.parameters,
            "score": smoothing.score,
        },
        "solver_success": bool(metadata["solver"]["success"]),
        "ik_trajectory_solvable": bool(metadata["ik_trajectory_solvable"]),
        "continuously_replayable": bool(metadata["continuously_replayable"]),
        "all_contacts_strictly_feasible": bool(metadata["contact_events"]["all_strictly_feasible"]),
        "contact_events_feasible": int(metadata["contact_events"]["strictly_feasible"]),
        "contact_events_total": int(metadata["contact_events"]["total"]),
        "individual_valid_targets": int(np.count_nonzero(result.frame_valid)),
        "both_valid_frames": int(np.count_nonzero(np.all(result.frame_valid, axis=1))),
        "position_error_p95_mm": float(metadata["tracking"]["position_error_p95_mm"]),
        "position_error_max_mm": float(1000 * np.max(result.position_error_m)),
        "orientation_error_p95_deg": float(metadata["tracking"]["orientation_error_p95_deg"]),
        "orientation_error_max_deg": float(np.degrees(np.max(result.orientation_error_rad))),
        "flow_direction_cosine_median": float(metadata["tracking"]["flow_direction_cosine_median"]),
        "time_scale": float(metadata["timing"]["time_scale"]),
        "timing_method": metadata["timing"]["method"],
        "vision_action_alignment_fraction_10mm_2deg": float(
            metadata["camera_alignment"]["vision_action_alignment_fraction_10mm_2deg"]
        ),
        "camera_alignment_status": metadata["camera_alignment"]["status"],
        "robot_delay_end_s": float(
            metadata["camera_alignment"]["temporal_mapping"]["robot_delay_end_s"]
        ),
        "failure_frames": {
            "umi1_left": np.flatnonzero(~result.frame_valid[:, 0]).tolist(),
            "umi2_right": np.flatnonzero(~result.frame_valid[:, 1]).tolist(),
            "either_arm": np.flatnonzero(~np.all(result.frame_valid, axis=1)).tolist(),
        },
        "failure_timing_bins": failure_timing,
        "tail_freeze": metadata["tail_freeze"],
        "validation": metadata["validation"],
    }


def _episode_failure_timing(result: Any) -> list[dict[str, int]]:
    frames = len(result.source_timestamps)
    bin_index = np.minimum(
        (np.arange(frames, dtype=np.int64) * FAILURE_PROGRESS_BIN_COUNT) // max(frames, 1),
        FAILURE_PROGRESS_BIN_COUNT - 1,
    )
    position_failure = result.position_error_m > 0.01
    orientation_failure = result.orientation_error_rad > np.deg2rad(8.0)
    arm_failure = position_failure | orientation_failure
    any_position = np.any(position_failure, axis=1)
    any_orientation = np.any(orientation_failure, axis=1)
    any_failure = np.any(arm_failure, axis=1)
    rows: list[dict[str, int]] = []
    for progress_bin in range(FAILURE_PROGRESS_BIN_COUNT):
        selected = bin_index == progress_bin
        rows.append(
            {
                "total_frames": int(np.count_nonzero(selected)),
                "failed_frames": int(np.count_nonzero(selected & any_failure)),
                "umi1_left_failed_frames": int(
                    np.count_nonzero(selected & arm_failure[:, 0])
                ),
                "umi2_right_failed_frames": int(
                    np.count_nonzero(selected & arm_failure[:, 1])
                ),
                "position_only_failed_frames": int(
                    np.count_nonzero(selected & any_position & ~any_orientation)
                ),
                "orientation_only_failed_frames": int(
                    np.count_nonzero(selected & ~any_position & any_orientation)
                ),
                "position_and_orientation_failed_frames": int(
                    np.count_nonzero(selected & any_position & any_orientation)
                ),
            }
        )
    return rows


def _aggregate_failure_timing(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    aggregated: list[dict[str, Any]] = []
    for progress_bin in range(FAILURE_PROGRESS_BIN_COUNT):
        episode_bins = [row["failure_timing_bins"][progress_bin] for row in rows]
        total_frames = sum(item["total_frames"] for item in episode_bins)
        failed_frames = sum(item["failed_frames"] for item in episode_bins)
        aggregated.append(
            {
                "progress_start_percent": 100 * progress_bin / FAILURE_PROGRESS_BIN_COUNT,
                "progress_end_percent": 100 * (progress_bin + 1) / FAILURE_PROGRESS_BIN_COUNT,
                "total_frames": total_frames,
                "failed_frames": failed_frames,
                "failure_rate": failed_frames / max(total_frames, 1),
                "episodes_affected": sum(item["failed_frames"] > 0 for item in episode_bins),
                "umi1_left_failed_frames": sum(
                    item["umi1_left_failed_frames"] for item in episode_bins
                ),
                "umi2_right_failed_frames": sum(
                    item["umi2_right_failed_frames"] for item in episode_bins
                ),
                "position_only_failed_frames": sum(
                    item["position_only_failed_frames"] for item in episode_bins
                ),
                "orientation_only_failed_frames": sum(
                    item["orientation_only_failed_frames"] for item in episode_bins
                ),
                "position_and_orientation_failed_frames": sum(
                    item["position_and_orientation_failed_frames"] for item in episode_bins
                ),
            }
        )
    return aggregated


def trim_pose_episode_end(episode: PoseEpisode, seconds: float) -> PoseEpisode:
    """Return an in-memory prefix; never mutate the source episode or dataset."""
    if seconds < 0:
        raise ValueError("seconds must be non-negative")
    trim_frames = int(round(seconds * episode.fps))
    if trim_frames == 0:
        return episode
    keep = len(episode.timestamps) - trim_frames
    if keep < 2:
        raise ValueError(
            f"episode {episode.episode_index} has {len(episode.timestamps)} frames; "
            f"cannot trim {trim_frames}"
        )
    return replace(
        episode,
        timestamps=episode.timestamps[:keep].copy(),
        positions=episode.positions[:keep].copy(),
        quaternions=episode.quaternions[:keep].copy(),
        grippers=None if episode.grippers is None else episode.grippers[:keep].copy(),
    )


def _aggregate_reachability(
    source: LeRobotSource, rows: list[dict[str, Any]], workers: int
) -> dict[str, Any]:
    frames = sum(row["frames"] for row in rows)
    left = sum(row["reachable_frames"]["umi1_left"] for row in rows)
    right = sum(row["reachable_frames"]["umi2_right"] for row in rows)
    both = sum(row["reachable_frames"]["both"] for row in rows)
    either = sum(row["reachable_frames"]["at_least_one"] for row in rows)
    left_events = sum(row["gripper_event_targets"]["umi1_left_total"] for row in rows)
    right_events = sum(row["gripper_event_targets"]["umi2_right_total"] for row in rows)
    left_events_reachable = sum(row["gripper_event_targets"]["umi1_left_reachable"] for row in rows)
    right_events_reachable = sum(
        row["gripper_event_targets"]["umi2_right_reachable"] for row in rows
    )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "source": source.source,
        "revision": source.location.revision,
        "read_only": True,
        "episodes": len(rows),
        "frames": frames,
        "duration_s": sum(row["duration_s"] for row in rows),
        "workers": workers,
        "smoothing": "lowest distortion-aware score per episode; gripper-event poses anchored",
        "ik_tolerances": {
            "normal_position_mm": 10.0,
            "gripper_event_position_mm": 5.0,
            "orientation_deg": 8.0,
        },
        "frame_weighted_reachability": {
            "umi1_left_fraction": left / frames,
            "umi2_right_fraction": right / frames,
            "individual_arm_target_fraction": (left + right) / (2 * frames),
            "both_arms_same_frame_fraction": both / frames,
            "at_least_one_arm_fraction": either / frames,
        },
        "gripper_event_reachability": {
            "umi1_left_fraction": left_events_reachable / max(left_events, 1),
            "umi2_right_fraction": right_events_reachable / max(right_events, 1),
            "individual_arm_event_fraction": (left_events_reachable + right_events_reachable)
            / max(left_events + right_events, 1),
            "umi1_left_reachable": left_events_reachable,
            "umi1_left_total": left_events,
            "umi2_right_reachable": right_events_reachable,
            "umi2_right_total": right_events,
        },
        "per_episode": rows,
    }
