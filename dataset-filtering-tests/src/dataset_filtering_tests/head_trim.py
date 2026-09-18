from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from dataset_filtering.hf_dataset import LeRobotSource, as_pose_episode
from dataset_filtering.retarget import retarget_pose_episode
from dataset_filtering.smoothing import smooth_pose_episode, with_smoothed_pose

from dataset_filtering_tests.baseline import BASELINE_REPORT, load_baseline
from dataset_filtering_tests.preprocessing import (
    choose_event_safe_head_trim,
    trim_pose_episode_head,
)


def screen_head_trim(
    *,
    source_name: str = "brandonyang/dual-lidar-umi-relative",
    baseline_report: str | Path = BASELINE_REPORT,
    candidates_s: tuple[float, ...] = (0.0, 1.0, 2.0, 3.0),
    workers: int = 8,
    tracking_failures_only: bool = True,
) -> dict[str, Any]:
    baseline = load_baseline(baseline_report)
    selected = [
        row
        for row in baseline["per_episode"]
        if not tracking_failures_only or not row["validation"]["passes_tracking_accuracy"]
    ]
    source = LeRobotSource(source_name)
    episode_indices = [int(row["episode_index"]) for row in selected]
    source.prefetch_episode_data(episode_indices)
    arguments = [
        (
            source_name,
            source.location.revision,
            str(source.cache_dir),
            int(row["episode_index"]),
            row["smoothing"]["method"],
            row["smoothing"]["parameters"],
            candidates_s,
        )
        for row in selected
    ]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(arguments)))) as executor:
        futures = [executor.submit(_screen_episode, argument) for argument in arguments]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            print(f"head-trim IK screen: {completed}/{len(futures)} episodes", flush=True)
    rows.sort(key=lambda row: row["episode_index"])
    aggregates: list[dict[str, Any]] = []
    for candidate in candidates_s:
        measurements = [row["candidates"][str(candidate)] for row in rows]
        aggregates.append(
            {
                "requested_trim_s": candidate,
                "episodes": len(measurements),
                "all_frames_both_ik_reachable": sum(
                    item["all_frames_both_ik_reachable"] for item in measurements
                ),
                "all_gripper_events_ik_reachable": sum(
                    item["all_gripper_events_ik_reachable"] for item in measurements
                ),
                "median_both_frame_fraction": float(
                    np.median([item["both_frame_fraction"] for item in measurements])
                ),
                "mean_actual_trim_s": float(
                    np.mean([item["actual_trim_s"] for item in measurements])
                ),
                "guard_clamped_episodes": sum(item["guard_clamped"] for item in measurements),
                "median_minimum_target_tcp_separation_mm": float(
                    np.median(
                        [item["minimum_target_tcp_separation_mm"] for item in measurements]
                    )
                ),
            }
        )
    return {
        "experiment": "event_safe_head_trim_framewise_ik_screen",
        "source": source_name,
        "baseline_report": str(Path(baseline_report).expanduser().resolve()),
        "scope": "tracking failures only" if tracking_failures_only else "all episodes",
        "gate_semantics_changed": False,
        "warning": (
            "Framewise IK is a screening proxy only; candidates must pass the continuous "
            "trajectory, dynamics, collision, contact, and video-alignment evaluation."
        ),
        "aggregates": aggregates,
        "per_episode": rows,
    }


def _screen_episode(
    arguments: tuple[str, str, str, int, str, dict[str, Any], tuple[float, ...]],
) -> dict[str, Any]:
    source_name, revision, cache_dir, episode_index, method, parameters, candidates = arguments
    source = LeRobotSource(source_name, revision=revision, cache_dir=cache_dir)
    original = as_pose_episode(source.load_episode(episode_index))
    measurements: dict[str, Any] = {}
    for requested_seconds in candidates:
        decision = choose_event_safe_head_trim(original, requested_seconds)
        trimmed = trim_pose_episode_head(original, decision.frame)
        smoothing = smooth_pose_episode(trimmed, method, parameters)
        filtered = with_smoothed_pose(trimmed, smoothing)
        result = retarget_pose_episode(
            filtered,
            position_tolerance_m=0.01,
            contact_position_tolerance_m=0.001,
            orientation_tolerance_rad=np.deg2rad(8.0),
            max_iterations=75,
        )
        both = result.arms[0].reachable & result.arms[1].reachable
        event_reachable = [
            result.arms[arm].reachable[result.event_indices[arm]] for arm in range(2)
        ]
        target_separation = np.linalg.norm(
            result.target_positions_world[:, 1] - result.target_positions_world[:, 0],
            axis=1,
        )
        measurements[str(requested_seconds)] = {
            "actual_trim_frames": decision.frame,
            "actual_trim_s": decision.seconds,
            "guard_clamped": decision.reason != "requested",
            "frames": len(filtered.timestamps),
            "both_frame_fraction": float(np.mean(both)),
            "all_frames_both_ik_reachable": bool(np.all(both)),
            "all_gripper_events_ik_reachable": bool(
                all(np.all(values) for values in event_reachable)
            ),
            "minimum_target_tcp_separation_mm": float(1000 * np.min(target_separation)),
        }
    return {"episode_index": episode_index, "candidates": measurements}
