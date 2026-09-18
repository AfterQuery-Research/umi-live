#!/usr/bin/env python3
"""Local-only frame-level diagnostics for the continuous IK quality gate.

Failed gate rows do not retain their joint arrays, so this script reproduces only
the selected final solve in a separate analysis directory. It never writes to the
dataset or to the quality-gate result cache.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from dataset_filtering.combined_pipeline import zero_pose_origins
from dataset_filtering.hf_dataset import LeRobotSource, as_pose_episode
from dataset_filtering.kinematics import YAMKinematics
from dataset_filtering.retarget import (
    DEFAULT_PAIR_SEPARATION_M,
    map_umi_targets_to_yam_world,
    yam_base_transforms,
)
from dataset_filtering.smoothing import smooth_pose_episode, with_smoothed_pose
from dataset_filtering.trajectory_optimization import (
    _COLLISION_RADII_M,
    _COLLISION_SAMPLE_FRACTIONS,
    _COLLISION_SEGMENTS,
    _SELF_COLLISION_SEGMENT_PAIRS,
    TrajectoryOptimizationConfig,
    _evaluate_trajectory,
    _numeric_collision_samples,
    _optimize_pose_episode_once,
    _orientation_error,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    return value


def _config(row: dict[str, Any]) -> TrajectoryOptimizationConfig:
    raw = dict(row["selected_result"]["config"])
    raw["contact_orientation_tolerance_rad"] = np.deg2rad(
        raw.pop("contact_orientation_tolerance_deg")
    )
    raw["tracking_orientation_tolerance_rad"] = np.deg2rad(
        raw.pop("tracking_orientation_tolerance_deg")
    )
    return TrajectoryOptimizationConfig(**raw)


def _apply_recorded_tail_freeze(episode: Any, row: dict[str, Any]) -> Any:
    positions = episode.positions.copy()
    quaternions = episode.quaternions.copy()
    arms = row["selected_result"]["tail_freeze"]["arms"]
    for arm, name in enumerate(("umi1_left", "umi2_right")):
        metadata = arms[name]
        if metadata["applied"]:
            cutoff = int(metadata["cutoff_frame"])
            positions[cutoff:, arm] = positions[cutoff, arm]
            quaternions[cutoff:, arm] = quaternions[cutoff, arm]
    return replace(episode, positions=positions, quaternions=quaternions)


def _clearance_profiles(
    links: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    samples = _numeric_collision_samples(links)
    radii = np.repeat(np.asarray(_COLLISION_RADII_M), len(_COLLISION_SAMPLE_FRACTIONS))
    left = samples[:, 0].reshape(len(links), -1, 3)
    right = samples[:, 1].reshape(len(links), -1, 3)
    inter_surface = (
        np.linalg.norm(left[:, :, None, :] - right[:, None, :, :], axis=-1)
        - (radii[:, None] + radii[None, :])[None, :, :]
    )
    inter = np.min(inter_surface, axis=(1, 2))
    inter_pair = np.argmin(inter_surface.reshape(len(links), -1), axis=1)

    segment_lookup = {segment: index for index, segment in enumerate(_COLLISION_SEGMENTS)}
    self_profile = np.full((len(links), 2), np.inf, dtype=np.float64)
    self_pair = np.full((len(links), 2), -1, dtype=np.int64)
    for arm in range(2):
        for pair_index, (first, second) in enumerate(_SELF_COLLISION_SEGMENT_PAIRS):
            first_points = samples[:, arm, segment_lookup[first]]
            second_points = samples[:, arm, segment_lookup[second]]
            surface = (
                np.linalg.norm(
                    first_points[:, :, None, :] - second_points[:, None, :, :], axis=-1
                )
                - (
                    _COLLISION_RADII_M[segment_lookup[first]]
                    + _COLLISION_RADII_M[segment_lookup[second]]
                )
            )
            pair_minimum = np.min(surface, axis=(1, 2))
            improved = pair_minimum < self_profile[:, arm]
            self_profile[improved, arm] = pair_minimum[improved]
            self_pair[improved, arm] = pair_index
    return inter, self_profile, inter_pair, self_pair


def _runs(mask: np.ndarray, values: np.ndarray, timestamps: np.ndarray) -> list[dict[str, Any]]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    splits = np.flatnonzero(np.diff(indices) > 1) + 1
    chunks = np.split(indices, splits)
    n = len(mask)
    result = []
    for chunk in chunks:
        start, end = int(chunk[0]), int(chunk[-1])
        local = values[chunk]
        worst_offset = int(np.argmax(local))
        result.append(
            {
                "start_frame": start,
                "end_frame": end,
                "frames": len(chunk),
                "start_s": float(timestamps[start] - timestamps[0]),
                "end_s": float(timestamps[end] - timestamps[0]),
                "start_fraction": start / max(1, n - 1),
                "end_fraction": end / max(1, n - 1),
                "worst_frame": int(chunk[worst_offset]),
                "worst_value": float(local[worst_offset]),
            }
        )
    return result


def _load_selected_trajectory(
    dataset: str, row_path: str, array_path: str | None
) -> tuple[Any, Any, TrajectoryOptimizationConfig, bool]:
    row = json.loads(Path(row_path).read_text())
    source = LeRobotSource(dataset)
    cfg = _config(row)
    original_pose = as_pose_episode(source.load_episode(row["source_episode_index"]))
    pose = original_pose if cfg.aligned_shared_frame else zero_pose_origins(original_pose)
    smoothing = row["selected_smoothing"]
    smoothed = smooth_pose_episode(
        pose, str(smoothing["method"]), dict(smoothing["parameters"])
    )
    filtered = with_smoothed_pose(pose, smoothed)
    reproduced = not row["accepted"]
    if reproduced:
        final_target = _apply_recorded_tail_freeze(filtered, row)
        result = _optimize_pose_episode_once(
            final_target, config=cfg, separation_m=DEFAULT_PAIR_SEPARATION_M
        )
    else:
        with np.load(str(array_path)) as artifact:
            q = np.asarray(artifact["q"], dtype=np.float64)
        final_target = _apply_recorded_tail_freeze(filtered, row)
        target_positions, target_quaternions = map_umi_targets_to_yam_world(
            final_target, aligned_shared_frame=cfg.aligned_shared_frame
        )
        links, achieved_positions, achieved_quaternions = _evaluate_trajectory(
            q, YAMKinematics(), yam_base_transforms(DEFAULT_PAIR_SEPARATION_M)
        )
        position_error = np.linalg.norm(achieved_positions - target_positions, axis=-1)
        orientation_error = _orientation_error(achieved_quaternions, target_quaternions)
        result = type("SavedResult", (), {})()
        result.q = q
        result.link_points_world = links
        result.achieved_positions_world = achieved_positions
        result.achieved_quaternions_world = achieved_quaternions
        result.target_positions_world = target_positions
        result.target_quaternions_world = target_quaternions
        result.position_error_m = position_error
        result.orientation_error_rad = orientation_error
        result.event_indices = tuple(
            np.asarray(anchor_frames, dtype=np.int64)
            for anchor_frames in (
                [a["frame"] for a in row.get("contact_anchors", []) if a["arm"] == 0],
                [a["frame"] for a in row.get("contact_anchors", []) if a["arm"] == 1],
            )
        )
        result.contact_anchors = ()
    return pose, result, cfg, reproduced


def _diagnose(dataset: str, row_path: str, output_dir: str) -> dict[str, Any]:
    row_file = Path(row_path)
    row = json.loads(row_file.read_text())
    array_path = row_file.with_suffix(".npz") if row["accepted"] else None
    pose, result, cfg, reproduced = _load_selected_trajectory(
        dataset, row_path, str(array_path) if array_path else None
    )
    inter, self_profile, inter_pair, self_pair = _clearance_profiles(
        result.link_points_world
    )
    tracking_position = np.max(result.position_error_m, axis=1)
    tracking_orientation = np.max(result.orientation_error_rad, axis=1)
    tracking_bad = np.any(
        (result.position_error_m > cfg.tracking_position_tolerance_m)
        | (result.orientation_error_rad > cfg.tracking_orientation_tolerance_rad),
        axis=1,
    )
    position_bad = result.position_error_m > cfg.tracking_position_tolerance_m
    orientation_bad = result.orientation_error_rad > cfg.tracking_orientation_tolerance_rad
    tracking_bad_by_arm = position_bad | orientation_bad
    contact_rows = []
    for anchor in result.contact_anchors:
        contact_rows.append(
            {
                "arm": int(anchor.arm),
                "frame": int(anchor.frame),
                "fraction": int(anchor.frame) / max(1, len(pose.timestamps) - 1),
                "feasible": bool(anchor.feasible),
                "position_error_mm": 1000 * float(anchor.position_error_m),
                "orientation_error_deg": float(np.degrees(anchor.orientation_error_rad)),
            }
        )
    target_tool_distance = np.linalg.norm(
        result.target_positions_world[:, 0] - result.target_positions_world[:, 1], axis=1
    )
    achieved_tool_distance = np.linalg.norm(
        result.achieved_positions_world[:, 0] - result.achieved_positions_world[:, 1], axis=1
    )
    expected = row["selected_result"]
    observed = {
        "position_error_max_mm": 1000 * float(np.max(result.position_error_m)),
        "orientation_error_max_deg": float(np.degrees(np.max(result.orientation_error_rad))),
        "minimum_interarm_surface_clearance_mm": 1000 * float(np.min(inter)),
        "minimum_self_surface_clearance_mm": 1000 * float(np.min(self_profile)),
    }
    comparison = {
        key: {
            "expected": float(
                expected["validation"][key]
                if key.startswith("minimum_")
                else expected[key]
            ),
            "observed": value,
        }
        for key, value in observed.items()
    }
    for item in comparison.values():
        item["absolute_delta"] = abs(item["observed"] - item["expected"])

    frame_count = len(pose.timestamps)
    inter_min_frame = int(np.argmin(inter))
    self_min_flat = int(np.argmin(self_profile))
    self_frame, self_arm = np.unravel_index(self_min_flat, self_profile.shape)
    collision_samples_per_arm = len(_COLLISION_SEGMENTS) * len(
        _COLLISION_SAMPLE_FRACTIONS
    )
    inter_pair_value = int(inter_pair[inter_min_frame])
    inter_left_sample = inter_pair_value // collision_samples_per_arm
    inter_right_sample = inter_pair_value % collision_samples_per_arm
    self_pair_value = int(self_pair[self_frame, self_arm])
    self_segments = _SELF_COLLISION_SEGMENT_PAIRS[self_pair_value]
    position_max_flat = int(np.argmax(result.position_error_m))
    position_frame, position_arm = np.unravel_index(
        position_max_flat, result.position_error_m.shape
    )
    orientation_max_flat = int(np.argmax(result.orientation_error_rad))
    orientation_frame, orientation_arm = np.unravel_index(
        orientation_max_flat, result.orientation_error_rad.shape
    )
    diagnostic = {
        "episode_index": row["source_episode_index"],
        "accepted": row["accepted"],
        "selected_profile": row["selected_profile"],
        "frames": frame_count,
        "duration_s": float(pose.timestamps[-1] - pose.timestamps[0]),
        "reproduced_failed_solve": reproduced,
        "reproduction_comparison": comparison,
        "interarm": {
            "minimum_clearance_mm": 1000 * float(inter[inter_min_frame]),
            "minimum_frame": inter_min_frame,
            "minimum_fraction": inter_min_frame / max(1, frame_count - 1),
            "minimum_time_s": float(pose.timestamps[inter_min_frame] - pose.timestamps[0]),
            "target_tool_distance_mm_at_minimum": 1000
            * float(target_tool_distance[inter_min_frame]),
            "achieved_tool_distance_mm_at_minimum": 1000
            * float(achieved_tool_distance[inter_min_frame]),
            "sample_pair_flat_index": int(inter_pair[inter_min_frame]),
            "minimum_link_segments": [
                int(_COLLISION_SEGMENTS[inter_left_sample // len(_COLLISION_SAMPLE_FRACTIONS)]),
                int(_COLLISION_SEGMENTS[inter_right_sample // len(_COLLISION_SAMPLE_FRACTIONS)]),
            ],
            "unsafe_runs": _runs(
                inter < cfg.interarm_collision_margin_m,
                cfg.interarm_collision_margin_m - inter,
                pose.timestamps,
            ),
        },
        "self_collision": {
            "minimum_clearance_mm": 1000 * float(self_profile[self_frame, self_arm]),
            "minimum_frame": int(self_frame),
            "minimum_arm": int(self_arm),
            "minimum_link_segments": [int(value) for value in self_segments],
            "minimum_fraction": int(self_frame) / max(1, frame_count - 1),
            "unsafe_runs": _runs(
                np.any(self_profile < cfg.self_collision_margin_m, axis=1),
                np.max(cfg.self_collision_margin_m - self_profile, axis=1),
                pose.timestamps,
            ),
        },
        "tracking": {
            "position_max_mm": 1000 * float(result.position_error_m[position_frame, position_arm]),
            "position_max_frame": int(position_frame),
            "position_max_arm": int(position_arm),
            "position_max_fraction": int(position_frame) / max(1, frame_count - 1),
            "orientation_max_deg": float(
                np.degrees(result.orientation_error_rad[orientation_frame, orientation_arm])
            ),
            "orientation_max_frame": int(orientation_frame),
            "orientation_max_arm": int(orientation_arm),
            "orientation_max_fraction": int(orientation_frame) / max(1, frame_count - 1),
            "bad_runs": _runs(
                tracking_bad,
                np.maximum(
                    tracking_position / cfg.tracking_position_tolerance_m,
                    tracking_orientation / cfg.tracking_orientation_tolerance_rad,
                ),
                pose.timestamps,
            ),
            "position_bad_runs": _runs(
                np.any(position_bad, axis=1),
                np.max(
                    result.position_error_m / cfg.tracking_position_tolerance_m,
                    axis=1,
                ),
                pose.timestamps,
            ),
            "orientation_bad_runs": _runs(
                np.any(orientation_bad, axis=1),
                np.max(
                    result.orientation_error_rad / cfg.tracking_orientation_tolerance_rad,
                    axis=1,
                ),
                pose.timestamps,
            ),
            "arms": [
                {
                    "arm": arm,
                    "bad_frames": int(np.count_nonzero(tracking_bad_by_arm[:, arm])),
                    "position_bad_frames": int(np.count_nonzero(position_bad[:, arm])),
                    "orientation_bad_frames": int(np.count_nonzero(orientation_bad[:, arm])),
                    "bad_runs": _runs(
                        tracking_bad_by_arm[:, arm],
                        np.maximum(
                            result.position_error_m[:, arm]
                            / cfg.tracking_position_tolerance_m,
                            result.orientation_error_rad[:, arm]
                            / cfg.tracking_orientation_tolerance_rad,
                        ),
                        pose.timestamps,
                    ),
                }
                for arm in range(2)
            ],
        },
        "contacts": contact_rows,
        "target_tool_distance": {
            "minimum_mm": 1000 * float(np.min(target_tool_distance)),
            "minimum_frame": int(np.argmin(target_tool_distance)),
            "minimum_fraction": int(np.argmin(target_tool_distance))
            / max(1, frame_count - 1),
            "p10_mm": 1000 * float(np.percentile(target_tool_distance, 10)),
        },
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / f"episode-{row['source_episode_index']:03d}.json"
    destination.write_text(json.dumps(_jsonable(diagnostic), indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        output / f"episode-{row['source_episode_index']:03d}.npz",
        q=np.asarray(result.q, dtype=np.float32),
        interarm_clearance_m=inter.astype(np.float32),
        self_clearance_m=self_profile.astype(np.float32),
        position_error_m=np.asarray(result.position_error_m, dtype=np.float32),
        orientation_error_rad=np.asarray(result.orientation_error_rad, dtype=np.float32),
        target_tool_distance_m=target_tool_distance.astype(np.float32),
        achieved_tool_distance_m=achieved_tool_distance.astype(np.float32),
        timestamps=np.asarray(pose.timestamps, dtype=np.float64),
        target_positions_world=np.asarray(result.target_positions_world, dtype=np.float32),
        achieved_positions_world=np.asarray(
            result.achieved_positions_world, dtype=np.float32
        ),
    )
    return diagnostic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", nargs="*", type=int)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    rows = []
    requested = set(args.episodes or [])
    for row_path in sorted(args.results.glob("input-*.json")):
        row = json.loads(row_path.read_text())
        if requested and row["source_episode_index"] not in requested:
            continue
        rows.append(row_path)
    if not rows:
        raise SystemExit("no matching completed result rows")

    with ProcessPoolExecutor(max_workers=min(args.workers, len(rows))) as pool:
        futures = {
            pool.submit(_diagnose, args.dataset, str(row), str(args.output)): row
            for row in rows
        }
        for future in as_completed(futures):
            row = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                print(f"ERROR {row.name}: {exc!r}", flush=True)
                continue
            deltas = result["reproduction_comparison"]
            print(
                f"episode {result['episode_index']}: done "
                f"interarm={result['interarm']['minimum_clearance_mm']:.1f} mm "
                f"tracking_runs={len(result['tracking']['bad_runs'])} "
                f"max_reproduction_delta={max(x['absolute_delta'] for x in deltas.values()):.4g}",
                flush=True,
            )


if __name__ == "__main__":
    main()
