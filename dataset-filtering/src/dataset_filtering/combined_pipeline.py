from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from dataset_filtering.export_dataset import optimized_world_poses_to_umi_state
from dataset_filtering.hf_dataset import LeRobotSource, as_pose_episode
from dataset_filtering.models import PoseEpisode, SmoothingResult
from dataset_filtering.retarget import map_umi_targets_to_yam_world, retarget_pose_episode
from dataset_filtering.smoothing import (
    benchmark_smoothers,
    smooth_pose_episode,
    with_smoothed_pose,
)
from dataset_filtering.trajectory_optimization import (
    TrajectoryOptimizationConfig,
    optimize_pose_episode,
)

PIPELINE_VERSION = "combined_aligned_shared_frame_v1_i2rt_220mm_umi_grasp_frame"
KINEMATICS_PROVENANCE = {
    "implementation": "AfterQuery-Research/i2rt",
    "revision": "7ed46f4e4e316133a0c39aa6cf34a73d2718e850",
    "arm_model": "i2rt/robot_models/arm/yam/yam.xml",
    "gripper_model": "i2rt/robot_models/gripper/linear_4310/linear_4310.xml",
    "grasp_offset_mm": 220.0,
    "grasp_axes": "UMI-compatible: +X up, +Y right, +Z forward",
}
ORIGINAL_REPO_ID = "brandonyang/dual-lidar-umi"
RELATIVE_REPO_ID = "brandonyang/dual-lidar-umi-relative"
LEFT_GRIPPER_WIDTH_LIMITS_MM = (40.376321, 122.999636)
RIGHT_GRIPPER_WIDTH_LIMITS_MM = (40.128507, 122.685643)
SAFE_SMOOTHING_CANDIDATES: tuple[tuple[str, dict[str, float | int]], ...] = (
    ("butterworth", {"cutoff_hz": 4.0, "order": 3}),
    ("butterworth", {"cutoff_hz": 3.0, "order": 3}),
    ("butterworth", {"cutoff_hz": 2.0, "order": 3}),
    ("butterworth", {"cutoff_hz": 1.5, "order": 3}),
    ("butterworth", {"cutoff_hz": 1.0, "order": 3}),
)


@dataclass(frozen=True)
class CombinedSource:
    repo_id: str
    revision: str


@dataclass(frozen=True)
class CombinedWorkItem:
    combined_input_index: int
    source_dataset: str
    source_revision: str
    source_episode_index: int
    cache_dir: str
    result_dir: str

    @property
    def signature(self) -> str:
        payload = json.dumps(
            {
                **asdict(self),
                "result_dir": None,
                "pipeline_version": PIPELINE_VERSION,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def zero_pose_origins(episode: PoseEpisode) -> PoseEpisode:
    """Express each UMI trajectory in its own first-pose SE(3) frame."""
    positions = episode.positions - episode.positions[0:1]
    quaternions = np.empty_like(episode.quaternions, dtype=np.float64)
    for arm in range(2):
        rotations = Rotation.from_quat(episode.quaternions[:, arm])
        relative = rotations[0].inv() * rotations
        if relative is NotImplemented:
            raise TypeError("could not remove UMI initial orientation")
        quaternions[:, arm] = relative.as_quat()
    positions[0] = 0.0
    quaternions[0, :, :3] = 0.0
    quaternions[0, :, 3] = 1.0
    return replace(episode, positions=positions, quaternions=quaternions)


def verify_zeroing_invariance(episode: PoseEpisode) -> dict[str, Any]:
    """Measure both mathematical and full auto-smoothing origin invariance."""
    zeroed = zero_pose_origins(episode)
    raw_positions, raw_quaternions = map_umi_targets_to_yam_world(episode)
    zero_positions, zero_quaternions = map_umi_targets_to_yam_world(zeroed)
    raw_smoothing = benchmark_smoothers(episode)[0]
    zero_smoothing = benchmark_smoothers(zeroed)[0]
    smoother_selection_same = (raw_smoothing.method, raw_smoothing.parameters) == (
        zero_smoothing.method,
        zero_smoothing.parameters,
    )
    raw_smoothed_positions, raw_smoothed_quaternions = map_umi_targets_to_yam_world(
        with_smoothed_pose(episode, raw_smoothing)
    )
    zero_smoothed_positions, zero_smoothed_quaternions = map_umi_targets_to_yam_world(
        with_smoothed_pose(zeroed, zero_smoothing)
    )
    # Applying the zeroed episode's chosen configuration to both inputs isolates
    # the coordinate transform itself from any auto-selector tie or roundoff.
    fixed_raw_smoothing = smooth_pose_episode(
        episode, zero_smoothing.method, zero_smoothing.parameters
    )
    fixed_raw_positions, fixed_raw_quaternions = map_umi_targets_to_yam_world(
        with_smoothed_pose(episode, fixed_raw_smoothing)
    )
    raw_rotation_error = _quaternion_angle(raw_quaternions, zero_quaternions)
    smooth_rotation_error = _quaternion_angle(raw_smoothed_quaternions, zero_smoothed_quaternions)
    fixed_smooth_rotation_error = _quaternion_angle(
        fixed_raw_quaternions, zero_smoothed_quaternions
    )
    return {
        "raw_smoothing_method": raw_smoothing.method,
        "raw_smoothing_parameters": raw_smoothing.parameters,
        "zeroed_smoothing_method": zero_smoothing.method,
        "zeroed_smoothing_parameters": zero_smoothing.parameters,
        "smoother_selection_same": smoother_selection_same,
        "raw_position_delta_max_m": float(np.max(np.abs(raw_positions - zero_positions))),
        "raw_rotation_delta_max_rad": float(np.max(raw_rotation_error)),
        "smoothed_position_delta_max_m": float(
            np.max(np.abs(raw_smoothed_positions - zero_smoothed_positions))
        ),
        "smoothed_rotation_delta_max_rad": float(np.max(smooth_rotation_error)),
        "fixed_smoother_position_delta_max_m": float(
            np.max(np.abs(fixed_raw_positions - zero_smoothed_positions))
        ),
        "fixed_smoother_rotation_delta_max_rad": float(np.max(fixed_smooth_rotation_error)),
    }


def process_combined_sources(
    *,
    sources: tuple[CombinedSource, ...],
    destination: str | Path,
    workers: int = 8,
) -> dict[str, Any]:
    """Run the unchanged-gate baseline/dense/safe-smoothing solver cascade."""
    if not sources:
        raise ValueError("at least one source dataset is required")
    result_root = Path(destination).expanduser().resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    cache_root = result_root / "episodes"
    cache_root.mkdir(parents=True, exist_ok=True)

    items: list[CombinedWorkItem] = []
    source_summaries: list[dict[str, Any]] = []
    combined_index = 0
    for source_spec in sources:
        source = LeRobotSource(source_spec.repo_id, revision=source_spec.revision)
        source.info()
        summaries = source.episode_summaries()
        indices = [summary.episode_index for summary in summaries]
        source.prefetch_episode_data(indices)
        for summary in summaries:
            items.append(
                CombinedWorkItem(
                    combined_input_index=combined_index,
                    source_dataset=source_spec.repo_id,
                    source_revision=source_spec.revision,
                    source_episode_index=summary.episode_index,
                    cache_dir=str(source.cache_dir),
                    result_dir=str(cache_root),
                )
            )
            combined_index += 1
        source_summaries.append(
            {
                "repo_id": source_spec.repo_id,
                "revision": source_spec.revision,
                "episodes": len(summaries),
                "frames": sum(summary.length for summary in summaries),
            }
        )

    started = monotonic()
    rows: list[dict[str, Any]] = []
    worker_count = max(1, min(workers, len(items), os.cpu_count() or workers))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_items = {executor.submit(_process_work_item, asdict(item)): item for item in items}
        for completed, future in enumerate(as_completed(future_items), start=1):
            item = future_items[future]
            try:
                row = future.result()
            except Exception as exc:  # pragma: no cover - protects long batch progress
                row = _exception_row(item, exc)
                _write_result_row(Path(item.result_dir), row)
            rows.append(row)
            status = "kept" if row["accepted"] else "rejected"
            cache_note = " cached" if row.get("cache_hit") else ""
            print(
                f"combined filtering: {completed}/{len(items)} {status}{cache_note} "
                f"input {item.combined_input_index} "
                f"({item.source_dataset} episode {item.source_episode_index})",
                flush=True,
            )
    rows.sort(key=lambda row: row["combined_input_index"])
    report = _aggregate_report(rows, source_summaries, worker_count, monotonic() - started)
    report_path = result_root / "pipeline-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def process_combined_source_shard(
    *,
    source_spec: CombinedSource,
    combined_offset: int,
    source_episode_from: int,
    source_episode_to: int,
    destination: str | Path,
    workers: int = 4,
) -> dict[str, Any]:
    """Precompute a nonoverlapping high-index shard into the shared result cache."""
    if source_episode_to < source_episode_from:
        raise ValueError("source_episode_to must be >= source_episode_from")
    result_root = Path(destination).expanduser().resolve()
    cache_root = result_root / "episodes"
    cache_root.mkdir(parents=True, exist_ok=True)
    source = LeRobotSource(source_spec.repo_id, revision=source_spec.revision)
    source.info()
    summaries = source.episode_summaries()
    indexed_summaries = list(enumerate(summaries))
    selected = [
        (ordinal, summary)
        for ordinal, summary in indexed_summaries
        if source_episode_from <= summary.episode_index <= source_episode_to
    ]
    if not selected:
        raise ValueError("source shard contains no episodes")
    source.prefetch_episode_data([summary.episode_index for _, summary in selected])
    items = [
        CombinedWorkItem(
            combined_input_index=combined_offset + ordinal,
            source_dataset=source_spec.repo_id,
            source_revision=source_spec.revision,
            source_episode_index=summary.episode_index,
            cache_dir=str(source.cache_dir),
            result_dir=str(cache_root),
        )
        for ordinal, summary in selected
    ]
    rows: list[dict[str, Any]] = []
    worker_count = max(1, min(workers, len(items), os.cpu_count() or workers))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_items = {executor.submit(_process_work_item, asdict(item)): item for item in items}
        for completed, future in enumerate(as_completed(future_items), start=1):
            item = future_items[future]
            try:
                row = future.result()
            except Exception as exc:  # pragma: no cover
                row = _exception_row(item, exc)
                _write_result_row(Path(item.result_dir), row)
            rows.append(row)
            print(
                f"combined shard: {completed}/{len(items)} "
                f"{'kept' if row['accepted'] else 'rejected'} input "
                f"{item.combined_input_index} ({source_spec.repo_id} episode "
                f"{item.source_episode_index})",
                flush=True,
            )
    return {
        "source": asdict(source_spec),
        "source_episode_from": source_episode_from,
        "source_episode_to": source_episode_to,
        "episodes": len(rows),
        "accepted": sum(row["accepted"] for row in rows),
        "rejected": sum(not row["accepted"] for row in rows),
        "workers": worker_count,
    }


_WORKER_SOURCES: dict[tuple[str, str, str], LeRobotSource] = {}


def _worker_source(repo_id: str, revision: str, cache_dir: str) -> LeRobotSource:
    key = (repo_id, revision, cache_dir)
    if key not in _WORKER_SOURCES:
        _WORKER_SOURCES[key] = LeRobotSource(repo_id, revision=revision, cache_dir=cache_dir)
    return _WORKER_SOURCES[key]


def _process_work_item(raw_item: dict[str, Any]) -> dict[str, Any]:
    item = CombinedWorkItem(**raw_item)
    result_dir = Path(item.result_dir)
    row_path = result_dir / f"input-{item.combined_input_index:03d}.json"
    array_path = result_dir / f"input-{item.combined_input_index:03d}.npz"
    if row_path.is_file():
        cached = json.loads(row_path.read_text(encoding="utf-8"))
        if cached.get("signature") == item.signature and (
            not cached.get("accepted") or array_path.is_file()
        ):
            cached["cache_hit"] = True
            return cached

    source = _worker_source(item.source_dataset, item.source_revision, item.cache_dir)
    original = as_pose_episode(source.load_episode(item.source_episode_index))
    episode = original
    initial_relative_translation = episode.positions[0, 1] - episode.positions[0, 0]
    initial_separation_m = float(np.linalg.norm(initial_relative_translation))
    if initial_separation_m < 0.05:
        raise RuntimeError(
            f"episode {item.source_episode_index}: expected aligned shared-frame poses, "
            f"but initial UMI separation is only {1000 * initial_separation_m:.1f} mm"
        )
    initial_relative_rotation = (
        Rotation.from_quat(episode.quaternions[0, 0]).inv()
        * Rotation.from_quat(episode.quaternions[0, 1])
    )
    if initial_relative_rotation is NotImplemented:
        raise TypeError("could not compute initial aligned UMI relative orientation")
    shared_frame = {
        "mode": "aligned_shared_umi1_initial",
        "initial_relative_translation_umi_m": initial_relative_translation.tolist(),
        "initial_separation_mm": 1000 * initial_separation_m,
        "initial_relative_orientation_deg": float(
            np.degrees(initial_relative_rotation.magnitude())
        ),
        "mapping": "preserve complete initial relative pose; center TCP pair over fixed bases",
    }

    smoothing = benchmark_smoothers(episode)[0]
    filtered = with_smoothed_pose(episode, smoothing)
    attempts: list[dict[str, Any]] = []
    baseline_config = TrajectoryOptimizationConfig(aligned_shared_frame=True)
    baseline = optimize_pose_episode(filtered, config=baseline_config)
    attempts.append(_attempt_summary("baseline_v3", smoothing, baseline))
    selected = baseline
    selected_smoothing = smoothing
    selected_filtered = filtered
    selected_profile = "baseline_v3"

    if not baseline.metadata["continuously_replayable"]:
        dense_config = replace(
            baseline_config,
            objective_stride=2,
            collision_stride=12,
            solver_max_iterations=500,
        )
        dense = optimize_pose_episode(filtered, config=dense_config)
        attempts.append(_attempt_summary("dense_objective", smoothing, dense))
        selected = dense
        selected_profile = "dense_objective"

        if (
            not dense.metadata["continuously_replayable"]
            and not dense.metadata["validation"]["passes_tracking_accuracy"]
        ):
            safe = _select_safe_smoothing_improvement(episode, smoothing)
            if safe is not None:
                safe_filtered = with_smoothed_pose(episode, safe)
                safe_result = optimize_pose_episode(safe_filtered, config=dense_config)
                safe_profile = f"dense_safe_smoothing_{safe.parameters['cutoff_hz']}hz"
                attempts.append(_attempt_summary(safe_profile, safe, safe_result))
                if safe_result.metadata["continuously_replayable"]:
                    selected = safe_result
                    selected_smoothing = safe
                    selected_filtered = safe_filtered
                    selected_profile = safe_profile
                elif _failed_result_key(safe_result) > _failed_result_key(selected):
                    selected = safe_result
                    selected_smoothing = safe
                    selected_filtered = safe_filtered
                    selected_profile = safe_profile

    accepted = bool(selected.metadata["continuously_replayable"])
    row = {
        "signature": item.signature,
        "pipeline_version": PIPELINE_VERSION,
        "combined_input_index": item.combined_input_index,
        "source_dataset": item.source_dataset,
        "source_revision": item.source_revision,
        "source_episode_index": item.source_episode_index,
        "frames": len(episode.timestamps),
        "accepted": accepted,
        "selected_profile": selected_profile,
        "selected_smoothing": _smoothing_summary(selected_smoothing),
        "selected_result": _result_summary(selected),
        "attempts": attempts,
        "shared_frame_mapping": shared_frame,
        "cache_hit": False,
    }
    if accepted:
        grippers = (
            episode.grippers
            if episode.grippers is not None
            else np.zeros((len(episode.timestamps), 2), dtype=np.float64)
        )
        state = optimized_world_poses_to_umi_state(
            selected.achieved_positions_world,
            selected.achieved_quaternions_world,
            selected_filtered.positions,
            selected_filtered.quaternions,
            aligned_shared_frame=True,
        )
        normalized_grippers = normalize_gripper_widths(grippers)
        np.savez_compressed(
            array_path,
            cartesian_state=state.astype(np.float32),
            q=selected.q.astype(np.float32),
            normalized_grippers=normalized_grippers.astype(np.float32),
        )
    _write_result_row(result_dir, row)
    return row


def normalize_gripper_widths(grippers_mm: np.ndarray) -> np.ndarray:
    grippers = np.asarray(grippers_mm, dtype=np.float64)
    if grippers.ndim != 2 or grippers.shape[1] != 2:
        raise ValueError("grippers must have shape (frames, 2)")
    if np.min(grippers) >= -1e-6 and np.max(grippers) <= 1.0 + 1e-6:
        return np.clip(grippers, 0.0, 1.0)
    result = np.empty_like(grippers)
    for arm, (closed, opened) in enumerate(
        (LEFT_GRIPPER_WIDTH_LIMITS_MM, RIGHT_GRIPPER_WIDTH_LIMITS_MM)
    ):
        result[:, arm] = np.clip((grippers[:, arm] - closed) / (opened - closed), 0.0, 1.0)
    return result


def joint_observation_state(q: np.ndarray, normalized_grippers: np.ndarray) -> np.ndarray:
    q_values = np.asarray(q, dtype=np.float64)
    grippers = np.asarray(normalized_grippers, dtype=np.float64)
    if q_values.ndim != 3 or q_values.shape[1:] != (2, 6):
        raise ValueError("q must have shape (frames, 2, 6)")
    if grippers.shape != (len(q_values), 2):
        raise ValueError("normalized_grippers must have shape (frames, 2)")
    return np.column_stack((q_values[:, 0], grippers[:, 0], q_values[:, 1], grippers[:, 1])).astype(
        np.float32
    )


def _select_safe_smoothing_improvement(
    episode: PoseEpisode,
    baseline_smoothing: SmoothingResult,
    *,
    aligned_shared_frame: bool = True,
) -> SmoothingResult | None:
    baseline_proxy = _framewise_proxy(
        episode, baseline_smoothing, aligned_shared_frame=aligned_shared_frame
    )
    candidates: list[tuple[SmoothingResult, dict[str, Any]]] = []
    for method, parameters in SAFE_SMOOTHING_CANDIDATES:
        smoothing = smooth_pose_episode(episode, method, parameters)
        proxy = _framewise_proxy(
            episode, smoothing, aligned_shared_frame=aligned_shared_frame
        )
        metrics = smoothing.metrics
        safe = bool(
            metrics["translation_distortion_p95_mm"] <= 10.0
            and metrics["rotation_distortion_rms_deg"] <= 5.0
            and metrics["direction_cosine_median"] >= 0.98
            and metrics["event_translation_error_max_mm"] <= 1e-6
            and metrics["event_rotation_error_max_deg"] <= 1e-6
            and proxy["all_strict_events_ik_reachable"]
        )
        if safe:
            candidates.append((smoothing, proxy))
    if not candidates:
        return None
    best_smoothing, best_proxy = max(
        candidates,
        key=lambda value: (
            value[1]["all_frames_both_ik_reachable"],
            value[1]["both_frame_fraction"],
            -value[0].metrics["translation_distortion_p95_mm"],
            -value[0].score,
        ),
    )
    if best_proxy["both_frame_fraction"] <= baseline_proxy["both_frame_fraction"] + 1e-6:
        return None
    return best_smoothing


def _framewise_proxy(
    episode: PoseEpisode,
    smoothing: SmoothingResult,
    *,
    aligned_shared_frame: bool = True,
) -> dict[str, Any]:
    filtered = with_smoothed_pose(episode, smoothing)
    result = retarget_pose_episode(
        filtered,
        position_tolerance_m=0.01,
        contact_position_tolerance_m=0.001,
        orientation_tolerance_rad=np.deg2rad(8.0),
        max_iterations=75,
        aligned_shared_frame=aligned_shared_frame,
    )
    both = result.arms[0].reachable & result.arms[1].reachable
    strict: list[np.ndarray] = []
    for arm in range(2):
        events = result.event_indices[arm]
        strict.append(
            (result.arms[arm].position_error_m[events] <= 0.001)
            & (result.arms[arm].orientation_error_rad[events] <= np.deg2rad(2.0))
        )
    return {
        "both_frame_fraction": float(np.mean(both)),
        "all_frames_both_ik_reachable": bool(np.all(both)),
        "all_strict_events_ik_reachable": bool(all(np.all(value) for value in strict)),
    }


def _attempt_summary(profile: str, smoothing: SmoothingResult, result: Any) -> dict[str, Any]:
    return {
        "profile": profile,
        "smoothing": _smoothing_summary(smoothing),
        "result": _result_summary(result),
    }


def _smoothing_summary(smoothing: SmoothingResult) -> dict[str, Any]:
    return {
        "method": smoothing.method,
        "parameters": smoothing.parameters,
        "score": float(smoothing.score),
        "metrics": smoothing.metrics,
    }


def _result_summary(result: Any) -> dict[str, Any]:
    metadata = result.metadata
    validation = metadata["validation"]
    return {
        "solver_success": bool(metadata["solver"]["success"]),
        "ik_trajectory_solvable": bool(metadata["ik_trajectory_solvable"]),
        "continuously_replayable": bool(metadata["continuously_replayable"]),
        "position_error_max_mm": float(1000 * np.max(result.position_error_m)),
        "orientation_error_max_deg": float(np.degrees(np.max(result.orientation_error_rad))),
        "flow_direction_cosine_median": float(metadata["tracking"]["flow_direction_cosine_median"]),
        "validation": {
            key: bool(value) if key.startswith("passes_") else value
            for key, value in validation.items()
        },
        "tail_freeze": metadata["tail_freeze"],
        "secondary_objective": metadata["secondary_objective"],
        "timing": metadata["timing"],
        "config": metadata["config"],
    }


def _failed_result_key(result: Any) -> tuple[float, ...]:
    validation = result.metadata["validation"]
    passes = sum(bool(value) for key, value in validation.items() if key.startswith("passes_"))
    clearance = min(
        float(validation["minimum_interarm_surface_clearance_mm"]) - 5.0,
        float(validation["minimum_self_surface_clearance_mm"]) - 1.0,
    )
    tracking = max(
        float(np.max(result.position_error_m)) / 0.01,
        float(np.max(result.orientation_error_rad)) / np.deg2rad(8.0),
    )
    return float(passes), clearance, -tracking


def _exception_row(item: CombinedWorkItem, exc: Exception) -> dict[str, Any]:
    return {
        "signature": item.signature,
        "pipeline_version": PIPELINE_VERSION,
        "combined_input_index": item.combined_input_index,
        "source_dataset": item.source_dataset,
        "source_revision": item.source_revision,
        "source_episode_index": item.source_episode_index,
        "frames": None,
        "accepted": False,
        "selected_profile": "pipeline_error",
        "selected_smoothing": None,
        "selected_result": None,
        "attempts": [],
        "shared_frame_mapping": None,
        "pipeline_error": f"{type(exc).__name__}: {exc}",
        "cache_hit": False,
    }


def _write_result_row(result_dir: Path, row: dict[str, Any]) -> None:
    path = result_dir / f"input-{int(row['combined_input_index']):03d}.json"
    path.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")


def _aggregate_report(
    rows: list[dict[str, Any]],
    sources: list[dict[str, Any]],
    workers: int,
    wall_runtime_s: float,
) -> dict[str, Any]:
    accepted = [row for row in rows if row["accepted"]]
    profiles = Counter(row["selected_profile"] for row in accepted)
    failure_signatures: Counter[tuple[str, ...]] = Counter()
    pipeline_errors = 0
    for row in rows:
        if row["accepted"]:
            continue
        if row.get("pipeline_error"):
            pipeline_errors += 1
            failure_signatures[("pipeline_error",)] += 1
            continue
        validation = row["selected_result"]["validation"]
        failure_signatures[
            tuple(
                key.removeprefix("passes_")
                for key, value in validation.items()
                if key.startswith("passes_") and not value
            )
        ] += 1
    source_results = []
    for source in sources:
        source_rows = [row for row in rows if row["source_dataset"] == source["repo_id"]]
        source_results.append(
            {
                **source,
                "accepted": sum(row["accepted"] for row in source_rows),
                "rejected": sum(not row["accepted"] for row in source_rows),
            }
        )
    invariance_values = [
        max(
            row["zero_origin_invariance"]["raw_position_delta_max_m"],
            row["zero_origin_invariance"]["raw_rotation_delta_max_rad"],
            row["zero_origin_invariance"]["fixed_smoother_position_delta_max_m"],
            row["zero_origin_invariance"]["fixed_smoother_rotation_delta_max_rad"],
        )
        for row in rows
        if row.get("zero_origin_invariance") is not None
    ]
    auto_invariance_values = [
        max(
            row["zero_origin_invariance"]["smoothed_position_delta_max_m"],
            row["zero_origin_invariance"]["smoothed_rotation_delta_max_rad"],
        )
        for row in rows
        if row.get("zero_origin_invariance") is not None
    ]
    smoother_selection_mismatches = sum(
        not row["zero_origin_invariance"]["smoother_selection_same"]
        for row in rows
        if row.get("zero_origin_invariance") is not None
    )
    return {
        "pipeline_version": PIPELINE_VERSION,
        "kinematics": KINEMATICS_PROVENANCE,
        "sources": source_results,
        "episodes": len(rows),
        "accepted": len(accepted),
        "rejected": len(rows) - len(accepted),
        "acceptance_rate": len(accepted) / max(len(rows), 1),
        "selection_counts": dict(sorted(profiles.items())),
        "pipeline_errors": pipeline_errors,
        "zero_origin": {
            "applied_per_arm_to_both_sources": True,
            "maximum_fixed_pipeline_world_target_delta": max(invariance_values, default=0.0),
            "auto_smoother_selection_mismatches": smoother_selection_mismatches,
            "maximum_auto_smoothed_world_target_delta": max(auto_invariance_values, default=0.0),
            "filtering_decisions_invariant": (
                max(invariance_values, default=0.0) <= 1e-9
                and smoother_selection_mismatches == 0
                and max(auto_invariance_values, default=0.0) <= 1e-9
            ),
        },
        "gate_semantics_changed": False,
        "gates": {
            "all_frame_position_mm": 10.0,
            "all_frame_orientation_deg": 8.0,
            "gripper_event_position_mm": 1.0,
            "gripper_event_orientation_deg": 2.0,
            "interarm_surface_clearance_mm": 5.0,
            "self_surface_clearance_mm": 1.0,
        },
        "failure_signatures": [
            {"reasons": list(signature), "episodes": count}
            for signature, count in sorted(
                failure_signatures.items(), key=lambda value: (-value[1], value[0])
            )
        ],
        "workers": workers,
        "wall_runtime_s": wall_runtime_s,
        "per_episode": rows,
    }


def _quaternion_angle(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    left = Rotation.from_quat(np.asarray(first).reshape(-1, 4))
    right = Rotation.from_quat(np.asarray(second).reshape(-1, 4))
    delta = left.inv() * right
    if delta is NotImplemented:
        raise TypeError("could not compare rotations")
    return delta.magnitude().reshape(np.asarray(first).shape[:-1])
