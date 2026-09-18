from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from functools import partial
from pathlib import Path
from time import monotonic
from typing import Any

import numpy as np
from dataset_filtering import trajectory_optimization as trajectory_module
from dataset_filtering.hf_dataset import LeRobotSource, as_pose_episode
from dataset_filtering.kinematics import YAMKinematics
from dataset_filtering.models import ContactAnchor
from dataset_filtering.smoothing import smooth_pose_episode, with_smoothed_pose
from dataset_filtering.trajectory_optimization import (
    TrajectoryOptimizationConfig,
    optimize_pose_episode,
)

from dataset_filtering_tests.baseline import BASELINE_REPORT, load_baseline
from dataset_filtering_tests.home_screen import HOME_CANDIDATES, _home_initializer

BASE_CONFIG = TrajectoryOptimizationConfig()
CONFIG_PROFILES: dict[str, TrajectoryOptimizationConfig] = {
    # Diagnostic: sample the Cartesian objective at 15 Hz instead of 5 Hz while
    # retaining the same spline expressiveness and every acceptance threshold.
    "dense_objective": replace(
        BASE_CONFIG,
        objective_stride=2,
        collision_stride=12,
        solver_max_iterations=500,
    ),
    # Add spline expressiveness as well as denser Cartesian supervision.
    "flexible_dense": replace(
        BASE_CONFIG,
        control_interval_s=0.12,
        objective_stride=2,
        collision_stride=12,
        contact_multistart_seeds=12,
        solver_max_iterations=650,
    ),
    # Keep regularization, but give tracking more influence than flow/posture.
    # This changes optimization behavior, not acceptance criteria.
    "tracking_priority": replace(
        BASE_CONFIG,
        control_interval_s=0.15,
        objective_stride=2,
        collision_stride=12,
        position_weight=2.0,
        orientation_weight=0.7,
        flow_weight=0.4,
        joint_center_weight=0.00025,
        solver_max_iterations=600,
    ),
    # The 20 mm production repair budget is a local-solver safeguard, not an
    # acceptance gate. Test whether a denser warm start can repair a deeply
    # colliding branch; final 5 mm/1 mm clearance checks stay unchanged.
    "dense_deep_repair": replace(
        BASE_CONFIG,
        objective_stride=2,
        collision_stride=12,
        collision_repair_max_penetration_m=0.12,
        collision_repair_iterations=3,
        solver_max_iterations=700,
    ),
    # Same unchanged-gate dense solve, but preserve the all-frame greedy IK path
    # in the warm start rather than replacing it with event-to-event straight lines.
    "dense_greedy_initializer": replace(
        BASE_CONFIG,
        objective_stride=2,
        collision_stride=12,
        solver_max_iterations=600,
    ),
    # Repair deep validation-resolution collisions through staged clearance
    # constraints, but require the final candidate to reach the original margins.
    "dense_homotopy_repair": replace(
        BASE_CONFIG,
        objective_stride=2,
        collision_stride=12,
        collision_repair_max_penetration_m=0.12,
        collision_repair_iterations=6,
        solver_max_iterations=550,
    ),
    # Let the coupled solver choose Cartesian-equivalent contact branches and an
    # outward, collision-aware posture from the first solve instead of waiting for
    # a validation failure after fixed per-arm branches have been imposed.
    "dense_secondary_from_start": replace(
        BASE_CONFIG,
        objective_stride=2,
        collision_stride=6,
        collision_repair_max_penetration_m=0.12,
        collision_repair_iterations=3,
        solver_max_iterations=650,
    ),
    # Turn the validation envelope into a true all-frame NLP constraint rather
    # than relying on a sampled soft objective and checking the gaps afterward.
    "all_frame_tracking_constraints": replace(
        BASE_CONFIG,
        control_interval_s=0.12,
        objective_stride=6,
        collision_stride=12,
        solver_max_iterations=900,
    ),
    "dense_home_balanced_0p6": replace(
        BASE_CONFIG, objective_stride=2, collision_stride=12, solver_max_iterations=500
    ),
    "dense_home_balanced_1p2": replace(
        BASE_CONFIG, objective_stride=2, collision_stride=12, solver_max_iterations=500
    ),
    "dense_home_balanced_pi_over_2": replace(
        BASE_CONFIG, objective_stride=2, collision_stride=12, solver_max_iterations=500
    ),
    "dense_home_balanced_1p8": replace(
        BASE_CONFIG, objective_stride=2, collision_stride=12, solver_max_iterations=500
    ),
}

_GREEDY_INITIALIZER_PROFILES = {"dense_greedy_initializer"}
_HOMOTOPY_REPAIR_PROFILES = {"dense_homotopy_repair"}
_SECONDARY_FROM_START_PROFILES = {"dense_secondary_from_start"}
_ALL_FRAME_TRACKING_PROFILES = {"all_frame_tracking_constraints"}
_HOME_BY_PROFILE = {
    "dense_home_balanced_0p6": HOME_CANDIDATES["balanced_0p6"],
    "dense_home_balanced_1p2": HOME_CANDIDATES["balanced_1p2"],
    "dense_home_balanced_pi_over_2": HOME_CANDIDATES["balanced_pi_over_2"],
    "dense_home_balanced_1p8": HOME_CANDIDATES["balanced_1p8"],
}


class _TrackingAugmentedActiveSet:
    def __init__(self, active_set: Any, frames: int) -> None:
        self.interarm = active_set.interarm
        self.self_collision = active_set.self_collision
        self.minimum_interarm_surface_m = active_set.minimum_interarm_surface_m
        self.minimum_self_surface_m = active_set.minimum_self_surface_m
        self.frames = set(range(frames)) | set(active_set.frames)


def select_experiment_episodes(
    report_path: str | Path,
    *,
    profile: str,
    criterion: str,
) -> list[int]:
    report = json.loads(Path(report_path).expanduser().resolve().read_text(encoding="utf-8"))
    selected: list[int] = []
    for row in report["per_episode"]:
        value = row["profiles"][profile]
        validation = value["validation"]
        matches = {
            "not-replayable": not value["continuously_replayable"],
            "tracking-fail": not validation["passes_tracking_accuracy"],
            "tracking-pass-collision-fail": (
                validation["passes_tracking_accuracy"]
                and (
                    not validation["passes_interarm_clearance"]
                    or not validation["passes_self_collision_clearance"]
                )
            ),
        }
        if criterion not in matches:
            raise ValueError(f"unknown candidate selection criterion: {criterion}")
        if matches[criterion]:
            selected.append(int(row["episode_index"]))
    return selected


def run_trajectory_sweep(
    *,
    episode_indices: list[int],
    profiles: tuple[str, ...],
    source_name: str = "brandonyang/dual-lidar-umi-relative",
    source_revision: str = "main",
    baseline_report: str | Path = BASELINE_REPORT,
    workers: int = 4,
    smoothing_overrides: dict[int, tuple[str, dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    if not episode_indices:
        raise ValueError("at least one episode index is required")
    unknown = sorted(set(profiles) - CONFIG_PROFILES.keys())
    if unknown:
        raise ValueError(f"unknown trajectory profiles: {unknown}")

    baseline = load_baseline(baseline_report)
    baseline_rows = {int(row["episode_index"]): row for row in baseline["per_episode"]}
    missing = sorted(set(episode_indices) - baseline_rows.keys())
    if missing:
        raise ValueError(f"episodes are missing from the baseline report: {missing}")

    source = LeRobotSource(source_name, revision=source_revision)
    unique_indices = sorted(set(episode_indices))
    source.prefetch_episode_data(unique_indices)
    overrides = smoothing_overrides or {}
    arguments = [
        (
            source_name,
            source.location.revision,
            str(source.cache_dir),
            episode_index,
            overrides.get(
                episode_index,
                (
                    baseline_rows[episode_index]["smoothing"]["method"],
                    baseline_rows[episode_index]["smoothing"]["parameters"],
                ),
            )[0],
            overrides.get(
                episode_index,
                (
                    baseline_rows[episode_index]["smoothing"]["method"],
                    baseline_rows[episode_index]["smoothing"]["parameters"],
                ),
            )[1],
            profiles,
        )
        for episode_index in unique_indices
    ]

    rows: list[dict[str, Any]] = []
    started = monotonic()
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(arguments)))) as executor:
        futures = [executor.submit(_measure_episode_profiles, item) for item in arguments]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            print(f"trajectory sweep: {completed}/{len(futures)} episodes", flush=True)
    rows.sort(key=lambda row: row["episode_index"])

    aggregates: list[dict[str, Any]] = []
    for profile in profiles:
        values = [row["profiles"][profile] for row in rows]
        aggregates.append(
            {
                "profile": profile,
                "episodes": len(values),
                "solver_success": sum(value["solver_success"] for value in values),
                "ik_trajectory_solvable": sum(
                    value["ik_trajectory_solvable"] for value in values
                ),
                "continuously_replayable": sum(
                    value["continuously_replayable"] for value in values
                ),
                "passes_tracking_accuracy": sum(
                    value["validation"]["passes_tracking_accuracy"] for value in values
                ),
                "passes_contact_accuracy": sum(
                    value["validation"]["passes_contact_accuracy"] for value in values
                ),
                "passes_interarm_clearance": sum(
                    value["validation"]["passes_interarm_clearance"] for value in values
                ),
                "passes_self_collision_clearance": sum(
                    value["validation"]["passes_self_collision_clearance"] for value in values
                ),
                "median_runtime_s": float(np.median([value["runtime_s"] for value in values])),
            }
        )
    return {
        "experiment": "continuous_trajectory_configuration_sweep",
        "source": source_name,
        "baseline_report": str(Path(baseline_report).expanduser().resolve()),
        "episodes": unique_indices,
        "profiles": list(profiles),
        "smoothing_overrides": {
            str(index): {"method": value[0], "parameters": value[1]}
            for index, value in sorted(overrides.items())
            if index in unique_indices
        },
        "gate_semantics_changed": False,
        "invariants": {
            "all_frame_position_mm": 10.0,
            "all_frame_orientation_deg": 8.0,
            "gripper_event_position_mm": 1.0,
            "gripper_event_orientation_deg": 2.0,
            "interarm_surface_clearance_mm": 5.0,
            "self_surface_clearance_mm": 1.0,
        },
        "wall_runtime_s": monotonic() - started,
        "aggregates": aggregates,
        "per_episode": rows,
    }


def _measure_episode_profiles(
    arguments: tuple[str, str, str, int, str, dict[str, Any], tuple[str, ...]],
) -> dict[str, Any]:
    source_name, revision, cache_dir, episode_index, method, parameters, profiles = arguments
    source = LeRobotSource(source_name, revision=revision, cache_dir=cache_dir)
    original = as_pose_episode(source.load_episode(episode_index))
    smoothing = smooth_pose_episode(original, method, parameters)
    filtered = with_smoothed_pose(original, smoothing)

    measurements: dict[str, Any] = {}
    for profile in profiles:
        started = monotonic()
        production_kinematics_init = YAMKinematics.__init__
        production_initializer = trajectory_module._continuous_initial_guess
        production_spline_solver = trajectory_module._solve_spline_nlp
        try:
            if profile in _HOME_BY_PROFILE:
                YAMKinematics.__init__ = _home_initializer(
                    production_kinematics_init, _HOME_BY_PROFILE[profile]
                )
            if profile in _GREEDY_INITIALIZER_PROFILES:
                trajectory_module._continuous_initial_guess = _greedy_event_corrected_guess
            if profile in _HOMOTOPY_REPAIR_PROFILES:
                trajectory_module._solve_spline_nlp = partial(
                    _homotopy_spline_nlp,
                    production_solver=production_spline_solver,
                )
            elif profile in _SECONDARY_FROM_START_PROFILES:
                trajectory_module._solve_spline_nlp = partial(
                    _secondary_from_start_spline_nlp,
                    production_solver=production_spline_solver,
                )
            elif profile in _ALL_FRAME_TRACKING_PROFILES:
                trajectory_module._solve_spline_nlp = partial(
                    _all_frame_tracking_spline_nlp,
                    production_solver=production_spline_solver,
                )
            result = optimize_pose_episode(filtered, config=CONFIG_PROFILES[profile])
        finally:
            YAMKinematics.__init__ = production_kinematics_init
            trajectory_module._continuous_initial_guess = production_initializer
            trajectory_module._solve_spline_nlp = production_spline_solver
        measurements[profile] = _result_summary(result, monotonic() - started)
        measurements[profile]["registration_home_q_rad"] = (
            _HOME_BY_PROFILE[profile].tolist() if profile in _HOME_BY_PROFILE else None
        )
    return {
        "episode_index": episode_index,
        "smoothing": {"method": method, "parameters": parameters},
        "profiles": measurements,
    }


def _result_summary(result: Any, runtime_s: float) -> dict[str, Any]:
    metadata = result.metadata
    validation = metadata["validation"]
    return {
        "runtime_s": runtime_s,
        "solver_success": bool(metadata["solver"]["success"]),
        "ik_trajectory_solvable": bool(metadata["ik_trajectory_solvable"]),
        "continuously_replayable": bool(metadata["continuously_replayable"]),
        "both_frame_fraction": float(np.mean(np.all(result.frame_valid, axis=1))),
        "failed_bimanual_frames": int(np.count_nonzero(~np.all(result.frame_valid, axis=1))),
        "position_error_max_mm": float(1000 * np.max(result.position_error_m)),
        "position_error_p95_mm": float(metadata["tracking"]["position_error_p95_mm"]),
        "orientation_error_max_deg": float(
            np.degrees(np.max(result.orientation_error_rad))
        ),
        "orientation_error_p95_deg": float(
            metadata["tracking"]["orientation_error_p95_deg"]
        ),
        "flow_direction_cosine_median": float(
            metadata["tracking"]["flow_direction_cosine_median"]
        ),
        "validation": {
            key: bool(value) if key.startswith("passes_") else value
            for key, value in validation.items()
        },
        "tail_freeze": metadata["tail_freeze"],
        "secondary_objective": metadata["secondary_objective"],
        "config": metadata["config"],
    }


def _greedy_event_corrected_guess(
    greedy: Any,
    anchors: list[ContactAnchor],
    frames: int,
    kin: YAMKinematics,
) -> np.ndarray:
    """Retain framewise IK structure while smoothly matching strict event branches."""
    q = np.column_stack((greedy.arms[0].playback_q, greedy.arms[1].playback_q)).copy()
    timeline = np.arange(frames, dtype=np.float64)
    for arm in range(2):
        arm_slice = slice(arm * 6, (arm + 1) * 6)
        q[0, arm_slice] = kin.home
        arm_anchors = [anchor for anchor in anchors if anchor.arm == arm and anchor.feasible]
        key_frames = [0, *[anchor.frame for anchor in arm_anchors]]
        corrections = [kin.home - q[0, arm_slice]]
        corrections.extend(anchor.q - q[anchor.frame, arm_slice] for anchor in arm_anchors)
        if key_frames[-1] != frames - 1:
            key_frames.append(frames - 1)
            corrections.append(np.zeros(6, dtype=np.float64))
        frame_array = np.asarray(key_frames, dtype=np.int64)
        correction_array = np.asarray(corrections, dtype=np.float64)
        unique_frames, unique_indices = np.unique(frame_array, return_index=True)
        correction_array = correction_array[unique_indices]
        for joint in range(6):
            q[:, arm * 6 + joint] += np.interp(
                timeline,
                unique_frames,
                correction_array[:, joint],
            )
        q[:, arm_slice] = np.clip(q[:, arm_slice], kin.lower, kin.upper)
        q[0, arm_slice] = kin.home
        for anchor in arm_anchors:
            q[anchor.frame, arm_slice] = anchor.q
    return q


def _homotopy_spline_nlp(
    basis: Any,
    target_positions: np.ndarray,
    target_quaternions: np.ndarray,
    anchors: list[ContactAnchor],
    cfg: TrajectoryOptimizationConfig,
    kin: YAMKinematics,
    bases: tuple[np.ndarray, np.ndarray],
    initial_controls: np.ndarray,
    fps: float,
    collision_active_set: Any = None,
    *,
    allow_contact_branch_adjustment: bool = False,
    enable_secondary_objective: bool = False,
    production_solver: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    staged_cfg = cfg
    if enable_secondary_objective and collision_active_set is not None:
        interarm_margin = cfg.interarm_collision_margin_m
        self_margin = cfg.self_collision_margin_m
        if collision_active_set.interarm:
            interarm_margin = min(
                interarm_margin,
                collision_active_set.minimum_interarm_surface_m + 0.015,
            )
        if collision_active_set.self_collision:
            self_margin = min(
                self_margin,
                collision_active_set.minimum_self_surface_m + 0.008,
            )
        staged_cfg = replace(
            cfg,
            interarm_collision_margin_m=interarm_margin,
            self_collision_margin_m=self_margin,
        )
    controls, metadata = production_solver(
        basis,
        target_positions,
        target_quaternions,
        anchors,
        staged_cfg,
        kin,
        bases,
        initial_controls,
        fps,
        collision_active_set,
        allow_contact_branch_adjustment=allow_contact_branch_adjustment,
        enable_secondary_objective=enable_secondary_objective,
    )
    metadata = dict(metadata)
    metadata["homotopy_constraint_margins_mm"] = {
        "interarm": 1000 * staged_cfg.interarm_collision_margin_m,
        "self": 1000 * staged_cfg.self_collision_margin_m,
        "final_required_interarm": 1000 * cfg.interarm_collision_margin_m,
        "final_required_self": 1000 * cfg.self_collision_margin_m,
    }
    return controls, metadata


def _secondary_from_start_spline_nlp(
    *args: Any,
    production_solver: Any,
    **kwargs: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    kwargs["allow_contact_branch_adjustment"] = True
    kwargs["enable_secondary_objective"] = True
    controls, metadata = production_solver(*args, **kwargs)
    metadata = dict(metadata)
    metadata["secondary_objective_from_initial_solve"] = True
    return controls, metadata


def _all_frame_tracking_spline_nlp(
    basis: Any,
    target_positions: np.ndarray,
    target_quaternions: np.ndarray,
    anchors: list[ContactAnchor],
    cfg: TrajectoryOptimizationConfig,
    kin: YAMKinematics,
    bases: tuple[np.ndarray, np.ndarray],
    initial_controls: np.ndarray,
    fps: float,
    collision_active_set: Any = None,
    *,
    allow_contact_branch_adjustment: bool = False,
    enable_secondary_objective: bool = False,
    production_solver: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    active_set = collision_active_set or trajectory_module._CollisionActiveSet()
    augmented = _TrackingAugmentedActiveSet(active_set, len(target_positions))
    controls, metadata = production_solver(
        basis,
        target_positions,
        target_quaternions,
        anchors,
        cfg,
        kin,
        bases,
        initial_controls,
        fps,
        augmented,
        allow_contact_branch_adjustment=allow_contact_branch_adjustment,
        enable_secondary_objective=enable_secondary_objective,
    )
    metadata = dict(metadata)
    metadata["hard_tracking_constraint_frames"] = len(augmented.frames)
    return controls, metadata
