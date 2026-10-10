from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from typing import Any

import casadi as ca
import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import csc_matrix
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from dataset_filtering.kinematics import (
    GRASP_SITE_ROTATION,
    GRASP_SITE_TRANSLATION,
    YAM_JOINTS,
    YAM_VELOCITY_LIMITS_RAD_S,
    YAMKinematics,
    transform_points,
    transform_pose_to_local,
)
from dataset_filtering.models import ContactAnchor, ContinuousRetargetResult, PoseEpisode
from dataset_filtering.retarget import (
    DEFAULT_PAIR_SEPARATION_M,
    map_umi_targets_to_yam_world,
    retarget_pose_episode,
    yam_base_transforms,
)
from dataset_filtering.retiming import time_parameterize, trajectory_dynamics
from dataset_filtering.smoothing import detect_gripper_events

# Conservative capsule proxy transcribed from the radii of the public YAM
# MuJoCo collision geoms. The kinematic point chain does not expose each geom's
# full local transform, so three interior samples per link are used in the NLP
# and validation. This catches segment crossings that joint-origin checks miss;
# final hardware deployment should still run the exact scene collision checker.
_COLLISION_SEGMENTS = (2, 3, 4, 5, 6)
_COLLISION_RADII_M = (0.032, 0.034, 0.030, 0.030, 0.040)
_COLLISION_SAMPLE_FRACTIONS = (0.15, 0.5, 0.85)
_SELF_COLLISION_SEGMENT_PAIRS = (
    (2, 4),
    (2, 5),
    (2, 6),
    (3, 5),
    (3, 6),
    (4, 6),
)


@dataclass(frozen=True)
class TrajectoryOptimizationConfig:
    control_interval_s: float = 0.2
    objective_stride: int = 6
    collision_stride: int = 24
    contact_position_tolerance_m: float = 0.001
    contact_orientation_tolerance_rad: float = np.deg2rad(2.0)
    tracking_position_tolerance_m: float = 0.01
    tracking_orientation_tolerance_rad: float = np.deg2rad(8.0)
    enable_tail_freeze: bool = True
    tail_freeze_grace_s: float = 0.5
    contact_multistart_seeds: int = 8
    contact_max_iterations: int = 120
    approach_window_s: float = 0.2
    planning_velocity_scale: float = 1.0
    planning_acceleration_limit_rad_s2: float = 20.0
    interarm_collision_margin_m: float = 0.005
    self_collision_margin_m: float = 0.001
    table_height_m: float | None = None
    table_clearance_m: float = 0.015
    position_weight: float = 1.0
    orientation_weight: float = 0.35
    flow_weight: float = 0.6
    bimanual_position_weight: float = 0.4
    bimanual_orientation_weight: float = 0.15
    camera_weight: float = 0.25
    joint_velocity_weight: float = 0.008
    joint_acceleration_weight: float = 0.002
    joint_jerk_weight: float = 0.0002
    joint_center_weight: float = 0.0005
    joint_reference_weight: float = 0.002
    joint_limit_margin_weight: float = 0.0002
    collision_avoidance_weight: float = 0.1
    collision_influence_m: float = 0.025
    collision_constraint_buffer_m: float = 1e-5
    collision_repair_max_penetration_m: float = 0.02
    collision_repair_iterations: int = 2
    solver_max_iterations: int = 350
    source_tool_to_camera_translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    source_tool_to_camera_rpy_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yam_tool_to_camera_translation_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yam_tool_to_camera_rpy_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_extrinsics_calibrated: bool = False
    prefer_toppra: bool = True
    aligned_shared_frame: bool = False


@dataclass(frozen=True)
class _CollisionActiveSet:
    interarm: tuple[tuple[int, int, int], ...] = ()
    self_collision: tuple[tuple[int, int, int, int], ...] = ()
    minimum_interarm_surface_m: float = float("inf")
    minimum_self_surface_m: float = float("inf")

    @property
    def frames(self) -> set[int]:
        return {item[0] for item in self.interarm} | {
            item[0] for item in self.self_collision
        }


def optimize_pose_episode(
    episode: PoseEpisode,
    *,
    config: TrajectoryOptimizationConfig | None = None,
    separation_m: float = DEFAULT_PAIR_SEPARATION_M,
) -> ContinuousRetargetResult:
    """Create a continuous trajectory with conditional per-arm put-down tail freezing."""
    cfg = config or TrajectoryOptimizationConfig()
    provisional = _optimize_pose_episode_once(
        episode,
        config=cfg,
        separation_m=separation_m,
    )
    processed, tail_metadata = apply_failed_tail_freeze(
        episode,
        provisional.event_indices,
        provisional.frame_valid,
        grace_period_s=cfg.tail_freeze_grace_s,
        position_tolerance_m=cfg.tracking_position_tolerance_m,
        orientation_tolerance_rad=cfg.tracking_orientation_tolerance_rad,
        enabled=cfg.enable_tail_freeze,
    )
    if tail_metadata["applied_any"]:
        final = _optimize_pose_episode_once(
            processed,
            config=cfg,
            separation_m=separation_m,
        )
    else:
        final = provisional

    tail_metadata["provisional_solver_success"] = bool(
        provisional.metadata["solver"]["success"]
    )
    tail_metadata["provisional_ik_trajectory_solvable"] = bool(
        provisional.metadata["ik_trajectory_solvable"]
    )
    tail_metadata["provisional_continuously_replayable"] = bool(
        provisional.metadata["continuously_replayable"]
    )
    tail_metadata["provisional_failed_arm_frames"] = int(
        np.count_nonzero(~provisional.frame_valid)
    )
    tail_metadata["provisional_failed_bimanual_frames"] = int(
        np.count_nonzero(~np.all(provisional.frame_valid, axis=1))
    )
    tail_metadata["final_failed_arm_frames"] = int(np.count_nonzero(~final.frame_valid))
    tail_metadata["final_failed_bimanual_frames"] = int(
        np.count_nonzero(~np.all(final.frame_valid, axis=1))
    )
    metadata = dict(final.metadata)
    metadata["base_algorithm"] = metadata["algorithm"]
    metadata["algorithm"] = (
        "contact_anchored_bimanual_cubic_bspline_nlp_with_tail_freeze_v3"
    )
    metadata["tail_freeze"] = tail_metadata
    return replace(final, metadata=metadata)


def apply_failed_tail_freeze(
    episode: PoseEpisode,
    event_indices: tuple[np.ndarray, np.ndarray],
    frame_valid: np.ndarray,
    *,
    grace_period_s: float = 0.5,
    position_tolerance_m: float = 0.01,
    orientation_tolerance_rad: float = np.deg2rad(8.0),
    enabled: bool = True,
) -> tuple[PoseEpisode, dict[str, Any]]:
    """Freeze each arm's target after its last event only when that tail contains a failure."""
    if grace_period_s < 0:
        raise ValueError("grace_period_s must be non-negative")
    if frame_valid.shape != (len(episode.timestamps), 2):
        raise ValueError("frame_valid must have shape (N, 2)")
    if len(event_indices) != 2:
        raise ValueError("event_indices must contain one array per arm")

    positions = episode.positions.copy()
    quaternions = episode.quaternions.copy()
    grace_frames = int(round(grace_period_s * episode.fps))
    arm_names = ("umi1_left", "umi2_right")
    arms: dict[str, Any] = {}
    applied_any = False
    final_frame = len(episode.timestamps) - 1
    for arm, name in enumerate(arm_names):
        events = np.asarray(event_indices[arm], dtype=np.int64)
        last_event = int(events[-1]) if len(events) else None
        cutoff = None if last_event is None else last_event + grace_frames
        eligible = cutoff is not None and cutoff < final_frame
        failed_tail_frames = (
            np.flatnonzero(~frame_valid[cutoff:, arm]).astype(np.int64) + cutoff
            if eligible and cutoff is not None
            else np.zeros(0, dtype=np.int64)
        )
        applied = bool(enabled and eligible and len(failed_tail_frames))
        if applied and cutoff is not None:
            positions[cutoff:, arm] = positions[cutoff, arm]
            quaternions[cutoff:, arm] = quaternions[cutoff, arm]
            applied_any = True
        arms[name] = {
            "last_event_frame": last_event,
            "cutoff_frame": cutoff if eligible else None,
            "eligible_tail": bool(eligible),
            "tail_frames_inspected": (len(episode.timestamps) - cutoff) if eligible else 0,
            "provisional_failed_tail_frames": failed_tail_frames.tolist(),
            "provisional_failed_tail_count": int(len(failed_tail_frames)),
            "applied": applied,
        }

    processed = replace(episode, positions=positions, quaternions=quaternions)
    return processed, {
        "enabled": bool(enabled),
        "grace_period_s": float(grace_period_s),
        "grace_frames": grace_frames,
        "trigger": "provisional_continuous_tracking_failure",
        "tracking_position_tolerance_mm": 1000 * position_tolerance_m,
        "tracking_orientation_tolerance_deg": float(np.degrees(orientation_tolerance_rad)),
        "frozen_value": "smoothed_6d_tool_pose_at_cutoff_frame",
        "applied_any": applied_any,
        "arms": arms,
    }


def _optimize_pose_episode_once(
    episode: PoseEpisode,
    *,
    config: TrajectoryOptimizationConfig,
    separation_m: float,
) -> ContinuousRetargetResult:
    """Run one continuous, contact-anchored solve without changing its target episode."""
    cfg = config
    kin = YAMKinematics()
    bases = yam_base_transforms(separation_m)
    target_positions, target_quaternions = map_umi_targets_to_yam_world(
        episode,
        separation_m=separation_m,
        kinematics=kin,
        aligned_shared_frame=cfg.aligned_shared_frame,
    )
    events = detect_gripper_events(episode.grippers, fps=episode.fps)
    greedy = retarget_pose_episode(
        episode,
        separation_m=separation_m,
        max_iterations=65,
        aligned_shared_frame=cfg.aligned_shared_frame,
    )
    anchors = _solve_contact_anchors(
        episode,
        target_positions,
        target_quaternions,
        events,
        greedy,
        cfg,
        kin,
        bases,
    )

    relative_time = episode.timestamps - episode.timestamps[0]
    basis = _bspline_basis(relative_time, cfg.control_interval_s)
    q_initial = _continuous_initial_guess(greedy, anchors, len(relative_time), kin)
    controls_initial = _fit_spline_controls(basis, q_initial)
    collision_active_set = _CollisionActiveSet()
    controls, solver_metadata = _solve_spline_nlp(
        basis,
        target_positions,
        target_quaternions,
        anchors,
        cfg,
        kin,
        bases,
        controls_initial,
        q_initial[0],
        episode.fps,
        collision_active_set,
        allow_contact_branch_adjustment=False,
        enable_secondary_objective=False,
    )
    q_flat = basis.values @ controls
    q = np.stack((q_flat[:, :6], q_flat[:, 6:]), axis=1)

    (
        links,
        achieved_positions,
        achieved_quaternions,
    ) = _evaluate_trajectory(q, kin, bases)
    repair_attempts: list[dict[str, Any]] = []
    active_set = _collect_active_collision_constraints(links, cfg)
    primary_feasible = _primary_tracking_feasible(
        achieved_positions,
        achieved_quaternions,
        target_positions,
        target_quaternions,
        events,
        cfg,
    )
    for iteration in range(cfg.collision_repair_iterations):
        penetration = max(
            cfg.interarm_collision_margin_m - active_set.minimum_interarm_surface_m,
            cfg.self_collision_margin_m - active_set.minimum_self_surface_m,
            0.0,
        )
        if not active_set.frames or not primary_feasible:
            break
        if penetration > cfg.collision_repair_max_penetration_m:
            repair_attempts.append(
                {
                    "iteration": iteration + 1,
                    "attempted": False,
                    "reason": "penetration_exceeds_repair_budget",
                    "penetration_mm": 1000 * penetration,
                }
            )
            break
        repaired_controls, repaired_solver = _solve_spline_nlp(
            basis,
            target_positions,
            target_quaternions,
            anchors,
            cfg,
            kin,
            bases,
            controls,
            q_initial[0],
            episode.fps,
            active_set,
            allow_contact_branch_adjustment=True,
            enable_secondary_objective=True,
        )
        repaired_q_flat = basis.values @ repaired_controls
        repaired_q = np.stack((repaired_q_flat[:, :6], repaired_q_flat[:, 6:]), axis=1)
        repaired_links, repaired_positions, repaired_quaternions = _evaluate_trajectory(
            repaired_q, kin, bases
        )
        repaired_active_set = _collect_active_collision_constraints(repaired_links, cfg)
        repaired_primary_feasible = _primary_tracking_feasible(
            repaired_positions,
            repaired_quaternions,
            target_positions,
            target_quaternions,
            events,
            cfg,
        )
        old_margin = min(
            active_set.minimum_interarm_surface_m - cfg.interarm_collision_margin_m,
            active_set.minimum_self_surface_m - cfg.self_collision_margin_m,
        )
        new_margin = min(
            repaired_active_set.minimum_interarm_surface_m - cfg.interarm_collision_margin_m,
            repaired_active_set.minimum_self_surface_m - cfg.self_collision_margin_m,
        )
        accepted = bool(
            repaired_solver["success"]
            and repaired_primary_feasible
            and new_margin > old_margin + 1e-6
        )
        repair_attempts.append(
            {
                "iteration": iteration + 1,
                "attempted": True,
                "accepted": accepted,
                "solver_success": bool(repaired_solver["success"]),
                "old_minimum_margin_mm": 1000 * old_margin,
                "new_minimum_margin_mm": 1000 * new_margin,
                "active_interarm_constraints": len(active_set.interarm),
                "active_self_collision_constraints": len(active_set.self_collision),
            }
        )
        if not accepted:
            break
        controls = repaired_controls
        solver_metadata = repaired_solver
        q_flat = repaired_q_flat
        q = repaired_q
        links = repaired_links
        achieved_positions = repaired_positions
        achieved_quaternions = repaired_quaternions
        active_set = repaired_active_set
        primary_feasible = repaired_primary_feasible
    position_error = np.linalg.norm(achieved_positions - target_positions, axis=-1)
    orientation_error = _orientation_error(achieved_quaternions, target_quaternions)

    velocity_limits = np.tile(YAM_VELOCITY_LIMITS_RAD_S * cfg.planning_velocity_scale, 2)
    acceleration_limits = np.full(12, cfg.planning_acceleration_limit_rad_s2)
    timing = time_parameterize(
        q_flat,
        episode.timestamps,
        velocity_limits,
        acceleration_limits,
        prefer_toppra=cfg.prefer_toppra,
    )
    dynamics = trajectory_dynamics(q_flat, timing.timestamps)
    validation = _validate_trajectory(
        q,
        links,
        achieved_positions,
        achieved_quaternions,
        target_positions,
        target_quaternions,
        anchors,
        events,
        timing.timestamps,
        velocity_limits,
        acceleration_limits,
        cfg,
        kin,
    )
    camera, camera_position_error, camera_orientation_error, video_weight = _camera_alignment(
        achieved_positions,
        achieved_quaternions,
        target_positions,
        target_quaternions,
        episode.timestamps,
        timing.timestamps,
        events,
        cfg,
    )
    frame_valid = (position_error <= cfg.tracking_position_tolerance_m) & (
        orientation_error <= cfg.tracking_orientation_tolerance_rad
    )
    metadata: dict[str, Any] = {
        "algorithm": "contact_anchored_bimanual_cubic_bspline_nlp_with_active_set_repair",
        "greedy_ik_role": "initializer_and_framewise_baseline_only",
        "solver": solver_metadata,
        "secondary_objective": {
            "contact_constraints": (
                "fixed_initializer_branch_on_baseline; cartesian_1mm_2deg_during_collision_repair"
            ),
            "joint_posture": "normalized_initial_branch_continuity_plus_joint_limit_barrier",
            "collision_avoidance": "coarse_global_constraints_plus_validation_sample_active_set",
            "active_interarm_constraints": len(collision_active_set.interarm),
            "active_self_collision_constraints": len(collision_active_set.self_collision),
            "collision_repair": repair_attempts,
        },
        "spline": {
            "degree": basis.degree,
            "control_points": basis.values.shape[1],
            "control_interval_s": cfg.control_interval_s,
        },
        "contact_events": {
            "total": len(anchors),
            "strictly_feasible": sum(anchor.feasible for anchor in anchors),
            "all_strictly_feasible": all(anchor.feasible for anchor in anchors),
            "umi1_left_frames": events[0].tolist(),
            "umi2_right_frames": events[1].tolist(),
            "position_tolerance_mm": 1000 * cfg.contact_position_tolerance_m,
            "orientation_tolerance_deg": float(np.degrees(cfg.contact_orientation_tolerance_rad)),
            "anchors": [
                {
                    "arm": "umi1_left" if anchor.arm == 0 else "umi2_right",
                    "frame": int(anchor.frame),
                    "feasible": bool(anchor.feasible),
                    "position_error_mm": float(1000 * anchor.position_error_m),
                    "orientation_error_deg": float(np.degrees(anchor.orientation_error_rad)),
                    "candidate_branches": int(anchor.candidates),
                }
                for anchor in anchors
            ],
        },
        "timing": {"method": timing.method, **timing.metadata},
        "dynamics": {
            "max_abs_velocity_rad_s": float(np.max(np.abs(dynamics["velocity_rad_s"]))),
            "max_abs_acceleration_rad_s2": float(np.max(np.abs(dynamics["acceleration_rad_s2"]))),
            "max_abs_jerk_rad_s3": float(np.max(np.abs(dynamics["jerk_rad_s3"]))),
            "velocity_limits_from_yam_urdf": True,
            "acceleration_limit_is_planning_assumption": True,
        },
        "tracking": {
            "individual_target_fraction_10mm_8deg": float(np.mean(frame_valid)),
            "both_target_fraction_10mm_8deg": float(np.mean(np.all(frame_valid, axis=1))),
            "all_frames_within_tolerance": bool(np.all(frame_valid)),
            "position_tolerance_mm": 1000 * cfg.tracking_position_tolerance_m,
            "orientation_tolerance_deg": float(
                np.degrees(cfg.tracking_orientation_tolerance_rad)
            ),
            "position_error_median_mm": float(1000 * np.median(position_error)),
            "position_error_p95_mm": float(1000 * np.percentile(position_error, 95)),
            "orientation_error_median_deg": float(np.degrees(np.median(orientation_error))),
            "orientation_error_p95_deg": float(np.degrees(np.percentile(orientation_error, 95))),
            "flow_direction_cosine_median": _flow_direction_cosine(
                achieved_positions, target_positions
            ),
        },
        "camera_alignment": camera,
        "validation": validation,
        "config": _serializable_config(cfg),
    }
    metadata["ik_trajectory_solvable"] = bool(
        solver_metadata["success"]
        and metadata["contact_events"]["all_strictly_feasible"]
        and validation["passes_joint_limits"]
        and validation["passes_tracking_accuracy"]
        and validation["passes_contact_accuracy"]
    )
    metadata["continuously_replayable"] = bool(
        metadata["ik_trajectory_solvable"]
        and validation["passes_velocity_limits"]
        and validation["passes_acceleration_limits"]
        and validation["passes_interarm_clearance"]
        and validation["passes_self_collision_clearance"]
        and validation["passes_table_clearance"]
    )
    return ContinuousRetargetResult(
        source_timestamps=episode.timestamps.copy(),
        robot_timestamps=timing.timestamps,
        q=q,
        link_points_world=links,
        achieved_positions_world=achieved_positions,
        achieved_quaternions_world=achieved_quaternions,
        target_positions_world=target_positions,
        target_quaternions_world=target_quaternions,
        position_error_m=position_error,
        orientation_error_rad=orientation_error,
        camera_position_error_m=camera_position_error,
        camera_orientation_error_rad=camera_orientation_error,
        video_alignment_weight=video_weight,
        event_indices=events,
        contact_anchors=tuple(anchors),
        frame_valid=frame_valid,
        metadata=metadata,
    )


@dataclass(frozen=True)
class _SplineBasis:
    degree: int
    knots: np.ndarray
    values: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray


def _bspline_basis(times: np.ndarray, control_interval_s: float) -> _SplineBasis:
    times = np.asarray(times, dtype=np.float64)
    duration = float(times[-1])
    degree = 3
    controls = max(8, int(np.ceil(duration / control_interval_s)) + degree)
    interior_count = controls - degree - 1
    interior = (
        np.linspace(0.0, duration, interior_count + 2)[1:-1] if interior_count else np.zeros(0)
    )
    knots = np.concatenate((np.repeat(0.0, degree + 1), interior, np.repeat(duration, degree + 1)))
    coefficient_identity = np.eye(controls)
    spline = BSpline(knots, coefficient_identity, degree, extrapolate=False)
    return _SplineBasis(
        degree=degree,
        knots=knots,
        values=np.asarray(spline(times), dtype=np.float64),
        velocity=np.asarray(spline.derivative(1)(times), dtype=np.float64),
        acceleration=np.asarray(spline.derivative(2)(times), dtype=np.float64),
        jerk=np.asarray(spline.derivative(3)(times), dtype=np.float64),
    )


def _fit_spline_controls(basis: _SplineBasis, q: np.ndarray) -> np.ndarray:
    regularization = 2e-4
    matrix = np.vstack((basis.values, np.sqrt(regularization) * basis.acceleration))
    target = np.vstack((q, np.zeros((len(q), q.shape[1]))))
    controls = np.linalg.lstsq(matrix, target, rcond=1e-8)[0]
    lower = np.tile(YAMKinematics().lower, 2)
    upper = np.tile(YAMKinematics().upper, 2)
    return np.clip(controls, lower, upper)


def _continuous_initial_guess(
    greedy: Any, anchors: list[ContactAnchor], frames: int, kin: YAMKinematics
) -> np.ndarray:
    q = np.column_stack((greedy.arms[0].playback_q, greedy.arms[1].playback_q))
    timeline = np.arange(frames)
    for arm in range(2):
        arm_anchors = [anchor for anchor in anchors if anchor.arm == arm and anchor.feasible]
        key_frames = [0, *[anchor.frame for anchor in arm_anchors]]
        key_q = [
            greedy.arms[arm].q_solutions[0],
            *[anchor.q for anchor in arm_anchors],
        ]
        if key_frames[-1] != frames - 1:
            key_frames.append(frames - 1)
            key_q.append(q[-1, arm * 6 : (arm + 1) * 6])
        order = np.argsort(key_frames)
        key_frames_array = np.asarray(key_frames, dtype=int)[order]
        key_q_array = np.asarray(key_q, dtype=np.float64)[order]
        unique_frames, unique_indices = np.unique(key_frames_array, return_index=True)
        key_q_array = key_q_array[unique_indices]
        for joint in range(6):
            q[:, arm * 6 + joint] = np.interp(timeline, unique_frames, key_q_array[:, joint])
    return q


def _solve_contact_anchors(
    episode: PoseEpisode,
    target_positions: np.ndarray,
    target_quaternions: np.ndarray,
    events: tuple[np.ndarray, np.ndarray],
    greedy: Any,
    cfg: TrajectoryOptimizationConfig,
    kin: YAMKinematics,
    bases: tuple[np.ndarray, np.ndarray],
) -> list[ContactAnchor]:
    all_anchors: list[ContactAnchor] = []
    for arm in range(2):
        candidate_sets: list[tuple[int, list[Any]]] = []
        sampler = qmc.Halton(d=6, scramble=True, seed=episode.episode_index * 17 + arm)
        random_unit = sampler.random(cfg.contact_multistart_seeds)
        random_seeds = kin.lower + random_unit * (kin.upper - kin.lower)
        for frame_value in events[arm]:
            frame = int(frame_value)
            position, quaternion = transform_pose_to_local(
                target_positions[frame, arm], target_quaternions[frame, arm], bases[arm]
            )
            explicit = np.vstack(
                (
                    greedy.arms[arm].q_solutions[frame],
                    kin.home,
                    0.5 * (kin.lower + kin.upper),
                    random_seeds,
                )
            )
            candidates = kin.solve_ik_multistart(
                position,
                quaternion,
                explicit,
                position_tolerance_m=cfg.contact_position_tolerance_m,
                orientation_tolerance_rad=cfg.contact_orientation_tolerance_rad,
                max_iterations=cfg.contact_max_iterations,
            )
            candidate_sets.append((frame, candidates))
        selected = _select_contact_branches(
            candidate_sets, kin, greedy.arms[arm].q_solutions[0]
        )
        for (frame, candidates), chosen in zip(candidate_sets, selected, strict=True):
            feasible = bool(chosen.success)
            all_anchors.append(
                ContactAnchor(
                    arm=arm,
                    frame=frame,
                    feasible=feasible,
                    q=chosen.q,
                    position_error_m=chosen.position_error_m,
                    orientation_error_rad=chosen.orientation_error_rad,
                    candidates=sum(candidate.success for candidate in candidates),
                )
            )
    return sorted(all_anchors, key=lambda anchor: (anchor.frame, anchor.arm))


def _select_contact_branches(
    candidate_sets: list[tuple[int, list[Any]]],
    kin: YAMKinematics,
    initial_q: np.ndarray,
) -> list[Any]:
    if not candidate_sets:
        return []
    selected: list[Any | None] = [None] * len(candidate_sets)
    feasible_indices = [
        index for index, (_, candidates) in enumerate(candidate_sets) if candidates[0].success
    ]
    for index, (_, candidates) in enumerate(candidate_sets):
        if not candidates[0].success:
            selected[index] = candidates[0]
    if not feasible_indices:
        return [item for item in selected if item is not None]

    costs: list[np.ndarray] = []
    parents: list[np.ndarray] = []
    joint_range = np.maximum(kin.upper - kin.lower, 1e-6)
    first_candidates = candidate_sets[feasible_indices[0]][1]
    costs.append(
        np.asarray(
            [
                np.sum(((candidate.q - initial_q) / joint_range) ** 2)
                for candidate in first_candidates
            ]
        )
    )
    parents.append(np.full(len(first_candidates), -1, dtype=int))
    for previous_slot, slot in zip(feasible_indices, feasible_indices[1:], strict=False):
        previous_candidates = candidate_sets[previous_slot][1]
        current_candidates = candidate_sets[slot][1]
        transition = np.empty((len(previous_candidates), len(current_candidates)))
        for previous_index, previous in enumerate(previous_candidates):
            for current_index, current in enumerate(current_candidates):
                transition[previous_index, current_index] = np.sum(
                    ((current.q - previous.q) / joint_range) ** 2
                )
        total = costs[-1][:, None] + transition
        parents.append(np.argmin(total, axis=0))
        costs.append(np.min(total, axis=0))
    current_index = int(np.argmin(costs[-1]))
    for path_index in range(len(feasible_indices) - 1, -1, -1):
        slot = feasible_indices[path_index]
        selected[slot] = candidate_sets[slot][1][current_index]
        current_index = int(parents[path_index][current_index])
    return [item for item in selected if item is not None]


def _solve_spline_nlp(
    basis: _SplineBasis,
    target_positions: np.ndarray,
    target_quaternions: np.ndarray,
    anchors: list[ContactAnchor],
    cfg: TrajectoryOptimizationConfig,
    kin: YAMKinematics,
    bases: tuple[np.ndarray, np.ndarray],
    initial_controls: np.ndarray,
    initial_q: np.ndarray,
    fps: float,
    collision_active_set: _CollisionActiveSet | None = None,
    *,
    allow_contact_branch_adjustment: bool = False,
    enable_secondary_objective: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    frames, controls_count = basis.values.shape
    active_set = collision_active_set or _CollisionActiveSet()
    opti = ca.Opti()
    controls = opti.variable(controls_count, 12)
    lower = np.tile(kin.lower, 2)
    upper = np.tile(kin.upper, 2)
    opti.subject_to(
        opti.bounded(
            ca.DM(np.tile(lower, (controls_count, 1))),
            controls,
            ca.DM(np.tile(upper, (controls_count, 1))),
        )
    )
    q_all = ca.DM(csc_matrix(basis.values)) @ controls
    opti.subject_to(q_all[0, :].T == ca.DM(initial_q))
    fk_symbolic = _symbolic_fk()
    target_rotations = (
        Rotation.from_quat(target_quaternions.reshape(-1, 4))
        .as_matrix()
        .reshape(frames, 2, 3, 3)
    )
    for anchor in anchors:
        if anchor.feasible:
            start = anchor.arm * 6
            q_event = ca.DM(csc_matrix(basis.values[anchor.frame : anchor.frame + 1])) @ controls
            if not allow_contact_branch_adjustment:
                opti.subject_to(q_event[0, start : start + 6].T == ca.DM(anchor.q))
                continue
            event_position, event_rotation, _ = fk_symbolic(
                q_event[0, start : start + 6].T
            )
            event_position = event_position + ca.DM(bases[anchor.arm][:3, 3])
            position_error = event_position - ca.DM(
                target_positions[anchor.frame, anchor.arm]
            )
            opti.subject_to(
                ca.sumsqr(position_error) <= (0.995 * cfg.contact_position_tolerance_m) ** 2
            )
            relative_rotation = (
                ca.DM(target_rotations[anchor.frame, anchor.arm]).T @ event_rotation
            )
            # trace(R) = 1 + 2 cos(theta), so this is an exact geodesic-angle
            # tolerance without committing the NLP to a preselected joint branch.
            opti.subject_to(
                ca.trace(relative_rotation)
                >= 1.0 + 2.0 * np.cos(0.995 * cfg.contact_orientation_tolerance_rad)
            )

    sample_indices = sorted(
        set(range(0, frames, max(1, cfg.objective_stride)))
        | {frames - 1}
        | {anchor.frame for anchor in anchors}
        | active_set.frames
    )
    event_frames = np.asarray([anchor.frame for anchor in anchors], dtype=int)
    source_camera_translation = ca.DM(cfg.source_tool_to_camera_translation_m)
    source_camera_rotation = ca.DM(
        Rotation.from_euler("xyz", cfg.source_tool_to_camera_rpy_rad).as_matrix()
    )
    yam_camera_translation = ca.DM(cfg.yam_tool_to_camera_translation_m)
    yam_camera_rotation = ca.DM(
        Rotation.from_euler("xyz", cfg.yam_tool_to_camera_rpy_rad).as_matrix()
    )
    achieved: dict[tuple[int, int], tuple[Any, Any, Any, Any, Any]] = {}
    objective: Any = 0
    approach_radius = max(1.0, cfg.approach_window_s * fps)
    for frame in sample_indices:
        if len(event_frames):
            distance = float(np.min(np.abs(event_frames - frame)))
            event_weight = 1.0 + 9.0 * np.exp(-((distance / approach_radius) ** 2))
        else:
            event_weight = 1.0
        for arm in range(2):
            start = arm * 6
            position, rotation, points = fk_symbolic(q_all[frame, start : start + 6].T)
            base_translation = ca.DM(bases[arm][:3, 3])
            world_position = position + base_translation
            world_points = points + ca.repmat(base_translation, 1, 8)
            camera_position = world_position + rotation @ yam_camera_translation
            camera_world_rotation = rotation @ yam_camera_rotation
            achieved[(frame, arm)] = (
                world_position,
                rotation,
                world_points,
                camera_position,
                camera_world_rotation,
            )
            target_position = ca.DM(target_positions[frame, arm])
            target_rotation = ca.DM(target_rotations[frame, arm])
            position_residual = world_position - target_position
            objective += (
                cfg.position_weight
                * event_weight
                * _pseudo_huber(_smooth_norm(position_residual) / 0.01)
            )
            objective += (
                cfg.orientation_weight
                * event_weight
                * _rotation_loss(rotation, target_rotation)
                / np.deg2rad(10.0) ** 2
            )
            target_camera_position = target_position + target_rotation @ source_camera_translation
            target_camera_rotation = target_rotation @ source_camera_rotation
            objective += cfg.camera_weight * _pseudo_huber(
                _smooth_norm(camera_position - target_camera_position) / 0.01
            )
            objective += (
                0.25
                * cfg.camera_weight
                * _rotation_loss(camera_world_rotation, target_camera_rotation)
                / np.deg2rad(10.0) ** 2
            )

        left_position, left_rotation, *_ = achieved[(frame, 0)]
        right_position, right_rotation, *_ = achieved[(frame, 1)]
        target_relative_position = ca.DM(target_positions[frame, 1] - target_positions[frame, 0])
        objective += cfg.bimanual_position_weight * ca.sumsqr(
            (right_position - left_position - target_relative_position) / 0.02
        )
        target_relative_rotation = ca.DM(target_rotations[frame, 0].T @ target_rotations[frame, 1])
        objective += (
            cfg.bimanual_orientation_weight
            * _rotation_loss(left_rotation.T @ right_rotation, target_relative_rotation)
            / np.deg2rad(15.0) ** 2
        )

    for previous, current in zip(sample_indices, sample_indices[1:], strict=False):
        dt = (current - previous) / fps
        if dt <= 0:
            continue
        for arm in range(2):
            achieved_velocity = (achieved[(current, arm)][0] - achieved[(previous, arm)][0]) / dt
            target_velocity = (
                target_positions[current, arm] - target_positions[previous, arm]
            ) / dt
            objective += cfg.flow_weight * ca.sumsqr(
                (achieved_velocity - ca.DM(target_velocity)) / 0.2
            )

    # Active collision frames are taken from the validation-resolution geometry
    # of the warm start. Keep primary tracking inside its hard envelope while the
    # secondary objective changes posture to clear the offending link pair.
    tracking_position_limit = 0.995 * cfg.tracking_position_tolerance_m
    tracking_orientation_limit = 0.995 * cfg.tracking_orientation_tolerance_rad
    for frame in sorted(active_set.frames):
        for arm in range(2):
            world_position, rotation, *_ = achieved[(frame, arm)]
            position_error = world_position - ca.DM(target_positions[frame, arm])
            opti.subject_to(ca.sumsqr(position_error) <= tracking_position_limit**2)
            relative_rotation = ca.DM(target_rotations[frame, arm]).T @ rotation
            opti.subject_to(
                ca.trace(relative_rotation)
                >= 1.0 + 2.0 * np.cos(tracking_orientation_limit)
            )

    derivative_indices = sample_indices
    velocity = ca.DM(csc_matrix(basis.velocity[derivative_indices])) @ controls
    acceleration = ca.DM(csc_matrix(basis.acceleration[derivative_indices])) @ controls
    jerk = ca.DM(csc_matrix(basis.jerk[derivative_indices])) @ controls
    objective += cfg.joint_velocity_weight * ca.sumsqr(velocity) / velocity.numel()
    objective += (
        cfg.joint_acceleration_weight * ca.sumsqr(acceleration) / (10.0**2 * acceleration.numel())
    )
    objective += cfg.joint_jerk_weight * ca.sumsqr(jerk) / (100.0**2 * jerk.numel())
    if enable_secondary_objective:
        joint_range = np.tile(np.maximum(kin.upper - kin.lower, 1e-6), 2)
        reference = ca.DM(initial_controls)
        range_grid = ca.repmat(ca.DM(joint_range).T, controls_count, 1)
        objective += (
            cfg.joint_reference_weight
            * ca.sumsqr((controls - reference) / range_grid)
            / controls.numel()
        )
        normalized = (controls - ca.DM(np.tile(lower, (controls_count, 1)))) / range_grid
        # A smooth symmetric barrier discourages joint-limit grazing without pulling
        # both arms toward the same, non-mirrored joint-space midpoint posture.
        limit_epsilon = 0.03
        objective += (
            cfg.joint_limit_margin_weight
            * ca.sum2(
                ca.sum1(
                    1.0 / (normalized + limit_epsilon) ** 2
                    + 1.0 / (1.0 - normalized + limit_epsilon) ** 2
                )
            )
            / controls.numel()
        )
    else:
        middle = ca.DM(np.tile(0.5 * (kin.lower + kin.upper), 2))
        objective += (
            cfg.joint_center_weight
            * ca.sumsqr(controls - ca.repmat(middle.T, controls_count, 1))
            / controls.numel()
        )

    collision_indices = sorted(
        set(sample_indices[:: max(1, cfg.collision_stride // cfg.objective_stride)])
        | {anchor.frame for anchor in anchors}
    )
    for frame in collision_indices:
        left_points = achieved[(frame, 0)][2]
        right_points = achieved[(frame, 1)][2]
        left_samples = _symbolic_collision_samples(left_points)
        right_samples = _symbolic_collision_samples(right_points)
        for left_point, left_radius, _ in left_samples:
            for right_point, right_radius, _ in right_samples:
                clearance = left_radius + right_radius + cfg.interarm_collision_margin_m
                distance = _smooth_norm(left_point - right_point)
                opti.subject_to(distance >= clearance)
                if enable_secondary_objective:
                    objective += cfg.collision_avoidance_weight * _squared_soft_margin(
                        distance,
                        clearance + cfg.collision_influence_m,
                        cfg.collision_influence_m,
                    )
        for points in (left_points, right_points):
            samples = _symbolic_collision_samples(points)
            for first, second in _SELF_COLLISION_SEGMENT_PAIRS:
                first_samples = [sample for sample in samples if sample[2] == first]
                second_samples = [sample for sample in samples if sample[2] == second]
                for first_point, first_radius, _ in first_samples:
                    for second_point, second_radius, _ in second_samples:
                        clearance = first_radius + second_radius + cfg.self_collision_margin_m
                        distance = _smooth_norm(first_point - second_point)
                        opti.subject_to(distance >= clearance)
                        if enable_secondary_objective:
                            objective += cfg.collision_avoidance_weight * _squared_soft_margin(
                                distance,
                                clearance + cfg.collision_influence_m,
                                cfg.collision_influence_m,
                            )
        if cfg.table_height_m is not None:
            minimum_z = cfg.table_height_m + cfg.table_clearance_m
            for points in (left_points, right_points):
                for point, radius, _ in _symbolic_collision_samples(points):
                    opti.subject_to(point[2] - radius >= minimum_z)

    for frame, left_index, right_index in active_set.interarm:
        left_point, left_radius, _ = _symbolic_collision_sample(
            achieved[(frame, 0)][2], left_index
        )
        right_point, right_radius, _ = _symbolic_collision_sample(
            achieved[(frame, 1)][2], right_index
        )
        clearance = (
            left_radius
            + right_radius
            + cfg.interarm_collision_margin_m
            + cfg.collision_constraint_buffer_m
        )
        distance = _smooth_norm(left_point - right_point)
        opti.subject_to(distance >= clearance)
        objective += cfg.collision_avoidance_weight * _squared_soft_margin(
            distance,
            clearance + cfg.collision_influence_m,
            cfg.collision_influence_m,
        )

    for frame, arm, first_index, second_index in active_set.self_collision:
        first_point, first_radius, _ = _symbolic_collision_sample(
            achieved[(frame, arm)][2], first_index
        )
        second_point, second_radius, _ = _symbolic_collision_sample(
            achieved[(frame, arm)][2], second_index
        )
        clearance = (
            first_radius
            + second_radius
            + cfg.self_collision_margin_m
            + cfg.collision_constraint_buffer_m
        )
        distance = _smooth_norm(first_point - second_point)
        opti.subject_to(distance >= clearance)
        objective += cfg.collision_avoidance_weight * _squared_soft_margin(
            distance,
            clearance + cfg.collision_influence_m,
            cfg.collision_influence_m,
        )

    opti.minimize(objective)
    opti.set_initial(controls, initial_controls)
    opti.solver(
        "ipopt",
        {"expand": False, "print_time": False},
        {
            "print_level": 0,
            "sb": "yes",
            "max_iter": cfg.solver_max_iterations,
            "tol": 1e-5,
            "acceptable_tol": 1e-3,
            "acceptable_iter": 8,
        },
    )
    try:
        solution = opti.solve()
        values = np.asarray(solution.value(controls), dtype=np.float64)
        stats = solution.stats()
        metadata = {
            "success": bool(stats.get("success", True)),
            "return_status": str(stats.get("return_status", "Solve_Succeeded")),
            "iterations": int(stats.get("iter_count", 0)),
            "objective": float(solution.value(objective)),
            "active_interarm_constraints": len(active_set.interarm),
            "active_self_collision_constraints": len(active_set.self_collision),
        }
    except RuntimeError as exc:
        try:
            values = np.asarray(opti.debug.value(controls), dtype=np.float64)
        except RuntimeError:
            values = initial_controls
        metadata = {
            "success": False,
            "return_status": "solver_failed",
            "iterations": 0,
            "objective": None,
            "active_interarm_constraints": len(active_set.interarm),
            "active_self_collision_constraints": len(active_set.self_collision),
            "error": str(exc).splitlines()[-1],
        }
    return values, metadata


@lru_cache(maxsize=1)
def _symbolic_fk() -> Any:
    q = ca.MX.sym("q", 6)
    transform: Any = ca.DM.eye(4)
    points: list[Any] = [ca.DM.zeros(3, 1)]
    for index, joint in enumerate(YAM_JOINTS):
        origin_rotation = Rotation.from_euler("xyz", joint.origin_rpy).as_matrix()
        transform = transform @ _homogeneous(ca.DM(origin_rotation), ca.DM(joint.origin_xyz))
        points.append(transform[:3, 3])
        axis = np.asarray(joint.axis, dtype=np.float64)
        skew = ca.DM(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ]
        )
        rotation = ca.DM.eye(3) + ca.sin(q[index]) * skew + (1 - ca.cos(q[index])) * (skew @ skew)
        transform = transform @ _homogeneous(rotation, ca.DM.zeros(3, 1))
    transform = transform @ _homogeneous(ca.DM(GRASP_SITE_ROTATION), ca.DM(GRASP_SITE_TRANSLATION))
    points.append(transform[:3, 3])
    return ca.Function("yam_fk", [q], [transform[:3, 3], transform[:3, :3], ca.horzcat(*points)])


def _homogeneous(rotation: Any, translation: Any) -> Any:
    return ca.vertcat(ca.horzcat(rotation, ca.reshape(translation, 3, 1)), ca.DM([[0, 0, 0, 1]]))


def _pseudo_huber(value: Any) -> Any:
    return ca.sqrt(1.0 + value**2) - 1.0


def _smooth_norm(value: Any) -> Any:
    return ca.sqrt(ca.sumsqr(value) + 1e-12)


def _squared_soft_margin(distance: Any, influence_distance: float, scale: float) -> Any:
    """Smooth squared hinge that repels geometry before a hard clearance activates."""
    violation = influence_distance - distance
    positive = 0.5 * (violation + ca.sqrt(violation**2 + 1e-10))
    return (positive / max(scale, 1e-6)) ** 2


def _rotation_loss(rotation: Any, target: Any) -> Any:
    return 3.0 - ca.trace(target.T @ rotation)


def _symbolic_collision_samples(points: Any) -> list[tuple[Any, float, int]]:
    samples: list[tuple[Any, float, int]] = []
    for segment, radius in zip(_COLLISION_SEGMENTS, _COLLISION_RADII_M, strict=True):
        for fraction in (0.5,):
            point = (1.0 - fraction) * points[:, segment] + fraction * points[:, segment + 1]
            samples.append((point, radius, segment))
    return samples


def _symbolic_collision_sample(points: Any, sample_index: int) -> tuple[Any, float, int]:
    fractions_count = len(_COLLISION_SAMPLE_FRACTIONS)
    segment_slot, fraction_slot = divmod(sample_index, fractions_count)
    segment = _COLLISION_SEGMENTS[segment_slot]
    radius = _COLLISION_RADII_M[segment_slot]
    fraction = _COLLISION_SAMPLE_FRACTIONS[fraction_slot]
    point = (1.0 - fraction) * points[:, segment] + fraction * points[:, segment + 1]
    return point, radius, segment


def _numeric_collision_samples(points: np.ndarray) -> np.ndarray:
    """Return (..., segment, sample, xyz) capsule-axis samples."""
    points = np.asarray(points, dtype=np.float64)
    starts = points[..., np.asarray(_COLLISION_SEGMENTS), :]
    ends = points[..., np.asarray(_COLLISION_SEGMENTS) + 1, :]
    fractions = np.asarray(_COLLISION_SAMPLE_FRACTIONS, dtype=np.float64)
    return (
        starts[..., :, None, :] * (1.0 - fractions[None, :, None])
        + ends[..., :, None, :] * fractions[None, :, None]
    )


def interarm_surface_clearance_profile_m(links: np.ndarray) -> np.ndarray:
    """Return the minimum sampled capsule surface clearance at every frame."""
    collision_samples = _numeric_collision_samples(links)
    radii = np.repeat(np.asarray(_COLLISION_RADII_M), len(_COLLISION_SAMPLE_FRACTIONS))
    left = collision_samples[:, 0].reshape(len(links), -1, 3)
    right = collision_samples[:, 1].reshape(len(links), -1, 3)
    surface = (
        np.linalg.norm(left[:, :, None, :] - right[:, None, :, :], axis=-1)
        - (radii[:, None] + radii[None, :])[None, :, :]
    )
    return np.min(surface, axis=(1, 2))


def _collect_active_collision_constraints(
    links: np.ndarray, cfg: TrajectoryOptimizationConfig
) -> _CollisionActiveSet:
    """Select the worst validation-resolution collision pair at each unsafe frame."""
    collision_samples = _numeric_collision_samples(links)
    radii = np.repeat(np.asarray(_COLLISION_RADII_M), len(_COLLISION_SAMPLE_FRACTIONS))
    left = collision_samples[:, 0].reshape(len(links), -1, 3)
    right = collision_samples[:, 1].reshape(len(links), -1, 3)
    interarm_surface = (
        np.linalg.norm(left[:, :, None, :] - right[:, None, :, :], axis=-1)
        - (radii[:, None] + radii[None, :])[None, :, :]
    )
    minimum_interarm_surface = float(np.min(interarm_surface))
    interarm: list[tuple[int, int, int]] = []
    unsafe_interarm_frames = np.flatnonzero(
        np.min(interarm_surface, axis=(1, 2)) < cfg.interarm_collision_margin_m
    )
    for frame in unsafe_interarm_frames:
        left_index, right_index = np.unravel_index(
            np.argmin(interarm_surface[frame]), interarm_surface[frame].shape
        )
        interarm.append((int(frame), int(left_index), int(right_index)))

    segment_lookup = {segment: index for index, segment in enumerate(_COLLISION_SEGMENTS)}
    self_collision: list[tuple[int, int, int, int]] = []
    minimum_self_surface = float("inf")
    samples_per_segment = len(_COLLISION_SAMPLE_FRACTIONS)
    for arm in range(2):
        per_frame: list[list[tuple[float, int, int]]] = [[] for _ in range(len(links))]
        for first_segment, second_segment in _SELF_COLLISION_SEGMENT_PAIRS:
            first_slot = segment_lookup[first_segment]
            second_slot = segment_lookup[second_segment]
            first = collision_samples[:, arm, first_slot]
            second = collision_samples[:, arm, second_slot]
            surface = (
                np.linalg.norm(first[:, :, None, :] - second[:, None, :, :], axis=-1)
                - (_COLLISION_RADII_M[first_slot] + _COLLISION_RADII_M[second_slot])
            )
            minimum_self_surface = min(minimum_self_surface, float(np.min(surface)))
            for frame in np.flatnonzero(
                np.min(surface, axis=(1, 2)) < cfg.self_collision_margin_m
            ):
                first_fraction, second_fraction = np.unravel_index(
                    np.argmin(surface[frame]), surface[frame].shape
                )
                first_index = first_slot * samples_per_segment + int(first_fraction)
                second_index = second_slot * samples_per_segment + int(second_fraction)
                per_frame[int(frame)].append(
                    (
                        float(surface[frame, first_fraction, second_fraction]),
                        first_index,
                        second_index,
                    )
                )
        for frame, candidates in enumerate(per_frame):
            if candidates:
                _, first_index, second_index = min(candidates)
                self_collision.append((frame, arm, first_index, second_index))
    return _CollisionActiveSet(
        tuple(interarm),
        tuple(self_collision),
        minimum_interarm_surface,
        minimum_self_surface,
    )


def _primary_tracking_feasible(
    achieved_positions: np.ndarray,
    achieved_quaternions: np.ndarray,
    target_positions: np.ndarray,
    target_quaternions: np.ndarray,
    events: tuple[np.ndarray, np.ndarray],
    cfg: TrajectoryOptimizationConfig,
) -> bool:
    position_error = np.linalg.norm(achieved_positions - target_positions, axis=-1)
    orientation_error = _orientation_error(achieved_quaternions, target_quaternions)
    if np.any(position_error > cfg.tracking_position_tolerance_m + 1e-6):
        return False
    if np.any(orientation_error > cfg.tracking_orientation_tolerance_rad + 1e-6):
        return False
    for arm in range(2):
        if np.any(position_error[events[arm], arm] > cfg.contact_position_tolerance_m + 1e-6):
            return False
        if np.any(
            orientation_error[events[arm], arm]
            > cfg.contact_orientation_tolerance_rad + 1e-6
        ):
            return False
    return True


def _evaluate_trajectory(
    q: np.ndarray, kin: YAMKinematics, bases: tuple[np.ndarray, np.ndarray]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = len(q)
    links = np.empty((frames, 2, 8, 3), dtype=np.float64)
    positions = np.empty((frames, 2, 3), dtype=np.float64)
    quaternions = np.empty((frames, 2, 4), dtype=np.float64)
    for arm in range(2):
        base_rotation = Rotation.from_matrix(bases[arm][:3, :3])
        for frame in range(frames):
            fk = kin.forward(q[frame, arm])
            links[frame, arm] = transform_points(fk.points, bases[arm])
            positions[frame, arm] = links[frame, arm, -1]
            world_rotation = base_rotation * Rotation.from_matrix(fk.rotation)
            if world_rotation is NotImplemented:
                raise TypeError("could not compose rotations")
            quaternions[frame, arm] = world_rotation.as_quat()
    return links, positions, quaternions


def _orientation_error(achieved: np.ndarray, target: np.ndarray) -> np.ndarray:
    achieved_rot = Rotation.from_quat(achieved.reshape(-1, 4))
    target_rot = Rotation.from_quat(target.reshape(-1, 4))
    delta = target_rot.inv() * achieved_rot
    if delta is NotImplemented:
        raise TypeError("could not compose rotations")
    return delta.magnitude().reshape(achieved.shape[:2])


def _camera_alignment(
    achieved_positions: np.ndarray,
    achieved_quaternions: np.ndarray,
    target_positions: np.ndarray,
    target_quaternions: np.ndarray,
    source_timestamps: np.ndarray,
    robot_timestamps: np.ndarray,
    events: tuple[np.ndarray, np.ndarray],
    cfg: TrajectoryOptimizationConfig,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    shape = achieved_positions.shape[:2]
    yam_translation = np.asarray(cfg.yam_tool_to_camera_translation_m, dtype=np.float64)
    source_translation = np.asarray(cfg.source_tool_to_camera_translation_m, dtype=np.float64)
    achieved_tool_rotation = Rotation.from_quat(achieved_quaternions.reshape(-1, 4))
    target_tool_rotation = Rotation.from_quat(target_quaternions.reshape(-1, 4))
    achieved_camera_positions = achieved_positions.reshape(-1, 3) + achieved_tool_rotation.apply(
        np.tile(yam_translation, (achieved_positions.size // 3, 1))
    )
    target_camera_positions = target_positions.reshape(-1, 3) + target_tool_rotation.apply(
        np.tile(source_translation, (target_positions.size // 3, 1))
    )
    achieved_camera_rotation = Rotation.from_matrix(
        achieved_tool_rotation.as_matrix()
        @ Rotation.from_euler("xyz", cfg.yam_tool_to_camera_rpy_rad).as_matrix()
    )
    target_camera_rotation = Rotation.from_matrix(
        target_tool_rotation.as_matrix()
        @ Rotation.from_euler("xyz", cfg.source_tool_to_camera_rpy_rad).as_matrix()
    )
    camera_delta = target_camera_rotation.inv() * achieved_camera_rotation
    if camera_delta is NotImplemented:
        raise TypeError("could not compose camera rotations")
    position_error = np.linalg.norm(
        achieved_camera_positions - target_camera_positions, axis=1
    ).reshape(shape)
    orientation_error = camera_delta.magnitude().reshape(shape)
    valid = (position_error <= 0.01) & (orientation_error <= np.deg2rad(2.0))
    proxy_weight = np.exp(
        -0.5 * ((position_error / 0.01) ** 2 + (orientation_error / np.deg2rad(2.0)) ** 2)
    )
    calibrated = cfg.camera_extrinsics_calibrated
    video_weight = proxy_weight if calibrated else np.ones_like(proxy_weight)
    fraction_valid = float(np.mean(valid))
    if not calibrated:
        status = "uncalibrated_proxy_do_not_filter"
    elif fraction_valid >= 0.9:
        status = "compatible_after_time_warp"
    else:
        status = "large_viewpoint_deviation"
    source_elapsed = source_timestamps - source_timestamps[0]
    robot_elapsed = robot_timestamps - robot_timestamps[0]
    temporal_delay = robot_elapsed - source_elapsed
    event_frames = (
        np.unique(np.concatenate(events))
        if any(len(item) for item in events)
        else np.array([], dtype=int)
    )
    event_delay = temporal_delay[event_frames]
    metadata = {
        "status": status,
        "extrinsics_calibrated": calibrated,
        "proxy_when_uncalibrated": not calibrated,
        "weights_actionable": calibrated,
        "source_tool_to_camera_translation_m": list(cfg.source_tool_to_camera_translation_m),
        "source_tool_to_camera_rpy_deg": np.degrees(cfg.source_tool_to_camera_rpy_rad).tolist(),
        "yam_tool_to_camera_translation_m": list(cfg.yam_tool_to_camera_translation_m),
        "yam_tool_to_camera_rpy_deg": np.degrees(cfg.yam_tool_to_camera_rpy_rad).tolist(),
        "position_error_median_mm": float(1000 * np.median(position_error)),
        "position_error_p95_mm": float(1000 * np.percentile(position_error, 95)),
        "orientation_error_median_deg": float(np.degrees(np.median(orientation_error))),
        "orientation_error_p95_deg": float(np.degrees(np.percentile(orientation_error, 95))),
        "vision_action_alignment_fraction_10mm_2deg": fraction_valid,
        "recommended_keep_fraction": float(np.mean(proxy_weight >= 0.25)),
        "requires_new_camera_recording": bool(calibrated and fraction_valid < 0.75),
        "temporal_mapping": {
            "method": "paired_source_and_robot_timestamps",
            "robot_delay_end_s": float(temporal_delay[-1]),
            "robot_delay_p95_s": float(np.percentile(temporal_delay, 95)),
            "gripper_event_delay_max_s": float(np.max(event_delay, initial=0.0)),
            "maximum_source_clock_rate": float(
                np.max(np.diff(source_timestamps) / np.diff(robot_timestamps))
            ),
            "video_must_not_use_robot_timestamps_as_source_frame_timestamps": True,
        },
    }
    return metadata, position_error, orientation_error, video_weight


def _validate_trajectory(
    q: np.ndarray,
    links: np.ndarray,
    achieved_positions: np.ndarray,
    achieved_quaternions: np.ndarray,
    target_positions: np.ndarray,
    target_quaternions: np.ndarray,
    anchors: list[ContactAnchor],
    events: tuple[np.ndarray, np.ndarray],
    timestamps: np.ndarray,
    velocity_limits: np.ndarray,
    acceleration_limits: np.ndarray,
    cfg: TrajectoryOptimizationConfig,
    kin: YAMKinematics,
) -> dict[str, Any]:
    q_flat = np.column_stack((q[:, 0], q[:, 1]))
    dynamics = trajectory_dynamics(q_flat, timestamps)
    lower = np.tile(kin.lower, 2)
    upper = np.tile(kin.upper, 2)
    velocity_ratio = np.max(np.abs(dynamics["velocity_rad_s"]) / velocity_limits[None, :])
    acceleration_ratio = np.max(
        np.abs(dynamics["acceleration_rad_s2"]) / acceleration_limits[None, :]
    )
    collision_samples = _numeric_collision_samples(links)
    radii = np.repeat(np.asarray(_COLLISION_RADII_M), len(_COLLISION_SAMPLE_FRACTIONS))
    left_samples = collision_samples[:, 0].reshape(len(q), -1, 3)
    right_samples = collision_samples[:, 1].reshape(len(q), -1, 3)
    interarm_surface_clearance = (
        np.linalg.norm(left_samples[:, :, None, :] - right_samples[:, None, :, :], axis=-1)
        - (radii[:, None] + radii[None, :])[None, :, :]
    )
    minimum_interarm_surface = float(np.min(interarm_surface_clearance))
    minimum_self_surface = float("inf")
    segment_lookup = {segment: index for index, segment in enumerate(_COLLISION_SEGMENTS)}
    for arm in range(2):
        for first, second in _SELF_COLLISION_SEGMENT_PAIRS:
            first_index = segment_lookup[first]
            second_index = segment_lookup[second]
            first_points = collision_samples[:, arm, first_index]
            second_points = collision_samples[:, arm, second_index]
            distances = np.linalg.norm(
                first_points[:, :, None, :] - second_points[:, None, :, :], axis=-1
            ) - (_COLLISION_RADII_M[first_index] + _COLLISION_RADII_M[second_index])
            minimum_self_surface = min(minimum_self_surface, float(np.min(distances)))
    if cfg.table_height_m is None:
        minimum_table_surface = None
        passes_table = True
    else:
        sample_z = collision_samples[..., 2]
        radius_grid = np.asarray(_COLLISION_RADII_M)[None, None, :, None]
        minimum_table_surface = float(np.min(sample_z - radius_grid - cfg.table_height_m))
        passes_table = minimum_table_surface + 1e-9 >= cfg.table_clearance_m
    orientation_error = _orientation_error(achieved_quaternions, target_quaternions)
    position_error = np.linalg.norm(achieved_positions - target_positions, axis=-1)
    event_position: list[float] = []
    event_orientation: list[float] = []
    for arm in range(2):
        event_position.extend(position_error[events[arm], arm].tolist())
        event_orientation.extend(orientation_error[events[arm], arm].tolist())
    max_event_position = max(event_position, default=0.0)
    max_event_orientation = max(event_orientation, default=0.0)
    frame_valid = (position_error <= cfg.tracking_position_tolerance_m) & (
        orientation_error <= cfg.tracking_orientation_tolerance_rad
    )
    return {
        "passes_joint_limits": bool(
            np.all(q_flat >= lower[None, :] - 1e-7) and np.all(q_flat <= upper[None, :] + 1e-7)
        ),
        "passes_velocity_limits": bool(velocity_ratio <= 1.001),
        "passes_acceleration_limits": bool(acceleration_ratio <= 1.001),
        "passes_interarm_clearance": bool(
            minimum_interarm_surface + 1e-9 >= cfg.interarm_collision_margin_m
        ),
        "passes_self_collision_clearance": bool(
            minimum_self_surface + 1e-9 >= cfg.self_collision_margin_m
        ),
        "passes_table_clearance": bool(passes_table),
        "passes_tracking_accuracy": bool(np.all(frame_valid)),
        "passes_contact_accuracy": bool(
            all(anchor.feasible for anchor in anchors)
            and max_event_position <= cfg.contact_position_tolerance_m + 1e-6
            and max_event_orientation <= cfg.contact_orientation_tolerance_rad + 1e-6
        ),
        "peak_velocity_limit_ratio": float(velocity_ratio),
        "peak_acceleration_limit_ratio": float(acceleration_ratio),
        "collision_model": "public_YAM_MuJoCo_radii_sampled_capsule_proxy",
        "exact_scene_collision_check_still_required_before_hardware": True,
        "minimum_interarm_surface_clearance_mm": 1000 * minimum_interarm_surface,
        "minimum_self_surface_clearance_mm": 1000 * minimum_self_surface,
        "minimum_table_surface_clearance_mm": (
            None if minimum_table_surface is None else 1000 * minimum_table_surface
        ),
        "contact_position_error_max_mm": 1000 * max_event_position,
        "contact_orientation_error_max_deg": float(np.degrees(max_event_orientation)),
        "tracking_position_tolerance_mm": 1000 * cfg.tracking_position_tolerance_m,
        "tracking_orientation_tolerance_deg": float(
            np.degrees(cfg.tracking_orientation_tolerance_rad)
        ),
        "failed_tracking_arm_frames": int(np.count_nonzero(~frame_valid)),
        "failed_tracking_bimanual_frames": int(
            np.count_nonzero(~np.all(frame_valid, axis=1))
        ),
        "strict_contact_anchors_feasible": sum(anchor.feasible for anchor in anchors),
        "strict_contact_anchors_total": len(anchors),
    }


def _flow_direction_cosine(achieved: np.ndarray, target: np.ndarray) -> float:
    achieved_delta = np.diff(achieved, axis=0).reshape(-1, 3)
    target_delta = np.diff(target, axis=0).reshape(-1, 3)
    moving = np.linalg.norm(target_delta, axis=1) > 1e-5
    if not np.any(moving):
        return 1.0
    numerator = np.sum(achieved_delta[moving] * target_delta[moving], axis=1)
    denominator = np.linalg.norm(achieved_delta[moving], axis=1) * np.linalg.norm(
        target_delta[moving], axis=1
    )
    return float(np.median(numerator / np.maximum(denominator, 1e-12)))


def _serializable_config(config: TrajectoryOptimizationConfig) -> dict[str, Any]:
    result = asdict(config)
    result["contact_orientation_tolerance_deg"] = float(
        np.degrees(result.pop("contact_orientation_tolerance_rad"))
    )
    result["tracking_orientation_tolerance_deg"] = float(
        np.degrees(result.pop("tracking_orientation_tolerance_rad"))
    )
    return result
