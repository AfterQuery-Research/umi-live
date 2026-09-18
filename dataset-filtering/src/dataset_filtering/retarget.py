from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import qmc

from dataset_filtering.kinematics import (
    YAMKinematics,
    transform_points,
    transform_pose_to_local,
)
from dataset_filtering.models import ArmRetargetResult, PoseEpisode, RetargetResult
from dataset_filtering.smoothing import detect_gripper_events

INCHES_TO_METRES = 0.0254
DEFAULT_PAIR_SEPARATION_M = 18.0 * INCHES_TO_METRES

# UMI state convention from umi_flow_core.geom: +X up, +Y right, +Z forward.
# Robot-world convention here: +X forward, +Y left, +Z up.
UMI_TO_ROBOT_BASIS = np.asarray(
    [
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)


def yam_base_transforms(
    separation_m: float = DEFAULT_PAIR_SEPARATION_M,
) -> tuple[np.ndarray, np.ndarray]:
    """umi1/YAM-left is +Y; umi2/YAM-right is -Y; both face +X."""
    left = np.eye(4, dtype=np.float64)
    right = np.eye(4, dtype=np.float64)
    left[1, 3] = separation_m / 2.0
    right[1, 3] = -separation_m / 2.0
    return left, right


def map_umi_targets_to_yam_world(
    episode: PoseEpisode,
    *,
    separation_m: float = DEFAULT_PAIR_SEPARATION_M,
    kinematics: YAMKinematics | None = None,
    aligned_shared_frame: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Map UMI poses into the fixed-base YAM world frame.

    Legacy/non-aligned episodes contain one independent origin per UMI, so each
    arm's first pose is anchored to its corresponding zero-joint TCP. Aligned
    episodes instead contain both UMIs in umi1's initial frame. For those, retain
    the complete initial relative transform and center the recorded TCP pair over
    the two fixed YAM bases.
    """
    kin = kinematics or YAMKinematics()
    home = kin.forward(kin.home)
    bases = yam_base_transforms(separation_m)
    n = len(episode.timestamps)
    positions_world = np.empty((n, 2, 3), dtype=np.float64)
    quaternions_world = np.empty((n, 2, 4), dtype=np.float64)
    if aligned_shared_frame:
        mapped_positions = episode.positions @ UMI_TO_ROBOT_BASIS.T
        home_positions = np.stack(
            [transform_points(home.position[None, :], base)[0] for base in bases]
        )
        translation = np.mean(home_positions, axis=0) - np.mean(
            mapped_positions[0], axis=0
        )
        positions_world[:] = mapped_positions + translation[None, None, :]
        for arm in range(2):
            source_rotations = Rotation.from_quat(episode.quaternions[:, arm])
            mapped_rotations = np.einsum(
                "ij,njk,kl->nil",
                UMI_TO_ROBOT_BASIS,
                source_rotations.as_matrix(),
                UMI_TO_ROBOT_BASIS.T,
            )
            local_targets = Rotation.from_matrix(
                mapped_rotations @ home.rotation[None, :, :]
            )
            base_rotation = Rotation.from_matrix(bases[arm][:3, :3])
            world_targets = base_rotation * local_targets
            if world_targets is NotImplemented:
                raise TypeError("could not compose rotations")
            quaternions_world[:, arm] = world_targets.as_quat()
        return positions_world, quaternions_world

    for arm in range(2):
        relative_position = episode.positions[:, arm] - episode.positions[0, arm]
        mapped_position = np.column_stack(
            (relative_position[:, 2], -relative_position[:, 1], relative_position[:, 0])
        )
        home_world = transform_points(home.position[None, :], bases[arm])[0]
        positions_world[:, arm] = home_world + mapped_position

        raw_rotations = Rotation.from_quat(episode.quaternions[:, arm])
        relative_rotations = raw_rotations[0].inv() * raw_rotations
        if relative_rotations is NotImplemented:
            raise TypeError("could not compose rotations")
        mapped_world_deltas = np.einsum(
            "ij,njk,kl->nil",
            UMI_TO_ROBOT_BASIS,
            relative_rotations.as_matrix(),
            UMI_TO_ROBOT_BASIS.T,
        )
        # The mapped delta is expressed in robot-world axes, so it must pre-multiply
        # the home grasp orientation. Post-multiplication would reinterpret a world
        # delta in the old tool's local axes and invert/permutate physical UMI motions.
        local_targets = Rotation.from_matrix(mapped_world_deltas @ home.rotation[None, :, :])
        base_rotation = Rotation.from_matrix(bases[arm][:3, :3])
        world_targets = base_rotation * local_targets
        if world_targets is NotImplemented:
            raise TypeError("could not compose rotations")
        quaternions_world[:, arm] = world_targets.as_quat()
    return positions_world, quaternions_world


def retarget_pose_episode(
    episode: PoseEpisode,
    *,
    separation_m: float = DEFAULT_PAIR_SEPARATION_M,
    position_tolerance_m: float = 0.01,
    contact_position_tolerance_m: float = 0.005,
    orientation_tolerance_rad: float = np.deg2rad(8.0),
    max_iterations: int = 90,
    aligned_shared_frame: bool = False,
) -> RetargetResult:
    kin = YAMKinematics()
    bases = yam_base_transforms(separation_m)
    positions_world, quaternions_world = map_umi_targets_to_yam_world(
        episode,
        separation_m=separation_m,
        kinematics=kin,
        aligned_shared_frame=aligned_shared_frame,
    )
    event_indices = detect_gripper_events(episode.grippers, fps=episode.fps)
    arms: list[ArmRetargetResult] = []
    for arm in range(2):
        arms.append(
            _retarget_arm(
                kin,
                bases[arm],
                positions_world[:, arm],
                quaternions_world[:, arm],
                event_indices[arm],
                position_tolerance_m,
                contact_position_tolerance_m,
                orientation_tolerance_rad,
                max_iterations,
                robust_initial=aligned_shared_frame,
            )
        )
    metadata: dict[str, Any] = {
        "pair_separation_m": separation_m,
        "pair_separation_in": separation_m / INCHES_TO_METRES,
        "source_pose_frame": (
            "aligned_shared_umi1_initial" if aligned_shared_frame else "independent_per_arm"
        ),
        "mapping": "UMI (+X up,+Y right,+Z forward) -> YAM world (+X forward,+Y left,+Z up)",
        "playback_on_ik_failure": "hold last reachable joint pose until the next reachable target",
        "position_tolerance_m": position_tolerance_m,
        "contact_position_tolerance_m": contact_position_tolerance_m,
        "orientation_tolerance_deg": float(np.degrees(orientation_tolerance_rad)),
        "reachability": {
            "umi1_left_fraction": float(np.mean(arms[0].reachable)),
            "umi2_right_fraction": float(np.mean(arms[1].reachable)),
            "both_fraction": float(np.mean(arms[0].reachable & arms[1].reachable)),
        },
        "event_frames": {
            "umi1_left": event_indices[0].tolist(),
            "umi2_right": event_indices[1].tolist(),
        },
        "event_position_error_mm": {
            "umi1_left_max": _event_max(arms[0].position_error_m, event_indices[0]) * 1000,
            "umi2_right_max": _event_max(arms[1].position_error_m, event_indices[1]) * 1000,
        },
        "contact_events_within_tolerance": {
            "umi1_left_fraction": _event_fraction(
                arms[0].position_error_m, event_indices[0], contact_position_tolerance_m
            ),
            "umi2_right_fraction": _event_fraction(
                arms[1].position_error_m, event_indices[1], contact_position_tolerance_m
            ),
        },
    }
    return RetargetResult(
        arms=(arms[0], arms[1]),
        target_positions_world=positions_world,
        target_quaternions_world=quaternions_world,
        event_indices=event_indices,
        metadata=metadata,
    )


def playback_link_points(
    result: RetargetResult, *, separation_m: float = DEFAULT_PAIR_SEPARATION_M
) -> np.ndarray:
    kin = YAMKinematics()
    bases = yam_base_transforms(separation_m)
    n = len(result.target_positions_world)
    points = np.empty((n, 2, 8, 3), dtype=np.float64)
    for arm in range(2):
        for frame, q in enumerate(result.arms[arm].playback_q):
            points[frame, arm] = transform_points(kin.forward(q).points, bases[arm])
    return points


def _retarget_arm(
    kin: YAMKinematics,
    base: np.ndarray,
    positions_world: np.ndarray,
    quaternions_world: np.ndarray,
    event_indices: np.ndarray,
    position_tolerance_m: float,
    contact_position_tolerance_m: float,
    orientation_tolerance_rad: float,
    max_iterations: int,
    *,
    robust_initial: bool = False,
) -> ArmRetargetResult:
    n = len(positions_world)
    solutions = np.empty((n, 6), dtype=np.float64)
    playback = np.empty((n, 6), dtype=np.float64)
    reachable = np.zeros(n, dtype=bool)
    projected_positions = np.empty((n, 3), dtype=np.float64)
    projected_quaternions = np.empty((n, 4), dtype=np.float64)
    position_errors = np.empty(n, dtype=np.float64)
    orientation_errors = np.empty(n, dtype=np.float64)
    seed = kin.home.copy()
    last_valid = kin.home.copy()
    event_set = set(int(index) for index in event_indices)
    for frame in range(n):
        local_position, local_quaternion = transform_pose_to_local(
            positions_world[frame], quaternions_world[frame], base
        )
        frame_position_tolerance = (
            contact_position_tolerance_m if frame in event_set else position_tolerance_m
        )
        if frame == 0 and robust_initial:
            sampler = qmc.Halton(d=6, scramble=True, seed=7919)
            random_unit = sampler.random(16)
            random_seeds = kin.lower + random_unit * (kin.upper - kin.lower)
            candidates = kin.solve_ik_multistart(
                local_position,
                local_quaternion,
                np.vstack((kin.home, 0.5 * (kin.lower + kin.upper), random_seeds)),
                position_tolerance_m=frame_position_tolerance,
                orientation_tolerance_rad=orientation_tolerance_rad,
                max_iterations=max(180, max_iterations),
            )
            ik = candidates[0]
        else:
            ik = kin.solve_ik(
                local_position,
                local_quaternion,
                initial_q=seed,
                position_tolerance_m=frame_position_tolerance,
                orientation_tolerance_rad=orientation_tolerance_rad,
                max_iterations=max_iterations,
            )
        solutions[frame] = ik.q
        reachable[frame] = ik.success
        position_errors[frame] = ik.position_error_m
        orientation_errors[frame] = ik.orientation_error_rad
        projected = kin.forward(ik.q)
        projected_positions[frame] = transform_points(projected.position[None, :], base)[0]
        base_rotation = Rotation.from_matrix(base[:3, :3])
        projected_world_rotation = base_rotation * Rotation.from_matrix(projected.rotation)
        if projected_world_rotation is NotImplemented:
            raise TypeError("could not compose rotations")
        projected_quaternions[frame] = projected_world_rotation.as_quat()
        if ik.success:
            seed = ik.q
            last_valid = ik.q
        playback[frame] = last_valid
    return ArmRetargetResult(
        q_solutions=solutions,
        playback_q=playback,
        reachable=reachable,
        projected_positions=projected_positions,
        projected_quaternions=projected_quaternions,
        position_error_m=position_errors,
        orientation_error_rad=orientation_errors,
    )


def _event_max(values: np.ndarray, events: np.ndarray) -> float:
    return float(np.max(values[events])) if len(events) else 0.0


def _event_fraction(values: np.ndarray, events: np.ndarray, tolerance: float) -> float:
    return float(np.mean(values[events] <= tolerance)) if len(events) else 1.0
