from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from dataset_filtering.models import IKResult


@dataclass(frozen=True)
class JointSpec:
    name: str
    origin_xyz: tuple[float, float, float]
    origin_rpy: tuple[float, float, float]
    axis: tuple[float, float, float]
    lower: float
    upper: float


@dataclass(frozen=True)
class ForwardKinematics:
    position: np.ndarray
    rotation: np.ndarray
    points: np.ndarray
    joint_origins: np.ndarray
    joint_axes_world: np.ndarray


# Exact arm geometry from the I2RT revision pinned by the current BiYAM driver:
# https://github.com/AfterQuery-Research/i2rt/tree/7ed46f4e4e316133a0c39aa6cf34a73d2718e850
#
# The local joint axes look different from the older public YAM URDF because the
# link frames were re-authored. Evaluated in the base frame, the joint order and
# signs remain the same. Keep the values here in the pinned model's native frames
# so FK agrees directly with the model used by the hardware stack.
YAM_JOINTS: tuple[JointSpec, ...] = (
    JointSpec(
        "joint1",
        (-2.49777947129965e-7, -9.17402363407408e-13, 0.0679999999995413),
        (0.0, 1.5708, 3.1415963267949),
        (-1.0, 0.0, 0.0),
        -2.61799,
        3.14159,
    ),
    JointSpec(
        "joint2",
        (-0.0455, -0.0339, -0.02),
        (-1.7953e-16, 5.55112e-17, -3.14159),
        (3.58252e-15, 1.0, -3.2007e-16),
        -8.88178e-16,
        3.66519,
    ),
    JointSpec(
        "joint3",
        (4.08431e-7, -0.0688, 0.264),
        (-2.99502e-14, -1.6015e-14, -3.14159),
        (4.54273e-15, 1.0, 1.04275e-14),
        0.0,
        3.14159,
    ),
    JointSpec(
        "joint4",
        (-0.0600003, -0.0688, -0.244999),
        (2.95536e-14, 8.52374e-14, -2.10585e-14),
        (0.0, 1.0, 0.0),
        -1.69297,
        1.5708,
    ),
    JointSpec(
        "joint5",
        (-0.0405003, 0.0338995, -0.0739989),
        (2.73264e-14, -4.32956e-16, -8.32667e-17),
        (1.0, 0.0, 0.0),
        -1.5708,
        1.5708,
    ),
    JointSpec(
        "joint6",
        (0.0404996, 2.39858e-7, -0.0356),
        (8.83309e-16, -5.88588e-16, 7.69861e-14),
        (1.46014e-15, 2.40808e-14, -1.0),
        -2.0944,
        2.0944,
    ),
)

# Retain the established pass's planning-speed assumption so this rerun isolates the
# corrected FK/tool geometry. The dataset remains observation-only; these limits are
# used only for the unchanged continuous-path acceptance and retiming gates.
YAM_VELOCITY_LIMITS_RAD_S = np.full(6, 10.0, dtype=np.float64)

# Exact custom-UMI gripper grasp site from the pinned I2RT linear_4310 model.
# MuJoCo's quat="0 0 1 0" is wxyz: a 180-degree rotation about local +Y.
# At YAM home this produces the UMI-compatible grasp axes X=up, Y=right,
# Z=forward and places the grasp point 220 mm from the flange.
GRASP_SITE_TRANSLATION = np.asarray([0.0, 0.0, -0.220], dtype=np.float64)
GRASP_SITE_ROTATION = Rotation.from_quat([0.0, 1.0, 0.0, 0.0]).as_matrix()


class YAMKinematics:
    def __init__(self) -> None:
        self.joints = YAM_JOINTS
        self.lower = np.asarray([joint.lower for joint in self.joints], dtype=np.float64)
        self.upper = np.asarray([joint.upper for joint in self.joints], dtype=np.float64)
        self.home = np.zeros(6, dtype=np.float64)

    def forward(self, q: np.ndarray) -> ForwardKinematics:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (6,):
            raise ValueError(f"expected six YAM joints, got {q.shape}")
        transform = np.eye(4, dtype=np.float64)
        points = [np.zeros(3, dtype=np.float64)]
        origins: list[np.ndarray] = []
        axes: list[np.ndarray] = []
        for angle, joint in zip(q, self.joints, strict=True):
            transform = transform @ _transform(joint.origin_xyz, joint.origin_rpy)
            origins.append(transform[:3, 3].copy())
            local_axis = np.asarray(joint.axis, dtype=np.float64)
            axes.append(transform[:3, :3] @ local_axis)
            points.append(transform[:3, 3].copy())
            transform = transform @ _axis_rotation(local_axis, float(angle))
        site = np.eye(4, dtype=np.float64)
        site[:3, :3] = GRASP_SITE_ROTATION
        site[:3, 3] = GRASP_SITE_TRANSLATION
        transform = transform @ site
        points.append(transform[:3, 3].copy())
        return ForwardKinematics(
            position=transform[:3, 3].copy(),
            rotation=transform[:3, :3].copy(),
            points=np.asarray(points),
            joint_origins=np.asarray(origins),
            joint_axes_world=np.asarray(axes),
        )

    def solve_ik(
        self,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
        *,
        initial_q: np.ndarray | None = None,
        position_tolerance_m: float = 0.01,
        orientation_tolerance_rad: float = np.deg2rad(8.0),
        max_iterations: int = 90,
    ) -> IKResult:
        target_position = np.asarray(target_position, dtype=np.float64)
        target_rotation = Rotation.from_quat(
            np.asarray(target_quaternion, dtype=np.float64)
        ).as_matrix()
        seed = np.clip(
            self.home if initial_q is None else np.asarray(initial_q, dtype=np.float64),
            self.lower,
            self.upper,
        )
        middle = 0.5 * (self.lower + self.upper)
        seeds = [seed, self.home, middle]
        best: IKResult | None = None
        for candidate in seeds:
            result = self._solve_from_seed(
                target_position,
                target_rotation,
                candidate,
                position_tolerance_m,
                orientation_tolerance_rad,
                max_iterations,
            )
            if result.success:
                return result
            if best is None or _normalized_error(result) < _normalized_error(best):
                best = result
        assert best is not None
        return best

    def solve_ik_multistart(
        self,
        target_position: np.ndarray,
        target_quaternion: np.ndarray,
        seeds: np.ndarray,
        *,
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
        max_iterations: int = 180,
    ) -> list[IKResult]:
        """Return every distinct strict-tolerance solution found from explicit seeds."""
        target_rotation = Rotation.from_quat(
            np.asarray(target_quaternion, dtype=np.float64)
        ).as_matrix()
        coarse: list[IKResult] = []
        for seed in np.asarray(seeds, dtype=np.float64):
            result = self._solve_from_seed(
                np.asarray(target_position, dtype=np.float64),
                target_rotation,
                seed,
                position_tolerance_m,
                orientation_tolerance_rad,
                max_iterations,
            )
            coarse.append(result)

        # Numerical least-squares is the accurate but expensive stage. Refine only the
        # best basins discovered by the broad damped-least-squares sweep, rather than
        # running a second optimizer from every low-discrepancy seed.
        ranked = sorted(coarse, key=_normalized_error)
        coarse_successes = [result for result in ranked if result.success]
        refinement_starts = (coarse_successes or ranked)[:4]
        solutions: list[IKResult] = []
        best_failure: IKResult | None = None
        for coarse_result in refinement_starts:
            result = self._refine_ik_least_squares(
                target_position,
                target_rotation,
                coarse_result.q,
                position_tolerance_m,
                orientation_tolerance_rad,
                max_iterations,
            )
            if result.success:
                if not any(np.linalg.norm(result.q - item.q) < 1e-3 for item in solutions):
                    solutions.append(result)
            elif best_failure is None or _normalized_error(result) < _normalized_error(
                best_failure
            ):
                best_failure = result
        if solutions:
            return solutions
        if best_failure is None:
            best_failure = ranked[0]
        assert best_failure is not None
        return [best_failure]

    def _refine_ik_least_squares(
        self,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        seed: np.ndarray,
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
        max_iterations: int,
    ) -> IKResult:
        """High-accuracy bounded refinement used for offline contact keyframes."""
        joint_range = np.maximum(self.upper - self.lower, 1e-6)

        def residual(q: np.ndarray) -> np.ndarray:
            fk = self.forward(q)
            rotation_error = Rotation.from_matrix(target_rotation @ fk.rotation.T).as_rotvec()
            return np.concatenate(
                (
                    (target_position - fk.position) / max(position_tolerance_m, 1e-6),
                    rotation_error / max(orientation_tolerance_rad, 1e-6),
                    1e-5 * (q - seed) / joint_range,
                )
            )

        refined = least_squares(
            residual,
            np.clip(seed, self.lower, self.upper),
            bounds=(self.lower, self.upper),
            method="trf",
            jac="3-point",
            x_scale="jac",
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=max(100, 2 * max_iterations),
        )
        fk = self.forward(refined.x)
        position_error = float(np.linalg.norm(target_position - fk.position))
        orientation_error = float(
            np.linalg.norm(Rotation.from_matrix(target_rotation @ fk.rotation.T).as_rotvec())
        )
        return IKResult(
            success=(
                position_error <= position_tolerance_m
                and orientation_error <= orientation_tolerance_rad
            ),
            q=refined.x,
            position_error_m=position_error,
            orientation_error_rad=orientation_error,
            iterations=int(refined.nfev),
        )

    def _solve_from_seed(
        self,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        seed: np.ndarray,
        position_tolerance_m: float,
        orientation_tolerance_rad: float,
        max_iterations: int,
    ) -> IKResult:
        q = np.clip(np.asarray(seed, dtype=np.float64).copy(), self.lower, self.upper)
        damping = 2e-3
        best_q = q.copy()
        best_errors = (float("inf"), float("inf"))
        best_cost = float("inf")
        for iteration in range(1, max_iterations + 1):
            fk = self.forward(q)
            pos_error = target_position - fk.position
            rot_error = Rotation.from_matrix(target_rotation @ fk.rotation.T).as_rotvec()
            pos_norm = float(np.linalg.norm(pos_error))
            rot_norm = float(np.linalg.norm(rot_error))
            cost = (pos_norm / 0.01) ** 2 + (rot_norm / np.deg2rad(8.0)) ** 2
            if cost < best_cost:
                best_q = q.copy()
                best_errors = (pos_norm, rot_norm)
                best_cost = cost
            if pos_norm <= position_tolerance_m and rot_norm <= orientation_tolerance_rad:
                return IKResult(True, q.copy(), pos_norm, rot_norm, iteration)

            jacobian = np.empty((6, 6), dtype=np.float64)
            for column in range(6):
                axis = fk.joint_axes_world[column]
                origin = fk.joint_origins[column]
                jacobian[:3, column] = np.cross(axis, fk.position - origin)
                jacobian[3:, column] = axis
            scale = np.asarray([100.0, 100.0, 100.0, 7.16197, 7.16197, 7.16197])
            weighted_jacobian = scale[:, None] * jacobian
            weighted_error = scale * np.concatenate((pos_error, rot_error))
            lhs = weighted_jacobian.T @ weighted_jacobian + damping * np.eye(6)
            rhs = weighted_jacobian.T @ weighted_error
            try:
                delta = np.linalg.solve(lhs, rhs)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
            delta = np.clip(delta, -0.18, 0.18)
            trial = np.clip(q + delta, self.lower, self.upper)
            if np.allclose(trial, q, atol=1e-10):
                damping = min(1.0, damping * 4.0)
            else:
                q = trial
                damping = max(1e-5, damping * 0.96)
        return IKResult(False, best_q, best_errors[0], best_errors[1], max_iterations)


def transform_points(points: np.ndarray, base_transform: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ base_transform[:3, :3].T + base_transform[:3, 3]


def transform_pose_to_local(
    position_world: np.ndarray, quaternion_world: np.ndarray, base_transform: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    base_rotation = Rotation.from_matrix(base_transform[:3, :3])
    local_position = base_rotation.inv().apply(
        np.asarray(position_world, dtype=np.float64) - base_transform[:3, 3]
    )
    local_rotation = base_rotation.inv() * Rotation.from_quat(quaternion_world)
    if local_rotation is NotImplemented:
        raise TypeError("could not compose rotations")
    return local_position, local_rotation.as_quat()


def _transform(xyz: tuple[float, float, float], rpy: tuple[float, float, float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    transform[:3, 3] = np.asarray(xyz)
    return transform


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
    return transform


def _normalized_error(result: IKResult) -> float:
    return (result.position_error_m / 0.01) ** 2 + (
        result.orientation_error_rad / np.deg2rad(8.0)
    ) ** 2
