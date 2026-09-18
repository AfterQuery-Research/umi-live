from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np


class EpisodeKind(StrEnum):
    UMI_POSE = "umi_pose"
    YAM_JOINT = "yam_joint"


@dataclass(frozen=True)
class EpisodeSummary:
    episode_index: int
    length: int
    duration_s: float
    task: str
    data_chunk_index: int
    data_file_index: int


@dataclass(frozen=True)
class Episode:
    """A validated LeRobot episode in either UMI-pose or YAM-joint space."""

    episode_index: int
    fps: float
    timestamps: np.ndarray
    state: np.ndarray
    action: np.ndarray | None
    grippers: np.ndarray | None
    task: str
    kind: EpisodeKind
    state_names: tuple[str, ...]
    source: str
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        n = len(self.timestamps)
        if self.state.ndim != 2 or len(self.state) != n:
            raise ValueError("state must have shape (N, D) on the timestamp timeline")
        if self.action is not None:
            valid_action_shapes = {self.state.shape}
            if self.kind is EpisodeKind.UMI_POSE and self.state.shape[1] == 14:
                valid_action_shapes.add((n, 12))
            if self.action.shape not in valid_action_shapes:
                raise ValueError(
                    "action must match state shape, except calibrated 14-D UMI state "
                    "may use 12-D pose-delta action"
                )
        if self.grippers is not None and self.grippers.shape != (n, 2):
            raise ValueError("grippers must have shape (N, 2)")


@dataclass(frozen=True)
class PoseEpisode:
    """Bimanual poses: positions in metres and rotations as scipy xyzw quaternions."""

    episode_index: int
    fps: float
    timestamps: np.ndarray
    positions: np.ndarray  # (N, 2, 3), UMI coordinates
    quaternions: np.ndarray  # (N, 2, 4), xyzw
    grippers: np.ndarray | None  # (N, 2)
    task: str
    source: str

    def __post_init__(self) -> None:
        n = len(self.timestamps)
        if self.positions.shape != (n, 2, 3):
            raise ValueError("positions must have shape (N, 2, 3)")
        if self.quaternions.shape != (n, 2, 4):
            raise ValueError("quaternions must have shape (N, 2, 4)")
        if self.grippers is not None and self.grippers.shape != (n, 2):
            raise ValueError("grippers must have shape (N, 2)")


@dataclass(frozen=True)
class SmoothingResult:
    method: str
    parameters: dict[str, Any]
    positions: np.ndarray
    quaternions: np.ndarray
    metrics: dict[str, float]
    score: float


@dataclass(frozen=True)
class IKResult:
    success: bool
    q: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    iterations: int


@dataclass(frozen=True)
class ArmRetargetResult:
    q_solutions: np.ndarray  # best solution per target, including failed projections
    playback_q: np.ndarray  # last-valid hold on failed targets
    reachable: np.ndarray
    projected_positions: np.ndarray
    projected_quaternions: np.ndarray
    position_error_m: np.ndarray
    orientation_error_rad: np.ndarray


@dataclass(frozen=True)
class RetargetResult:
    arms: tuple[ArmRetargetResult, ArmRetargetResult]
    target_positions_world: np.ndarray
    target_quaternions_world: np.ndarray
    event_indices: tuple[np.ndarray, np.ndarray]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ContactAnchor:
    arm: int
    frame: int
    feasible: bool
    q: np.ndarray
    position_error_m: float
    orientation_error_rad: float
    candidates: int


@dataclass(frozen=True)
class ContinuousRetargetResult:
    """A jointly optimized, time-parameterized bimanual YAM trajectory."""

    source_timestamps: np.ndarray
    robot_timestamps: np.ndarray
    q: np.ndarray  # (N, 2, 6)
    link_points_world: np.ndarray  # (N, 2, 8, 3)
    achieved_positions_world: np.ndarray  # (N, 2, 3)
    achieved_quaternions_world: np.ndarray  # (N, 2, 4)
    target_positions_world: np.ndarray  # (N, 2, 3)
    target_quaternions_world: np.ndarray  # (N, 2, 4)
    position_error_m: np.ndarray  # (N, 2)
    orientation_error_rad: np.ndarray  # (N, 2)
    camera_position_error_m: np.ndarray  # (N, 2)
    camera_orientation_error_rad: np.ndarray  # (N, 2)
    video_alignment_weight: np.ndarray  # (N, 2), actionable only when calibrated
    event_indices: tuple[np.ndarray, np.ndarray]
    contact_anchors: tuple[ContactAnchor, ...]
    frame_valid: np.ndarray  # (N, 2)
    metadata: dict[str, Any]
