from __future__ import annotations

import json
import threading
import time
import urllib.parse
import urllib.request
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from scipy.spatial.transform import Rotation

from dataset_filtering.combined_pipeline import zero_pose_origins
from dataset_filtering.hf_dataset import DEFAULT_REPO_ID, LeRobotSource, as_pose_episode
from dataset_filtering.kinematics import (
    YAM_VELOCITY_LIMITS_RAD_S,
    YAMKinematics,
    transform_points,
)
from dataset_filtering.models import EpisodeKind
from dataset_filtering.noise import pose_noise_metrics, vector_noise_metrics
from dataset_filtering.retarget import (
    DEFAULT_PAIR_SEPARATION_M,
    map_umi_targets_to_yam_world,
    playback_link_points,
    retarget_pose_episode,
    yam_base_transforms,
)
from dataset_filtering.retiming import time_parameterize
from dataset_filtering.smoothing import (
    benchmark_smoothers,
    detect_gripper_events,
    smooth_pose_episode,
    with_smoothed_pose,
)
from dataset_filtering.trajectory_optimization import (
    interarm_surface_clearance_profile_m,
    optimize_pose_episode,
)

METHOD_DEFAULTS: dict[str, dict[str, float | int]] = {
    "savgol": {"window": 15, "polyorder": 3},
    "gaussian": {"sigma_frames": 3.0},
    "butterworth": {"cutoff_hz": 4.0, "order": 3},
}


def create_app(
    source_name: str = DEFAULT_REPO_ID,
    *,
    revision: str = "main",
    cache_dir: str | None = None,
    precomputed_results: str | Path | None = None,
) -> FastAPI:
    source = LeRobotSource(source_name, revision=revision, cache_dir=cache_dir)
    saved_results = (
        Path(precomputed_results).expanduser().resolve() if precomputed_results else None
    )
    requested_revision = revision
    source_lock = threading.RLock()
    next_revision_check = 0.0
    app = FastAPI(title="UMI trajectory filtering and YAM IK viewer")

    def current_source() -> LeRobotSource:
        """Follow a mutable Hub ``main`` branch without sacrificing offline caches."""
        nonlocal source, next_revision_check
        with source_lock:
            if source.location.repo_id is None or requested_revision != "main":
                return source
            now = time.monotonic()
            if now < next_revision_check:
                return source
            next_revision_check = now + 10.0
            try:
                revision_sha = _hub_revision_sha(source.location.repo_id, requested_revision)
            except Exception:
                return source
            if revision_sha != source.location.revision:
                source = LeRobotSource(
                    source_name,
                    revision=revision_sha,
                    cache_dir=cache_dir,
                )
            return source

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        html = files("dataset_filtering.web").joinpath("index.html").read_text(encoding="utf-8")
        return HTMLResponse(html)

    @app.get("/api/episodes")
    def episodes() -> dict[str, Any]:
        try:
            active_source = current_source()
            summaries = active_source.episode_summaries()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {
            "source": active_source.source,
            "revision": active_source.location.revision,
            "read_only": True,
            "precomputed_results": saved_results is not None,
            "total_duration_s": sum(summary.duration_s for summary in summaries),
            "episodes": [
                {
                    "episode_index": summary.episode_index,
                    "length": summary.length,
                    "duration_s": summary.duration_s,
                    "task": summary.task,
                }
                for summary in summaries
            ],
        }

    @app.get("/api/episode/{episode_index}")
    def episode(
        episode_index: int,
        smoother: str = Query("auto", pattern="^(auto|savgol|gaussian|butterworth)$"),
        solver: str = Query("continuous", pattern="^(continuous|greedy)$"),
    ) -> dict[str, Any]:
        try:
            active_source = current_source()
            return _cached_payload(
                active_source.location.revision,
                episode_index,
                smoother,
                solver,
                _saved_result_token(saved_results, episode_index)
                if saved_results is not None and solver == "continuous"
                else None,
            )
        except SavedSolveUnavailable as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except IndexError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=500, detail=f"episode processing failed: {exc}"
            ) from exc

    @lru_cache(maxsize=6)
    def _cached_payload(
        source_revision: str,
        episode_index: int,
        smoother: str,
        solver: str,
        saved_result_token: str | None,
    ) -> dict[str, Any]:
        with source_lock:
            payload_source = source
            if payload_source.location.revision != source_revision:
                payload_source = LeRobotSource(
                    source_name,
                    revision=source_revision,
                    cache_dir=cache_dir,
                )
        episode_data = payload_source.load_episode(episode_index)
        if episode_data.kind is EpisodeKind.UMI_POSE:
            if saved_results is not None and solver == "continuous":
                return _saved_pose_payload(episode_data, saved_results, saved_result_token)
            return _pose_payload(episode_data, smoother, solver)
        return _joint_payload(episode_data)

    return app


class SavedSolveUnavailable(RuntimeError):
    """The requested gate result is unfinished or has no retained replay arrays."""


def _saved_result_token(results: Path, episode_index: int) -> str:
    row_path = results / f"input-{episode_index:03d}.json"
    array_path = row_path.with_suffix(".npz")
    if not row_path.is_file():
        raise SavedSolveUnavailable(
            f"episode {episode_index} is still processing; no completed Lambda solve exists yet"
        )
    row = json.loads(row_path.read_text(encoding="utf-8"))
    if not row.get("accepted"):
        profile = row.get("selected_profile", "unknown")
        raise SavedSolveUnavailable(
            f"episode {episode_index} completed with {profile}, but failed trajectories were not "
            "retained by this run"
        )
    if not array_path.is_file():
        raise SavedSolveUnavailable(
            f"episode {episode_index} passed, but its saved Lambda trajectory is missing"
        )
    return f"{row_path.stat().st_mtime_ns}:{array_path.stat().st_mtime_ns}"


def _saved_pose_payload(
    episode: Any,
    results: Path,
    saved_result_token: str | None,
) -> dict[str, Any]:
    if saved_result_token is None:
        raise SavedSolveUnavailable("no saved Lambda solve was selected")
    row_path = results / f"input-{episode.episode_index:03d}.json"
    row = json.loads(row_path.read_text(encoding="utf-8"))
    with np.load(row_path.with_suffix(".npz")) as artifact:
        q = np.asarray(artifact["q"], dtype=np.float64)

    result_row = row["selected_result"]
    config = result_row["config"]
    aligned_shared_frame = bool(config.get("aligned_shared_frame", False))
    original_pose = as_pose_episode(episode)
    pose = original_pose if aligned_shared_frame else zero_pose_origins(original_pose)
    if q.shape != (len(pose.timestamps), 2, 6):
        raise SavedSolveUnavailable(
            f"episode {episode.episode_index}: saved q shape {q.shape} does not match "
            f"{len(pose.timestamps)} source frames"
        )

    smoothing_row = row["selected_smoothing"]
    smoothing = smooth_pose_episode(
        pose,
        str(smoothing_row["method"]),
        dict(smoothing_row["parameters"]),
    )
    filtered = with_smoothed_pose(pose, smoothing)
    raw_positions, _ = map_umi_targets_to_yam_world(
        pose, aligned_shared_frame=aligned_shared_frame
    )
    target_positions, target_quaternions = map_umi_targets_to_yam_world(
        filtered, aligned_shared_frame=aligned_shared_frame
    )
    links, achieved_positions, achieved_quaternions = _forward_trajectory(q)
    position_error = np.linalg.norm(achieved_positions - target_positions, axis=-1)
    orientation_error = _orientation_error(achieved_quaternions, target_quaternions)

    timing = time_parameterize(
        q.reshape(len(q), 12),
        pose.timestamps,
        np.tile(
            YAM_VELOCITY_LIMITS_RAD_S * float(config["planning_velocity_scale"]),
            2,
        ),
        np.full(12, float(config["planning_acceleration_limit_rad_s2"])),
        prefer_toppra=bool(config["prefer_toppra"]),
    )
    events = detect_gripper_events(pose.grippers, fps=pose.fps)
    position_tolerance = float(config["tracking_position_tolerance_m"])
    orientation_tolerance = np.deg2rad(float(config["tracking_orientation_tolerance_deg"]))
    frame_valid = (position_error <= position_tolerance) & (
        orientation_error <= orientation_tolerance
    )
    strict_position = float(config["contact_position_tolerance_m"])
    strict_orientation = np.deg2rad(float(config["contact_orientation_tolerance_deg"]))
    strict_total = sum(len(indices) for indices in events)
    strict_feasible = sum(
        int(
            np.count_nonzero(
                (position_error[indices, arm] <= strict_position)
                & (orientation_error[indices, arm] <= strict_orientation)
            )
        )
        for arm, indices in enumerate(events)
    )
    tracking = {
        "both_target_fraction_10mm_8deg": float(np.mean(np.all(frame_valid, axis=1))),
        "individual_target_fraction_10mm_8deg": float(np.mean(frame_valid)),
        "flow_direction_cosine_median": result_row["flow_direction_cosine_median"],
    }
    retarget_metadata = {
        **result_row,
        "reachability": {
            "both_fraction": tracking["both_target_fraction_10mm_8deg"],
            "individual_fraction": tracking["individual_target_fraction_10mm_8deg"],
        },
        "tracking": tracking,
        "contact_events": {
            "total": strict_total,
            "strictly_feasible": strict_feasible,
            "umi1_left_frames": events[0].tolist(),
            "umi2_right_frames": events[1].tolist(),
        },
        "timing": {"method": timing.method, **timing.metadata},
    }
    grippers = pose.grippers if pose.grippers is not None else np.zeros((len(q), 2))
    return {
        "mode": "umi_pose_continuous",
        "solver": "continuous",
        "precomputed": True,
        "selected_profile": row["selected_profile"],
        "source": episode.source,
        "episode_index": episode.episode_index,
        "task": episode.task,
        "fps": episode.fps,
        "frames": len(q),
        "duration_s": float(timing.timestamps[-1] - timing.timestamps[0]),
        "source_duration_s": float(pose.timestamps[-1] - pose.timestamps[0]),
        "timestamps": _array(timing.timestamps),
        "source_timestamps": _array(pose.timestamps),
        "raw_positions": _array(raw_positions),
        "smoothed_positions": _array(target_positions),
        "smoothed_quaternions": _array(target_quaternions),
        "achieved_positions": _array(achieved_positions),
        "achieved_quaternions": _array(achieved_quaternions),
        "grippers": _array(grippers),
        "robot_links": _array(links),
        "interarm_clearance_mm": _array(1000 * interarm_surface_clearance_profile_m(links)),
        "interarm_clearance_gate_mm": 5.0,
        "robot_joints": _array(q),
        "reachable": frame_valid.tolist(),
        "position_error_mm": _array(1000 * position_error),
        "orientation_error_deg": _array(np.degrees(orientation_error)),
        "camera_position_error_mm": _array(1000 * position_error),
        "camera_orientation_error_deg": _array(np.degrees(orientation_error)),
        "video_alignment_weight": np.ones_like(position_error).tolist(),
        "gripper_events": [indices.tolist() for indices in events],
        "noise": pose_noise_metrics(pose)["aggregate"],
        "smoothing": smoothing_row,
        "retarget": retarget_metadata,
        "warnings": [
            *episode.warnings,
            "Loaded the saved Lambda quality-gate solve "
            f"({row['selected_profile']}); no IK re-solve.",
        ],
    }


def _forward_trajectory(q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kin = YAMKinematics()
    bases = yam_base_transforms(DEFAULT_PAIR_SEPARATION_M)
    links = np.empty((len(q), 2, 8, 3), dtype=np.float64)
    positions = np.empty((len(q), 2, 3), dtype=np.float64)
    quaternions = np.empty((len(q), 2, 4), dtype=np.float64)
    for arm in range(2):
        base_rotation = Rotation.from_matrix(bases[arm][:3, :3])
        for frame, joints in enumerate(q[:, arm]):
            fk = kin.forward(joints)
            links[frame, arm] = transform_points(fk.points, bases[arm])
            positions[frame, arm] = links[frame, arm, -1]
            world_rotation = base_rotation * Rotation.from_matrix(fk.rotation)
            if world_rotation is NotImplemented:
                raise TypeError("could not compose rotations")
            quaternions[frame, arm] = world_rotation.as_quat()
    return links, positions, quaternions


def _orientation_error(achieved: np.ndarray, targets: np.ndarray) -> np.ndarray:
    achieved_rotations = Rotation.from_quat(achieved.reshape(-1, 4))
    target_rotations = Rotation.from_quat(targets.reshape(-1, 4))
    relative = achieved_rotations.inv() * target_rotations
    if relative is NotImplemented:
        raise TypeError("could not compare rotations")
    return relative.magnitude().reshape(achieved.shape[:2])


def _hub_revision_sha(repo_id: str, revision: str) -> str:
    quoted_repo = urllib.parse.quote(repo_id, safe="/")
    quoted_revision = urllib.parse.quote(revision, safe="")
    url = f"https://huggingface.co/api/datasets/{quoted_repo}/revision/{quoted_revision}"
    request = urllib.request.Request(url, headers={"User-Agent": "umi-replay-viewer/0.1"})
    with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
        payload = json.load(response)
    sha = str(payload.get("sha", "")).strip()
    if not sha:
        raise RuntimeError(f"Hub returned no commit SHA for {repo_id}@{revision}")
    return sha


def _pose_payload(episode: Any, smoother: str, solver: str) -> dict[str, Any]:
    pose = as_pose_episode(episode)
    if smoother == "auto":
        smoothing = benchmark_smoothers(pose)[0]
    else:
        smoothing = smooth_pose_episode(pose, smoother, METHOD_DEFAULTS[smoother])
    filtered = with_smoothed_pose(pose, smoothing)
    if solver == "continuous":
        return _continuous_pose_payload(episode, pose, filtered, smoothing)

    retargeted = retarget_pose_episode(filtered, max_iterations=60)
    raw_targets, _ = map_umi_targets_to_yam_world(pose)
    links = playback_link_points(retargeted)
    grippers = pose.grippers if pose.grippers is not None else np.zeros((len(pose.timestamps), 2))
    return {
        "mode": "umi_pose_ik",
        "solver": "greedy",
        "source": episode.source,
        "episode_index": episode.episode_index,
        "task": episode.task,
        "fps": episode.fps,
        "frames": len(episode.timestamps),
        "duration_s": len(episode.timestamps) / episode.fps,
        "timestamps": _array(episode.timestamps),
        "raw_positions": _array(raw_targets),
        "smoothed_positions": _array(retargeted.target_positions_world),
        "smoothed_quaternions": _array(retargeted.target_quaternions_world),
        "grippers": _array(grippers),
        "robot_links": _array(links),
        "interarm_clearance_mm": _array(1000 * interarm_surface_clearance_profile_m(links)),
        "interarm_clearance_gate_mm": 5.0,
        "robot_joints": _array(np.stack([arm.playback_q for arm in retargeted.arms], axis=1)),
        "reachable": np.stack([arm.reachable for arm in retargeted.arms], axis=1).tolist(),
        "position_error_mm": _array(
            1000 * np.stack([arm.position_error_m for arm in retargeted.arms], axis=1)
        ),
        "orientation_error_deg": _array(
            np.degrees(np.stack([arm.orientation_error_rad for arm in retargeted.arms], axis=1))
        ),
        "gripper_events": [indices.tolist() for indices in retargeted.event_indices],
        "noise": pose_noise_metrics(pose)["aggregate"],
        "smoothing": {
            "method": smoothing.method,
            "parameters": smoothing.parameters,
            "score": smoothing.score,
            "metrics": smoothing.metrics,
        },
        "retarget": retargeted.metadata,
        "warnings": list(episode.warnings),
    }


def _continuous_pose_payload(
    episode: Any, pose: Any, filtered: Any, smoothing: Any
) -> dict[str, Any]:
    optimized = optimize_pose_episode(filtered)
    raw_targets, _ = map_umi_targets_to_yam_world(pose)
    grippers = pose.grippers if pose.grippers is not None else np.zeros((len(pose.timestamps), 2))
    tracking = optimized.metadata["tracking"]
    retarget_metadata = {
        **optimized.metadata,
        "reachability": {
            "both_fraction": tracking["both_target_fraction_10mm_8deg"],
            "individual_fraction": tracking["individual_target_fraction_10mm_8deg"],
        },
        "orientation_tolerance_deg": 8.0,
    }
    return {
        "mode": "umi_pose_continuous",
        "solver": "continuous",
        "source": episode.source,
        "episode_index": episode.episode_index,
        "task": episode.task,
        "fps": episode.fps,
        "frames": len(pose.timestamps),
        "duration_s": float(optimized.robot_timestamps[-1] - optimized.robot_timestamps[0]),
        "source_duration_s": len(pose.timestamps) / episode.fps,
        "timestamps": _array(optimized.robot_timestamps),
        "source_timestamps": _array(optimized.source_timestamps),
        "raw_positions": _array(raw_targets),
        "smoothed_positions": _array(optimized.target_positions_world),
        "smoothed_quaternions": _array(optimized.target_quaternions_world),
        "achieved_positions": _array(optimized.achieved_positions_world),
        "achieved_quaternions": _array(optimized.achieved_quaternions_world),
        "grippers": _array(grippers),
        "robot_links": _array(optimized.link_points_world),
        "interarm_clearance_mm": _array(
            1000 * interarm_surface_clearance_profile_m(optimized.link_points_world)
        ),
        "interarm_clearance_gate_mm": 5.0,
        "robot_joints": _array(optimized.q),
        "reachable": optimized.frame_valid.tolist(),
        "position_error_mm": _array(1000 * optimized.position_error_m),
        "orientation_error_deg": _array(np.degrees(optimized.orientation_error_rad)),
        "camera_position_error_mm": _array(1000 * optimized.camera_position_error_m),
        "camera_orientation_error_deg": _array(np.degrees(optimized.camera_orientation_error_rad)),
        "video_alignment_weight": _array(optimized.video_alignment_weight),
        "gripper_events": [indices.tolist() for indices in optimized.event_indices],
        "noise": pose_noise_metrics(pose)["aggregate"],
        "smoothing": {
            "method": smoothing.method,
            "parameters": smoothing.parameters,
            "score": smoothing.score,
            "metrics": smoothing.metrics,
        },
        "retarget": retarget_metadata,
        "warnings": list(episode.warnings),
    }


def _joint_payload(episode: Any) -> dict[str, Any]:
    kin = YAMKinematics()
    bases = yam_base_transforms(DEFAULT_PAIR_SEPARATION_M)
    q = np.stack((episode.state[:, :6], episode.state[:, 7:13]), axis=1)
    links = np.empty((len(q), 2, 8, 3), dtype=np.float64)
    positions = np.empty((len(q), 2, 3), dtype=np.float64)
    quaternions = np.empty((len(q), 2, 4), dtype=np.float64)
    reachable = np.ones((len(q), 2), dtype=bool)
    for arm in range(2):
        reachable[:, arm] = np.all((q[:, arm] >= kin.lower) & (q[:, arm] <= kin.upper), axis=1)
        base_rotation = Rotation.from_matrix(bases[arm][:3, :3])
        for frame, joints in enumerate(q[:, arm]):
            fk = kin.forward(np.clip(joints, kin.lower, kin.upper))
            links[frame, arm] = transform_points(fk.points, bases[arm])
            positions[frame, arm] = links[frame, arm, -1]
            world_rotation = base_rotation * Rotation.from_matrix(fk.rotation)
            if world_rotation is NotImplemented:
                raise TypeError("could not compose rotations")
            quaternions[frame, arm] = world_rotation.as_quat()
    grippers = episode.grippers if episode.grippers is not None else np.zeros((len(q), 2))
    return {
        "mode": "yam_joint_reference",
        "source": episode.source,
        "episode_index": episode.episode_index,
        "task": episode.task,
        "fps": episode.fps,
        "frames": len(episode.timestamps),
        "duration_s": len(episode.timestamps) / episode.fps,
        "timestamps": _array(episode.timestamps),
        "raw_positions": _array(positions),
        "smoothed_positions": _array(positions),
        "smoothed_quaternions": _array(quaternions),
        "grippers": _array(grippers),
        "robot_links": _array(links),
        "interarm_clearance_mm": _array(1000 * interarm_surface_clearance_profile_m(links)),
        "interarm_clearance_gate_mm": 5.0,
        "robot_joints": _array(q),
        "reachable": reachable.tolist(),
        "position_error_mm": np.zeros((len(q), 2)).tolist(),
        "orientation_error_deg": np.zeros((len(q), 2)).tolist(),
        "gripper_events": [[], []],
        "noise": vector_noise_metrics(episode),
        "smoothing": None,
        "retarget": {"reachability": {"both_fraction": float(np.mean(np.all(reachable, axis=1)))}},
        "warnings": list(episode.warnings),
    }


def _array(values: np.ndarray, decimals: int = 6) -> list[Any]:
    return np.round(np.asarray(values, dtype=np.float64), decimals=decimals).tolist()
