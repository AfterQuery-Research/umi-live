from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from dataset_filtering.hf_dataset import LeRobotSource, as_pose_episode
from dataset_filtering.kinematics import YAMKinematics
from dataset_filtering.retarget import retarget_pose_episode
from dataset_filtering.smoothing import smooth_pose_episode, with_smoothed_pose

from dataset_filtering_tests.baseline import BASELINE_REPORT, load_baseline

HOME_CANDIDATES: dict[str, np.ndarray] = {
    "zero": np.zeros(6, dtype=np.float64),
    "balanced_0p6": np.asarray([0.0, 0.6, 0.6, 0.0, 0.0, 0.0]),
    "balanced_1p2": np.asarray([0.0, 1.2, 1.2, 0.0, 0.0, 0.0]),
    "balanced_pi_over_2": np.asarray([0.0, np.pi / 2, np.pi / 2, 0.0, 0.0, 0.0]),
    "balanced_1p8": np.asarray([0.0, 1.8, 1.8, 0.0, 0.0, 0.0]),
}


def screen_global_home_postures(
    *,
    episode_indices: list[int],
    source_name: str = "brandonyang/dual-lidar-umi-relative",
    baseline_report: str | Path = BASELINE_REPORT,
    workers: int = 8,
) -> dict[str, Any]:
    baseline = load_baseline(baseline_report)
    rows_by_index = {int(row["episode_index"]): row for row in baseline["per_episode"]}
    unique_indices = sorted(set(episode_indices))
    missing = sorted(set(unique_indices) - rows_by_index.keys())
    if missing:
        raise ValueError(f"episodes are missing from the baseline: {missing}")

    source = LeRobotSource(source_name)
    source.prefetch_episode_data(unique_indices)
    arguments = [
        (
            source_name,
            source.location.revision,
            str(source.cache_dir),
            episode_index,
            rows_by_index[episode_index]["smoothing"]["method"],
            rows_by_index[episode_index]["smoothing"]["parameters"],
        )
        for episode_index in unique_indices
    ]
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(arguments)))) as executor:
        futures = [executor.submit(_screen_episode, item) for item in arguments]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            print(f"home-posture screen: {completed}/{len(futures)} episodes", flush=True)
    rows.sort(key=lambda row: row["episode_index"])

    aggregates = []
    for name in HOME_CANDIDATES:
        values = [row["candidates"][name] for row in rows]
        aggregates.append(
            {
                "home": name,
                "q_rad": HOME_CANDIDATES[name].tolist(),
                "episodes": len(values),
                "all_frames_both_ik_reachable": sum(
                    value["all_frames_both_ik_reachable"] for value in values
                ),
                "all_strict_events_ik_reachable": sum(
                    value["all_strict_events_ik_reachable"] for value in values
                ),
                "median_both_frame_fraction": float(
                    np.median([value["both_frame_fraction"] for value in values])
                ),
            }
        )
    kin = YAMKinematics()
    home_geometry = {}
    reference_rotation = kin.forward(HOME_CANDIDATES["zero"]).rotation
    for name, q in HOME_CANDIDATES.items():
        forward = kin.forward(q)
        home_geometry[name] = {
            "tcp_position_local_m": forward.position.tolist(),
            "orientation_delta_from_zero_deg": float(
                np.degrees(
                    np.arccos(
                        np.clip((np.trace(reference_rotation.T @ forward.rotation) - 1) / 2, -1, 1)
                    )
                )
            ),
            "within_joint_limits": bool(np.all(q >= kin.lower) and np.all(q <= kin.upper)),
        }
    return {
        "experiment": "global_yam_home_registration_framewise_ik_screen",
        "source": source_name,
        "baseline_report": str(Path(baseline_report).expanduser().resolve()),
        "episodes": unique_indices,
        "gate_semantics_changed": False,
        "registration_semantics": (
            "The UMI relative displacement is unchanged; source zero is anchored to a "
            "different collision-free YAM joint posture with the same nominal TCP orientation."
        ),
        "hardware_review_required": True,
        "warning": "Framewise IK is a proxy; use one global home in the complete pipeline.",
        "home_geometry": home_geometry,
        "aggregates": aggregates,
        "per_episode": rows,
    }


def _screen_episode(
    arguments: tuple[str, str, str, int, str, dict[str, Any]],
) -> dict[str, Any]:
    source_name, revision, cache_dir, episode_index, method, parameters = arguments
    source = LeRobotSource(source_name, revision=revision, cache_dir=cache_dir)
    episode = as_pose_episode(source.load_episode(episode_index))
    filtered = with_smoothed_pose(episode, smooth_pose_episode(episode, method, parameters))
    production_init = YAMKinematics.__init__
    candidates: dict[str, Any] = {}
    try:
        for name, home in HOME_CANDIDATES.items():
            YAMKinematics.__init__ = _home_initializer(production_init, home)
            result = retarget_pose_episode(
                filtered,
                position_tolerance_m=0.01,
                contact_position_tolerance_m=0.001,
                orientation_tolerance_rad=np.deg2rad(8.0),
                max_iterations=75,
            )
            both = result.arms[0].reachable & result.arms[1].reachable
            strict_events = []
            for arm in range(2):
                events = result.event_indices[arm]
                strict_events.append(
                    (result.arms[arm].position_error_m[events] <= 0.001)
                    & (result.arms[arm].orientation_error_rad[events] <= np.deg2rad(2.0))
                )
            candidates[name] = {
                "both_frame_fraction": float(np.mean(both)),
                "all_frames_both_ik_reachable": bool(np.all(both)),
                "all_strict_events_ik_reachable": bool(
                    all(np.all(values) for values in strict_events)
                ),
            }
    finally:
        YAMKinematics.__init__ = production_init
    return {"episode_index": episode_index, "candidates": candidates}


def _home_initializer(production_init: Any, home: np.ndarray) -> Any:
    def initialize(instance: YAMKinematics) -> None:
        production_init(instance)
        instance.home = home.copy()

    return initialize
