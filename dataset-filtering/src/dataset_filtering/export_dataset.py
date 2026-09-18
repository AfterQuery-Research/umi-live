from __future__ import annotations

import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from dataset_filtering.hf_dataset import DEFAULT_REPO_ID, LeRobotSource, as_pose_episode
from dataset_filtering.kinematics import YAMKinematics, transform_points
from dataset_filtering.retarget import (
    DEFAULT_PAIR_SEPARATION_M,
    UMI_TO_ROBOT_BASIS,
    yam_base_transforms,
)
from dataset_filtering.smoothing import benchmark_smoothers, with_smoothed_pose
from dataset_filtering.trajectory_optimization import optimize_pose_episode

DEFAULT_FILTER_REPORT = Path("reports/dual-lidar-umi-tail-freeze-v2-replayability.json")
DEFAULT_FILTERED_REPO_ID = "brandonyang/dual-lidar-umi-filtered"
STAT_NAMES = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
NUMERIC_FEATURES = (
    "observation.state",
    "observation.gripper_width.umi1",
    "observation.gripper_width.umi2",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)
VIDEO_FEATURES = ("observation.images.umi1", "observation.images.umi2")


@dataclass(frozen=True)
class ExportEpisode:
    output_episode_index: int
    source_episode_index: int
    frames: int
    source_file_index: int


def build_filtered_dataset(
    source: LeRobotSource,
    *,
    report_path: str | Path = DEFAULT_FILTER_REPORT,
    destination: str | Path,
    include_videos: bool = True,
    workers: int = 8,
) -> dict[str, Any]:
    """Build a local observation-only LeRobot-v3 dataset from complete replay passes."""
    report_file = Path(report_path).expanduser().resolve()
    report = json.loads(report_file.read_text(encoding="utf-8"))
    _validate_report(source, report)
    passing_rows = [row for row in report["per_episode"] if row["continuously_replayable"]]
    if not passing_rows:
        raise ValueError("filter report contains no continuously replayable episodes")

    output_root = Path(destination).expanduser().resolve()
    in_progress = output_root / ".build-in-progress"
    if output_root.exists() and any(output_root.iterdir()) and not in_progress.is_file():
        raise FileExistsError(f"destination is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    in_progress.touch()
    cache_root = output_root / ".build-cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    selected_source_indices = [int(row["episode_index"]) for row in passing_rows]
    source.prefetch_episode_data(selected_source_indices)
    source_meta = source.episode_metadata_frame().set_index("episode_index", drop=False)
    exports = [
        ExportEpisode(
            output_episode_index=output_index,
            source_episode_index=source_index,
            frames=int(source_meta.loc[source_index, "length"]),
            source_file_index=int(source_meta.loc[source_index, "data/file_index"]),
        )
        for output_index, source_index in enumerate(selected_source_indices)
    ]

    _make_layout(output_root)
    episode_stats: list[dict[str, dict[str, np.ndarray]]] = []
    metadata_rows: list[dict[str, Any]] = []
    optimizer_rows: list[dict[str, Any]] = []
    global_from_index = 0
    for item in exports:
        optimized_state, optimizer_metadata = _optimized_state_for_episode(
            source, item.source_episode_index, cache_root
        )
        if len(optimized_state) != item.frames:
            raise RuntimeError(
                f"episode {item.source_episode_index}: optimized rows {len(optimized_state)} "
                f"!= source rows {item.frames}"
            )
        data_path = source.materialize(f"data/chunk-000/file-{item.source_file_index:03d}.parquet")
        source_table = pq.read_table(data_path)
        episode_table = _rewrite_episode_table(
            source_table,
            optimized_state,
            output_episode_index=item.output_episode_index,
            global_from_index=global_from_index,
        )
        output_data_path = (
            output_root / "data/chunk-000" / f"file-{item.output_episode_index:03d}.parquet"
        )
        pq.write_table(episode_table, output_data_path, compression="zstd")

        numeric = _numeric_episode_data(episode_table)
        stats = {key: _feature_stats(value) for key, value in numeric.items()}
        source_episode_row = source_meta.loc[item.source_episode_index].to_dict()
        for video_key in VIDEO_FEATURES:
            stats[video_key] = _stats_from_metadata_row(source_episode_row, video_key)
        episode_stats.append(stats)
        metadata_rows.append(
            _episode_metadata_row(
                source_episode_row,
                item,
                stats,
                global_from_index=global_from_index,
            )
        )
        optimizer_rows.append(
            {
                "output_episode_index": item.output_episode_index,
                "source_episode_index": item.source_episode_index,
                "frames": item.frames,
                "profile": optimizer_metadata.get("profile", "baseline_v3"),
                "smoothing": optimizer_metadata["smoothing"],
                "retarget": optimizer_metadata["retarget"],
            }
        )
        global_from_index += item.frames
        print(
            f"filtered dataset: optimized {item.output_episode_index + 1}/{len(exports)} "
            f"(source episode {item.source_episode_index})",
            flush=True,
        )

    _write_episode_metadata(output_root, metadata_rows)
    global_stats = _aggregate_stats(episode_stats)
    _write_json(output_root / "meta/stats.json", _jsonable_stats(global_stats))
    source_info = source.info()
    output_info = _filtered_info(source_info, episodes=len(exports), frames=global_from_index)
    _write_json(output_root / "meta/info.json", output_info)
    shutil.copy2(source.materialize("meta/tasks.parquet"), output_root / "meta/tasks.parquet")
    _copy_optional_source_file(source, ".gitattributes", output_root / ".gitattributes")

    if include_videos:
        _materialize_videos(source, output_root, exports, workers=workers)

    provenance = {
        "source_dataset": source.source,
        "source_revision": source.location.revision,
        "selection_policy": (
            "per_episode.continuously_replayable == true, including strict all-frame "
            "10 mm/8 deg tracking after conditional per-arm tail freezing"
        ),
        "selection_report": str(report_file),
        "algorithm": report["algorithm"],
        "state_semantics": (
            "continuous optimizer achieved YAM FK pose mapped back into the original "
            "12-D UMI Cartesian observation.state convention"
        ),
        "action_removed": True,
        "episodes": len(exports),
        "frames": global_from_index,
        "episode_index_map": [
            {
                "output_episode_index": item.output_episode_index,
                "source_episode_index": item.source_episode_index,
            }
            for item in exports
        ],
        "optimizer_results": optimizer_rows,
    }
    _write_json(output_root / "meta/filtering.json", provenance)
    _write_dataset_card(output_root, provenance)
    shutil.rmtree(cache_root)
    in_progress.unlink()
    summary = validate_filtered_dataset(output_root)
    summary["destination"] = str(output_root)
    return summary


def validate_filtered_dataset(root: str | Path) -> dict[str, Any]:
    dataset_root = Path(root).expanduser().resolve()
    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    if "action" in info["features"]:
        raise ValueError("filtered info.json still contains action")
    expected_columns = [
        "observation.state",
        "observation.gripper_width.umi1",
        "observation.gripper_width.umi2",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    data_files = sorted((dataset_root / "data").rglob("*.parquet"))
    episode_files = sorted((dataset_root / "meta/episodes").rglob("*.parquet"))
    if len(data_files) != int(info["total_episodes"]):
        raise ValueError("data file count does not match info.total_episodes")
    metadata = pd.concat([pd.read_parquet(path) for path in episode_files], ignore_index=True)
    expected_episodes = list(range(int(info["total_episodes"])))
    if metadata["episode_index"].astype(int).tolist() != expected_episodes:
        raise ValueError("episode indices are not dense and ordered")
    total_rows = 0
    next_global_index = 0
    for episode_index, path in enumerate(data_files):
        table = pq.read_table(path)
        if table.column_names != expected_columns:
            raise ValueError(f"{path}: unexpected columns {table.column_names}")
        if table.schema.field("observation.state").type != pa.list_(pa.float32(), 12):
            raise ValueError(f"{path}: observation.state is not fixed float32[12]")
        rows = table.num_rows
        episode_values = table["episode_index"].to_numpy()
        global_values = table["index"].to_numpy()
        if not np.all(episode_values == episode_index):
            raise ValueError(f"{path}: wrong episode_index values")
        if not np.array_equal(
            global_values, np.arange(next_global_index, next_global_index + rows)
        ):
            raise ValueError(f"{path}: global index is not contiguous")
        next_global_index += rows
        total_rows += rows
    if total_rows != int(info["total_frames"]):
        raise ValueError("row count does not match info.total_frames")
    video_files = sorted((dataset_root / "videos").rglob("*.mp4"))
    expected_video_files = 2 * int(info["total_episodes"])
    if video_files and len(video_files) != expected_video_files:
        raise ValueError("video file count does not match two streams per episode")
    return {
        "episodes": int(info["total_episodes"]),
        "frames": total_rows,
        "data_files": len(data_files),
        "video_files": len(video_files),
        "features": list(info["features"]),
        "action_present": False,
    }


def push_filtered_dataset(
    root: str | Path,
    *,
    repo_id: str = DEFAULT_FILTERED_REPO_ID,
    private: bool = False,
) -> str:
    """Upload a validated local artifact. This is the only Hub-mutating entrypoint."""
    validate_filtered_dataset(root)
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the 'hub' extra before pushing") from exc
    dataset_root = Path(root).expanduser().resolve()
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(dataset_root),
        ignore_patterns=[".build-cache/**"],
        delete_patterns=["meta/**", "data/**", "videos/**", "README.md", ".gitattributes"],
        commit_message="Upload smoothed, continuously replayable observation-only dataset",
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def optimized_world_poses_to_umi_state(
    achieved_positions_world: np.ndarray,
    achieved_quaternions_world: np.ndarray,
    original_positions: np.ndarray,
    original_quaternions: np.ndarray,
    *,
    separation_m: float = DEFAULT_PAIR_SEPARATION_M,
    aligned_shared_frame: bool = False,
) -> np.ndarray:
    """Invert UMI-to-YAM anchoring while retaining the source pose frame."""
    frames = len(achieved_positions_world)
    state = np.empty((frames, 12), dtype=np.float64)
    kin = YAMKinematics()
    home = kin.forward(kin.home)
    bases = yam_base_transforms(separation_m)
    if aligned_shared_frame:
        mapped_original = original_positions @ UMI_TO_ROBOT_BASIS.T
        home_positions = np.stack(
            [transform_points(home.position[None, :], base)[0] for base in bases]
        )
        translation = np.mean(home_positions, axis=0) - np.mean(
            mapped_original[0], axis=0
        )
        mapped_positions = achieved_positions_world - translation[None, None, :]
        positions = mapped_positions @ UMI_TO_ROBOT_BASIS
        for arm in range(2):
            base_rotation = Rotation.from_matrix(bases[arm][:3, :3])
            world_rotation = Rotation.from_quat(achieved_quaternions_world[:, arm])
            local_rotation = base_rotation.inv() * world_rotation
            if local_rotation is NotImplemented:
                raise TypeError("could not remove YAM base rotation")
            mapped_source = local_rotation * Rotation.from_matrix(home.rotation).inv()
            if mapped_source is NotImplemented:
                raise TypeError("could not remove YAM home rotation")
            source_matrices = np.einsum(
                "ij,njk,kl->nil",
                UMI_TO_ROBOT_BASIS.T,
                mapped_source.as_matrix(),
                UMI_TO_ROBOT_BASIS,
            )
            offset = 0 if arm == 0 else 6
            state[:, offset : offset + 3] = positions[:, arm]
            state[:, offset + 3 : offset + 6] = Rotation.from_matrix(
                source_matrices
            ).as_rotvec()
        return state.astype(np.float32)

    for arm in range(2):
        home_world = transform_points(home.position[None, :], bases[arm])[0]
        robot_relative = achieved_positions_world[:, arm] - home_world
        umi_relative = np.column_stack(
            (robot_relative[:, 2], -robot_relative[:, 1], robot_relative[:, 0])
        )
        positions = original_positions[0, arm] + umi_relative

        base_rotation = Rotation.from_matrix(bases[arm][:3, :3])
        world_rotation = Rotation.from_quat(achieved_quaternions_world[:, arm])
        local_rotation = base_rotation.inv() * world_rotation
        if local_rotation is NotImplemented:
            raise TypeError("could not remove YAM base rotation")
        # Forward retargeting applies the mapped world delta before the home grasp
        # orientation. Remove home on the right to recover that same world delta.
        mapped_relative = local_rotation * Rotation.from_matrix(home.rotation).inv()
        if mapped_relative is NotImplemented:
            raise TypeError("could not remove YAM home rotation")
        umi_relative_matrices = np.einsum(
            "ij,njk,kl->nil",
            UMI_TO_ROBOT_BASIS.T,
            mapped_relative.as_matrix(),
            UMI_TO_ROBOT_BASIS,
        )
        output_rotation = Rotation.from_quat(original_quaternions[0, arm]) * Rotation.from_matrix(
            umi_relative_matrices
        )
        if output_rotation is NotImplemented:
            raise TypeError("could not restore UMI episode orientation origin")
        offset = 0 if arm == 0 else 6
        state[:, offset : offset + 3] = positions
        state[:, offset + 3 : offset + 6] = output_rotation.as_rotvec()
    return state.astype(np.float32)


def _optimized_state_for_episode(
    source: LeRobotSource, episode_index: int, cache_root: Path
) -> tuple[np.ndarray, dict[str, Any]]:
    cache_path = cache_root / f"source-episode-{episode_index:03d}.npz"
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            return cached["state"], json.loads(str(cached["metadata_json"]))
    episode = as_pose_episode(source.load_episode(episode_index))
    smoothing = benchmark_smoothers(episode)[0]
    filtered = with_smoothed_pose(episode, smoothing)
    result = optimize_pose_episode(filtered)
    if not result.metadata["continuously_replayable"]:
        raise RuntimeError(
            f"source episode {episode_index} no longer passes continuously_replayable "
            "when recomputed; refusing to export a stale report selection"
        )
    state = optimized_world_poses_to_umi_state(
        result.achieved_positions_world,
        result.achieved_quaternions_world,
        episode.positions,
        episode.quaternions,
    )
    metadata = {
        "smoothing": {
            "method": smoothing.method,
            "parameters": smoothing.parameters,
            "metrics": smoothing.metrics,
        },
        "retarget": result.metadata,
    }
    np.savez_compressed(cache_path, state=state, metadata_json=json.dumps(metadata))
    return state, metadata


def _rewrite_episode_table(
    source_table: pa.Table,
    optimized_state: np.ndarray,
    *,
    output_episode_index: int,
    global_from_index: int,
) -> pa.Table:
    if len(source_table) != len(optimized_state):
        raise ValueError("source and optimized episode lengths differ")
    columns: dict[str, pa.Array | pa.ChunkedArray] = {}
    flat_state = pa.array(optimized_state.reshape(-1), type=pa.float32())
    columns["observation.state"] = pa.FixedSizeListArray.from_arrays(flat_state, 12)
    for name in source_table.column_names:
        if name in {"observation.state", "action", "episode_index", "index"}:
            continue
        columns[name] = source_table[name]
    columns["episode_index"] = pa.array(
        np.full(len(source_table), output_episode_index, dtype=np.int64)
    )
    columns["index"] = pa.array(
        np.arange(global_from_index, global_from_index + len(source_table), dtype=np.int64)
    )
    ordered = [
        "observation.state",
        "observation.gripper_width.umi1",
        "observation.gripper_width.umi2",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    table = pa.table({name: columns[name] for name in ordered})
    hf_metadata = {
        "info": {
            "features": {
                "observation.state": {
                    "feature": {"dtype": "float32", "_type": "Value"},
                    "length": 12,
                    "_type": "List",
                },
                "observation.gripper_width.umi1": {"dtype": "float32", "_type": "Value"},
                "observation.gripper_width.umi2": {"dtype": "float32", "_type": "Value"},
                "timestamp": {"dtype": "float32", "_type": "Value"},
                "frame_index": {"dtype": "int64", "_type": "Value"},
                "episode_index": {"dtype": "int64", "_type": "Value"},
                "index": {"dtype": "int64", "_type": "Value"},
                "task_index": {"dtype": "int64", "_type": "Value"},
            }
        },
        "fingerprint": hashlib.sha256(optimized_state.tobytes()).hexdigest()[:16],
    }
    return table.replace_schema_metadata(
        {b"huggingface": json.dumps(hf_metadata, separators=(",", ":")).encode()}
    )


def _numeric_episode_data(table: pa.Table) -> dict[str, np.ndarray]:
    state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    result = {"observation.state": state}
    for key in NUMERIC_FEATURES[1:]:
        values = table[key].to_numpy(zero_copy_only=False)
        result[key] = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    return result


def _feature_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    return {
        "min": np.min(values, axis=0),
        "max": np.max(values, axis=0),
        "mean": np.mean(values, axis=0),
        "std": np.std(values, axis=0),
        "count": np.asarray([len(values)], dtype=np.int64),
        "q01": np.quantile(values, 0.01, axis=0),
        "q10": np.quantile(values, 0.10, axis=0),
        "q50": np.quantile(values, 0.50, axis=0),
        "q90": np.quantile(values, 0.90, axis=0),
        "q99": np.quantile(values, 0.99, axis=0),
    }


def _stats_from_metadata_row(row: dict[str, Any], feature: str) -> dict[str, np.ndarray]:
    return {
        stat: np.asarray(_nested_python_values(row[f"stats/{feature}/{stat}"]))
        for stat in STAT_NAMES
    }


def _nested_python_values(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _nested_python_values(value.tolist())
    if isinstance(value, list):
        return [_nested_python_values(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _aggregate_stats(
    stats_list: list[dict[str, dict[str, np.ndarray]]],
) -> dict[str, dict[str, np.ndarray]]:
    result: dict[str, dict[str, np.ndarray]] = {}
    for feature in stats_list[0]:
        feature_stats = [stats[feature] for stats in stats_list]
        counts = np.asarray([float(stats["count"][0]) for stats in feature_stats])
        total_count = float(np.sum(counts))
        means = np.stack([np.asarray(stats["mean"], dtype=np.float64) for stats in feature_stats])
        variances = np.stack(
            [np.asarray(stats["std"], dtype=np.float64) ** 2 for stats in feature_stats]
        )
        weight_shape = (len(counts),) + (1,) * (means.ndim - 1)
        weights = counts.reshape(weight_shape)
        total_mean = np.sum(means * weights, axis=0) / total_count
        total_variance = (
            np.sum((variances + (means - total_mean) ** 2) * weights, axis=0) / total_count
        )
        aggregate = {
            "min": np.min(np.stack([stats["min"] for stats in feature_stats]), axis=0),
            "max": np.max(np.stack([stats["max"] for stats in feature_stats]), axis=0),
            "mean": total_mean,
            "std": np.sqrt(total_variance),
            "count": np.asarray([int(total_count)], dtype=np.int64),
        }
        for quantile in ("q01", "q10", "q50", "q90", "q99"):
            values = np.stack([np.asarray(stats[quantile]) for stats in feature_stats])
            aggregate[quantile] = np.sum(values * weights, axis=0) / total_count
        result[feature] = aggregate
    return result


def _episode_metadata_row(
    source_row: dict[str, Any],
    item: ExportEpisode,
    stats: dict[str, dict[str, np.ndarray]],
    *,
    global_from_index: int,
) -> dict[str, Any]:
    row = {
        key: value
        for key, value in source_row.items()
        if not key.startswith("stats/") and key != "episode_index"
    }
    row.update(
        {
            "episode_index": item.output_episode_index,
            "length": item.frames,
            "data/chunk_index": 0,
            "data/file_index": item.output_episode_index,
            "dataset_from_index": global_from_index,
            "dataset_to_index": global_from_index + item.frames,
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": item.output_episode_index,
        }
    )
    for video_key in VIDEO_FEATURES:
        row[f"videos/{video_key}/chunk_index"] = 0
        row[f"videos/{video_key}/file_index"] = item.output_episode_index
    for feature, feature_stats in stats.items():
        for stat, value in feature_stats.items():
            row[f"stats/{feature}/{stat}"] = value
    return row


def _write_episode_metadata(root: Path, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        episode_index = int(row["episode_index"])
        parquet_row = {key: _nested_python_values(value) for key, value in row.items()}
        frame = pd.DataFrame([parquet_row])
        path = root / "meta/episodes/chunk-000" / f"file-{episode_index:03d}.parquet"
        frame.to_parquet(path, index=False)


def _filtered_info(source_info: dict[str, Any], *, episodes: int, frames: int) -> dict[str, Any]:
    info = json.loads(json.dumps(source_info))
    info["features"].pop("action", None)
    info["total_episodes"] = episodes
    info["total_frames"] = frames
    info["splits"] = {"train": f"0:{episodes}"}
    return info


def _materialize_videos(
    source: LeRobotSource,
    output_root: Path,
    exports: list[ExportEpisode],
    *,
    workers: int,
) -> None:
    jobs = [(item, video_key) for item in exports for video_key in VIDEO_FEATURES]

    def copy(job: tuple[ExportEpisode, str]) -> None:
        item, video_key = job
        relative = f"videos/{video_key}/chunk-000/file-{item.source_file_index:03d}.mp4"
        source_path = source.materialize(relative)
        destination = (
            output_root
            / "videos"
            / video_key
            / "chunk-000"
            / f"file-{item.output_episode_index:03d}.mp4"
        )
        if destination.is_file() and destination.stat().st_size == source_path.stat().st_size:
            return
        try:
            os.link(source_path, destination)
        except OSError:
            shutil.copy2(source_path, destination)

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as pool:
        list(pool.map(copy, jobs))


def _make_layout(root: Path) -> None:
    for relative in (
        "data/chunk-000",
        "meta/episodes/chunk-000",
        "videos/observation.images.umi1/chunk-000",
        "videos/observation.images.umi2/chunk-000",
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)


def _validate_report(source: LeRobotSource, report: dict[str, Any]) -> None:
    if report.get("source") not in {source.source, DEFAULT_REPO_ID}:
        raise ValueError(f"report source {report.get('source')!r} does not match {source.source!r}")
    supported_algorithms = {
        "contact_anchored_bimanual_cubic_bspline_nlp_with_tail_freeze_v2",
        "contact_anchored_bimanual_cubic_bspline_nlp_with_tail_freeze_v3",
    }
    if report.get("algorithm") not in supported_algorithms:
        raise ValueError("unsupported filtering report algorithm")
    invalid_passes = [
        int(row["episode_index"])
        for row in report.get("per_episode", [])
        if row.get("continuously_replayable")
        and row.get("both_valid_frames") != row.get("frames")
    ]
    if invalid_passes:
        raise ValueError(
            "filter report marks episodes continuously replayable despite failed tracking "
            f"frames: {invalid_passes}"
        )


def _jsonable_stats(stats: dict[str, dict[str, np.ndarray]]) -> dict[str, Any]:
    return {
        feature: {stat: np.asarray(value).tolist() for stat, value in values.items()}
        for feature, values in stats.items()
    }


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _copy_optional_source_file(source: LeRobotSource, relative: str, destination: Path) -> None:
    try:
        shutil.copy2(source.materialize(relative), destination)
    except (FileNotFoundError, RuntimeError):
        return


def _write_dataset_card(root: Path, provenance: dict[str, Any]) -> None:
    source = provenance["source_dataset"]
    root.joinpath("README.md").write_text(
        "---\n"
        "license: apache-2.0\n"
        "task_categories:\n"
        "- robotics\n"
        "tags:\n"
        "- LeRobot\n"
        "configs:\n"
        "- config_name: default\n"
        "  data_files: data/*/*.parquet\n"
        "---\n\n"
        "# Filtered dual-LiDAR UMI demonstrations\n\n"
        f"Observation-only LeRobot v3 derivative of [{source}]"
        f"(https://huggingface.co/datasets/{source}). It contains "
        f"{provenance['episodes']} demonstrations ({provenance['frames']} frames) that pass "
        "the complete continuous YAM replayability classification.\n\n"
        "`observation.state` retains the original 12-D UMI Cartesian schema. Its values are "
        "the smoothed, trajectory-optimized YAM-achievable FK path mapped back into the UMI "
        "coordinate convention. Gripper observations, videos, timestamps, and tasks are "
        "preserved on the source-frame timeline. The `action` feature is intentionally absent.\n\n"
        "See `meta/filtering.json` for source episode mapping and optimizer metadata. The "
        "collision test is a sampled-capsule proxy; exact scene validation remains required "
        "before robot playback.\n",
        encoding="utf-8",
    )
