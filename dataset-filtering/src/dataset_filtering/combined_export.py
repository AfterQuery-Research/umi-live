from __future__ import annotations

import hashlib
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from dataset_filtering.combined_pipeline import (
    KINEMATICS_PROVENANCE,
    LEFT_GRIPPER_WIDTH_LIMITS_MM,
    PIPELINE_VERSION,
    RIGHT_GRIPPER_WIDTH_LIMITS_MM,
    joint_observation_state,
)
from dataset_filtering.export_dataset import (
    VIDEO_FEATURES,
    ExportEpisode,
    _aggregate_stats,
    _episode_metadata_row,
    _feature_stats,
    _jsonable_stats,
    _make_layout,
    _stats_from_metadata_row,
    _write_episode_metadata,
    _write_json,
)
from dataset_filtering.gripper_calibration import load_single_gripper_calibration
from dataset_filtering.hf_dataset import LeRobotSource

CARTESIAN_REPO_ID = "brandonyang/dual-lidar-combined-filtered-long-gripper"
JOINT_REPO_ID = "brandonyang/dual-lidar-combined-filtered-joint-positions-long-gripper"
CARTESIAN_STATE_NAMES = (
    "umi1_x",
    "umi1_y",
    "umi1_z",
    "umi1_rx",
    "umi1_ry",
    "umi1_rz",
    "umi1_gripper.pos",
    "umi2_x",
    "umi2_y",
    "umi2_z",
    "umi2_rx",
    "umi2_ry",
    "umi2_rz",
    "umi2_gripper.pos",
)
YAM_STATE_NAMES = (
    "left_joint_0.pos",
    "left_joint_1.pos",
    "left_joint_2.pos",
    "left_joint_3.pos",
    "left_joint_4.pos",
    "left_joint_5.pos",
    "left_gripper.pos",
    "right_joint_0.pos",
    "right_joint_1.pos",
    "right_joint_2.pos",
    "right_joint_3.pos",
    "right_joint_4.pos",
    "right_joint_5.pos",
    "right_gripper.pos",
)

Variant = Literal["cartesian", "joint"]
CARTESIAN_POSE_INDICES = (0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12)
GRIPPER_STATE_INDICES = (6, 13)


@dataclass(frozen=True)
class CombinedExportEpisode:
    output_episode_index: int
    combined_input_index: int
    source_dataset: str
    source_revision: str
    source_episode_index: int
    frames: int
    source_data_chunk: int
    source_data_file: int
    source_video_chunks: tuple[int, int]
    source_video_files: tuple[int, int]
    result_array: Path


def build_combined_datasets(
    *,
    report_path: str | Path,
    cartesian_destination: str | Path,
    joint_destination: str | Path,
    umi1_gripper_calibration: str | Path | None = None,
    umi2_gripper_calibration: str | Path | None = None,
    workers: int = 12,
) -> dict[str, Any]:
    """Build matching Cartesian and YAM-joint observation-only LeRobot datasets."""
    report_file = Path(report_path).expanduser().resolve()
    report = json.loads(report_file.read_text(encoding="utf-8"))
    _validate_pipeline_report(report)
    accepted_rows = [row for row in report["per_episode"] if row["accepted"]]
    if not accepted_rows:
        raise ValueError("pipeline report contains no accepted episodes")

    roots = {
        "cartesian": Path(cartesian_destination).expanduser().resolve(),
        "joint": Path(joint_destination).expanduser().resolve(),
    }
    for root in roots.values():
        _prepare_output_root(root)

    sources: dict[tuple[str, str], LeRobotSource] = {}
    source_meta: dict[tuple[str, str], pd.DataFrame] = {}
    for source_spec in report["sources"]:
        key = (str(source_spec["repo_id"]), str(source_spec["revision"]))
        source = LeRobotSource(key[0], revision=key[1])
        sources[key] = source
        source_meta[key] = source.episode_metadata_frame().set_index("episode_index", drop=False)

    exports: list[CombinedExportEpisode] = []
    episodes_dir = report_file.parent / "episodes"
    for output_index, row in enumerate(accepted_rows):
        key = (str(row["source_dataset"]), str(row["source_revision"]))
        if key not in source_meta:
            raise ValueError(f"accepted row references unknown source {key}")
        source_episode_index = int(row["source_episode_index"])
        metadata = source_meta[key].loc[source_episode_index]
        exports.append(
            CombinedExportEpisode(
                output_episode_index=output_index,
                combined_input_index=int(row["combined_input_index"]),
                source_dataset=key[0],
                source_revision=key[1],
                source_episode_index=source_episode_index,
                frames=int(metadata["length"]),
                source_data_chunk=int(metadata["data/chunk_index"]),
                source_data_file=int(metadata["data/file_index"]),
                source_video_chunks=tuple(
                    int(metadata[f"videos/{feature}/chunk_index"]) for feature in VIDEO_FEATURES
                ),
                source_video_files=tuple(
                    int(metadata[f"videos/{feature}/file_index"]) for feature in VIDEO_FEATURES
                ),
                result_array=episodes_dir / f"input-{int(row['combined_input_index']):03d}.npz",
            )
        )

    gripper_width_limits_mm, gripper_calibration_provenance = _resolve_gripper_calibrations(
        umi1_gripper_calibration, umi2_gripper_calibration
    )
    gripper_width_diagnostics = _accepted_gripper_width_diagnostics(
        sources, exports, gripper_width_limits_mm
    )
    stats_by_variant: dict[Variant, list[dict[str, dict[str, np.ndarray]]]] = {
        "cartesian": [],
        "joint": [],
    }
    numeric_values_by_variant: dict[Variant, dict[str, list[np.ndarray]]] = {
        "cartesian": {},
        "joint": {},
    }
    metadata_by_variant: dict[Variant, list[dict[str, Any]]] = {
        "cartesian": [],
        "joint": [],
    }
    global_from_index = 0
    for item in exports:
        source = sources[(item.source_dataset, item.source_revision)]
        metadata = (
            source_meta[(item.source_dataset, item.source_revision)]
            .loc[item.source_episode_index]
            .to_dict()
        )
        source_table = _source_episode_table(source, item)
        arrays = _load_result_arrays(item)
        normalized_grippers = _normalize_gripper_widths(
            _raw_gripper_widths(source_table), gripper_width_limits_mm
        )
        states = {
            "cartesian": _cartesian_observation_state(
                arrays["cartesian_state"], normalized_grippers
            ),
            "joint": joint_observation_state(arrays["q"], normalized_grippers),
        }
        if np.max(np.abs(states["cartesian"][0, CARTESIAN_POSE_INDICES])) > 1e-4:
            raise ValueError(
                f"combined input {item.combined_input_index}: optimized Cartesian first "
                "pose does not start at zero"
            )
        states["cartesian"][0, CARTESIAN_POSE_INDICES] = 0.0
        if np.max(np.abs(arrays["q"][0])) > 1e-4:
            raise ValueError(
                f"combined input {item.combined_input_index}: optimized YAM joints do not "
                "start at home"
            )
        states["joint"][0, :6] = 0.0
        states["joint"][0, 7:13] = 0.0

        for variant in ("cartesian", "joint"):
            table = _rewrite_combined_episode_table(
                source_table,
                states[variant],
                variant=variant,
                output_episode_index=item.output_episode_index,
                global_from_index=global_from_index,
            )
            path = (
                roots[variant] / "data/chunk-000" / f"file-{item.output_episode_index:03d}.parquet"
            )
            pq.write_table(table, path, compression="zstd")
            numeric = _numeric_table_data(table)
            episode_stats = {feature: _feature_stats(values) for feature, values in numeric.items()}
            for feature, values in numeric.items():
                numeric_values_by_variant[variant].setdefault(feature, []).append(values)
            for video_feature in VIDEO_FEATURES:
                episode_stats[video_feature] = _stats_from_metadata_row(metadata, video_feature)
            stats_by_variant[variant].append(episode_stats)
            metadata_by_variant[variant].append(
                _episode_metadata_row(
                    metadata,
                    ExportEpisode(
                        output_episode_index=item.output_episode_index,
                        source_episode_index=item.source_episode_index,
                        frames=item.frames,
                        source_file_index=item.source_data_file,
                    ),
                    episode_stats,
                    global_from_index=global_from_index,
                )
            )
        global_from_index += item.frames
        print(
            f"combined export: {item.output_episode_index + 1}/{len(exports)} "
            f"(combined input {item.combined_input_index})",
            flush=True,
        )

    first_source = next(iter(sources.values()))
    first_info = first_source.info()
    first_tasks = pq.read_table(first_source.materialize("meta/tasks.parquet"))
    for source in sources.values():
        if not pq.read_table(source.materialize("meta/tasks.parquet")).equals(first_tasks):
            raise ValueError("source task tables differ; cannot preserve one task index")
    for variant in ("cartesian", "joint"):
        root = roots[variant]
        _write_episode_metadata(root, metadata_by_variant[variant])
        global_stats = _aggregate_stats(stats_by_variant[variant])
        global_stats.update(_exact_numeric_stats(numeric_values_by_variant[variant]))
        _write_json(
            root / "meta/stats.json",
            _jsonable_stats(global_stats),
        )
        _write_json(
            root / "meta/info.json",
            _combined_info(
                first_info,
                variant=variant,
                episodes=len(exports),
                frames=global_from_index,
            ),
        )
        pq.write_table(first_tasks, root / "meta/tasks.parquet")

    _materialize_combined_videos(sources, roots, exports, workers=workers)
    provenance_common = {
        "pipeline_version": report["pipeline_version"],
        "kinematics": report["kinematics"],
        "sources": report["sources"],
        "zero_origin": report["zero_origin"],
        "gate_semantics_changed": report["gate_semantics_changed"],
        "gates": report["gates"],
        "gripper_normalization": _gripper_normalization_metadata(
            gripper_width_limits_mm,
            gripper_width_diagnostics,
            gripper_calibration_provenance,
        ),
        "action_removed": True,
        "episodes": len(exports),
        "frames": global_from_index,
        "episode_index_map": [
            {
                "output_episode_index": item.output_episode_index,
                "combined_input_index": item.combined_input_index,
                "source_dataset": item.source_dataset,
                "source_revision": item.source_revision,
                "source_episode_index": item.source_episode_index,
            }
            for item in exports
        ],
        "optimizer_results": accepted_rows,
    }
    for variant in ("cartesian", "joint"):
        provenance = {
            **provenance_common,
            "state_semantics": (
                "smoothed and continuously optimized YAM-achievable FK poses mapped to "
                "14-D [UMI1 zero-origin pose, normalized UMI1 gripper, UMI2 "
                "zero-origin pose, normalized UMI2 gripper] observations"
                if variant == "cartesian"
                else "14-D [left YAM 6 joints, normalized left gripper, right YAM 6 "
                "joints, normalized right gripper] observations"
            ),
        }
        _write_json(roots[variant] / "meta/filtering.json", provenance)
        _write_dataset_card(roots[variant], provenance, variant)
        marker = roots[variant] / ".build-in-progress"
        if marker.exists():
            marker.unlink()

    summaries = {
        variant: validate_combined_dataset(root, variant=variant) for variant, root in roots.items()
    }
    _validate_matching_outputs(roots["cartesian"], roots["joint"])
    return summaries


def _raw_gripper_widths(source: pa.Table) -> np.ndarray:
    columns = ("observation.gripper_width.umi1", "observation.gripper_width.umi2")
    if any(column not in source.column_names for column in columns):
        raise ValueError("source episode is missing one or both UMI gripper-width observations")
    widths = np.column_stack(
        [
            np.asarray(source[column].to_numpy(zero_copy_only=False), dtype=np.float64)
            for column in columns
        ]
    )
    if widths.ndim != 2 or widths.shape[1] != 2 or not np.all(np.isfinite(widths)):
        raise ValueError("source gripper widths must be finite with shape (frames, 2)")
    return widths


def _resolve_gripper_calibrations(
    umi1_path: str | Path | None, umi2_path: str | Path | None
) -> tuple[np.ndarray, dict[str, Any]]:
    built_in = (LEFT_GRIPPER_WIDTH_LIMITS_MM, RIGHT_GRIPPER_WIDTH_LIMITS_MM)
    paths = (umi1_path, umi2_path)
    names = ("umi1", "umi2")
    limits: list[tuple[float, float]] = []
    provenance: dict[str, Any] = {}
    for name, path, fallback in zip(names, paths, built_in, strict=True):
        if path is None:
            limits.append(fallback)
            provenance[name] = {"source": "built_in_i2rt_linear_4310_calibration"}
        else:
            calibrated, details = load_single_gripper_calibration(path, expected_gripper=name)
            limits.append(calibrated)
            provenance[name] = details
    return np.asarray(limits, dtype=np.float64), provenance


def _accepted_gripper_width_diagnostics(
    sources: dict[tuple[str, str], LeRobotSource],
    exports: list[CombinedExportEpisode],
    calibrated_limits_mm: np.ndarray,
) -> dict[str, Any]:
    """Summarize accepted raw widths and values clipped by physical calibration."""
    limits = np.asarray(calibrated_limits_mm, dtype=np.float64)
    if limits.shape != (2, 2):
        raise ValueError("calibrated gripper limits must have shape (2, 2)")
    minimum = np.full(2, np.inf, dtype=np.float64)
    maximum = np.full(2, -np.inf, dtype=np.float64)
    below = np.zeros(2, dtype=np.int64)
    above = np.zeros(2, dtype=np.int64)
    rows = 0
    for item in exports:
        source = sources[(item.source_dataset, item.source_revision)]
        widths = _raw_gripper_widths(_source_episode_table(source, item))
        minimum = np.minimum(minimum, np.min(widths, axis=0))
        maximum = np.maximum(maximum, np.max(widths, axis=0))
        below += np.sum(widths < limits[:, 0], axis=0)
        above += np.sum(widths > limits[:, 1], axis=0)
        rows += len(widths)
    if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
        raise ValueError("could not compute accepted-dataset raw gripper diagnostics")
    return {
        "rows": rows,
        "umi1": {
            "observed_min_width_mm": float(minimum[0]),
            "observed_max_width_mm": float(maximum[0]),
            "values_clipped_below": int(below[0]),
            "values_clipped_above": int(above[0]),
        },
        "umi2": {
            "observed_min_width_mm": float(minimum[1]),
            "observed_max_width_mm": float(maximum[1]),
            "values_clipped_below": int(below[1]),
            "values_clipped_above": int(above[1]),
        },
    }


def _normalize_gripper_widths(widths_mm: np.ndarray, limits_mm: np.ndarray) -> np.ndarray:
    widths = np.asarray(widths_mm, dtype=np.float64)
    limits = np.asarray(limits_mm, dtype=np.float64)
    if widths.ndim != 2 or widths.shape[1] != 2:
        raise ValueError("gripper widths must have shape (frames, 2)")
    if limits.shape != (2, 2):
        raise ValueError("gripper limits must have shape (2, 2) as [arm, min/max]")
    span = limits[:, 1] - limits[:, 0]
    if np.any(span <= 0):
        raise ValueError("each gripper must have a positive observed width range")
    return np.clip((widths - limits[:, 0]) / span, 0.0, 1.0)


def _cartesian_observation_state(
    cartesian_pose: np.ndarray, normalized_grippers: np.ndarray
) -> np.ndarray:
    pose = np.asarray(cartesian_pose, dtype=np.float64)
    grippers = np.asarray(normalized_grippers, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != 12:
        raise ValueError("Cartesian pose must have shape (frames, 12)")
    if grippers.shape != (len(pose), 2):
        raise ValueError("normalized grippers must have shape (frames, 2)")
    return np.column_stack((pose[:, :6], grippers[:, 0], pose[:, 6:], grippers[:, 1])).astype(
        np.float32
    )


def _gripper_normalization_metadata(
    limits_mm: np.ndarray,
    diagnostics: dict[str, Any] | None = None,
    calibration_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    limits = np.asarray(limits_mm, dtype=np.float64)
    result = {
        "method": "per_arm_calibrated_physical_min_max",
        "source_units": "millimeters",
        "formula": "clip((width_mm - min_width_mm) / (max_width_mm - min_width_mm), 0, 1)",
        "normalized_semantics": {"0": "closed", "1": "open"},
        "umi1": {
            "min_width_mm": float(limits[0, 0]),
            "max_width_mm": float(limits[0, 1]),
        },
        "umi2": {
            "min_width_mm": float(limits[1, 0]),
            "max_width_mm": float(limits[1, 1]),
        },
        "state_indices": {"umi1": 6, "umi2": 13},
    }
    if calibration_provenance is not None:
        result["calibration_provenance"] = calibration_provenance
    if diagnostics is not None:
        result["accepted_raw_width_diagnostics"] = diagnostics
    return result


def _exact_numeric_stats(
    values_by_feature: dict[str, list[np.ndarray]],
) -> dict[str, dict[str, np.ndarray]]:
    """Compute global numeric statistics over the exact exported rows.

    Quantiles are not mergeable by averaging per-episode quantiles, even when
    weighting by episode length. Numeric LeRobot features are small enough to
    retain during export, so compute their final statistics from the full value
    population. Video statistics continue to use their source metadata.
    """
    return {
        feature: _feature_stats(np.concatenate(parts, axis=0))
        for feature, parts in values_by_feature.items()
    }


def validate_combined_dataset(root: str | Path, *, variant: Variant) -> dict[str, Any]:
    dataset_root = Path(root).expanduser().resolve()
    info = json.loads((dataset_root / "meta/info.json").read_text(encoding="utf-8"))
    if "action" in info["features"]:
        raise ValueError("combined dataset still contains action")
    width = 14
    expected_columns = [
        "observation.state",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    data_files = sorted((dataset_root / "data").rglob("*.parquet"))
    episode_files = sorted((dataset_root / "meta/episodes").rglob("*.parquet"))
    episodes = int(info["total_episodes"])
    if len(data_files) != episodes or len(episode_files) != episodes:
        raise ValueError("data or episode-metadata file count does not match total_episodes")
    next_index = 0
    for episode_index, path in enumerate(data_files):
        table = pq.read_table(path)
        if table.column_names != expected_columns:
            raise ValueError(f"{path}: unexpected columns {table.column_names}")
        if table.schema.field("observation.state").type != pa.list_(pa.float32(), width):
            raise ValueError(f"{path}: observation.state is not float32[{width}]")
        state = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
        if variant == "cartesian" and not np.array_equal(
            state[0, CARTESIAN_POSE_INDICES], np.zeros(12)
        ):
            raise ValueError(f"{path}: Cartesian poses do not start exactly at zero")
        if variant == "joint":
            if np.max(np.abs(state[0, CARTESIAN_POSE_INDICES])) > 0:
                raise ValueError(f"{path}: YAM joints do not start exactly at home")
        grippers = state[:, GRIPPER_STATE_INDICES]
        if np.min(grippers) < 0 or np.max(grippers) > 1:
            raise ValueError(f"{path}: normalized gripper values leave [0, 1]")
        rows = len(table)
        if not np.all(table["episode_index"].to_numpy() == episode_index):
            raise ValueError(f"{path}: wrong episode_index")
        if not np.array_equal(table["index"].to_numpy(), np.arange(next_index, next_index + rows)):
            raise ValueError(f"{path}: global index is not contiguous")
        next_index += rows
    if next_index != int(info["total_frames"]):
        raise ValueError("data row count does not match total_frames")
    videos = sorted((dataset_root / "videos").rglob("*.mp4"))
    if len(videos) != 2 * episodes:
        raise ValueError("video file count does not match two UMI streams per episode")
    return {
        "variant": variant,
        "episodes": episodes,
        "frames": next_index,
        "data_files": len(data_files),
        "video_files": len(videos),
        "state_width": width,
        "action_present": False,
    }


def push_combined_dataset(
    root: str | Path,
    *,
    variant: Variant,
    repo_id: str,
    private: bool = False,
    workers: int = 12,
) -> str:
    """Validate and upload one completed combined artifact using resumable workers."""
    validate_combined_dataset(root, variant=variant)
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the 'hub' extra before pushing") from exc
    dataset_root = Path(root).expanduser().resolve()
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=dataset_root,
        num_workers=max(1, workers),
        print_report=True,
        print_report_every=30,
    )
    return f"https://huggingface.co/datasets/{repo_id}"


def verify_remote_combined_dataset(repo_id: str, *, variant: Variant) -> dict[str, Any]:
    """Verify the committed Hub tree and representative Parquet schemas."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("install the 'hub' extra before remote verification") from exc
    api = HfApi()
    repository = api.repo_info(repo_id=repo_id, repo_type="dataset")
    revision = repository.sha
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=revision)
    info_path = hf_hub_download(repo_id, "meta/info.json", repo_type="dataset", revision=revision)
    info = json.loads(Path(info_path).read_text(encoding="utf-8"))
    episodes = int(info["total_episodes"])
    width = 14
    if "action" in info["features"]:
        raise ValueError(f"{repo_id}: remote info contains action")
    if info["features"]["observation.state"]["shape"] != [width]:
        raise ValueError(f"{repo_id}: remote state width is not {width}")
    categories = {
        "data_files": sorted(
            path for path in files if path.startswith("data/") and path.endswith(".parquet")
        ),
        "episode_metadata_files": sorted(
            path
            for path in files
            if path.startswith("meta/episodes/") and path.endswith(".parquet")
        ),
        "video_files": sorted(
            path for path in files if path.startswith("videos/") and path.endswith(".mp4")
        ),
    }
    if len(categories["data_files"]) != episodes:
        raise ValueError(f"{repo_id}: remote data file count mismatch")
    if len(categories["episode_metadata_files"]) != episodes:
        raise ValueError(f"{repo_id}: remote episode metadata count mismatch")
    if len(categories["video_files"]) != 2 * episodes:
        raise ValueError(f"{repo_id}: remote video count mismatch")
    expected_columns = [
        "observation.state",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    sample_paths = [categories["data_files"][0], categories["data_files"][-1]]
    sample_rows = 0
    for remote_path in sample_paths:
        local_path = hf_hub_download(repo_id, remote_path, repo_type="dataset", revision=revision)
        table = pq.read_table(local_path)
        if table.column_names != expected_columns:
            raise ValueError(f"{repo_id}/{remote_path}: unexpected columns")
        if table.schema.field("observation.state").type != pa.list_(pa.float32(), width):
            raise ValueError(f"{repo_id}/{remote_path}: unexpected observation.state type")
        sample_rows += len(table)
    return {
        "repo_id": repo_id,
        "revision": revision,
        "private": bool(repository.private),
        "episodes": episodes,
        "frames": int(info["total_frames"]),
        "data_files": len(categories["data_files"]),
        "episode_metadata_files": len(categories["episode_metadata_files"]),
        "video_files": len(categories["video_files"]),
        "sample_data_rows_checked": sample_rows,
        "state_width": width,
        "action_present": False,
    }


def _prepare_output_root(root: Path) -> None:
    marker = root / ".build-in-progress"
    if root.exists() and any(root.iterdir()) and not marker.is_file():
        raise FileExistsError(f"destination is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    marker.touch()
    _make_layout(root)


def _validate_pipeline_report(report: dict[str, Any]) -> None:
    if report.get("pipeline_version") != PIPELINE_VERSION:
        raise ValueError("combined report was produced by a different pipeline version")
    if report.get("kinematics") != KINEMATICS_PROVENANCE:
        raise ValueError("combined report does not use the pinned 220 mm I2RT kinematics")
    if report.get("gate_semantics_changed") is not False:
        raise ValueError("combined report did not preserve filtering gate semantics")
    if len(report.get("per_episode", [])) != int(report.get("episodes", -1)):
        raise ValueError("combined report is missing per-episode results")
    for row in report["per_episode"]:
        if row.get("accepted") and not row.get("selected_result", {}).get(
            "continuously_replayable"
        ):
            raise ValueError("accepted row is not continuously replayable")


def _source_episode_table(source: LeRobotSource, item: CombinedExportEpisode) -> pa.Table:
    relative = f"data/chunk-{item.source_data_chunk:03d}/file-{item.source_data_file:03d}.parquet"
    table = pq.read_table(source.materialize(relative))
    table = table.filter(pc.equal(table["episode_index"], item.source_episode_index))
    order = np.argsort(table["frame_index"].to_numpy())
    table = table.take(pa.array(order, type=pa.int64()))
    if len(table) != item.frames:
        raise ValueError(
            f"{item.source_dataset} episode {item.source_episode_index}: expected "
            f"{item.frames} source rows, found {len(table)}"
        )
    return table


def _load_result_arrays(item: CombinedExportEpisode) -> dict[str, np.ndarray]:
    if not item.result_array.is_file():
        raise FileNotFoundError(item.result_array)
    with np.load(item.result_array, allow_pickle=False) as stored:
        arrays = {key: np.asarray(stored[key]).copy() for key in stored.files}
    expected = {"cartesian_state", "q", "normalized_grippers"}
    if set(arrays) != expected:
        raise ValueError(f"{item.result_array}: unexpected arrays {sorted(arrays)}")
    if arrays["cartesian_state"].shape != (item.frames, 12):
        raise ValueError(f"{item.result_array}: incorrect Cartesian state shape")
    if arrays["q"].shape != (item.frames, 2, 6):
        raise ValueError(f"{item.result_array}: incorrect q shape")
    if arrays["normalized_grippers"].shape != (item.frames, 2):
        raise ValueError(f"{item.result_array}: incorrect gripper shape")
    return arrays


def _rewrite_combined_episode_table(
    source: pa.Table,
    state: np.ndarray,
    *,
    variant: Variant,
    output_episode_index: int,
    global_from_index: int,
) -> pa.Table:
    state = np.asarray(state, dtype=np.float32)
    if len(source) != len(state):
        raise ValueError("source and optimized state lengths differ")
    width = state.shape[1]
    state_array = pa.FixedSizeListArray.from_arrays(
        pa.array(state.reshape(-1), type=pa.float32()), width
    )
    columns: dict[str, pa.Array | pa.ChunkedArray] = {"observation.state": state_array}
    for key in ("timestamp", "frame_index", "task_index"):
        columns[key] = source[key]
    columns["episode_index"] = pa.array(np.full(len(source), output_episode_index, dtype=np.int64))
    columns["index"] = pa.array(
        np.arange(global_from_index, global_from_index + len(source), dtype=np.int64)
    )
    ordered = [
        "observation.state",
        "timestamp",
        "frame_index",
        "episode_index",
        "index",
        "task_index",
    ]
    table = pa.table({name: columns[name] for name in ordered})
    metadata = {
        "info": {
            "features": {
                name: _arrow_feature_metadata(table.schema.field(name).type, width)
                for name in ordered
            }
        },
        "fingerprint": hashlib.sha256(state.tobytes()).hexdigest()[:16],
    }
    return table.replace_schema_metadata(
        {b"huggingface": json.dumps(metadata, separators=(",", ":")).encode()}
    )


def _arrow_feature_metadata(data_type: pa.DataType, state_width: int) -> dict[str, Any]:
    if pa.types.is_fixed_size_list(data_type):
        return {
            "feature": {"dtype": "float32", "_type": "Value"},
            "length": state_width,
            "_type": "List",
        }
    dtype = "float32" if pa.types.is_float32(data_type) else "int64"
    return {"dtype": dtype, "_type": "Value"}


def _numeric_table_data(table: pa.Table) -> dict[str, np.ndarray]:
    result = {
        "observation.state": np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    }
    for key in table.column_names[1:]:
        result[key] = np.asarray(
            table[key].to_numpy(zero_copy_only=False), dtype=np.float64
        ).reshape(-1, 1)
    return result


def _combined_info(
    source_info: dict[str, Any], *, variant: Variant, episodes: int, frames: int
) -> dict[str, Any]:
    info = json.loads(json.dumps(source_info))
    info["features"].pop("action", None)
    info["features"].pop("observation.gripper_width.umi1", None)
    info["features"].pop("observation.gripper_width.umi2", None)
    if variant == "joint":
        info["features"]["observation.state"] = {
            "dtype": "float32",
            "shape": [14],
            "names": list(YAM_STATE_NAMES),
        }
        info["robot_type"] = "bi_yam_follower"
    else:
        info["features"]["observation.state"] = {
            "dtype": "float32",
            "shape": [14],
            "names": list(CARTESIAN_STATE_NAMES),
        }
    info["total_episodes"] = episodes
    info["total_frames"] = frames
    info["splits"] = {"train": f"0:{episodes}"}
    return info


def _materialize_combined_videos(
    sources: dict[tuple[str, str], LeRobotSource],
    roots: dict[str, Path],
    exports: list[CombinedExportEpisode],
    *,
    workers: int,
) -> None:
    jobs = [(item, arm, feature) for item in exports for arm, feature in enumerate(VIDEO_FEATURES)]

    def copy(job: tuple[CombinedExportEpisode, int, str]) -> None:
        item, arm, feature = job
        source = sources[(item.source_dataset, item.source_revision)]
        relative = (
            f"videos/{feature}/chunk-{item.source_video_chunks[arm]:03d}/"
            f"file-{item.source_video_files[arm]:03d}.mp4"
        )
        source_path = source.materialize(relative)
        for root in roots.values():
            destination = (
                root
                / "videos"
                / feature
                / "chunk-000"
                / f"file-{item.output_episode_index:03d}.mp4"
            )
            if destination.is_file() and destination.stat().st_size == source_path.stat().st_size:
                continue
            if destination.exists():
                destination.unlink()
            try:
                os.link(source_path, destination)
            except OSError:
                shutil.copy2(source_path, destination)

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as pool:
        list(pool.map(copy, jobs))


def _write_dataset_card(root: Path, provenance: dict[str, Any], variant: Variant) -> None:
    sources = ", ".join(
        f"[{source['repo_id']}](https://huggingface.co/datasets/{source['repo_id']})"
        for source in provenance["sources"]
    )
    state_text = (
        "The 14-D `observation.state` contains the smoothed, trajectory-optimized "
        "YAM-achievable UMI1 pose, normalized UMI1 gripper, UMI2 pose, and normalized "
        "UMI2 gripper."
        if variant == "cartesian"
        else "The 14-D `observation.state` contains left YAM joints 0–5, normalized left "
        "gripper, right YAM joints 0–5, and normalized right gripper."
    )
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
        "# Combined filtered dual-LiDAR UMI demonstrations\n\n"
        f"Observation-only LeRobot v3 derivative of {sources}. It contains "
        f"{provenance['episodes']} demonstrations ({provenance['frames']} frames) accepted "
        "by the continuous bimanual YAM replayability pipeline.\n\n"
        f"{state_text} The two original UMI videos, timestamps, frame cadence, and task are "
        "preserved; `action` is intentionally absent.\n\n"
        "Each gripper is normalized independently from its calibrated physical closed/open "
        "width and clipped to `[0, 1]`. The calibration, formula, observed raw extrema, and "
        "clipping counts are recorded in `meta/filtering.json`.\n\n"
        "IK and FK use the current BiYAM-pinned I2RT YAM model with the custom 220 mm "
        "linear gripper and UMI-compatible grasp frame.\n\n"
        "Both UMI trajectories are expressed relative to their own first pose, so both begin "
        "at zero. See `meta/filtering.json` for pinned source revisions, gates, provenance, "
        "and the output-to-source episode map. Collision checks are sampled-capsule proxies; "
        "validate the exact scene before hardware playback.\n",
        encoding="utf-8",
    )


def _validate_matching_outputs(cartesian_root: Path, joint_root: Path) -> None:
    cart_files = sorted((cartesian_root / "data").rglob("*.parquet"))
    joint_files = sorted((joint_root / "data").rglob("*.parquet"))
    if len(cart_files) != len(joint_files):
        raise ValueError("Cartesian and joint outputs have different episode counts")
    for cart_path, joint_path in zip(cart_files, joint_files, strict=True):
        cart = pq.read_table(cart_path)
        joint = pq.read_table(joint_path)
        for column in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            if not cart[column].equals(joint[column]):
                raise ValueError(f"output timelines differ at {cart_path.name}: {column}")
    for relative in (
        "meta/filtering.json",
        "meta/tasks.parquet",
    ):
        if not (cartesian_root / relative).is_file() or not (joint_root / relative).is_file():
            raise ValueError(f"matching output is missing {relative}")
