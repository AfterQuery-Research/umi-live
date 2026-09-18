from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from dataset_filtering.export_dataset import (
    build_filtered_dataset,
    optimized_world_poses_to_umi_state,
    validate_filtered_dataset,
)
from dataset_filtering.hf_dataset import LeRobotSource, as_pose_episode
from dataset_filtering.smoothing import smooth_pose_episode, with_smoothed_pose
from dataset_filtering.trajectory_optimization import (
    TrajectoryOptimizationConfig,
    optimize_pose_episode,
)

from dataset_filtering_tests.trajectory_sweep import CONFIG_PROFILES

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PORTFOLIO_REPORT = PACKAGE_ROOT / "reports/portfolio-final-zero-home.json"
DEFAULT_SOURCE = "brandonyang/dual-lidar-umi-relative"
DEFAULT_REPO_ID = "brandonyang/dual-lidar-umi-relative-filtered"


@dataclass(frozen=True)
class PortfolioSelection:
    episode_index: int
    profile: str
    smoothing_method: str
    smoothing_parameters: dict[str, Any]
    evidence_report: str

    @property
    def signature(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_portfolio_selections(
    portfolio_path: str | Path = DEFAULT_PORTFOLIO_REPORT,
) -> tuple[dict[str, Any], list[PortfolioSelection]]:
    """Resolve every accepted portfolio row to its exact solver and smoother."""
    path = Path(portfolio_path).expanduser().resolve()
    portfolio = json.loads(path.read_text(encoding="utf-8"))
    baseline_path = Path(portfolio["baseline_report"]).expanduser().resolve()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_rows = {
        int(row["episode_index"]): row for row in baseline["per_episode"]
    }

    experiment_reports: list[tuple[dict[str, Any], str, str]] = []
    for source in portfolio["experiment_sources"]:
        report_path = Path(source["path"]).expanduser().resolve()
        experiment_reports.append(
            (
                json.loads(report_path.read_text(encoding="utf-8")),
                str(source["profile"]),
                str(report_path),
            )
        )

    selections: list[PortfolioSelection] = []
    for selected in portfolio["per_episode"]:
        if not selected["accepted"]:
            continue
        episode_index = int(selected["episode_index"])
        profile = str(selected["selected_profile"])
        baseline_smoothing = baseline_rows[episode_index]["smoothing"]
        if profile == "baseline_v3":
            selections.append(
                PortfolioSelection(
                    episode_index=episode_index,
                    profile=profile,
                    smoothing_method=str(baseline_smoothing["method"]),
                    smoothing_parameters=dict(baseline_smoothing["parameters"]),
                    evidence_report=str(baseline_path),
                )
            )
            continue

        matching: list[tuple[float, dict[str, Any], dict[str, Any], str]] = []
        for report, report_profile, report_path in experiment_reports:
            if report_profile != profile:
                continue
            rows = {
                int(row["episode_index"]): row for row in report.get("per_episode", [])
            }
            row = rows.get(episode_index)
            if row is None:
                continue
            result = row["profiles"][profile]
            if not result["continuously_replayable"]:
                continue
            distance = _quality_distance(selected["quality"], result)
            matching.append((distance, row, report, report_path))
        if not matching:
            raise ValueError(
                f"episode {episode_index}: no passing evidence for selected profile {profile}"
            )
        distance, row, report, evidence_path = min(matching, key=lambda item: item[0])
        if distance > 1e-8:
            raise ValueError(
                f"episode {episode_index}: selected quality does not match experiment evidence"
            )
        smoothing = row.get("smoothing")
        if smoothing is None:
            smoothing = report.get("smoothing_overrides", {}).get(str(episode_index))
        if smoothing is None:
            smoothing = baseline_smoothing
        selections.append(
            PortfolioSelection(
                episode_index=episode_index,
                profile=profile,
                smoothing_method=str(smoothing["method"]),
                smoothing_parameters=dict(smoothing["parameters"]),
                evidence_report=evidence_path,
            )
        )

    selections.sort(key=lambda value: value.episode_index)
    if len(selections) != int(portfolio["accepted"]):
        raise ValueError(
            f"portfolio says {portfolio['accepted']} accepted episodes, resolved {len(selections)}"
        )
    return portfolio, selections


def build_portfolio_dataset(
    *,
    destination: str | Path,
    portfolio_path: str | Path = DEFAULT_PORTFOLIO_REPORT,
    source_name: str = DEFAULT_SOURCE,
    source_revision: str = "main",
    include_videos: bool = True,
    workers: int = 8,
) -> dict[str, Any]:
    """Regenerate, gate-check, and package the final observation-only portfolio."""
    portfolio_file = Path(portfolio_path).expanduser().resolve()
    portfolio, selections = load_portfolio_selections(portfolio_file)
    output_root = Path(destination).expanduser().resolve()
    marker = output_root / ".build-in-progress"
    cache_root = output_root / ".build-cache"
    if output_root.exists() and any(output_root.iterdir()) and not marker.is_file():
        raise FileExistsError(f"destination is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    marker.touch()
    cache_root.mkdir(parents=True, exist_ok=True)

    source = LeRobotSource(source_name, revision=source_revision)
    indices = [selection.episode_index for selection in selections]
    # Materialize shared metadata in the parent before worker processes start.
    # LeRobotSource's download cache is intentionally simple; concurrent first-use
    # of the same ``meta/info.json.partial`` path would otherwise race.
    source.info()
    source.prefetch_episode_data(indices)
    source_metadata = source.episode_metadata_frame().set_index("episode_index", drop=False)
    arguments = [
        (
            source_name,
            source_revision,
            str(source.cache_dir),
            asdict(selection),
            str(cache_root / f"source-episode-{selection.episode_index:03d}.npz"),
        )
        for selection in selections
    ]

    completed_rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(arguments)))) as executor:
        futures = [executor.submit(_solve_and_cache_episode, argument) for argument in arguments]
        for completed, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            completed_rows.append(row)
            status = "cached" if row["cache_hit"] else "solved"
            print(
                f"portfolio export: {completed}/{len(futures)} {status} "
                f"source episode {row['episode_index']}",
                flush=True,
            )
    completed_rows.sort(key=lambda row: row["episode_index"])

    selection_report = PACKAGE_ROOT / "reports/portfolio-export-selection.json"
    report_rows = []
    for selection in selections:
        frames = int(source_metadata.loc[selection.episode_index, "length"])
        report_rows.append(
            {
                "episode_index": selection.episode_index,
                "frames": frames,
                "both_valid_frames": frames,
                "continuously_replayable": True,
            }
        )
    selection_payload = {
        "source": source_name,
        "algorithm": "contact_anchored_bimanual_cubic_bspline_nlp_with_tail_freeze_v3",
        "portfolio_report": str(portfolio_file),
        "per_episode": report_rows,
    }
    selection_report.parent.mkdir(parents=True, exist_ok=True)
    selection_report.write_text(json.dumps(selection_payload, indent=2) + "\n", encoding="utf-8")

    build_filtered_dataset(
        source,
        report_path=selection_report,
        destination=output_root,
        include_videos=include_videos,
        workers=workers,
    )
    _rewrite_portfolio_provenance(
        output_root,
        portfolio=portfolio,
        portfolio_path=portfolio_file,
        selections=selections,
        source_revision=source_revision,
        solve_rows=completed_rows,
    )
    validated = validate_filtered_dataset(output_root)
    validated["destination"] = str(output_root)
    validated["selection_profiles"] = dict(portfolio["selection_counts"])
    return validated


def _solve_and_cache_episode(
    arguments: tuple[str, str, str, dict[str, Any], str],
) -> dict[str, Any]:
    source_name, revision, source_cache, raw_selection, raw_cache_path = arguments
    selection = PortfolioSelection(**raw_selection)
    cache_path = Path(raw_cache_path)
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            metadata = json.loads(str(cached["metadata_json"]))
            state = cached["state"]
        if (
            metadata.get("selection_signature") == selection.signature
            and int(metadata.get("source_episode_index", -1)) == selection.episode_index
            and state.ndim == 2
            and state.shape[1] == 12
        ):
            return {
                "episode_index": selection.episode_index,
                "profile": selection.profile,
                "cache_hit": True,
            }
        cache_path.unlink()

    source = LeRobotSource(source_name, revision=revision, cache_dir=source_cache)
    original = as_pose_episode(source.load_episode(selection.episode_index))
    smoothing = smooth_pose_episode(
        original,
        selection.smoothing_method,
        selection.smoothing_parameters,
    )
    filtered = with_smoothed_pose(original, smoothing)
    if selection.profile == "baseline_v3":
        config = TrajectoryOptimizationConfig()
    else:
        try:
            config = CONFIG_PROFILES[selection.profile]
        except KeyError as exc:
            raise ValueError(f"unsupported portfolio profile: {selection.profile}") from exc
    result = optimize_pose_episode(filtered, config=config)
    if not result.metadata["continuously_replayable"]:
        validation = result.metadata["validation"]
        failures = [
            key.removeprefix("passes_")
            for key, value in validation.items()
            if key.startswith("passes_") and not value
        ]
        raise RuntimeError(
            f"source episode {selection.episode_index} no longer passes its selected "
            f"{selection.profile} solve; failures={failures}"
        )
    state = optimized_world_poses_to_umi_state(
        result.achieved_positions_world,
        result.achieved_quaternions_world,
        original.positions,
        original.quaternions,
    )
    if len(state) != len(original.timestamps):
        raise RuntimeError(
            f"source episode {selection.episode_index}: optimized path changed frame count"
        )
    metadata = {
        "source_episode_index": selection.episode_index,
        "profile": selection.profile,
        "selection_signature": selection.signature,
        "evidence_report": selection.evidence_report,
        "smoothing": {
            "method": smoothing.method,
            "parameters": smoothing.parameters,
            "metrics": smoothing.metrics,
        },
        "retarget": result.metadata,
    }
    np.savez_compressed(cache_path, state=state, metadata_json=json.dumps(metadata))
    return {
        "episode_index": selection.episode_index,
        "profile": selection.profile,
        "cache_hit": False,
    }


def _quality_distance(expected: dict[str, Any], result: dict[str, Any]) -> float:
    return float(
        abs(float(expected["position_error_max_mm"]) - float(result["position_error_max_mm"]))
        + abs(
            float(expected["orientation_error_max_deg"])
            - float(result["orientation_error_max_deg"])
        )
        + abs(
            float(expected["flow_direction_cosine_median"])
            - float(result["flow_direction_cosine_median"])
        )
    )


def _rewrite_portfolio_provenance(
    output_root: Path,
    *,
    portfolio: dict[str, Any],
    portfolio_path: Path,
    selections: list[PortfolioSelection],
    source_revision: str,
    solve_rows: list[dict[str, Any]],
) -> None:
    path = output_root / "meta/filtering.json"
    provenance = json.loads(path.read_text(encoding="utf-8"))
    provenance.update(
        {
            "selection_policy": (
                "accepted by the final zero-home solver portfolio and independently "
                "regenerated through every unchanged replayability gate"
            ),
            "selection_report": str(portfolio_path),
            "portfolio_experiment": portfolio["experiment"],
            "portfolio_selection_counts": portfolio["selection_counts"],
            "gate_semantics_changed": bool(portfolio["gate_semantics_changed"]),
            "source_revision": source_revision,
            "regenerated_episode_count": len(solve_rows),
            "episode_index_map": [
                {
                    "output_episode_index": output_index,
                    "source_episode_index": selection.episode_index,
                    "profile": selection.profile,
                    "smoothing": {
                        "method": selection.smoothing_method,
                        "parameters": selection.smoothing_parameters,
                    },
                }
                for output_index, selection in enumerate(selections)
            ],
        }
    )
    path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
