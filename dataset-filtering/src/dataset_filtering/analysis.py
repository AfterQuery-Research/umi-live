from __future__ import annotations

import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from dataset_filtering.hf_dataset import LeRobotSource, as_pose_episode
from dataset_filtering.models import EpisodeKind
from dataset_filtering.noise import aggregate_pose_reports, pose_noise_metrics, vector_noise_metrics
from dataset_filtering.smoothing import DEFAULT_CANDIDATES, benchmark_smoothers


def analyze_dataset(
    source: LeRobotSource,
    *,
    episode_indices: list[int] | None = None,
    benchmark_smoothing: bool = True,
) -> dict[str, Any]:
    summaries = source.episode_summaries()
    selected = (
        summaries
        if episode_indices is None
        else [summary for summary in summaries if summary.episode_index in set(episode_indices)]
    )
    if not selected:
        raise ValueError("no episodes selected")
    source.prefetch_episode_data([summary.episode_index for summary in selected])
    pose_reports: list[dict[str, Any]] = []
    vector_reports: list[dict[str, Any]] = []
    smoothing_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    per_episode: list[dict[str, Any]] = []
    for number, summary in enumerate(selected, start=1):
        episode = source.load_episode(summary.episode_index)
        item: dict[str, Any] = {
            "episode_index": episode.episode_index,
            "frames": len(episode.timestamps),
            "duration_s": len(episode.timestamps) / episode.fps,
            "kind": episode.kind.value,
        }
        if episode.kind is EpisodeKind.UMI_POSE:
            pose = as_pose_episode(episode)
            noise = pose_noise_metrics(pose)
            pose_reports.append(noise)
            item["noise"] = noise["aggregate"]
            if benchmark_smoothing:
                results = benchmark_smoothers(pose)
                item["best_smoother"] = _result_summary(results[0])
                for result in results:
                    key = _candidate_key(result.method, result.parameters)
                    smoothing_by_candidate[key].append(_result_summary(result))
        else:
            metrics = vector_noise_metrics(episode)
            vector_reports.append(metrics)
            item["noise"] = metrics
        per_episode.append(item)
        print(
            f"analyze: {number}/{len(selected)} episode {episode.episode_index} "
            f"({episode.kind.value}, {len(episode.timestamps)} frames)",
            flush=True,
        )

    schema_kind = "umi_pose" if pose_reports and not vector_reports else "mixed_or_joint"
    result: dict[str, Any] = {
        "source": source.source,
        "revision": source.location.revision,
        "generated_at": datetime.now(UTC).isoformat(),
        "read_only": True,
        "schema_kind": schema_kind,
        "episodes_analyzed": len(selected),
        "dataset_info": {
            key: source.info().get(key)
            for key in ("fps", "total_episodes", "total_frames", "total_tasks", "robot_type")
        },
        "per_episode": per_episode,
    }
    if pose_reports:
        result["noise"] = aggregate_pose_reports(pose_reports)
    if vector_reports:
        result["joint_or_vector_noise"] = vector_reports
    if smoothing_by_candidate:
        candidates = [
            _aggregate_candidate(key, values) for key, values in smoothing_by_candidate.items()
        ]
        candidates.sort(key=lambda candidate: candidate["score_median"])
        result["smoothing_experiment"] = {
            "selection_rule": (
                "minimum median score across episodes; score balances >6 Hz residual, jerk, "
                "trajectory distortion, direction preservation, and exact gripper-event anchors"
            ),
            "candidates": candidates,
            "recommended": candidates[0],
        }
    return result


def write_analysis(report: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def _candidate_key(method: str, parameters: dict[str, Any]) -> str:
    return f"{method}:{json.dumps(parameters, sort_keys=True, separators=(',', ':'))}"


def _result_summary(result: Any) -> dict[str, Any]:
    return {
        "method": result.method,
        "parameters": result.parameters,
        "score": result.score,
        "metrics": result.metrics,
    }


def _aggregate_candidate(key: str, values: list[dict[str, Any]]) -> dict[str, Any]:
    del key
    metric_names = values[0]["metrics"].keys()
    return {
        "method": values[0]["method"],
        "parameters": values[0]["parameters"],
        "episodes": len(values),
        "score_median": float(np.median([value["score"] for value in values])),
        "score_p90": float(np.percentile([value["score"] for value in values], 90)),
        "metrics_median": {
            metric: float(np.median([value["metrics"][metric] for value in values]))
            for metric in metric_names
        },
    }


def candidate_from_report(report: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    smoothing = report.get("smoothing_experiment")
    if not smoothing:
        return DEFAULT_CANDIDATES[0][0], dict(DEFAULT_CANDIDATES[0][1])
    recommended = smoothing["recommended"]
    return str(recommended["method"]), dict(recommended["parameters"])
