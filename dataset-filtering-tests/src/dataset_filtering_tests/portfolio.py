from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from dataset_filtering_tests.baseline import BASELINE_REPORT, load_baseline


def build_solver_portfolio(
    *,
    baseline_report: str | Path = BASELINE_REPORT,
    experiments: list[tuple[str | Path, str]],
) -> dict[str, Any]:
    baseline = load_baseline(baseline_report)
    candidates: dict[int, list[dict[str, Any]]] = {
        int(row["episode_index"]): [_baseline_candidate(row)]
        for row in baseline["per_episode"]
    }
    experiment_sources: list[dict[str, str]] = []
    for report_value, profile in experiments:
        path = Path(report_value).expanduser().resolve()
        report = json.loads(path.read_text(encoding="utf-8"))
        experiment_sources.append({"path": str(path), "profile": profile})
        for row in report["per_episode"]:
            episode_index = int(row["episode_index"])
            candidates[episode_index].append(
                _experiment_candidate(row["profiles"][profile], profile, str(path))
            )

    selected_rows: list[dict[str, Any]] = []
    selection_counts: Counter[str] = Counter()
    for episode_index in sorted(candidates):
        values = candidates[episode_index]
        passing = [value for value in values if value["continuously_replayable"]]
        if passing:
            baseline_passing = [value for value in passing if value["profile"] == "baseline_v3"]
            selected = baseline_passing[0] if baseline_passing else max(passing, key=_quality_key)
            selection_counts[selected["profile"]] += 1
        else:
            selected = max(values, key=_failed_candidate_key)
        selected_rows.append(
            {
                "episode_index": episode_index,
                "accepted": bool(passing),
                "selected_profile": selected["profile"],
                "quality": selected["quality"],
                "validation": selected["validation"],
                "available_profiles": [value["profile"] for value in values],
            }
        )

    accepted = [row for row in selected_rows if row["accepted"]]
    rejection_signatures = Counter(
        tuple(
            key.removeprefix("passes_")
            for key, value in row["validation"].items()
            if key.startswith("passes_") and not value
        )
        for row in selected_rows
        if not row["accepted"]
    )
    return {
        "experiment": "unchanged_gate_solver_portfolio",
        "baseline_report": str(Path(baseline_report).expanduser().resolve()),
        "experiment_sources": experiment_sources,
        "episodes": len(selected_rows),
        "accepted": len(accepted),
        "acceptance_rate": len(accepted) / max(len(selected_rows), 1),
        "target_70_percent_episodes": 105,
        "additional_needed_for_70_percent": max(0, 105 - len(accepted)),
        "selection_counts": dict(sorted(selection_counts.items())),
        "gate_semantics_changed": False,
        "rejection_signatures": [
            {"reasons": list(signature), "episodes": count}
            for signature, count in sorted(
                rejection_signatures.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        "per_episode": selected_rows,
    }


def _baseline_candidate(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile": "baseline_v3",
        "continuously_replayable": bool(row["continuously_replayable"]),
        "validation": row["validation"],
        "quality": {
            "position_error_max_mm": float(row["position_error_max_mm"]),
            "orientation_error_max_deg": float(row["orientation_error_max_deg"]),
            "flow_direction_cosine_median": float(row["flow_direction_cosine_median"]),
        },
    }


def _experiment_candidate(
    value: dict[str, Any], profile: str, source: str
) -> dict[str, Any]:
    return {
        "profile": profile,
        "source": source,
        "continuously_replayable": bool(value["continuously_replayable"]),
        "validation": value["validation"],
        "quality": {
            "position_error_max_mm": float(value["position_error_max_mm"]),
            "orientation_error_max_deg": float(value["orientation_error_max_deg"]),
            "flow_direction_cosine_median": float(value["flow_direction_cosine_median"]),
        },
    }


def _quality_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    validation = candidate["validation"]
    interarm_margin = float(validation["minimum_interarm_surface_clearance_mm"]) - 5.0
    self_margin = float(validation["minimum_self_surface_clearance_mm"]) - 1.0
    clearance_margin = min(interarm_margin, self_margin)
    quality = candidate["quality"]
    normalized_tracking = max(
        quality["position_error_max_mm"] / 10.0,
        quality["orientation_error_max_deg"] / 8.0,
    )
    return (
        clearance_margin,
        -normalized_tracking,
        quality["flow_direction_cosine_median"],
    )


def _failed_candidate_key(candidate: dict[str, Any]) -> tuple[float, ...]:
    passes = sum(
        bool(value)
        for key, value in candidate["validation"].items()
        if key.startswith("passes_")
    )
    return (float(passes), *_quality_key(candidate))
