from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

BASELINE_REPORT = (
    Path(__file__).resolve().parents[3]
    / "dataset-filtering"
    / "reports"
    / "dual-lidar-umi-relative-tail-freeze-v3-secondary-objective-replayability.json"
)


def load_baseline(path: str | Path = BASELINE_REPORT) -> dict[str, Any]:
    report_path = Path(path).expanduser().resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows = report.get("per_episode")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"baseline report has no per-episode results: {report_path}")
    if report.get("episodes") != len(rows):
        raise ValueError(
            f"baseline episode count does not match per-episode rows: {report_path}"
        )
    return report


def failure_signature(row: dict[str, Any]) -> tuple[str, ...]:
    reasons: list[str] = []
    if not row["solver_success"]:
        reasons.append("solver")
    if not row["validation"]["passes_tracking_accuracy"]:
        reasons.append("tracking")
    if not row["validation"]["passes_contact_accuracy"]:
        reasons.append("contact")
    if not row["validation"]["passes_interarm_clearance"]:
        reasons.append("interarm")
    if not row["validation"]["passes_self_collision_clearance"]:
        reasons.append("self_collision")
    if not row["validation"]["passes_velocity_limits"]:
        reasons.append("velocity")
    if not row["validation"]["passes_acceleration_limits"]:
        reasons.append("acceleration")
    return tuple(reasons)


def summarize_baseline(report: dict[str, Any]) -> dict[str, Any]:
    rows = report["per_episode"]
    signatures = Counter(
        failure_signature(row) for row in rows if not row["continuously_replayable"]
    )
    return {
        "source": report["source"],
        "algorithm": report["algorithm"],
        "episodes": report["episodes"],
        "kept": report["episode_replayability"]["continuously_replayable"],
        "acceptance_rate": (
            report["episode_replayability"]["continuously_replayable"] / report["episodes"]
        ),
        "target_kept_for_70_percent": 105,
        "additional_episodes_needed": (
            105 - report["episode_replayability"]["continuously_replayable"]
        ),
        "gate_pass_counts": report["validation_pass_episodes"],
        "failure_signatures": [
            {"reasons": list(signature), "episodes": count}
            for signature, count in sorted(
                signatures.items(), key=lambda item: (-item[1], item[0])
            )
        ],
        "invariants": {
            "all_frame_position_mm": 10.0,
            "all_frame_orientation_deg": 8.0,
            "gripper_event_position_mm": 1.0,
            "gripper_event_orientation_deg": 2.0,
            "interarm_surface_clearance_mm": 5.0,
            "self_surface_clearance_mm": 1.0,
            "gate_tuning_to_hit_target_forbidden": True,
        },
    }
