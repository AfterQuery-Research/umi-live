from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

CALIBRATION_SCHEMA_VERSION = 1
GripperName = Literal["umi1", "umi2"]


def build_two_point_calibration_report(
    *,
    gripper: GripperName,
    closed_samples_mm: np.ndarray,
    open_samples_mm: np.ndarray,
    measurement: dict[str, Any],
    minimum_span_mm: float = 5.0,
) -> dict[str, Any]:
    """Build one-gripper calibration metadata from two stable endpoint bursts."""
    closed = _endpoint_summary(closed_samples_mm, "closed")
    opened = _endpoint_summary(open_samples_mm, "open")
    closed_width = float(closed["median_width_mm"])
    open_width = float(opened["median_width_mm"])
    span = open_width - closed_width
    if span < minimum_span_mm:
        raise ValueError(
            f"{gripper} open width must exceed closed width by at least "
            f"{minimum_span_mm:g} mm; measured span={span:.3f} mm"
        )
    return {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "calibration_type": "umi_gripper_width_two_point",
        "gripper": gripper,
        "source_units": "millimeters",
        "closed_width_mm": closed_width,
        "open_width_mm": open_width,
        "span_mm": span,
        "normalization_formula": (
            "clip((width_mm - closed_width_mm) / "
            "(open_width_mm - closed_width_mm), 0, 1)"
        ),
        "normalized_semantics": {"0": "closed", "1": "open"},
        "endpoints": {"closed": closed, "open": opened},
        "measurement": measurement,
    }


def write_gripper_calibration(report: dict[str, Any], destination: str | Path) -> Path:
    path = Path(destination).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return path


def load_single_gripper_calibration(
    path: str | Path, *, expected_gripper: GripperName
) -> tuple[tuple[float, float], dict[str, Any]]:
    calibration_path = Path(path).expanduser().resolve()
    raw = calibration_path.read_bytes()
    report = json.loads(raw)
    if report.get("schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise ValueError("unsupported gripper calibration schema version")
    if report.get("calibration_type") != "umi_gripper_width_two_point":
        raise ValueError("file is not an interactive two-point UMI gripper calibration")
    if report.get("gripper") != expected_gripper:
        raise ValueError(
            f"calibration is for {report.get('gripper')!r}, expected {expected_gripper!r}"
        )
    closed = float(report["closed_width_mm"])
    opened = float(report["open_width_mm"])
    if not np.isfinite(closed) or not np.isfinite(opened) or opened <= closed:
        raise ValueError("calibrated open width must be finite and exceed closed width")
    provenance = {
        "source": "interactive_two_point_calibration",
        "report_sha256": hashlib.sha256(raw).hexdigest(),
        "report_path": str(calibration_path),
        "measurement": report.get("measurement"),
    }
    return (closed, opened), provenance


def _endpoint_summary(samples_mm: np.ndarray, endpoint: str) -> dict[str, Any]:
    samples = np.asarray(samples_mm, dtype=np.float64).reshape(-1)
    samples = samples[np.isfinite(samples)]
    if len(samples) < 3:
        raise ValueError(f"{endpoint} endpoint needs at least three finite width samples")
    median = float(np.median(samples))
    mad = float(np.median(np.abs(samples - median)))
    return {
        "samples": int(len(samples)),
        "median_width_mm": median,
        "mean_width_mm": float(np.mean(samples)),
        "std_width_mm": float(np.std(samples)),
        "mad_width_mm": mad,
        "min_width_mm": float(np.min(samples)),
        "max_width_mm": float(np.max(samples)),
    }
