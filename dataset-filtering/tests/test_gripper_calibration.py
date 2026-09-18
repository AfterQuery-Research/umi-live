from __future__ import annotations

import json

import numpy as np
import pytest

from dataset_filtering.gripper_calibration import (
    build_two_point_calibration_report,
    load_single_gripper_calibration,
    write_gripper_calibration,
)


def test_two_point_report_uses_endpoint_medians_and_loads_one_gripper(tmp_path) -> None:
    report = build_two_point_calibration_report(
        gripper="umi1",
        closed_samples_mm=np.asarray([40.0, 40.2, 39.9, 1000.0]),
        open_samples_mm=np.asarray([122.0, 122.2, 121.9, 122.1]),
        measurement={"host": "umi1@umi1.local"},
    )
    path = write_gripper_calibration(report, tmp_path / "umi1.json")

    limits, provenance = load_single_gripper_calibration(path, expected_gripper="umi1")

    np.testing.assert_allclose(limits, [40.1, 122.05])
    assert report["normalized_semantics"] == {"0": "closed", "1": "open"}
    assert provenance["source"] == "interactive_two_point_calibration"
    assert len(provenance["report_sha256"]) == 64


def test_single_gripper_calibration_rejects_wrong_arm(tmp_path) -> None:
    report = build_two_point_calibration_report(
        gripper="umi2",
        closed_samples_mm=np.asarray([40.0, 40.1, 40.2]),
        open_samples_mm=np.asarray([120.0, 120.1, 120.2]),
        measurement={},
    )
    path = tmp_path / "umi2.json"
    path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="expected 'umi1'"):
        load_single_gripper_calibration(path, expected_gripper="umi1")


def test_two_point_report_rejects_reversed_or_tiny_span() -> None:
    with pytest.raises(ValueError, match="must exceed closed width"):
        build_two_point_calibration_report(
            gripper="umi1",
            closed_samples_mm=np.asarray([80.0, 80.1, 79.9]),
            open_samples_mm=np.asarray([81.0, 81.1, 80.9]),
            measurement={},
        )
