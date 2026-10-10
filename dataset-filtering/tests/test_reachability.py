from types import SimpleNamespace

import numpy as np
import pytest

from dataset_filtering.reachability import (
    FAILURE_PROGRESS_BIN_COUNT,
    _aggregate_failure_timing,
    _aggregate_reachability,
    _episode_failure_timing,
)


def test_aggregate_reachability_is_frame_weighted() -> None:
    source = SimpleNamespace(source="local", location=SimpleNamespace(revision="main"))
    rows = [
        {
            "frames": 4,
            "duration_s": 2.0,
            "reachable_frames": {
                "umi1_left": 4,
                "umi2_right": 2,
                "both": 2,
                "at_least_one": 4,
            },
            "gripper_event_targets": {
                "umi1_left_total": 2,
                "umi1_left_reachable": 2,
                "umi2_right_total": 1,
                "umi2_right_reachable": 0,
            },
        },
        {
            "frames": 6,
            "duration_s": 3.0,
            "reachable_frames": {
                "umi1_left": 3,
                "umi2_right": 6,
                "both": 3,
                "at_least_one": 6,
            },
            "gripper_event_targets": {
                "umi1_left_total": 1,
                "umi1_left_reachable": 0,
                "umi2_right_total": 2,
                "umi2_right_reachable": 2,
            },
        },
    ]

    report = _aggregate_reachability(source, rows, workers=1)
    reachability = report["frame_weighted_reachability"]
    events = report["gripper_event_reachability"]

    assert reachability["umi1_left_fraction"] == pytest.approx(0.7)
    assert reachability["umi2_right_fraction"] == pytest.approx(0.8)
    assert reachability["individual_arm_target_fraction"] == pytest.approx(0.75)
    assert reachability["both_arms_same_frame_fraction"] == pytest.approx(0.5)
    assert reachability["at_least_one_arm_fraction"] == pytest.approx(1.0)
    assert events["individual_arm_event_fraction"] == pytest.approx(4 / 6)


def test_failure_timing_counts_normalized_progress_and_failure_type() -> None:
    frames = 20
    position_error = np.zeros((frames, 2))
    orientation_error = np.zeros((frames, 2))
    position_error[10, 0] = 0.011
    orientation_error[19, 1] = np.deg2rad(9.0)
    result = SimpleNamespace(
        source_timestamps=np.arange(frames) / 30,
        position_error_m=position_error,
        orientation_error_rad=orientation_error,
    )

    bins = _episode_failure_timing(result)

    assert len(bins) == FAILURE_PROGRESS_BIN_COUNT
    assert bins[10]["failed_frames"] == 1
    assert bins[10]["position_only_failed_frames"] == 1
    assert bins[19]["failed_frames"] == 1
    assert bins[19]["orientation_only_failed_frames"] == 1
    aggregate = _aggregate_failure_timing(
        [{"failure_timing_bins": bins}, {"failure_timing_bins": bins}]
    )
    assert aggregate[10]["failed_frames"] == 2
    assert aggregate[10]["episodes_affected"] == 2
    assert aggregate[10]["failure_rate"] == pytest.approx(1.0)
