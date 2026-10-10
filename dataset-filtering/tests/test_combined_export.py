from __future__ import annotations

import numpy as np
import pyarrow as pa

from dataset_filtering.combined_export import (
    CARTESIAN_STATE_NAMES,
    YAM_STATE_NAMES,
    _cartesian_observation_state,
    _combined_info,
    _exact_numeric_stats,
    _gripper_normalization_metadata,
    _normalize_gripper_widths,
    _resolve_gripper_calibrations,
    _rewrite_combined_episode_table,
)
from dataset_filtering.gripper_calibration import (
    build_two_point_calibration_report,
    write_gripper_calibration,
)


def _source_table(frames: int = 3) -> pa.Table:
    state = pa.FixedSizeListArray.from_arrays(pa.array(np.zeros(frames * 12, dtype=np.float32)), 12)
    action = pa.FixedSizeListArray.from_arrays(
        pa.array(np.zeros(frames * 12, dtype=np.float32)), 12
    )
    return pa.table(
        {
            "observation.state": state,
            "action": action,
            "observation.gripper_width.umi1": pa.array(np.arange(frames, dtype=np.float32)),
            "observation.gripper_width.umi2": pa.array(np.arange(frames, dtype=np.float32) + 10),
            "timestamp": pa.array(np.arange(frames, dtype=np.float32) / 30),
            "frame_index": pa.array(np.arange(frames, dtype=np.int64)),
            "episode_index": pa.array(np.full(frames, 99, dtype=np.int64)),
            "index": pa.array(np.arange(frames, dtype=np.int64) + 500),
            "task_index": pa.array(np.zeros(frames, dtype=np.int64)),
        }
    )


def test_combined_episode_tables_are_observation_only_and_reindexed() -> None:
    source = _source_table()
    cartesian = _rewrite_combined_episode_table(
        source,
        np.zeros((3, 14), dtype=np.float32),
        variant="cartesian",
        output_episode_index=4,
        global_from_index=100,
    )
    joint = _rewrite_combined_episode_table(
        source,
        np.zeros((3, 14), dtype=np.float32),
        variant="joint",
        output_episode_index=4,
        global_from_index=100,
    )

    assert "action" not in cartesian.column_names
    assert "action" not in joint.column_names
    assert "observation.gripper_width.umi1" not in cartesian.column_names
    assert "observation.gripper_width.umi1" not in joint.column_names
    assert cartesian.schema.field("observation.state").type == pa.list_(pa.float32(), 14)
    assert joint.schema.field("observation.state").type == pa.list_(pa.float32(), 14)
    np.testing.assert_array_equal(cartesian["episode_index"].to_numpy(), [4, 4, 4])
    np.testing.assert_array_equal(joint["index"].to_numpy(), [100, 101, 102])


def test_joint_info_matches_yam_state_layout_without_actions() -> None:
    source_info = {
        "features": {
            "observation.state": {"dtype": "float32", "shape": [12], "names": []},
            "action": {"dtype": "float32", "shape": [12], "names": []},
            "observation.gripper_width.umi1": {"dtype": "float32", "shape": [1]},
            "observation.gripper_width.umi2": {"dtype": "float32", "shape": [1]},
        },
        "robot_type": None,
    }
    info = _combined_info(source_info, variant="joint", episodes=8, frames=123)
    assert info["features"]["observation.state"]["shape"] == [14]
    assert info["features"]["observation.state"]["names"] == list(YAM_STATE_NAMES)
    assert "action" not in info["features"]
    assert "observation.gripper_width.umi1" not in info["features"]
    assert info["robot_type"] == "bi_yam_follower"
    assert info["splits"] == {"train": "0:8"}


def test_cartesian_info_embeds_normalized_grippers_in_state() -> None:
    source_info = {
        "features": {
            "observation.state": {"dtype": "float32", "shape": [12], "names": []},
            "action": {"dtype": "float32", "shape": [12], "names": []},
            "observation.gripper_width.umi1": {"dtype": "float32", "shape": [1]},
            "observation.gripper_width.umi2": {"dtype": "float32", "shape": [1]},
        }
    }

    info = _combined_info(source_info, variant="cartesian", episodes=2, frames=20)

    assert info["features"]["observation.state"]["shape"] == [14]
    assert info["features"]["observation.state"]["names"] == list(CARTESIAN_STATE_NAMES)
    assert "observation.gripper_width.umi1" not in info["features"]
    assert "observation.gripper_width.umi2" not in info["features"]


def test_grippers_use_per_arm_calibration_and_enter_both_state_layouts() -> None:
    widths = np.asarray([[40.0, 50.0], [60.0, 80.0], [100.0, 110.0]])
    limits = np.asarray([[40.0, 100.0], [50.0, 110.0]])
    normalized = _normalize_gripper_widths(widths, limits)
    pose = np.arange(36, dtype=np.float64).reshape(3, 12)

    np.testing.assert_allclose(normalized, [[0.0, 0.0], [1 / 3, 0.5], [1.0, 1.0]])
    cartesian = _cartesian_observation_state(pose, normalized)
    np.testing.assert_allclose(cartesian[:, 6], normalized[:, 0])
    np.testing.assert_allclose(cartesian[:, 13], normalized[:, 1])
    np.testing.assert_allclose(cartesian[:, :6], pose[:, :6])
    np.testing.assert_allclose(cartesian[:, 7:13], pose[:, 6:])

    metadata = _gripper_normalization_metadata(limits)
    assert metadata["method"] == "per_arm_calibrated_physical_min_max"
    assert metadata["normalized_semantics"] == {"0": "closed", "1": "open"}
    assert metadata["umi1"] == {"min_width_mm": 40.0, "max_width_mm": 100.0}
    assert metadata["umi2"] == {"min_width_mm": 50.0, "max_width_mm": 110.0}
    assert metadata["state_indices"] == {"umi1": 6, "umi2": 13}
    np.testing.assert_array_equal(
        _normalize_gripper_widths(np.asarray([[0.0, 200.0]]), limits), [[0.0, 1.0]]
    )


def test_future_export_can_load_one_arm_calibration_and_fallback_for_other(tmp_path) -> None:
    report = build_two_point_calibration_report(
        gripper="umi1",
        closed_samples_mm=np.asarray([41.0, 41.1, 40.9]),
        open_samples_mm=np.asarray([121.0, 121.1, 120.9]),
        measurement={},
    )
    path = write_gripper_calibration(report, tmp_path / "umi1.json")

    limits, provenance = _resolve_gripper_calibrations(path, None)

    np.testing.assert_allclose(limits[0], [41.0, 121.0])
    assert provenance["umi1"]["source"] == "interactive_two_point_calibration"
    assert provenance["umi2"]["source"] == "built_in_i2rt_linear_4310_calibration"


def test_exact_numeric_stats_use_all_rows_not_episode_quantiles() -> None:
    episodes = [
        np.asarray([[0.0], [1.0], [2.0]]),
        np.asarray([[100.0], [101.0]]),
    ]

    stats = _exact_numeric_stats({"observation.state": episodes})["observation.state"]
    values = np.concatenate(episodes, axis=0)

    np.testing.assert_allclose(stats["q01"], np.quantile(values, 0.01, axis=0))
    np.testing.assert_allclose(stats["q99"], np.quantile(values, 0.99, axis=0))
    np.testing.assert_array_equal(stats["count"], [5])
