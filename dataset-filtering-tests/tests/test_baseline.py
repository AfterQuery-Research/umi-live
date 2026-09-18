from __future__ import annotations

from dataset_filtering_tests.baseline import load_baseline, summarize_baseline


def test_baseline_is_the_verified_v3_result() -> None:
    summary = summarize_baseline(load_baseline())

    assert summary["episodes"] == 150
    assert summary["kept"] == 48
    assert summary["additional_episodes_needed"] == 57
    assert summary["gate_pass_counts"]["tracking_accuracy"] == 65
    assert summary["invariants"]["gate_tuning_to_hit_target_forbidden"]
