from __future__ import annotations

import json

from dataset_filtering.progress_viewer import summarize_progress


def test_progress_categories_include_requested_totals_and_dense_subsets(tmp_path) -> None:
    rows = [
        {"accepted": True, "selected_profile": "baseline_v3", "attempts": []},
        {
            "accepted": True,
            "selected_profile": "dense_objective",
            "attempts": [{"profile": "baseline_v3"}, {"profile": "dense_objective"}],
        },
        {
            "accepted": False,
            "selected_profile": "dense_objective",
            "attempts": [{"profile": "baseline_v3"}, {"profile": "dense_objective"}],
        },
        {
            "accepted": False,
            "selected_profile": "pipeline_error",
            "attempts": [],
            "pipeline_error": "test",
        },
    ]
    for index, row in enumerate(rows):
        row["combined_input_index"] = index
        (tmp_path / f"input-{index:03d}.json").write_text(json.dumps(row))

    status = summarize_progress(tmp_path, total_episodes=10)
    assert status["completed"] == 4
    assert status["passed"] == 2
    assert status["passed_with_dense"] == 1
    assert status["failed_with_dense"] == 1
    assert status["failed"] == 2
    assert status["pipeline_errors"] == 1
    assert status["remaining"] == 6
