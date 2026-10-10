from __future__ import annotations

import json
from pathlib import Path

from filter_watcher import active_dataset, status_payload


def test_active_dataset_follows_server_env(tmp_path: Path) -> None:
    repo = tmp_path / "umi-flow"
    repo.mkdir()
    (repo / ".env").write_text(
        'DATA_DIR="data/current-campaign"\nHF_REPO_ID="owner/live"\n', encoding="utf-8"
    )

    dataset, hub_repo = active_dataset(repo)

    assert dataset == (repo / "data/current-campaign/lerobot/local/umi-flow").resolve()
    assert hub_repo == "owner/live"


def test_status_distinguishes_gate_failures_from_pending_and_errors(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    rows = {
        0: {"accepted": True, "selected_profile": "baseline_v3"},
        1: {
            "accepted": False,
            "selected_profile": "dense_objective",
            "selected_result": {
                "validation": {
                    "passes_tracking_accuracy": False,
                    "passes_joint_limits": True,
                }
            },
        },
    }

    payload = status_payload(
        dataset=dataset,
        hub_repo="owner/live",
        episode_indices=[0, 1, 2, 3],
        rows=rows,
        processing={2},
        errors={3: "test error"},
        workers=16,
        code="abc",
    )

    assert payload["passed"] == 1
    assert payload["failed"] == 1
    assert payload["evaluated"] == 2
    assert [row["state"] for row in payload["episodes"]] == [
        "pass",
        "fail",
        "processing",
        "error",
    ]
    assert payload["failed_episode_indices"] == [1]
    assert payload["episodes"][1]["failure_reasons"] == ["tracking_accuracy"]
    json.dumps(payload)
