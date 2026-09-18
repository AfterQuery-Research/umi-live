import json

from dataset_filtering_tests.portfolio import build_solver_portfolio


def test_portfolio_preserves_passing_baseline_and_adds_only_passing_candidate(
    tmp_path, monkeypatch
) -> None:
    baseline = {
        "episodes": 150,
        "per_episode": [
            _baseline_row(1, kept=True),
            _baseline_row(2, kept=False),
            *[_baseline_row(index, kept=False) for index in range(3, 151)],
        ],
    }
    experiment = {
        "per_episode": [
            _experiment_row(1, kept=False),
            _experiment_row(2, kept=True),
        ]
    }
    baseline_path = tmp_path / "baseline.json"
    experiment_path = tmp_path / "experiment.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    experiment_path.write_text(json.dumps(experiment), encoding="utf-8")
    monkeypatch.setattr(
        "dataset_filtering_tests.portfolio.load_baseline", lambda _: baseline
    )

    result = build_solver_portfolio(
        baseline_report=baseline_path,
        experiments=[(experiment_path, "candidate")],
    )

    assert result["accepted"] == 2
    assert result["per_episode"][0]["selected_profile"] == "baseline_v3"
    assert result["per_episode"][1]["selected_profile"] == "candidate"


def _validation(kept: bool) -> dict[str, object]:
    return {
        "passes_tracking_accuracy": kept,
        "passes_contact_accuracy": True,
        "passes_interarm_clearance": True,
        "passes_self_collision_clearance": True,
        "minimum_interarm_surface_clearance_mm": 10.0,
        "minimum_self_surface_clearance_mm": 2.0,
    }


def _baseline_row(index: int, *, kept: bool) -> dict[str, object]:
    return {
        "episode_index": index,
        "continuously_replayable": kept,
        "position_error_max_mm": 1.0,
        "orientation_error_max_deg": 1.0,
        "flow_direction_cosine_median": 1.0,
        "validation": _validation(kept),
    }


def _experiment_row(index: int, *, kept: bool) -> dict[str, object]:
    return {
        "episode_index": index,
        "profiles": {
            "candidate": {
                "continuously_replayable": kept,
                "position_error_max_mm": 1.0,
                "orientation_error_max_deg": 1.0,
                "flow_direction_cosine_median": 1.0,
                "validation": _validation(kept),
            }
        },
    }
