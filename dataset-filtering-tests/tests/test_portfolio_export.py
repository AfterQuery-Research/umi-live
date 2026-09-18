from __future__ import annotations

from collections import Counter

from dataset_filtering_tests.portfolio_export import load_portfolio_selections


def test_final_portfolio_resolves_exact_export_profiles() -> None:
    portfolio, selections = load_portfolio_selections()
    assert len(selections) == 65 == portfolio["accepted"]
    assert Counter(value.profile for value in selections) == {
        "baseline_v3": 48,
        "dense_objective": 17,
    }
    assert [value.episode_index for value in selections] == sorted(
        value.episode_index for value in selections
    )

    episode_127 = next(value for value in selections if value.episode_index == 127)
    assert episode_127.profile == "dense_objective"
    assert episode_127.smoothing_method == "butterworth"
    assert episode_127.smoothing_parameters == {"cutoff_hz": 1.5, "order": 3}

    episode_3 = next(value for value in selections if value.episode_index == 3)
    assert episode_3.smoothing_parameters == {"cutoff_hz": 4.0, "order": 3}
