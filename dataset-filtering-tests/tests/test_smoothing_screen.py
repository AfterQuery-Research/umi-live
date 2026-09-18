import json

from dataset_filtering_tests.smoothing_screen import (
    _candidate_id,
    select_safe_smoothing_improvements,
)


def test_candidate_id_is_stable_across_parameter_order() -> None:
    assert _candidate_id("butterworth", {"order": 3, "cutoff_hz": 2.0}) == (
        "butterworth:cutoff_hz=2.0,order=3"
    )


def test_select_safe_smoothing_requires_reachability_improvement(tmp_path) -> None:
    def candidate(cutoff, fraction, safe=True):
        return {
            "method": "butterworth",
            "parameters": {"cutoff_hz": cutoff, "order": 3},
            "score": cutoff,
            "fidelity_guard_pass": safe,
            "all_strict_events_ik_reachable": True,
            "all_frames_both_ik_reachable": fraction == 1.0,
            "both_frame_fraction": fraction,
            "metrics": {"translation_distortion_p95_mm": 2.0},
        }

    baseline_id = "butterworth:cutoff_hz=4.0,order=3"
    report = {
        "per_episode": [
            {
                "episode_index": 8,
                "candidates": {
                    baseline_id: candidate(4.0, 0.8),
                    "butterworth:cutoff_hz=2.0,order=3": candidate(2.0, 0.9),
                    "butterworth:cutoff_hz=1.0,order=3": candidate(1.0, 1.0, safe=False),
                },
            }
        ]
    }
    path = tmp_path / "screen.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    assert select_safe_smoothing_improvements(path) == {
        8: ("butterworth", {"cutoff_hz": 2.0, "order": 3})
    }
