from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from dataset_filtering.models import PoseEpisode
from dataset_filtering.smoothing import detect_gripper_events, smooth_pose_episode


def synthetic_episode(seed: int = 4) -> PoseEpisode:
    rng = np.random.default_rng(seed)
    fps = 30.0
    t = np.arange(300) / fps
    clean = np.stack(
        (
            0.12 * np.sin(0.7 * t),
            0.08 * np.cos(0.5 * t),
            0.04 * np.sin(0.3 * t),
        ),
        axis=1,
    )
    noise = rng.normal(scale=0.0016, size=(len(t), 2, 3))
    positions = np.stack((clean, clean * np.array([1, -1, 1])), axis=1) + noise
    rotvec = np.stack((0.1 * np.sin(0.4 * t), 0.04 * np.cos(0.4 * t), 0.03 * t), axis=1)
    rotvecs = np.stack((rotvec, -rotvec), axis=1) + rng.normal(scale=0.002, size=(len(t), 2, 3))
    quaternions = Rotation.from_rotvec(rotvecs.reshape(-1, 3)).as_quat().reshape(-1, 2, 4)
    grippers = np.full((len(t), 2), 110.0)
    grippers[120:, 0] = 35.0
    grippers[210:, 0] = 110.0
    grippers[170:, 1] = 30.0
    return PoseEpisode(0, fps, t, positions, quaternions, grippers, "synthetic", "test")


def test_gripper_events_and_exact_pose_anchors() -> None:
    episode = synthetic_episode()
    events = detect_gripper_events(episode.grippers, fps=episode.fps)
    assert events[0].tolist() == [120, 210]
    assert events[1].tolist() == [170]
    result = smooth_pose_episode(episode, "gaussian", {"sigma_frames": 2.0})
    for arm in range(2):
        np.testing.assert_allclose(
            result.positions[events[arm], arm], episode.positions[events[arm], arm]
        )
        raw = Rotation.from_quat(episode.quaternions[events[arm], arm])
        filtered = Rotation.from_quat(result.quaternions[events[arm], arm])
        delta = raw.inv() * filtered
        assert delta is not NotImplemented
        assert np.max(delta.magnitude(), initial=0.0) < 1e-10


def test_smoothing_reduces_high_frequency_noise_without_large_drift() -> None:
    result = smooth_pose_episode(synthetic_episode(), "gaussian", {"sigma_frames": 2.0})
    assert result.metrics["translation_hf_remaining_ratio"] < 0.25
    assert result.metrics["rotation_hf_remaining_ratio"] < 0.25
    assert result.metrics["translation_distortion_rms_mm"] < 3.0
    assert result.metrics["direction_cosine_median"] > 0.98
    assert result.metrics["event_translation_error_max_mm"] < 1e-8
