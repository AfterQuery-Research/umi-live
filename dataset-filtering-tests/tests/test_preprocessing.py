from __future__ import annotations

import numpy as np
from dataset_filtering.models import PoseEpisode
from scipy.spatial.transform import Rotation

from dataset_filtering_tests.preprocessing import (
    choose_event_safe_head_trim,
    trim_pose_episode_head,
)


def _episode() -> PoseEpisode:
    frames = 120
    grippers = np.full((frames, 2), 80.0)
    grippers[60:, 0] = 20.0
    return PoseEpisode(
        episode_index=1,
        fps=30.0,
        timestamps=np.arange(frames) / 30.0,
        positions=np.zeros((frames, 2, 3)),
        quaternions=np.tile(Rotation.identity().as_quat(), (frames, 2, 1)),
        grippers=grippers,
        task="test",
        source="local",
    )


def test_head_trim_preserves_half_second_event_approach() -> None:
    episode = _episode()
    decision = choose_event_safe_head_trim(episode, 3.0, event_guard_seconds=0.5)

    assert decision.frame == 45
    assert decision.seconds == 1.5
    assert decision.reason == "clamped_to_preserve_pre_event_approach"

    trimmed = trim_pose_episode_head(episode, decision.frame)
    assert len(trimmed.timestamps) == 75
    assert trimmed.timestamps[0] == 0.0
    assert trimmed.grippers is not None
    assert trimmed.grippers[15, 0] == 20.0
    assert not np.shares_memory(trimmed.positions, episode.positions)
