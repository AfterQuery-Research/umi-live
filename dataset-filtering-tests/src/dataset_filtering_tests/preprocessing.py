from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from dataset_filtering.models import PoseEpisode
from dataset_filtering.smoothing import detect_gripper_events


@dataclass(frozen=True)
class HeadTrimDecision:
    frame: int
    seconds: float
    first_event_frame: int | None
    event_guard_frames: int
    reason: str


def trim_pose_episode_head(episode: PoseEpisode, frames: int) -> PoseEpisode:
    """Return a copied suffix with timestamps and frame-relative semantics reset."""
    frame = int(np.clip(frames, 0, max(0, len(episode.timestamps) - 2)))
    timestamps = episode.timestamps[frame:].copy()
    timestamps -= timestamps[0]
    return replace(
        episode,
        timestamps=timestamps,
        positions=episode.positions[frame:].copy(),
        quaternions=episode.quaternions[frame:].copy(),
        grippers=None if episode.grippers is None else episode.grippers[frame:].copy(),
    )


def choose_event_safe_head_trim(
    episode: PoseEpisode,
    requested_seconds: float,
    *,
    event_guard_seconds: float = 0.5,
) -> HeadTrimDecision:
    """Clip setup motion without removing the approach window before the first event."""
    requested = max(0, int(round(requested_seconds * episode.fps)))
    guard = max(0, int(round(event_guard_seconds * episode.fps)))
    events = detect_gripper_events(episode.grippers, fps=episode.fps)
    nonempty = [int(values[0]) for values in events if len(values)]
    first_event = min(nonempty) if nonempty else None
    maximum = len(episode.timestamps) - 2
    reason = "requested"
    if first_event is not None:
        guarded_maximum = max(0, first_event - guard)
        if requested > guarded_maximum:
            reason = "clamped_to_preserve_pre_event_approach"
        maximum = min(maximum, guarded_maximum)
    frame = min(requested, maximum)
    return HeadTrimDecision(
        frame=frame,
        seconds=frame / episode.fps,
        first_event_frame=first_event,
        event_guard_frames=guard,
        reason=reason,
    )
