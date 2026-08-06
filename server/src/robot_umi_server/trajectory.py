"""GLIM trajectory interchange, resampling, and per-frame states.

The one interchange contract between SLAM engines and the dataset stage:
GLIM exports ``trajectory.tum`` (``t x y z qx qy qz qw``, t = unix seconds
on the episode's PTP axis). The dataset stage consumes that and resamples onto
camera frame timestamps with linear position interpolation and
quaternion Slerp — the proven pattern from the previous rig.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


@dataclass(frozen=True)
class PoseTrack:
    """Timestamped world poses: t (unix seconds, float64), pos (N,3), quat_xyzw (N,4)."""

    t: np.ndarray
    pos: np.ndarray
    quat: np.ndarray

    def __len__(self) -> int:
        return len(self.t)


def read_tum(path: Path) -> PoseTrack:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 8:
            raise ValueError(f"{path}: malformed TUM line: {line!r}")
        rows.append([float(x) for x in parts])
    if not rows:
        raise ValueError(f"{path}: empty trajectory")
    arr = np.asarray(rows, dtype=np.float64)
    order = np.argsort(arr[:, 0], kind="stable")
    arr = arr[order]
    return PoseTrack(t=arr[:, 0], pos=arr[:, 1:4], quat=arr[:, 4:8])


def write_tum(path: Path, track: PoseTrack) -> None:
    with path.open("w") as f:
        f.write("# t x y z qx qy qz qw\n")
        for i in range(len(track)):
            f.write(
                f"{track.t[i]:.9f} "
                + " ".join(f"{v:.9f}" for v in track.pos[i])
                + " "
                + " ".join(f"{v:.9f}" for v in track.quat[i])
                + "\n"
            )


def _dedup_increasing(track: PoseTrack) -> PoseTrack:
    keep = np.concatenate([[True], np.diff(track.t) > 0])
    return PoseTrack(t=track.t[keep], pos=track.pos[keep], quat=track.quat[keep])


def resample(track: PoseTrack, frame_t: np.ndarray) -> PoseTrack:
    """Sample the track at camera frame times: interp1d position + Slerp rotation.

    Frame times outside the track's span are clamped to the endpoints (no
    extrapolation — the proven behavior from the previous pipeline).
    """
    track = _dedup_increasing(track)
    if len(track) == 1:
        n = len(frame_t)
        return PoseTrack(
            t=np.asarray(frame_t, dtype=np.float64),
            pos=np.repeat(track.pos, n, axis=0),
            quat=np.repeat(track.quat, n, axis=0),
        )
    clamped = np.clip(frame_t, track.t[0], track.t[-1])
    pos_f = interp1d(track.t, track.pos, axis=0, assume_sorted=True)
    slerp = Slerp(track.t, Rotation.from_quat(track.quat))
    return PoseTrack(
        t=np.asarray(frame_t, dtype=np.float64),
        pos=pos_f(clamped),
        quat=slerp(clamped).as_quat(),
    )


def states_from_track(
    track: PoseTrack,
    ref_pos: np.ndarray | None = None,
    ref_quat: np.ndarray | None = None,
    basis: np.ndarray | None = None,
) -> np.ndarray:
    """Episode-relative (N, 6) float32 state: position + rotation-vector.

    Zeros to the track's own first pose by default; pass ``ref_pos``/``ref_quat``
    to zero to a SHARED origin (2-UMI rigs: both arms use umi1's first pose so
    umi2 is expressed in umi1's start frame). ``basis`` re-expresses the
    episode-relative state in another body convention (e.g.
    ``geom.UMI_FROM_LIDAR_BODY`` for lidar tracks: forward=+Z, right=+Y, up=+X);
    positions map as ``basis @ p`` and rotation-vectors likewise (conjugation).
    """
    if len(track) == 0:
        return np.zeros((0, 6), dtype=np.float32)
    p0 = track.pos[0] if ref_pos is None else np.asarray(ref_pos, dtype=np.float64)
    q0 = track.quat[0] if ref_quat is None else np.asarray(ref_quat, dtype=np.float64)
    r0_inv = Rotation.from_quat(q0).inv()
    rel_pos = r0_inv.apply(track.pos - p0)
    rel_rot = (r0_inv * Rotation.from_quat(track.quat)).as_rotvec()
    if basis is not None:
        rel_pos = rel_pos @ basis.T
        rel_rot = rel_rot @ basis.T
    return np.concatenate([rel_pos, rel_rot], axis=1).astype(np.float32)


def actions_from_states(states: np.ndarray) -> np.ndarray:
    """action[i] = state[i+1] - state[i]; the final action is zero (episode end)."""
    if len(states) == 0:
        return states.copy()
    actions = np.zeros_like(states)
    actions[:-1] = states[1:] - states[:-1]
    return actions
