"""Trajectory math: TUM IO, resampling, episode-relative states, actions."""

from __future__ import annotations

import numpy as np
import pytest
from robot_umi_server.trajectory import (
    PoseTrack,
    actions_from_states,
    read_tum,
    resample,
    states_from_track,
    write_tum,
)
from scipy.spatial.transform import Rotation


def straight_track(n: int = 11, t0: float = 100.0) -> PoseTrack:
    t = t0 + np.arange(n) * 0.1
    pos = np.stack([np.linspace(0, 1, n), np.zeros(n), np.zeros(n)], axis=1)
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
    return PoseTrack(t=t, pos=pos, quat=quat)


def test_tum_roundtrip(tmp_path):
    track = straight_track()
    path = tmp_path / "trajectory.tum"
    write_tum(path, track)
    back = read_tum(path)
    assert np.allclose(back.t, track.t)
    assert np.allclose(back.pos, track.pos)
    assert np.allclose(back.quat, track.quat)


def test_read_tum_sorts_and_rejects_malformed(tmp_path):
    p = tmp_path / "t.tum"
    p.write_text("2.0 0 0 0 0 0 0 1\n1.0 1 1 1 0 0 0 1\n")
    track = read_tum(p)
    assert track.t[0] == 1.0
    p.write_text("1.0 0 0\n")
    with pytest.raises(ValueError, match="malformed"):
        read_tum(p)


def test_resample_interpolates_and_clamps():
    track = straight_track()  # x goes 0 -> 1 over t in [100, 101]
    frames = np.array([99.5, 100.05, 100.5, 101.5])  # before, mid, mid, after
    out = resample(track, frames)
    assert out.pos[0, 0] == pytest.approx(0.0)  # clamped to start
    assert out.pos[1, 0] == pytest.approx(0.05, abs=1e-9)
    assert out.pos[2, 0] == pytest.approx(0.5, abs=1e-9)
    assert out.pos[3, 0] == pytest.approx(1.0)  # clamped to end


def test_resample_slerps_rotation():
    t = np.array([0.0, 1.0])
    quat = np.stack(
        [
            Rotation.identity().as_quat(),
            Rotation.from_euler("z", 90, degrees=True).as_quat(),
        ]
    )
    track = PoseTrack(t=t, pos=np.zeros((2, 3)), quat=quat)
    out = resample(track, np.array([0.5]))
    angle = Rotation.from_quat(out.quat[0]).magnitude()
    assert np.degrees(angle) == pytest.approx(45.0, abs=1e-6)


def test_states_zeroed_to_first_pose():
    track = straight_track()
    states = states_from_track(track)
    assert states.shape == (11, 6)
    assert np.allclose(states[0], 0.0)
    assert states[-1, 0] == pytest.approx(1.0)


def test_states_shared_origin():
    track = straight_track()
    ref_pos = np.array([0.5, 0.0, 0.0])
    ref_quat = np.array([0.0, 0.0, 0.0, 1.0])
    states = states_from_track(track, ref_pos, ref_quat)
    assert states[0, 0] == pytest.approx(-0.5)  # expressed in the shared frame


def test_states_basis_remap_umi_convention():
    from robot_umi_core import geom

    # GLIM-frame motions: forward=-y, right=-x, up=+z (Airy mount, recalibrated
    # 2026-07-16 after the factory IMU extrinsic fix).
    t = np.arange(4) * 0.1
    pos = np.array([[0, 0, 0], [0, -1, 0], [-1, -1, 0], [-1, -1, 1]], dtype=np.float64)
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (4, 1))
    track = PoseTrack(t=t, pos=pos, quat=quat)
    states = states_from_track(track, basis=geom.UMI_FROM_LIDAR_BODY)
    # forward (-y) -> +Z ; right (-x) -> +Y ; up (+z) -> +X
    assert states[1, :3] == pytest.approx([0, 0, 1])   # moved forward
    assert states[2, :3] == pytest.approx([0, 1, 1])   # then right
    assert states[3, :3] == pytest.approx([1, 1, 1])   # then up


def test_actions_are_deltas_with_zero_tail():
    states = np.array([[0.0] * 6, [1.0] * 6, [3.0] * 6], dtype=np.float32)
    actions = actions_from_states(states)
    assert np.allclose(actions[0], 1.0)
    assert np.allclose(actions[1], 2.0)
    assert np.allclose(actions[2], 0.0)
