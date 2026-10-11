"""Cross-lidar align stage: pure-geometry + parsing coverage (no heavy deps), plus
registration/pose-graph tests gated behind open3d/gtsam (importorskip)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R
from umi_live_core import geom
from umi_live_server import align_stage as A
from umi_live_server.trajectory import PoseTrack


def _track(n: int) -> PoseTrack:
    t = np.arange(float(n))
    pos = np.stack([t, np.zeros(n), np.zeros(n)], axis=1)
    quat = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
    return PoseTrack(t=t, pos=pos, quat=quat)


def test_apply_transform_to_track_matches_manual_compose():
    track = _track(5)
    tq = R.from_euler("z", 90, degrees=True).as_quat()  # xyzw
    tp = np.array([1.0, 2.0, 3.0])
    out = A.apply_transform_to_track(track, (tp, tq))
    # origin pose -> the transform itself; point (1,0,0) rot 90 about z -> (0,1,0)+tp.
    assert np.allclose(out.pos[0], tp)
    assert np.allclose(out.pos[1], [1.0, 3.0, 3.0])
    # timestamps untouched
    assert np.allclose(out.t, track.t)


def test_crop_bbox_keeps_only_inside():
    pts = np.array([[0, 0, 0], [5, 5, 5], [-1, 0, 0], [0.5, 0.5, 0.5]], dtype=float)
    out = A.crop_bbox(pts, (-0.5, -0.5, -0.5, 1.0, 1.0, 1.0))
    assert len(out) == 2
    assert A.crop_bbox(pts, None) is pts  # None => passthrough


def test_se3_mat_roundtrip():
    tq = R.from_euler("xyz", [10, 20, 30], degrees=True).as_quat()
    tp = np.array([0.3, -0.4, 1.2])
    p2, q2 = A._mat_to_se3(A._se3_to_mat((tp, tq)))
    assert np.allclose(p2, tp)
    assert np.allclose(q2, tq) or np.allclose(q2, -tq)


def test_transform_points_matches_se3():
    pts = np.random.default_rng(0).normal(size=(50, 3))
    tq = R.from_euler("z", 45, degrees=True).as_quat()
    tp = np.array([1.0, 0.0, 0.0])
    out = A.transform_points(pts, (tp, tq))
    exp = R.from_quat(tq).apply(pts) + tp
    assert np.allclose(out, exp)


def test_piecewise_correct_identity_is_noop():
    track = _track(6)
    init = [(np.zeros(3), np.array([0.0, 0, 0, 1])) for _ in range(3)]
    stamps = np.array([0.0, 2.0, 4.0])
    out = A.piecewise_correct(track, init, init, stamps, A.IDENTITY)
    assert np.allclose(out.pos, track.pos)
    assert np.allclose(out.quat, track.quat)


def test_piecewise_correct_uniform_correction_equals_rigid():
    track = _track(6)
    init = [(np.zeros(3), np.array([0.0, 0, 0, 1])) for _ in range(3)]
    stamps = np.array([0.0, 2.0, 4.0])
    cq = R.from_euler("z", 30, degrees=True).as_quat()
    cp = np.array([0.5, 0.0, 0.0])
    opt = [geom.pose_multiply(cp, cq, *p) for p in init]
    out = A.piecewise_correct(track, init, opt, stamps, A.IDENTITY)
    exp = A.apply_transform_to_track(track, (cp, cq))
    assert np.allclose(out.pos, exp.pos, atol=1e-9)
    assert np.allclose(np.abs((out.quat * exp.quat).sum(1)), 1.0, atol=1e-9)


def test_read_submaps_parses_poses(tmp_path):
    d = tmp_path / "glim"
    (d / "submaps").mkdir(parents=True)
    (d / "submaps" / "poses.txt").write_text(
        "# id stamp x y z qx qy qz qw\n0 100.5 1 2 3 0 0 0 1\n1 101.5 4 5 6 0 0 0 1\n"
    )
    sm = A.read_submaps(d, load_clouds=False)
    assert np.allclose(sm.stamps, [100.5, 101.5])
    assert np.allclose(sm.poses[0][0], [1, 2, 3])
    assert np.allclose(sm.poses[1][0], [4, 5, 6])
    assert sm.clouds == []


def test_clock_offset_reads_manifest():
    assert A._clock_offset_s({}) == 0.0
    m = {"clock_sync": {"estimated_cross_pi_offset_ns": 500_000_000}}
    assert A._clock_offset_s(m) == pytest.approx(0.5)


def test_match_cross_agent_caps_and_spreads():
    ref = A.Submaps(stamps=np.arange(20.0), poses=[], clouds=[])
    other = A.Submaps(stamps=np.arange(20.0), poses=[], clouds=[])
    pairs = A._match_cross_agent(ref, other, offset_s=0.0, max_pairs=5)
    assert len(pairs) <= 5
    # each ref index maps to the nearest other index (== itself at zero offset)
    assert all(i == j for i, j in pairs)


# --- registration (open3d) -------------------------------------------------------------


def _structured_cloud(rng) -> np.ndarray:
    """Three orthogonal planes (a room corner) — distinctive geometry for GICP."""
    n = 1500
    xy = rng.uniform(0, 3, size=(n, 2))
    floor = np.column_stack([xy[:, 0], xy[:, 1], np.zeros(n)])
    yz = rng.uniform(0, 3, size=(n, 2))
    wall_x = np.column_stack([np.zeros(n), yz[:, 0], yz[:, 1]])
    xz = rng.uniform(0, 3, size=(n, 2))
    wall_y = np.column_stack([xz[:, 0], np.zeros(n), xz[:, 1]])
    return np.vstack([floor, wall_x, wall_y])


def test_gicp_pair_recovers_small_transform():
    pytest.importorskip("open3d")
    rng = np.random.default_rng(1)
    target = _structured_cloud(rng)
    # small known misalignment: 8 deg yaw + 10 cm translation
    gq = R.from_euler("z", 8, degrees=True).as_quat()
    gp = np.array([0.1, -0.05, 0.02])
    source = R.from_quat(gq).apply(target) + gp
    # gicp_pair(source, target, init=identity) -> T(target<-source) ~ inverse of G
    t = A.gicp_pair(source, target, A.IDENTITY, voxel=0.1)
    aligned = A.transform_points(source, t)
    # residual: every aligned source point close to the target it came from
    err = np.linalg.norm(aligned - target, axis=1)
    assert np.median(err) < 0.05


def test_register_clouds_recovers_transform():
    pytest.importorskip("open3d")
    rng = np.random.default_rng(2)
    target = _structured_cloud(rng)
    gq = R.from_euler("z", 25, degrees=True).as_quat()
    gp = np.array([0.4, 0.2, 0.0])
    source = R.from_quat(gq).apply(target) + gp
    t = A.register_clouds(source, target, voxel=0.1)
    aligned = A.transform_points(source, t)
    err = np.linalg.norm(aligned - target, axis=1)
    assert np.median(err) < 0.1
