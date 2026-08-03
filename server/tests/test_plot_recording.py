"""Offline pose-plot video (mac/plot_recording.py): pure data/transform functions.

Rendering itself is matplotlib-gated (a viz-only dep) and validated by running the tool;
here we cover the EE transform + TUM parsing + episode discovery, which need only numpy."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parents[2] / "mac" / "plot_recording.py"
_SPEC = importlib.util.spec_from_file_location("plot_recording", _PATH)
P = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(P)


def _tum(path: Path, rows: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rows)


def test_ee_from_pose_identity_is_lever():
    ee, head = P.ee_from_pose(np.zeros(3), np.array([0.0, 0, 0, 1]))
    assert np.allclose(ee, P.LIDAR_EE_LEVER_M)
    assert np.allclose(head, [0, 0, 1])


def test_ee_from_pose_remaps_glim_to_umi_axes():
    # glim +x -> umi (up, right, fwd) = M_UMI @ x, plus the lever.
    ee, _ = P.ee_from_pose(np.array([1.0, 0, 0]), np.array([0.0, 0, 0, 1]))
    assert np.allclose(ee, P.M_UMI @ np.array([1.0, 0, 0]) + P.LIDAR_EE_LEVER_M)


def test_ee_track_reads_and_sorts(tmp_path):
    tum = tmp_path / "t.tum"
    _tum(tum, "# hdr\n1 1 0 0 0 0 0 1\n0 0 0 0 0 0 0 1\n")
    t, ee, head = P.ee_track(tum)
    assert np.allclose(t, [0, 1])  # sorted by timestamp
    assert ee.shape == (2, 3) and head.shape == (2, 3)


def test_discover_prefers_align_when_present(tmp_path):
    for u in ("umi1", "umi2"):
        _tum(tmp_path / u / "glim" / "trajectory.tum", "0 0 0 0 0 0 0 1\n")
    srcs, joint = P.discover_sources(tmp_path, "auto")
    assert [s[0] for s in srcs] == ["umi1", "umi2"]
    assert joint is False and "glim" in str(srcs[0][1])

    for u in ("umi1", "umi2"):
        _tum(tmp_path / "align" / u / "trajectory.tum", "0 0 0 0 0 0 0 1\n")
    srcs, joint = P.discover_sources(tmp_path, "auto")
    assert joint is True and "align" in str(srcs[0][1])

    srcs, joint = P.discover_sources(tmp_path, "off")  # force independent
    assert joint is False and "glim" in str(srcs[0][1])


def test_discover_joint_on_requires_align(tmp_path):
    for u in ("umi1", "umi2"):
        _tum(tmp_path / u / "glim" / "trajectory.tum", "0 0 0 0 0 0 0 1\n")
    with pytest.raises(SystemExit):
        P.discover_sources(tmp_path, "on")


def test_discover_no_trajectories_errors(tmp_path):
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit):
        P.discover_sources(tmp_path, "auto")
