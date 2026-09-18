from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))

import render_slam_demo as demo  # noqa: E402


def test_nearest_past_indices_uses_master_timeline() -> None:
    source = np.array([10.0, 10.1, 10.2])
    target = np.array([9.9, 10.0, 10.19, 10.4])
    np.testing.assert_array_equal(demo.nearest_past_indices(source, target), [0, 0, 1, 2])


def test_operator_offset_maps_umi_start_into_external_video() -> None:
    assert demo.operator_frame_index(0.0, 2.5, 30.0) == 75
    assert demo.operator_frame_index(1.0, -2.0, 30.0) is None
    assert demo.operator_frame_index(2.0, -2.0, 30.0) == 0


def test_shared_camera_duration_stops_before_shorter_camera_freezes(monkeypatch) -> None:
    streams = {
        "umi1": type("Stream", (), {"t": np.array([10.0, 12.0])})(),
        "umi2": type("Stream", (), {"t": np.array([9.9, 11.5])})(),
    }
    monkeypatch.setattr(demo, "load_frame_stream", lambda _episode, name: streams[name])
    assert demo.shared_camera_duration(Path("episode")) == 1.5


def test_letterbox_has_exact_requested_size() -> None:
    image = np.full((60, 80, 3), 255, dtype=np.uint8)
    framed = demo.letterbox(image, 120, 120)
    assert framed.shape == (120, 120, 3)
    assert framed[15:105].mean() > 240
    assert framed[:10].mean() < 50


def test_isometric_projection_fits_requested_rect() -> None:
    points = np.array(
        [
            [-1.0, -0.5, 0.0],
            [1.0, 0.5, 0.8],
            [0.0, 0.0, 0.2],
        ]
    )
    projector = demo.IsoProjector.fit(points, (50, 70, 700, 500))
    projected = projector.project(points)
    assert np.isfinite(projected).all()
    assert ((projected[:, 0] >= 50) & (projected[:, 0] <= 750)).all()
    assert ((projected[:, 1] >= 70) & (projected[:, 1] <= 570)).all()


def test_slam_panel_is_exact_panel_size() -> None:
    n = 4
    state = np.zeros((n, 6), dtype=np.float32)
    state[:, 2] = np.linspace(0.0, 0.3, n)
    states = {"umi1": state, "umi2": state.copy()}
    positions = np.concatenate([demo._umi_to_display(value[:, :3]) for value in states.values()])
    projector = demo.IsoProjector.fit(positions, (50, 100, 800, 700))
    panel = demo.slam_panel(states, 2, projector, (960, 1080))
    assert panel.shape == (1080, 960, 3)
