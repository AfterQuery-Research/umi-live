from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[2] / "tools"
sys.path.insert(0, str(TOOLS))
import calibrate_camera as calibration  # noqa: E402
import make_charuco as generator  # noqa: E402


def synthetic_views():
    object_points = (
        calibration.make_board().getChessboardCorners().reshape(-1, 1, 3).astype(np.float64)
    )
    expected_k = np.array([[330.0, 0, 400.0], [0, 328.0, 300.0], [0, 0, 1.0]])
    expected_d = np.array([[-0.03], [0.005], [-0.001], [0.0001]])
    ids = np.arange(len(object_points), dtype=np.int32).reshape(-1, 1)
    views = []
    for index in range(24):
        rotation = np.array(
            [
                [-0.35 + 0.7 * ((index % 4) / 3)],
                [-0.4 + 0.8 * (((index // 4) % 3) / 2)],
                [-0.2 + 0.4 * ((index % 5) / 4)],
            ]
        )
        translation = np.array(
            [
                [-0.09 + 0.18 * ((index % 6) / 5)],
                [-0.06 + 0.12 * ((index // 6) / 3)],
                [0.28 + 0.04 * (index % 3)],
            ]
        )
        image_points, _ = cv2.fisheye.projectPoints(
            object_points, rotation, translation, expected_k, expected_d
        )
        views.append((image_points.astype(np.float32), ids.copy()))
    return views, expected_k, expected_d


def test_generated_board_matches_calibrator() -> None:
    assert (generator.SQUARES_X, generator.SQUARES_Y) == (
        calibration.SQUARES_X,
        calibration.SQUARES_Y,
    )
    assert generator.BOARD_DICT == calibration.BOARD_DICT
    assert generator.SQUARE_LEN == calibration.SQUARE_LEN
    assert generator.MARKER_LEN == calibration.MARKER_LEN

    image = generator.generate_board(1600, 1000, 40)
    detector = cv2.aruco.CharucoDetector(calibration.make_board())
    corners, ids, _, _ = detector.detectBoard(image)
    assert corners is not None
    assert ids is not None
    assert len(ids) == (calibration.SQUARES_X - 1) * (calibration.SQUARES_Y - 1)


def test_split_jpegs_preserves_an_incomplete_frame() -> None:
    first = cv2.imencode(".jpg", np.zeros((20, 30, 3), dtype=np.uint8))[1].tobytes()
    second = cv2.imencode(".jpg", np.full((20, 30, 3), 255, dtype=np.uint8))[1].tobytes()

    frames, remainder = calibration.split_jpegs(b"discard me" + first + second[:100])
    assert frames == [first]
    assert remainder == second[:100]
    frames, remainder = calibration.split_jpegs(remainder + second[100:])
    assert frames == [second]
    assert remainder == b""


def test_live_preflight_reports_missing_provisioning(monkeypatch) -> None:
    result = subprocess.CompletedProcess(
        args=[],
        returncode=20,
        stdout="",
        stderr="v4l2-ctl missing; run ./bin/provision-pair first\n",
    )
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return result

    monkeypatch.setattr(calibration.subprocess, "run", fake_run)

    with pytest.raises(SystemExit, match="run ./bin/provision-pair first"):
        calibration.preflight_live_host("umi1@umi1.local")

    assert calls[0][0][:2] == ["ssh", "-o"]
    assert calls[0][0][-2] == "umi1@umi1.local"
    assert calls[0][1]["check"] is False
    remote_command = calls[0][0][-1]
    for mutating_command in ("sudo", "apt-get", "rm ", "mv ", "cp "):
        assert mutating_command not in remote_command


def test_fisheye_solver_writes_the_runtime_contract(tmp_path: Path) -> None:
    views, expected_k, expected_d = synthetic_views()

    output = tmp_path / "camera.npz"
    rms, actual_k, actual_d = calibration.solve_and_save(views, (800, 600), output)

    assert rms < 0.01
    np.testing.assert_allclose(actual_k, expected_k, atol=0.1)
    np.testing.assert_allclose(actual_d, expected_d, atol=1e-4)
    with np.load(output) as saved:
        assert set(saved.files) == {"K", "dist", "image_size", "model"}
        assert saved["image_size"].tolist() == [800, 600]
        assert saved["model"].item() == "fisheye"


def test_solver_does_not_replace_output_when_quality_is_low(tmp_path: Path, monkeypatch) -> None:
    views, _, _ = synthetic_views()
    output = tmp_path / "camera.npz"
    original = b"existing calibration"
    output.write_bytes(original)
    monkeypatch.setattr(
        calibration,
        "turning_point_deg",
        lambda _distortion: calibration.MIN_INVERTIBLE_DEG,
    )

    with pytest.raises(SystemExit, match="existing output was not changed"):
        calibration.solve_and_save(views, (800, 600), output)

    assert output.read_bytes() == original
