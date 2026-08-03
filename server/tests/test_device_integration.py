"""End-to-end: device records -> tar -> ingest -> runner -> LeRobot dataset.

The device side is the REAL recorder (mock lidar pcap + cv2 camera); the ingest is
the real extract/validate; the runner is the real state machine. Only decode/glim
are faked — glim's fake emits a genuine TUM trajectory spanning the camera frame
timestamps, so the REAL dataset stage exercises resampling, shared-origin states,
and the LeRobot append.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from robot_umi_server.dataset_stage import run as dataset_run
from robot_umi_server.runner import Executor, Runner, default_stages
from robot_umi_server.server import ServerConfig, create_app
from robot_umi_server.trajectory import PoseTrack, write_tum

from tests.helpers import frame_unix_times, make_episode_tar, make_v2_episode

pytest.importorskip("lerobot", reason="dataset stage needs lerobot")


def fake_glim_from_frames(episode_dir: Path):
    """A fake SLAM engine: straight-line trajectory covering the frame span."""

    def fn(_episode_dir: Path, out_dir: Path, scope: str) -> None:
        ts = np.asarray(frame_unix_times(episode_dir, "umi1"), dtype=np.float64) / 1e9
        n = len(ts)
        pos = np.stack([np.linspace(0.0, 0.5, n), np.zeros(n), np.zeros(n)], axis=1)
        quat = np.tile([0.0, 0.0, 0.0, 1.0], (n, 1))
        write_tum(out_dir / "trajectory.tum", PoseTrack(t=ts, pos=pos, quat=quat))

    return fn


def test_device_to_lerobot_dataset(tmp_path: Path, monkeypatch):
    # 1. Device records a real episode (mock lidar pcap + real MJPEG).
    episode = make_v2_episode(tmp_path, n_umis=1, frames=6)

    # 2. Upload through the real ingest.
    data_dir = tmp_path / "data"
    tar_path = make_episode_tar(episode, tmp_path / "out")
    app = create_app(ServerConfig(data_dir=data_dir))
    with TestClient(app) as client:
        r = client.post(
            "/upload",
            content=tar_path.read_bytes(),
            headers={"X-Episode-Id": episode.name},
        )
        assert r.status_code == 202
        worker = app.state.worker
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not worker.snapshot_results():
            time.sleep(0.02)
        assert worker.snapshot_results()[0].success
    published = data_dir / "raw_episodes" / episode.name

    # 3. Pipeline: fake decode/glim, REAL dataset stage.
    monkeypatch.setenv("ROBOT_UMI_REPO_ID", "test/umi-e2e")
    monkeypatch.setenv("ROBOT_UMI_DATASET_ROOT", str(tmp_path / "lerobot"))
    monkeypatch.setenv("ROBOT_UMI_PUSH_TO_HUB", "0")

    stages = default_stages(
        dataset_fn=lambda ep, out, scope: dataset_run(ep, out, scope),
        gripper_fn=None,
    )
    for s in stages:
        if s.fn is None:
            s.fn = (
                fake_glim_from_frames(published)
                if s.name == "glim"
                else lambda ep, out, scope: None  # decode: contents unused by fakes
            )
    runner = Runner(data_dir=data_dir, stages=stages, executor=Executor(data_dir))
    executed = runner.scan_once()
    assert executed == 3, f"failed stages: {runner.failed}"

    # 4. The LeRobot dataset exists with the derived schema and real states.
    receipt = json.loads(
        (data_dir / "derived" / episode.name / "dataset" / "receipt.json").read_text()
    )
    assert receipt["frames"] == 6
    assert receipt["state_dim"] == 6

    ds_root = tmp_path / "lerobot" / "test/umi-e2e"
    info = json.loads((ds_root / "meta" / "info.json").read_text())
    assert info["fps"] == 30
    assert tuple(info["features"]["observation.state"]["shape"]) == (6,)
    assert info["features"]["observation.state"]["names"][0] == "umi1_x"
    assert "observation.images.umi1" in info["features"]
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 6

    # States follow the fake trajectory: zeroed at frame 0, +0.5 m in GLIM-x at
    # the end — which the UMI body convention (forward=+Z, right=+Y, up=+X;
    # geom.UMI_FROM_LIDAR_BODY) reports as Y = -x.
    import pandas as pd

    parquets = sorted(ds_root.glob("data/**/*.parquet"))
    assert parquets
    df = pd.concat([pd.read_parquet(p) for p in parquets])
    states = np.stack(df["observation.state"].to_numpy())
    assert states.shape == (6, 6)
    assert np.allclose(states[0], 0.0, atol=1e-6)
    assert states[-1, 1] == pytest.approx(-0.5, abs=1e-6)
    assert states[-1, 0] == pytest.approx(0.0, abs=1e-6)
    assert states[-1, 2] == pytest.approx(0.0, abs=1e-6)
