"""Dataset stage: append-ledger idempotence + align/glim trajectory selection
(no lerobot/cv2 needed here)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest
from umi_flow_core import config as core_config
from umi_flow_server.dataset_stage import (
    _ENSURED_HUB_REPOS,
    _ensure_hub_repo,
    _reconcile_unpushed_receipts,
    _SequentialVideoReader,
    _wait_for_video_encoder,
    already_appended,
    mark_appended,
    normalize_gripper_widths,
    pose_track_for_umi,
)
from umi_flow_server.trajectory import PoseTrack, write_tum


def _fake_encoder():
    queue = Queue(maxsize=1)
    return SimpleNamespace(
        _frame_queues={"camera": queue},
        _threads={"camera": SimpleNamespace(is_alive=lambda: True)},
        _dropped_frames={},
    )


def test_video_encoder_waits_for_capacity_without_growing_queue(monkeypatch):
    from umi_flow_server import dataset_stage

    encoder = _fake_encoder()
    queue = encoder._frame_queues["camera"]
    queue.put("previous frame")
    waits = []

    def consume(_seconds):
        waits.append(True)
        assert queue.full()
        queue.get_nowait()

    monkeypatch.setattr(dataset_stage.time, "sleep", consume)
    _wait_for_video_encoder(encoder)
    assert waits == [True]
    assert queue.empty()
    assert queue.maxsize == 1


def test_video_encoder_fails_if_stalled():
    encoder = _fake_encoder()
    encoder._frame_queues["camera"].put("previous frame")
    with pytest.raises(RuntimeError, match="stalled"):
        _wait_for_video_encoder(encoder, timeout_s=0)


def test_video_encoder_fails_if_worker_dies():
    encoder = _fake_encoder()
    encoder._threads["camera"].is_alive = lambda: False
    with pytest.raises(RuntimeError, match="stopped"):
        _wait_for_video_encoder(encoder)


def test_video_encoder_refuses_any_dropped_frames():
    encoder = _fake_encoder()
    encoder._dropped_frames["camera"] = 1
    with pytest.raises(RuntimeError, match="dropped video frames"):
        _wait_for_video_encoder(encoder)


def test_append_ledger_records_and_detects_episodes(tmp_path: Path):
    dataset_dir = tmp_path / "org" / "umi-dataset"  # repo_id can be nested

    assert not already_appended(dataset_dir, "episode_1")
    mark_appended(dataset_dir, "episode_1")
    assert already_appended(dataset_dir, "episode_1")
    assert not already_appended(dataset_dir, "episode_2")

    mark_appended(dataset_dir, "episode_2")
    assert already_appended(dataset_dir, "episode_1")
    assert already_appended(dataset_dir, "episode_2")

    ledger = (dataset_dir / ".umi_appended_episodes.txt").read_text().split()
    assert ledger == ["episode_1", "episode_2"]


def test_append_ledger_is_substring_safe(tmp_path: Path):
    # "ep" must not match "episode_10"; membership is per-line, not substring.
    dataset_dir = tmp_path / "ds"
    mark_appended(dataset_dir, "episode_10")
    assert already_appended(dataset_dir, "episode_10")
    assert not already_appended(dataset_dir, "episode_1")
    assert not already_appended(dataset_dir, "ep")


class _FakeHubApi:
    def __init__(self, *, corrupt: bool = False, repo_exists: bool = False):
        self.corrupt = corrupt
        self._repo_exists = repo_exists
        self.created = 0
        self.exists_checks = 0
        self.uploaded: list[str] = []
        self.raw_dirs: dict[str, Path] = {}

    def repo_exists(self, **_kwargs):
        self.exists_checks += 1
        return self._repo_exists

    def create_repo(self, **_kwargs):
        self.created += 1

    def upload_folder(self, *, folder_path: str, path_in_repo: str, **_kwargs):
        self.uploaded.append(path_in_repo)
        self.raw_dirs[path_in_repo] = Path(folder_path)
        return SimpleNamespace(oid=f"revision-{len(self.uploaded)}")

    def list_repo_tree(self, *, path_in_repo: str, **_kwargs):
        raw_dir = self.raw_dirs[path_in_repo]
        result = []
        for path in raw_dir.iterdir():
            if not path.is_file():
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if self.corrupt and path.name == "umi1_lidar.pcap":
                digest = "0" * 64
            result.append(
                SimpleNamespace(
                    path=f"{path_in_repo}/{path.name}",
                    size=path.stat().st_size,
                    lfs=SimpleNamespace(sha256=digest),
                )
            )
        return result


def _stale_publication(tmp_path: Path, episode_id: str = "paired_sess_1") -> tuple[Path, Path]:
    data_dir = tmp_path / "data"
    dataset_dir = data_dir / "lerobot" / "local" / "umi-flow"
    mark_appended(dataset_dir, episode_id)
    raw_dir = data_dir / "raw_episodes" / episode_id
    raw_dir.mkdir(parents=True)
    for name in (
        "umi1_camera.mjpeg.avi",
        "umi1_lidar.pcap",
        "umi2_camera.mjpeg.avi",
        "umi2_lidar.pcap",
    ):
        (raw_dir / name).write_bytes(f"payload:{name}".encode())
    receipt_dir = data_dir / "derived" / episode_id / "dataset"
    receipt_dir.mkdir(parents=True)
    (receipt_dir / "COMPLETE").write_text("complete\n")
    receipt_path = receipt_dir / "receipt.json"
    receipt_path.write_text(json.dumps({"episode_id": episode_id, "pushed": False}))
    return data_dir, receipt_path


def test_hub_repo_is_ensured_once_per_process():
    api = _FakeHubApi()
    _ENSURED_HUB_REPOS.clear()
    _ensure_hub_repo(api, "org/dataset")
    _ensure_hub_repo(api, "org/dataset")
    assert api.created == 1
    assert api.exists_checks == 1


def test_existing_hub_repo_does_not_call_create_endpoint():
    api = _FakeHubApi(repo_exists=True)
    _ENSURED_HUB_REPOS.clear()
    _ensure_hub_repo(api, "org/existing")
    assert api.created == 0
    assert api.exists_checks == 1


def test_reconcile_reuploads_and_exactly_verifies_stale_raw(tmp_path: Path):
    data_dir, receipt_path = _stale_publication(tmp_path)
    api = _FakeHubApi()

    repaired, failed = _reconcile_unpushed_receipts(api, data_dir, "org/dataset")

    receipt = json.loads(receipt_path.read_text())
    assert (repaired, failed) == (1, 0)
    assert receipt["pushed"] is True
    assert receipt["hub_revision"] == "revision-1"
    assert "publication_reconciled_utc" in receipt
    assert api.uploaded == ["raw_episodes/paired_sess_1"]


def test_reconcile_leaves_receipt_false_on_remote_hash_mismatch(tmp_path: Path):
    data_dir, receipt_path = _stale_publication(tmp_path)
    api = _FakeHubApi(corrupt=True)

    repaired, failed = _reconcile_unpushed_receipts(api, data_dir, "org/dataset")

    assert (repaired, failed) == (0, 1)
    assert json.loads(receipt_path.read_text())["pushed"] is False


def test_gripper_width_normalization_clips_to_closed_open_range():
    config = core_config.GripperConfig(
        tag_ids=(2, 3), closed_width_mm=40.0, open_width_mm=120.0
    )
    normalized = normalize_gripper_widths(
        np.asarray([20.0, 40.0, 80.0, 120.0, 140.0]), config
    )
    assert normalized.tolist() == pytest.approx([0.0, 0.0, 0.5, 1.0, 1.0])


class _FakeCapture:
    def __init__(self, frames: list[np.ndarray]):
        self.frames = frames
        self.read_count = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, _prop: int) -> float:
        return float(len(self.frames))

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.read_count >= len(self.frames):
            return False, None
        frame = self.frames[self.read_count]
        self.read_count += 1
        return True, frame

    def release(self) -> None:
        self.released = True


def test_sequential_video_reader_keeps_only_current_frame(tmp_path: Path, monkeypatch):
    import cv2

    capture = _FakeCapture([np.full((2, 3, 3), value, np.uint8) for value in range(4)])
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: capture)

    with _SequentialVideoReader(tmp_path / "camera.avi", expected_frames=4) as reader:
        first = reader.frame(0)
        assert reader.frame(0) is first  # repeated alignment index reuses the cached frame
        assert capture.read_count == 1
        assert np.all(reader.frame(3) == 3)
        assert capture.read_count == 4
        with pytest.raises(ValueError, match="cannot seek backward"):
            reader.frame(2)

    assert capture.released


def test_sequential_video_reader_rejects_frame_count_mismatch(tmp_path: Path, monkeypatch):
    import cv2

    capture = _FakeCapture([np.zeros((2, 3, 3), np.uint8)])
    monkeypatch.setattr(cv2, "VideoCapture", lambda _path: capture)

    with pytest.raises(ValueError, match="frames jsonl count 2 != video frames 1"):
        _SequentialVideoReader(tmp_path / "camera.avi", expected_frames=2)
    assert capture.released


def _write_tum(path: Path, x: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_tum(
        path,
        PoseTrack(
            t=np.array([0.0, 1.0]),
            pos=np.array([[x, 0.0, 0.0], [x, 0.0, 0.0]]),
            quat=np.tile([0.0, 0.0, 0.0, 1.0], (2, 1)),
        ),
    )


def _lidar_rig() -> core_config.RigConfig:
    return core_config.parse_rig_config(
        {
            "version": 1,
            "task": "t",
            "umis": {
                "umi1": {
                    "camera": {
                        "device": "/d",
                        "width": 64,
                        "height": 48,
                        "fps": 30,
                        "calib": "c.npz",
                    },
                    "lidar": {"iface": "eth0"},
                    "gripper": {"tag_ids": [2, 3]},
                }
            },
        }
    )


def test_pose_track_prefers_align_falls_back_to_glim(tmp_path: Path):
    # No pcap on disk => clock-skew realignment is a no-op, so the trajectory is read verbatim.
    umi = _lidar_rig().umis[0]
    m = {"sensors": {"umi1/lidar": {"kind": "lidar"}}}
    derived = tmp_path / "derived_ep"
    _write_tum(derived / "umi1" / "glim" / "trajectory.tum", 1.0)

    # No align output yet -> falls back to the per-lidar GLIM trajectory (non-joint path).
    track, basis = pose_track_for_umi(tmp_path, derived, m, umi)
    assert track.pos[0][0] == 1.0
    assert basis is not None  # lidar body-convention basis applied downstream

    # Align output present -> preferred (common-frame trajectory).
    _write_tum(derived / "align" / "umi1" / "trajectory.tum", 2.0)
    track2, _ = pose_track_for_umi(tmp_path, derived, m, umi)
    assert track2.pos[0][0] == 2.0
