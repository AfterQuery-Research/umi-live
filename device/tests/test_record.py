from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from robot_umi_device.episode import DeviceConfig, EpisodeRecorder, monotonic_ns
from robot_umi_device.lidar import LidarEpisodeStats
from robot_umi_device.mock_lidar_sdk import DeviceInfo

RECORD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "record.py"
spec = importlib.util.spec_from_file_location("record", RECORD_PATH)
assert spec is not None and spec.loader is not None
record = importlib.util.module_from_spec(spec)
sys.modules["record"] = record
spec.loader.exec_module(record)


class FakeCamera:
    def __init__(self, error: str = ""):
        self.error = error

    def prepare(self) -> None:
        pass

    def start(self, output_path: Path, timestamps_path: Path, stats_path: Path) -> int:
        output_path.write_bytes(b"fake video")
        timestamps_path.write_text("{}\n", encoding="utf-8")
        stats_path.write_text("{}\n", encoding="utf-8")
        return monotonic_ns()

    def stop(self) -> int:
        return monotonic_ns()

    def cancel(self) -> None:
        pass

    def poll_error(self) -> str | None:
        return None

    def stats(self) -> dict:
        stats: dict[str, Any] = {
            "camera_frame_count": 1,
            "camera_dataset_eligible": not self.error,
        }
        if self.error:
            stats["camera_error"] = self.error
        return stats


class FakeLidar:
    def __init__(self):
        self.closed = False

    def connect(self) -> DeviceInfo:
        return DeviceInfo(
            model="FAKE-LDS",
            serial="FAKE-LDS-00000001",
            firmware="0.0.1",
            frame_rate_hz=10,
            points_per_frame=20_000,
            points_per_second=200_000,
            horizontal_fov_deg=360.0,
            vertical_fov_min_deg=-7.0,
            vertical_fov_max_deg=52.0,
            range_min_m=0.1,
            range_max_m=70.0,
        )

    def close(self) -> None:
        self.closed = True

    def begin_episode(self, index_path: Path, points_path: Path) -> None:
        index_path.write_text('{"sensor":"mock_lidar"}\n', encoding="utf-8")
        points_path.write_bytes(b"\x00" * 16)

    def end_episode(self) -> LidarEpisodeStats:
        return LidarEpisodeStats(
            frame_count=1,
            points_total=1,
            dropped_packets=0,
            max_gap_ms=100.0,
            model="FAKE-LDS",
        )

    def cancel_episode(self) -> None:
        pass

    def poll_error(self) -> str | None:
        return None


class DummyUploader:
    def __init__(self):
        self.submitted: list[Path] = []
        self.closed: list[bool] = []

    def submit(self, episode_dir: Path) -> None:
        self.submitted.append(episode_dir)

    def close(self, wait: bool = False) -> None:
        self.closed.append(wait)


def make_recorder(
    tmp_path: Path, camera: FakeCamera | None = None
) -> tuple[EpisodeRecorder, FakeLidar]:
    cfg = DeviceConfig(server_url="http://server/upload", spool_dir=tmp_path)
    lidar = FakeLidar()
    return EpisodeRecorder(cfg, camera=camera or FakeCamera(), lidar=lidar), lidar


def test_spool_has_data_ignores_empty_scaffolding(tmp_path: Path):
    spool = tmp_path / "spool"
    assert not record.spool_has_data(spool)

    make_recorder(spool)  # creates empty in_progress/complete/sent
    assert not record.spool_has_data(spool)

    (spool / "staging").mkdir()
    (spool / "staging" / "episode_1.tar").write_bytes(b"tar")
    assert not record.spool_has_data(spool)  # derived artifacts do not count

    (spool / "complete" / "episode_1").mkdir()
    assert record.spool_has_data(spool)


def test_spool_has_data_counts_stray_entries(tmp_path: Path):
    spool = tmp_path / "spool"
    make_recorder(spool)
    (spool / "notes.txt").write_text("hi", encoding="utf-8")
    assert record.spool_has_data(spool)


def test_shutdown_saves_active_episode_and_closes_lidar(tmp_path: Path, capsys):
    recorder, lidar = make_recorder(tmp_path)
    uploader: Any = DummyUploader()
    recorder.start()

    record.shutdown(recorder, uploader, wait=False)

    assert recorder.active is None
    assert len(uploader.submitted) == 1
    assert uploader.submitted[0].parent == tmp_path / "complete"
    assert uploader.closed == [False]
    assert lidar.closed
    assert "stopping active episode before exit" in capsys.readouterr().out


def test_shutdown_without_active_episode_only_closes(tmp_path: Path):
    recorder, lidar = make_recorder(tmp_path)
    uploader: Any = DummyUploader()

    record.shutdown(recorder, uploader, wait=True)

    assert uploader.submitted == []
    assert uploader.closed == [True]
    assert lidar.closed


def test_finish_active_surfaces_sensor_warnings(tmp_path: Path, capsys):
    recorder, _lidar = make_recorder(
        tmp_path, FakeCamera(error="camera recorder exited with code 1: boom")
    )
    uploader: Any = DummyUploader()
    recorder.start()

    record.finish_active(recorder, uploader)

    out = capsys.readouterr().out
    assert "warning: camera" in out
    assert "completed" in out
    assert len(uploader.submitted) == 1
    # The failed episode is preserved and queued, never deleted.
    assert uploader.submitted[0].exists()
