from __future__ import annotations

import io
import os
import signal
import subprocess
from pathlib import Path
from types import SimpleNamespace

from robot_umi_device.camera import RustV4L2Camera, join_errors


def make_fresh_rust_project(root: Path) -> Path:
    project = root / "v4l2_recorder"
    binary = project / "target" / "release" / "robot-umi-v4l2-recorder"
    (project / "src").mkdir(parents=True)
    binary.parent.mkdir(parents=True)
    (project / "Cargo.toml").write_text("[package]\nname='x'\n", encoding="utf-8")
    (project / "Cargo.lock").write_text("", encoding="utf-8")
    (project / "src" / "main.rs").write_text("fn main() {}\n", encoding="utf-8")
    binary.write_text("#!/bin/sh\n", encoding="utf-8")
    os.utime(binary, (2_000_000_000, 2_000_000_000))
    return project


GOOD_STATS = (
    '{"camera_frame_count":1,"camera_dropped_sequence_frames":0,'
    '"camera_queue_full_count":0,"camera_dataset_eligible":true}\n'
)


class FakeProcess:
    returncode_after_wait: int | None = 0
    stats_json: str | None = GOOD_STATS
    stderr_bytes: bytes = b"recorder metrics"
    timeouts_before_exit: int = 0

    def __init__(self, cmd: list[str], **_kwargs):
        self.cmd = cmd
        self.pid = 123
        self.returncode: int | None = None
        self.stderr = io.BytesIO(self.stderr_bytes)
        self._timeouts_left = self.timeouts_before_exit

    def poll(self):
        return self.returncode

    def wait(self, timeout: float):
        if self._timeouts_left > 0:
            self._timeouts_left -= 1
            raise subprocess.TimeoutExpired(self.cmd, timeout)
        if self.stats_json is not None:
            stats_path = Path(self.cmd[self.cmd.index("--stats") + 1])
            stats_path.write_text(self.stats_json, encoding="utf-8")
        self.returncode = self.returncode_after_wait
        return self.returncode


def make_camera(
    tmp_path: Path, monkeypatch, process_cls: type[FakeProcess]
) -> tuple[RustV4L2Camera, list[tuple[int, int]], list[list[str]]]:
    make_fresh_rust_project(tmp_path)
    killed: list[tuple[int, int]] = []
    builds: list[list[str]] = []

    def fake_run(cmd, **_kwargs):
        builds.append(list(cmd))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("robot_umi_device.camera.subprocess.Popen", process_cls)
    monkeypatch.setattr(
        "robot_umi_device.camera.os.killpg", lambda pid, sig: killed.append((pid, sig))
    )
    monkeypatch.setattr("robot_umi_device.camera.subprocess.run", fake_run)
    # Builds are faked above, so the toolchain probe must be too (CI has no cargo).
    monkeypatch.setattr("robot_umi_device.camera.shutil.which", lambda _: "/usr/bin/cargo")
    camera = RustV4L2Camera("/dev/video0", 1920, 1200, 30, repo_root=tmp_path)
    return camera, killed, builds


def start_camera(camera: RustV4L2Camera, tmp_path: Path) -> None:
    camera.start(
        tmp_path / "camera_front.mjpeg.avi",
        tmp_path / "camera_frames.jsonl",
        tmp_path / "camera_stats.json",
    )


def test_rust_camera_stop_sends_sigint_and_reads_stats(tmp_path: Path, monkeypatch):
    camera, killed, _builds = make_camera(tmp_path, monkeypatch, FakeProcess)
    start_camera(camera, tmp_path)
    camera.stop()

    assert killed == [(123, signal.SIGINT)]
    assert camera.stats()["camera_frame_count"] == 1
    assert camera.stats()["camera_dataset_eligible"] is True
    assert "camera_error" not in camera.stats()


def test_stop_synthesizes_error_when_recorder_flags_ineligible_silently(
    tmp_path: Path, monkeypatch
):
    # The recorder exits 0 but declares the capture ineligible (dropped frames,
    # queue overflow) with an EMPTY camera_error. That verdict must become a
    # sensor error so the manifest marks the episode ineligible.
    class SilentlyIneligible(FakeProcess):
        returncode_after_wait = 0
        stats_json = (
            '{"camera_frame_count":100,"avi_frame_count":100,'
            '"camera_dropped_sequence_frames":7,"camera_queue_full_count":3,'
            '"camera_dataset_eligible":false,"camera_error":""}\n'
        )

    camera, _killed, _builds = make_camera(tmp_path, monkeypatch, SilentlyIneligible)
    start_camera(camera, tmp_path)
    camera.stop()

    stats = camera.stats()
    assert stats["camera_dataset_eligible"] is False
    assert "dropped frames (7)" in stats["camera_error"]
    assert "writer queue full (3)" in stats["camera_error"]


def test_stop_reports_recorder_failure_without_raising(tmp_path: Path, monkeypatch):
    class FailingProcess(FakeProcess):
        returncode_after_wait = 1
        stats_json = (
            '{"camera_frame_count":5,"camera_dataset_eligible":false,'
            '"camera_error":"write avi frame: boom"}\n'
        )
        stderr_bytes = b"recording failed: write avi frame: boom"

    camera, killed, _builds = make_camera(tmp_path, monkeypatch, FailingProcess)
    start_camera(camera, tmp_path)
    camera.stop()

    stats = camera.stats()
    assert killed == [(123, signal.SIGINT)]
    assert stats["camera_dataset_eligible"] is False
    assert "write avi frame: boom" in stats["camera_error"]
    assert "exited with code 1" in stats["camera_error"]
    assert stats["camera_frame_count"] == 5


def test_stop_reports_missing_stats_without_raising(tmp_path: Path, monkeypatch):
    class NoStatsProcess(FakeProcess):
        returncode_after_wait = 1
        stats_json = None
        stderr_bytes = b"recording failed: open /dev/video0: busy"

    camera, _killed, _builds = make_camera(tmp_path, monkeypatch, NoStatsProcess)
    start_camera(camera, tmp_path)
    camera.stop()

    stats = camera.stats()
    assert stats["camera_dataset_eligible"] is False
    assert "exited with code 1" in stats["camera_error"]
    assert "busy" in stats["camera_error"]


def test_stop_escalates_to_sigkill_after_graceful_timeout(tmp_path: Path, monkeypatch):
    class StuckProcess(FakeProcess):
        returncode_after_wait = -9
        stats_json = None
        timeouts_before_exit = 1

    camera, killed, _builds = make_camera(tmp_path, monkeypatch, StuckProcess)
    start_camera(camera, tmp_path)
    camera.stop()

    assert killed == [(123, signal.SIGINT), (123, signal.SIGKILL)]
    assert camera.stats()["camera_dataset_eligible"] is False


def test_cancel_escalates_to_sigkill_after_timeout(tmp_path: Path, monkeypatch):
    class StuckProcess(FakeProcess):
        returncode_after_wait = -9
        stats_json = None
        timeouts_before_exit = 1

    camera, killed, _builds = make_camera(tmp_path, monkeypatch, StuckProcess)
    start_camera(camera, tmp_path)
    camera.cancel()

    assert killed == [(123, signal.SIGTERM), (123, signal.SIGKILL)]
    assert camera.stats() == {}


def test_poll_error_reports_early_death(tmp_path: Path, monkeypatch):
    camera, killed, _builds = make_camera(tmp_path, monkeypatch, FakeProcess)
    start_camera(camera, tmp_path)
    assert camera.poll_error() is None

    assert camera.process is not None
    camera.process.returncode = 1
    assert "exited early" in (camera.poll_error() or "")

    camera.stop()
    assert killed == []  # already dead: no signals sent
    assert "exited with code 1" in camera.stats()["camera_error"]
    assert camera.poll_error() is None


def test_prepare_builds_stale_binary_once(tmp_path: Path, monkeypatch):
    camera, _killed, builds = make_camera(tmp_path, monkeypatch, FakeProcess)
    os.utime(camera.binary_path, (0, 0))

    camera.prepare()

    assert len(builds) == 1
    assert "build" in builds[0]


def test_join_errors_merges_distinct_messages():
    assert join_errors(None, None) is None
    assert join_errors("a", None) == "a"
    assert join_errors(None, "b") == "b"
    assert join_errors("a", "b") == "a; b"
    assert join_errors("a; b", "b") == "a; b"
