"""Camera capture helpers for the Raspberry Pi device."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Protocol

# Graceful drain can legitimately take a while on a slow SD card: the Rust
# recorder flushes up to writer-queue-frames buffered frames plus the AVI
# index before exiting.
STOP_TIMEOUT_SECONDS = 60.0
CANCEL_TIMEOUT_SECONDS = 10.0
KILL_TIMEOUT_SECONDS = 10.0


class CameraProcess(Protocol):
    def prepare(self) -> None: ...
    def start(self, output_path: Path, timestamps_path: Path, stats_path: Path) -> int: ...
    def stop(self) -> int: ...
    def cancel(self) -> None: ...
    def stats(self) -> dict[str, Any]: ...
    def poll_error(self) -> str | None: ...


class RustV4L2Camera:
    def __init__(
        self,
        device: str,
        width: int,
        height: int,
        fps: int,
        *,
        exposure: int | None = None,
        warmup_frames: int = 8,
        buffer_count: int = 32,
        writer_queue_frames: int = 120,
        repo_root: Path | None = None,
    ):
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.exposure = exposure
        self.warmup_frames = warmup_frames
        self.buffer_count = buffer_count
        self.writer_queue_frames = writer_queue_frames
        self.repo_root = repo_root or Path(__file__).resolve().parents[2]
        self.project_dir = self.repo_root / "v4l2_recorder"
        self.binary_path = self.project_dir / "target" / "release" / "umi-flow-v4l2-recorder"
        self.process: subprocess.Popen[bytes] | None = None
        self.stats_path: Path | None = None
        self._stats: dict[str, Any] = {}

    def prepare(self) -> None:
        """Build the recorder binary if needed. Run at program startup so the
        (potentially minutes-long) cargo build never delays a record start."""
        self._ensure_binary()

    def start(self, output_path: Path, timestamps_path: Path, stats_path: Path) -> int:
        self._ensure_binary()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.stats_path = stats_path
        self._stats = {}
        cmd = [
            str(self.binary_path),
            "--device",
            self.device,
            "--output",
            str(output_path),
            "--timestamps",
            str(timestamps_path),
            "--stats",
            str(stats_path),
            "--width",
            str(self.width),
            "--height",
            str(self.height),
            "--fps",
            str(self.fps),
            *(["--exposure", str(self.exposure)] if self.exposure is not None else []),
            "--warmup-frames",
            str(self.warmup_frames),
            "--buffer-count",
            str(self.buffer_count),
            "--writer-queue-frames",
            str(self.writer_queue_frames),
        ]
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        return time.monotonic_ns()

    def poll_error(self) -> str | None:
        """Report when the recorder process died while it should be running."""
        if self.process is None:
            return None
        returncode = self.process.poll()
        if returncode is None:
            return None
        return f"camera recorder exited early (code {returncode})"

    def stop(self) -> int:
        """Stop the recorder and collect stats.

        Never raises for recorder failures: whatever the recorder salvaged
        stays on disk, and the failure is reported via stats()['camera_error']
        with camera_dataset_eligible forced to False.
        """
        stop_ns = time.monotonic_ns()
        if self.process is None:
            return stop_ns
        process = self.process
        self.process = None
        if process.poll() is None:
            # SIGINT asks for a graceful drain + AVI finalization. SIGTERM is
            # trapped by the recorder as the same graceful flag, so the only
            # meaningful escalation is SIGKILL.
            self._signal(process, signal.SIGINT)
            if not self._wait(process, STOP_TIMEOUT_SECONDS):
                self._signal(process, signal.SIGKILL)
                self._wait(process, KILL_TIMEOUT_SECONDS)
        stderr = b""
        if process.stderr is not None:
            stderr = process.stderr.read()
        error: str | None = None
        if process.returncode is None:
            error = "camera recorder did not exit after SIGKILL"
        elif process.returncode != 0:
            detail = stderr.decode(errors="replace").strip()
            error = f"camera recorder exited with code {process.returncode}: {detail}"
        self._stats = self._read_stats(error)
        return stop_ns

    def cancel(self) -> None:
        process = self.process
        self.process = None
        self._stats = {}
        if process is None:
            return
        if process.poll() is None:
            self._signal(process, signal.SIGTERM)
            if not self._wait(process, CANCEL_TIMEOUT_SECONDS):
                self._signal(process, signal.SIGKILL)
                self._wait(process, KILL_TIMEOUT_SECONDS)
        if process.stderr is not None:
            process.stderr.read()

    def stats(self) -> dict[str, Any]:
        return dict(self._stats)

    @staticmethod
    def _signal(process: subprocess.Popen[bytes], sig: int) -> None:
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass

    @staticmethod
    def _wait(process: subprocess.Popen[bytes], timeout_seconds: float) -> bool:
        try:
            process.wait(timeout=timeout_seconds)
            return True
        except subprocess.TimeoutExpired:
            return False

    def _read_stats(self, error: str | None) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        if self.stats_path is not None and self.stats_path.exists():
            try:
                stats = json.loads(self.stats_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                error = join_errors(error, f"unreadable camera_stats.json: {exc}")
        elif error is None:
            error = "camera_stats.json was not written by the Rust recorder"
        # The recorder can flag a capture ineligible (dropped/late/faulty frames)
        # while exiting 0 with an EMPTY camera_error. Manifest eligibility is derived
        # purely from the sensor error string (sensors.py -> manifest.py), so that
        # verdict would otherwise be lost and bad frames would enter the dataset.
        # Synthesize an error from the anomaly counters so it propagates.
        if stats.get("camera_dataset_eligible") is False and not stats.get("camera_error"):
            error = join_errors(error, _camera_ineligible_reason(stats))
        error = join_errors(stats.get("camera_error") or None, error)
        if error:
            stats["camera_error"] = error
            stats["camera_dataset_eligible"] = False
        return stats

    def _ensure_binary(self) -> None:
        if self._binary_is_fresh():
            return  # a fresh prebuilt binary needs no toolchain
        cargo = shutil.which("cargo")
        if cargo is None:
            raise RuntimeError("cargo is required to build the Rust V4L2 recorder")
        subprocess.run(
            [
                cargo,
                "build",
                "--release",
                "--manifest-path",
                str(self.project_dir / "Cargo.toml"),
            ],
            cwd=self.repo_root,
            check=True,
        )

    def _binary_is_fresh(self) -> bool:
        if not self.binary_path.exists():
            return False
        binary_mtime = self.binary_path.stat().st_mtime
        sources = [
            self.project_dir / "Cargo.toml",
            self.project_dir / "Cargo.lock",
            self.project_dir / "src" / "main.rs",
        ]
        return all(path.exists() and path.stat().st_mtime <= binary_mtime for path in sources)


def join_errors(first: str | None, second: str | None) -> str | None:
    if first and second and second not in first:
        return f"{first}; {second}"
    return first or second


# Anomaly counters the Rust recorder writes (see v4l2_recorder write_stats); any
# non-zero one makes the capture dataset-ineligible with camera_error left empty.
_CAMERA_ANOMALY_COUNTERS = {
    "camera_dropped_sequence_frames": "dropped frames",
    "camera_queue_full_count": "writer queue full",
    "camera_non_monotonic_timestamps": "non-monotonic timestamps",
    "camera_bad_timestamp_metadata": "faulty timestamp metadata",
    "delta_bad_count": "bad frame-interval deltas",
}


def _camera_ineligible_reason(stats: dict[str, Any]) -> str:
    parts = [
        f"{label} ({stats[key]})"
        for key, label in _CAMERA_ANOMALY_COUNTERS.items()
        if stats.get(key)
    ]
    frame_count = stats.get("camera_frame_count")
    written = stats.get("avi_frame_count")
    if frame_count is not None and written is not None and frame_count != written:
        parts.append(f"only {written} of {frame_count} frames written")
    detail = ", ".join(parts) if parts else "recorder flagged the capture ineligible"
    return f"camera capture not dataset-eligible: {detail}"
