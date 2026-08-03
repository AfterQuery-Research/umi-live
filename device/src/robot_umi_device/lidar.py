"""Lidar episode recording on top of the (mock) 3D lidar SDK.

The sensor lives in the SDK's device process and produces ~20k-point frames at
10 Hz on its own clock (Mid-360 class); this module only subscribes and writes
them to the episode. Each episode gets two files:

- ``mock_lidar.jsonl``: one index line per frame (timestamps, seq, point
  count, byte offset/length into the points file, drop counter);
- ``mock_lidar_points.bin``: the frames' points, packed float32
  (x, y, z, intensity), concatenated in index order.

The recorder connects once per session (a real sensor keeps spinning between
episodes) and starts/stops the packet stream per episode.
"""

from __future__ import annotations

import contextlib
import json
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Protocol, TextIO

from robot_umi_device.mock_lidar_sdk import (
    DEFAULT_PORT,
    MID360_PROFILE,
    DeviceInfo,
    LidarFrame,
    LidarProfile,
    MockLidar,
)


class LidarError(RuntimeError):
    """The lidar stream failed; the episode is missing lidar data."""


# Frames buffered in RAM between the socket reader and the disk writer. SD
# cards stall for ~1s under load; a single read+write thread blocking on such
# a stall overflows the (~208 KB) socket buffer and loses whole frames. 64
# frames buffer ~6.4 s at 10 Hz (~20 MB for Mid-360, ~88 MB worst case for
# Airy) - safe on a 4 GB Pi, and measured to absorb every stall in practice.
RING_BUFFER_FRAMES = 64


@dataclass(frozen=True)
class LidarEpisodeStats:
    frame_count: int
    points_total: int
    dropped_packets: int
    max_gap_ms: float
    model: str
    host_dropped_frames: int = 0


class LidarSensor(Protocol):
    def connect(self) -> DeviceInfo: ...
    def close(self) -> None: ...
    def begin_episode(self, index_path: Path, points_path: Path) -> None: ...
    def end_episode(self) -> LidarEpisodeStats: ...
    def cancel_episode(self) -> None: ...
    def poll_error(self) -> str | None: ...


class LidarRecorder:
    def __init__(
        self,
        port: str = DEFAULT_PORT,
        profile: LidarProfile = MID360_PROFILE,
        seed: int = 0,
    ):
        self.sdk = MockLidar(port, profile=profile, seed=seed)
        self.info: DeviceInfo | None = None
        self.error: str | None = None
        self._reader_thread: threading.Thread | None = None
        self._writer_thread: threading.Thread | None = None
        self._buffer: queue.Queue[LidarFrame] = queue.Queue(maxsize=RING_BUFFER_FRAMES)
        self._stop_requested = threading.Event()
        self._writer_stop = threading.Event()
        self._first_frame_event = threading.Event()
        self._index_file: TextIO | None = None
        self._points_file: BinaryIO | None = None
        self._points_offset = 0
        self._frame_count = 0
        self._points_total = 0
        self._dropped_packets = 0
        self._host_dropped_frames = 0
        self._max_gap_ms = 0.0
        self._previous_timestamp_ns: int | None = None

    # -- session lifecycle -----------------------------------------------

    def connect(self) -> DeviceInfo:
        self.info = self.sdk.connect()
        return self.info

    def close(self) -> None:
        if self._reader_thread is not None:
            self.cancel_episode()
        self.sdk.disconnect()
        self.info = None

    # -- episode lifecycle --------------------------------------------------

    def begin_episode(self, index_path: Path, points_path: Path) -> None:
        if self._reader_thread is not None:
            raise LidarError("lidar episode already active")
        if not self.sdk.alive():
            raise LidarError("lidar device is not connected")
        self.error = None
        self._stop_requested.clear()
        self._writer_stop.clear()
        self._first_frame_event.clear()
        while not self._buffer.empty():
            self._buffer.get_nowait()
        self._points_offset = 0
        self._frame_count = 0
        self._points_total = 0
        self._dropped_packets = 0
        self._host_dropped_frames = 0
        self._max_gap_ms = 0.0
        self._previous_timestamp_ns = None
        self._index_file = index_path.open("w", encoding="utf-8")
        self._points_file = points_path.open("wb")
        self.sdk.start_scanning()
        self._reader_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._reader_thread.start()
        self._writer_thread = threading.Thread(target=self._write_loop, daemon=True)
        self._writer_thread.start()
        period = 1.0 / self.sdk.profile.frame_rate_hz
        # The first frame finishes assembling roughly one period after start.
        self._first_frame_event.wait(timeout=max(1.0, 3 * period))
        if self.error is not None:
            self._finish_threads()
            raise LidarError(self.error)

    def end_episode(self) -> LidarEpisodeStats:
        """Stop the stream and finish the episode files. Raises LidarError if
        the stream failed, so a missing/truncated lidar stream can never look
        like a successful episode."""
        if self._reader_thread is None:
            # Guards against returning stale stats from a previous episode.
            raise LidarError("no active lidar episode")
        self._finish_threads()
        if self.error is not None:
            raise LidarError(self.error)
        if self._frame_count == 0:
            raise LidarError("lidar produced no frames for this episode")
        assert self.info is not None
        return LidarEpisodeStats(
            frame_count=self._frame_count,
            points_total=self._points_total,
            dropped_packets=self._dropped_packets,
            max_gap_ms=self._max_gap_ms,
            model=self.info.model,
            host_dropped_frames=self._host_dropped_frames,
        )

    def cancel_episode(self) -> None:
        with contextlib.suppress(Exception):
            self._finish_threads()

    def poll_error(self) -> str | None:
        """Report a lidar stream that failed mid-episode."""
        if self.error is not None:
            return self.error
        if self._reader_thread is not None and not self.sdk.alive():
            return "lidar device process died"
        return None

    # -- internals ------------------------------------------------------------

    def _finish_threads(self) -> None:
        reader = self._reader_thread
        if reader is None:
            self._close_files()
            return
        self._stop_requested.set()
        if self.sdk.alive():
            with contextlib.suppress(Exception):
                self.sdk.stop_scanning()
        reader.join(timeout=10)
        reader_alive = reader.is_alive()
        self._reader_thread = None
        writer = self._writer_thread
        self._writer_thread = None
        writer_alive = False
        if writer is not None:
            self._writer_stop.set()
            # May legitimately take a while: it is flushing the buffered
            # frames to a possibly-stalling SD card.
            writer.join(timeout=30)
            writer_alive = writer.is_alive()
        self._close_files()
        if reader_alive:
            raise LidarError("lidar reader thread did not stop within 10s")
        if writer_alive:
            raise LidarError("lidar writer thread did not stop within 30s")

    def _close_files(self) -> None:
        for attr in ("_index_file", "_points_file"):
            handle = getattr(self, attr)
            if handle is not None:
                with contextlib.suppress(Exception):
                    handle.close()
                setattr(self, attr, None)

    def _read_loop(self) -> None:
        """Drain the SDK socket into the RAM buffer; never blocks on disk.

        A single thread doing both socket reads and SD writes loses frames:
        an SD write-latency spike (~1 s is routine under load) overflows the
        socket buffer in well under a frame period.
        """
        try:
            while True:
                frame = self.sdk.read_frame(timeout=0.1)
                if frame is None:
                    if self._stop_requested.is_set():
                        # Stream is quiet after stop: keep any partial frame.
                        pending = self.sdk.flush_pending_frame()
                        if pending is not None and pending.point_count > 0:
                            self._enqueue(pending)
                        return
                    continue
                self._enqueue(frame)
        except Exception as exc:  # surfaced via poll_error()/end_episode()
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            self._first_frame_event.set()

    def _enqueue(self, frame: LidarFrame) -> None:
        try:
            self._buffer.put_nowait(frame)
        except queue.Full:
            # A disk stall outlasted the buffer: drop honestly, keep draining
            # the socket so recovery is immediate when the disk returns.
            self._host_dropped_frames += 1

    def _write_loop(self) -> None:
        """Persist buffered frames; free to block on a slow SD card."""
        while True:
            try:
                frame = self._buffer.get(timeout=0.2)
            except queue.Empty:
                if self._writer_stop.is_set():
                    return
                continue
            try:
                self._write_frame(frame)
            except Exception as exc:
                self.error = f"{type(exc).__name__}: {exc}"
                self._first_frame_event.set()
                return

    def _write_frame(self, frame: LidarFrame) -> None:
        assert self._index_file is not None and self._points_file is not None
        assert self.info is not None
        self._points_file.write(frame.points)
        record = {
            "sensor": "mock_lidar",
            "model": self.info.model,
            "seq": frame.seq,
            "time_monotonic_ns": frame.time_monotonic_ns,
            "time_unix_ns": frame.time_unix_ns,
            "point_count": frame.point_count,
            "offset": self._points_offset,
            "length": len(frame.points),
            "dropped_packets_total": frame.dropped_packets_total,
        }
        self._index_file.write(json.dumps(record, separators=(",", ":")) + "\n")
        # Flush periodically, not per frame: a synchronous 320 KB flush to a
        # slow SD card can stall this reader long enough to overflow the
        # socket buffer and drop packets.
        if self._frame_count % 10 == 0:
            self._index_file.flush()
            self._points_file.flush()
        self._points_offset += len(frame.points)
        if self._previous_timestamp_ns is not None:
            gap_ms = (frame.time_monotonic_ns - self._previous_timestamp_ns) / 1e6
            self._max_gap_ms = max(self._max_gap_ms, gap_ms)
        self._previous_timestamp_ns = frame.time_monotonic_ns
        self._frame_count += 1
        self._points_total += frame.point_count
        self._dropped_packets = frame.dropped_packets_total
        self._first_frame_event.set()
