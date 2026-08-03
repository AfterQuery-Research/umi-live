"""Mock 3D lidar SDK, shaped like a real vendor library (Livox/RoboSense
style) and matched to real sensor specs.

Profiles model real hardware:

- ``MID360_PROFILE`` (default): Livox Mid-360 class. 200,000 points/s at
  10 Hz frames (~20,000 points per frame), 360deg x 59deg FOV (-7..+52deg),
  0.1 m blind zone, 70 m max range.
- ``AIRY_PROFILE``: RoboSense Airy class (single return). ~860,000 points/s
  at 10 Hz (~86,000 points per frame), 360deg x 90deg hemispherical FOV,
  0.1-60 m.

The simulated device runs as its own process (own interpreter and GIL, like a
driver daemon) and takes frames at the profile's rate on its own clock:
host-side load can delay when frames are *read*, never when they are *taken*.

Like real lidars, the device streams each frame as many small packets (the
real Mid-360 emits ~1380-byte UDP packets); the SDK client reassembles frames.
A slow consumer loses packets - counted, never blocking the device - and a
frame missing packets is delivered with fewer points, exactly like real
hardware. Each point is packed binary float32 (x, y, z, intensity).

Typical use::

    lidar = MockLidar()                  # Mid-360 profile
    info = lidar.connect()               # spawns the device process
    lidar.start_scanning()
    frame = lidar.read_frame(timeout=1.0)
    lidar.stop_scanning()
    lidar.disconnect()
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import select
import socket
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PORT = "mock://lidar0"
FIRMWARE = "1.2.0"
CONNECT_TIMEOUT_SECONDS = 10.0
POINT_STRUCT = "<4f"  # x, y, z, intensity as little-endian float32
POINT_BYTES = struct.calcsize(POINT_STRUCT)
POINTS_PER_PACKET = 1600
# Ask for large buffers (the OS caps unprivileged requests, e.g. ~208 KiB on
# stock Linux); packets are sized well under any cap.
SOCKET_BUFFER_BYTES = 4 << 20
_HEADER_LEN_STRUCT = "!I"


class MockLidarError(RuntimeError):
    """SDK-level failure: device unreachable, dead, or protocol error."""


@dataclass(frozen=True)
class LidarProfile:
    model: str
    frame_rate_hz: int
    points_per_frame: int
    horizontal_fov_deg: float
    vertical_fov_min_deg: float
    vertical_fov_max_deg: float
    range_min_m: float
    range_max_m: float

    @property
    def points_per_second(self) -> int:
        return self.frame_rate_hz * self.points_per_frame


# Livox Mid-360: 200k pts/s first return, 10 Hz, 360 x 59 deg (-7..+52),
# 0.1 m blind zone, 40 m @ 10% / 70 m max.
MID360_PROFILE = LidarProfile(
    model="MOCK-MID360",
    frame_rate_hz=10,
    points_per_frame=20_000,
    horizontal_fov_deg=360.0,
    vertical_fov_min_deg=-7.0,
    vertical_fov_max_deg=52.0,
    range_min_m=0.1,
    range_max_m=70.0,
)

# RoboSense Airy: 96-beam hemispherical, ~860k pts/s single return, 10 Hz,
# 360 x 90 deg, 30 m @ 10% / 60 m max.
AIRY_PROFILE = LidarProfile(
    model="MOCK-AIRY",
    frame_rate_hz=10,
    points_per_frame=86_000,
    horizontal_fov_deg=360.0,
    vertical_fov_min_deg=0.0,
    vertical_fov_max_deg=90.0,
    range_min_m=0.1,
    range_max_m=60.0,
)

PROFILES = {profile.model: profile for profile in (MID360_PROFILE, AIRY_PROFILE)}


@dataclass(frozen=True)
class DeviceInfo:
    model: str
    serial: str
    firmware: str
    frame_rate_hz: int
    points_per_frame: int
    points_per_second: int
    horizontal_fov_deg: float
    vertical_fov_min_deg: float
    vertical_fov_max_deg: float
    range_min_m: float
    range_max_m: float


@dataclass(frozen=True)
class LidarFrame:
    seq: int
    time_monotonic_ns: int
    time_unix_ns: int
    point_count: int
    points: bytes  # packed POINT_STRUCT * point_count
    dropped_packets_total: int


def _set_socket_buffers(sock: socket.socket) -> None:
    for option in (socket.SO_SNDBUF, socket.SO_RCVBUF):
        with contextlib.suppress(OSError):
            sock.setsockopt(socket.SOL_SOCKET, option, SOCKET_BUFFER_BYTES)


class MockLidar:
    def __init__(
        self,
        port: str = DEFAULT_PORT,
        profile: LidarProfile = MID360_PROFILE,
        seed: int = 0,
    ):
        self.port = port
        self.profile = profile
        self.seed = seed
        self._runtime_dir: Path | None = None
        self._socket: socket.socket | None = None
        self._device_path: str | None = None
        self._process: subprocess.Popen[bytes] | None = None
        self.info: DeviceInfo | None = None
        self._pending_seq: int | None = None
        self._pending: dict | None = None

    # -- lifecycle ---------------------------------------------------------

    def connect(self) -> DeviceInfo:
        if self._process is not None:
            raise MockLidarError("already connected")
        self._runtime_dir = Path(tempfile.mkdtemp(prefix="mock-lidar-"))
        client_path = str(self._runtime_dir / "client.sock")
        self._device_path = str(self._runtime_dir / "device.sock")
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        _set_socket_buffers(self._socket)
        self._socket.bind(client_path)
        self._process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "robot_umi_device.mock_lidar_sdk",
                self._device_path,
                client_path,
                self.profile.model,
                str(self.seed),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        deadline = time.monotonic() + CONNECT_TIMEOUT_SECONDS
        # The device binds its socket then announces itself; poke it until the
        # handshake lands (its socket may not exist for the first few ms).
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.disconnect()
                raise MockLidarError("timed out connecting to lidar device")
            if self._process.poll() is not None:
                raise MockLidarError(
                    f"lidar device process exited (code {self._process.returncode})"
                )
            try:
                self._send({"cmd": "hello"})
            except (ConnectionRefusedError, FileNotFoundError):
                time.sleep(0.02)
                continue
            message = self._recv_control(timeout=min(0.2, remaining))
            if message is not None and message.get("type") == "info":
                self.info = DeviceInfo(
                    model=str(message["model"]),
                    serial=str(message["serial"]),
                    firmware=str(message["firmware"]),
                    frame_rate_hz=int(message["frame_rate_hz"]),
                    points_per_frame=int(message["points_per_frame"]),
                    points_per_second=int(message["points_per_second"]),
                    horizontal_fov_deg=float(message["horizontal_fov_deg"]),
                    vertical_fov_min_deg=float(message["vertical_fov_min_deg"]),
                    vertical_fov_max_deg=float(message["vertical_fov_max_deg"]),
                    range_min_m=float(message["range_min_m"]),
                    range_max_m=float(message["range_max_m"]),
                )
                return self.info

    def disconnect(self) -> None:
        process = self._process
        self._process = None
        if process is not None and process.poll() is None:
            try:
                self._send({"cmd": "bye"})
            except OSError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        if self._runtime_dir is not None:
            for name in ("client.sock", "device.sock"):
                (self._runtime_dir / name).unlink(missing_ok=True)
            with contextlib.suppress(OSError):
                self._runtime_dir.rmdir()
            self._runtime_dir = None
        self.info = None
        self._pending = None
        self._pending_seq = None

    def alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    # -- scanning ----------------------------------------------------------

    def start_scanning(self) -> None:
        self._require_connected()
        self._send({"cmd": "start"})

    def stop_scanning(self) -> None:
        self._require_connected()
        self._send({"cmd": "stop"})

    def read_frame(self, timeout: float) -> LidarFrame | None:
        """Next assembled frame, or None if none completed within timeout.

        A frame is emitted when its last packet arrives, or - if packets were
        lost - when the next frame begins (partial frame, fewer points).
        Raises MockLidarError when the device process has died.
        """
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if not self.alive():
                    raise MockLidarError("lidar device process died")
                return None
            datagram = self._recv_raw(timeout=remaining)
            if datagram is None:
                continue
            if datagram[:1] == b"{":
                continue  # control message; ignore during streaming
            frame = self._ingest_packet(datagram)
            if frame is not None:
                return frame

    def flush_pending_frame(self) -> LidarFrame | None:
        """Emit a partially assembled frame (used at stream end)."""
        frame = self._finish_pending()
        self._pending = None
        self._pending_seq = None
        return frame

    # -- plumbing ----------------------------------------------------------

    def _ingest_packet(self, datagram: bytes) -> LidarFrame | None:
        header_len = struct.unpack(_HEADER_LEN_STRUCT, datagram[:4])[0]
        header = json.loads(datagram[4 : 4 + header_len].decode("utf-8"))
        payload = datagram[4 + header_len :]
        completed: LidarFrame | None = None
        if self._pending_seq != header["fs"]:
            completed = self._finish_pending()
            self._pending_seq = header["fs"]
            self._pending = {
                "seq": header["fs"],
                "tm": header["tm"],
                "tu": header["tu"],
                "dp": header["dp"],
                "packets": {},
                "packet_count": header["pc"],
            }
        assert self._pending is not None
        self._pending["packets"][header["pi"]] = payload
        self._pending["dp"] = header["dp"]
        if len(self._pending["packets"]) == self._pending["packet_count"]:
            complete = self._finish_pending()
            self._pending = None
            self._pending_seq = None
            # By construction only one frame can complete per packet.
            return completed or complete
        return completed

    def _finish_pending(self) -> LidarFrame | None:
        if self._pending is None:
            return None
        packets = self._pending["packets"]
        points = b"".join(packets[i] for i in sorted(packets))
        return LidarFrame(
            seq=self._pending["seq"],
            time_monotonic_ns=self._pending["tm"],
            time_unix_ns=self._pending["tu"],
            point_count=len(points) // POINT_BYTES,
            points=points,
            dropped_packets_total=self._pending["dp"],
        )

    def _require_connected(self) -> None:
        if self._process is None or self._socket is None:
            raise MockLidarError("not connected")
        if not self.alive():
            raise MockLidarError("lidar device process died")

    def _send(self, message: dict) -> None:
        assert self._socket is not None and self._device_path is not None
        self._socket.sendto(json.dumps(message).encode("utf-8"), self._device_path)

    def _recv_raw(self, timeout: float) -> bytes | None:
        assert self._socket is not None
        readable, _, _ = select.select([self._socket], [], [], max(0.0, timeout))
        if not readable:
            return None
        data, _ = self._socket.recvfrom(4 << 20)
        return data

    def _recv_control(self, timeout: float) -> dict | None:
        data = self._recv_raw(timeout)
        if data is None or data[:1] != b"{":
            return None
        return json.loads(data.decode("utf-8"))


# -- device process ----------------------------------------------------------


def _device_loop(
    device_path: str,
    client_path: str,
    model: str,
    seed: int,
) -> None:
    """The simulated sensor. Runs in its own process; frames are taken on
    this process's schedule and stamped here, independent of the consumer.

    Each frame's points stream as POINTS_PER_PACKET-sized packets paced
    across the frame period, like a real lidar's continuous packet stream.
    """
    import numpy as np

    profile = PROFILES[model]
    rng = np.random.default_rng(seed)
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
    _set_socket_buffers(sock)
    sock.bind(device_path)
    sock.setblocking(False)
    serial = f"{profile.model}-{os.getpid():08d}"
    scanning = False
    seq = 0
    dropped_packets_total = 0
    period = 1.0 / profile.frame_rate_hz
    packets_per_frame = math.ceil(profile.points_per_frame / POINTS_PER_PACKET)
    packet_spacing = period / (packets_per_frame + 1)
    next_frame = time.monotonic()
    outbox: list[bytes] = []
    next_packet_at = 0.0

    def send_packet(payload: bytes) -> bool:
        try:
            sock.sendto(payload, client_path)
            return True
        except (BlockingIOError, OSError):
            return False

    def make_frame_packets() -> list[bytes]:
        points = _generate_frame_points(profile, seq, rng)
        time_monotonic_ns = time.monotonic_ns()
        time_unix_ns = time.time_ns()
        packets = []
        for index in range(packets_per_frame):
            chunk = points[
                index * POINTS_PER_PACKET * POINT_BYTES : (index + 1)
                * POINTS_PER_PACKET
                * POINT_BYTES
            ]
            header = json.dumps(
                {
                    "fs": seq,
                    "pi": index,
                    "pc": packets_per_frame,
                    "tm": time_monotonic_ns,
                    "tu": time_unix_ns,
                    "dp": dropped_packets_total,
                }
            ).encode("utf-8")
            packets.append(
                struct.pack(_HEADER_LEN_STRUCT, len(header)) + header + chunk
            )
        return packets

    while True:
        now = time.monotonic()
        if scanning and outbox and now >= next_packet_at:
            # Drain every due packet; absolute-slot scheduling so select()
            # wakeup latency does not accumulate across the frame.
            while outbox and time.monotonic() >= next_packet_at:
                if not send_packet(outbox.pop(0)):
                    dropped_packets_total += 1
                next_packet_at += packet_spacing
        if scanning and now >= next_frame:
            # Flush stragglers at the boundary; only a full consumer-side
            # buffer (send failure) actually drops packets.
            for payload in outbox:
                if not send_packet(payload):
                    dropped_packets_total += 1
            outbox = make_frame_packets()
            next_packet_at = time.monotonic()
            seq += 1
            next_frame += period
            # If the host paused this process past a whole frame, skip the
            # missed slots rather than bursting stale frames.
            if next_frame < now - period:
                missed = int((now - next_frame) / period) + 1
                dropped_packets_total += missed * packets_per_frame
                seq += missed
                next_frame += missed * period
        if scanning:
            timeout = max(
                0.0,
                min(
                    next_frame - time.monotonic(),
                    (next_packet_at - time.monotonic()) if outbox else period,
                ),
            )
        else:
            timeout = 0.5
        readable, _, _ = select.select([sock], [], [], max(0.0, timeout))
        if not readable:
            continue
        try:
            data, _ = sock.recvfrom(65536)
            command = json.loads(data.decode("utf-8")).get("cmd")
        except (OSError, ValueError):
            command = None
        if command == "hello":
            info = {
                "type": "info",
                "model": profile.model,
                "serial": serial,
                "firmware": FIRMWARE,
                "frame_rate_hz": profile.frame_rate_hz,
                "points_per_frame": profile.points_per_frame,
                "points_per_second": profile.points_per_second,
                "horizontal_fov_deg": profile.horizontal_fov_deg,
                "vertical_fov_min_deg": profile.vertical_fov_min_deg,
                "vertical_fov_max_deg": profile.vertical_fov_max_deg,
                "range_min_m": profile.range_min_m,
                "range_max_m": profile.range_max_m,
            }
            send_packet(json.dumps(info).encode("utf-8"))
        elif command == "start":
            scanning = True
            outbox = []
            next_frame = time.monotonic()
        elif command == "stop":
            scanning = False
            outbox = []
        elif command == "bye":
            sock.close()
            Path(device_path).unlink(missing_ok=True)
            return


def _generate_frame_points(profile: LidarProfile, seq: int, rng) -> bytes:
    """One frame of packed float32 (x, y, z, intensity) points.

    Non-repetitive scanning pattern (like the Mid-360's rotating-mirror
    sweep): golden-angle azimuth progression that never exactly repeats,
    against a synthetic scene of walls with slow-moving structure.
    """
    import numpy as np

    n = profile.points_per_frame
    golden = math.pi * (3.0 - math.sqrt(5.0))
    base_index = np.arange(n, dtype=np.float64) + float(seq) * n
    azimuth = np.mod(base_index * golden, 2.0 * math.pi)
    v_min = math.radians(profile.vertical_fov_min_deg)
    v_max = math.radians(profile.vertical_fov_max_deg)
    elevation = v_min + np.mod(base_index * 0.61803398875, 1.0) * (v_max - v_min)

    phase = seq * 0.05
    wall = 3.0 + 0.7 * np.sin(3.0 * azimuth + phase)
    noise = rng.normal(0.0, 0.05, n)
    horizontal = np.clip(wall + noise, profile.range_min_m, profile.range_max_m)
    cos_elev = np.clip(np.cos(elevation), 0.05, None)
    slant = np.clip(horizontal / cos_elev, profile.range_min_m, profile.range_max_m)

    x = slant * np.cos(elevation) * np.cos(azimuth)
    y = slant * np.cos(elevation) * np.sin(azimuth)
    z = slant * np.sin(elevation)
    intensity = rng.uniform(0.0, 255.0, n)

    points = np.empty((n, 4), dtype=np.float32)
    points[:, 0] = x
    points[:, 1] = y
    points[:, 2] = z
    points[:, 3] = intensity
    return points.tobytes()


if __name__ == "__main__":
    _device_loop(
        device_path=sys.argv[1],
        client_path=sys.argv[2],
        model=sys.argv[3],
        seed=int(sys.argv[4]),
    )
