"""Vive pose reception on the Pi: UDP wire codec + kernel-stamped receiver.

The box-side sender (pose_udp_sender, unchanged from the previous stack) ships
already-converted body-frame samples (name + [x,y,z,qx,qy,qz,qw]) as JSON datagrams.
This module receives them and stamps each datagram with the KERNEL receive time
(``SO_TIMESTAMPNS``): under heavy load the Python reader thread can be starved for
~1 s, and read-time stamping would bunch a second of packets at one late instant.
The kernel stamps on arrival (CLOCK_REALTIME — the PTP-disciplined axis), and a
large ``SO_RCVBUF`` keeps the backlog from dropping meanwhile.

Recording is RAW (record raw, process later): both the sender's own timestamp and
the kernel arrival time are kept per sample; EE calibration, zeroing, and frame
conversion happen server-side. The receiver is session-long; per-episode capture is
the :class:`robot_umi_device.sensors.ViveRecorder`'s job.
"""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass

PROTOCOL = 1
RECV_BUF = 65535
RCVBUF_BYTES = 8 << 20  # 8 MiB: hold seconds of poses if the reader thread is starved
_SO_TIMESTAMPNS = getattr(socket, "SO_TIMESTAMPNS", 35)
_SCM_TIMESTAMPNS = getattr(socket, "SCM_TIMESTAMPNS", _SO_TIMESTAMPNS)


@dataclass(frozen=True)
class PoseSample:
    """One raw pose sample: tracker/arm name + pos(3)+quat_xyzw(4), two clocks."""

    name: str
    recv_unix_ns: int  # kernel arrival time (PTP axis)
    sender_ts: float  # the box's own sample clock (kept verbatim; not our axis)
    data: tuple[float, ...]  # [x, y, z, qx, qy, qz, qw]


def encode_samples(samples: list[tuple[str, float, list[float]]], send_ts: float) -> bytes:
    """Pack (name, ts, data) tuples into one datagram. Sender-side; kept for tests."""
    return json.dumps(
        {
            "v": PROTOCOL,
            "t": float(send_ts),
            "s": [{"n": n, "ts": float(ts), "d": [float(x) for x in d]} for n, ts, d in samples],
        }
    ).encode("utf-8")


def decode_samples(data: bytes, recv_unix_ns: int) -> list[PoseSample]:
    pkt = json.loads(data.decode("utf-8"))
    if pkt.get("v") != PROTOCOL:
        raise ValueError(f"unknown pose packet version {pkt.get('v')!r}")
    return [
        PoseSample(
            name=str(e["n"]),
            recv_unix_ns=recv_unix_ns,
            sender_ts=float(e["ts"]),
            data=tuple(float(x) for x in e["d"]),
        )
        for e in pkt["s"]
    ]


def _kernel_stamp_ns(ancdata: list, fallback_ns: int) -> int:
    for level, typ, cdata in ancdata:
        if level == socket.SOL_SOCKET and typ == _SCM_TIMESTAMPNS and len(cdata) >= 16:
            sec, nsec = struct.unpack("qq", cdata[:16])
            return sec * 1_000_000_000 + nsec
    return fallback_ns


class NetworkPoseSensor:
    """Session-long UDP listener for Vive poses; one instance serves all trackers.

    Incoming samples are dispatched into PER-TRACKER queues keyed by the sample
    name, so each UMI's :class:`ViveRecorder` can drain only its own tracker via
    ``read(arm_name)`` without discarding a co-recorder's samples. ``read()`` with
    no argument drains every queue (single-tracker rigs, tests)."""

    def __init__(self, port: int = 9130, host: str = "0.0.0.0", queue_size: int = 8192):
        self.port = port
        self.host = host
        self._queue_size = queue_size
        self._queues: dict[str, deque[PoseSample]] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._kernel_ts = False

    def _queue_for(self, name: str) -> deque[PoseSample]:
        q = self._queues.get(name)
        if q is None:
            q = deque(maxlen=self._queue_size)
            self._queues[name] = q
        return q

    def start(self) -> None:
        if self._sock is not None:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, RCVBUF_BYTES)
        except OSError:
            pass
        try:
            sock.setsockopt(socket.SOL_SOCKET, _SO_TIMESTAMPNS, 1)
            self._kernel_ts = True
        except OSError:
            self._kernel_ts = False
        sock.bind((self.host, self.port))
        self.port = sock.getsockname()[1]  # resolve port 0 (OS-assigned, for tests)
        sock.settimeout(0.5)
        self._sock = sock
        self._stop.clear()
        self._thread = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def _recv_loop(self) -> None:
        ancbufsize = socket.CMSG_SPACE(16)
        while not self._stop.is_set():
            try:
                if self._kernel_ts:
                    data, ancdata, _flags, _addr = self._sock.recvmsg(RECV_BUF, ancbufsize)  # type: ignore[union-attr]
                    stamp_ns = _kernel_stamp_ns(ancdata, time.time_ns())
                else:
                    data, _ = self._sock.recvfrom(RECV_BUF)  # type: ignore[union-attr]
                    stamp_ns = time.time_ns()
            except (TimeoutError, OSError):
                continue
            try:
                samples = decode_samples(data, stamp_ns)
            except (ValueError, KeyError, TypeError, UnicodeDecodeError):
                continue  # one malformed datagram must not kill the stream
            with self._lock:
                for s in samples:
                    self._queue_for(s.name).append(s)

    def read(self, name: str | None = None) -> list[PoseSample]:
        """Drain and return buffered samples. With ``name``, only that tracker's
        queue (non-destructive to other trackers); without, every queue."""
        with self._lock:
            if name is not None:
                q = self._queues.get(name)
                if q is None:
                    return []
                out = list(q)
                q.clear()
                return out
            out = []
            for q in self._queues.values():
                out.extend(q)
                q.clear()
            return out

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        with self._lock:
            self._queues.clear()
