"""Leader/follower control channel for coordinated two-Pi bimanual capture.

The leader Pi keeps the buttons; on each start/stop/cancel it signals the follower over a
tiny newline-delimited JSON socket so both Pis record the SAME take, tagged with a shared
``session_id`` (the key the server merges the two single-arm episodes on). The follower
runs headless, driving its own ``EpisodeRecorder`` from the commands it receives.

The reply to every command carries the follower's ``time.time_ns()`` — an independent
leader<->follower clock-offset estimate that rides into the manifests alongside chrony's
self-report (see ``ptp.chrony_status``). Solo (single-Pi) capture uses neither side.

stdlib sockets only (matches ``uploader``'s dependency-free style); the leader opens a
fresh short-lived connection per command, so a dropped link fails that one command
cleanly rather than wedging a long-lived socket.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass
from queue import Empty, Queue

DEFAULT_PORT = 9140
_TIMEOUT_S = 5.0
_MAX_LINE = 65536

CMD_START = "start"
CMD_STOP = "stop"
CMD_CANCEL = "cancel"
CMD_PING = "ping"


@dataclass
class Command:
    cmd: str
    session_id: str | None = None
    mode: str | None = None
    leader_unix_ns: int | None = None


def _read_line(sock: socket.socket) -> bytes:
    buf = b""
    while b"\n" not in buf and len(buf) < _MAX_LINE:
        try:
            chunk = sock.recv(4096)
        except OSError:
            break
        if not chunk:
            break
        buf += chunk
    return buf.split(b"\n", 1)[0]


class Leader:
    """Client side: signals the follower. Every method returns the follower's ack dict
    (``{"ok": true, "follower_unix_ns": ...}``) or None if it was unreachable."""

    def __init__(self, peer: str, *, timeout_s: float = _TIMEOUT_S):
        host, _, port = peer.partition(":")
        self.host = host
        self.port = int(port) if port else DEFAULT_PORT
        self.timeout_s = timeout_s

    def _send(self, obj: dict) -> dict | None:
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as s:
                s.settimeout(self.timeout_s)
                s.sendall((json.dumps(obj) + "\n").encode())
                line = _read_line(s)
            return json.loads(line) if line else None
        except (OSError, ValueError):
            return None

    def start_take(
        self, session_id: str, mode: str, leader_unix_ns: int | None = None
    ) -> dict | None:
        return self._send(
            {
                "cmd": CMD_START,
                "session_id": session_id,
                "mode": mode,
                "leader_unix_ns": leader_unix_ns if leader_unix_ns is not None else time.time_ns(),
            }
        )

    def stop_take(self) -> dict | None:
        return self._send({"cmd": CMD_STOP})

    def cancel_take(self) -> dict | None:
        return self._send({"cmd": CMD_CANCEL})

    def ping(self) -> bool:
        return self._send({"cmd": CMD_PING}) is not None


class Follower:
    """Server side: a background thread accepts commands and enqueues them; the follower
    recorder loop drains the queue. Replies immediately (with the time handshake) so a
    slow recorder never blocks the leader's ack."""

    def __init__(self, listen_port: int = DEFAULT_PORT, host: str = "0.0.0.0"):
        self.host = host
        self.port = listen_port
        self._q: Queue[Command] = Queue()
        self._srv: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    def start(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        self.port = srv.getsockname()[1]  # resolve an ephemeral (0) port to the real one
        srv.listen(4)
        srv.settimeout(0.5)
        self._srv = srv
        self._thread = threading.Thread(target=self._serve, name="follower-ctl", daemon=True)
        self._thread.start()

    def _serve(self) -> None:
        assert self._srv is not None
        while not self._stop.is_set():
            try:
                conn, _addr = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            with conn:
                conn.settimeout(_TIMEOUT_S)
                line = _read_line(conn)
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                # Reply first (the ack IS the time handshake), then enqueue non-ping cmds.
                try:
                    conn.sendall((json.dumps({"ok": True, "follower_unix_ns": time.time_ns()}) + "\n").encode())
                except OSError:
                    pass
                cmd = str(obj.get("cmd", ""))
                if cmd and cmd != CMD_PING:
                    self._q.put(
                        Command(
                            cmd=cmd,
                            session_id=obj.get("session_id"),
                            mode=obj.get("mode"),
                            leader_unix_ns=obj.get("leader_unix_ns"),
                        )
                    )

    def poll(self) -> Command | None:
        try:
            return self._q.get_nowait()
        except Empty:
            return None

    def close(self) -> None:
        self._stop.set()
        if self._srv is not None:
            self._srv.close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
