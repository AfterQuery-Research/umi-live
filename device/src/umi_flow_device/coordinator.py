"""Leader/follower control channel for coordinated two-Pi bimanual capture.

The leader Pi keeps the buttons; on each start/stop/cancel it signals the follower over a
tiny newline-delimited JSON socket so both Pis record the SAME take, tagged with a shared
``session_id`` (the key the server merges the two one-arm transport episodes on). The follower
runs headless, driving its own ``EpisodeRecorder`` from the commands it receives.

The reply to every command carries the follower's ``time.time_ns()`` — an independent
leader<->follower clock-offset estimate that rides into the manifests alongside chrony's
self-report (see ``ptp.chrony_status``).

stdlib sockets only (matches ``uploader``'s dependency-free style); the leader opens a
fresh short-lived connection per command, so a dropped link fails that one command
cleanly rather than wedging a long-lived socket.
"""

from __future__ import annotations

import json
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from queue import Empty, Queue

DEFAULT_PORT = 9140
_TIMEOUT_S = 5.0
_COMMAND_ACK_TIMEOUT_S = 30.0
_MAX_LINE = 65536

CMD_START = "start"
CMD_STOP = "stop"
CMD_CANCEL = "cancel"
CMD_PING = "ping"
CMD_PRECHECK = "precheck"
CMD_STATUS = "status"
CMD_PREPARE_START = "prepare_start"
CMD_COMMIT_START = "commit_start"
CMD_ABORT_START = "abort_start"
CMD_HEARTBEAT = "heartbeat"
PROTOCOL_VERSION = 2
_REPLAY_CACHE_SIZE = 256
PEER_LEASE_GRACE_S = 5.0


@dataclass
class Command:
    cmd: str
    session_id: str | None = None
    leader_unix_ns: int | None = None
    command_id: str | None = None
    expected_generation: str | None = None
    _reply: Queue[dict] | None = field(default=None, repr=False)
    _expired: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def expired(self) -> bool:
        return self._expired.is_set()

    def expire(self) -> None:
        self._expired.set()

    def respond(self, ok: bool, error: str = "", result: dict[str, object] | None = None) -> None:
        """Complete the command after the follower recorder applied it."""
        if self._reply is None:
            return
        reply: dict = {
            "ok": ok,
            "protocol_version": PROTOCOL_VERSION,
            "command_id": self.command_id,
            "follower_unix_ns": time.time_ns(),
        }
        if error:
            reply["error"] = error
        if result is not None:
            reply["result"] = result
        self._reply.put_nowait(reply)


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
        self.peer_generation: str | None = None

    def _send(self, obj: dict) -> dict | None:
        request = dict(obj)
        request.setdefault("protocol_version", PROTOCOL_VERSION)
        request.setdefault("command_id", uuid.uuid4().hex)
        if (
            request.get("cmd")
            in {
                CMD_START,
                CMD_PREPARE_START,
                CMD_COMMIT_START,
                CMD_ABORT_START,
                CMD_HEARTBEAT,
                CMD_STOP,
                CMD_CANCEL,
            }
            and self.peer_generation is not None
        ):
            request.setdefault("expected_generation", self.peer_generation)
        reply = self._exchange(request)
        if reply is None:
            # Reuse the same command_id: if the first acknowledgement was lost,
            # the follower returns its cached result instead of applying twice.
            reply = self._exchange(request)
        if reply is None:
            return None
        if reply.get("protocol_version") != PROTOCOL_VERSION:
            return {
                "ok": False,
                "error": "incompatible follower coordinator protocol",
                "protocol_version": reply.get("protocol_version"),
            }
        if reply.get("command_id") != request["command_id"]:
            return {
                "ok": False,
                "error": "follower acknowledgement command_id mismatch",
                "protocol_version": PROTOCOL_VERSION,
            }
        if reply.get("ok") and isinstance(reply.get("generation"), str):
            self.peer_generation = reply["generation"]
        return reply

    def _exchange(self, request: dict) -> dict | None:
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as s:
                command = request.get("cmd")
                applied = command not in {CMD_PING, CMD_STATUS}
                s.settimeout(
                    max(self.timeout_s, _COMMAND_ACK_TIMEOUT_S + 2) if applied else self.timeout_s
                )
                s.sendall((json.dumps(request) + "\n").encode())
                line = _read_line(s)
            reply = json.loads(line) if line else None
            return reply if isinstance(reply, dict) else None
        except (OSError, ValueError):
            return None

    def start_take(self, session_id: str, leader_unix_ns: int | None = None) -> dict | None:
        return self._send(
            {
                "cmd": CMD_START,
                "session_id": session_id,
                "leader_unix_ns": leader_unix_ns if leader_unix_ns is not None else time.time_ns(),
            }
        )

    def prepare_start(self, session_id: str) -> dict | None:
        return self._send(
            {
                "cmd": CMD_PREPARE_START,
                "session_id": session_id,
                "leader_unix_ns": time.time_ns(),
            }
        )

    def commit_start(self, session_id: str) -> dict | None:
        return self._send(
            {
                "cmd": CMD_COMMIT_START,
                "session_id": session_id,
                "leader_unix_ns": time.time_ns(),
            }
        )

    def abort_start(self, session_id: str) -> dict | None:
        return self._send({"cmd": CMD_ABORT_START, "session_id": session_id})

    def heartbeat(self, session_id: str) -> dict | None:
        return self._send({"cmd": CMD_HEARTBEAT, "session_id": session_id})

    def precheck(self) -> dict | None:
        return self._send({"cmd": CMD_PRECHECK})

    def status(self) -> dict | None:
        return self._send({"cmd": CMD_STATUS})

    def stop_take(self) -> dict | None:
        return self._send({"cmd": CMD_STOP})

    def cancel_take(self) -> dict | None:
        return self._send({"cmd": CMD_CANCEL})

    def ping(self) -> bool:
        reply = self._send({"cmd": CMD_PING})
        return bool(reply and reply.get("ok"))


class Follower:
    """Server side: a background thread accepts commands and enqueues them; the follower
    recorder loop drains the queue. Replies immediately (with the time handshake) so a
    slow recorder never blocks the leader's ack."""

    def __init__(
        self,
        listen_port: int = DEFAULT_PORT,
        host: str = "0.0.0.0",
        *,
        status: dict[str, object] | None = None,
    ):
        self.host = host
        self.port = listen_port
        self._q: Queue[Command] = Queue()
        self._srv: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.generation = uuid.uuid4().hex
        self._status = dict(status or {})
        self._status_lock = threading.Lock()
        self._replies: dict[str, dict] = {}
        self._heartbeat_lock = threading.Lock()
        self._heartbeat_session: str | None = None
        self._last_heartbeat = 0.0

    def set_status(self, **values: object) -> None:
        with self._status_lock:
            self._status.update(values)

    def status(self) -> dict[str, object]:
        with self._status_lock:
            return dict(self._status)

    def lease_expired(self, session_id: str, grace_s: float = PEER_LEASE_GRACE_S) -> bool:
        with self._heartbeat_lock:
            return (
                self._heartbeat_session == session_id
                and time.monotonic() - self._last_heartbeat > grace_s
            )

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
            except TimeoutError:
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
                cmd = str(obj.get("cmd", ""))
                command_id = obj.get("command_id")
                if obj.get("protocol_version") != PROTOCOL_VERSION or not isinstance(
                    command_id, str
                ):
                    reply = {
                        "ok": False,
                        "error": "incompatible or malformed coordinator protocol",
                        "protocol_version": PROTOCOL_VERSION,
                        "command_id": command_id,
                        "generation": self.generation,
                        "follower_unix_ns": time.time_ns(),
                    }
                elif command_id in self._replies:
                    reply = self._replies[command_id]
                elif obj.get("expected_generation") not in {None, self.generation}:
                    reply = {
                        "ok": False,
                        "error": "follower generation changed; run precheck again",
                        "protocol_version": PROTOCOL_VERSION,
                        "command_id": command_id,
                        "generation": self.generation,
                        "follower_unix_ns": time.time_ns(),
                    }
                elif cmd == CMD_PING:
                    reply = {
                        "ok": True,
                        "protocol_version": PROTOCOL_VERSION,
                        "command_id": command_id,
                        "generation": self.generation,
                        "follower_unix_ns": time.time_ns(),
                    }
                elif cmd == CMD_STATUS:
                    reply = {
                        "ok": True,
                        "protocol_version": PROTOCOL_VERSION,
                        "command_id": command_id,
                        "generation": self.generation,
                        "follower_unix_ns": time.time_ns(),
                        "status": self.status(),
                    }
                elif cmd == CMD_HEARTBEAT:
                    session_id = obj.get("session_id")
                    status = self.status()
                    matches = (
                        isinstance(session_id, str)
                        and status.get("session_id") == session_id
                        and status.get("state") in {"prepared", "recording"}
                    )
                    if matches:
                        with self._heartbeat_lock:
                            self._heartbeat_session = session_id
                            self._last_heartbeat = time.monotonic()
                    reply = {
                        "ok": matches,
                        "error": "" if matches else "session lease mismatch",
                        "protocol_version": PROTOCOL_VERSION,
                        "command_id": command_id,
                        "generation": self.generation,
                        "follower_unix_ns": time.time_ns(),
                    }
                elif cmd in {
                    CMD_PRECHECK,
                    CMD_START,
                    CMD_PREPARE_START,
                    CMD_COMMIT_START,
                    CMD_ABORT_START,
                    CMD_STOP,
                    CMD_CANCEL,
                }:
                    if cmd == CMD_COMMIT_START and isinstance(obj.get("session_id"), str):
                        with self._heartbeat_lock:
                            self._heartbeat_session = obj["session_id"]
                            self._last_heartbeat = time.monotonic()
                    replies: Queue[dict] = Queue(maxsize=1)
                    queued = Command(
                        cmd=cmd,
                        session_id=obj.get("session_id"),
                        leader_unix_ns=obj.get("leader_unix_ns"),
                        command_id=command_id,
                        expected_generation=obj.get("expected_generation"),
                        _reply=replies,
                    )
                    self._q.put(queued)
                    try:
                        reply = replies.get(timeout=_COMMAND_ACK_TIMEOUT_S)
                    except Empty:
                        queued.expire()
                        reply = {
                            "ok": False,
                            "error": "follower recorder did not apply command",
                            "protocol_version": PROTOCOL_VERSION,
                            "command_id": command_id,
                            "follower_unix_ns": time.time_ns(),
                        }
                else:
                    reply = {
                        "ok": False,
                        "error": f"unknown command {cmd!r}",
                        "protocol_version": PROTOCOL_VERSION,
                        "command_id": command_id,
                        "follower_unix_ns": time.time_ns(),
                    }
                reply["generation"] = self.generation
                if isinstance(command_id, str):
                    self._replies[command_id] = reply
                    while len(self._replies) > _REPLAY_CACHE_SIZE:
                        self._replies.pop(next(iter(self._replies)))
                try:
                    conn.sendall((json.dumps(reply) + "\n").encode())
                except OSError:
                    pass

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
