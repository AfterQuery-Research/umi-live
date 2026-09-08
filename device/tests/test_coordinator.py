"""Leader/follower control channel: command delivery, ack/time-handshake, unreachable."""

from __future__ import annotations

import socket
import threading
import time

from umi_flow_device import coordinator


def _drain(f: coordinator.Follower, timeout: float = 1.0) -> coordinator.Command:
    # The server enqueues AFTER replying, so poll with a short retry.
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        cmd = f.poll()
        if cmd is not None:
            return cmd
        time.sleep(0.01)
    raise AssertionError("no command received")


def test_leader_follower_roundtrip():
    f = coordinator.Follower(listen_port=0, host="127.0.0.1")
    f.start()
    try:
        leader = coordinator.Leader(f"127.0.0.1:{f.port}")

        result: list[dict | None] = []
        request = threading.Thread(target=lambda: result.append(leader.precheck()))
        request.start()
        cmd = _drain(f)
        assert cmd.cmd == coordinator.CMD_PRECHECK
        cmd.respond(True, result={"ok": True, "checks": []})
        request.join(timeout=1)
        assert result[0] is not None and result[0]["result"]["ok"] is True
        assert result[0]["protocol_version"] == coordinator.PROTOCOL_VERSION
        assert leader.peer_generation == f.generation

        result.clear()
        request = threading.Thread(target=lambda: result.append(leader.start_take("sess_1")))
        request.start()
        cmd = _drain(f)
        assert request.is_alive()  # no ack until the recorder applies START
        cmd.respond(True)
        request.join(timeout=1)
        ack = result[0]
        assert ack is not None and "follower_unix_ns" in ack
        assert cmd.cmd == coordinator.CMD_START
        assert cmd.session_id == "sess_1"
        assert cmd.leader_unix_ns is not None  # the time handshake rode along

        result.clear()
        request = threading.Thread(target=lambda: result.append(leader.stop_take()))
        request.start()
        cmd = _drain(f)
        assert cmd.cmd == coordinator.CMD_STOP
        cmd.respond(True)
        request.join(timeout=1)
        assert result[0] is not None

        f.set_status(state="idle", task="pour water", config_sha256="abc")
        status = leader.status()
        assert status is not None
        assert status["status"] == {
            "state": "idle",
            "task": "pour water",
            "config_sha256": "abc",
        }

        # Heartbeats are accepted only for the live prepared/recording session.
        f.set_status(state="recording", session_id="sess_1")
        assert leader.heartbeat("sess_1")["ok"] is True
        assert f.lease_expired("sess_1") is False
        assert f.lease_expired("sess_1", grace_s=-1) is True
        assert leader.heartbeat("different")["ok"] is False

        result.clear()
        request = threading.Thread(target=lambda: result.append(leader.cancel_take()))
        request.start()
        cmd = _drain(f)
        assert cmd.cmd == coordinator.CMD_CANCEL
        cmd.respond(True)
        request.join(timeout=1)
        assert result[0] is not None

        # ping acks but does NOT enqueue a command.
        assert leader.ping() is True
        time.sleep(0.05)
        assert f.poll() is None
    finally:
        f.close()


def test_leader_unreachable_returns_none():
    # Grab a definitely-closed port.
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    leader = coordinator.Leader(f"127.0.0.1:{port}", timeout_s=0.5)
    assert leader.precheck() is None
    assert leader.start_take("s") is None
    assert leader.stop_take() is None
    assert leader.ping() is False


def test_follower_rejects_stale_generation_without_enqueuing():
    f = coordinator.Follower(listen_port=0, host="127.0.0.1")
    f.start()
    try:
        leader = coordinator.Leader(f"127.0.0.1:{f.port}")
        leader.peer_generation = "old-process"
        reply = leader.start_take("sess_1")
        assert reply is not None
        assert reply["ok"] is False
        assert "generation changed" in reply["error"]
        assert f.poll() is None
    finally:
        f.close()


def test_duplicate_command_id_replays_ack_without_second_enqueue():
    f = coordinator.Follower(listen_port=0, host="127.0.0.1")
    f.start()
    try:
        leader = coordinator.Leader(f"127.0.0.1:{f.port}")
        command = {
            "cmd": coordinator.CMD_START,
            "session_id": "sess_1",
            "command_id": "fixed-command",
        }
        replies: list[dict | None] = []
        first = threading.Thread(target=lambda: replies.append(leader._send(command)))
        first.start()
        received = _drain(f)
        received.respond(True)
        first.join(timeout=1)

        replay = leader._send(command)
        assert replay == replies[0]
        assert f.poll() is None
    finally:
        f.close()


def test_transport_retry_reuses_command_id(monkeypatch):
    leader = coordinator.Leader("127.0.0.1:1")
    requests: list[dict] = []

    def exchange(request: dict):
        requests.append(dict(request))
        if len(requests) == 1:
            return None
        return {
            "ok": True,
            "protocol_version": coordinator.PROTOCOL_VERSION,
            "command_id": request["command_id"],
            "generation": "generation-1",
        }

    monkeypatch.setattr(leader, "_exchange", exchange)

    assert leader.commit_start("session-1") == {
        "ok": True,
        "protocol_version": coordinator.PROTOCOL_VERSION,
        "command_id": requests[0]["command_id"],
        "generation": "generation-1",
    }
    assert requests[0]["command_id"] == requests[1]["command_id"]
