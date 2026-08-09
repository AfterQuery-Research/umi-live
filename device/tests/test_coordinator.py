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
    assert leader.start_take("s") is None
    assert leader.stop_take() is None
    assert leader.ping() is False
