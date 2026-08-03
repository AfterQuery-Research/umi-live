"""Vive receiver: per-tracker queue isolation (dual-rig data integrity)."""

from __future__ import annotations

import socket
import time

from robot_umi_device.vive import NetworkPoseSensor, encode_samples


def _wait_for(recv: NetworkPoseSensor, count: int, timeout: float = 2.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with recv._lock:
            total = sum(len(q) for q in recv._queues.values())
        if total >= count:
            return
        time.sleep(0.02)


def test_read_per_arm_does_not_consume_other_arms():
    """Two ViveRecorders share one receiver. Draining one tracker must not discard
    the other's samples — the bug that silently halved dual-rig Vive data."""
    recv = NetworkPoseSensor(port=0)  # OS-assigned port
    recv.start()
    try:
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        addr = ("127.0.0.1", recv.port)
        sender.sendto(encode_samples([("LHR-A", 1.0, [0.0] * 7)], 1.0), addr)
        sender.sendto(encode_samples([("LHR-B", 1.0, [1.0] * 7)], 1.0), addr)
        sender.sendto(encode_samples([("LHR-A", 2.0, [0.0] * 7)], 2.0), addr)
        _wait_for(recv, 3)

        a = recv.read("LHR-A")
        assert [s.name for s in a] == ["LHR-A", "LHR-A"]
        # Draining A must leave B's sample intact.
        b = recv.read("LHR-B")
        assert [s.name for s in b] == ["LHR-B"]
        # Everything consumed now; unknown arm reads empty.
        assert recv.read("LHR-A") == []
        assert recv.read() == []
    finally:
        recv.stop()


def test_read_all_drains_every_queue():
    recv = NetworkPoseSensor(port=0)
    recv.start()
    try:
        sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        addr = ("127.0.0.1", recv.port)
        sender.sendto(encode_samples([("LHR-A", 1.0, [0.0] * 7)], 1.0), addr)
        sender.sendto(encode_samples([("LHR-B", 1.0, [1.0] * 7)], 1.0), addr)
        _wait_for(recv, 2)

        names = sorted(s.name for s in recv.read())
        assert names == ["LHR-A", "LHR-B"]
        assert recv.read() == []
    finally:
        recv.stop()
