from __future__ import annotations

import json
import os
from pathlib import Path

from umi_flow_server.retention import DEFAULT_RETENTION_SECONDS, PRUNED_FILE, prune_completed


def make_session(data_dir: Path, session_id: str, *, age_seconds: int, complete: bool) -> Path:
    derived = data_dir / "derived" / f"paired_{session_id}"
    for role in ("umi1", "umi2"):
        decode = derived / role / "decode"
        (decode / "bag").mkdir(parents=True)
        (decode / "bag" / "bag_0.mcap").write_bytes(b"decoded")
        (decode / "COMPLETE").write_text("complete\n")
    (derived / "glim" / "map").mkdir(parents=True)
    (derived / "glim" / "map" / "keep.ply").write_bytes(b"slam")
    if complete:
        marker = derived / "dataset" / "COMPLETE"
        marker.parent.mkdir(parents=True)
        marker.write_text("complete\n")
        os.utime(marker, ns=(1_000_000_000, 1_000_000_000))

    for role in ("umi1", "umi2"):
        source = data_dir / "paired_sources" / f"episode_{session_id}_{role}"
        source.mkdir(parents=True)
        (source / "camera.avi").write_bytes(b"source")
        (source / "manifest.json").write_text(json.dumps({"session_id": session_id, "role": role}))
    raw = data_dir / "raw_episodes" / f"paired_{session_id}"
    raw.mkdir(parents=True)
    (raw / "canonical.pcap").write_bytes(b"canonical")
    return derived


def test_prunes_only_reconstructible_artifacts_after_retention(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    derived = make_session(data_dir, "sess_old", age_seconds=10_000, complete=True)

    result = prune_completed(data_dir, retention_seconds=3_600, now_ns=10_000_000_000_000)

    assert result.eligible_sessions == 1
    assert result.source_dirs_removed == 2
    assert result.decode_dirs_pruned == 2
    assert not any((data_dir / "paired_sources").iterdir())
    for role in ("umi1", "umi2"):
        decode = derived / role / "decode"
        assert {item.name for item in decode.iterdir()} == {"COMPLETE", PRUNED_FILE}
    assert (derived / "glim/map/keep.ply").read_bytes() == b"slam"
    assert (data_dir / "raw_episodes/paired_sess_old/canonical.pcap").read_bytes() == b"canonical"


def test_keeps_young_and_incomplete_sessions(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    young = make_session(data_dir, "sess_young", age_seconds=0, complete=True)
    incomplete = make_session(data_dir, "sess_incomplete", age_seconds=0, complete=False)
    marker = young / "dataset" / "COMPLETE"
    os.utime(marker, ns=(9_000_000_000_000, 9_000_000_000_000))

    result = prune_completed(data_dir, retention_seconds=3_600, now_ns=10_000_000_000_000)

    assert result.eligible_sessions == 0
    assert (young / "umi1/decode/bag/bag_0.mcap").exists()
    assert (incomplete / "umi1/decode/bag/bag_0.mcap").exists()
    assert len(list((data_dir / "paired_sources").iterdir())) == 4


def test_default_retention_expires_after_thirty_minutes(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    now_ns = 10_000_000_000_000
    for name, age_minutes in (("sess_29min", 29), ("sess_31min", 31)):
        derived = make_session(data_dir, name, age_seconds=0, complete=True)
        completed_ns = now_ns - age_minutes * 60 * 1_000_000_000
        os.utime(derived / "dataset/COMPLETE", ns=(completed_ns, completed_ns))

    result = prune_completed(data_dir, now_ns=now_ns)

    assert result.eligible_sessions == 1
    assert result.source_dirs_removed == 2
    assert (data_dir / "derived/paired_sess_29min/umi1/decode/bag").exists()
    assert not (data_dir / "derived/paired_sess_31min/umi1/decode/bag").exists()


def test_is_idempotent_and_preserves_incomplete_source_pair(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    derived = make_session(data_dir, "sess_partial", age_seconds=10_000, complete=True)
    follower = data_dir / "paired_sources/episode_sess_partial_umi2"
    for item in follower.iterdir():
        item.unlink()
    follower.rmdir()
    messages: list[str] = []

    first = prune_completed(
        data_dir,
        retention_seconds=0,
        now_ns=10_000_000_000_000,
        log=messages.append,
    )
    second = prune_completed(data_dir, retention_seconds=0, now_ns=10_000_000_000_000)

    assert first.decode_dirs_pruned == 2
    assert first.source_dirs_removed == 0
    assert second.decode_dirs_pruned == 0
    assert (data_dir / "paired_sources/episode_sess_partial_umi1").exists()
    assert (derived / "umi1/decode" / PRUNED_FILE).exists()
    assert messages and "kept incomplete paired sources" in messages[0]


def test_retention_is_wired_into_server_provisioning() -> None:
    root = Path(__file__).resolve().parents[2]
    installer = (root / "server/scripts/install_server.sh").read_text()
    service = (root / "config/systemd/umi-retention.service").read_text()
    timer = (root / "config/systemd/umi-retention.timer").read_text()

    assert "umi-retention.service" in installer
    assert "umi-retention.timer" in installer
    assert "enable --now umi-retention.timer" in installer
    assert DEFAULT_RETENTION_SECONDS == 1_800
    assert "--retention-hours 0.5" in service
    assert "TimeoutStartSec=30min" in service
    assert "OnUnitActiveSec=5min" in timer
