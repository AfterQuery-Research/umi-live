from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import yaml
from umi_flow_core import layout, manifest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "recover_interrupted_episode.py"
spec = importlib.util.spec_from_file_location("recover_interrupted_episode", SCRIPT)
assert spec is not None and spec.loader is not None
recover_module = importlib.util.module_from_spec(spec)
sys.modules["recover_interrupted_episode"] = recover_module
spec.loader.exec_module(recover_module)


def interrupted_episode(tmp_path: Path) -> tuple[Path, Path, Path]:
    spool = tmp_path / "spool"
    episode = spool / "in_progress" / "episode_20260914T201233Z_umi2_918511547"
    snapshot = episode / "config_snapshot"
    snapshot.mkdir(parents=True)
    (spool / "complete").mkdir()
    (spool / ".recorder-status.json").write_text(
        json.dumps({"state": "idle", "session_id": None}) + "\n"
    )
    config = {
        "version": 1,
        "task": "pour water",
        "umis": {
            "umi2": {
                "camera": {
                    "device": "/dev/video0",
                    "width": 800,
                    "height": 600,
                    "fps": 30,
                    "calib": "config/calib/umi2_calib.npz",
                },
                "lidar": {"iface": "eth0"},
                "gripper": {"tag_ids": [0, 1]},
            }
        },
    }
    (snapshot / "umi.yaml").write_text(yaml.safe_dump(config))
    (episode / "events.jsonl").write_text(json.dumps({"event": "start", "time_unix_ns": 1}) + "\n")
    (episode / "umi2_camera.mjpeg.avi").write_bytes(b"AVI")
    (episode / "umi2_camera_frames.jsonl").write_text('{"frame":0}\n')
    (episode / "umi2_camera_stats.json").write_text(
        json.dumps({"camera_dataset_eligible": True, "camera_frame_count": 1})
    )
    (episode / "umi2_lidar.pcap").write_bytes(b"pcap evidence")
    proc_root = tmp_path / "proc"
    proc_root.mkdir()
    return spool, episode, proc_root


def test_recovery_is_dry_run_by_default(tmp_path: Path):
    spool, episode, proc_root = interrupted_episode(tmp_path)
    result = recover_module.recover(
        spool,
        episode,
        role="umi2",
        session_id="sess_1",
        reason="service stopped during take",
        apply=False,
        proc_root=proc_root,
    )
    assert result["state"] == "planned"
    assert episode.exists()
    assert not (episode / layout.MANIFEST_FILE).exists()


def test_recovery_preserves_files_and_marks_episode_ineligible(tmp_path: Path):
    spool, episode, proc_root = interrupted_episode(tmp_path)
    result = recover_module.recover(
        spool,
        episode,
        role="umi2",
        session_id="sess_1",
        reason="service stopped during take",
        apply=True,
        proc_root=proc_root,
    )
    final = spool / "complete" / episode.name
    assert result["state"] == "complete"
    assert final.exists() and not episode.exists()
    assert (final / "umi2_camera.mjpeg.avi").read_bytes() == b"AVI"
    assert (final / "umi2_lidar.pcap").read_bytes() == b"pcap evidence"
    recovered = manifest.read_manifest(final / layout.MANIFEST_FILE)
    assert recovered["dataset_eligible"] is False
    assert recovered["session_id"] == "sess_1"
    assert recovered["recovery"]["kind"] == "interrupted_recorder"
    assert all(entry["error"] for entry in recovered["sensors"].values())
    assert (final / layout.CHECKSUMS_FILE).is_file()
    assert (final / layout.COMPLETE_FILE).is_file()


def test_recovery_synthesizes_missing_camera_stats(tmp_path: Path):
    spool, episode, proc_root = interrupted_episode(tmp_path)
    (episode / "umi2_camera_stats.json").unlink()

    planned = recover_module.recover(
        spool,
        episode,
        role="umi2",
        session_id="sess_1",
        reason="abrupt recorder loss",
        apply=False,
        proc_root=proc_root,
    )
    assert planned["state"] == "planned"
    assert planned["camera_stats_source"] == "missing"

    recover_module.recover(
        spool,
        episode,
        role="umi2",
        session_id="sess_1",
        reason="abrupt recorder loss",
        apply=True,
        proc_root=proc_root,
    )
    final = spool / "complete" / episode.name
    stats = json.loads((final / "umi2_camera_stats.json").read_text())
    assert stats["camera_dataset_eligible"] is False
    assert stats["camera_frame_count"] == 1
    assert stats["camera_stats_recovered"] is True
    assert stats["camera_stats_source"] == "missing"
    recovered = manifest.read_manifest(final / layout.MANIFEST_FILE)
    assert recovered["dataset_eligible"] is False
    assert recovered["sensors"]["umi2/camera"]["error"]


def test_recovery_refuses_non_idle_recorder(tmp_path: Path):
    spool, episode, proc_root = interrupted_episode(tmp_path)
    (spool / ".recorder-status.json").write_text(
        json.dumps({"state": "recording", "session_id": "sess_live"}) + "\n"
    )
    with pytest.raises(RuntimeError, match="not idle"):
        recover_module.recover(
            spool,
            episode,
            role="umi2",
            session_id="sess_1",
            reason="service stopped during take",
            apply=True,
            proc_root=proc_root,
        )


def test_recovery_refuses_live_capture_process(tmp_path: Path):
    spool, episode, proc_root = interrupted_episode(tmp_path)
    process = proc_root / "123"
    process.mkdir()
    (process / "cmdline").write_bytes(
        b"tcpdump\0-w\0" + str(spool / "in_progress" / "live.pcap").encode()
    )
    with pytest.raises(RuntimeError, match="capture processes are still active"):
        recover_module.recover(
            spool,
            episode,
            role="umi2",
            session_id="sess_1",
            reason="service stopped during take",
            apply=True,
            proc_root=proc_root,
        )
