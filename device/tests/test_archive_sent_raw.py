from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "archive_sent_raw.py"
spec = importlib.util.spec_from_file_location("archive_sent_raw", SCRIPT)
assert spec is not None and spec.loader is not None
archive = importlib.util.module_from_spec(spec)
sys.modules["archive_sent_raw"] = archive
spec.loader.exec_module(archive)


def test_interrupted_post_delete_archive_finishes_receipt(tmp_path: Path) -> None:
    episode = tmp_path / "episode_1"
    episode.mkdir()
    intent = {
        "state": "in_progress",
        "files": [
            {"name": "umi1_camera.mjpeg.avi", "size": 10, "allocated_bytes": 4096},
            {"name": "umi1_lidar.pcap", "size": 20, "allocated_bytes": 4096},
        ],
    }
    archive._write_json(episode / archive.IN_PROGRESS_MARKER, intent)

    removed = archive._resume_durable_intent(episode, archive.PAYLOADS_BY_ROLE["umi1"])

    assert removed == 8192
    assert not (episode / archive.IN_PROGRESS_MARKER).exists()
    receipt = archive._json(episode / archive.ARCHIVED_MARKER)
    assert receipt is not None
    assert receipt["state"] == "complete"


def test_interrupted_mixed_archive_refuses_further_mutation(tmp_path: Path) -> None:
    episode = tmp_path / "episode_1"
    episode.mkdir()
    archive._write_json(episode / archive.IN_PROGRESS_MARKER, {"files": []})
    (episode / "umi1_lidar.pcap").write_bytes(b"still here")

    try:
        archive._resume_durable_intent(episode, archive.PAYLOADS_BY_ROLE["umi1"])
    except RuntimeError as exc:
        assert "mixed payload state" in str(exc)
    else:
        raise AssertionError("mixed archive state was not rejected")

    assert (episode / "umi1_lidar.pcap").exists()
