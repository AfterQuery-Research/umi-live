from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "archive_published_raw.py"
spec = importlib.util.spec_from_file_location("archive_published_raw", SCRIPT)
assert spec is not None and spec.loader is not None
archive = importlib.util.module_from_spec(spec)
sys.modules["archive_published_raw"] = archive
spec.loader.exec_module(archive)


def test_post_delete_server_archive_resumes_receipts(tmp_path: Path) -> None:
    raw = tmp_path / "raw_episodes" / "paired_sess_1"
    raw.mkdir(parents=True)
    intent_path = tmp_path / "quarantine" / "archived-raw-cleanup-20260904" / "intent.json"
    archive._write_json(
        intent_path,
        {
            "state": "verified",
            "episode_id": raw.name,
            "files": [{"name": name, "allocated_bytes": 4096} for name in archive.PAYLOAD_NAMES],
        },
    )

    resumed, removed = archive._resume_archive_intents(tmp_path)

    assert resumed == 1
    assert removed == 4 * 4096
    assert archive._load_json(intent_path)["state"] == "complete"
    assert archive._load_json(raw / archive.ARCHIVED_MARKER)["state"] == "complete"
