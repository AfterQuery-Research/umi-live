from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from umi_live_server.operation_lock import PipelineLeaseBusyError

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


def test_cli_can_defer_cleanly_when_pipeline_owns_lease(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    def busy(*_args, **_kwargs):
        raise PipelineLeaseBusyError("pipeline/maintenance lease is busy: test")

    monkeypatch.setattr(archive, "archive_published_raw", busy)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--data-dir",
            str(tmp_path),
            "--repo-id",
            "brandonyang/artem-pour-water",
            "--apply",
            "--skip-if-busy",
        ],
    )

    archive.main()

    report = json.loads(capsys.readouterr().out)
    assert report["state"] == "deferred"
    assert report["reason"] == "pipeline_busy"


def test_cli_surfaces_busy_lease_without_skip_flag(tmp_path: Path, monkeypatch) -> None:
    def busy(*_args, **_kwargs):
        raise PipelineLeaseBusyError("pipeline/maintenance lease is busy: test")

    monkeypatch.setattr(archive, "archive_published_raw", busy)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--data-dir",
            str(tmp_path),
            "--repo-id",
            "brandonyang/artem-pour-water",
            "--apply",
        ],
    )

    with pytest.raises(PipelineLeaseBusyError, match="lease is busy"):
        archive.main()
