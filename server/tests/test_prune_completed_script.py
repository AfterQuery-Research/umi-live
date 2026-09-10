from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prune_completed.py"
spec = importlib.util.spec_from_file_location("prune_completed_script", SCRIPT)
assert spec is not None and spec.loader is not None
maintenance = importlib.util.module_from_spec(spec)
sys.modules["prune_completed_script"] = maintenance
spec.loader.exec_module(maintenance)


def test_admission_ready_does_not_run_raw_archive(tmp_path: Path, monkeypatch) -> None:
    required = 80 * maintenance.GIB
    monkeypatch.setattr(maintenance, "admission_required_bytes", lambda *_args: required)
    monkeypatch.setattr(
        maintenance.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=required),
    )
    monkeypatch.setattr(
        maintenance.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("must not run")),
    )

    result = maintenance.ensure_admission_headroom(
        tmp_path,
        archive_target_gib=96,
        archive_min_age_hours=24,
    )

    assert result.free_bytes == required
    assert result.required_free_bytes == required
    assert result.raw_archive_state == "not_needed"


def test_blocked_admission_runs_verified_archiver_to_high_watermark(
    tmp_path: Path, monkeypatch
) -> None:
    required = 80 * maintenance.GIB
    free_values = iter((76 * maintenance.GIB, 96 * maintenance.GIB))
    calls: list[list[str]] = []
    monkeypatch.setenv("HF_REPO_ID", "brandonyang/artem-pour-water")
    monkeypatch.setattr(maintenance, "admission_required_bytes", lambda *_args: required)
    monkeypatch.setattr(
        maintenance.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=next(free_values)),
    )

    def run(command: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, '{"state": "complete"}\n', "")

    monkeypatch.setattr(maintenance.subprocess, "run", run)

    result = maintenance.ensure_admission_headroom(
        tmp_path,
        archive_target_gib=96,
        archive_min_age_hours=24,
    )

    assert result.free_bytes == 96 * maintenance.GIB
    assert result.raw_archive_state == "complete"
    assert len(calls) == 1
    assert calls[0][-2:] == ["--apply", "--skip-if-busy"]
    assert calls[0][calls[0].index("--target-free-gib") + 1] == "96.0"


def test_archive_target_cannot_undercut_admission_requirement(tmp_path: Path, monkeypatch) -> None:
    required = 84 * maintenance.GIB
    free_values = iter((76 * maintenance.GIB, 84 * maintenance.GIB))
    calls: list[list[str]] = []
    monkeypatch.setenv("HF_REPO_ID", "brandonyang/artem-pour-water")
    monkeypatch.setattr(maintenance, "admission_required_bytes", lambda *_args: required)
    monkeypatch.setattr(
        maintenance.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=next(free_values)),
    )

    def run(command: list[str], **_kwargs) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, '{"state": "complete"}\n', "")

    monkeypatch.setattr(maintenance.subprocess, "run", run)

    maintenance.ensure_admission_headroom(
        tmp_path,
        archive_target_gib=64,
        archive_min_age_hours=24,
    )

    assert calls[0][calls[0].index("--target-free-gib") + 1] == "84.0"
