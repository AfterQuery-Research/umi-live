#!/usr/bin/env python3
"""Prune reconstructible server artifacts after a short recovery window."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from umi_live_server.retention import DEFAULT_RETENTION_SECONDS, prune_completed
from umi_live_server.server import ServerConfig, admission_required_bytes

GIB = 1024**3
DEFAULT_ARCHIVE_TARGET_GIB = 96.0
DEFAULT_ARCHIVE_MIN_AGE_HOURS = 24.0
DEFAULT_ARCHIVE_WAIT_SECONDS = 10 * 60.0


@dataclass(frozen=True)
class AdmissionMaintenance:
    required_free_bytes: int
    cleanup_trigger_bytes: int
    free_bytes: int
    raw_archive_state: str


def _archive_state(stdout: str) -> str:
    for line in reversed(stdout.splitlines()):
        try:
            value = json.loads(line)
        except (TypeError, ValueError):
            continue
        if isinstance(value, dict) and isinstance(value.get("state"), str):
            return value["state"]
    return "unknown"


def ensure_admission_headroom(
    data_dir: Path,
    *,
    archive_target_gib: float,
    archive_min_age_hours: float,
    archive_wait_seconds: float = DEFAULT_ARCHIVE_WAIT_SECONDS,
) -> AdmissionMaintenance:
    """Archive verified old raw payloads before the ingest admission gate is blocked."""
    recording_reserve_gib = int(os.environ.get("UMI_LIVE_RECORDING_RESERVE_GIB", "64"))
    cfg = ServerConfig(
        data_dir=data_dir,
        recording_reserve_bytes=recording_reserve_gib * GIB,
    )
    required_free_bytes = admission_required_bytes(cfg, cfg.max_upload_bytes)
    # Start no later than the configured high-water target. The timer must reclaim
    # at the warning line, not wait until only one upload remains before rejection.
    cleanup_trigger_bytes = max(
        required_free_bytes + cfg.max_upload_bytes,
        int(archive_target_gib * GIB),
    )
    free_bytes = shutil.disk_usage(data_dir).free
    if free_bytes >= cleanup_trigger_bytes:
        return AdmissionMaintenance(
            required_free_bytes,
            cleanup_trigger_bytes,
            free_bytes,
            "not_needed",
        )

    repo_id = os.environ.get("HF_REPO_ID")
    if not repo_id:
        raise RuntimeError(
            "ingest admission is blocked and HF_REPO_ID is unavailable for verified raw archival"
        )

    # Never configure a cleanup target below the actual admission requirement.
    target_free_bytes = max(int(archive_target_gib * GIB), cleanup_trigger_bytes)
    command = [
        sys.executable,
        str(Path(__file__).with_name("archive_published_raw.py")),
        "--data-dir",
        str(data_dir),
        "--repo-id",
        repo_id,
        "--target-free-gib",
        str(target_free_bytes / GIB),
        "--min-age-hours",
        str(archive_min_age_hours),
        "--apply",
        "--wait-if-busy-seconds",
        str(archive_wait_seconds),
        "--skip-if-busy",
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n", flush=True)
    if completed.stderr:
        print(
            completed.stderr,
            end="" if completed.stderr.endswith("\n") else "\n",
            file=sys.stderr,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"verified raw archival failed with exit status {completed.returncode}")

    free_bytes = shutil.disk_usage(data_dir).free
    return AdmissionMaintenance(
        required_free_bytes,
        cleanup_trigger_bytes,
        free_bytes,
        _archive_state(completed.stdout),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("DATA_DIR", "data")),
    )
    parser.add_argument(
        "--retention-hours",
        type=float,
        default=DEFAULT_RETENTION_SECONDS / 3600,
    )
    parser.add_argument(
        "--archive-target-free-gib",
        type=float,
        default=float(
            os.environ.get("UMI_LIVE_ARCHIVE_TARGET_FREE_GIB", DEFAULT_ARCHIVE_TARGET_GIB)
        ),
    )
    parser.add_argument(
        "--archive-min-age-hours",
        type=float,
        default=float(
            os.environ.get("UMI_LIVE_ARCHIVE_MIN_AGE_HOURS", DEFAULT_ARCHIVE_MIN_AGE_HOURS)
        ),
    )
    parser.add_argument(
        "--archive-wait-seconds",
        type=float,
        default=float(
            os.environ.get("UMI_LIVE_ARCHIVE_WAIT_SECONDS", DEFAULT_ARCHIVE_WAIT_SECONDS)
        ),
    )
    args = parser.parse_args()
    if (
        args.archive_target_free_gib <= 0
        or args.archive_min_age_hours < 0
        or args.archive_wait_seconds < 0
    ):
        parser.error("archive target must be positive; minimum age and wait cannot be negative")
    summary = prune_completed(
        args.data_dir,
        retention_seconds=args.retention_hours * 3600,
    )
    admission = ensure_admission_headroom(
        args.data_dir,
        archive_target_gib=args.archive_target_free_gib,
        archive_min_age_hours=args.archive_min_age_hours,
        archive_wait_seconds=args.archive_wait_seconds,
    )
    print(
        "retention complete: "
        f"eligible={summary.eligible_sessions} "
        f"source_dirs={summary.source_dirs_removed} "
        f"decode_dirs={summary.decode_dirs_pruned} "
        f"files={summary.files_removed} bytes={summary.bytes_removed} "
        f"raw_archive={admission.raw_archive_state} "
        f"admission_ready={admission.free_bytes >= admission.required_free_bytes} "
        f"free_bytes={admission.free_bytes} "
        f"required_free_bytes={admission.required_free_bytes} "
        f"cleanup_trigger_bytes={admission.cleanup_trigger_bytes}",
        flush=True,
    )


if __name__ == "__main__":
    main()
