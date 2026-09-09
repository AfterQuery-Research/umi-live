#!/usr/bin/env python3
"""Prune reconstructible server artifacts after a short recovery window."""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from umi_flow_server.retention import DEFAULT_RETENTION_SECONDS, prune_completed
from umi_flow_server.server import ServerConfig, admission_required_bytes


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
    args = parser.parse_args()
    summary = prune_completed(
        args.data_dir,
        retention_seconds=args.retention_hours * 3600,
    )
    recording_reserve_gib = int(os.environ.get("UMI_FLOW_RECORDING_RESERVE_GIB", "64"))
    cfg = ServerConfig(
        data_dir=args.data_dir,
        recording_reserve_bytes=recording_reserve_gib * 1024**3,
    )
    # The configured maximum upload is the conservative paired-take estimate used
    # by deployed clients. Report readiness from the ingest admission calculation.
    required_free_bytes = admission_required_bytes(cfg, cfg.max_upload_bytes)
    free_bytes = shutil.disk_usage(args.data_dir).free
    print(
        "retention complete: "
        f"eligible={summary.eligible_sessions} "
        f"source_dirs={summary.source_dirs_removed} "
        f"decode_dirs={summary.decode_dirs_pruned} "
        f"files={summary.files_removed} bytes={summary.bytes_removed} "
        f"admission_ready={free_bytes >= required_free_bytes} "
        f"free_bytes={free_bytes} required_free_bytes={required_free_bytes}",
        flush=True,
    )


if __name__ == "__main__":
    main()
