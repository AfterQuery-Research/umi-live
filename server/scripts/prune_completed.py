#!/usr/bin/env python3
"""Prune reconstructible server artifacts after a short recovery window."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from umi_flow_server.retention import DEFAULT_RETENTION_SECONDS, prune_completed


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
    print(
        "retention complete: "
        f"eligible={summary.eligible_sessions} "
        f"source_dirs={summary.source_dirs_removed} "
        f"decode_dirs={summary.decode_dirs_pruned} "
        f"files={summary.files_removed} bytes={summary.bytes_removed}",
        flush=True,
    )


if __name__ == "__main__":
    main()
