#!/usr/bin/env python3
"""Safely repair stale pushed=False receipts after a successful Hub resync."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from umi_live_server.dataset_stage import reconcile_hub_publication
from umi_live_server.operation_lock import PipelineLease, PipelineLeaseBusyError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("DATA_DIR", "data")))
    parser.add_argument("--repo-id", default=os.environ.get("HF_REPO_ID"))
    parser.add_argument("--wait-if-busy-seconds", type=float, default=0.0)
    args = parser.parse_args()
    if not args.repo_id:
        parser.error("--repo-id or HF_REPO_ID is required")
    if args.wait_if_busy_seconds < 0:
        parser.error("--wait-if-busy-seconds cannot be negative")

    lease = PipelineLease.for_data_dir(args.data_dir, exclusive=True)
    try:
        lease.acquire(wait_timeout_s=args.wait_if_busy_seconds)
        repaired, failed = reconcile_hub_publication(args.data_dir, args.repo_id)
    except PipelineLeaseBusyError as exc:
        print(json.dumps({"state": "deferred", "reason": str(exc)}, sort_keys=True))
        raise SystemExit(75) from exc
    finally:
        lease.release()

    print(
        json.dumps(
            {
                "state": "complete" if failed == 0 else "incomplete",
                "repo": args.repo_id,
                "receipts_repaired": repaired,
                "receipts_failed": failed,
            },
            sort_keys=True,
        )
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
