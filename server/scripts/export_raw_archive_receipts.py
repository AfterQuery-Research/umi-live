#!/usr/bin/env python3
"""Export trusted server RAW_ARCHIVED receipts as one JSON object."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    receipts: dict[str, dict[str, Any]] = {}
    for marker in (args.data_dir / "raw_episodes").glob("paired_sess_*/RAW_ARCHIVED.json"):
        value = _load(marker)
        episode_id = marker.parent.name
        if (
            value
            and value.get("state") == "complete"
            and value.get("episode_id") == episode_id
            and value.get("metadata_preserved") is True
            and isinstance(value.get("revision"), str)
            and isinstance(value.get("files"), list)
        ):
            receipts[episode_id] = value
    print(json.dumps(receipts, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
