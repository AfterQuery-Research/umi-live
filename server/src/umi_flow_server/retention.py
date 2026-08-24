"""Retention for reconstructible server artifacts after dataset publication.

The merged episode in ``raw_episodes`` is the canonical reprocessing source. Once the
dataset stage is COMPLETE, the two transported arm copies in ``paired_sources`` and the
decoded ROS bag payloads are duplicates. This module removes only those duplicates while
preserving stage COMPLETE markers and a machine-readable pruning receipt.
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from umi_flow_core import layout

DEFAULT_RETENTION_SECONDS = 3 * 60 * 60
PRUNED_FILE = "PRUNED.json"
PAIRED_ROLES = {"umi1", "umi2"}


@dataclass(frozen=True)
class PruneSummary:
    eligible_sessions: int = 0
    source_dirs_removed: int = 0
    decode_dirs_pruned: int = 0
    files_removed: int = 0
    bytes_removed: int = 0


def _tree_stats(path: Path) -> tuple[int, int]:
    files = 0
    size = 0
    for item in path.rglob("*"):
        try:
            if item.is_file():
                files += 1
                size += item.stat().st_size
        except FileNotFoundError:
            pass
    return files, size


def _paired_sources(data_dir: Path) -> dict[str, list[tuple[str, Path]]]:
    sessions: dict[str, list[tuple[str, Path]]] = {}
    root = data_dir / "paired_sources"
    if not root.exists():
        return sessions
    for manifest_path in sorted(root.glob("episode_*/manifest.json")):
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        session_id = raw.get("session_id")
        role = raw.get("role")
        if isinstance(session_id, str) and role in PAIRED_ROLES:
            sessions.setdefault(session_id, []).append((role, manifest_path.parent))
    return sessions


def _prune_decode(
    decode_dir: Path,
    *,
    episode_id: str,
    scope: str,
    now_ns: int,
) -> tuple[bool, int, int]:
    if not (decode_dir / layout.COMPLETE_FILE).is_file():
        return False, 0, 0
    payload = [
        item
        for item in decode_dir.iterdir()
        if item.name not in {layout.COMPLETE_FILE, PRUNED_FILE}
    ]
    files_removed = 0
    bytes_removed = 0
    for item in payload:
        files, size = _tree_stats(item) if item.is_dir() else (1, item.stat().st_size)
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
        files_removed += files
        bytes_removed += size

    receipt_path = decode_dir / PRUNED_FILE
    if payload or not receipt_path.exists():
        receipt = {
            "artifact": "decoded ROS2 bag",
            "episode_id": episode_id,
            "pruned_unix_ns": now_ns,
            "reason": (
                "dataset stage complete for at least the retention period; "
                "canonical paired raw episode retained"
            ),
            "scope": scope,
        }
        temporary = decode_dir / f".{PRUNED_FILE}.tmp"
        temporary.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(receipt_path)
    return bool(payload), files_removed, bytes_removed


def prune_completed(
    data_dir: Path,
    *,
    retention_seconds: float = DEFAULT_RETENTION_SECONDS,
    now_ns: int | None = None,
    log: Callable[[str], None] = print,
) -> PruneSummary:
    """Prune duplicates for dataset-complete sessions older than the retention period.

    The operation is idempotent. A partial pair in ``paired_sources`` is deliberately
    retained because deleting one unexplained half would make incident recovery harder.
    """
    if retention_seconds < 0:
        raise ValueError("retention_seconds must be non-negative")
    now_ns = time.time_ns() if now_ns is None else now_ns
    cutoff_ns = now_ns - int(retention_seconds * 1_000_000_000)
    sources = _paired_sources(data_dir)
    derived_root = data_dir / "derived"

    eligible = source_dirs = decode_dirs = files_removed = bytes_removed = 0
    if not derived_root.exists():
        return PruneSummary()

    for episode_dir in sorted(derived_root.glob("paired_sess_*")):
        dataset_complete = episode_dir / "dataset" / layout.COMPLETE_FILE
        try:
            completed_ns = dataset_complete.stat().st_mtime_ns
        except FileNotFoundError:
            continue
        if completed_ns > cutoff_ns:
            continue
        eligible += 1
        session_id = episode_dir.name.removeprefix("paired_")

        for decode_dir in sorted(episode_dir.glob("*/decode")):
            pruned, files, size = _prune_decode(
                decode_dir,
                episode_id=episode_dir.name,
                scope=decode_dir.parent.name,
                now_ns=now_ns,
            )
            decode_dirs += int(pruned)
            files_removed += files
            bytes_removed += size

        halves = sources.get(session_id, [])
        roles = {role for role, _path in halves}
        if halves and (len(halves) != 2 or roles != PAIRED_ROLES):
            log(f"retention: kept incomplete paired sources for {session_id}: roles={roles}")
            continue
        for _role, source_dir in halves:
            files, size = _tree_stats(source_dir)
            shutil.rmtree(source_dir)
            source_dirs += 1
            files_removed += files
            bytes_removed += size

    return PruneSummary(
        eligible_sessions=eligible,
        source_dirs_removed=source_dirs,
        decode_dirs_pruned=decode_dirs,
        files_removed=files_removed,
        bytes_removed=bytes_removed,
    )
