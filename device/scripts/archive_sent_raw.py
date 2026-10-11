#!/usr/bin/env python3
"""Archive verified raw payloads from a recorder Pi's sent spool.

The trusted receipt bundle is produced from server-side ``RAW_ARCHIVED.json``
markers.  Those markers are only created after the paired episode is in the
dataset ledger, has a pushed=True receipt, and its raw payloads exactly match a
pinned Hugging Face revision.  This helper independently hashes the Pi copies
again immediately before removing only the camera AVI and lidar PCAP.
"""

from __future__ import annotations

import argparse
import atexit
import collections
import hashlib
import json
import os
import socket
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from umi_live_device.operation_lock import OperationLock

ARCHIVED_MARKER = "RAW_ARCHIVED.json"
IN_PROGRESS_MARKER = ".RAW_ARCHIVE_IN_PROGRESS.json"
PAYLOADS_BY_ROLE = {
    "umi1": ("umi1_camera.mjpeg.avi", "umi1_lidar.pcap"),
    "umi2": ("umi2_camera.mjpeg.avi", "umi2_lidar.pcap"),
}


def _json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _signature(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_ctime_ns, stat.st_mtime_ns)


def _free_bytes(path: Path) -> int:
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _nonempty(path: Path) -> bool:
    return path.is_dir() and next(path.iterdir(), None) is not None


def _capture_processes(spool_dir: Path, proc_root: Path = Path("/proc")) -> list[int]:
    """Find local capture children that are writing into this spool."""
    needle = str(spool_dir.resolve()).encode()
    active: list[int] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if needle in cmdline and (b"v4l2" in cmdline or b"tcpdump" in cmdline):
            active.append(int(entry.name))
    return active


def _require_idle_spool(spool_dir: Path) -> None:
    pending = [name for name in ("complete", "staging") if _nonempty(spool_dir / name)]
    if pending:
        raise RuntimeError(f"refusing maintenance with pending queues: {', '.join(pending)}")
    capture_pids = _capture_processes(spool_dir)
    if capture_pids:
        raise RuntimeError(f"refusing maintenance while capture is active: pids={capture_pids}")


def _checksums(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return result
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            result[parts[1].lstrip("* ")] = parts[0]
    return result


def _paired_episode_id(session_id: object) -> str | None:
    if not isinstance(session_id, str) or not session_id.startswith("sess_"):
        return None
    return f"paired_{session_id}"


def _load_receipts(path: str) -> dict[str, dict[str, Any]]:
    stream = sys.stdin if path == "-" else open(path, encoding="utf-8")
    try:
        value = json.load(stream)
    finally:
        if stream is not sys.stdin:
            stream.close()
    if not isinstance(value, dict):
        raise ValueError("receipt bundle must be an object keyed by paired episode id")
    return {str(key): item for key, item in value.items() if isinstance(item, dict)}


def _complete_durable_intent(episode_dir: Path, intent: dict[str, Any]) -> int:
    """Finish the receipt side of an archive interrupted after payload deletion."""
    payloads = [item for item in intent.get("files", []) if isinstance(item, dict)]
    removed = sum(int(item.get("allocated_bytes", item.get("size", 0))) for item in payloads)
    finished = {
        **intent,
        "state": "complete",
        "metadata_preserved": True,
        "bytes_removed": removed,
        "finished_utc": datetime.now(UTC).isoformat(),
        "restore": (
            "Download the listed repo_path files at the pinned revision and verify sha256."
        ),
        "host": socket.gethostname(),
    }
    _write_json(episode_dir / ARCHIVED_MARKER, finished)
    (episode_dir / IN_PROGRESS_MARKER).unlink(missing_ok=True)
    return removed


def _resume_durable_intent(episode_dir: Path, payload_names: tuple[str, ...]) -> int | None:
    intent = _json(episode_dir / IN_PROGRESS_MARKER)
    if not intent:
        return None
    present = [(episode_dir / name).exists() for name in payload_names]
    if all(present):
        return None  # deletion never started; normal fresh verification can proceed
    if any(present):
        raise RuntimeError(
            f"{episode_dir}: interrupted archive has a mixed payload state; refusing mutation"
        )
    return _complete_durable_intent(episode_dir, intent)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spool-dir", type=Path, required=True)
    parser.add_argument("--role", choices=sorted(PAYLOADS_BY_ROLE), required=True)
    parser.add_argument("--receipts-json", default="-")
    parser.add_argument("--target-free-gib", type=float, default=64.0)
    parser.add_argument("--min-age-hours", type=float, default=24.0)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    operation_lock = OperationLock.for_spool(args.spool_dir)
    operation_lock.acquire()
    atexit.register(operation_lock.release)
    _require_idle_spool(args.spool_dir)

    sent_dir = args.spool_dir / "sent"
    if not sent_dir.is_dir():
        raise SystemExit(f"missing sent directory: {sent_dir}")
    receipts = _load_receipts(args.receipts_json)
    payload_names = PAYLOADS_BY_ROLE[args.role]
    cutoff_ns = time.time_ns() - int(args.min_age_hours * 3600 * 1_000_000_000)
    target_free = int(args.target_free_gib * 1024**3)
    initial_free = _free_bytes(sent_dir)
    archived = 0
    verified_count = 0
    bytes_removed = 0
    skipped_counts: collections.Counter[str] = collections.Counter()
    skipped_examples: list[dict[str, str]] = []

    if args.apply:
        for episode_dir in sorted(path for path in sent_dir.iterdir() if path.is_dir()):
            resumed = _resume_durable_intent(episode_dir, payload_names)
            if resumed is not None:
                archived += 1
                bytes_removed += resumed
                print(f"resumed archive receipt for {episode_dir.name}")

    candidates: list[tuple[int, Path, dict[str, Any]]] = []
    for episode_dir in sent_dir.iterdir():
        if not episode_dir.is_dir() or (episode_dir / ARCHIVED_MARKER).exists():
            continue
        manifest = _json(episode_dir / "manifest.json")
        if not manifest:
            continue
        stop_ns = manifest.get("stop_unix_ns")
        if type(stop_ns) is not int or stop_ns > cutoff_ns:
            continue
        candidates.append((stop_ns, episode_dir, manifest))
    candidates.sort(key=lambda item: item[0])

    for _stop_ns, episode_dir, manifest in candidates:
        if _free_bytes(sent_dir) >= target_free:
            break
        if args.max_episodes is not None and verified_count >= args.max_episodes:
            break
        reason = ""
        paired_id = _paired_episode_id(manifest.get("session_id"))
        receipt = receipts.get(paired_id or "")
        if manifest.get("episode_id") != episode_dir.name or manifest.get("role") != args.role:
            reason = "manifest identity/role mismatch"
        elif not paired_id or not receipt:
            reason = "no trusted server archive receipt"
        elif receipt.get("state") != "complete" or receipt.get("metadata_preserved") is not True:
            reason = "server receipt is not complete"
        elif receipt.get("remote_modified") not in (False, None):
            reason = "server receipt reports remote modification"

        expected: dict[str, dict[str, Any]] = {}
        if not reason:
            for item in receipt.get("files", []):
                if isinstance(item, dict) and isinstance(item.get("name"), str):
                    expected[item["name"]] = item
            if any(name not in expected for name in payload_names):
                reason = "server receipt lacks role payloads"

        files = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
        checksum_entries = _checksums(episode_dir / "checksums.sha256")
        signatures: dict[str, tuple[int, int, int, int, int]] = {}
        verified: list[dict[str, Any]] = []
        if not reason:
            for name in payload_names:
                path = episode_dir / name
                manifest_item = files.get(name) if isinstance(files.get(name), dict) else {}
                expected_item = expected[name]
                expected_size = expected_item.get("size")
                expected_sha = expected_item.get("sha256")
                if not path.is_file():
                    reason = f"missing payload {name}"
                    break
                before = _signature(path)
                if (
                    type(expected_size) is not int
                    or not isinstance(expected_sha, str)
                    or before[2] != expected_size
                    or manifest_item.get("bytes") != expected_size
                    or manifest_item.get("sha256") != expected_sha
                    or checksum_entries.get(name) != expected_sha
                ):
                    reason = f"recorded metadata mismatch for {name}"
                    break
                actual_sha = _sha256(path)
                after = _signature(path)
                if before != after or actual_sha != expected_sha:
                    reason = f"fresh verification mismatch for {name}"
                    break
                signatures[name] = after
                verified.append(
                    {
                        "name": name,
                        "size": expected_size,
                        "allocated_bytes": path.stat().st_blocks * 512,
                        "sha256": expected_sha,
                        "repo_path": expected_item.get("repo_path"),
                    }
                )

        if reason:
            skipped_counts[reason] += 1
            if len(skipped_examples) < 20:
                skipped_examples.append({"episode_id": episode_dir.name, "reason": reason})
            continue
        if not args.apply:
            verified_bytes = sum(item["size"] for item in verified)
            print(f"would archive {episode_dir.name}: bytes={verified_bytes}")
            verified_count += 1
            continue

        started = {
            "state": "in_progress",
            "episode_id": episode_dir.name,
            "paired_episode_id": paired_id,
            "role": args.role,
            "files": verified,
            "repo": receipt.get("repo"),
            "revision": receipt.get("revision"),
            "started_utc": datetime.now(UTC).isoformat(),
        }
        _write_json(episode_dir / IN_PROGRESS_MARKER, started)
        removed = 0
        for name in payload_names:
            path = episode_dir / name
            if _signature(path) != signatures[name]:
                raise RuntimeError(f"payload changed after verification: {path}")
            removed += path.stat().st_blocks * 512
            path.unlink()
        _complete_durable_intent(episode_dir, started)
        archived += 1
        verified_count += 1
        bytes_removed += removed
        print(
            f"archived {episode_dir.name}: allocated_bytes={removed} free={_free_bytes(sent_dir)}"
        )

    summary = {
        "state": "complete",
        "apply": args.apply,
        "role": args.role,
        "initial_free_bytes": initial_free,
        "final_free_bytes": _free_bytes(sent_dir),
        "target_free_bytes": target_free,
        "eligible_candidates": len(candidates),
        "episodes_archived": archived,
        "episodes_verified": verified_count,
        "bytes_removed": bytes_removed,
        "skipped_counts": dict(sorted(skipped_counts.items())),
        "skipped_examples": skipped_examples,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
