#!/usr/bin/env python3
"""Archive old local raw payloads after exact Hugging Face verification.

Only the two camera AVIs and two lidar PCAPs are removed. Manifests, frame
timestamps, camera statistics, calibration snapshots, derived artifacts, and the
local LeRobot dataset remain. Every deletion is recoverable from a pinned Hub
revision and is recorded both in quarantine and in the raw episode directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from huggingface_hub.errors import RemoteEntryNotFoundError
from umi_flow_server.operation_lock import PipelineLease, PipelineLeaseBusyError

PAYLOAD_NAMES = (
    "umi1_camera.mjpeg.avi",
    "umi1_lidar.pcap",
    "umi2_camera.mjpeg.avi",
    "umi2_lidar.pcap",
)
ARCHIVED_MARKER = "RAW_ARCHIVED.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _signature(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_ctime_ns, stat.st_mtime_ns)


def _allocated_bytes(path: Path) -> int:
    stat = path.stat()
    return stat.st_blocks * 512


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _load_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _eligible_candidates(data_dir: Path, *, min_age_seconds: float) -> list[Path]:
    ledger_path = data_dir / "lerobot/local/umi-flow/.umi_appended_episodes.txt"
    ledger = {line.strip() for line in ledger_path.read_text().splitlines() if line.strip()}
    cutoff = time.time() - min_age_seconds
    candidates: list[tuple[float, Path]] = []
    for marker in (data_dir / "derived").glob("paired_sess_*/dataset/COMPLETE"):
        episode_id = marker.parent.parent.name
        raw_dir = data_dir / "raw_episodes" / episode_id
        receipt = _load_json(marker.parent / "receipt.json")
        if marker.stat().st_mtime > cutoff:
            continue
        if episode_id not in ledger or not receipt or receipt.get("pushed") is not True:
            continue
        if (raw_dir / ARCHIVED_MARKER).exists():
            continue
        if not all((raw_dir / name).is_file() for name in PAYLOAD_NAMES):
            continue
        candidates.append((marker.stat().st_mtime, raw_dir))
    return [raw_dir for _mtime, raw_dir in sorted(candidates)]


def _remote_payloads(
    api: HfApi,
    *,
    repo_id: str,
    revision: str,
    episode_id: str,
) -> dict[str, Any]:
    root = f"raw_episodes/{episode_id}"
    try:
        entries = api.list_repo_tree(
            repo_id=repo_id,
            repo_type="dataset",
            revision=revision,
            path_in_repo=root,
            recursive=True,
            expand=True,
        )
        return {entry.path: entry for entry in entries if hasattr(entry, "size")}
    except RemoteEntryNotFoundError:
        return {}


def _verify_episode(
    api: HfApi,
    *,
    raw_dir: Path,
    repo_id: str,
    revision: str,
) -> tuple[list[dict[str, Any]], int]:
    remote = _remote_payloads(
        api,
        repo_id=repo_id,
        revision=revision,
        episode_id=raw_dir.name,
    )
    verified: list[dict[str, Any]] = []
    allocated_total = 0
    for name in PAYLOAD_NAMES:
        path = raw_dir / name
        signature = _signature(path)
        repo_path = f"raw_episodes/{raw_dir.name}/{name}"
        remote_file = remote.get(repo_path)
        remote_lfs = getattr(remote_file, "lfs", None)
        remote_sha256 = getattr(remote_lfs, "sha256", None)
        remote_size = getattr(remote_file, "size", None)
        if remote_sha256 is None or remote_size != signature[2]:
            raise ValueError(f"{repo_path}: missing or size mismatch at revision {revision}")
        digest = _sha256(path)
        if _signature(path) != signature:
            raise ValueError(f"{path}: changed while hashing")
        if digest != remote_sha256:
            raise ValueError(f"{repo_path}: SHA-256 mismatch at revision {revision}")
        allocated = _allocated_bytes(path)
        allocated_total += allocated
        verified.append(
            {
                "name": name,
                "repo_path": repo_path,
                "size": signature[2],
                "allocated_bytes": allocated,
                "sha256": digest,
                "signature": list(signature),
            }
        )
    return verified, allocated_total


def _resume_archive_intents(data_dir: Path) -> tuple[int, int]:
    """Complete receipts after a crash that removed all verified payloads."""
    resumed = 0
    bytes_removed = 0
    for receipt_path in sorted((data_dir / "quarantine").glob("archived-raw-cleanup-*/*.json")):
        receipt = _load_json(receipt_path)
        if not receipt or receipt.get("state") != "verified":
            continue
        episode_id = receipt.get("episode_id")
        if not isinstance(episode_id, str):
            continue
        raw_dir = data_dir / "raw_episodes" / episode_id
        present = [(raw_dir / name).exists() for name in PAYLOAD_NAMES]
        if all(present):
            continue
        if any(present):
            raise RuntimeError(
                f"{raw_dir}: interrupted archive has a mixed payload state; refusing mutation"
            )
        removed = sum(
            int(item.get("allocated_bytes", item.get("size", 0)))
            for item in receipt.get("files", [])
            if isinstance(item, dict)
        )
        receipt.update(
            {
                "state": "complete",
                "finished_utc": datetime.now(UTC).isoformat(),
                "bytes_removed": removed,
                "removed_files": list(PAYLOAD_NAMES),
            }
        )
        _write_json(receipt_path, receipt)
        _write_json(raw_dir / ARCHIVED_MARKER, receipt)
        resumed += 1
        bytes_removed += removed
        print(f"resumed archive receipt for {episode_id}", flush=True)
    return resumed, bytes_removed


def archive_published_raw(
    data_dir: Path,
    *,
    repo_id: str,
    target_free_bytes: int,
    min_age_seconds: float,
    apply: bool,
    max_episodes: int | None,
) -> dict[str, Any]:
    lease = PipelineLease.for_data_dir(data_dir, exclusive=True)
    lease.acquire()
    try:
        return _archive_published_raw_locked(
            data_dir,
            repo_id=repo_id,
            target_free_bytes=target_free_bytes,
            min_age_seconds=min_age_seconds,
            apply=apply,
            max_episodes=max_episodes,
        )
    finally:
        lease.release()


def _archive_published_raw_locked(
    data_dir: Path,
    *,
    repo_id: str,
    target_free_bytes: int,
    min_age_seconds: float,
    apply: bool,
    max_episodes: int | None,
) -> dict[str, Any]:
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    revision = api.repo_info(repo_id=repo_id, repo_type="dataset").sha
    if not revision:
        raise RuntimeError(f"could not resolve pinned revision for {repo_id}")
    quarantine = data_dir / "quarantine" / f"archived-raw-cleanup-{datetime.now(UTC):%Y%m%d}"
    resumed, resumed_bytes = _resume_archive_intents(data_dir) if apply else (0, 0)
    initial_free = shutil.disk_usage(data_dir).free
    candidates = _eligible_candidates(data_dir, min_age_seconds=min_age_seconds)
    archived = resumed
    bytes_removed = resumed_bytes
    skipped: list[dict[str, str]] = []

    for raw_dir in candidates:
        if shutil.disk_usage(data_dir).free >= target_free_bytes:
            break
        if max_episodes is not None and archived >= max_episodes:
            break
        try:
            files, allocated = _verify_episode(
                api,
                raw_dir=raw_dir,
                repo_id=repo_id,
                revision=revision,
            )
        except (OSError, ValueError) as exc:
            skipped.append({"episode_id": raw_dir.name, "reason": str(exc)})
            print(f"skip {raw_dir.name}: {exc}", flush=True)
            continue

        now = datetime.now(UTC).isoformat()
        receipt = {
            "state": "verified" if apply else "dry_run_verified",
            "episode_id": raw_dir.name,
            "repo": repo_id,
            "revision": revision,
            "verified_utc": now,
            "files": files,
            "bytes_removed": 0,
            "removed_files": [],
            "metadata_preserved": True,
            "remote_modified": False,
            "restore": (
                "Download files from repo at the pinned revision using repo_path; verify sha256."
            ),
        }
        receipt_path = quarantine / f"{raw_dir.name}.json"
        if apply:
            _write_json(receipt_path, receipt)
            for item in files:
                path = raw_dir / item["name"]
                if list(_signature(path)) != item["signature"]:
                    raise RuntimeError(f"{path}: changed after verification; refusing deletion")
                path.unlink()
            finished = datetime.now(UTC).isoformat()
            receipt.update(
                {
                    "state": "complete",
                    "finished_utc": finished,
                    "bytes_removed": allocated,
                    "removed_files": list(PAYLOAD_NAMES),
                }
            )
            _write_json(receipt_path, receipt)
            _write_json(raw_dir / ARCHIVED_MARKER, receipt)
            bytes_removed += allocated
        archived += 1
        print(
            f"{'archived' if apply else 'verified'} {raw_dir.name}: "
            f"allocated_bytes={allocated} free={shutil.disk_usage(data_dir).free}",
            flush=True,
        )

    result = {
        "state": "complete",
        "apply": apply,
        "repo": repo_id,
        "revision": revision,
        "initial_free_bytes": initial_free,
        "final_free_bytes": shutil.disk_usage(data_dir).free,
        "target_free_bytes": target_free_bytes,
        "eligible_candidates": len(candidates),
        "episodes_archived" if apply else "episodes_verified": archived,
        "bytes_removed": bytes_removed,
        "skipped": skipped,
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("DATA_DIR", "data")))
    parser.add_argument("--repo-id", default=os.environ.get("HF_REPO_ID"))
    parser.add_argument("--target-free-gib", type=float, default=96.0)
    parser.add_argument("--min-age-hours", type=float, default=24.0)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument(
        "--skip-if-busy",
        action="store_true",
        help="defer successfully when a pipeline stage currently owns the maintenance lease",
    )
    args = parser.parse_args()
    if not args.repo_id:
        parser.error("--repo-id or HF_REPO_ID is required")
    if args.target_free_gib <= 0 or args.min_age_hours < 0:
        parser.error("target free space must be positive and minimum age cannot be negative")
    try:
        archive_published_raw(
            args.data_dir,
            repo_id=args.repo_id,
            target_free_bytes=int(args.target_free_gib * 1024**3),
            min_age_seconds=args.min_age_hours * 3600,
            apply=args.apply,
            max_episodes=args.max_episodes,
        )
    except PipelineLeaseBusyError as exc:
        if not args.skip_if_busy:
            raise
        print(
            json.dumps(
                {
                    "state": "deferred",
                    "reason": "pipeline_busy",
                    "detail": str(exc),
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
