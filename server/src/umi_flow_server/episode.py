"""Episode extraction and validation."""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

COMPLETE_FILE = "COMPLETE"
CHECKSUMS_FILE = "checksums.sha256"
MANIFEST_FILE = "manifest.json"
# Written in extraction staging immediately before publish. Pair timeout must measure
# how long a half has waited ON THIS SERVER, not how old the recording is: an offline
# client may legitimately upload both halves hours later.
INGESTED_FILE = ".INGESTED"

# Defense in depth against a decompression/expansion bomb: even a plain tar is
# rejected if the extracted members would exceed this budget. Sized for the
# largest expected episode (a ~3 min 1920x1200@30 MJPEG capture is a few GiB).
DEFAULT_MAX_EXTRACT_BYTES = 8 * 1024 * 1024 * 1024
# Keep enough space for the OS and for stages to fail cleanly instead of filling
# the processing server's filesystem completely.
DEFAULT_MIN_FREE_BYTES = 16 * 1024 * 1024 * 1024


def extract_episode(
    tar_path: Path,
    raw_root: Path,
    expected_root: str | None = None,
    max_extract_bytes: int = DEFAULT_MAX_EXTRACT_BYTES,
    validate: Callable[[Path], None] | None = None,
) -> Path:
    """Extract one episode atomically, validating before it is published.

    The tar is opened as an uncompressed archive only (no transparent
    gzip/bz2/xz, which would let a tiny body expand without bound), extracted
    into a private staging directory, optionally ``validate``d THERE, and only
    then published to its final location with a single atomic rename. Because
    validation runs in quarantine, the episode appears under ``raw_root`` (where
    the pipeline runner scans for the COMPLETE marker) only after it passes — the
    runner can never race a stage against an unvalidated or corrupt episode. A
    crash, checksum failure, or error mid-extraction leaves no episode directory,
    and two concurrent uploads of the same episode cannot interleave into one
    directory: the second rename loses cleanly.
    """
    raw_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:") as tar:
        members = tar.getmembers()
        roots = {Path(member.name).parts[0] for member in members if Path(member.name).parts}
        if len(roots) != 1:
            raise ValueError("episode tar must contain exactly one top-level folder")
        root = next(iter(roots))
        if root != safe_name(root):
            raise ValueError(f"unsafe episode folder name: {root}")
        if expected_root is not None and root != expected_root:
            # The stored identity (tar root) must match the declared identity
            # (X-Episode-Id), or results/errors would be misattributed.
            raise ValueError(
                f"episode id mismatch: header {expected_root!r} != tar folder {root!r}"
            )
        total_bytes = 0
        for member in members:
            member_path = Path(member.name)
            parts = member_path.parts
            if not parts or member_path.is_absolute() or ".." in parts:
                raise ValueError(f"unsafe tar path: {member.name}")
            if parts[0] != root:
                raise ValueError("episode tar must contain exactly one top-level folder")
            if not (member.isfile() or member.isdir()):
                raise ValueError(f"unsupported tar member type: {member.name}")
            total_bytes += member.size
            if total_bytes > max_extract_bytes:
                raise ValueError("episode tar exceeds maximum extracted size")

        dest = raw_root / root
        if dest.exists():
            raise ValueError(f"episode already exists: {root}")
        staging = Path(tempfile.mkdtemp(prefix=f".{root}.staging.", dir=raw_root))
        try:
            tar.extractall(staging, filter="data")
            extracted = staging / root
            if validate is not None:
                validate(extracted)  # in quarantine, before the episode is visible
            (extracted / INGESTED_FILE).write_text(f"{time.time_ns()}\n", encoding="utf-8")
            try:
                extracted.rename(dest)
            except OSError as exc:
                # Lost a race with a concurrent upload of the same episode.
                if dest.exists():
                    raise ValueError(f"episode already exists: {root}") from exc
                raise
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    return dest


def validate_episode(episode_dir: Path) -> None:
    if not (episode_dir / COMPLETE_FILE).exists():
        raise ValueError("episode is missing COMPLETE marker")
    if not (episode_dir / MANIFEST_FILE).exists():
        raise ValueError("episode is missing manifest.json")
    checksum_path = episode_dir / CHECKSUMS_FILE
    if not checksum_path.exists():
        raise ValueError("episode is missing checksums.sha256")
    lines = [
        line for line in checksum_path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not lines:
        raise ValueError("episode has empty checksums.sha256; integrity cannot be verified")
    for line in lines:
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise ValueError("malformed checksum line")
        expected, name = parts
        file_path = episode_dir / name.strip()
        if not file_path.resolve().is_relative_to(episode_dir.resolve()):
            raise ValueError(f"unsafe checksum target: {name}")
        if not file_path.exists():
            raise ValueError(f"checksum target missing: {name}")
        actual = sha256_file(file_path)
        if actual != expected:
            raise ValueError(f"checksum mismatch for {name}")


def safe_name(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
    cleaned = "".join(ch if ch in allowed else "_" for ch in value)
    return cleaned.strip("._") or "episode"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
