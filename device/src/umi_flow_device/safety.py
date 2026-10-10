"""Operational safety policy shared by preflight and the episode recorder.

Rig YAML describes the physical capture contract. Disk reserves are deployment
policy, so they are configured through validated environment variables instead
of being copied into every episode snapshot.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

GIB = 1024**3
MIB = 1024**2

MIN_FREE_GIB_ENV = "UMI_FLOW_MIN_RECORDING_FREE_GIB"
ESTIMATED_TAKE_GIB_ENV = "UMI_FLOW_ESTIMATED_TAKE_GIB"
METADATA_RESERVE_MIB_ENV = "UMI_FLOW_METADATA_RESERVE_MIB"


def _nonnegative_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    try:
        value = default if raw is None else float(raw)
    except ValueError as exc:
        raise ValueError(f"${name} must be a number, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"${name} must be non-negative, got {value}")
    return value


@dataclass(frozen=True)
class SafetyPolicy:
    """Conservative local capacity required before a recording may start."""

    min_free_bytes_after_take: int = 12 * GIB
    estimated_take_bytes: int = 4 * GIB
    metadata_reserve_bytes: int = 64 * MIB

    @classmethod
    def from_env(cls) -> SafetyPolicy:
        return cls(
            min_free_bytes_after_take=int(_nonnegative_float(MIN_FREE_GIB_ENV, 12.0) * GIB),
            estimated_take_bytes=int(_nonnegative_float(ESTIMATED_TAKE_GIB_ENV, 4.0) * GIB),
            metadata_reserve_bytes=int(
                _nonnegative_float(METADATA_RESERVE_MIB_ENV, 64.0) * MIB
            ),
        )

    @property
    def required_free_bytes(self) -> int:
        return (
            self.min_free_bytes_after_take
            + self.estimated_take_bytes
            + self.metadata_reserve_bytes
        )


@dataclass(frozen=True)
class StorageHeadroom:
    free_bytes: int
    required_free_bytes: int

    @property
    def ok(self) -> bool:
        return self.free_bytes >= self.required_free_bytes

    def require(self) -> None:
        if not self.ok:
            raise RuntimeError(
                "insufficient recording storage: "
                f"free={self.free_bytes} required={self.required_free_bytes}"
            )


DiskUsage = Callable[[Path], object]


def storage_headroom(
    path: Path,
    policy: SafetyPolicy,
    *,
    disk_usage: DiskUsage | None = None,
) -> StorageHeadroom:
    usage = (disk_usage or shutil.disk_usage)(path)
    free = getattr(usage, "free", None)
    if type(free) is not int:
        raise TypeError("disk usage result must expose integer free bytes")
    return StorageHeadroom(free_bytes=free, required_free_bytes=policy.required_free_bytes)
