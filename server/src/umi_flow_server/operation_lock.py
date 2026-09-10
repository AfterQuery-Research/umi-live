"""Cross-process lease shared by pipeline stages and destructive maintenance."""

from __future__ import annotations

import fcntl
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

LOCK_FILE = ".pipeline-or-maintenance.lock"


class PipelineLeaseBusyError(RuntimeError):
    """Raised when another pipeline or maintenance process owns the lease."""


@dataclass
class PipelineLease:
    path: Path
    exclusive: bool
    _handle: TextIO | None = None

    @classmethod
    def for_data_dir(cls, data_dir: Path, *, exclusive: bool) -> PipelineLease:
        return cls(data_dir / LOCK_FILE, exclusive)

    def acquire(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = self.path.open("a+", encoding="utf-8")
        mode = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(handle.fileno(), mode | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            handle.close()
            owner = self.path.read_text(encoding="utf-8", errors="replace").strip()
            raise PipelineLeaseBusyError(
                f"pipeline/maintenance lease is busy: {owner or 'owner unknown'}"
            ) from exc
        if self.exclusive:
            handle.seek(0)
            handle.truncate()
            json.dump(
                {"pid": os.getpid(), "started_unix": time.time(), "mode": "maintenance"},
                handle,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        self._handle = handle

    def release(self) -> None:
        if self._handle is None:
            return
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None

    def __enter__(self) -> PipelineLease:
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()
