"""Process lock used to exclude capture and spool maintenance."""

from __future__ import annotations

import fcntl
from pathlib import Path
from types import TracebackType
from typing import TextIO

CAPTURE_LOCK_FILE = ".capture-or-maintenance.lock"


class OperationLock:
    """Non-blocking advisory lock with process-lifetime ownership."""

    def __init__(self, path: Path):
        self.path = path
        self._stream: TextIO | None = None

    @classmethod
    def for_spool(cls, spool_dir: Path) -> OperationLock:
        return cls(spool_dir / CAPTURE_LOCK_FILE)

    def acquire(self) -> None:
        if self._stream is not None:
            raise RuntimeError(f"operation lock is already held: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            stream.close()
            raise RuntimeError(f"capture or maintenance is already active: {self.path}") from exc
        self._stream = stream

    def release(self) -> None:
        if self._stream is None:
            return
        stream, self._stream = self._stream, None
        fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()

    def __enter__(self) -> OperationLock:
        self.acquire()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc, traceback
        self.release()

