"""Process lock used to exclude capture and spool maintenance."""

from __future__ import annotations

import fcntl
from pathlib import Path
from types import TracebackType
from typing import TextIO

CAPTURE_LOCK_FILE = ".capture-or-maintenance.lock"
RECORDER_LOCK_FILE = ".recorder-process.lock"


class OperationLock:
    """Non-blocking advisory lock with process-lifetime ownership."""

    def __init__(self, path: Path, *, busy_description: str = "capture or maintenance"):
        self.path = path
        self.busy_description = busy_description
        self._stream: TextIO | None = None

    @classmethod
    def for_spool(cls, spool_dir: Path) -> OperationLock:
        return cls(spool_dir / CAPTURE_LOCK_FILE)

    @classmethod
    def for_recorder(cls, spool_dir: Path) -> OperationLock:
        """Lock one long-running recorder process to one spool.

        This is deliberately separate from the capture/maintenance lock: the recorder
        stays alive while idle maintenance runs, but a second recorder entrypoint must
        never share its buttons, coordinator, runtime status, or spool.
        """

        return cls(
            spool_dir / RECORDER_LOCK_FILE,
            busy_description="another recorder process",
        )

    def acquire(self) -> None:
        if self._stream is not None:
            raise RuntimeError(f"operation lock is already held: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            stream.close()
            raise RuntimeError(f"{self.busy_description} is already active: {self.path}") from exc
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
