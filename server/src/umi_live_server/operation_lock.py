"""Cross-process lease shared by pipeline stages and destructive maintenance."""

from __future__ import annotations

import fcntl
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

LOCK_FILE = ".pipeline-or-maintenance.lock"
MAINTENANCE_REQUEST_FILE = ".pipeline-maintenance-request.json"
MAINTENANCE_REQUEST_STALE_SECONDS = 30 * 60
LEASE_RETRY_SECONDS = 0.25


class PipelineLeaseBusyError(RuntimeError):
    """Raised when another pipeline or maintenance process owns the lease."""


@dataclass
class PipelineLease:
    path: Path
    exclusive: bool
    _handle: TextIO | None = None
    _request_id: str | None = None

    @classmethod
    def for_data_dir(cls, data_dir: Path, *, exclusive: bool) -> PipelineLease:
        return cls(data_dir / LOCK_FILE, exclusive)

    @property
    def request_path(self) -> Path:
        return self.path.with_name(MAINTENANCE_REQUEST_FILE)

    def _maintenance_requested(self) -> bool:
        try:
            value = json.loads(self.request_path.read_text(encoding="utf-8"))
            requested_unix = float(value["requested_unix"])
            pid = int(value["pid"])
        except (FileNotFoundError, OSError, ValueError, KeyError, TypeError):
            return False
        if time.time() - requested_unix > MAINTENANCE_REQUEST_STALE_SECONDS:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            # The runner is unprivileged while retention runs as root.
            pass
        return True

    def _request_maintenance(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        request_id = uuid.uuid4().hex
        temporary = self.request_path.with_name(
            f".{self.request_path.name}.{os.getpid()}.{request_id}.tmp"
        )
        value = {
            "pid": os.getpid(),
            "request_id": request_id,
            "requested_unix": time.time(),
        }
        try:
            with temporary.open("w", encoding="utf-8") as handle:
                json.dump(value, handle, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(self.request_path)
        finally:
            temporary.unlink(missing_ok=True)
        self._request_id = request_id

    def _clear_maintenance_request(self) -> None:
        if self._request_id is None:
            return
        try:
            value = json.loads(self.request_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, ValueError, TypeError):
            value = None
        if isinstance(value, dict) and value.get("request_id") == self._request_id:
            self.request_path.unlink(missing_ok=True)
        self._request_id = None

    def acquire(
        self,
        *,
        wait_timeout_s: float = 0.0,
        retry_interval_s: float = LEASE_RETRY_SECONDS,
    ) -> None:
        if wait_timeout_s < 0:
            raise ValueError("lease wait timeout cannot be negative")
        if retry_interval_s <= 0:
            raise ValueError("lease retry interval must be positive")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.exclusive and wait_timeout_s > 0:
            # Announce intent before waiting. New pipeline jobs then defer, while
            # already-running jobs keep their shared leases and finish normally.
            self._request_maintenance()
        handle = self.path.open("a+", encoding="utf-8")
        mode = fcntl.LOCK_EX if self.exclusive else fcntl.LOCK_SH
        deadline = time.monotonic() + wait_timeout_s
        try:
            while True:
                if not self.exclusive and self._maintenance_requested():
                    raise BlockingIOError
                try:
                    fcntl.flock(handle.fileno(), mode | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if wait_timeout_s <= 0 or time.monotonic() >= deadline:
                        raise
                    time.sleep(min(retry_interval_s, max(0.0, deadline - time.monotonic())))
        except BlockingIOError as exc:
            handle.close()
            owner = self.path.read_text(encoding="utf-8", errors="replace").strip()
            self._clear_maintenance_request()
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
            self._clear_maintenance_request()
            return
        fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
        self._handle.close()
        self._handle = None
        self._clear_maintenance_request()

    def __enter__(self) -> PipelineLease:
        self.acquire()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.release()
