"""Durable loaded-process status for SSH monitoring and stale-code detection."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

STATUS_FILE = ".recorder-status.json"


class RuntimeStatus:
    def __init__(self, spool_dir: Path, initial: dict[str, Any]):
        self.path = spool_dir / STATUS_FILE
        self._lock = threading.Lock()
        self._value = {
            **initial,
            "pid": os.getpid(),
            "process_started_unix": time.time(),
        }
        self.update()

    def update(self, **values: Any) -> None:
        with self._lock:
            self._value.update(values)
            self._value["updated_unix"] = time.time()
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(f".{self.path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
            try:
                with tmp.open("w", encoding="utf-8") as handle:
                    json.dump(self._value, handle, indent=2, sort_keys=True)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                tmp.replace(self.path)
            finally:
                tmp.unlink(missing_ok=True)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._value)
