from __future__ import annotations

import json
from pathlib import Path

from umi_flow_device.runtime_status import RuntimeStatus


def test_runtime_status_is_atomic_and_retains_loaded_identity(tmp_path: Path) -> None:
    status = RuntimeStatus(tmp_path, {"task": "pour water", "state": "idle"})
    status.update(state="recording", session_id="sess_1")

    value = json.loads(status.path.read_text())
    assert value["task"] == "pour water"
    assert value["state"] == "recording"
    assert value["session_id"] == "sess_1"
    assert isinstance(value["pid"], int)
    assert not list(tmp_path.glob("*.tmp"))
