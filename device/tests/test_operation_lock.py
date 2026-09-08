from __future__ import annotations

from pathlib import Path

import pytest
from umi_flow_device.operation_lock import OperationLock


def test_operation_lock_is_exclusive_and_reusable(tmp_path: Path) -> None:
    first = OperationLock.for_spool(tmp_path)
    second = OperationLock.for_spool(tmp_path)

    first.acquire()
    with pytest.raises(RuntimeError, match="capture or maintenance is already active"):
        second.acquire()

    first.release()
    second.acquire()
    second.release()
