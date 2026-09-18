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


def test_recorder_process_lock_is_exclusive_but_does_not_block_maintenance(
    tmp_path: Path,
) -> None:
    recorder = OperationLock.for_recorder(tmp_path)
    duplicate = OperationLock.for_recorder(tmp_path)
    maintenance = OperationLock.for_spool(tmp_path)

    recorder.acquire()
    maintenance.acquire()
    maintenance.release()
    with pytest.raises(RuntimeError, match="another recorder process is already active"):
        duplicate.acquire()
    recorder.release()

    duplicate.acquire()
    duplicate.release()
