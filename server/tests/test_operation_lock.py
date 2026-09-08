from __future__ import annotations

from pathlib import Path

import pytest
from umi_flow_server.operation_lock import PipelineLease


def test_maintenance_excludes_stages_and_other_maintenance(tmp_path: Path) -> None:
    maintenance = PipelineLease.for_data_dir(tmp_path, exclusive=True)
    maintenance.acquire()
    try:
        with pytest.raises(RuntimeError, match="lease is busy"):
            PipelineLease.for_data_dir(tmp_path, exclusive=False).acquire()
        with pytest.raises(RuntimeError, match="lease is busy"):
            PipelineLease.for_data_dir(tmp_path, exclusive=True).acquire()
    finally:
        maintenance.release()


def test_parallel_stages_share_lease_but_block_maintenance(tmp_path: Path) -> None:
    first = PipelineLease.for_data_dir(tmp_path, exclusive=False)
    second = PipelineLease.for_data_dir(tmp_path, exclusive=False)
    first.acquire()
    second.acquire()
    try:
        with pytest.raises(RuntimeError, match="lease is busy"):
            PipelineLease.for_data_dir(tmp_path, exclusive=True).acquire()
    finally:
        second.release()
        first.release()

    with PipelineLease.for_data_dir(tmp_path, exclusive=True):
        pass
