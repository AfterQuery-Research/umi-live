from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from umi_live_server.operation_lock import PipelineLease, PipelineLeaseBusyError


def test_maintenance_excludes_stages_and_other_maintenance(tmp_path: Path) -> None:
    maintenance = PipelineLease.for_data_dir(tmp_path, exclusive=True)
    maintenance.acquire()
    try:
        with pytest.raises(PipelineLeaseBusyError, match="lease is busy"):
            PipelineLease.for_data_dir(tmp_path, exclusive=False).acquire()
        with pytest.raises(PipelineLeaseBusyError, match="lease is busy"):
            PipelineLease.for_data_dir(tmp_path, exclusive=True).acquire()
    finally:
        maintenance.release()


def test_parallel_stages_share_lease_but_block_maintenance(tmp_path: Path) -> None:
    first = PipelineLease.for_data_dir(tmp_path, exclusive=False)
    second = PipelineLease.for_data_dir(tmp_path, exclusive=False)
    first.acquire()
    second.acquire()
    try:
        with pytest.raises(PipelineLeaseBusyError, match="lease is busy"):
            PipelineLease.for_data_dir(tmp_path, exclusive=True).acquire()
    finally:
        second.release()
        first.release()

    with PipelineLease.for_data_dir(tmp_path, exclusive=True):
        pass


def test_waiting_maintenance_stops_new_stages_and_acquires_after_drain(tmp_path: Path) -> None:
    stage = PipelineLease.for_data_dir(tmp_path, exclusive=False)
    stage.acquire()
    maintenance = PipelineLease.for_data_dir(tmp_path, exclusive=True)
    acquired = threading.Event()
    release = threading.Event()

    def wait_for_maintenance() -> None:
        maintenance.acquire(wait_timeout_s=2, retry_interval_s=0.01)
        acquired.set()
        release.wait(timeout=2)
        maintenance.release()

    thread = threading.Thread(target=wait_for_maintenance)
    thread.start()
    deadline = time.monotonic() + 1
    while not maintenance.request_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert maintenance.request_path.exists()
    with pytest.raises(PipelineLeaseBusyError, match="lease is busy"):
        PipelineLease.for_data_dir(tmp_path, exclusive=False).acquire()

    stage.release()
    assert acquired.wait(timeout=1)
    release.set()
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert not maintenance.request_path.exists()


def test_timed_out_maintenance_clears_request(tmp_path: Path) -> None:
    stage = PipelineLease.for_data_dir(tmp_path, exclusive=False)
    stage.acquire()
    maintenance = PipelineLease.for_data_dir(tmp_path, exclusive=True)
    try:
        with pytest.raises(PipelineLeaseBusyError, match="lease is busy"):
            maintenance.acquire(wait_timeout_s=0.02, retry_interval_s=0.01)
        assert not maintenance.request_path.exists()
    finally:
        stage.release()

    with PipelineLease.for_data_dir(tmp_path, exclusive=False):
        pass


def test_dead_maintenance_request_does_not_starve_pipeline(tmp_path: Path) -> None:
    lease = PipelineLease.for_data_dir(tmp_path, exclusive=False)
    lease.request_path.write_text(
        json.dumps(
            {
                "pid": 2**30,
                "request_id": "abandoned",
                "requested_unix": time.time(),
            }
        )
    )

    with lease:
        pass
