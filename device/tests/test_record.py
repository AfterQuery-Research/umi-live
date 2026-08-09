"""Tests for the fixed leader/follower recorder entrypoint."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from umi_flow_device.buttons import Event

RECORD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "record.py"
spec = importlib.util.spec_from_file_location("record", RECORD_PATH)
assert spec is not None and spec.loader is not None
record = importlib.util.module_from_spec(spec)
sys.modules["record"] = record
spec.loader.exec_module(record)


class FakeDisplay:
    def __init__(self) -> None:
        self.uploads: list[str] = []

    def notify_upload(self, state: str) -> None:
        self.uploads.append(state)

    def set_lidar_health(self, *_args) -> None: ...
    def set_episodes(self, *_args) -> None: ...
    def set_sending(self, *_args) -> None: ...
    def set_recording(self, *_args) -> None: ...


class FakeUploader:
    def current_upload(self):
        return None

    def snapshot_results(self):
        return []


class FakeRecorder:
    active = None

    def __init__(self) -> None:
        self.starts = 0

    def start(self, **_kwargs):
        self.starts += 1
        raise AssertionError("local recording must not start without the follower")


class SequenceButtons:
    def __init__(self, *events: Event) -> None:
        self.events = iter(events)

    def poll(self):
        return next(self.events)


class UnavailableFollower:
    def start_take(self, _session_id: str):
        return None


def test_legacy_environment_names_remain_accepted(tmp_path: Path, monkeypatch):
    config = tmp_path / "legacy.yaml"
    spool = tmp_path / "legacy-spool"
    monkeypatch.delenv(record.core_config.CONFIG_ENV_VAR, raising=False)
    monkeypatch.delenv(record.SPOOL_DIR_ENV, raising=False)
    monkeypatch.setenv(record.core_config.LEGACY_CONFIG_ENV_VAR, str(config))
    monkeypatch.setenv(record.LEGACY_SPOOL_DIR_ENV, str(spool))

    assert record.resolve_config_path(record.Args()) == config
    assert record.resolve_spool_dir(None) == spool


def test_existing_legacy_default_spool_is_not_stranded(tmp_path: Path, monkeypatch):
    legacy = tmp_path / record.LEGACY_DEFAULT_SPOOL_DIR
    legacy.mkdir()
    monkeypatch.delenv(record.SPOOL_DIR_ENV, raising=False)
    monkeypatch.delenv(record.LEGACY_SPOOL_DIR_ENV, raising=False)
    monkeypatch.setattr(record, "device_root", lambda: tmp_path)

    assert record.resolve_spool_dir(None) == legacy


def test_spool_has_data_ignores_empty_scaffolding(tmp_path: Path):
    spool = tmp_path / "spool"
    assert not record.spool_has_data(spool)
    for name in ("in_progress", "complete", "sent", "staging"):
        (spool / name).mkdir(parents=True, exist_ok=True)
    assert not record.spool_has_data(spool)

    (spool / "complete" / "episode_1").mkdir()
    assert record.spool_has_data(spool)


def test_leader_start_records_clock_handshake():
    display = FakeDisplay()
    handshake = {"v": None}

    class AvailableFollower:
        def start_take(self, _session_id: str):
            return {"ok": True, "follower_unix_ns": 1}

    assert record._leader_start(AvailableFollower(), "sess_1", handshake, display)
    assert isinstance(handshake["v"], int)
    assert display.uploads == []


def test_unavailable_follower_aborts_before_local_recording(tmp_path: Path):
    recorder = FakeRecorder()
    display = FakeDisplay()
    record.button_loop(
        recorder,
        FakeUploader(),
        {},
        display,
        tmp_path,
        SequenceButtons(Event.START, Event.QUIT),
        UnavailableFollower(),
        "umi1",
        {"v": None},
    )

    assert recorder.starts == 0
    assert display.uploads == [record.UPLOAD_FAIL]


def test_follower_start_rejection_aborts_before_local_recording(tmp_path: Path):
    recorder = FakeRecorder()
    display = FakeDisplay()

    class RejectedFollower:
        def start_take(self, _session_id: str):
            return {"ok": False, "error": "camera failed"}

    record.button_loop(
        recorder,
        FakeUploader(),
        {},
        display,
        tmp_path,
        SequenceButtons(Event.START, Event.QUIT),
        RejectedFollower(),
        "umi1",
        {"v": None},
    )

    assert recorder.starts == 0
    assert display.uploads == [record.UPLOAD_FAIL]
