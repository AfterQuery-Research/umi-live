"""Tests for the fixed leader/follower recorder entrypoint."""

from __future__ import annotations

import importlib.util
import signal
import sys
import threading
from pathlib import Path
from queue import Queue

from umi_live_device.buttons import Event
from umi_live_device.preflight import CheckResult, PipelineResult

RECORD_PATH = Path(__file__).resolve().parents[1] / "scripts" / "record.py"
spec = importlib.util.spec_from_file_location("record", RECORD_PATH)
assert spec is not None and spec.loader is not None
record = importlib.util.module_from_spec(spec)
sys.modules["record"] = record
spec.loader.exec_module(record)


class FakeDisplay:
    def __init__(self) -> None:
        self.uploads: list[str] = []
        self.check_failures: list[tuple[str, ...]] = []
        self.check_clears = 0

    def notify_upload(self, state: str) -> None:
        self.uploads.append(state)

    def set_lidar_health(self, *_args) -> None: ...
    def set_episodes(self, *_args) -> None: ...
    def set_sending(self, *_args) -> None: ...
    def set_recording(self, *_args) -> None: ...

    def notify_check_failure(self, lines: tuple[str, ...]) -> None:
        self.check_failures.append(lines)

    def clear_check_failure(self) -> None:
        self.check_clears += 1


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
    def precheck(self):
        return None

    def start_take(self, _session_id: str):
        return None

    def prepare_start(self, _session_id: str):
        return None

    def abort_start(self, _session_id: str):
        return {"ok": True}


class FakePipeline:
    def __init__(self, result: PipelineResult):
        self.result = result
        self.runs = 0

    def run(self) -> PipelineResult:
        self.runs += 1
        return self.result


def passing_pipeline(umi: str = "umi1") -> FakePipeline:
    return FakePipeline(PipelineResult((CheckResult("gripper_tags", umi, True),)))


def precheck_ack(result: PipelineResult) -> dict:
    return {"ok": result.ok, "result": result.to_dict(), "follower_unix_ns": 1}


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
    (spool / record.CAPTURE_LOCK_FILE).touch()
    (spool / record.RECORDER_LOCK_FILE).touch()
    (spool / record.STATUS_FILE).write_text("{}\n")
    assert not record.spool_has_data(spool)

    (spool / "complete" / "episode_1").mkdir()
    assert record.spool_has_data(spool)


def test_button_loop_honors_shutdown_request_without_polling(tmp_path: Path):
    shutdown = threading.Event()
    shutdown.set()

    class MustNotPoll:
        def poll(self):
            raise AssertionError("buttons must not be polled after shutdown")

    record.button_loop(
        FakeRecorder(),
        FakeUploader(),
        {},
        FakeDisplay(),
        tmp_path,
        MustNotPoll(),
        UnavailableFollower(),
        "umi1",
        {"v": None},
        passing_pipeline(),
        shutdown_requested=shutdown,
    )


def test_sigterm_handler_sets_shutdown_event(monkeypatch):
    shutdown = threading.Event()
    installed = {}

    def fake_signal(signum, handler):
        installed[signum] = handler
        return signal.SIG_DFL

    monkeypatch.setattr(record.signal, "signal", fake_signal)
    previous = record._install_sigterm_handler(shutdown)
    assert previous == signal.SIG_DFL

    installed[signal.SIGTERM](signal.SIGTERM, None)
    assert shutdown.is_set()


def test_leader_start_records_clock_handshake():
    display = FakeDisplay()
    handshake = {"v": None}

    class AvailableFollower:
        def start_take(self, _session_id: str):
            return {"ok": True, "follower_unix_ns": 1}

    assert record._leader_start(AvailableFollower(), "sess_1", handshake, display)
    assert isinstance(handshake["v"], int)
    assert display.uploads == []


def test_finish_paired_starts_local_and_follower_stops_concurrently(tmp_path: Path) -> None:
    local_started = threading.Event()
    follower_started = threading.Event()
    episode_dir = tmp_path / "episode_test"

    class StoppingRecorder:
        last_stop_issues: tuple = ()

        def stop(self):
            local_started.set()
            assert follower_started.wait(timeout=1)
            return episode_dir

    class StoppingFollower:
        def stop_take(self):
            follower_started.set()
            assert local_started.wait(timeout=1)
            return {"ok": True}

    submitted: list[Path] = []

    class CapturingUploader:
        def submit(self, path: Path) -> None:
            submitted.append(path)

    record.finish_paired(StoppingRecorder(), CapturingUploader(), StoppingFollower())

    assert submitted == [episode_dir]


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
        passing_pipeline(),
    )

    assert recorder.starts == 0
    assert display.uploads == []
    assert display.check_failures == [("umi2: follower unavailable",)]


def test_follower_commit_rejection_cancels_local_start(tmp_path: Path):
    class CancelableRecorder:
        active = None

        def __init__(self) -> None:
            self.starts = 0
            self.cancels = 0

        def poll_active_error(self):
            return None

        def start(self, **_kwargs):
            self.starts += 1
            self.active = type("Active", (), {"episode_id": "episode_test"})()
            return self.active

        def cancel(self):
            self.cancels += 1
            self.active = None

    recorder = CancelableRecorder()
    display = FakeDisplay()

    class RejectedFollower:
        def prepare_start(self, _session_id: str):
            return precheck_ack(PipelineResult((CheckResult("gripper_tags", "umi2", True),)))

        def commit_start(self, _session_id: str):
            return {"ok": False, "error": "camera failed"}

        def abort_start(self, _session_id: str):
            return {"ok": True}

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
        passing_pipeline(),
    )

    assert recorder.starts == 1
    assert recorder.cancels == 1
    assert display.uploads == [record.UPLOAD_FAIL]


def test_preflight_reports_missing_tags_from_both_clients() -> None:
    local = FakePipeline(PipelineResult((CheckResult("gripper_tags", "umi1", False, (2,)),)))
    remote = PipelineResult((CheckResult("gripper_tags", "umi2", False, (0, 1)),))

    class MissingFollower:
        def precheck(self):
            return precheck_ack(remote)

    display = FakeDisplay()
    assert not record._preflight_before_start(local, MissingFollower(), display)
    assert local.runs == 1
    assert display.check_failures == [("umi1 tags missing: 2", "umi2 tags missing: 0,1")]


def test_preflight_passes_only_after_both_clients_pass() -> None:
    local = passing_pipeline()
    remote = PipelineResult((CheckResult("gripper_tags", "umi2", True),))

    class ReadyFollower:
        def precheck(self):
            return precheck_ack(remote)

    display = FakeDisplay()
    assert record._preflight_before_start(local, ReadyFollower(), display)
    assert display.check_failures == []


def test_follower_precheck_response_carries_structured_result() -> None:
    replies: Queue[dict] = Queue(maxsize=1)
    command = record.coordinator.Command(cmd=record.coordinator.CMD_PRECHECK, _reply=replies)
    pipeline = FakePipeline(PipelineResult((CheckResult("gripper_tags", "umi2", False, (1,)),)))

    record._respond_to_precheck(command, pipeline)

    reply = replies.get_nowait()
    assert reply["ok"] is False
    result = PipelineResult.from_dict(reply["result"])
    assert result.failures[0].missing_tag_ids == (1,)


def test_button_start_runs_both_prechecks_before_starting_either_recorder(tmp_path: Path) -> None:
    calls: list[str] = []

    class StartableRecorder:
        active = None

        def poll_active_error(self):
            return None

        def start(self, **_kwargs):
            calls.append("local-start")
            self.active = type(
                "Active",
                (),
                {"episode_id": "episode_test", "session_id": _kwargs["session_id"]},
            )()
            return self.active

    remote_result = PipelineResult((CheckResult("gripper_tags", "umi2", True),))

    class ReadyFollower:
        def prepare_start(self, _session_id: str):
            calls.append("remote-prepare")
            return precheck_ack(remote_result)

        def commit_start(self, _session_id: str):
            calls.append("remote-commit")
            return {"ok": True, "follower_unix_ns": 1}

        def abort_start(self, _session_id: str):
            calls.append("remote-abort")
            return {"ok": True}

        def heartbeat(self, _session_id: str):
            return {"ok": True}

    class OrderedPipeline(FakePipeline):
        def run(self):
            calls.append("local-check")
            return super().run()

    recorder = StartableRecorder()
    record.button_loop(
        recorder,
        FakeUploader(),
        {},
        FakeDisplay(),
        tmp_path,
        SequenceButtons(Event.START, Event.QUIT),
        ReadyFollower(),
        "umi1",
        {"v": None},
        OrderedPipeline(PipelineResult((CheckResult("gripper_tags", "umi1", True),))),
    )

    assert calls == ["local-check", "remote-prepare", "local-start", "remote-commit"]


def test_follower_prepare_then_commit_starts_reserved_session(tmp_path: Path) -> None:
    replies: Queue[dict] = Queue()
    commands = iter(
        [
            record.coordinator.Command(
                cmd=record.coordinator.CMD_PREPARE_START,
                session_id="sess_1",
                _reply=replies,
            ),
            record.coordinator.Command(
                cmd=record.coordinator.CMD_COMMIT_START,
                session_id="sess_1",
                leader_unix_ns=1,
                _reply=replies,
            ),
        ]
    )

    class Done(Exception):
        pass

    class Follower:
        def __init__(self) -> None:
            self.statuses: list[dict] = []

        def poll(self):
            try:
                return next(commands)
            except StopIteration as exc:
                raise Done from exc

        def set_status(self, **values):
            self.statuses.append(values)

        def lease_expired(self, _session_id: str):
            return False

    class Recorder:
        active = None

        def poll_active_error(self):
            return None

        def start(self, *, session_id, role):
            assert (session_id, role) == ("sess_1", "umi2")
            self.active = type("Active", (), {"episode_id": "episode_1"})()
            return self.active

    follower = Follower()
    recorder = Recorder()
    try:
        record.follower_loop(
            recorder,
            FakeUploader(),
            {},
            FakeDisplay(),
            tmp_path,
            follower,
            "umi2",
            {"v": None},
            passing_pipeline("umi2"),
        )
    except Done:
        pass
    else:
        raise AssertionError("test follower did not finish")

    assert recorder.active is not None
    assert replies.get_nowait()["ok"] is True
    assert replies.get_nowait()["ok"] is True
    assert follower.statuses[-1]["state"] == "recording"
