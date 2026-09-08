"""Pipeline-runner tests: the filesystem state machine, dependencies, recovery."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import umi_flow_server.runner as runner_module
from umi_flow_server.runner import Executor, Runner, StageRun, default_stages

from tests.helpers import make_paired_episode


def publish(tmp_path: Path) -> tuple[Path, Path]:
    """A validated episode in data/raw_episodes (as the ingest would leave it)."""
    episode = make_paired_episode(tmp_path)
    data_dir = tmp_path / "data"
    raw_root = data_dir / "raw_episodes"
    raw_root.mkdir(parents=True)
    target = raw_root / episode.name
    episode.rename(target)
    return data_dir, target


def make_runner(data_dir: Path, calls: list[tuple[str, str]]):
    """A runner whose container stages are in-process fakes that log calls."""

    def fake_stage(name):
        def fn(episode_dir: Path, out_dir: Path, scope: str) -> None:
            calls.append((name, scope))
            if name == "glim":
                (out_dir / "trajectory.tum").write_text("# t x y z qx qy qz qw\n")

        return fn

    stages = default_stages(dataset_fn=fake_stage("dataset"), gripper_fn=fake_stage("gripper"))
    for s in stages:
        if s.fn is None:  # decode/glim: replace compose execution with fakes
            s.fn = fake_stage(s.name)
    return Runner(data_dir=data_dir, stages=stages, executor=Executor(data_dir))


def test_full_graph_runs_in_dependency_order(tmp_path: Path):
    data_dir, episode = publish(tmp_path)
    calls: list[tuple[str, str]] = []
    runner = make_runner(data_dir, calls)
    executed = runner.scan_once()
    # Both fixed arms run decode + GLIM; gripper and dataset run once on the pair.
    assert ("decode", "umi1") in calls
    assert ("glim", "umi1") in calls
    assert ("decode", "umi2") in calls
    assert ("glim", "umi2") in calls
    assert ("gripper", "") in calls
    assert ("dataset", "") in calls
    assert calls.index(("decode", "umi1")) < calls.index(("glim", "umi1"))
    assert calls.index(("glim", "umi1")) < calls.index(("dataset", ""))
    assert executed == 6

    derived = data_dir / "derived" / episode.name
    assert (derived / "umi1" / "decode" / "COMPLETE").exists()
    assert (derived / "umi1" / "glim" / "trajectory.tum").exists()
    assert (derived / "dataset" / "COMPLETE").exists()


def test_rescan_is_idempotent(tmp_path: Path):
    data_dir, _ = publish(tmp_path)
    calls: list[tuple[str, str]] = []
    runner = make_runner(data_dir, calls)
    assert runner.scan_once() == 6
    assert runner.scan_once() == 0  # everything COMPLETE; nothing re-runs
    assert len(calls) == 6


def test_recovery_reruns_incomplete_stage(tmp_path: Path):
    data_dir, episode = publish(tmp_path)
    calls: list[tuple[str, str]] = []
    runner = make_runner(data_dir, calls)
    runner.scan_once()
    # Simulate a crash between rename and COMPLETE: marker missing.
    decode_out = data_dir / "derived" / episode.name / "umi1" / "decode"
    (decode_out / "COMPLETE").unlink()
    calls.clear()
    runner2 = make_runner(data_dir, calls)
    executed = runner2.scan_once()
    assert ("decode", "umi1") in calls  # torn down and re-run
    assert executed == 1
    assert (decode_out / "COMPLETE").exists()


def test_failed_stage_blocks_dependents_not_siblings(tmp_path: Path):
    data_dir, _ = publish(tmp_path)
    calls: list[tuple[str, str]] = []

    def fake_stage(name):
        def fn(episode_dir: Path, out_dir: Path, scope: str) -> None:
            if name == "decode" and scope == "umi2":
                raise RuntimeError("boom")
            calls.append((name, scope))
            if name == "glim":
                (out_dir / "trajectory.tum").write_text("#\n")

        return fn

    stages = default_stages(dataset_fn=fake_stage("dataset"), gripper_fn=fake_stage("gripper"))
    for s in stages:
        if s.fn is None:
            s.fn = fake_stage(s.name)
    runner = Runner(data_dir=data_dir, stages=stages, executor=Executor(data_dir))
    runner.scan_once()
    assert ("decode", "umi1") in calls and ("glim", "umi1") in calls
    assert ("glim", "umi2") not in calls  # blocked by its failed decode
    assert ("dataset", "") not in calls  # needs glim for BOTH lidar umis
    assert len(runner.failed) == 1


def test_failed_stage_retries_after_backoff(tmp_path: Path):
    data_dir, _ = publish(tmp_path)
    calls = 0
    now = [100.0]

    def flaky(_episode_dir: Path, _out_dir: Path, _scope: str) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient")

    stage = runner_module.StageDef(name="flaky", scopes=lambda _m, _rig: [""], fn=flaky)
    runner = Runner(
        data_dir=data_dir,
        stages=[stage],
        executor=Executor(data_dir),
        retry_initial_s=10.0,
        retry_max_s=60.0,
        now_fn=lambda: now[0],
    )

    assert runner.scan_once() == 0
    assert calls == 1
    assert runner.scan_once() == 0
    assert calls == 1
    now[0] += 10.0
    assert runner.scan_once() == 1
    assert calls == 2
    assert runner.failed == {}


def test_retry_state_survives_runner_restart(tmp_path: Path):
    data_dir, episode = publish(tmp_path)
    calls = 0
    now = [100.0]

    def fail(*_args) -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("bad input")

    stage = runner_module.StageDef(name="decode", scopes=lambda _m, _rig: ["umi1"], fn=fail)
    first = Runner(
        data_dir=data_dir,
        stages=[stage],
        retry_initial_s=10,
        now_fn=lambda: now[0],
    )
    assert first.scan_once() == 0
    run = StageRun(episode.name, "decode", "umi1")
    assert first.failure_path(run).exists()

    restarted = Runner(
        data_dir=data_dir,
        stages=[stage],
        retry_initial_s=10,
        now_fn=lambda: now[0],
    )
    assert restarted.scan_once() == 0
    assert calls == 1
    now[0] += 10
    assert restarted.scan_once() == 0
    assert calls == 2


def test_deterministic_failure_becomes_terminal_dataset_skip(tmp_path: Path):
    data_dir, episode = publish(tmp_path)
    now = [100.0]

    def fail(*_args) -> None:
        raise ValueError("permanently malformed input")

    stage = runner_module.StageDef(name="decode", scopes=lambda _m, _rig: ["umi1"], fn=fail)
    runner = Runner(
        data_dir=data_dir,
        stages=[stage],
        retry_initial_s=1,
        max_deterministic_attempts=2,
        now_fn=lambda: now[0],
    )
    assert runner.scan_once() == 0
    now[0] += 1
    assert runner.scan_once() == 0

    dataset = data_dir / "derived" / episode.name / "dataset"
    receipt = (dataset / "receipt.json").read_text()
    assert (dataset / "COMPLETE").exists()
    assert "terminal deterministic_input failure after 2 attempts" in receipt
    assert runner.ready_runs() == []


def test_known_ineligible_episode_skips_expensive_stages(tmp_path: Path):
    data_dir, episode = publish(tmp_path)
    manifest_path = episode / "manifest.json"
    value = runner_module.manifest.read_manifest(manifest_path)
    value["dataset_eligible"] = False
    runner_module.manifest.write_manifest(manifest_path, value)
    calls: list[tuple[str, str]] = []
    runner = make_runner(data_dir, calls)

    assert runner.scan_once() == 0
    assert calls == []
    receipt = data_dir / "derived" / episode.name / "dataset" / "receipt.json"
    assert "episode not dataset_eligible" in receipt.read_text()


def test_clock_invalid_episode_skips_expensive_stages(tmp_path: Path):
    data_dir, episode = publish(tmp_path)
    manifest_path = episode / "manifest.json"
    value = runner_module.manifest.read_manifest(manifest_path)
    value["clock_sync"] = {
        "synced": False,
        "estimated_cross_pi_offset_ns": None,
    }
    runner_module.manifest.write_manifest(manifest_path, value)
    calls: list[tuple[str, str]] = []
    runner = make_runner(data_dir, calls)

    assert runner.scan_once() == 0
    assert calls == []
    receipt = data_dir / "derived" / episode.name / "dataset" / "receipt.json"
    assert "cross-Pi clock offset" in receipt.read_text()


def test_resource_failure_is_not_terminalized(tmp_path: Path):
    data_dir, episode = publish(tmp_path)
    now = [100.0]

    def fail(*_args) -> None:
        raise OSError("[Errno 28] No space left on device")

    stage = runner_module.StageDef(name="decode", scopes=lambda _m, _rig: ["umi1"], fn=fail)
    runner = Runner(
        data_dir=data_dir,
        stages=[stage],
        retry_initial_s=1,
        max_deterministic_attempts=1,
        now_fn=lambda: now[0],
    )
    assert runner.scan_once() == 0
    state = runner._failure(StageRun(episode.name, "decode", "umi1"))
    assert state is not None
    assert state.classification == "retryable_resource"
    assert not state.terminal
    assert not (data_dir / "derived" / episode.name / "dataset" / "COMPLETE").exists()


def test_runner_pauses_when_disk_reserve_is_reached(tmp_path: Path, monkeypatch):
    data_dir, _ = publish(tmp_path)
    runner = make_runner(data_dir, [])
    runner.min_free_bytes = 100
    monkeypatch.setattr(runner_module.shutil, "disk_usage", lambda _path: SimpleNamespace(free=99))

    assert runner.ready_runs() == []


def test_low_disk_blocks_decode_but_allows_downstream_completion(tmp_path: Path, monkeypatch):
    data_dir, episode = publish(tmp_path)
    runner = make_runner(data_dir, [])
    runner.min_free_bytes = 100
    runner.decode_min_free_bytes = 200
    decoded = runner.out_dir(StageRun(episode.name, "decode", "umi1"))
    decoded.mkdir(parents=True)
    (decoded / "COMPLETE").touch()
    monkeypatch.setattr(runner_module.shutil, "disk_usage", lambda _path: SimpleNamespace(free=150))

    ready = runner.ready_runs()

    assert all(stage.name != "decode" for stage, _, _ in ready)
    assert any(stage.name == "glim" and run.scope == "umi1" for stage, run, _ in ready)


def test_run_forever_never_queues_more_than_worker_limit(tmp_path: Path, monkeypatch):
    data_dir, episode = publish(tmp_path)
    runner = make_runner(data_dir, [])
    stage = runner.stages[0]
    ready = [
        (stage, StageRun(f"episode_{index}", stage.name, "umi1"), episode) for index in range(20)
    ]
    submitted = []

    class PendingFuture:
        @staticmethod
        def done() -> bool:
            return False

    class RecordingPool:
        def __init__(self, max_workers: int):
            assert max_workers == 3

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def submit(self, fn, *args):
            submitted.append((fn, args))
            return PendingFuture()

    class StopLoop(Exception):
        pass

    monkeypatch.setattr(runner, "ready_runs", lambda: ready)
    monkeypatch.setattr("umi_flow_server.pair_stage.pair_once", lambda _data_dir: None)
    monkeypatch.setattr("concurrent.futures.ThreadPoolExecutor", RecordingPool)
    monkeypatch.setattr(
        runner_module.time, "sleep", lambda _seconds: (_ for _ in ()).throw(StopLoop)
    )

    with pytest.raises(StopLoop):
        runner.run_forever(interval_s=0, workers=3)

    assert len(submitted) == 3


def test_executor_reports_stage_timeout(tmp_path: Path, monkeypatch):
    calls: list[list[str]] = []

    def timeout(cmd, **_kwargs):
        calls.append(cmd)
        if "compose" in cmd:
            raise subprocess.TimeoutExpired(cmd=cmd, timeout=7)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(runner_module.subprocess, "run", timeout)
    executor = Executor(tmp_path, timeout_s=7)
    stage = default_stages()[0]
    run = StageRun("episode", stage.name, "umi1")

    with pytest.raises(RuntimeError, match="timed out after 7s"):
        executor.run(stage, run, tmp_path / "input", tmp_path / "output")
    assert calls[0][4:6] == ["--name", executor._container_name(run)]
    assert calls[1] == ["docker", "rm", "-f", executor._container_name(run)]


def test_glim_stage_carries_extrinsic_env_fn():
    # The GPU glim stage must inject the per-lidar extrinsic; the compose stage decode
    # (and the in-process gripper/dataset) must not.
    stages = {s.name: s for s in default_stages()}
    assert stages["glim"].env_fn is not None
    assert stages["decode"].env_fn is None


def test_executor_threads_env_into_compose(tmp_path: Path):
    # A compose stage's env dict becomes `-e KEY=VALUE` flags before the service name.
    captured: dict = {}

    class RecordingExecutor(Executor):
        def run(self, stage, run, input_dir, staging, env=None):
            captured["env"] = env
            captured["cmd"] = [
                *self.compose_cmd,
                "run",
                "--rm",
                *[x for k, v in (env or {}).items() for x in ("-e", f"{k}={v}")],
                f"umi-{stage.name}",
            ]

    data_dir, _ = publish(tmp_path)
    stages = default_stages(dataset_fn=lambda *a: None, gripper_fn=lambda *a: None)
    glim = next(s for s in stages if s.name == "glim")
    # Drive execute() for the glim stage directly with a recording executor.
    runner = Runner(data_dir=data_dir, stages=[], executor=RecordingExecutor(data_dir))
    episode = next((data_dir / "raw_episodes").iterdir())
    # decode must be done first (glim's input_from); fake its output.
    decode_out = data_dir / "derived" / episode.name / "umi1" / "decode"
    decode_out.mkdir(parents=True)
    (decode_out / "COMPLETE").write_text("complete\n")
    runner.execute(glim, StageRun(episode.name, "glim", "umi1"), episode)
    assert "T_LIDAR_IMU" in (captured["env"] or {})
    assert "-e" in captured["cmd"]
    assert len(captured["env"]["T_LIDAR_IMU"].split()) == 7


def test_scoped_outputs_layout(tmp_path: Path):
    data_dir, episode = publish(tmp_path)
    calls: list[tuple[str, str]] = []
    runner = make_runner(data_dir, calls)
    runner.scan_once()
    derived = data_dir / "derived" / episode.name
    assert (derived / "umi1" / "glim" / "COMPLETE").exists()
    assert (derived / "umi2" / "glim" / "COMPLETE").exists()
    run = StageRun(episode.name, "glim", "umi2")
    assert runner.is_done(run)
