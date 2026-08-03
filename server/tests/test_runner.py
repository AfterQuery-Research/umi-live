"""Pipeline-runner tests: the filesystem state machine, dependencies, recovery."""

from __future__ import annotations

from pathlib import Path

from robot_umi_server.runner import Executor, Runner, StageRun, default_stages

from tests.helpers import make_v2_episode


def publish(tmp_path: Path, n_umis: int = 1) -> tuple[Path, Path]:
    """A validated episode in data/raw_episodes (as the ingest would leave it)."""
    episode = make_v2_episode(tmp_path, n_umis=n_umis)
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
    # lidar present, no gripper config -> decode + glim + dataset (no gripper).
    assert ("decode", "umi1") in calls
    assert ("glim", "umi1") in calls
    assert ("dataset", "") in calls
    assert calls.index(("decode", "umi1")) < calls.index(("glim", "umi1"))
    assert calls.index(("glim", "umi1")) < calls.index(("dataset", ""))
    assert executed == 3

    derived = data_dir / "derived" / episode.name
    assert (derived / "umi1" / "decode" / "COMPLETE").exists()
    assert (derived / "umi1" / "glim" / "trajectory.tum").exists()
    assert (derived / "dataset" / "COMPLETE").exists()


def test_rescan_is_idempotent(tmp_path: Path):
    data_dir, _ = publish(tmp_path)
    calls: list[tuple[str, str]] = []
    runner = make_runner(data_dir, calls)
    assert runner.scan_once() == 3
    assert runner.scan_once() == 0  # everything COMPLETE; nothing re-runs
    assert len(calls) == 3


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
    data_dir, _ = publish(tmp_path, n_umis=2)
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
                *self.compose_cmd, "run", "--rm",
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
    data_dir, episode = publish(tmp_path, n_umis=2)
    calls: list[tuple[str, str]] = []
    runner = make_runner(data_dir, calls)
    runner.scan_once()
    derived = data_dir / "derived" / episode.name
    assert (derived / "umi1" / "glim" / "COMPLETE").exists()
    assert (derived / "umi2" / "glim" / "COMPLETE").exists()
    run = StageRun(episode.name, "glim", "umi2")
    assert runner.is_done(run)
