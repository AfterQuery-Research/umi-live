"""The pipeline runner: a filesystem state machine over stage-CLI containers.

The filesystem is the queue. Ingest publishes hermetic episodes into
``data/raw_episodes/<id>/``; each stage writes ``data/derived/<id>/<scope>/<stage>/``
(scope = a UMI name for per-lidar stages, "" for episode-wide ones). The runner
rescans for stages whose inputs are COMPLETE and outputs aren't, and executes them
— via ``docker compose run --rm <service> <in> <out>`` in production, or an
in-process callable for pure-Python stages and tests. Recovery IS the rescan: kill
the runner mid-pipeline and the next scan resumes exactly the missing stages. No
per-stage pending markers, no cross-container calls; containers meet at directories.

Stage protocol: the runner hands the stage a private ``.staging`` directory; on
success it atomically renames it to the final output dir and guarantees a COMPLETE
marker inside. A final dir without COMPLETE (crash between rename and marker) is
torn down and re-run.

Dispatch is by stream PRESENCE in the manifest (never the mode label): decode runs
iff a lidar pcap exists, engines iff decode is done, gripper iff a camera + gripper
config exist, dataset when every input it needs is done.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from robot_umi_core import config as core_config
from robot_umi_core import layout, manifest

from robot_umi_server import extrinsic

COMPLETE = layout.COMPLETE_FILE

# A stage callable runs in-process: (episode_dir, out_dir, scope) -> None.
StageFn = Callable[[Path, Path, str], None]


@dataclass(frozen=True)
class StageRun:
    """One executable unit of work: a stage applied to one episode (+ scope)."""

    episode_id: str
    stage: str
    scope: str  # UMI name for per-lidar stages, "" for episode-wide stages

    @property
    def rel_out(self) -> Path:
        return Path(self.episode_id) / (Path(self.scope) if self.scope else Path()) / self.stage


# (stage_name, scope) -> bool: is that output COMPLETE for this episode?
DoneCheck = Callable[[str, str], bool]


@dataclass
class StageDef:
    """A stage: when it applies, what it needs, and how to run it."""

    name: str
    # (manifest dict, rig config) -> scopes this stage should run for. [] = not applicable.
    scopes: Callable[[dict, core_config.RigConfig], list[str]]
    # stage names (same scope) whose COMPLETE outputs this stage needs first.
    needs: tuple[str, ...] = ()
    # Cross-scope readiness gate (e.g. dataset needs glim@every-lidar-umi + gripper@"").
    ready: Callable[[dict, core_config.RigConfig, DoneCheck], bool] | None = None
    # The stage (same scope) whose output dir is this stage's INPUT; None = the episode dir.
    input_from: str | None = None
    # in-process callable; None means "execute via the compose service of the same name".
    fn: StageFn | None = None
    # (episode_dir, scope) -> extra env for the compose run (e.g. the resolved T_lidar_imu
    # for glim). Ignored for in-process fn stages, which resolve their own env.
    env_fn: Callable[[Path, str], dict[str, str]] | None = None


class Executor:
    """Runs one StageRun into a staging dir. Compose in production, callable in-process."""

    def __init__(self, data_dir: Path, compose_cmd: list[str] | None = None):
        self.data_dir = data_dir
        self.compose_cmd = compose_cmd or ["docker", "compose"]

    def _containerize(self, p: Path) -> str:
        """Host path -> the stage container's view (DATA_DIR is mounted at /data)."""
        try:
            return str(Path("/data") / p.resolve().relative_to(self.data_dir.resolve()))
        except ValueError:
            return str(p)

    def run(
        self,
        stage: StageDef,
        run: StageRun,
        input_dir: Path,
        staging: Path,
        env: dict[str, str] | None = None,
    ) -> None:
        if stage.fn is not None:
            stage.fn(input_dir, staging, run.scope)
            return
        env_flags: list[str] = []
        for key, value in (env or {}).items():
            env_flags += ["-e", f"{key}={value}"]
        cmd = [
            *self.compose_cmd,
            "run",
            "--rm",
            *env_flags,
            f"umi-{stage.name}",
            self._containerize(input_dir),
            self._containerize(staging),
            *([run.scope] if run.scope else []),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"stage {stage.name} failed (rc={result.returncode}): "
                f"{result.stderr.strip()[:500]}"
            )


@dataclass
class Runner:
    data_dir: Path
    stages: list[StageDef]
    executor: Executor | None = None
    failed: dict[tuple[str, str, str], str] = field(default_factory=dict)
    # When pairing is enabled, skip role-tagged single-arm episodes: they are one half of
    # a coordinated pair and must be merged (pair_stage) before processing, never run
    # alone into a 6-D dataset. Solo/merged episodes have no `role` and process normally.
    pair_sessions: bool = False

    def __post_init__(self) -> None:
        if self.executor is None:
            self.executor = Executor(self.data_dir)

    # --- filesystem state ---------------------------------------------------------

    @property
    def raw_root(self) -> Path:
        return self.data_dir / "raw_episodes"

    @property
    def derived_root(self) -> Path:
        return self.data_dir / "derived"

    def out_dir(self, run: StageRun) -> Path:
        return self.derived_root / run.rel_out

    def is_done(self, run: StageRun) -> bool:
        return (self.out_dir(run) / COMPLETE).exists()

    def _episodes(self) -> list[Path]:
        if not self.raw_root.exists():
            return []
        return sorted(
            p
            for p in self.raw_root.iterdir()
            if p.is_dir() and (p / COMPLETE).exists()
        )

    # --- planning -------------------------------------------------------------------

    def ready_runs(self) -> list[tuple[StageDef, StageRun, Path]]:
        """Every (stage, run) whose inputs are complete and output isn't."""
        ready = []
        for episode_dir in self._episodes():
            try:
                m = manifest.read_manifest(episode_dir / layout.MANIFEST_FILE)
                rig = core_config.load_rig_config(
                    episode_dir / layout.CONFIG_SNAPSHOT_DIR / layout.CONFIG_SNAPSHOT_YAML
                )
            except (ValueError, OSError) as exc:
                print(f"skipping unreadable episode {episode_dir.name}: {exc}", flush=True)
                continue
            if self.pair_sessions and m.get("role") is not None:
                continue  # an unmerged single arm of a pair; pair_stage will consume it
            for stage in self.stages:
                for scope in stage.scopes(m, rig):
                    run = StageRun(episode_dir.name, stage.name, scope)
                    key = (run.episode_id, run.stage, run.scope)
                    if key in self.failed or self.is_done(run):
                        continue
                    deps_done = all(
                        self.is_done(StageRun(run.episode_id, dep, scope))
                        for dep in stage.needs
                    )
                    if deps_done and stage.ready is not None:
                        eid = run.episode_id

                        def _done(name: str, sc: str, _eid: str = eid) -> bool:
                            return self.is_done(StageRun(_eid, name, sc))

                        deps_done = stage.ready(m, rig, _done)
                    if deps_done:
                        ready.append((stage, run, episode_dir))
        return ready

    # --- execution --------------------------------------------------------------------

    def execute(self, stage: StageDef, run: StageRun, episode_dir: Path) -> bool:
        input_dir = episode_dir
        if stage.input_from is not None:
            input_dir = self.out_dir(StageRun(run.episode_id, stage.input_from, run.scope))
        out = self.out_dir(run)
        staging = out.parent / f".{run.stage}.staging"
        # Stage containers write their output as root (no USER in the images). If a
        # prior run failed and left a root-owned tree here, an unguarded rmtree
        # raises PermissionError — which, before this guard, propagated out of
        # scan_once() and killed the runner on EVERY rescan. Treat it as a stage
        # failure (blacklisted until restart) with an actionable message instead.
        try:
            if out.exists() and not (out / COMPLETE).exists():
                shutil.rmtree(out)  # crashed half-output: tear down and re-run
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True)
        except OSError as exc:
            self.failed[(run.episode_id, run.stage, run.scope)] = str(exc)
            print(
                f"[{run.episode_id}] {run.stage}({run.scope}) FAILED to clear stale "
                f"output (root-owned stage files? try `sudo rm -rf {out} {staging}`): "
                f"{exc}",
                flush=True,
            )
            return False
        try:
            assert self.executor is not None
            # env_fn feeds the compose run only; in-process fn stages resolve their own.
            env = (
                stage.env_fn(episode_dir, run.scope)
                if stage.fn is None and stage.env_fn is not None
                else None
            )
            self.executor.run(stage, run, input_dir, staging, env=env)
        except Exception as exc:
            self.failed[(run.episode_id, run.stage, run.scope)] = str(exc)
            print(f"[{run.episode_id}] {run.stage}({run.scope}) FAILED: {exc}", flush=True)
            shutil.rmtree(staging, ignore_errors=True)
            return False
        staging.rename(out)
        marker = out / COMPLETE
        if not marker.exists():
            marker.write_text("complete\n")
        print(f"[{run.episode_id}] {run.stage}({run.scope}) done -> {out}", flush=True)
        return True

    def scan_once(self) -> int:
        """One full pass; returns the number of stage runs executed."""
        executed = 0
        progressed = True
        while progressed:  # keep going while completions unlock dependents
            progressed = False
            for stage, run, episode_dir in self.ready_runs():
                if self.execute(stage, run, episode_dir):
                    executed += 1
                    progressed = True
        return executed

    def run_forever(self, interval_s: float = 5.0) -> None:
        print(f"pipeline runner watching {self.raw_root}", flush=True)
        while True:
            self.scan_once()
            time.sleep(interval_s)


# --- the standard stage graph ----------------------------------------------------------


def _lidar_scopes(m: dict, _rig: core_config.RigConfig) -> list[str]:
    return [key.split("/")[0] for key in manifest.present_streams(m, layout.KIND_LIDAR)]


def _glim_env(episode_dir: Path, scope: str) -> dict[str, str]:
    """Compose the GLIM container env: this lidar's T_lidar_imu, plus EXPORT_MAP=1 when the
    rig opts into alignment (so the container also exports its map for the align stage).
    With alignment off (default) the env is exactly the historical T_lidar_imu only."""
    env = dict(extrinsic.t_lidar_imu_env(episode_dir, scope))
    try:
        rig = core_config.load_rig_config(
            episode_dir / layout.CONFIG_SNAPSHOT_DIR / layout.CONFIG_SNAPSHOT_YAML
        )
        if rig.alignment.enabled:
            env["EXPORT_MAP"] = "1"
    except (ValueError, OSError):
        pass
    return env


def _gripper_scope(m: dict, rig: core_config.RigConfig) -> list[str]:
    cams = {k.split("/")[0] for k in manifest.present_streams(m, layout.KIND_CAMERA)}
    if any(u.gripper is not None and u.name in cams for u in rig.umis):
        return [""]  # one episode-wide run; the stage iterates UMIs itself
    return []


def _always(_m: dict, _rig: core_config.RigConfig) -> list[str]:
    return [""]


def _align_scopes(m: dict, rig: core_config.RigConfig) -> list[str]:
    """The cross-lidar align stage runs once (episode-wide) ONLY when alignment is
    explicitly enabled AND the episode has >=2 lidars. Disabled (the default) or a
    single lidar => [] so the stage is never scheduled and each lidar's GLIM trajectory
    is consumed directly — the historical non-joint path."""
    if not rig.alignment.enabled:
        return []
    if len(_lidar_scopes(m, rig)) < 2:
        return []
    return [""]


def _align_ready(m: dict, rig: core_config.RigConfig, done: DoneCheck) -> bool:
    """Align needs every lidar UMI's GLIM stage (trajectory + map export) done first."""
    return all(done("glim", umi) for umi in _lidar_scopes(m, rig))


def _dataset_ready(m: dict, rig: core_config.RigConfig, done: DoneCheck) -> bool:
    """Dataset needs the SLAM trajectory for every lidar stream and the gripper
    widths when they apply. Vive-only episodes have no lidar deps. When alignment is
    enabled on a multi-lidar episode it also waits for the align stage (which rewrites
    the per-lidar trajectories into one common frame)."""
    lidar_umis = _lidar_scopes(m, rig)
    for umi in lidar_umis:
        if not done("glim", umi):
            return False
    if rig.alignment.enabled and len(lidar_umis) >= 2 and not done("align", ""):
        return False
    if _gripper_scope(m, rig) and not done("gripper", ""):
        return False
    return True


def default_stages(
    dataset_fn: StageFn | None = None,
    gripper_fn: StageFn | None = None,
    align_fn: StageFn | None = None,
) -> list[StageDef]:
    """The production (GPU) graph. decode/glim run in their compose services; gripper,
    align, and dataset run in-process by default (pure Python) unless overridden.

    ``align`` is scheduled only when the rig config enables it on a multi-lidar episode
    (``_align_scopes``); otherwise the graph is exactly decode -> glim -> gripper/dataset."""
    return [
        StageDef(name="decode", scopes=_lidar_scopes),
        StageDef(
            name="glim",
            scopes=_lidar_scopes,
            needs=("decode",),
            input_from="decode",
            env_fn=_glim_env,
        ),
        StageDef(name="align", scopes=_align_scopes, fn=align_fn, ready=_align_ready),
        StageDef(name="gripper", scopes=_gripper_scope, fn=gripper_fn),
        StageDef(name="dataset", scopes=_always, fn=dataset_fn, ready=_dataset_ready),
    ]


def mac_stages(
    glim_fn: StageFn,
    dataset_fn: StageFn | None = None,
    gripper_fn: StageFn | None = None,
    align_fn: StageFn | None = None,
) -> list[StageDef]:
    """The Mac (GPU-less) graph: a single in-process ``glim`` stage replays each lidar
    UMI's pcap through the arm64 CPU GLIM container straight to trajectory.tum — no
    separate decode/rosbag stage (rslidar_sdk replays the pcap directly). gripper, align,
    and dataset are the same in-process stages as default_stages().

    The stage keeps the name ``glim`` on purpose: the dataset stage reads
    ``<umi>/glim/trajectory.tum`` and _dataset_ready gates on ``done("glim", ...)``,
    so this is a drop-in engine swap, not a new artifact layout. ``align`` is scheduled
    only when the rig config enables it on a multi-lidar episode (``_align_scopes``)."""
    return [
        StageDef(name="glim", scopes=_lidar_scopes, fn=glim_fn),
        StageDef(name="align", scopes=_align_scopes, fn=align_fn, ready=_align_ready),
        StageDef(name="gripper", scopes=_gripper_scope, fn=gripper_fn),
        StageDef(name="dataset", scopes=_always, fn=dataset_fn, ready=_dataset_ready),
    ]
