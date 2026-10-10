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

Dispatch is by stream presence in the manifest: decode runs
iff a lidar pcap exists, engines iff decode is done, gripper iff a camera + gripper
config exist, dataset when every input it needs is done.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from umi_flow_core import config as core_config
from umi_flow_core import layout, manifest

from umi_flow_server import extrinsic
from umi_flow_server.episode import DEFAULT_MIN_FREE_BYTES
from umi_flow_server.operation_lock import PipelineLease

COMPLETE = layout.COMPLETE_FILE

# Concurrent stage slots for run_forever. Sized for the box (52 cores, 2x H100):
# decode is the heavy stage (~2-4 cores each); GLIM holds the GPU ~10 s per arm;
# dataset never parallelizes (see run_forever). 5 keeps 2-3 episodes fully in
# flight without starving anything.
MAX_STAGE_WORKERS = 5
DEFAULT_STAGE_TIMEOUT_S = 2 * 60 * 60
DEFAULT_RETRY_INITIAL_S = 30.0
DEFAULT_RETRY_MAX_S = 15 * 60.0
DEFAULT_MAX_DETERMINISTIC_ATTEMPTS = 3
# Stop expanding raw PCAPs before the ingest reserve is reached. Downstream
# stages can still finish and make their decoded bags eligible for retention.
DEFAULT_DECODE_MIN_FREE_BYTES = 32 * 1024 * 1024 * 1024

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

    def __init__(
        self,
        data_dir: Path,
        compose_cmd: list[str] | None = None,
        timeout_s: float = DEFAULT_STAGE_TIMEOUT_S,
    ):
        self.data_dir = data_dir
        self.compose_cmd = compose_cmd or ["docker", "compose"]
        self.timeout_s = timeout_s

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
            "--name",
            self._container_name(run),
            *env_flags,
            f"umi-{stage.name}",
            self._containerize(input_dir),
            self._containerize(staging),
            *([run.scope] if run.scope else []),
        ]
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            # Killing the compose client alone can leave its one-shot container
            # running and writing into the staging directory during a retry.
            subprocess.run(
                [*self._docker_cli(), "rm", "-f", self._container_name(run)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            raise RuntimeError(f"stage {stage.name} timed out after {self.timeout_s:g}s") from exc
        if result.returncode != 0:
            raise RuntimeError(
                f"stage {stage.name} failed (rc={result.returncode}): {result.stderr.strip()[:500]}"
            )

    def _container_name(self, run: StageRun) -> str:
        identity = f"{run.episode_id}\0{run.stage}\0{run.scope}".encode()
        return f"umi-flow-{run.stage}-{hashlib.sha256(identity).hexdigest()[:12]}"

    def _docker_cli(self) -> list[str]:
        # Production uses `docker compose`; preserve wrappers such as `sudo docker`.
        if self.compose_cmd and self.compose_cmd[-1] == "compose":
            return self.compose_cmd[:-1]
        return ["docker"]


@dataclass(frozen=True)
class FailureState:
    error: str
    attempts: int
    retry_at: float
    classification: str = "algorithmic"
    terminal: bool = False
    first_failed_at_unix: float | None = None
    last_failed_at_unix: float | None = None
    source_manifest_sha256: str = ""
    runner_sha256: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "error": self.error,
            "attempts": self.attempts,
            "retry_at_unix": self.retry_at,
            "classification": self.classification,
            "terminal": self.terminal,
            "first_failed_at_unix": self.first_failed_at_unix,
            "last_failed_at_unix": self.last_failed_at_unix,
            "source_manifest_sha256": self.source_manifest_sha256,
            "runner_sha256": self.runner_sha256,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> FailureState:
        return cls(
            error=str(value["error"]),
            attempts=int(value["attempts"]),
            retry_at=float(value["retry_at_unix"]),
            classification=str(value.get("classification", "algorithmic")),
            terminal=bool(value.get("terminal", False)),
            first_failed_at_unix=(
                float(value["first_failed_at_unix"])
                if value.get("first_failed_at_unix") is not None
                else None
            ),
            last_failed_at_unix=(
                float(value["last_failed_at_unix"])
                if value.get("last_failed_at_unix") is not None
                else None
            ),
            source_manifest_sha256=str(value.get("source_manifest_sha256", "")),
            runner_sha256=str(value.get("runner_sha256", "")),
        )


@dataclass
class Runner:
    data_dir: Path
    stages: list[StageDef]
    executor: Executor | None = None
    failed: dict[tuple[str, str, str], FailureState] = field(default_factory=dict)
    retry_initial_s: float = DEFAULT_RETRY_INITIAL_S
    retry_max_s: float = DEFAULT_RETRY_MAX_S
    max_deterministic_attempts: int = DEFAULT_MAX_DETERMINISTIC_ATTEMPTS
    min_free_bytes: int = DEFAULT_MIN_FREE_BYTES
    decode_min_free_bytes: int = DEFAULT_DECODE_MIN_FREE_BYTES
    now_fn: Callable[[], float] = field(default=time.time, repr=False)
    _disk_blocked: bool = field(default=False, init=False, repr=False)

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

    def failure_path(self, run: StageRun) -> Path:
        scope = run.scope or "_episode"
        return self.derived_root / run.episode_id / ".failures" / scope / f"{run.stage}.json"

    def _write_failure(self, run: StageRun, state: FailureState) -> None:
        path = self.failure_path(run)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                json.dump(state.to_dict(), handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)

    def _failure(self, run: StageRun) -> FailureState | None:
        key = (run.episode_id, run.stage, run.scope)
        if key in self.failed:
            return self.failed[key]
        path = self.failure_path(run)
        if not path.exists():
            return None
        try:
            state = FailureState.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"ignoring malformed failure state {path}: {exc}", flush=True)
            return None
        self.failed[key] = state
        return state

    def _clear_failure(self, run: StageRun) -> None:
        self.failed.pop((run.episode_id, run.stage, run.scope), None)
        self.failure_path(run).unlink(missing_ok=True)

    def _terminalize_episode(
        self,
        episode_id: str,
        reason: str,
        *,
        failed_run: StageRun | None = None,
    ) -> None:
        """Write the ordinary dataset-stage terminal contract for unusable input."""
        out = self.derived_root / episode_id / "dataset"
        if (out / COMPLETE).exists():
            return
        out.mkdir(parents=True, exist_ok=True)
        receipt: dict[str, object] = {
            "episode_id": episode_id,
            "skipped": reason,
            "terminal": True,
        }
        if failed_run is not None:
            receipt["failed_stage"] = failed_run.stage
            receipt["failed_scope"] = failed_run.scope
        receipt_path = out / "receipt.json"
        tmp = receipt_path.with_name(f".{receipt_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                json.dump(receipt, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            tmp.replace(receipt_path)
        finally:
            tmp.unlink(missing_ok=True)
        (out / COMPLETE).write_text("complete\n")
        print(f"[{episode_id}] terminal dataset skip: {reason}", flush=True)

    def _episode_terminal(self, episode_id: str) -> bool:
        receipt = self.derived_root / episode_id / "dataset" / "receipt.json"
        try:
            terminal = bool(json.loads(receipt.read_text(encoding="utf-8")).get("terminal"))
        except (OSError, ValueError, TypeError):
            return False
        if terminal:
            marker = receipt.parent / COMPLETE
            if not marker.exists():
                marker.write_text("complete\n")
        return terminal

    @staticmethod
    def _eligibility_failure(m: dict) -> str | None:
        if not m.get("dataset_eligible", False):
            return "episode not dataset_eligible"
        if not manifest.lidar_ptp_acceptable(m):
            return "PTP not locked / residual too large for lidar SLAM"
        if not manifest.clock_sync_acceptable(m):
            return "cross-Pi clock offset too large / unsynced (bimanual merge)"
        return None

    def _has_disk_headroom(self) -> bool:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.data_dir).free
        blocked = free < self.min_free_bytes
        if blocked and not self._disk_blocked:
            print(
                f"pipeline paused: only {free} bytes free; requires {self.min_free_bytes}",
                flush=True,
            )
        elif not blocked and self._disk_blocked:
            print("pipeline resumed: disk space recovered", flush=True)
        self._disk_blocked = blocked
        return not blocked

    def _episodes(self) -> list[Path]:
        if not self.raw_root.exists():
            return []
        return sorted(p for p in self.raw_root.iterdir() if p.is_dir() and (p / COMPLETE).exists())

    # --- planning -------------------------------------------------------------------

    def ready_runs(self) -> list[tuple[StageDef, StageRun, Path]]:
        """Every (stage, run) whose inputs are complete and output isn't."""
        if not self._has_disk_headroom():
            return []
        can_decode = shutil.disk_usage(self.data_dir).free >= self.decode_min_free_bytes
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
            if m.get("role") is not None:
                continue  # an unmerged single arm of a pair; pair_stage will consume it
            if self._episode_terminal(episode_dir.name):
                continue
            eligibility_failure = self._eligibility_failure(m)
            if eligibility_failure is not None:
                self._terminalize_episode(episode_dir.name, eligibility_failure)
                continue
            for stage in self.stages:
                if stage.name == "decode" and not can_decode:
                    continue
                for scope in stage.scopes(m, rig):
                    run = StageRun(episode_dir.name, stage.name, scope)
                    failure = self._failure(run)
                    if failure is not None and (
                        failure.terminal or self.now_fn() < failure.retry_at
                    ):
                        continue
                    if self.is_done(run):
                        continue
                    deps_done = all(
                        self.is_done(StageRun(run.episode_id, dep, scope)) for dep in stage.needs
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

    def _record_failure(self, run: StageRun, exc: Exception) -> float:
        key = (run.episode_id, run.stage, run.scope)
        previous = self._failure(run)
        attempts = 1 if previous is None else previous.attempts + 1
        error = str(exc)
        lowered = error.lower()
        retryable_markers = (
            "no space left",
            "enospc",
            "http 507",
            "connection reset",
            "connection refused",
            "temporarily unavailable",
            "timed out",
        )
        if any(marker in lowered for marker in retryable_markers):
            classification = "retryable_resource"
        elif isinstance(exc, (FileNotFoundError, ValueError)) or "rc=66" in lowered:
            classification = "deterministic_input"
        else:
            classification = "algorithmic"
        terminal = (
            classification != "retryable_resource" and attempts >= self.max_deterministic_attempts
        )
        now = self.now_fn()
        manifest_path = self.raw_root / run.episode_id / layout.MANIFEST_FILE
        delay = min(
            self.retry_max_s,
            self.retry_initial_s * (2 ** min(attempts - 1, 20)),
        )
        state = FailureState(
            error=error,
            attempts=attempts,
            retry_at=now + delay,
            classification=classification,
            terminal=terminal,
            first_failed_at_unix=(
                previous.first_failed_at_unix
                if previous is not None and previous.first_failed_at_unix is not None
                else now
            ),
            last_failed_at_unix=now,
            source_manifest_sha256=(
                layout.sha256_file(manifest_path) if manifest_path.is_file() else ""
            ),
            runner_sha256=layout.sha256_file(Path(__file__)),
        )
        self.failed[key] = state
        self._write_failure(run, state)
        if terminal:
            self._terminalize_episode(
                run.episode_id,
                f"pipeline stage {run.stage} reached terminal {classification} failure "
                f"after {attempts} attempts",
                failed_run=run,
            )
        return delay

    def execute(self, stage: StageDef, run: StageRun, episode_dir: Path) -> bool:
        lease = PipelineLease.for_data_dir(self.data_dir, exclusive=False)
        try:
            lease.acquire()
        except RuntimeError as exc:
            print(f"[{run.episode_id}] {run.stage} deferred: {exc}", flush=True)
            return False
        try:
            return self._execute_locked(stage, run, episode_dir)
        finally:
            lease.release()

    def _execute_locked(self, stage: StageDef, run: StageRun, episode_dir: Path) -> bool:
        input_dir = episode_dir
        if stage.input_from is not None:
            input_dir = self.out_dir(StageRun(run.episode_id, stage.input_from, run.scope))
        out = self.out_dir(run)
        staging = out.parent / f".{run.stage}.staging"
        # Stage containers write their output as root (no USER in the images). If a
        # prior run failed and left a root-owned tree here, an unguarded rmtree
        # raises PermissionError — which, before this guard, propagated out of
        # scan_once() and killed the runner on EVERY rescan. Treat it as a stage
        # failure with bounded-backoff retry and an actionable message instead.
        try:
            if out.exists() and not (out / COMPLETE).exists():
                shutil.rmtree(out)  # crashed half-output: tear down and re-run
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True)
        except OSError as exc:
            delay = self._record_failure(run, exc)
            print(
                f"[{run.episode_id}] {run.stage}({run.scope}) FAILED to clear stale "
                f"output (root-owned stage files? try `sudo rm -rf {out} {staging}`): "
                f"{exc}; retry in {delay:g}s",
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
            delay = self._record_failure(run, exc)
            print(
                f"[{run.episode_id}] {run.stage}({run.scope}) FAILED: {exc}; retry in {delay:g}s",
                flush=True,
            )
            shutil.rmtree(staging, ignore_errors=True)
            return False
        staging.rename(out)
        marker = out / COMPLETE
        if not marker.exists():
            marker.write_text("complete\n")
        self._clear_failure(run)
        print(f"[{run.episode_id}] {run.stage}({run.scope}) done -> {out}", flush=True)
        return True

    def scan_once(self) -> int:
        """Pair all complete arm uploads, then run every newly-ready stage (sequentially)."""
        from umi_flow_server import pair_stage

        pair_stage.pair_once(self.data_dir)
        executed = 0
        progressed = True
        while progressed:  # keep going while completions unlock dependents
            progressed = False
            for stage, run, episode_dir in self.ready_runs():
                if self.execute(stage, run, episode_dir):
                    executed += 1
                    progressed = True
        return executed

    def run_forever(self, interval_s: float = 5.0, workers: int = MAX_STAGE_WORKERS) -> None:
        """Concurrent pipeline loop: up to ``workers`` stages execute at once across
        episodes (and across the two arms of one episode), so a rapid-fire recording
        session drains in parallel instead of one take at a time. Two invariants:

        * The DATASET stage is globally serialized (``_dataset_lock``) — one LeRobot
          writer, one append ledger, one Hub mirror; concurrent appends would corrupt
          it. Everything upstream (decode/glim/gripper/align) parallelizes freely:
          stages meet only at per-episode directories.
        * Pairing runs every tick on the main thread, so freshly-uploaded arms merge
          immediately even while stages are executing (previously pairing waited for
          the whole sequential scan).

        Episode indices in the dataset follow append-completion order, which under
        concurrency may differ from recording order (a short take can finish before
        an earlier long one) — harmless for training data.
        """
        import threading
        from concurrent.futures import Future, ThreadPoolExecutor

        from umi_flow_server import pair_stage

        dataset_lock = threading.Lock()
        inflight: dict[tuple[str, str, str], Future] = {}

        def _job(stage: StageDef, run: StageRun, episode_dir: Path) -> None:
            if stage.name == "dataset":
                with dataset_lock:
                    self.execute(stage, run, episode_dir)
            else:
                self.execute(stage, run, episode_dir)

        print(
            f"pipeline runner watching {self.raw_root} ({workers} stage workers)",
            flush=True,
        )
        with ThreadPoolExecutor(max_workers=workers) as pool:
            while True:
                try:
                    pair_stage.pair_once(self.data_dir)
                except Exception as exc:  # pairing must never take down the loop
                    print(f"pair: pass failed, continuing: {exc}", flush=True)
                for key in [k for k, f in inflight.items() if f.done()]:
                    del inflight[key]
                for stage, run, episode_dir in self.ready_runs():
                    # ThreadPoolExecutor's work queue is unbounded. Keep the admitted
                    # set bounded too, or a large backlog can enqueue hundreds of jobs
                    # while disk is healthy and execute them after the reserve trips.
                    if len(inflight) >= workers:
                        break
                    key = (run.episode_id, run.stage, run.scope)
                    if key in inflight:
                        continue
                    inflight[key] = pool.submit(_job, stage, run, episode_dir)
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
    widths. When alignment is enabled it also waits for the align stage."""
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
    """The production graph. decode/glim run in their compose services; gripper,
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
