"""Episode recording: N sensors behind one seam, hermetic episode directories.

An episode directory is the complete device->server contract (see
robot_umi_core.layout): raw sensor streams + a resolved config snapshot (the YAML
plus every referenced calib blob) + a versioned manifest + checksums + COMPLETE.

The recorder holds a flat list of :class:`SensorRecorder`s chosen by the episode's
mode (the enable mask, from ``robot_umi_core.config.capture_plan``). Non-camera
sensors start first so a camera-start failure never orphans a detached recorder
process; failures at stop never discard data — they are reported in the manifest
(per-sensor ``error``, ``dataset_eligible=false``).
"""

from __future__ import annotations

import json
import shutil
import socket
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from robot_umi_core import config as core_config
from robot_umi_core import layout, manifest

from robot_umi_device.sensors import CameraRecorder, SensorRecorder

PtpProbe = Callable[[], dict[str, manifest.PtpStatus]]
ClockSyncProbe = Callable[[], manifest.ClockSyncStatus | None]


def _no_ptp() -> dict[str, manifest.PtpStatus]:
    return {}


def _no_clock_sync() -> manifest.ClockSyncStatus | None:
    return None


@dataclass
class ActiveEpisode:
    episode_id: str
    path: Path
    mode: str
    sensors: list[SensorRecorder]
    start_unix_ns: int
    session_id: str | None = None  # coordinated multi-Pi pairing key (None = solo)
    role: str | None = None  # this Pi's arm, e.g. "umi1"/"umi2" (None = solo)


class EpisodeRecorder:
    def __init__(
        self,
        rig: core_config.RigConfig,
        config_path: Path,
        spool_dir: Path,
        sensors_by_key: dict[str, SensorRecorder],
        *,
        config_root: Path | None = None,
        ptp_probe: PtpProbe = _no_ptp,
        clock_sync_probe: ClockSyncProbe = _no_clock_sync,
    ):
        """``sensors_by_key`` holds one recorder per configured sensor (all modes'
        union); each episode arms the subset its mode selects. ``config_root`` is
        the base for the config's relative blob paths (defaults to the config
        file's repo root)."""
        self.rig = rig
        self.config_path = config_path
        self.config_root = config_root or config_path.resolve().parent.parent
        self.spool_dir = spool_dir.expanduser()
        self.sensors_by_key = sensors_by_key
        self.ptp_probe = ptp_probe
        self.clock_sync_probe = clock_sync_probe
        self.in_progress_dir = self.spool_dir / "in_progress"
        self.complete_dir = self.spool_dir / "complete"
        self.sent_dir = self.spool_dir / "sent"
        self.active: ActiveEpisode | None = None
        self.last_stop_issues: list[str] = []
        for path in [self.in_progress_dir, self.complete_dir, self.sent_dir]:
            path.mkdir(parents=True, exist_ok=True)

    # --- lifecycle -------------------------------------------------------------------

    def start(
        self, mode: str, *, session_id: str | None = None, role: str | None = None
    ) -> ActiveEpisode:
        if self.active is not None:
            raise RuntimeError("episode already recording")
        plan = core_config.capture_plan(self.rig, mode)
        armed: list[SensorRecorder] = []
        for key, _kind in plan:
            recorder = self.sensors_by_key.get(key)
            if recorder is None:
                raise RuntimeError(f"mode {mode!r} arms {key} but no recorder is wired")
            armed.append(recorder)

        eid = layout.episode_id()
        path = self.in_progress_dir / eid
        path.mkdir(parents=True, exist_ok=False)
        self._write_event(path, "start")
        self._write_config_snapshot(path)
        start_unix = layout.unix_ns()

        # Cameras last: a camera-start failure then has no detached process to
        # orphan — everything started so far is cancellable in-process.
        ordered = [s for s in armed if not isinstance(s, CameraRecorder)] + [
            s for s in armed if isinstance(s, CameraRecorder)
        ]
        started: list[SensorRecorder] = []
        try:
            for sensor in ordered:
                sensor.begin_episode(path)
                started.append(sensor)
        except Exception:
            for sensor in started:
                sensor.cancel_episode()
            shutil.rmtree(path, ignore_errors=True)
            raise

        self.active = ActiveEpisode(
            eid, path, mode, ordered, start_unix, session_id=session_id, role=role
        )
        return self.active

    def poll_active_error(self) -> str | None:
        if self.active is None:
            return None
        for sensor in self.active.sensors:
            error = sensor.poll_error()
            if error is not None:
                return f"{sensor.sensor_key}: {error}"
        return None

    def stop(self) -> Path:
        """Finish the active episode, preserving data even when sensors failed."""
        if self.active is None:
            raise RuntimeError("no active episode")
        active = self.active
        self.last_stop_issues = []
        self._write_event(active.path, "stop")

        entries: dict[str, manifest.SensorEntry] = {}
        for sensor in active.sensors:
            try:
                result = sensor.end_episode()
            except Exception as exc:  # a sensor's failure must not lose the others
                result_stats: dict = {}
                result_error: str | None = str(exc)
            else:
                result_stats, result_error = result.stats, result.error
            kind = sensor.sensor_key.split("/")[1]
            entries[sensor.sensor_key] = manifest.SensorEntry(
                kind=kind,
                armed=True,
                files=layout.sensor_files(sensor.sensor_key),
                stats=_json_safe(result_stats),
                error=result_error,
            )
            if result_error:
                self.last_stop_issues.append(f"{sensor.sensor_key}: {result_error}")

        stop_unix = layout.unix_ns()
        snapshot_yaml = active.path / layout.CONFIG_SNAPSHOT_DIR / layout.CONFIG_SNAPSHOT_YAML
        m = manifest.build_manifest(
            episode_id=active.episode_id,
            device_id=socket.gethostname(),
            mode=active.mode,
            task=self.rig.task,
            sensors=entries,
            ptp=self.ptp_probe(),
            config_sha256=layout.sha256_file(snapshot_yaml),
            start_unix_ns=active.start_unix_ns,
            stop_unix_ns=stop_unix,
            files={},
            session_id=active.session_id,
            role=active.role,
            clock_sync=self.clock_sync_probe(),
        )
        manifest.write_manifest(active.path / layout.MANIFEST_FILE, m)
        hashes = write_checksums(active.path)
        self._update_manifest_hashes(active.path, hashes)
        (active.path / layout.COMPLETE_FILE).write_text("complete\n", encoding="utf-8")
        final_path = self.complete_dir / active.episode_id
        shutil.move(str(active.path), final_path)
        self.active = None
        return final_path

    def cancel(self) -> None:
        if self.active is None:
            return
        active = self.active
        self._write_event(active.path, "cancel")
        for sensor in active.sensors:
            sensor.cancel_episode()
        shutil.rmtree(active.path, ignore_errors=True)
        self.active = None

    # --- internals -------------------------------------------------------------------

    def _write_config_snapshot(self, path: Path) -> None:
        """The hermetic-episode core: the resolved YAML + every referenced blob."""
        snap = path / layout.CONFIG_SNAPSHOT_DIR
        snap.mkdir()
        shutil.copyfile(self.config_path, snap / layout.CONFIG_SNAPSHOT_YAML)
        for rel in core_config.referenced_blobs(self.rig):
            src = self.config_root / rel
            dst = snap / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)

    def _write_event(self, path: Path, event: str) -> None:
        with (path / layout.EVENTS_FILE).open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "event": event,
                        "time_monotonic_ns": layout.monotonic_ns(),
                        "time_unix_ns": layout.unix_ns(),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )

    def _update_manifest_hashes(self, path: Path, hashes: dict[str, str]) -> None:
        manifest_path = path / layout.MANIFEST_FILE
        m = json.loads(manifest_path.read_text(encoding="utf-8"))
        m["files"] = {
            name: {"sha256": digest, "bytes": (path / name).stat().st_size}
            for name, digest in sorted(hashes.items())
        }
        manifest_path.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_checksums(path: Path) -> dict[str, str]:
    """Hash every data file, RECURSIVELY (config_snapshot/ ships blobs in subdirs)."""
    skip = {layout.CHECKSUMS_FILE, layout.COMPLETE_FILE, layout.MANIFEST_FILE}
    hashes: dict[str, str] = {}
    for child in sorted(path.rglob("*")):
        if not child.is_file():
            continue
        rel = child.relative_to(path).as_posix()
        if rel in skip:
            continue
        hashes[rel] = layout.sha256_file(child)
    with (path / layout.CHECKSUMS_FILE).open("w", encoding="utf-8") as f:
        for name, digest in hashes.items():
            f.write(f"{digest}  {name}\n")
    return hashes


def _json_safe(d: dict) -> dict:
    return json.loads(json.dumps(d, default=str))
