"""The Pi recorder session: rig config -> sensors -> button-driven episodes.

Each client records one camera and one RoboSense Airy lidar. The ``umi1`` config is
the button-driven leader; ``umi2`` is the headless follower. Solo recording and
runtime sensor-mode selection are intentionally unsupported.

Keyboard: s=start e=stop c=cancel q=quit (or tap/hold GPIO buttons).
"""

from __future__ import annotations

import contextlib
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import tyro
from umi_flow_core import config as core_config
from umi_flow_core import layout
from umi_flow_core.ports import bpf_filter
from umi_flow_device import coordinator
from umi_flow_device.buttons import Buttons, Event, GpioButtons, KeyboardButtons
from umi_flow_device.camera import RustV4L2Camera
from umi_flow_device.display import (
    ST7789,
    UPLOAD_FAIL,
    UPLOAD_OK,
    Display,
    NullDisplay,
    StatusDisplay,
)
from umi_flow_device.episode import EpisodeRecorder
from umi_flow_device.lidar_monitor import LidarMonitor
from umi_flow_device.operation_lock import CAPTURE_LOCK_FILE
from umi_flow_device.preflight import (
    CheckResult,
    PipelineResult,
    PreRecordingPipeline,
    build_pre_recording_pipeline,
)
from umi_flow_device.ptp import chrony_status, probe_ptp
from umi_flow_device.runtime_status import STATUS_FILE, RuntimeStatus
from umi_flow_device.safety import SafetyPolicy
from umi_flow_device.sensors import CameraRecorder, LidarPcapRecorder, SensorRecorder
from umi_flow_device.uploader import (
    DEFAULT_UPLOAD_TIMEOUT_SECONDS,
    AsyncUploader,
    server_admission,
)

SPOOL_DIR_ENV = "UMI_FLOW_SPOOL_DIR"
LEGACY_SPOOL_DIR_ENV = "ROBOT_UMI_SPOOL_DIR"
PEER_ENV = "UMI_FLOW_PEER"
LEGACY_PEER_ENV = "ROBOT_UMI_PEER"
DEFAULT_SPOOL_DIR = "umi_flow_spool"
LEGACY_DEFAULT_SPOOL_DIR = "robot_umi_spool"
UPLOAD_URL = "http://127.0.0.1:8080/upload"

# Follower watchdog: if a STOP is ever lost (leader crash / WiFi blip), auto-finish a take
# that has run this long so the follower never records forever.
MAX_TAKE_S = 1800.0
PREPARE_TTL_S = 30.0
HEARTBEAT_INTERVAL_S = 1.0


def _install_sigterm_handler(shutdown_requested: threading.Event):
    """Turn systemd's SIGTERM into an orderly loop exit.

    The main ``finally`` block owns capture finalization.  The handler therefore
    only sets a flag; it must not run sensor or filesystem code asynchronously.
    """

    def _handle_sigterm(_signum, _frame) -> None:
        if not shutdown_requested.is_set():
            print("SIGTERM received; finishing active episode before exit", flush=True)
        shutdown_requested.set()

    return signal.signal(signal.SIGTERM, _handle_sigterm)


@dataclass
class Args:
    config: Path | None = None  # rig YAML; falls back to $UMI_FLOW_CONFIG
    spool_dir: Path | None = None  # falls back to $UMI_FLOW_SPOOL_DIR, else umi_flow_spool
    buttons: Literal["keyboard", "gpio"] = "keyboard"
    # "auto" = take the LCD kind + control pins from the rig config's controls.display;
    # "none"/"lcd" force it (dev off-Pi wants --display none). Button pins always come
    # from controls.buttons.
    display: Literal["auto", "none", "lcd"] = "auto"
    retry_seconds: float = 5.0
    upload_timeout_seconds: float = DEFAULT_UPLOAD_TIMEOUT_SECONDS
    wait_on_quit: bool = False


def device_root() -> Path:
    """The device package dir (contains v4l2_recorder/) — NOT the repo root."""
    return Path(__file__).resolve().parents[1]


def resolve_config_path(args: Args) -> Path:
    if args.config is not None:
        return args.config.expanduser().resolve()
    env = os.environ.get(core_config.CONFIG_ENV_VAR) or os.environ.get(
        core_config.LEGACY_CONFIG_ENV_VAR
    )
    if env:
        return Path(env).expanduser().resolve()
    raise SystemExit(
        f"no rig config: pass --config or set ${core_config.CONFIG_ENV_VAR} "
        f"(legacy ${core_config.LEGACY_CONFIG_ENV_VAR} is also accepted)"
    )


def resolve_spool_dir(path: Path | None) -> Path:
    if path is None:
        env = os.environ.get(SPOOL_DIR_ENV) or os.environ.get(LEGACY_SPOOL_DIR_ENV)
        if env:
            path = Path(env)
        else:
            new_default = device_root() / DEFAULT_SPOOL_DIR
            legacy_default = device_root() / LEGACY_DEFAULT_SPOOL_DIR
            # Do not strand pending data during the naming migration. New installs use the
            # UMI Flow path; an existing legacy spool remains authoritative until moved.
            return (
                legacy_default
                if legacy_default.exists() and not new_default.exists()
                else new_default
            )
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else device_root() / expanded


def spool_has_data(path: Path) -> bool:
    if not path.exists():
        return False
    scaffolding = {"in_progress", "complete", "sent"}
    for child in path.iterdir():
        if child.name in {"staging", CAPTURE_LOCK_FILE, STATUS_FILE}:
            continue
        if child.name in scaffolding and child.is_dir():
            if any(child.iterdir()):
                return True
        else:
            return True
    return False


def confirm_existing_spool(path: Path) -> None:
    if not spool_has_data(path):
        return
    print(f"spool already has data: {path}")
    print("Complete episodes will be retried for upload on startup.")
    if not sys.stdin.isatty():
        # Scripted/piped runs must never block on (or worse, eat) stdin —
        # the pipe's first byte would be consumed as the confirmation.
        print("stdin is not a TTY; continuing")
        return
    try:
        input("Press Enter to continue, or Ctrl-C to quit.")
    except EOFError:
        print("stdin is not interactive; continuing")


def build_sensors(
    rig: core_config.RigConfig, args: Args, device_dir: Path
) -> tuple[dict[str, SensorRecorder], dict[str, LidarMonitor]]:
    """Build the fixed camera and lidar recorders plus advisory lidar monitoring."""
    sensors: dict[str, SensorRecorder] = {}
    monitors: dict[str, LidarMonitor] = {}

    for u in rig.umis:
        cam_key = layout.sensor_key(u.name, layout.KIND_CAMERA)
        camera = RustV4L2Camera(
            u.camera.device,
            u.camera.width,
            u.camera.height,
            u.camera.fps,
            exposure=u.camera.exposure,
            repo_root=device_dir,
        )
        cam = CameraRecorder(cam_key, camera)
        cam.prepare()
        sensors[cam_key] = cam

        lidar_key = layout.sensor_key(u.name, layout.KIND_LIDAR)
        sensors[lidar_key] = LidarPcapRecorder(
            lidar_key, u.lidar.iface, bpf_filter(u.lidar.host_ip)
        )
        monitor = LidarMonitor()
        monitor.start()
        monitors[u.name] = monitor

    return sensors, monitors


def make_buttons(rig: core_config.RigConfig, args: Args) -> Buttons:
    if args.buttons == "gpio":
        b = rig.controls.buttons
        return GpioButtons(b.record, b.quit)
    return KeyboardButtons()


def make_display(rig: core_config.RigConfig, args: Args, role: str) -> Display:
    kind = rig.controls.display.kind if args.display == "auto" else args.display
    if kind != "lcd":
        return NullDisplay()
    d = rig.controls.display
    # A follower's LCD is a lidar-health monitor only (the operator drives recording
    # from the leader), so hide the REC/IDLE box + episode count there.
    title = "Leader" if role == "leader" else "Follower"
    return StatusDisplay(
        driver=ST7789(dc=d.dc, rst=d.rst, bl=d.bl), show_status=role != "follower", title=title
    )


def pending_episodes(spool_dir: Path) -> int:
    complete = spool_dir / "complete"
    if not complete.exists():
        return 0
    return sum(1 for p in complete.iterdir() if p.is_dir())


def finish_active(recorder: EpisodeRecorder, uploader: AsyncUploader) -> None:
    episode_dir = recorder.stop()
    for issue in recorder.last_stop_issues:
        print(f"warning: {issue} (episode kept, marked not dataset eligible)")
    print(f"completed {episode_dir.name}")
    uploader.submit(episode_dir)


def finish_paired(recorder: EpisodeRecorder, uploader: AsyncUploader, leader) -> None:
    """Stop both recorders concurrently, then wait for the follower acknowledgement."""
    if leader is None:
        finish_active(recorder, uploader)
        return

    with ThreadPoolExecutor(max_workers=1) as executor:
        follower_stop = executor.submit(leader.stop_take)
        finish_active(recorder, uploader)
        follower_stop.result()


def _refresh_status(display, monitors, spool_dir, uploader, state) -> None:
    """The 2 Hz LCD/monitor/upload housekeeping shared by both loops."""
    now = time.monotonic()
    if now - state["last_monitor"] <= 0.5:
        return
    state["last_monitor"] = now
    for umi, monitor in monitors.items():
        display.set_lidar_health(umi, monitor.poll())
    display.set_episodes(pending_episodes(spool_dir))
    display.set_sending(uploader.current_upload())
    results = uploader.snapshot_results()
    if len(results) > state["last_result_count"]:
        state["last_result_count"] = len(results)
        display.notify_upload(UPLOAD_OK if results[-1].success else UPLOAD_FAIL)


def _leader_start(leader, session_id, handshake, display) -> bool:
    """Start the follower first; never record a useless unpaired half."""
    ack = leader.start_take(session_id)
    if ack is None or not ack.get("ok"):
        display.notify_upload(UPLOAD_FAIL)
        reason = ack.get("error", "unreachable") if ack else "unreachable"
        print(f"follower unavailable ({reason}); take aborted")
        return False
    fu = ack.get("follower_unix_ns")
    if fu is not None:
        handshake["v"] = time.time_ns() - int(fu)
    return True


def _failed_precheck(umi: str, message: str) -> PipelineResult:
    return PipelineResult((CheckResult(check="coordination", umi=umi, ok=False, message=message),))


def _remote_precheck(leader) -> PipelineResult:
    ack = leader.precheck()
    if ack is None:
        return _failed_precheck(core_config.FOLLOWER_UMI, "follower unavailable")
    try:
        result = PipelineResult.from_dict(ack.get("result"))
    except (TypeError, ValueError) as exc:
        return _failed_precheck(core_config.FOLLOWER_UMI, f"invalid follower result: {exc}")
    if bool(ack.get("ok")) != result.ok:
        return _failed_precheck(core_config.FOLLOWER_UMI, "inconsistent follower result")
    return result


def _check_failure_lines(failures: tuple[CheckResult, ...]) -> tuple[str, ...]:
    lines = []
    for failure in failures:
        if failure.missing_tag_ids:
            tag_ids = ",".join(str(tag_id) for tag_id in failure.missing_tag_ids)
            lines.append(f"{failure.umi} tags missing: {tag_ids}")
        else:
            message = failure.message or f"{failure.check} failed"
            lines.append(f"{failure.umi}: {message}"[:34])
    return tuple(lines)


def _preflight_before_start(local_pipeline: PreRecordingPipeline, leader, display: Display) -> bool:
    """Run both clients' checks before sending either client a START command."""
    display.clear_check_failure()
    local = local_pipeline.run()
    remote = _remote_precheck(leader)
    failures = (*local.failures, *remote.failures)
    if not failures:
        return True
    lines = _check_failure_lines(failures)
    display.notify_check_failure(lines)
    print(f"recording blocked by pre-recording check: {'; '.join(lines)}")
    return False


def _prepare_before_start(
    local_pipeline: PreRecordingPipeline,
    leader,
    session_id: str,
    display: Display,
) -> bool:
    """Prepare both arms without opening capture resources on either one."""
    display.clear_check_failure()
    local = local_pipeline.run()
    ack = leader.prepare_start(session_id)
    if ack is None:
        remote = _failed_precheck(core_config.FOLLOWER_UMI, "follower unavailable")
    else:
        try:
            remote = PipelineResult.from_dict(ack.get("result"))
        except (TypeError, ValueError) as exc:
            remote = _failed_precheck(core_config.FOLLOWER_UMI, f"invalid follower result: {exc}")
        else:
            if bool(ack.get("ok")) != remote.ok:
                remote = _failed_precheck(core_config.FOLLOWER_UMI, "inconsistent follower result")
    failures = (*local.failures, *remote.failures)
    if not failures:
        return True
    leader.abort_start(session_id)
    lines = _check_failure_lines(failures)
    display.notify_check_failure(lines)
    print(f"recording blocked by prepare: {'; '.join(lines)}")
    return False


def _respond_to_precheck(cmd: coordinator.Command, pipeline: PreRecordingPipeline) -> None:
    result = pipeline.run()
    cmd.respond(
        result.ok,
        "" if result.ok else "pre-recording check failed",
        result=result.to_dict(),
    )


def _set_runtime_status(runtime_status: RuntimeStatus | None, **values: object) -> None:
    if runtime_status is not None:
        runtime_status.update(**values)


def button_loop(
    recorder,
    uploader,
    monitors,
    display,
    spool_dir,
    buttons,
    leader,
    role,
    handshake,
    preflight_pipeline,
    runtime_status: RuntimeStatus | None = None,
    shutdown_requested: threading.Event | None = None,
) -> None:
    """The leader buttons drive synchronized two-client episodes."""
    state = {
        "last_monitor": 0.0,
        "last_result_count": 0,
        "last_heartbeat": 0.0,
        "last_peer_ok": time.monotonic(),
    }
    while not (shutdown_requested and shutdown_requested.is_set()):
        event = buttons.poll()
        _refresh_status(display, monitors, spool_dir, uploader, state)

        if recorder.active is not None:
            now = time.monotonic()
            if now - state["last_heartbeat"] >= HEARTBEAT_INTERVAL_S:
                state["last_heartbeat"] = now
                ack = leader.heartbeat(recorder.active.session_id)
                if ack is not None and ack.get("ok"):
                    state["last_peer_ok"] = now
                elif now - state["last_peer_ok"] > coordinator.PEER_LEASE_GRACE_S:
                    print("follower lease expired; stopping + keeping local data")
                    finish_active(recorder, uploader)
                    _set_runtime_status(
                        runtime_status, state="idle", session_id=None, episode_id=None
                    )
                    display.set_recording(False)
                    continue
            error = recorder.poll_active_error()
            if error is not None:
                print(f"sensor failed mid-episode ({error}); stopping + keeping data")
                finish_paired(recorder, uploader, leader)
                _set_runtime_status(runtime_status, state="idle", session_id=None, episode_id=None)
                display.set_recording(False)

        if event is None:
            time.sleep(0.02)
            continue

        recording = recorder.active is not None
        if event in (Event.TOGGLE, Event.START) and not recording:
            session_id = f"sess_{layout.unix_ns()}"
            if not _prepare_before_start(preflight_pipeline, leader, session_id, display):
                continue
            try:
                active = recorder.start(session_id=session_id, role=role)
            except Exception as exc:
                print(f"failed to start: {exc}")
                leader.abort_start(session_id)
                continue
            ack = leader.commit_start(session_id)
            if ack is None or not ack.get("ok"):
                reason = ack.get("error", "unreachable") if ack else "unreachable"
                print(f"follower commit failed ({reason}); cancelling local start")
                recorder.cancel()
                _set_runtime_status(runtime_status, state="idle", session_id=None, episode_id=None)
                display.notify_upload(UPLOAD_FAIL)
                continue
            fu = ack.get("follower_unix_ns")
            if fu is not None:
                handshake["v"] = time.time_ns() - int(fu)
            display.set_recording(True)
            state["last_peer_ok"] = time.monotonic()
            state["last_heartbeat"] = 0.0
            _set_runtime_status(
                runtime_status,
                state="recording",
                session_id=session_id,
                episode_id=active.episode_id,
            )
            print(f"recording {active.episode_id} (session {session_id})")
        elif event in (Event.TOGGLE, Event.STOP) and recording:
            finish_paired(recorder, uploader, leader)
            _set_runtime_status(runtime_status, state="idle", session_id=None, episode_id=None)
            display.set_recording(False)
        elif event is Event.CANCEL and recording:
            if leader is not None:
                leader.cancel_take()
            recorder.cancel()
            _set_runtime_status(runtime_status, state="idle", session_id=None, episode_id=None)
            display.set_recording(False)
            print("cancelled (episode discarded)")
        elif event is Event.QUIT:
            break


def follower_loop(
    recorder,
    uploader,
    monitors,
    display,
    spool_dir,
    follower,
    role,
    handshake,
    preflight_pipeline,
    runtime_status: RuntimeStatus | None = None,
    shutdown_requested: threading.Event | None = None,
) -> None:
    """The headless follower loop: leader commands (not buttons) drive episodes."""
    state = {
        "last_monitor": 0.0,
        "last_result_count": 0,
        "started_at": 0.0,
        "prepared_session": None,
        "prepared_at": 0.0,
    }
    while not (shutdown_requested and shutdown_requested.is_set()):
        cmd = follower.poll()
        _refresh_status(display, monitors, spool_dir, uploader, state)

        if (
            state["prepared_session"] is not None
            and time.monotonic() - state["prepared_at"] > PREPARE_TTL_S
        ):
            state["prepared_session"] = None
            follower.set_status(state="idle", session_id=None, episode_id=None)
            _set_runtime_status(runtime_status, state="idle", session_id=None, episode_id=None)

        if recorder.active is not None:
            if follower.lease_expired(recorder.active.session_id):
                print("leader lease expired; stopping + keeping follower data")
                finish_active(recorder, uploader)
                follower.set_status(state="idle", session_id=None, episode_id=None)
                _set_runtime_status(runtime_status, state="idle", session_id=None, episode_id=None)
                display.set_recording(False)
                continue
            error = recorder.poll_active_error()
            if error is not None:
                print(f"sensor failed mid-episode ({error}); stopping + keeping data")
                finish_active(recorder, uploader)
                follower.set_status(state="idle", session_id=None, episode_id=None)
                _set_runtime_status(runtime_status, state="idle", session_id=None, episode_id=None)
                display.set_recording(False)
            elif time.monotonic() - state["started_at"] > MAX_TAKE_S:
                print("follower watchdog: max take duration exceeded; stopping")
                finish_active(recorder, uploader)
                follower.set_status(state="idle", session_id=None, episode_id=None)
                _set_runtime_status(runtime_status, state="idle", session_id=None, episode_id=None)
                display.set_recording(False)

        if cmd is None:
            time.sleep(0.02)
            continue
        if cmd.expired:
            continue

        if (
            cmd.cmd == coordinator.CMD_PRECHECK
            and recorder.active is None
            and state["prepared_session"] is None
        ):
            _respond_to_precheck(cmd, preflight_pipeline)
        elif (
            cmd.cmd == coordinator.CMD_PREPARE_START
            and recorder.active is None
            and state["prepared_session"] in {None, cmd.session_id}
        ):
            result = preflight_pipeline.run()
            if result.ok:
                state["prepared_session"] = cmd.session_id
                state["prepared_at"] = time.monotonic()
                follower.set_status(state="prepared", session_id=cmd.session_id)
                _set_runtime_status(runtime_status, state="prepared", session_id=cmd.session_id)
            cmd.respond(
                result.ok,
                "" if result.ok else "pre-recording check failed",
                result=result.to_dict(),
            )
        elif (
            cmd.cmd in {coordinator.CMD_COMMIT_START, coordinator.CMD_START}
            and recorder.active is None
            and (cmd.cmd == coordinator.CMD_START or state["prepared_session"] == cmd.session_id)
        ):
            if cmd.leader_unix_ns is not None:
                handshake["v"] = time.time_ns() - int(cmd.leader_unix_ns)
            try:
                active = recorder.start(session_id=cmd.session_id or "", role=role)
                state["started_at"] = time.monotonic()
                state["prepared_session"] = None
                follower.set_status(
                    state="recording",
                    session_id=cmd.session_id,
                    episode_id=active.episode_id,
                )
                _set_runtime_status(
                    runtime_status,
                    state="recording",
                    session_id=cmd.session_id,
                    episode_id=active.episode_id,
                )
                display.set_recording(True)
                print(f"follower recording {active.episode_id} (session {cmd.session_id})")
            except Exception as exc:
                print(f"follower failed to start: {exc}")
                cmd.respond(False, str(exc))
            else:
                cmd.respond(True)
        elif (
            cmd.cmd == coordinator.CMD_ABORT_START
            and recorder.active is None
            and state["prepared_session"] in {None, cmd.session_id}
        ):
            state["prepared_session"] = None
            follower.set_status(state="idle", session_id=None, episode_id=None)
            _set_runtime_status(runtime_status, state="idle", session_id=None, episode_id=None)
            cmd.respond(True)
        elif cmd.cmd == coordinator.CMD_STOP and recorder.active is not None:
            try:
                finish_active(recorder, uploader)
            except Exception as exc:
                cmd.respond(False, str(exc))
                print(f"follower failed to stop: {exc}")
            else:
                follower.set_status(state="idle", session_id=None, episode_id=None)
                _set_runtime_status(runtime_status, state="idle", session_id=None, episode_id=None)
                display.set_recording(False)
                cmd.respond(True)
        elif cmd.cmd == coordinator.CMD_CANCEL and recorder.active is not None:
            try:
                recorder.cancel()
            except Exception as exc:
                cmd.respond(False, str(exc))
                print(f"follower failed to cancel: {exc}")
            else:
                follower.set_status(state="idle", session_id=None, episode_id=None)
                _set_runtime_status(runtime_status, state="idle", session_id=None, episode_id=None)
                display.set_recording(False)
                print("follower cancelled (episode discarded)")
                cmd.respond(True)
        else:
            cmd.respond(False, "command is invalid for the current recorder state")


def main(args: Args) -> None:
    config_path = resolve_config_path(args)
    rig = core_config.load_rig_config(config_path)
    role = core_config.client_role(rig)
    dev_dir = device_root()
    spool_dir = resolve_spool_dir(args.spool_dir)
    confirm_existing_spool(spool_dir)

    manifest_role = rig.umis[0].name
    loaded_identity = {
        "state": "idle",
        "session_id": None,
        "episode_id": None,
        "task": rig.task,
        "config_sha256": layout.sha256_file(config_path),
        "role": manifest_role,
        "coordinator_sha256": layout.sha256_file(Path(coordinator.__file__)),
        "recorder_script_sha256": layout.sha256_file(Path(__file__)),
    }
    runtime_status = RuntimeStatus(spool_dir, loaded_identity)
    handshake: dict[str, int | None] = {"v": None}
    safety_policy = SafetyPolicy.from_env()
    preflight_pipeline = build_pre_recording_pipeline(
        rig.umis[0],
        spool_dir=spool_dir,
        safety_policy=safety_policy,
        server_admission_probe=lambda estimated: server_admission(UPLOAD_URL, estimated),
    )

    def clock_sync_probe():
        cs = chrony_status("system-chrony")
        if cs is not None and handshake["v"] is not None:
            cs = replace(cs, handshake_offset_ns=handshake["v"])
        return cs

    sensors, monitors = build_sensors(rig, args, dev_dir)
    lidar_ifaces = {u.lidar.iface: u.lidar.lidar_ip for u in rig.umis}
    recorder = EpisodeRecorder(
        rig,
        config_path,
        spool_dir,
        sensors,
        ptp_probe=lambda: probe_ptp(lidar_ifaces),
        clock_sync_probe=clock_sync_probe,
        safety_policy=safety_policy,
    )

    display = make_display(rig, args, role)
    display.start()
    uploader = AsyncUploader(
        recorder.complete_dir,
        UPLOAD_URL,
        recorder.sent_dir,
        retry_seconds=args.retry_seconds,
        upload_timeout_seconds=args.upload_timeout_seconds,
    )
    uploader.retry_existing()

    buttons: Buttons | None = None
    leader: coordinator.Leader | None = None
    follower: coordinator.Follower | None = None
    if role == "follower":
        follower = coordinator.Follower(status=loaded_identity)
        follower.start()
        print(
            f"rig: {rig.n_umis} UMI(s) FOLLOWER role={manifest_role}; "
            f"awaiting leader on :{coordinator.DEFAULT_PORT}"
        )
    else:
        buttons = make_buttons(rig, args)
        peer = os.environ.get(PEER_ENV) or os.environ.get(LEGACY_PEER_ENV)
        if not peer:
            raise SystemExit(
                f"leader config requires ${PEER_ENV} (legacy ${LEGACY_PEER_ENV} is also accepted)"
            )
        leader = coordinator.Leader(peer)
        print(f"rig: {rig.n_umis} UMI, role leader; s/e/c/q or buttons")

    shutdown_requested = threading.Event()

    previous_sigterm = _install_sigterm_handler(shutdown_requested)
    try:
        if role == "follower":
            assert follower is not None
            follower_loop(
                recorder,
                uploader,
                monitors,
                display,
                spool_dir,
                follower,
                manifest_role,
                handshake,
                preflight_pipeline,
                runtime_status,
                shutdown_requested,
            )
        else:
            assert buttons is not None and leader is not None
            button_loop(
                recorder,
                uploader,
                monitors,
                display,
                spool_dir,
                buttons,
                leader,
                manifest_role,
                handshake,
                preflight_pipeline,
                runtime_status,
                shutdown_requested,
            )
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if recorder.active is not None:
            print("stopping active episode before exit")
            with contextlib.suppress(Exception):
                finish_active(recorder, uploader)
        with contextlib.suppress(Exception, KeyboardInterrupt):
            uploader.close(wait=args.wait_on_quit)
        if follower is not None:
            follower.close()
        for monitor in monitors.values():
            monitor.close()
        display.close()
        if buttons is not None:
            close = getattr(buttons, "close", None)
            if callable(close):
                close()
        runtime_status.update(state="stopped", session_id=None, episode_id=None)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    main(tyro.cli(Args))
