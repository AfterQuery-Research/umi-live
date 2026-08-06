"""The Pi recorder session: rig config -> sensors -> button-driven episodes.

Each client records one camera and one RoboSense Airy lidar. The ``umi1`` config is
the button-driven leader; ``umi2`` is the headless follower. Solo recording and
runtime sensor-mode selection are intentionally unsupported.

Keyboard: s=start e=stop c=cancel q=quit (or tap/hold GPIO buttons).
"""

from __future__ import annotations

import contextlib
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import tyro
from robot_umi_core import config as core_config
from robot_umi_core import layout
from robot_umi_core.ports import bpf_filter
from robot_umi_device import coordinator
from robot_umi_device.buttons import Buttons, Event, GpioButtons, KeyboardButtons
from robot_umi_device.camera import RustV4L2Camera
from robot_umi_device.display import (
    ST7789,
    UPLOAD_FAIL,
    UPLOAD_OK,
    Display,
    NullDisplay,
    StatusDisplay,
)
from robot_umi_device.episode import EpisodeRecorder
from robot_umi_device.lidar_monitor import LidarMonitor
from robot_umi_device.mock_lidar import MockLidarPcapRecorder
from robot_umi_device.ptp import chrony_status, probe_ptp
from robot_umi_device.sensors import CameraRecorder, LidarPcapRecorder, SensorRecorder
from robot_umi_device.uploader import DEFAULT_UPLOAD_TIMEOUT_SECONDS, AsyncUploader

SPOOL_DIR_ENV = "ROBOT_UMI_SPOOL_DIR"
PEER_ENV = "ROBOT_UMI_PEER"
UPLOAD_URL = "http://127.0.0.1:8080/upload"

# Follower watchdog: if a STOP is ever lost (leader crash / WiFi blip), auto-finish a take
# that has run this long so the follower never records forever.
MAX_TAKE_S = 1800.0


@dataclass
class Args:
    config: Path | None = None  # rig YAML; falls back to $ROBOT_UMI_CONFIG
    spool_dir: Path | None = None  # falls back to $ROBOT_UMI_SPOOL_DIR, else robot_umi_spool
    buttons: Literal["keyboard", "gpio"] = "keyboard"
    # "auto" = take the LCD kind + control pins from the rig config's controls.display;
    # "none"/"lcd" force it (dev off-Pi wants --display none). Button pins always come
    # from controls.buttons.
    display: Literal["auto", "none", "lcd"] = "auto"
    mock_lidar: bool = False  # synthetic pcap instead of tcpdump (dev/CI)
    retry_seconds: float = 5.0
    upload_timeout_seconds: float = DEFAULT_UPLOAD_TIMEOUT_SECONDS
    wait_on_quit: bool = False


def device_root() -> Path:
    """The device package dir (contains v4l2_recorder/) — NOT the repo root."""
    return Path(__file__).resolve().parents[1]


def resolve_config_path(args: Args) -> Path:
    if args.config is not None:
        return args.config.expanduser().resolve()
    env = os.environ.get(core_config.CONFIG_ENV_VAR)
    if env:
        return Path(env).expanduser().resolve()
    raise SystemExit(
        f"no rig config: pass --config or set ${core_config.CONFIG_ENV_VAR}"
    )


def resolve_spool_dir(path: Path | None) -> Path:
    if path is None:
        env = os.environ.get(SPOOL_DIR_ENV)
        path = Path(env) if env else Path("robot_umi_spool")
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else device_root() / expanded


def spool_has_data(path: Path) -> bool:
    if not path.exists():
        return False
    scaffolding = {"in_progress", "complete", "sent"}
    for child in path.iterdir():
        if child.name == "staging":
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
        if args.mock_lidar:
            sensors[lidar_key] = MockLidarPcapRecorder(lidar_key)
        else:
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
    return StatusDisplay(driver=ST7789(dc=d.dc, rst=d.rst, bl=d.bl),
                         show_status=role != "follower", title=title)


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


def button_loop(
    recorder, uploader, monitors, display, spool_dir, buttons, leader, role, handshake
) -> None:
    """The leader buttons drive synchronized two-client episodes."""
    state = {"last_monitor": 0.0, "last_result_count": 0}
    while True:
        event = buttons.poll()
        _refresh_status(display, monitors, spool_dir, uploader, state)

        if recorder.active is not None:
            error = recorder.poll_active_error()
            if error is not None:
                print(f"sensor failed mid-episode ({error}); stopping + keeping data")
                if leader is not None:
                    leader.stop_take()
                finish_active(recorder, uploader)
                display.set_recording(False)

        if event is None:
            time.sleep(0.02)
            continue

        recording = recorder.active is not None
        if event in (Event.TOGGLE, Event.START) and not recording:
            session_id = f"sess_{layout.unix_ns()}"
            if not _leader_start(leader, session_id, handshake, display):
                continue
            try:
                active = recorder.start(session_id=session_id, role=role)
                display.set_recording(True)
                print(f"recording {active.episode_id} (session {session_id})")
            except Exception as exc:
                print(f"failed to start: {exc}")
                if leader is not None:
                    leader.cancel_take()  # keep the follower from recording alone
        elif event in (Event.TOGGLE, Event.STOP) and recording:
            if leader is not None:
                leader.stop_take()
            finish_active(recorder, uploader)
            display.set_recording(False)
        elif event is Event.CANCEL and recording:
            if leader is not None:
                leader.cancel_take()
            recorder.cancel()
            display.set_recording(False)
            print("cancelled (episode discarded)")
        elif event is Event.QUIT:
            break


def follower_loop(
    recorder, uploader, monitors, display, spool_dir, follower, role, handshake
) -> None:
    """The headless follower loop: leader commands (not buttons) drive episodes."""
    state = {"last_monitor": 0.0, "last_result_count": 0, "started_at": 0.0}
    while True:
        cmd = follower.poll()
        _refresh_status(display, monitors, spool_dir, uploader, state)

        if recorder.active is not None:
            error = recorder.poll_active_error()
            if error is not None:
                print(f"sensor failed mid-episode ({error}); stopping + keeping data")
                finish_active(recorder, uploader)
                display.set_recording(False)
            elif time.monotonic() - state["started_at"] > MAX_TAKE_S:
                print("follower watchdog: max take duration exceeded; stopping")
                finish_active(recorder, uploader)
                display.set_recording(False)

        if cmd is None:
            time.sleep(0.02)
            continue

        if cmd.cmd == coordinator.CMD_START and recorder.active is None:
            if cmd.leader_unix_ns is not None:
                handshake["v"] = time.time_ns() - int(cmd.leader_unix_ns)
            try:
                active = recorder.start(session_id=cmd.session_id or "", role=role)
                state["started_at"] = time.monotonic()
                display.set_recording(True)
                print(f"follower recording {active.episode_id} (session {cmd.session_id})")
            except Exception as exc:
                print(f"follower failed to start: {exc}")
                cmd.respond(False, str(exc))
            else:
                cmd.respond(True)
        elif cmd.cmd == coordinator.CMD_STOP and recorder.active is not None:
            try:
                finish_active(recorder, uploader)
            except Exception as exc:
                cmd.respond(False, str(exc))
                print(f"follower failed to stop: {exc}")
            else:
                display.set_recording(False)
                cmd.respond(True)
        elif cmd.cmd == coordinator.CMD_CANCEL and recorder.active is not None:
            try:
                recorder.cancel()
            except Exception as exc:
                cmd.respond(False, str(exc))
                print(f"follower failed to cancel: {exc}")
            else:
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
    handshake: dict[str, int | None] = {"v": None}

    def clock_sync_probe():
        cs = chrony_status("system-chrony")
        if cs is not None and handshake["v"] is not None:
            cs = replace(cs, handshake_offset_ns=handshake["v"])
        return cs

    sensors, monitors = build_sensors(rig, args, dev_dir)
    lidar_ifaces = sorted({u.lidar.iface for u in rig.umis})
    recorder = EpisodeRecorder(
        rig,
        config_path,
        spool_dir,
        sensors,
        ptp_probe=lambda: probe_ptp(lidar_ifaces),
        clock_sync_probe=clock_sync_probe,
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
        follower = coordinator.Follower()
        follower.start()
        print(
            f"rig: {rig.n_umis} UMI(s) FOLLOWER role={manifest_role}; "
            f"awaiting leader on :{coordinator.DEFAULT_PORT}"
        )
    else:
        buttons = make_buttons(rig, args)
        peer = os.environ.get(PEER_ENV)
        if not peer:
            raise SystemExit(f"leader config requires ${PEER_ENV}")
        leader = coordinator.Leader(peer)
        print(f"rig: {rig.n_umis} UMI, role leader; s/e/c/q or buttons")

    try:
        if role == "follower":
            assert follower is not None
            follower_loop(recorder, uploader, monitors, display, spool_dir, follower,
                          manifest_role, handshake)
        else:
            assert buttons is not None and leader is not None
            button_loop(recorder, uploader, monitors, display, spool_dir, buttons, leader,
                        manifest_role, handshake)
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


if __name__ == "__main__":
    main(tyro.cli(Args))
