"""The Pi recorder session: rig config -> sensors -> button-driven episodes.

Wires the whole device stack from ONE rig config (``--config`` or
``$ROBOT_UMI_CONFIG``): per-UMI cameras (Rust V4L2), lidar pcap capture, Vive
recording, the advisory lidar monitor + LCD, GPIO/keyboard buttons, and the
resilient episode spool + uploader.

Modes (the enable mask): the MODE button cycles lidar -> vive -> lidar+vive while
idle; each episode arms the sensors its mode selects (cameras always). The episode
manifest records the mode actually used — the config's motion_source is only the
boot default.

Keyboard: s=start e=stop c=cancel q=quit m=mode (or tap/hold GPIO buttons).
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
from robot_umi_device.sensors import (
    CameraRecorder,
    LidarPcapRecorder,
    SensorRecorder,
    ViveRecorder,
)
from robot_umi_device.uploader import DEFAULT_UPLOAD_TIMEOUT_SECONDS, AsyncUploader
from robot_umi_device.vive import NetworkPoseSensor


SERVER_URL_ENV = "ROBOT_UMI_SERVER_URL"
SPOOL_DIR_ENV = "ROBOT_UMI_SPOOL_DIR"
# Coordination knobs also accepted from the .env (so the generic systemd unit needs no
# per-role edit — the provisioner just writes these into the Pi's .env).
ROLE_ENV = "ROBOT_UMI_ROLE"
PEER_ENV = "ROBOT_UMI_PEER"
LISTEN_PORT_ENV = "ROBOT_UMI_LISTEN_PORT"
TIME_REF_ENV = "ROBOT_UMI_TIME_REF"

# Follower watchdog: if a STOP is ever lost (leader crash / WiFi blip), auto-finish a take
# that has run this long so the follower never records forever.
MAX_TAKE_S = 1800.0


@dataclass
class Args:
    server_url: str | None = None  # upload endpoint; falls back to $ROBOT_UMI_SERVER_URL
    config: Path | None = None  # rig YAML; falls back to $ROBOT_UMI_CONFIG
    spool_dir: Path | None = None  # falls back to $ROBOT_UMI_SPOOL_DIR, else robot_umi_spool
    buttons: Literal["keyboard", "gpio"] = "keyboard"
    # "auto" = take the LCD kind + control pins from the rig config's controls.display;
    # "none"/"lcd" force it (dev off-Pi wants --display none). Button pins always come
    # from controls.buttons.
    display: Literal["auto", "none", "lcd"] = "auto"
    mock_lidar: bool = False  # synthetic pcap instead of tcpdump (dev/CI)
    vive_port: int = 9130
    retry_seconds: float = 5.0
    upload_timeout_seconds: float = DEFAULT_UPLOAD_TIMEOUT_SECONDS
    wait_on_quit: bool = False
    # Two-Pi bimanual coordination. solo = single-Pi (default, unchanged). leader has the
    # buttons and signals the follower (needs --peer). follower runs headless, driven by
    # the leader over the control socket. The manifest role is the rig's UMI name.
    role: Literal["solo", "leader", "follower"] = "solo"
    peer: str | None = None  # leader: follower host[:port]
    listen_port: int = coordinator.DEFAULT_PORT  # follower: control port
    require_peer: bool = True  # leader: abort a take if the follower is unreachable
    time_ref: str | None = None  # shared NTP ref recorded in the clock_sync manifest block


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


def resolve_server_url(args: Args) -> str:
    url = args.server_url or os.environ.get(SERVER_URL_ENV)
    if not url:
        raise SystemExit(f"no server URL: pass --server-url or set ${SERVER_URL_ENV}")
    return url


def apply_env_coordination(args: Args) -> None:
    """Fill coordination knobs from the .env when not given on the CLI, so the generic
    ``umi-recorder`` unit (which just runs ``record.py --buttons gpio``) can drive a
    leader/follower pair purely from the provisioned .env."""
    if args.role == "solo":
        env_role = os.environ.get(ROLE_ENV)
        if env_role in ("leader", "follower"):
            args.role = env_role  # type: ignore[assignment]
    if args.peer is None:
        args.peer = os.environ.get(PEER_ENV) or None
    if args.time_ref is None:
        args.time_ref = os.environ.get(TIME_REF_ENV) or None
    env_listen = os.environ.get(LISTEN_PORT_ENV)
    if env_listen:
        args.listen_port = int(env_listen)


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
) -> tuple[dict[str, SensorRecorder], NetworkPoseSensor | None, dict[str, LidarMonitor]]:
    """One recorder per configured sensor (all modes' union), a shared Vive
    receiver, and one advisory monitor per lidar."""
    sensors: dict[str, SensorRecorder] = {}
    monitors: dict[str, LidarMonitor] = {}
    receiver: NetworkPoseSensor | None = None

    if any(u.vive for u in rig.umis):
        receiver = NetworkPoseSensor(port=args.vive_port)
        receiver.start()

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

        if u.lidar is not None:
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

        if u.vive is not None and receiver is not None:
            vive_key = layout.sensor_key(u.name, layout.KIND_VIVE)
            sensors[vive_key] = ViveRecorder(vive_key, receiver, u.name)

    return sensors, receiver, monitors


def make_buttons(rig: core_config.RigConfig, args: Args) -> Buttons:
    if args.buttons == "gpio":
        b = rig.controls.buttons
        return GpioButtons(b.record, b.quit, b.mode)
    return KeyboardButtons()


def make_display(rig: core_config.RigConfig, args: Args) -> Display:
    kind = rig.controls.display.kind if args.display == "auto" else args.display
    if kind != "lcd":
        return NullDisplay()
    d = rig.controls.display
    # A follower's LCD is a lidar-health monitor only (the operator drives recording
    # from the leader), so hide the REC/IDLE box + episode count there.
    title = {"leader": "Leader", "follower": "Follower"}.get(args.role, "robot-umi")
    return StatusDisplay(driver=ST7789(dc=d.dc, rst=d.rst, bl=d.bl),
                         show_status=args.role != "follower", title=title)


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


def _leader_start(leader, session_id, mode, require_peer, handshake, display) -> bool:
    """Signal the follower to start; return True if we should start our own take.

    Unreachable follower: abort (require_peer) so we never record a useless half-pair,
    else proceed (the server orphan policy handles the missing mate). On success, stash
    the rough handshake offset for the clock_sync manifest block."""
    if leader is None:
        return True  # solo
    ack = leader.start_take(session_id, mode)
    if ack is None:
        display.notify_upload(UPLOAD_FAIL)
        print("follower unreachable" + ("; take aborted" if require_peer else "; recording solo half"))
        return not require_peer
    fu = ack.get("follower_unix_ns")
    if fu is not None:
        handshake["v"] = time.time_ns() - int(fu)
    return True


def button_loop(recorder, uploader, monitors, display, spool_dir, buttons, leader,
                role, handshake, mode, require_peer) -> None:
    """The leader/solo loop: buttons drive episodes; a leader also signals the follower."""
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
        if event is Event.MODE:
            if recording:
                print("mode locked while recording")
            else:
                mode = core_config.cycle_mode(mode)
                display.set_mode(mode)
                print(f"mode -> {mode}")
        elif event in (Event.TOGGLE, Event.START) and not recording:
            session_id = f"sess_{layout.unix_ns()}" if role is not None else None
            if not _leader_start(leader, session_id, mode, require_peer, handshake, display):
                continue
            try:
                active = recorder.start(mode, session_id=session_id, role=role)
                display.set_recording(True)
                print(f"recording {active.episode_id} (mode {mode}"
                      + (f", session {session_id})" if session_id else ")"))
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


def follower_loop(recorder, uploader, monitors, display, spool_dir, follower,
                  role, handshake, boot_mode) -> None:
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
                active = recorder.start(cmd.mode or boot_mode, session_id=cmd.session_id, role=role)
                state["started_at"] = time.monotonic()
                display.set_recording(True)
                print(f"follower recording {active.episode_id} (session {cmd.session_id})")
            except Exception as exc:
                print(f"follower failed to start: {exc}")
        elif cmd.cmd == coordinator.CMD_STOP and recorder.active is not None:
            finish_active(recorder, uploader)
            display.set_recording(False)
        elif cmd.cmd == coordinator.CMD_CANCEL and recorder.active is not None:
            recorder.cancel()
            display.set_recording(False)
            print("follower cancelled (episode discarded)")


def main(args: Args) -> None:
    apply_env_coordination(args)
    server_url = resolve_server_url(args)
    config_path = resolve_config_path(args)
    rig = core_config.load_rig_config(config_path)
    dev_dir = device_root()
    spool_dir = resolve_spool_dir(args.spool_dir)
    confirm_existing_spool(spool_dir)

    # The manifest role (the pairing key's arm) is the rig's single UMI name; solo = none.
    manifest_role = None if args.role == "solo" else rig.umis[0].name
    handshake: dict[str, int | None] = {"v": None}

    def clock_sync_probe():
        if args.role == "solo":
            return None
        cs = chrony_status(args.time_ref or "")
        if cs is not None and handshake["v"] is not None:
            cs = replace(cs, handshake_offset_ns=handshake["v"])
        return cs

    sensors, receiver, monitors = build_sensors(rig, args, dev_dir)
    lidar_ifaces = sorted({u.lidar.iface for u in rig.umis if u.lidar is not None})
    recorder = EpisodeRecorder(
        rig,
        config_path,
        spool_dir,
        sensors,
        ptp_probe=lambda: probe_ptp(lidar_ifaces),
        clock_sync_probe=clock_sync_probe,
    )

    display = make_display(rig, args)
    display.start()
    mode = rig.motion_source
    display.set_mode(mode)

    uploader = AsyncUploader(
        recorder.complete_dir,
        server_url,
        recorder.sent_dir,
        retry_seconds=args.retry_seconds,
        upload_timeout_seconds=args.upload_timeout_seconds,
    )
    uploader.retry_existing()

    buttons: Buttons | None = None
    leader: coordinator.Leader | None = None
    follower: coordinator.Follower | None = None
    if args.role == "follower":
        follower = coordinator.Follower(listen_port=args.listen_port)
        follower.start()
        print(f"rig: {rig.n_umis} UMI(s) FOLLOWER role={manifest_role}; awaiting leader on :{args.listen_port}")
    else:
        buttons = make_buttons(rig, args)
        if args.role == "leader":
            if not args.peer:
                raise SystemExit("--role leader requires --peer host[:port]")
            leader = coordinator.Leader(args.peer)
        print(f"rig: {rig.n_umis} UMI(s), boot mode {mode}, role {args.role}; s/e/c/q/m or buttons")

    try:
        if args.role == "follower":
            assert follower is not None
            follower_loop(recorder, uploader, monitors, display, spool_dir, follower,
                          manifest_role, handshake, mode)
        else:
            assert buttons is not None
            button_loop(recorder, uploader, monitors, display, spool_dir, buttons, leader,
                        manifest_role, handshake, mode, args.require_peer)
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
        if receiver is not None:
            receiver.stop()
        for monitor in monitors.values():
            monitor.close()
        display.close()
        if buttons is not None:
            close = getattr(buttons, "close", None)
            if callable(close):
                close()


if __name__ == "__main__":
    main(tyro.cli(Args))
