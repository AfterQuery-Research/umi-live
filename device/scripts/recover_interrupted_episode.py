#!/usr/bin/env python3
"""Safely finalize an interrupted, preserved Pi episode as dataset-ineligible.

This is an evidence-preserving recovery path for a recorder process that was
terminated after capture files were closed but before it could write a manifest,
checksums, and ``COMPLETE``.  It never claims that the interrupted capture was
healthy: every recovered sensor is marked with the supplied interruption reason,
PTP/clock evidence is explicitly unknown, and the resulting pair is therefore
terminally dataset-ineligible after upload.
"""

from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path
from typing import Any

from umi_flow_core import config as core_config
from umi_flow_core import layout, manifest
from umi_flow_device.episode import write_checksums
from umi_flow_device.operation_lock import OperationLock
from umi_flow_device.runtime_status import STATUS_FILE


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _capture_processes(spool_dir: Path, proc_root: Path = Path("/proc")) -> list[int]:
    needle = str(spool_dir.resolve()).encode()
    active: list[int] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if needle in cmdline and (b"v4l2" in cmdline or b"tcpdump" in cmdline):
            active.append(int(entry.name))
    return active


def _start_unix_ns(events_path: Path) -> int:
    for line in events_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("event") == "start" and type(event.get("time_unix_ns")) is int:
            return event["time_unix_ns"]
    raise ValueError(f"no valid start event in {events_path}")


def _required_capture_paths(episode_dir: Path, role: str) -> dict[str, Path]:
    camera = layout.sensor_files(layout.sensor_key(role, layout.KIND_CAMERA))
    lidar = layout.sensor_files(layout.sensor_key(role, layout.KIND_LIDAR))
    return {
        "camera_video": episode_dir / camera["video"],
        "camera_frames": episode_dir / camera["frames"],
        "camera_stats": episode_dir / camera["stats"],
        "lidar_pcap": episode_dir / lidar["pcap"],
    }


def _camera_stats_for_recovery(path: Path, frames_path: Path) -> tuple[dict[str, Any], str]:
    """Load camera stats when present, or synthesize conservative interrupted stats.

    The camera recorder writes its stats only during an orderly stop. Abrupt process
    loss can therefore leave valid video and frame metadata without this optional
    summary. Recovery must preserve that evidence without ever claiming it is healthy.
    """

    try:
        return _json(path), "captured"
    except FileNotFoundError:
        source = "missing"
    except (OSError, ValueError):
        source = "unreadable"
    try:
        frame_count = sum(
            1 for line in frames_path.read_text(encoding="utf-8").splitlines() if line
        )
    except OSError:
        frame_count = None
    return {
        "camera_dataset_eligible": False,
        "camera_frame_count": frame_count,
        "camera_stats_recovered": True,
    }, source


def recover(
    spool_dir: Path,
    episode_dir: Path,
    *,
    role: str,
    session_id: str,
    reason: str,
    apply: bool,
    proc_root: Path = Path("/proc"),
) -> dict[str, Any]:
    spool_dir = spool_dir.resolve()
    episode_dir = episode_dir.resolve()
    in_progress = (spool_dir / "in_progress").resolve()
    if episode_dir.parent != in_progress or not episode_dir.name.startswith("episode_"):
        raise ValueError("episode must be a direct episode_* child of spool/in_progress")
    if not session_id.startswith("sess_"):
        raise ValueError("session_id must start with 'sess_'")
    if not reason.strip():
        raise ValueError("a non-empty interruption reason is required")

    status = _json(spool_dir / STATUS_FILE)
    if status.get("state") != "idle" or status.get("session_id") is not None:
        raise RuntimeError(f"recorder is not idle: {status.get('state')!r}")
    capture_pids = _capture_processes(spool_dir, proc_root)
    if capture_pids:
        raise RuntimeError(f"capture processes are still active: pids={capture_pids}")
    for forbidden in (layout.MANIFEST_FILE, layout.CHECKSUMS_FILE, layout.COMPLETE_FILE):
        if (episode_dir / forbidden).exists():
            raise RuntimeError(
                f"refusing already-finalized or partially-finalized episode: {forbidden}"
            )

    config_path = episode_dir / layout.CONFIG_SNAPSHOT_DIR / layout.CONFIG_SNAPSHOT_YAML
    rig = core_config.load_rig_config(config_path)
    if len(rig.umis) != 1 or rig.umis[0].name != role:
        raise ValueError(f"snapshot role does not match requested role {role!r}")
    if f"_{role}_" not in episode_dir.name:
        raise ValueError(f"episode id does not match requested role {role!r}")

    required = _required_capture_paths(episode_dir, role)
    required_payloads = {name: path for name, path in required.items() if name != "camera_stats"}
    missing = [
        name
        for name, path in required_payloads.items()
        if not path.is_file() or path.stat().st_size == 0
    ]
    if missing:
        raise RuntimeError(
            f"required interrupted capture files are missing/empty: {', '.join(missing)}"
        )
    camera_stats, camera_stats_source = _camera_stats_for_recovery(
        required["camera_stats"], required["camera_frames"]
    )
    start_ns = _start_unix_ns(episode_dir / layout.EVENTS_FILE)
    stop_ns = max(path.stat().st_mtime_ns for path in required_payloads.values())
    if stop_ns < start_ns:
        raise RuntimeError("capture file timestamps precede the start event")

    final_dir = spool_dir / "complete" / episode_dir.name
    if final_dir.exists():
        raise FileExistsError(final_dir)
    result = {
        "state": "planned" if not apply else "complete",
        "episode_id": episode_dir.name,
        "role": role,
        "session_id": session_id,
        "source": str(episode_dir),
        "destination": str(final_dir),
        "dataset_eligible": False,
        "reason": reason,
        "camera_stats_source": camera_stats_source,
    }
    if not apply:
        return result

    with OperationLock.for_spool(spool_dir):
        # Recheck the live safety facts after acquiring the same lock capture uses.
        status = _json(spool_dir / STATUS_FILE)
        if status.get("state") != "idle" or status.get("session_id") is not None:
            raise RuntimeError("recorder left idle state while recovery was starting")
        capture_pids = _capture_processes(spool_dir, proc_root)
        if capture_pids:
            raise RuntimeError(f"capture processes appeared during recovery: pids={capture_pids}")

        recovered_at = time.time_ns()
        with (episode_dir / layout.EVENTS_FILE).open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "event": "recover_interrupted",
                        "time_unix_ns": recovered_at,
                        "reason": reason,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )

        lidar_stats_path = (
            episode_dir / layout.sensor_files(layout.sensor_key(role, layout.KIND_LIDAR))["stats"]
        )
        lidar_stats_path.write_text(
            json.dumps(
                {
                    "packets_captured": None,
                    "packets_dropped_kernel": None,
                    "recovered_after_interruption": True,
                    "recovery_reason": reason,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        camera_stats = {
            **camera_stats,
            "camera_dataset_eligible": False,
            "camera_stats_source": camera_stats_source,
            "recovered_after_interruption": True,
            "recovery_reason": reason,
        }
        required["camera_stats"].write_text(
            json.dumps(camera_stats, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        interruption_error = f"interrupted recorder recovery: {reason}"
        sensors = {
            layout.sensor_key(role, layout.KIND_CAMERA): manifest.SensorEntry(
                kind=layout.KIND_CAMERA,
                armed=True,
                files=layout.sensor_files(layout.sensor_key(role, layout.KIND_CAMERA)),
                stats=camera_stats,
                error=interruption_error,
            ),
            layout.sensor_key(role, layout.KIND_LIDAR): manifest.SensorEntry(
                kind=layout.KIND_LIDAR,
                armed=True,
                files=layout.sensor_files(layout.sensor_key(role, layout.KIND_LIDAR)),
                stats=_json(lidar_stats_path),
                error=interruption_error,
            ),
        }
        iface = rig.umis[0].lidar.iface
        value = manifest.build_manifest(
            episode_id=episode_dir.name,
            device_id=socket.gethostname(),
            task=rig.task,
            sensors=sensors,
            ptp={iface: manifest.PtpStatus(locked=False, timestamping_mode="none")},
            config_sha256=layout.sha256_file(config_path),
            start_unix_ns=start_ns,
            stop_unix_ns=stop_ns,
            files={},
            session_id=session_id,
            role=role,
            clock_sync=manifest.ClockSyncStatus(synced=False, ref=""),
        )
        value["recovery"] = {
            "kind": "interrupted_recorder",
            "reason": reason,
            "recovered_at_unix_ns": recovered_at,
            "original_capture_preserved": True,
        }
        manifest.write_manifest(episode_dir / layout.MANIFEST_FILE, value)
        hashes = write_checksums(episode_dir)
        value["files"] = {
            name: {"sha256": digest, "bytes": (episode_dir / name).stat().st_size}
            for name, digest in sorted(hashes.items())
        }
        manifest.write_manifest(episode_dir / layout.MANIFEST_FILE, value)
        (episode_dir / layout.COMPLETE_FILE).write_text("complete\n", encoding="utf-8")
        final_dir.parent.mkdir(parents=True, exist_ok=True)
        episode_dir.rename(final_dir)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spool-dir", type=Path, required=True)
    parser.add_argument("--episode-dir", type=Path, required=True)
    parser.add_argument("--role", choices=("umi1", "umi2"), required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            recover(
                args.spool_dir,
                args.episode_dir,
                role=args.role,
                session_id=args.session_id,
                reason=args.reason,
                apply=args.apply,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
