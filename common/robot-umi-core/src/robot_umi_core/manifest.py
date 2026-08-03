"""The versioned episode manifest (v2): written by the device, validated by the server.

v2 replaces the flat ``camera_*``/``lidar_*`` field namespace (which cannot hold
N UMIs x M sensors) with a nested ``sensors`` mapping keyed by sensor key
(``"umi1/camera"``), plus first-class ``ptp``, ``mode`` and ``config_sha256`` blocks.

The manifest is the per-episode authority: the requested mode, the realized stream
inventory, and eligibility live HERE, not in any live config file. Both sides import
this module — the schema exists exactly once.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from robot_umi_core import config as core_config
from robot_umi_core import layout

MANIFEST_VERSION = 2

PTP_MODE_HARDWARE = "hardware"
PTP_MODE_SOFTWARE = "software"
PTP_MODE_NONE = "none"  # no PTP running (e.g. vive-only rig, or bring-up)
PTP_MODES = (PTP_MODE_HARDWARE, PTP_MODE_SOFTWARE, PTP_MODE_NONE)

# Episodes whose PTP residual exceeds this are rejected for lidar SLAM processing
# (the de-skew/IMU alignment error would dominate). Advisory streams still process.
MAX_PTP_RESIDUAL_NS = 5_000_000  # 5 ms

# Merged bimanual episodes whose estimated cross-Pi clock offset exceeds this are
# rejected: at 30 fps a >5 ms skew shifts the follower arm past a half-frame, corrupting
# the resample/nearest-past alignment. Single-Pi episodes carry no clock_sync block and
# are never gated by this.
MAX_CROSS_PI_OFFSET_NS = 5_000_000  # 5 ms (< one 30 fps frame)


@dataclass(frozen=True)
class PtpStatus:
    """Per-episode PTP sync quality, recorded per lidar interface."""

    locked: bool
    timestamping_mode: str  # one of PTP_MODES
    residual_offset_ns: int | None = None  # last |offset| reported by ptp4l, if locked

    def to_dict(self) -> dict[str, Any]:
        return {
            "locked": self.locked,
            "timestamping_mode": self.timestamping_mode,
            "residual_offset_ns": self.residual_offset_ns,
        }


@dataclass(frozen=True)
class ClockSyncStatus:
    """Per-episode cross-Pi clock discipline quality (chrony to a common NTP reference).

    Only meaningful for coordinated multi-Pi (leader/follower) captures — single-Pi
    episodes omit the block entirely, so their manifests are unaffected. ``ref`` is the
    shared time source both Pis discipline to; ``offset_to_ref_ns`` is THIS Pi's last
    SIGNED offset to it (same sign convention on both Pis, so leader.offset -
    follower.offset gives the true signed cross-Pi skew). ``estimated_cross_pi_offset_ns``
    is set only on a MERGED bimanual manifest (computed by the pair-merge)."""

    synced: bool
    ref: str = ""  # the common time reference (NTP server ip/host); "" if unknown
    offset_to_ref_ns: int | None = None  # this Pi's last SIGNED offset to the shared ref
    rms_offset_ns: int | None = None  # chrony RMS offset, if reported
    handshake_offset_ns: int | None = None  # independent leader<->follower socket estimate
    estimated_cross_pi_offset_ns: int | None = None  # merged-only: leader - follower

    def to_dict(self) -> dict[str, Any]:
        return {
            "synced": self.synced,
            "ref": self.ref,
            "offset_to_ref_ns": self.offset_to_ref_ns,
            "rms_offset_ns": self.rms_offset_ns,
            "handshake_offset_ns": self.handshake_offset_ns,
            "estimated_cross_pi_offset_ns": self.estimated_cross_pi_offset_ns,
        }


@dataclass(frozen=True)
class SensorEntry:
    """One realized sensor stream in the episode."""

    kind: str  # layout.SENSOR_KINDS
    armed: bool  # was this sensor required by the episode's mode?
    files: dict[str, str]  # role -> filename (layout.sensor_files)
    stats: dict[str, Any] = field(default_factory=dict)  # kind-specific counters
    error: str | None = None  # non-None => the stream failed mid-episode

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "armed": self.armed,
            "files": self.files,
            "stats": self.stats,
            "error": self.error,
        }


def build_manifest(
    *,
    episode_id: str,
    device_id: str,
    mode: str,
    task: str,
    sensors: dict[str, SensorEntry],
    ptp: dict[str, PtpStatus],
    config_sha256: str,
    start_unix_ns: int,
    stop_unix_ns: int,
    files: dict[str, dict[str, Any]],
    session_id: str | None = None,
    role: str | None = None,
    clock_sync: ClockSyncStatus | None = None,
) -> dict[str, Any]:
    """Assemble the manifest dict. Eligibility: every ARMED sensor must have produced
    a stream without error. Unarmed/absent sensors never affect eligibility.

    ``session_id``/``role``/``clock_sync`` are the coordinated multi-Pi (leader/follower)
    fields — the pairing key, this Pi's arm, and its cross-Pi clock quality. They are
    OMITTED from the dict when ``None`` so a single-Pi manifest is byte-identical to
    before these fields existed."""
    if mode not in core_config.RECORD_MODES:
        raise ValueError(f"unknown mode {mode!r}")
    for key, entry in sensors.items():
        layout.sensor_key(*key.split("/"))  # validates shape + kind
        if entry.kind not in layout.SENSOR_KINDS:
            raise ValueError(f"sensor {key}: unknown kind {entry.kind!r}")
    for iface, status in ptp.items():
        if status.timestamping_mode not in PTP_MODES:
            raise ValueError(f"ptp[{iface}]: bad timestamping_mode")

    dataset_eligible = all(
        entry.error is None for entry in sensors.values() if entry.armed
    )
    m: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "episode_id": episode_id,
        "device_id": device_id,
        "mode": mode,
        "task": task,
        "sensors": {k: v.to_dict() for k, v in sensors.items()},
        "ptp": {iface: s.to_dict() for iface, s in ptp.items()},
        "config_sha256": config_sha256,
        "start_unix_ns": start_unix_ns,
        "stop_unix_ns": stop_unix_ns,
        "dataset_eligible": dataset_eligible,
        "files": files,
    }
    if session_id is not None:
        m["session_id"] = session_id
    if role is not None:
        m["role"] = role
    if clock_sync is not None:
        m["clock_sync"] = clock_sync.to_dict()
    return m


REQUIRED_TOP_LEVEL = (
    "manifest_version",
    "episode_id",
    "device_id",
    "mode",
    "sensors",
    "ptp",
    "config_sha256",
    "dataset_eligible",
    "files",
)


def validate_manifest(d: dict[str, Any]) -> list[str]:
    """Structural validation; returns a list of problems (empty = valid)."""
    problems = []
    for key in REQUIRED_TOP_LEVEL:
        if key not in d:
            problems.append(f"missing top-level key {key!r}")
    if d.get("manifest_version") != MANIFEST_VERSION:
        problems.append(
            f"manifest_version {d.get('manifest_version')!r} != {MANIFEST_VERSION}"
        )
    if d.get("mode") not in core_config.RECORD_MODES:
        problems.append(f"unknown mode {d.get('mode')!r}")
    sensors = d.get("sensors")
    if not isinstance(sensors, dict) or not sensors:
        problems.append("'sensors' must be a non-empty mapping")
    else:
        for key, entry in sensors.items():
            parts = key.split("/")
            if len(parts) != 2 or parts[1] not in layout.SENSOR_KINDS:
                problems.append(f"bad sensor key {key!r}")
            if not isinstance(entry, dict) or "files" not in entry:
                problems.append(f"sensor {key!r} entry malformed")
    return problems


def present_streams(d: dict[str, Any], kind: str) -> list[str]:
    """Sensor keys of the given kind that actually produced data (presence over mode).

    Server stages dispatch on THIS — never on the mode label."""
    out = []
    for key, entry in d.get("sensors", {}).items():
        if entry.get("kind") == kind and entry.get("error") is None:
            out.append(key)
    return sorted(out)


def lidar_ptp_acceptable(d: dict[str, Any], max_residual_ns: int = MAX_PTP_RESIDUAL_NS) -> bool:
    """The reject-unlocked-PTP rule, in one place. Applies only when lidar streams exist."""
    if not present_streams(d, layout.KIND_LIDAR):
        return True
    ptp = d.get("ptp", {})
    if not ptp:
        return False
    for status in ptp.values():
        if not status.get("locked"):
            return False
        residual = status.get("residual_offset_ns")
        if residual is not None and abs(residual) > max_residual_ns:
            return False
    return True


def clock_sync_acceptable(
    d: dict[str, Any], max_cross_pi_ns: int = MAX_CROSS_PI_OFFSET_NS
) -> bool:
    """The cross-Pi clock gate, in one place. A no-op for episodes without a
    ``clock_sync`` block (single-Pi captures always pass). When present, the shared
    clock must be synced and, if the merged cross-Pi offset estimate is known, within
    ``max_cross_pi_ns``."""
    cs = d.get("clock_sync")
    if not cs:
        return True
    if not cs.get("synced", False):
        return False
    est = cs.get("estimated_cross_pi_offset_ns")
    if est is not None and abs(est) > max_cross_pi_ns:
        return False
    return True


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def read_manifest(path: Path) -> dict[str, Any]:
    d = json.loads(path.read_text())
    problems = validate_manifest(d)
    if problems:
        raise ValueError(f"invalid manifest {path}: " + "; ".join(problems))
    return d
