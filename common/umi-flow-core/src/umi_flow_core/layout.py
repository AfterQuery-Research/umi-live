"""Episode directory layout: file naming, markers, hashing.

An episode is a hermetic directory — every byte the server needs travels inside it.
Sensor streams are named by *sensor key* (``"umi1/camera"``, ``"umi2/lidar"``, ...);
the key's ``/`` becomes ``_`` on disk.

Layout of one episode directory::

    episode_<UTC>_<host>_<ns9>/
      umi1_camera.mjpeg.avi        umi1_camera_frames.jsonl   umi1_camera_stats.json
      umi1_lidar.pcap              umi1_lidar_stats.json
      umi2_...                     (present after the server pairs both clients)
      config_snapshot/             resolved umi.yaml + referenced calib blobs
      events.jsonl                 start/stop/cancel with both clocks
      manifest.json                versioned manifest (see umi_flow_core.manifest)
      checksums.sha256             every data file, re-verified server-side
      COMPLETE                     sentinel written last
"""

from __future__ import annotations

import hashlib
import socket
import time
from pathlib import Path

# --- episode-level files -----------------------------------------------------------
MANIFEST_FILE = "manifest.json"
CHECKSUMS_FILE = "checksums.sha256"
COMPLETE_FILE = "COMPLETE"
EVENTS_FILE = "events.jsonl"
CONFIG_SNAPSHOT_DIR = "config_snapshot"
CONFIG_SNAPSHOT_YAML = "umi.yaml"  # inside CONFIG_SNAPSHOT_DIR

# Camera frame timestamps: V4L2 CLOCK_MONOTONIC at start of exposure, plus a unix_ns
# derived from a realtime-minus-monotonic offset captured at start (the Rust recorder).
CAMERA_SYNC_MODE = "v4l2_monotonic_start_of_exposure_v1"

# --- sensor kinds ------------------------------------------------------------------
KIND_CAMERA = "camera"
KIND_LIDAR = "lidar"
SENSOR_KINDS = (KIND_CAMERA, KIND_LIDAR)


def sensor_key(umi: str, kind: str) -> str:
    """Canonical sensor key, e.g. ``sensor_key("umi1", "lidar") == "umi1/lidar"``."""
    if kind not in SENSOR_KINDS:
        raise ValueError(f"unknown sensor kind {kind!r}, expected one of {SENSOR_KINDS}")
    return f"{umi}/{kind}"


def sensor_stem(key: str) -> str:
    """On-disk file stem for a sensor key (``"umi1/lidar"`` -> ``"umi1_lidar"``)."""
    return key.replace("/", "_")


def sensor_files(key: str) -> dict[str, str]:
    """The data files a sensor of this key writes, as {role: filename}.

    Roles are stable identifiers used by the manifest and the server stages.
    """
    umi, _, kind = key.partition("/")
    stem = sensor_stem(sensor_key(umi, kind))  # validates kind
    if kind == KIND_CAMERA:
        return {
            "video": f"{stem}.mjpeg.avi",
            "frames": f"{stem}_frames.jsonl",
            "stats": f"{stem}_stats.json",
        }
    if kind == KIND_LIDAR:
        return {
            "pcap": f"{stem}.pcap",
            "stats": f"{stem}_stats.json",
        }
    raise ValueError(f"unknown sensor kind {kind!r}")


# --- clocks + ids ------------------------------------------------------------------


def monotonic_ns() -> int:
    return time.monotonic_ns()


def unix_ns() -> int:
    return time.time_ns()


def episode_id(now_ns: int | None = None, hostname: str | None = None) -> str:
    now_ns = unix_ns() if now_ns is None else now_ns
    hostname = socket.gethostname() if hostname is None else hostname
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime(now_ns / 1_000_000_000))
    return f"episode_{stamp}_{hostname}_{now_ns % 1_000_000_000:09d}"


# --- hashing -----------------------------------------------------------------------


def sha256_file(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()
