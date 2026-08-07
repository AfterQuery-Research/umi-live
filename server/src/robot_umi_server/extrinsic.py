"""Resolve each lidar's IMU->lidar extrinsic (``T_lidar_imu``) for the GLIM stage.

Precedence, highest first:
  1. an explicit per-UMI ``lidar.t_lidar_imu`` override in the episode's config snapshot,
  2. the lidar's OWN factory value, parsed from its DIFOP packet in this episode's pcap,
  3. a hardcoded default (a known-good unit's value).

Both engines inject the result into the GLIM container via the ``T_LIDAR_IMU`` env var,
which the entrypoint splices into ``config_sensors.json`` before running GLIM. Resolving
from the EPISODE pcap (not the decoded bag) keeps the GPU and Mac paths identical, and
makes each of a bimanual rig's two lidars get its own calibration automatically.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from robot_umi_core import config as core_config
from robot_umi_core import difop, layout

ENV_VAR = "T_LIDAR_IMU"


def resolve_t_lidar_imu(episode_dir: Path, scope: str) -> tuple[tuple[float, ...], str]:
    """``(extrinsic, source)`` for UMI ``scope``; source is one of
    ``config_override`` / ``difop`` / ``hardcoded_default``."""
    # 1. explicit per-UMI override in the snapshot config.
    try:
        rig = core_config.load_rig_config(
            episode_dir / layout.CONFIG_SNAPSHOT_DIR / layout.CONFIG_SNAPSHOT_YAML
        )
        umi = next((u for u in rig.umis if u.name == scope), None)
        if umi is not None and umi.lidar is not None and umi.lidar.t_lidar_imu is not None:
            return tuple(umi.lidar.t_lidar_imu), "config_override"
    except (ValueError, OSError, yaml.YAMLError):
        pass  # unreadable/corrupt snapshot -> fall through to DIFOP/default

    # 2. this lidar's own DIFOP.
    pcap = episode_dir / layout.sensor_files(layout.sensor_key(scope, layout.KIND_LIDAR))["pcap"]
    if pcap.is_file():
        ext = difop.extract_t_lidar_imu(pcap)
        if ext is not None:
            return ext, "difop"

    # 3. hardcoded fallback.
    return difop.DEFAULT_AIRY_T_LIDAR_IMU, "hardcoded_default"


def t_lidar_imu_env(episode_dir: Path, scope: str) -> dict[str, str]:
    """The GLIM container's env: ``{T_LIDAR_IMU: "x y z qx qy qz qw"}``. Logs the source."""
    ext, source = resolve_t_lidar_imu(episode_dir, scope)
    value = " ".join(repr(float(v)) for v in ext)
    note = " (no DIFOP in pcap)" if source == "hardcoded_default" else ""
    print(
        f"[{episode_dir.name}] {scope} T_lidar_imu source={source}{note} -> [{value}]",
        flush=True,
    )
    return {ENV_VAR: value}
