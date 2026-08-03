"""Mac SLAM stage: replay an episode's lidar pcap through the arm64 CPU GLIM
container and export ``trajectory.tum`` — the GPU-less equivalent of the amd64
``docker/slam-glim`` stage, so a Mac can process episodes without a GPU box.

The container (``mac/Dockerfile`` -> ``umi-glim-live-arm64``, its ``entrypoint.sh
replay`` mode) does the heavy lifting: rslidar_sdk pcap replay -> rosbag ->
glim_rosbag (auto_quit) -> export_tum. This module is the in-process StageFn the
runner invokes per lidar UMI: it locates that UMI's pcap in the episode, runs the
container over it, and leaves ``trajectory.tum`` in the runner's staging dir (which
the runner promotes to ``derived/<id>/<umi>/glim/``).

Selected via ``runner.py --engine mac`` (which uses ``runner.mac_stages``). The
stage is still named ``glim`` so the dataset stage + readiness gate are unchanged.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

from robot_umi_core import config as core_config
from robot_umi_core import layout

from robot_umi_server import extrinsic

DEFAULT_IMAGE = "umi-glim-live-arm64"

StageFn = Callable[[Path, Path, str], None]


def _alignment_enabled(episode_dir: Path) -> bool:
    """Whether this episode's rig config opts into cross-lidar alignment. When True the
    GLIM container also exports its map (map.ply + submaps/) for the align stage; when
    False (the default) the container behaves exactly as the non-joint pipeline."""
    try:
        rig = core_config.load_rig_config(
            episode_dir / layout.CONFIG_SNAPSHOT_DIR / layout.CONFIG_SNAPSHOT_YAML
        )
    except (ValueError, OSError):
        return False
    return rig.alignment.enabled


def make_glim_fn(image: str = DEFAULT_IMAGE) -> StageFn:
    """Return a StageFn that SLAMs one lidar UMI's pcap with the arm64 GLIM image."""

    def run(episode_dir: Path, out_dir: Path, scope: str) -> None:
        if not scope:
            raise ValueError("mac glim stage requires a per-UMI scope")
        pcap_name = layout.sensor_files(layout.sensor_key(scope, layout.KIND_LIDAR))["pcap"]
        pcap = episode_dir / pcap_name
        if not pcap.is_file():
            raise FileNotFoundError(f"no lidar pcap for {scope}: {pcap}")

        episode_dir = episode_dir.resolve()
        out_dir = out_dir.resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        # This lidar's own extrinsic (config override > its DIFOP > default); the entrypoint
        # splices T_LIDAR_IMU into config_sensors.json so each arm gets its own calibration.
        env = extrinsic.t_lidar_imu_env(episode_dir, scope)
        # Only request the map export when the rig opts into alignment — otherwise the
        # container does exactly what the non-joint pipeline does (trajectory.tum only).
        export_map = _alignment_enabled(episode_dir)
        env_flags = ["-e", f"{extrinsic.ENV_VAR}={env[extrinsic.ENV_VAR]}"]
        if export_map:
            env_flags += ["-e", "EXPORT_MAP=1"]
        # The episode is mounted read-only; the staging out_dir is writable. Paths inside
        # the container are fixed (/ep, /out) so the entrypoint stays host-agnostic.
        cmd = [
            "docker", "run", "--rm",
            *env_flags,
            "-v", f"{episode_dir}:/ep:ro",
            "-v", f"{out_dir}:/out",
            image, "replay", f"/ep/{pcap_name}", "/out",
        ]
        subprocess.run(cmd, check=True)

        traj = out_dir / "trajectory.tum"
        if not traj.is_file() or traj.stat().st_size == 0:
            raise RuntimeError(f"mac glim produced no trajectory.tum in {out_dir}")
        # When alignment is on, the align stage needs the exported map artifacts.
        if export_map and not (out_dir / "map.ply").is_file():
            raise RuntimeError(f"mac glim (EXPORT_MAP) produced no map.ply in {out_dir}")
        if export_map and not (out_dir / "submaps" / "poses.txt").is_file():
            raise RuntimeError(f"mac glim (EXPORT_MAP) produced no submaps/poses.txt in {out_dir}")

    return run
