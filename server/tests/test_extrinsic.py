"""Per-lidar T_lidar_imu resolution: DIFOP extraction from a pcap + precedence."""

from __future__ import annotations

import numpy as np
import yaml
from umi_flow_core import difop, layout
from umi_flow_server import extrinsic
from umi_flow_test_support import PcapWriter

OTHER = (0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0)  # a distinct, identity-rotation extrinsic


def _pcap_name(scope: str = "umi1") -> str:
    return layout.sensor_files(layout.sensor_key(scope, layout.KIND_LIDAR))["pcap"]


def _write_pcap(path, *, difop_ext=None, msop=False):
    w = PcapWriter(path)
    if difop_ext is not None:
        w.write_udp(
            1,
            difop.synth_difop_packet(difop_ext),
            src_ip="192.168.1.200",
            src_port=7788,
            dst_ip="192.168.1.102",
            dst_port=7788,  # ROBOSENSE_DIFOP_PORT
        )
    if msop:
        w.write_udp(
            2,
            b"\x00" * 32,
            src_ip="192.168.1.200",
            src_port=6699,
            dst_ip="192.168.1.102",
            dst_port=6699,
        )
    w.close()


def _write_snapshot(episode_dir, *, override=None):
    snap = episode_dir / layout.CONFIG_SNAPSHOT_DIR
    snap.mkdir(parents=True, exist_ok=True)
    lidar = {"iface": "eth0"}
    if override is not None:
        lidar["t_lidar_imu"] = list(override)
    cfg = {
        "version": 1,
        "umis": {
            "umi1": {
                "camera": {"device": "d", "width": 800, "height": 600, "fps": 30, "calib": "c.npz"},
                "lidar": lidar,
            }
        },
    }
    (snap / layout.CONFIG_SNAPSHOT_YAML).write_text(yaml.safe_dump(cfg))


def test_extract_from_pcap(tmp_path):
    p = tmp_path / _pcap_name()
    _write_pcap(p, difop_ext=difop.DEFAULT_AIRY_T_LIDAR_IMU)
    actual = difop.extract_t_lidar_imu(p)
    assert actual is not None
    assert np.allclose(actual, difop.DEFAULT_AIRY_T_LIDAR_IMU, atol=1e-5)


def test_extract_none_without_difop(tmp_path):
    p = tmp_path / _pcap_name()
    _write_pcap(p, msop=True)  # MSOP only, no DIFOP
    assert difop.extract_t_lidar_imu(p) is None


def test_resolve_config_override_wins(tmp_path):
    _write_snapshot(tmp_path, override=OTHER)
    _write_pcap(tmp_path / _pcap_name(), difop_ext=difop.DEFAULT_AIRY_T_LIDAR_IMU)
    ext, source = extrinsic.resolve_t_lidar_imu(tmp_path, "umi1")
    assert source == "config_override"
    assert np.allclose(ext, OTHER)


def test_resolve_difop_when_no_override(tmp_path):
    _write_snapshot(tmp_path, override=None)
    _write_pcap(tmp_path / _pcap_name(), difop_ext=OTHER)
    ext, source = extrinsic.resolve_t_lidar_imu(tmp_path, "umi1")
    assert source == "difop"
    assert np.allclose(ext, OTHER, atol=1e-5)


def test_resolve_default_when_no_difop(tmp_path):
    _write_snapshot(tmp_path, override=None)
    _write_pcap(tmp_path / _pcap_name(), msop=True)
    ext, source = extrinsic.resolve_t_lidar_imu(tmp_path, "umi1")
    assert source == "hardcoded_default"
    assert np.allclose(ext, difop.DEFAULT_AIRY_T_LIDAR_IMU)


def test_extract_none_on_corrupt_pcap(tmp_path):
    # A present-but-garbage pcap (bad magic) must degrade to None, not raise, so it can
    # never block the SLAM stage.
    p = tmp_path / _pcap_name()
    p.write_bytes(b"this is not a pcap file, just garbage" * 4)
    assert difop.extract_t_lidar_imu(p) is None


def test_resolve_falls_back_on_corrupt_snapshot(tmp_path):
    # A syntactically-broken snapshot yaml (yaml.YAMLError, not ValueError/OSError) must
    # fall through step 1 to the DIFOP value, not raise.
    snap = tmp_path / layout.CONFIG_SNAPSHOT_DIR
    snap.mkdir(parents=True)
    (snap / layout.CONFIG_SNAPSHOT_YAML).write_text("[unclosed list")
    _write_pcap(tmp_path / _pcap_name(), difop_ext=OTHER)
    ext, source = extrinsic.resolve_t_lidar_imu(tmp_path, "umi1")
    assert source == "difop"
    assert np.allclose(ext, OTHER, atol=1e-5)


def test_env_format(tmp_path):
    _write_snapshot(tmp_path, override=OTHER)
    _write_pcap(tmp_path / _pcap_name(), msop=True)
    env = extrinsic.t_lidar_imu_env(tmp_path, "umi1")
    assert extrinsic.ENV_VAR in env
    parts = env[extrinsic.ENV_VAR].split()
    assert len(parts) == 7
    assert np.allclose([float(x) for x in parts], OTHER)
