from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from umi_flow_core import ports
from umi_flow_core.lidar_clock import pcap_host_clock_mapping
from umi_flow_test_support.lidar import PcapWriter, synth_msop_packet


def _write_clock_pcap(
    path: Path,
    *,
    rate: float,
    host_step_s: float = 0.0,
) -> tuple[float, float]:
    lidar_origin = 1_700_000_000.0
    host_origin = 1_800_000_000.0
    writer = PcapWriter(path)
    try:
        for index in range(500):
            lidar_time = lidar_origin + index * 0.02
            host_time = host_origin + rate * (lidar_time - lidar_origin)
            if index >= 250:
                host_time += host_step_s
            writer.write_udp(
                round(host_time * 1e9),
                synth_msop_packet(unix_ns=round(lidar_time * 1e9)),
                src_ip=ports.ROBOSENSE_DEFAULT_LIDAR_IP,
                src_port=49_609,
                dst_ip=ports.ROBOSENSE_DEFAULT_HOST_IP,
                dst_port=ports.ROBOSENSE_MSOP_PORT,
            )
    finally:
        writer.close()
    return lidar_origin, host_origin


def test_affine_clock_mapping_recovers_offset_and_frequency_error(tmp_path: Path) -> None:
    path = tmp_path / "clock.pcap"
    lidar_origin, host_origin = _write_clock_pcap(path, rate=0.94)

    mapping = pcap_host_clock_mapping(path)

    assert mapping is not None
    assert mapping.rate == pytest.approx(0.94, abs=2e-6)
    lidar_times = lidar_origin + np.asarray([0.0, 3.0, 9.0])
    expected = host_origin + 0.94 * np.asarray([0.0, 3.0, 9.0])
    np.testing.assert_allclose(mapping.to_host(lidar_times), expected, atol=2e-5)
    assert mapping.residual_p99_s - mapping.residual_p01_s < 1e-4


def test_affine_clock_mapping_rejects_a_mid_episode_clock_step(tmp_path: Path) -> None:
    path = tmp_path / "clock-step.pcap"
    _write_clock_pcap(path, rate=1.0, host_step_s=0.2)

    with pytest.raises(ValueError, match="not affine"):
        pcap_host_clock_mapping(path)
