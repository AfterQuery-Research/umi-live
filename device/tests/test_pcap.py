"""pcap writer/reader roundtrip + synthetic packet shapes."""

from __future__ import annotations

from robot_umi_core import ports

from robot_umi_device.lidar_monitor import (
    parse_robosense_temp_c,
    synth_imu_packet,
    synth_msop_packet,
)
from robot_umi_device.pcap import PcapWriter, read_udp_packets


def test_pcap_roundtrip(tmp_path):
    path = tmp_path / "x.pcap"
    w = PcapWriter(path)
    t0 = 1_752_600_000_123_456_789
    w.write_udp(
        t0,
        b"hello",
        src_ip="192.168.1.200",
        src_port=49610,
        dst_ip="192.168.1.102",
        dst_port=6699,
    )
    w.write_udp(
        t0 + 450_000,
        b"world!!",
        src_ip="192.168.1.200",
        src_port=49607,
        dst_ip="192.168.1.102",
        dst_port=6688,
    )
    w.close()

    pkts = list(read_udp_packets(path))
    assert len(pkts) == 2
    assert pkts[0].payload == b"hello"
    assert pkts[0].ts_unix_ns == t0  # nanosecond precision preserved
    assert pkts[0].src_ip == "192.168.1.200"
    assert pkts[0].dst_port == 6699
    assert pkts[1].dst_port == 6688
    assert pkts[1].ts_unix_ns - pkts[0].ts_unix_ns == 450_000


def test_synth_msop_packet_shape():
    pkt = synth_msop_packet(temp_c=41.5)
    assert len(pkt) == ports.ROBOSENSE_MSOP_LEN
    assert pkt[:4] == ports.ROBOSENSE_MSOP_MAGIC
    assert abs(parse_robosense_temp_c(pkt) - 41.5) < ports.ROBOSENSE_TEMP_RES_C


def test_synth_msop_negative_temp_roundtrip():
    pkt = synth_msop_packet(temp_c=-10.0)
    assert abs(parse_robosense_temp_c(pkt) - (-10.0)) < ports.ROBOSENSE_TEMP_RES_C


def test_synth_imu_packet_shape():
    pkt = synth_imu_packet()
    assert len(pkt) == ports.ROBOSENSE_IMU_LEN
    assert pkt[:4] == ports.ROBOSENSE_IMU_MAGIC
