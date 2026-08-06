"""Minimal classic-pcap READER, shared by device and server.

The reader half of the pcap plumbing lives in core so both the device (capture-side
stats/tests) and the server's offline DIFOP parsing can iterate a pcap's UDP
packets without a device->server dependency. The device's ``pcap.py`` re-exports these
names for back-compat and keeps the capture/writer halves (``PcapCapture``/``PcapWriter``),
which are device-only.

No scapy/dpkt: the Ethernet/IPv4/UDP framing we need is ~30 lines and fully under test.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

PCAP_MAGIC_US = 0xA1B2C3D4
PCAP_MAGIC_NS = 0xA1B23C4D
LINKTYPE_ETHERNET = 1

_UDP_HEADER = struct.Struct("!HHHH")


@dataclass(frozen=True)
class UdpPacket:
    ts_unix_ns: int
    src_ip: str
    src_port: int
    dst_ip: str
    dst_port: int
    payload: bytes


def read_udp_packets(path: Path) -> Iterator[UdpPacket]:
    """Iterate UDP packets in a classic pcap (usec or nanosec, EN10MB)."""
    with path.open("rb") as f:
        header = f.read(24)
        if len(header) < 24:
            return
        magic = struct.unpack("<I", header[:4])[0]
        if magic == PCAP_MAGIC_NS:
            frac_scale = 1
        elif magic == PCAP_MAGIC_US:
            frac_scale = 1_000
        else:
            raise ValueError(f"{path}: not a classic pcap (magic {magic:#x})")
        linktype = struct.unpack("<I", header[20:24])[0]
        if linktype != LINKTYPE_ETHERNET:
            raise ValueError(f"{path}: unsupported linktype {linktype}")

        while True:
            rec = f.read(16)
            if len(rec) < 16:
                return
            ts_sec, ts_frac, incl_len, _orig_len = struct.unpack("<IIII", rec)
            frame = f.read(incl_len)
            if len(frame) < incl_len:
                return
            pkt = _parse_udp(frame)
            if pkt is None:
                continue
            src_ip, src_port, dst_ip, dst_port, payload = pkt
            yield UdpPacket(
                ts_unix_ns=ts_sec * 1_000_000_000 + ts_frac * frac_scale,
                src_ip=src_ip,
                src_port=src_port,
                dst_ip=dst_ip,
                dst_port=dst_port,
                payload=payload,
            )


def _parse_udp(frame: bytes) -> tuple[str, int, str, int, bytes] | None:
    if len(frame) < 14 + 20 + 8:
        return None
    ethertype = struct.unpack("!H", frame[12:14])[0]
    if ethertype != 0x0800:
        return None
    ihl = (frame[14] & 0x0F) * 4
    if frame[14 + 9] != 17:  # not UDP
        return None
    src_ip = ".".join(str(b) for b in frame[14 + 12 : 14 + 16])
    dst_ip = ".".join(str(b) for b in frame[14 + 16 : 14 + 20])
    udp_off = 14 + ihl
    src_port, dst_port, udp_len, _ = _UDP_HEADER.unpack(frame[udp_off : udp_off + 8])
    payload = frame[udp_off + 8 : udp_off + max(8, udp_len)]
    return src_ip, src_port, dst_ip, dst_port, payload
