"""Standard-pcap plumbing for raw lidar capture.

Three pieces, all speaking the same on-disk format (classic pcap, EN10MB link type):

- :class:`PcapCapture` — wraps ``tcpdump`` for the real capture path. AF_PACKET taps
  packets pre-routing (so capture works regardless of rp_filter), the kernel stamps
  every packet on arrival, and kernel drop counts are parsed from tcpdump's exit
  summary so losses are visible, never silent.
- :class:`PcapWriter` — a pure-Python nanosecond-pcap writer that synthesizes
  Ethernet/IPv4/UDP framing around payloads. Used by the CI mock lidar so mock and
  hardware exercise byte-compatible files (rslidar_sdk pcap replay reads either).
- :func:`read_udp_packets` — a minimal reader (both microsecond and nanosecond
  magics) for stats, tests, and the server's DIFOP parser. This half now lives
  in :mod:`robot_umi_core.pcapio` (so the server can read pcaps without depending on
  this device package); it is re-exported here for back-compat.

No scapy/dpkt dependency: the framing we need is ~40 lines and fully under test.
"""

from __future__ import annotations

import re
import shutil
import signal
import struct
import subprocess
from pathlib import Path

# The reader half + wire constants live in core; re-export for back-compat.
from robot_umi_core.pcapio import (  # noqa: F401
    LINKTYPE_ETHERNET,
    PCAP_MAGIC_NS,
    PCAP_MAGIC_US,
    UdpPacket,
    read_udp_packets,
)

_ETH_HEADER = struct.Struct("!6s6sH")
_IP_HEADER = struct.Struct("!BBHHHBBH4s4s")
_UDP_HEADER = struct.Struct("!HHHH")


def _ip_checksum(header: bytes) -> int:
    total = 0
    for i in range(0, len(header), 2):
        total += (header[i] << 8) | header[i + 1]
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return ~total & 0xFFFF


def _ip_to_bytes(ip: str) -> bytes:
    return bytes(int(part) for part in ip.split("."))


class PcapWriter:
    """Nanosecond classic pcap with synthesized Ethernet/IPv4/UDP framing."""

    def __init__(self, path: Path):
        self._f = path.open("wb")
        self._f.write(struct.pack("<IHHiIII", PCAP_MAGIC_NS, 2, 4, 0, 0, 0xFFFF, LINKTYPE_ETHERNET))

    def write_udp(
        self,
        ts_unix_ns: int,
        payload: bytes,
        *,
        src_ip: str,
        src_port: int,
        dst_ip: str,
        dst_port: int,
    ) -> None:
        udp = _UDP_HEADER.pack(src_port, dst_port, 8 + len(payload), 0) + payload
        ip_len = 20 + len(udp)
        ip_wo_checksum = _IP_HEADER.pack(
            0x45, 0, ip_len, 0, 0, 64, 17, 0, _ip_to_bytes(src_ip), _ip_to_bytes(dst_ip)
        )
        checksum = _ip_checksum(ip_wo_checksum)
        ip = _IP_HEADER.pack(
            0x45,
            0,
            ip_len,
            0,
            0,
            64,
            17,
            checksum,
            _ip_to_bytes(src_ip),
            _ip_to_bytes(dst_ip),
        )
        frame = _ETH_HEADER.pack(b"\x02" * 6, b"\x04" * 6, 0x0800) + ip + udp
        self._f.write(
            struct.pack(
                "<IIII",
                ts_unix_ns // 1_000_000_000,
                ts_unix_ns % 1_000_000_000,
                len(frame),
                len(frame),
            )
        )
        self._f.write(frame)

    def flush(self) -> None:
        self._f.flush()

    def close(self) -> None:
        self._f.close()


# --- live capture --------------------------------------------------------------------

_DROP_RE = re.compile(r"(\d+)\s+packets? dropped by kernel")
_CAPTURED_RE = re.compile(r"(\d+)\s+packets? captured")
STOP_TIMEOUT_SECONDS = 15.0
KILL_TIMEOUT_SECONDS = 5.0


class PcapCapture:
    """One tcpdump process writing one pcap file (one lidar interface).

    Capture is passive: an unplugged lidar is just a packet gap, never an error here
    (liveness/QC is the monitor's job). Errors reported by ``poll_error`` are process
    failures only (tcpdump died: missing binary, no permission, bad interface).
    """

    def __init__(self, iface: str, bpf: str, *, buffer_kib: int = 8192):
        self.iface = iface
        self.bpf = bpf
        self.buffer_kib = buffer_kib
        self.process: subprocess.Popen[bytes] | None = None
        self._stats: dict[str, int] = {}
        self._error: str | None = None

    def start(self, output_path: Path) -> None:
        tcpdump = shutil.which("tcpdump")
        if tcpdump is None:
            raise RuntimeError("tcpdump is required for lidar capture")
        cmd = [
            tcpdump,
            "-i",
            self.iface,
            "-w",
            str(output_path),
            "--time-stamp-precision=nano",
            "-B",
            str(self.buffer_kib),
            "-q",
            self.bpf,
        ]
        self._stats = {}
        self._error = None
        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )

    def poll_error(self) -> str | None:
        if self._error is not None:
            return self._error
        if self.process is None:
            return None
        code = self.process.poll()
        if code is None:
            return None
        stderr = self.process.stderr.read().decode(errors="replace") if self.process.stderr else ""
        self._error = f"tcpdump exited early (code {code}): {stderr.strip()[:300]}"
        self.process = None
        return self._error

    def stop(self) -> dict[str, int]:
        """SIGINT (flush + stats summary), parse captured/dropped counts."""
        process = self.process
        self.process = None
        if process is None:
            return dict(self._stats)
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=STOP_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=KILL_TIMEOUT_SECONDS)
        stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
        captured = _CAPTURED_RE.search(stderr)
        dropped = _DROP_RE.search(stderr)
        self._stats = {
            "packets_captured": int(captured.group(1)) if captured else 0,
            "packets_dropped_kernel": int(dropped.group(1)) if dropped else 0,
        }
        return dict(self._stats)

    def cancel(self) -> None:
        process = self.process
        self.process = None
        if process is None:
            return
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=KILL_TIMEOUT_SECONDS)
            except subprocess.TimeoutExpired:
                process.kill()
        if process.stderr is not None:
            process.stderr.read()
