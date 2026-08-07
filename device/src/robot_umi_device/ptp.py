"""Best-effort PTP status probe for the episode manifest.

The Pi runs ptp4l as MASTER per lidar interface (systemd template ``umi-ptp@``),
disciplining the lidar's clock to Pi time. What we can observe host-side:

- whether the per-interface PTP service is running (``locked`` proxy),
- whether the NIC offers hardware timestamping (``ethtool -T``; the Pi 5's onboard
  eth0 has a PHC, USB-GbE dongles usually don't -> software),
- the phc2sys system<->PHC residual when hardware timestamping is active.

The lidar-side (slave) offset is not directly observable from the master; the
decode stage cross-checks it offline (payload time vs pcap arrival time). Every
probe failure degrades gracefully to an honest "not locked / none" — the manifest
must never lie about sync quality.
"""

from __future__ import annotations

import re
import shutil
import subprocess

from robot_umi_core import manifest

_ETHTOOL_HW_RE = re.compile(r"hardware-transmit|SOF_TIMESTAMPING_TX_HARDWARE")
_PHC2SYS_OFFSET_RE = re.compile(r"offset\s+(-?\d+)")

# `chronyc tracking` fields (seconds; kept SIGNED so the merge's leader-minus-follower
# cross-Pi subtraction gives the true skew direction).
_CHRONY_REFID_RE = re.compile(r"Reference ID\s*:\s*([0-9A-Fa-f]+)")
_CHRONY_LEAP_RE = re.compile(r"Leap status\s*:\s*(\w+)")
_CHRONY_LAST_OFFSET_RE = re.compile(r"Last offset\s*:\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)")
_CHRONY_RMS_RE = re.compile(r"RMS offset\s*:\s*([-+]?\d+\.?\d*(?:[eE][-+]?\d+)?)")


def _run(cmd: list[str], timeout: float = 3.0) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout


def timestamping_mode(iface: str) -> str:
    ethtool = shutil.which("ethtool")
    if ethtool is None:
        return manifest.PTP_MODE_NONE
    out = _run([ethtool, "-T", iface])
    if out is None:
        return manifest.PTP_MODE_NONE
    if _ETHTOOL_HW_RE.search(out):
        return manifest.PTP_MODE_HARDWARE
    if "software-transmit" in out or "SOF_TIMESTAMPING_TX_SOFTWARE" in out:
        return manifest.PTP_MODE_SOFTWARE
    return manifest.PTP_MODE_NONE


def service_active(iface: str) -> bool:
    out = _run(["systemctl", "is-active", f"umi-ptp@{iface}"])
    return out is not None and out.strip() == "active"


def phc2sys_residual_ns(iface: str) -> int | None:
    """Last phc2sys offset from the journal, if hardware timestamping is in play."""
    out = _run(["journalctl", "-u", f"umi-phc2sys@{iface}", "-n", "1", "--no-pager", "-o", "cat"])
    if not out:
        return None
    m = _PHC2SYS_OFFSET_RE.search(out)
    return int(m.group(1)) if m else None


def chrony_status(ref: str = "") -> manifest.ClockSyncStatus | None:
    """Best-effort cross-Pi clock quality from ``chronyc tracking``, for the manifest's
    ``clock_sync`` block. ``ref`` is the shared NTP source both Pis discipline to (recorded
    for provenance). Returns None if chrony isn't present/reachable; ``synced=False`` if it
    isn't locked to a real reference — the manifest must never overstate sync quality."""
    chronyc = shutil.which("chronyc")
    if chronyc is None:
        return None
    out = _run([chronyc, "tracking"])
    if out is None:
        return None

    def _to_ns(pattern: re.Pattern[str]) -> int | None:
        m = pattern.search(out)
        return int(round(float(m.group(1)) * 1e9)) if m else None

    refid = _CHRONY_REFID_RE.search(out)
    leap = _CHRONY_LEAP_RE.search(out)
    synced = bool(
        leap
        and leap.group(1) == "Normal"
        and refid
        and int(refid.group(1), 16) != 0  # 00000000 = not yet synced to any source
    )
    return manifest.ClockSyncStatus(
        synced=synced,
        ref=ref,
        offset_to_ref_ns=_to_ns(_CHRONY_LAST_OFFSET_RE),
        rms_offset_ns=_to_ns(_CHRONY_RMS_RE),
    )


def probe_ptp(ifaces: list[str]) -> dict[str, manifest.PtpStatus]:
    """One PtpStatus per lidar interface, all best-effort."""
    statuses: dict[str, manifest.PtpStatus] = {}
    for iface in ifaces:
        mode = timestamping_mode(iface)
        active = service_active(iface)
        residual = phc2sys_residual_ns(iface) if mode == manifest.PTP_MODE_HARDWARE else None
        statuses[iface] = manifest.PtpStatus(
            locked=active,
            timestamping_mode=mode,
            residual_offset_ns=residual,
        )
    return statuses
