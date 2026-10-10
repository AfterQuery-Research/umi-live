"""chrony_status parsing for the manifest clock_sync block."""

from __future__ import annotations

from umi_flow_device import ptp

_TRACKING = """Reference ID    : 0A000001 (10.0.0.1)
Stratum         : 3
Ref time (UTC)  : Wed Jul 23 00:00:00 2026
System time     : 0.000012345 seconds slow of NTP time
Last offset     : -0.000001234 seconds
RMS offset      : 0.000005678 seconds
Frequency       : 1.234 ppm slow
Residual freq   : +0.000 ppm
Skew            : 0.100 ppm
Root delay      : 0.001000 seconds
Root dispersion : 0.001000 seconds
Update interval : 4.0 seconds
Leap status     : Normal
"""


def test_chrony_status_parses_signed_offsets(monkeypatch):
    monkeypatch.setattr(ptp.shutil, "which", lambda _n: "/usr/bin/chronyc")
    monkeypatch.setattr(ptp, "_run", lambda *a, **k: _TRACKING)
    cs = ptp.chrony_status("10.0.0.1")
    assert cs is not None
    assert cs.synced is True
    assert cs.ref == "10.0.0.1"
    assert cs.offset_to_ref_ns == -1234  # -0.000001234 s, SIGNED
    assert cs.rms_offset_ns == 5678


def test_chrony_status_unsynced_when_refid_zero(monkeypatch):
    txt = _TRACKING.replace("0A000001", "00000000")
    monkeypatch.setattr(ptp.shutil, "which", lambda _n: "/usr/bin/chronyc")
    monkeypatch.setattr(ptp, "_run", lambda *a, **k: txt)
    cs = ptp.chrony_status()
    assert cs is not None and cs.synced is False


def test_chrony_status_none_when_chronyc_absent(monkeypatch):
    monkeypatch.setattr(ptp.shutil, "which", lambda _n: None)
    assert ptp.chrony_status() is None


def test_chrony_status_none_when_command_fails(monkeypatch):
    monkeypatch.setattr(ptp.shutil, "which", lambda _n: "/usr/bin/chronyc")
    monkeypatch.setattr(ptp, "_run", lambda *a, **k: None)
    assert ptp.chrony_status() is None


def test_ptp_probe_requires_both_master_service_and_airy_lock(monkeypatch):
    monkeypatch.setattr(ptp, "timestamping_mode", lambda _iface: "hardware")
    monkeypatch.setattr(ptp, "service_active", lambda _iface: True)
    monkeypatch.setattr(ptp, "phc2sys_residual_ns", lambda _iface: 123)
    monkeypatch.setattr(ptp, "_airy_reports_lock", lambda _ip: False)

    status = ptp.probe_ptp({"eth0": "192.168.1.200"})["eth0"]

    assert status.locked is False
    assert status.timestamping_mode == "hardware"
    assert status.residual_offset_ns == 123

    monkeypatch.setattr(ptp, "_airy_reports_lock", lambda _ip: True)
    assert ptp.probe_ptp({"eth0": "192.168.1.200"})["eth0"].locked is True
