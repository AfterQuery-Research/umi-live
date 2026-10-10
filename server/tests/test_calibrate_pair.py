from __future__ import annotations

import os
import pty
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "bin" / "calibrate-pair"


def test_dry_run_describes_pair_without_contacting_clients(tmp_path: Path) -> None:
    pair_config = tmp_path / "pair.env"
    pair_config.write_text(
        'LEADER="alice@leader.local"\n'
        'FOLLOWER="bob@follower.local"\n'
        'SERVER="server@example.local"\n'
        'TASK="test"\n'
    )
    env = os.environ.copy()
    env["PAIR_CONFIG"] = str(pair_config)

    result = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "no local or remote changes" in result.stdout
    assert "alice@leader.local" in result.stdout
    assert "bob@follower.local" in result.stdout
    assert "verify -> generate board -> calibrate both -> deploy both" in result.stdout


def test_pair_flow_uses_mocked_clients_and_replaces_both_local_files(tmp_path: Path) -> None:
    test_root = tmp_path / "repo"
    (test_root / "bin").mkdir(parents=True)
    (test_root / "tools").mkdir()
    calibration_dir = test_root / "config" / "calib"
    calibration_dir.mkdir(parents=True)
    shutil.copy2(SCRIPT, test_root / "bin" / "calibrate-pair")
    leader_calibration = calibration_dir / "umi1_calib.npz"
    follower_calibration = calibration_dir / "umi2_calib.npz"
    leader_calibration.write_bytes(b"old leader")
    follower_calibration.write_bytes(b"old follower")

    pair_config = tmp_path / "pair.env"
    pair_config.write_text('LEADER="alice@leader.local"\nFOLLOWER="bob@follower.local"\n')
    call_log = tmp_path / "calls.log"
    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_uv = fake_bin / "uv"
    fake_uv.write_text(
        "#!/bin/sh\n"
        'printf \'uv %s\\n\' "$*" >> "$CALL_LOG"\n'
        'args="$*"; out=; host=\n'
        "while [ $# -gt 0 ]; do\n"
        '  case "$1" in\n'
        "    --out) out=$2; shift 2 ;;\n"
        "    --host) host=$2; shift 2 ;;\n"
        "    *) shift ;;\n"
        "  esac\n"
        "done\n"
        'case "$args" in\n'
        '  *make_charuco.py*) printf board > "$out" ;;\n'
        "  *--preflight-host*) : ;;\n"
        '  *calibrate_camera.py*) printf \'new:%s\' "$host" > "$out" ;;\n'
        "esac\n"
    )
    fake_ssh = fake_bin / "ssh"
    fake_ssh.write_text('#!/bin/sh\nprintf \'ssh %s\\n\' "$*" >> "$CALL_LOG"\n')
    fake_scp = fake_bin / "scp"
    fake_scp.write_text('#!/bin/sh\nprintf \'scp %s\\n\' "$*" >> "$CALL_LOG"\n')
    for command in (fake_uv, fake_ssh, fake_scp):
        command.chmod(0o755)

    env = os.environ.copy()
    env.update(
        PAIR_CONFIG=str(pair_config),
        CALL_LOG=str(call_log),
        PATH=f"{fake_bin}:{env['PATH']}",
        TMPDIR=str(tmp_path),
    )
    master, slave = pty.openpty()
    process = subprocess.Popen(
        ["bash", str(test_root / "bin" / "calibrate-pair")],
        cwd=test_root,
        env=env,
        stdin=slave,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    os.close(slave)
    try:
        os.write(master, b"\n")
        stdout, stderr = process.communicate(timeout=10)
    finally:
        os.close(master)

    assert process.returncode == 0, stderr
    assert "calibration complete" in stdout
    assert leader_calibration.read_bytes() == b"new:alice@leader.local"
    assert follower_calibration.read_bytes() == b"new:bob@follower.local"
    calls = call_log.read_text()
    assert calls.count("--preflight-host") == 4
    assert calls.count("\nscp ") == 2
    assert "umi1_calib.npz.new" in calls
    assert "umi2_calib.npz.new" in calls
