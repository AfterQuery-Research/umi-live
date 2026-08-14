"""Repository-level guards for the single supported deployment workflow."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from umi_flow_core import config

ROOT = Path(__file__).resolve().parents[3]


def test_pair_inventory_has_only_supported_inputs() -> None:
    lines = [
        line.split("=", 1)[0]
        for line in (ROOT / "provisioning/pair.env.example").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    assert lines == [
        "SERVER",
        "LEADER",
        "FOLLOWER",
        "TASK",
        "DATA_NAMESPACE",
        "ALIGNMENT_ENABLED",
        "PAIR_WIFI_SSID",
        "PAIR_WIFI_PASSWORD",
        "PAIR_WIFI_CHANNEL",
        "PAIR_WIFI_COUNTRY",
        "HF_REPO_ID",
        "HF_TOKEN",
    ]


def test_only_one_readme_and_no_legacy_inventory() -> None:
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.splitlines()
    readmes = [ROOT / relative for relative in tracked if Path(relative).name.startswith("README")]
    assert readmes == [ROOT / "README.md"]
    for relative in (
        ".env.example",
        "bin/switch-target",
        "provisioning/secrets.example",
        "provisioning/server.env.example",
        "provisioning/rigs/example.env",
        "config/umi.pair-leader.example.yaml",
        "config/umi.pair-follower.example.yaml",
    ):
        assert not (ROOT / relative).exists()


def test_provisioning_scripts_are_valid_and_have_no_removed_inputs() -> None:
    scripts = [
        ROOT / "bin/provision-server",
        ROOT / "bin/provision-pair",
        ROOT / "bin/provision-pi",
        ROOT / "device/scripts/install_pi.sh",
        ROOT / "device/scripts/pair_wifi_service.sh",
    ]
    subprocess.run(["bash", "-n", *map(str, scripts)], check=True)
    text = "\n".join(path.read_text() for path in scripts)
    for removed in (
        "UPLOAD_TOKEN",
        "RUNNER_ARGS",
        "RIG_TARGET",
        "TIME_REF",
        "ALLOW_SD",
        "SERVER_URL",
    ):
        assert removed not in text


def test_rsync_deployment_excludes_internal_workspaces() -> None:
    exclude_file = ROOT / "config" / "rsync-excludes"
    patterns = {
        line for line in exclude_file.read_text().splitlines() if line and not line.startswith("#")
    }
    assert patterns == {
        "/dataset-filtering/",
        "/dataset-filtering-tests/",
        "/*CODEBASE_REVIEW.md",
        "/*.SCOPE.md",
    }
    for script in (ROOT / "bin/provision-server", ROOT / "bin/provision-pi"):
        assert '--exclude-from "$ROOT/config/rsync-excludes"' in script.read_text()


def test_upload_service_is_loopback_only() -> None:
    compose = (ROOT / "docker-compose.yml").read_text()
    assert '"127.0.0.1:8080:8080"' in compose


def test_private_pair_wifi_preserves_tcp_control_and_offline_clock_sync() -> None:
    provision = (ROOT / "bin/provision-pi").read_text()
    runtime = (ROOT / "device/scripts/pair_wifi_service.sh").read_text()
    installer = (ROOT / "device/scripts/install_pi.sh").read_text()
    unit = (ROOT / "config/systemd/umi-pair-wifi@.service").read_text()

    assert "UMI_FLOW_PEER=$PAIR_WIFI_FOLLOWER_IP:9140" in provision
    assert 'exec hostapd "$HOSTAPD_CONFIG"' in runtime
    assert 'exec wpa_supplicant -D nl80211' in runtime
    assert "local stratum 10 orphan" in installer
    assert "server $UMI_PAIR_LEADER_IP" in installer
    assert "Before=umi-recorder.service umi-upload-tunnel.service" in unit


def test_gptp_sync_uses_the_802_1as_multicast_address() -> None:
    config_text = (ROOT / "config/ptp/umi-gptp.cfg").read_text()
    assert "transportSpecific 0x1" in config_text
    assert "ptp_dst_mac 01:80:C2:00:00:0E" in config_text


def test_phc_follows_system_clock_across_late_ntp_steps() -> None:
    service = (ROOT / "config/systemd/umi-phc2sys@.service").read_text()
    assert "-s CLOCK_REALTIME -c %i" in service
    assert "-S 0.001" in service


def test_password_bootstrap_survives_a_verified_client_reboot(tmp_path: Path) -> None:
    """A password-only Pi must reboot before its temporary sudo grant is removed."""
    mock_bin = tmp_path / "bin"
    state = tmp_path / "state"
    mock_bin.mkdir()
    state.mkdir()

    ssh = mock_bin / "ssh"
    ssh.write_text(
        r"""#!/usr/bin/env bash
set -eu
remote="${!#}"
printf '%s\n' "$remote" >> "$MOCK_STATE_DIR/commands"
case "$remote" in
  "sudo -n true")
    test -f "$MOCK_STATE_DIR/sudo"
    ;;
  *"sh -c"*"90-umi-provision"*)
    cat >/dev/null || true
    touch "$MOCK_STATE_DIR/sudo"
    ;;
  *"rm -f /etc/sudoers.d/90-umi-provision"*)
    test -f "$MOCK_STATE_DIR/sudo"
    if test -f "$MOCK_STATE_DIR/rebooted"; then
      touch "$MOCK_STATE_DIR/cleanup_after_reboot"
    else
      touch "$MOCK_STATE_DIR/cleanup_before_reboot"
    fi
    rm -f "$MOCK_STATE_DIR/sudo"
    ;;
  *"ls /dev/v4l/by-path/"*)
    echo /dev/v4l/by-path/mock-usb-video-index0
    ;;
  "cat ~/.ssh/id_ed25519.pub")
    echo 'ssh-ed25519 mock-key provision-test'
    ;;
  *"authorized_keys"*)
    cat >/dev/null || true
    ;;
  *"sudo tee /etc/systemd/system/umi-upload-tunnel.service"*)
    cat >/dev/null || true
    ;;
  "cat /proc/sys/kernel/random/boot_id")
    if test -f "$MOCK_STATE_DIR/rebooted"; then echo boot-after; else echo boot-before; fi
    ;;
  *"systemctl reboot"*)
    test -f "$MOCK_STATE_DIR/sudo"
    touch "$MOCK_STATE_DIR/rebooted"
    exit 255
    ;;
  *"systemctl is-active --quiet umi-recorder"*)
    test -f "$MOCK_STATE_DIR/rebooted"
    ;;
  *)
    ;;
esac
"""
    )
    ssh.chmod(0o755)
    for name in ("rsync", "scp", "sleep"):
        tool = mock_bin / name
        tool.write_text("#!/usr/bin/env bash\nexit 0\n")
        tool.chmod(0o755)

    pair_config = tmp_path / "pair.env"
    pair_config.write_text(
        'SERVER="server@test"\n'
        'LEADER="leader@test"\n'
        'FOLLOWER="follower@test"\n'
        'TASK="deployment test"\n'
        'PAIR_WIFI_SSID="umi-flow-test"\n'
        'PAIR_WIFI_PASSWORD="test-password"\n'
        'PAIR_WIFI_CHANNEL=44\n'
        'PAIR_WIFI_COUNTRY="US"\n'
    )
    gripper_calib = tmp_path / "gripper-calib"
    gripper_calib.mkdir()
    for name in ("umi1", "umi2"):
        (gripper_calib / f"{name}_gripper_range.json").write_text(
            '{"gripper":"' + name + '","closed_width_mm":40,"open_width_mm":120}'
        )
    env = os.environ.copy()
    env.update(
        {
            "MOCK_STATE_DIR": str(state),
            "PAIR_CONFIG": str(pair_config),
            "GRIPPER_CALIB_DIR": str(gripper_calib),
            "UMI_SUDO_PASSWORD": "aq",
            "PATH": f"{mock_bin}{os.pathsep}{env['PATH']}",
        }
    )
    result = subprocess.run(
        [str(ROOT / "bin/provision-pi"), "--role", "follower"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "follower client ready" in result.stdout
    follower_tags = config.load_gripper_tag_pairs(ROOT / "config/gripper_tags.yaml")["umi2"]
    assert f"tags=[{follower_tags[0]}, {follower_tags[1]}]" in result.stdout
    assert (state / "rebooted").exists()
    assert (state / "cleanup_after_reboot").exists()
    assert not (state / "cleanup_before_reboot").exists()
    assert not (state / "sudo").exists()

    commands = (state / "commands").read_text().splitlines()
    reboot = next(i for i, command in enumerate(commands) if "systemctl reboot" in command)
    cleanup = next(i for i, command in enumerate(commands) if "rm -f /etc/sudoers.d" in command)
    assert reboot < cleanup
