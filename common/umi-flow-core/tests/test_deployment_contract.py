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
        "LEADER_HOTSPOT_ENABLED",
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


def test_pi_installer_installs_and_verifies_the_lcd_font() -> None:
    installer = (ROOT / "device/scripts/install_pi.sh").read_text()
    display = (ROOT / "device/src/umi_flow_device/display.py").read_text()

    assert "fonts-dejavu-core" in installer
    assert "DejaVuSans-Bold.ttf" in installer
    assert "DejaVuSans-Bold.ttf" in display


def test_pi_installer_waits_for_ubuntu_maintenance_and_retries_downloads() -> None:
    installer = (ROOT / "device/scripts/install_pi.sh").read_text()

    assert 'APT_LOCK_TIMEOUT_SECONDS="${APT_LOCK_TIMEOUT_SECONDS:-1800}"' in installer
    assert "wait_for_ubuntu_package_maintenance" in installer
    assert "systemctl is-active --quiet apt-daily-upgrade.service" in installer
    assert "sudo env DEBIAN_FRONTEND=noninteractive dpkg --configure -a" in installer
    assert "reinstalling interrupted package(s)" in installer
    assert installer.count('DPkg::Lock::Timeout="$APT_LOCK_TIMEOUT_SECONDS"') == 3
    assert installer.count("Acquire::Retries=10") == 3
    assert "--retry 12 --retry-all-errors" in installer
    assert 'curl --proto \'=https\' --tlsv1.2' in installer
    assert '-o "$installer" "$url"' in installer
    assert "install_from_url https://astral.sh/uv/install.sh" in installer
    assert "install_from_url https://sh.rustup.rs -y --profile minimal" in installer


def test_private_pair_wifi_preserves_tcp_control_and_offline_clock_sync() -> None:
    provision = (ROOT / "bin/provision-pi").read_text()
    runtime = (ROOT / "device/scripts/pair_wifi_service.sh").read_text()
    installer = (ROOT / "device/scripts/install_pi.sh").read_text()
    unit = (ROOT / "config/systemd/umi-pair-wifi@.service").read_text()

    assert "UMI_FLOW_PEER=$PAIR_WIFI_FOLLOWER_IP:9140" in provision
    assert 'COORD_ENV="UMI_FLOW_PEER=${FOLLOWER#*@}:9140"' in provision
    assert "ID_BUS=usb" in provision
    assert 'PAIR_WIFI_RADIO_IFACE="wlan0"' in provision
    assert 'PAIR_WIFI_UPLINK_IFACE="wlan0"' in provision
    assert 'PAIR_WIFI_IFACE="$PAIR_WIFI_RADIO_IFACE"' in provision
    assert "waiting for follower to rejoin the private network" in provision
    assert 'exec hostapd "$HOSTAPD_CONFIG"' in runtime
    assert 'wpa_cli -i "$UMI_PAIR_IFACE" select_network' in runtime
    assert 'wpa_cli -i "$UMI_PAIR_IFACE" set ap_scan 2' in runtime
    assert 'wpa_cli -i "$UMI_PAIR_IFACE" reconfigure' in runtime
    assert 'resolvectl dns "$UMI_PAIR_IFACE"' in runtime
    assert "waiting for follower on configured network" in provision
    assert "systemctl disable --now umi-pair-wifi@follower.service" in provision
    assert "local stratum 10 orphan" in installer
    assert "server $UMI_PAIR_LEADER_IP" in installer
    assert 'if [ "$UMI_LEADER_HOTSPOT_ENABLED" = true ]; then' in installer
    assert "Before=umi-recorder.service umi-upload-tunnel.service" in unit


def test_pair_provisioning_generates_and_reuses_private_wifi_secret(tmp_path: Path) -> None:
    pair_config = tmp_path / "pair.env"
    pair_config.write_text(
        'SERVER="server@test"\n'
        'LEADER="leader@test"\n'
        'FOLLOWER="follower@test"\n'
        'TASK="deployment test"\n'
        "LEADER_HOTSPOT_ENABLED=true\n"
    )
    secret_file = tmp_path / "pair-wifi-psk"
    calls = tmp_path / "calls"
    provision_pi = tmp_path / "provision-pi"
    provision_pi.write_text(
        '#!/usr/bin/env bash\nprintf "%s %s\\n" "$*" "$UMI_PAIR_PROVISION_PSK" >> "$CALLS_FILE"\n'
    )
    provision_pi.chmod(0o755)
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    ssh = mock_bin / "ssh"
    ssh.write_text("#!/usr/bin/env bash\nexit 1\n")
    ssh.chmod(0o755)
    env = os.environ.copy()
    env.update(
        {
            "CALLS_FILE": str(calls),
            "PAIR_CONFIG": str(pair_config),
            "PAIR_WIFI_SECRET_FILE": str(secret_file),
            "PROVISION_PI": str(provision_pi),
            "PATH": f"{mock_bin}{os.pathsep}{env['PATH']}",
        }
    )

    first = subprocess.run(
        [str(ROOT / "bin/provision-pair"), "--dry-run"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    secret = secret_file.read_text().strip()
    assert "generated" in first.stdout
    assert secret not in first.stdout
    assert 8 <= len(secret) <= 63
    assert secret_file.stat().st_mode & 0o777 == 0o600

    second = subprocess.run(
        [str(ROOT / "bin/provision-pair"), "--dry-run"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "reused" in second.stdout
    assert secret_file.read_text().strip() == secret
    assert calls.read_text().splitlines() == [
        f"--role leader --dry-run {secret}",
        f"--role follower --dry-run {secret}",
        f"--role leader --dry-run {secret}",
        f"--role follower --dry-run {secret}",
    ]


def test_gptp_sync_uses_the_802_1as_multicast_address() -> None:
    config_text = (ROOT / "config/ptp/umi-gptp.cfg").read_text()
    assert "transportSpecific 0x1" in config_text
    assert "ptp_dst_mac 01:80:C2:00:00:0E" in config_text


def test_phc_follows_system_clock_across_late_ntp_steps() -> None:
    service = (ROOT / "config/systemd/umi-phc2sys@.service").read_text()
    assert "-s CLOCK_REALTIME -c %i" in service
    assert "-S 0.001" in service


def test_hotspot_dry_run_auto_detects_the_leader_usb_wifi_adapter(tmp_path: Path) -> None:
    mock_bin = tmp_path / "bin"
    mock_bin.mkdir()
    ssh = mock_bin / "ssh"
    ssh.write_text(
        r"""#!/usr/bin/env bash
remote="${!#}"
case "$remote" in
  *"ls /dev/v4l/by-path/"*) echo /dev/v4l/by-path/mock-usb-video-index0 ;;
  *"/sys/class/net/"*"ID_BUS=usb"*) echo wlxmockusb ;;
esac
"""
    )
    ssh.chmod(0o755)
    pair_config = tmp_path / "pair.env"
    pair_config.write_text(
        'SERVER="server@test"\n'
        'LEADER="leader@test"\n'
        'FOLLOWER="follower@test"\n'
        'TASK="deployment test"\n'
        "LEADER_HOTSPOT_ENABLED=true\n"
    )
    env = os.environ.copy()
    env.update(
        {
            "PAIR_CONFIG": str(pair_config),
            "UMI_PAIR_PROVISION_PSK": "generated-test-password",
            "PATH": f"{mock_bin}{os.pathsep}{env['PATH']}",
        }
    )

    result = subprocess.run(
        [str(ROOT / "bin/provision-pi"), "--role", "leader", "--dry-run"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "radio=wlxmockusb uplink=wlan0" in result.stdout


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
