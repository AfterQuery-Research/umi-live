#!/usr/bin/env bash
# install_pi.sh — take a freshly-flashed Ubuntu 24.04 Raspberry Pi 5 to a working
# UMI recorder in one run. Idempotent: safe to re-run after editing .env/config.
#
#   cp .env.example .env && $EDITOR .env      # fill in server URL + spool dir
#   device/scripts/install_pi.sh              # run as the LOGIN user (not root)
#
# It does everything the old bring-up left as tribal knowledge: apt deps, uv +
# Rust toolchains, the Python env + Rust recorder build, tcpdump capture
# capability (setcap — else every lidar capture dies), SPI overlay + group
# membership for the LCD/GPIO/camera, lidar network + PTP from the rig config,
# the episode spool, the rig config file, a recorder systemd unit, and a
# self-test that turns each old silent failure into a loud check.
set -euo pipefail

# --- preflight -----------------------------------------------------------------
if [ "$(id -u)" -eq 0 ]; then
  echo "run as your login user (it sudo's what it needs); running as root would" \
       "leave the uv/cargo/venv files root-owned." >&2
  exit 1
fi
command -v apt-get >/dev/null || { echo "this installer targets Debian/Ubuntu (apt)." >&2; exit 1; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
USER_NAME="$(id -un)"
cd "$REPO"

# Load .env (server URL, spool dir, rig config path) if present.
if [ -f "$REPO/.env" ]; then
  set -a; . "$REPO/.env"; set +a
else
  echo "note: no .env yet — copying .env.example -> .env (edit it and re-run)."
  cp "$REPO/.env.example" "$REPO/.env"
fi
CONFIG_REL="${ROBOT_UMI_CONFIG:-config/umi.yaml}"
CONFIG_PATH="$REPO/$CONFIG_REL"
SPOOL_DIR="${ROBOT_UMI_SPOOL_DIR:-$REPO/device/robot_umi_spool}"
[ -n "${ROBOT_UMI_UPLOAD_TOKEN:-}" ] || echo "WARN: ROBOT_UMI_UPLOAD_TOKEN is blank" \
  "in .env — uploads will 401 unless the server is tokenless. Set the SAME value as the GPU box."

# needrestart (Ubuntu's post-apt hook) auto-restarts services after upgrades — including
# systemd-networkd/resolved, which briefly kills DNS mid-install and makes the uv/rust
# downloads fail ("could not resolve host"). Suspend it; the final reboot restarts all.
export NEEDRESTART_SUSPEND=1

echo "=== [1/11] apt packages ==="
# Fresh RPi Ubuntu images sometimes ship -updates packages already installed but omit
# the -updates POCKET from the sources (only <codename> + <codename>-security). Then
# `apt install <foo>-dev` can only see the OLDER -dev, which conflicts with the newer
# installed runtime lib -> "held broken packages". Add <codename>-updates if missing.
. /etc/os-release 2>/dev/null || true
APT_SRC=/etc/apt/sources.list.d/ubuntu.sources
if [ -f "$APT_SRC" ] && [ -n "${VERSION_CODENAME:-}" ] \
   && ! grep -qE "^Suites:.*${VERSION_CODENAME}-updates" "$APT_SRC"; then
  sudo sed -i "s/^Suites: ${VERSION_CODENAME}\$/Suites: ${VERSION_CODENAME} ${VERSION_CODENAME}-updates/" "$APT_SRC"
  echo "added ${VERSION_CODENAME}-updates pocket to $APT_SRC (was missing on this image)"
fi
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
  git curl ca-certificates build-essential pkg-config \
  tcpdump ethtool linuxptp libcap2-bin v4l-utils python3-dev

echo "=== [2/11] uv (Python) ==="
if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "=== [3/11] Rust (camera recorder) ==="
if ! command -v cargo >/dev/null && [ ! -x "$HOME/.cargo/bin/cargo" ]; then
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
export PATH="$HOME/.cargo/bin:$PATH"
cargo --version

echo "=== [4/11] Python env (device + Pi extras: GPIO/LCD) ==="
uv sync --package robot-umi-device --extra pi

echo "=== [5/11] build the Rust V4L2 recorder ==="
# Force a recompile of the current source. rsync -a preserves the source's (possibly
# older) mtime, which can be OLDER than a stale binary left by a prior build — cargo
# would then skip the rebuild and ship outdated code (e.g. a camera binary missing
# --exposure). Touching the sources guarantees cargo sees them as newest.
find "$REPO/device/v4l2_recorder/src" -name '*.rs' -exec touch {} +
cargo build --release --manifest-path "$REPO/device/v4l2_recorder/Cargo.toml"

echo "=== [6/11] tcpdump capture capability (setcap) ==="
# Without this, record.py's tcpdump dies instantly and every episode records
# camera-only, marked dataset_eligible=false — the #1 silent bring-up failure.
TCPDUMP="$(command -v tcpdump)"
sudo setcap cap_net_raw,cap_net_admin+eip "$TCPDUMP"
getcap "$TCPDUMP"

echo "=== [7/11] SPI overlay + device groups + udev (LCD / GPIO / camera) ==="
BOOTCFG=/boot/firmware/config.txt
[ -f "$BOOTCFG" ] || BOOTCFG=/boot/config.txt
if [ -f "$BOOTCFG" ] && ! grep -q '^dtparam=spi=on' "$BOOTCFG"; then
  echo 'dtparam=spi=on' | sudo tee -a "$BOOTCFG" >/dev/null
  echo "enabled SPI in $BOOTCFG (LCD needs a reboot to see /dev/spidev0.0)"
fi
# The recorder runs as a USER systemd service; its unit SupplementaryGroups are
# video gpio spi dialout i2c. Raspberry Pi OS ships these groups + a udev rule making
# /dev/gpiochip* group-accessible; Ubuntu Server for Pi does NOT — MISSING groups make
# systemd fail the unit with 216/GROUP, and /dev/gpiochip* is root:root 0600 so a user
# process can't reach the GPIO buttons. Create the groups + a udev rule for parity.
# (Creating them before the post-install reboot means systemd resolves them cleanly on
# the next boot — no daemon-reexec needed.)
NEED_RELOGIN=0
for grp in video gpio spi dialout plugdev i2c; do
  getent group "$grp" >/dev/null || { sudo groupadd --system "$grp"; echo "created group $grp"; }
  if ! id -nG "$USER_NAME" | tr ' ' '\n' | grep -qx "$grp"; then
    sudo usermod -aG "$grp" "$USER_NAME"; NEED_RELOGIN=1
  fi
done
sudo tee /etc/udev/rules.d/99-umi-gpio.rules >/dev/null <<'UDEV'
KERNEL=="gpiochip*", GROUP="gpio", MODE="0660"
UDEV
sudo udevadm control --reload-rules && sudo udevadm trigger

echo "=== [8/11] rig config ==="
if [ ! -f "$CONFIG_PATH" ]; then
  mkdir -p "$(dirname "$CONFIG_PATH")"
  cp "$REPO/config/umi.single.example.yaml" "$CONFIG_PATH"
  echo "seeded $CONFIG_REL from umi.single.example.yaml — EDIT it (camera device, calib, lidar)."
fi
# Fail loudly on an unparseable config NOW (unknown key, bad YAML) instead of
# silently skipping lidar/PTP below and reporting a false "install complete".
if ! ERR="$("$REPO/.venv/bin/python" -c \
    'import sys; from robot_umi_core.config import load_rig_config; load_rig_config(sys.argv[1])' \
    "$CONFIG_PATH" 2>&1)"; then
  echo "config error in $CONFIG_REL:" >&2; echo "$ERR" >&2
  echo "edit $CONFIG_REL and re-run." >&2; exit 1
fi

echo "=== [9/11] lidar network + PTP (from the rig config) ==="
IFACES=""
LIDARS="$("$REPO/.venv/bin/python" - "$CONFIG_PATH" <<'PY'
import sys
from robot_umi_core.config import load_rig_config
rig = load_rig_config(sys.argv[1])   # already validated above
for u in rig.umis:
    if u.lidar is not None:
        print(f"{u.lidar.iface} {u.lidar.host_ip} {u.lidar.lidar_ip}")
PY
)"
if [ -n "$LIDARS" ]; then
  while read -r IFACE HOST_IP LIDAR_IP; do
    [ -z "${IFACE:-}" ] && continue
    echo "--- lidar on $IFACE (host $HOST_IP, lidar $LIDAR_IP) ---"
    sudo "$REPO/device/scripts/setup_lidar_net.sh" "$IFACE" "$HOST_IP/24" "$LIDAR_IP" || \
      echo "WARN: lidar-net setup failed on $IFACE (cabled? powered?) — re-run later"
    IFACES="$IFACES $IFACE"
  done <<< "$LIDARS"
  # shellcheck disable=SC2086
  sudo "$REPO/device/scripts/setup_ptp.sh" $IFACES || echo "WARN: PTP setup incomplete"
else
  echo "no lidar in $CONFIG_REL (vive-only rig, or config unedited) — skipping lidar net/PTP."
fi

echo "=== [10/11] episode spool ==="
mkdir -p "$SPOOL_DIR"
# The spool holds raw pcap + video (GB per episode). On the SD card (root fs) it fills
# fast and wears the card, so require an SSD mount unless explicitly overridden.
if [ "$(df --output=target "$SPOOL_DIR" | tail -1)" = "/" ]; then
  MSG="spool $SPOOL_DIR is on the SD card (root fs), not an SSD."
  if [ "${ROBOT_UMI_ALLOW_SD_SPOOL:-0}" = "1" ]; then
    echo "WARN: $MSG ROBOT_UMI_ALLOW_SD_SPOOL=1 set — continuing."
  else
    echo "ERROR: $MSG Mount the SSD and set ROBOT_UMI_SPOOL_DIR in .env to a path on" >&2
    echo "it, then re-run. (Set ROBOT_UMI_ALLOW_SD_SPOOL=1 to override on a dev box.)" >&2
    exit 1
  fi
fi

echo "=== [11/11] recorder systemd unit ==="
sed -e "s#@REPO@#$REPO#g" -e "s#@USER@#$USER_NAME#g" \
    "$REPO/config/systemd/umi-recorder.service" | sudo tee /etc/systemd/system/umi-recorder.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable umi-recorder >/dev/null   # autostart on EVERY boot (the whole point)
if systemctl is-active --quiet umi-recorder; then
  sudo systemctl restart umi-recorder            # already up: pick up new code/config
  echo "umi-recorder enabled + restarted."
else
  echo "umi-recorder enabled (autostart on boot). It comes up on the next boot — which is"
  echo "needed anyway to activate the SPI overlay + group membership from step 7."
fi
echo "dev session by hand instead (keyboard buttons, no LCD):"
echo "    cd $REPO && .venv/bin/python device/scripts/record.py --buttons keyboard --display none"

# --- self-test -----------------------------------------------------------------
echo
echo "=== self-test ==="
fail=0
getcap "$TCPDUMP" | grep -q cap_net_raw && echo "OK  tcpdump can capture" || { echo "FAIL tcpdump missing capability"; fail=1; }
[ -x "$REPO/device/v4l2_recorder/target/release/robot-umi-v4l2-recorder" ] \
  && echo "OK  camera recorder built" || { echo "FAIL recorder binary missing"; fail=1; }
"$REPO/.venv/bin/python" -c "import robot_umi_device.sensors, robot_umi_device.camera" 2>/dev/null \
  && echo "OK  recorder package imports" || { echo "FAIL recorder import"; fail=1; }
CAM="$("$REPO/.venv/bin/python" -c \
  'import sys; from robot_umi_core.config import load_rig_config; print(load_rig_config(sys.argv[1]).umis[0].camera.device)' \
  "$CONFIG_PATH" 2>/dev/null)"
if [ -n "$CAM" ] && [ -e "$CAM" ]; then echo "OK  camera present ($CAM)"; else echo "CHECK camera device ${CAM:-?} not found — verify $CONFIG_REL"; fi
for i in $IFACES; do
  systemctl is-active --quiet "umi-ptp@$i" && echo "OK  PTP master running on $i" \
    || echo "CHECK PTP not active on $i — journalctl -u umi-ptp@$i"
done
[ "$NEED_RELOGIN" = "1" ] && echo "ACTION log out/in (or reboot) for new group membership (GPIO/SPI/camera)."
grep -q '^dtparam=spi=on' "$BOOTCFG" 2>/dev/null && ! [ -e /dev/spidev0.0 ] && echo "ACTION reboot to activate SPI (LCD)."

echo
if [ "$fail" = "0" ]; then
  echo "install complete. umi-recorder is enabled; reboot to bring it up with SPI/groups:"
  echo "    sudo reboot"
else
  echo "install finished with FAILUREs above — fix them before recording."
  exit 1
fi
