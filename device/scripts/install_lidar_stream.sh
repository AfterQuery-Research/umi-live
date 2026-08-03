#!/usr/bin/env bash
# One-file Pi setup for a lidar-STREAMING UMI (ROS-free tap; GLIM runs on the Mac).
#
# Brings a freshly-flashed Ubuntu Pi to a state where the Mac can tap its lidar
# and run GLIM remotely. Unlike install_pi.sh (the offline RECORDER: camera + pcap
# to disk), this sets up ONLY what the live stream needs: the lidar network, PTP,
# and the raw-UDP tap (pi_tap.py, driven from the Mac -- see mac/feed.sh).
#
# Run as root on the Pi (from the repo checkout):
#   sudo device/scripts/install_lidar_stream.sh [iface] [host_cidr] [lidar_ip]
# Defaults target a single RoboSense Airy: eth0 / 192.168.1.102/24 / 192.168.1.200
set -eo pipefail

IFACE="${1:-eth0}"
HOST_CIDR="${2:-192.168.1.102/24}"
LIDAR_IP="${3:-192.168.1.200}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ "$(id -u)" -ne 0 ]; then
  echo "run as root: sudo $0 $*" >&2
  exit 1
fi

echo "=== [1/3] lidar network: ${IFACE}=${HOST_CIDR}, lidar=${LIDAR_IP} ==="
# persistent static IP (survives NetworkManager) + rp_filter=0 (unicast delivery)
"${REPO}/device/scripts/setup_lidar_net.sh" "$IFACE" "$HOST_CIDR" "$LIDAR_IP"

echo "=== [2/3] PTP master on ${IFACE} (time-sync for GLIM) ==="
"${REPO}/device/scripts/setup_ptp.sh" "$IFACE"

echo "=== [3/3] tap self-check (can it bind the lidar ports?) ==="
if python3 "${REPO}/device/scripts/pi_tap.py" 6699 7788 6688 >/dev/null 2>/tmp/pi_tap_check.err &
then
  TAP=$!; sleep 2
  if kill -0 "$TAP" 2>/dev/null; then
    kill "$TAP" 2>/dev/null; echo "OK  pi_tap binds 6699/7788/6688"
  else
    echo "FAIL pi_tap exited -- ports busy? see /tmp/pi_tap_check.err"; cat /tmp/pi_tap_check.err
  fi
fi

echo
echo "DONE. This Pi streams when the Mac runs (from mac/):"
echo "    ./setup_glim_stream.sh <this-pi-ip> <umi-name>"
echo "then view with:  python3 mac/live_pose_plot.py --pi-ip <this-pi-ip>"
