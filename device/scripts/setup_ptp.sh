#!/usr/bin/env bash
# Install linuxptp + the per-interface PTP units and start them for each lidar
# interface. Run as root:
#   sudo device/scripts/setup_ptp.sh eth0 [eth1 ...]
set -eo pipefail

if [ "$#" -lt 1 ]; then
  echo "usage: setup_ptp.sh <iface> [iface ...]" >&2
  exit 1
fi

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "=== install linuxptp + ethtool ==="
apt-get install -y linuxptp ethtool

echo "=== install gPTP config + systemd units ==="
mkdir -p /etc/linuxptp
install -m 0644 "${REPO}/config/ptp/umi-gptp.cfg" /etc/linuxptp/
install -m 0644 "${REPO}/config/systemd/umi-ptp@.service" /etc/systemd/system/
install -m 0644 "${REPO}/config/systemd/umi-phc2sys@.service" /etc/systemd/system/
systemctl daemon-reload

for IFACE in "$@"; do
  echo "=== enable PTP master on ${IFACE} ==="
  systemctl enable --now "umi-ptp@${IFACE}"
  systemctl enable --now "umi-phc2sys@${IFACE}" || true  # no-op without a PHC
  echo "--- ${IFACE} timestamping capabilities ---"
  ethtool -T "${IFACE}" | sed -n '1,8p'
done

echo "verify: journalctl -fu umi-ptp@eth0   (expect port state MASTER + peer delay)"
