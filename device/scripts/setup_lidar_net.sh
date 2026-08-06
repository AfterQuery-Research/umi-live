#!/usr/bin/env bash
# Configure ONE Pi interface for ONE lidar. Run as root, once per lidar:
#   sudo device/scripts/setup_lidar_net.sh eth0 192.168.1.102/24 192.168.1.200
#
# Two silent failure modes this handles (both proven on hardware):
#   1. A network manager wipes a manual `ip addr add`, so we write a PERSISTENT
#      static IP via whichever stack the Pi runs — NetworkManager (nmcli) OR
#      netplan/systemd-networkd (Ubuntu Server default; no NetworkManager). No
#      default route is set, so internet stays on wlan0.
#   2. rp_filter drops the lidar's unicast on a multi-homed Pi. Note: pcap
#      capture (AF_PACKET) taps pre-routing and would see packets anyway, but
#      the advisory monitor uses UDP sockets, which do not.
set -eo pipefail

IFACE="${1:?usage: setup_lidar_net.sh <iface> <host_cidr> <lidar_ip>}"
HOST_CIDR="${2:?usage: setup_lidar_net.sh <iface> <host_cidr> <lidar_ip>}"
LIDAR_IP="${3:?usage: setup_lidar_net.sh <iface> <host_cidr> <lidar_ip>}"
CON="lidar-${IFACE}"

# --- persistent static IP: use whichever network stack this Pi actually runs ---
if command -v nmcli >/dev/null 2>&1 && systemctl is-active --quiet NetworkManager 2>/dev/null; then
  echo "=== persistent static IP on ${IFACE} via NetworkManager ==="
  nmcli con delete "${CON}" >/dev/null 2>&1 || true
  nmcli con add type ethernet ifname "${IFACE}" con-name "${CON}" \
    ipv4.method manual ipv4.addresses "${HOST_CIDR}" \
    ipv4.never-default yes ipv6.method disabled
  nmcli con up "${CON}"
elif command -v netplan >/dev/null 2>&1; then
  echo "=== persistent static IP on ${IFACE} via netplan (systemd-networkd) ==="
  # A higher-numbered file overrides cloud-init's dhcp on this iface. No gateway/route,
  # so ${IFACE} never becomes the default route (internet stays on wlan0).
  NP="/etc/netplan/99-lidar-${IFACE}.yaml"
  cat > "${NP}" <<EOF
network:
  version: 2
  ethernets:
    ${IFACE}:
      dhcp4: false
      dhcp6: false
      optional: true
      addresses: [${HOST_CIDR}]
EOF
  chmod 600 "${NP}"
  netplan generate
  netplan apply
else
  echo "ERROR: neither nmcli (NetworkManager) nor netplan found — cannot set a" >&2
  echo "persistent static IP on ${IFACE}. Install one, or set it by hand:" >&2
  echo "  ip addr add ${HOST_CIDR} dev ${IFACE} && ip link set ${IFACE} up" >&2
  exit 1
fi

echo "=== disable rp_filter (persist) ==="
cat > "/etc/sysctl.d/99-lidar-rpfilter-${IFACE}.conf" <<EOF
net.ipv4.conf.all.rp_filter=0
net.ipv4.conf.${IFACE}.rp_filter=0
EOF
sysctl -p "/etc/sysctl.d/99-lidar-rpfilter-${IFACE}.conf" >/dev/null

echo "=== verify ==="
sleep 1
ip addr show "${IFACE}" | grep "inet " || echo "WARN: no IPv4 on ${IFACE} yet (link up? carrier?)"
ping -c1 -W2 "${LIDAR_IP}" >/dev/null 2>&1 && echo "lidar ${LIDAR_IP} reachable" \
  || echo "note: lidar ${LIDAR_IP} did not answer ping (many lidars ignore ICMP — check with tcpdump)"
echo "quick data check: tcpdump -ni ${IFACE} udp port 6699 -c 3"
