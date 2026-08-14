#!/usr/bin/env bash
# Runtime for the dedicated UMI-to-UMI Wi-Fi link.
#
# The leader creates a virtual AP while preserving wlan0 as its optional uplink.
# The follower creates a second managed interface, leaving its normal wlan0 connection
# untouched. Static addresses keep the recorder control path independent of DHCP/mDNS.
set -euo pipefail

die() { echo "pair-wifi: $*" >&2; exit 1; }
need() { [ -n "${!1:-}" ] || die "$1 is unset"; }

[ "$(id -u)" -eq 0 ] || die "must run as root"
[ "$#" -eq 1 ] || die "usage: pair_wifi_service.sh up|run|down"
ACTION="$1"
for name in UMI_PAIR_ROLE UMI_PAIR_RADIO_IFACE UMI_PAIR_IFACE UMI_PAIR_SSID \
  UMI_PAIR_PSK UMI_PAIR_CHANNEL UMI_PAIR_COUNTRY UMI_PAIR_LEADER_IP \
  UMI_PAIR_FOLLOWER_IP UMI_PAIR_SUBNET UMI_PAIR_UPLINK_IFACE; do
  need "$name"
done
case "$UMI_PAIR_ROLE" in leader|follower) ;; *) die "invalid role $UMI_PAIR_ROLE" ;; esac

RUNTIME_DIR=/run/umi-flow-pair
HOSTAPD_CONFIG="$RUNTIME_DIR/hostapd.conf"
WPA_CONFIG="$RUNTIME_DIR/wpa_supplicant.conf"

interface_exists() {
  iw dev "$UMI_PAIR_IFACE" info >/dev/null 2>&1
}

ensure_interface() {
  if ! interface_exists; then
    if [ "$UMI_PAIR_ROLE" = leader ]; then
      iw dev "$UMI_PAIR_RADIO_IFACE" interface add "$UMI_PAIR_IFACE" type __ap
    else
      iw dev "$UMI_PAIR_RADIO_IFACE" interface add "$UMI_PAIR_IFACE" type managed
    fi
  fi
  ip link set "$UMI_PAIR_IFACE" up
}

add_forwarding_rules() {
  sysctl -q -w net.ipv4.ip_forward=1
  iptables -w -t nat -C POSTROUTING -s "$UMI_PAIR_SUBNET" \
    -o "$UMI_PAIR_UPLINK_IFACE" -j MASQUERADE 2>/dev/null \
    || iptables -w -t nat -A POSTROUTING -s "$UMI_PAIR_SUBNET" \
      -o "$UMI_PAIR_UPLINK_IFACE" -j MASQUERADE
  iptables -w -C FORWARD -i "$UMI_PAIR_IFACE" -o "$UMI_PAIR_UPLINK_IFACE" \
    -j ACCEPT 2>/dev/null \
    || iptables -w -A FORWARD -i "$UMI_PAIR_IFACE" -o "$UMI_PAIR_UPLINK_IFACE" \
      -j ACCEPT
  iptables -w -C FORWARD -i "$UMI_PAIR_UPLINK_IFACE" -o "$UMI_PAIR_IFACE" \
    -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT 2>/dev/null \
    || iptables -w -A FORWARD -i "$UMI_PAIR_UPLINK_IFACE" -o "$UMI_PAIR_IFACE" \
      -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT
}

remove_forwarding_rules() {
  iptables -w -t nat -D POSTROUTING -s "$UMI_PAIR_SUBNET" \
    -o "$UMI_PAIR_UPLINK_IFACE" -j MASQUERADE 2>/dev/null || true
  iptables -w -D FORWARD -i "$UMI_PAIR_IFACE" -o "$UMI_PAIR_UPLINK_IFACE" \
    -j ACCEPT 2>/dev/null || true
  iptables -w -D FORWARD -i "$UMI_PAIR_UPLINK_IFACE" -o "$UMI_PAIR_IFACE" \
    -m conntrack --ctstate RELATED,ESTABLISHED -j ACCEPT 2>/dev/null || true
}

write_leader_config() {
  local mode=g
  [ "$UMI_PAIR_CHANNEL" -le 14 ] || mode=a
  umask 077
  mkdir -p "$RUNTIME_DIR"
  {
    printf 'interface=%s\n' "$UMI_PAIR_IFACE"
    printf 'driver=nl80211\n'
    printf 'country_code=%s\n' "$UMI_PAIR_COUNTRY"
    printf 'ssid=%s\n' "$UMI_PAIR_SSID"
    printf 'hw_mode=%s\n' "$mode"
    printf 'channel=%s\n' "$UMI_PAIR_CHANNEL"
    printf 'ieee80211n=1\nwmm_enabled=1\nauth_algs=1\n'
    printf 'wpa=2\nwpa_key_mgmt=WPA-PSK\nrsn_pairwise=CCMP\n'
    printf 'wpa_passphrase=%s\n' "$UMI_PAIR_PSK"
  } > "$HOSTAPD_CONFIG"
}

write_follower_config() {
  umask 077
  mkdir -p "$RUNTIME_DIR"
  {
    printf 'ctrl_interface=/run/wpa_supplicant\n'
    printf 'country=%s\nap_scan=1\n' "$UMI_PAIR_COUNTRY"
    # wlan0 already owns the radio's single scheduled-scan plan and P2P device.
    # Force ordinary periodic scans on this second managed interface instead.
    printf 'disable_scan_offload=1\np2p_disabled=1\nautoscan=periodic:5\n'
    printf 'network={\n'
    printf '  ssid="%s"\n' "$UMI_PAIR_SSID"
    printf '  psk="%s"\n' "$UMI_PAIR_PSK"
    printf '  key_mgmt=WPA-PSK\n  scan_ssid=1\n'
    printf '}\n'
  } > "$WPA_CONFIG"
}

case "$ACTION" in
  up)
    ensure_interface
    if [ "$UMI_PAIR_ROLE" = leader ]; then
      ip address replace "$UMI_PAIR_LEADER_IP/24" dev "$UMI_PAIR_IFACE"
      write_leader_config
      add_forwarding_rules
    else
      ip address replace "$UMI_PAIR_FOLLOWER_IP/24" dev "$UMI_PAIR_IFACE"
      ip route replace default via "$UMI_PAIR_LEADER_IP" dev "$UMI_PAIR_IFACE" metric 900
      write_follower_config
    fi
    ;;
  run)
    if [ "$UMI_PAIR_ROLE" = leader ]; then
      exec hostapd "$HOSTAPD_CONFIG"
    else
      exec wpa_supplicant -D nl80211 -i "$UMI_PAIR_IFACE" -c "$WPA_CONFIG"
    fi
    ;;
  down)
    if [ "$UMI_PAIR_ROLE" = leader ]; then
      remove_forwarding_rules
    else
      ip route del default via "$UMI_PAIR_LEADER_IP" dev "$UMI_PAIR_IFACE" \
        metric 900 2>/dev/null || true
    fi
    if interface_exists; then
      iw dev "$UMI_PAIR_IFACE" del
    fi
    rm -f "$HOSTAPD_CONFIG" "$WPA_CONFIG"
    rmdir "$RUNTIME_DIR" 2>/dev/null || true
    ;;
  *) die "usage: pair_wifi_service.sh up|run|down" ;;
esac
