#!/usr/bin/env bash
# Runtime for the dedicated UMI-to-UMI Wi-Fi link.
#
# The leader creates a virtual AP while preserving wlan0 as its optional uplink.
# The follower temporarily dedicates its configured radio to the leader network. Static
# addresses keep the recorder control path independent of DHCP and mDNS.
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

RUNTIME_DIR=/run/umi-live-pair
HOSTAPD_CONFIG="$RUNTIME_DIR/hostapd.conf"
WPA_NETWORK_ID_FILE="$RUNTIME_DIR/follower-network-id"

interface_exists() {
  iw dev "$UMI_PAIR_IFACE" info >/dev/null 2>&1
}

ensure_interface() {
  if ! interface_exists; then
    if [ "$UMI_PAIR_ROLE" = leader ]; then
      iw dev "$UMI_PAIR_RADIO_IFACE" interface add "$UMI_PAIR_IFACE" type __ap
    else
      [ "$UMI_PAIR_IFACE" = "$UMI_PAIR_RADIO_IFACE" ] \
        || iw dev "$UMI_PAIR_RADIO_IFACE" interface add "$UMI_PAIR_IFACE" type managed
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

wait_for_follower_supplicant() {
  for _ in $(seq 1 30); do
    [ "$(wpa_cli -i "$UMI_PAIR_IFACE" ping 2>/dev/null | tail -1)" = PONG ] && return
    sleep 1
  done
  die "system wpa_supplicant did not become ready on $UMI_PAIR_IFACE"
}

configure_follower_network() {
  local network_id frequency result
  mkdir -p "$RUNTIME_DIR"
  wait_for_follower_supplicant
  [ "$(wpa_cli -i "$UMI_PAIR_IFACE" set ap_scan 2 | tail -1)" = OK ] \
    || die "could not enable direct follower association"
  network_id="$(wpa_cli -i "$UMI_PAIR_IFACE" add_network | tail -1)"
  case "$network_id" in *[!0-9]*|"") die "could not add follower Wi-Fi network" ;; esac
  printf '%s\n' "$network_id" > "$WPA_NETWORK_ID_FILE"
  if [ "$UMI_PAIR_CHANNEL" -eq 14 ]; then
    frequency=2484
  elif [ "$UMI_PAIR_CHANNEL" -le 13 ]; then
    frequency=$((2407 + UMI_PAIR_CHANNEL * 5))
  else
    frequency=$((5000 + UMI_PAIR_CHANNEL * 5))
  fi
  for setting in \
    "ssid \"$UMI_PAIR_SSID\"" \
    "psk \"$UMI_PAIR_PSK\"" \
    "key_mgmt WPA-PSK" \
    "scan_ssid 1" \
    "scan_freq $frequency"; do
    # setting is generated solely from the validated provisioning inventory.
    # shellcheck disable=SC2086
    result="$(wpa_cli -i "$UMI_PAIR_IFACE" set_network "$network_id" $setting | tail -1)"
    [ "$result" = OK ] || die "could not configure follower Wi-Fi network"
  done
  [ "$(wpa_cli -i "$UMI_PAIR_IFACE" select_network "$network_id" | tail -1)" = OK ] \
    || die "could not select follower Wi-Fi network"
}

configure_follower_ip() {
  ip -4 address flush dev "$UMI_PAIR_IFACE"
  ip -4 route flush dev "$UMI_PAIR_IFACE"
  ip address replace "$UMI_PAIR_FOLLOWER_IP/24" dev "$UMI_PAIR_IFACE"
  ip route replace default via "$UMI_PAIR_LEADER_IP" dev "$UMI_PAIR_IFACE" metric 900
  resolvectl dns "$UMI_PAIR_IFACE" 1.1.1.1 1.0.0.1
  resolvectl default-route "$UMI_PAIR_IFACE" yes
}

clear_follower_ip() {
  resolvectl revert "$UMI_PAIR_IFACE" 2>/dev/null || true
  ip route del default via "$UMI_PAIR_LEADER_IP" dev "$UMI_PAIR_IFACE" \
    metric 900 2>/dev/null || true
  ip address del "$UMI_PAIR_FOLLOWER_IP/24" dev "$UMI_PAIR_IFACE" 2>/dev/null || true
}

run_follower() {
  local configured=0 missed=0 state ssid status
  while true; do
    status="$(wpa_cli -i "$UMI_PAIR_IFACE" status 2>/dev/null || true)"
    state="$(printf '%s\n' "$status" | sed -n 's/^wpa_state=//p')"
    ssid="$(printf '%s\n' "$status" | sed -n 's/^ssid=//p')"
    if [ "$state" = COMPLETED ] && [ "$ssid" = "$UMI_PAIR_SSID" ]; then
      if [ "$configured" = 0 ]; then
        configure_follower_ip
        configured=1
      fi
      missed=0
    else
      configured=0
      missed=$((missed + 1))
      if [ "$missed" -ge 60 ]; then
        echo "pair-wifi: leader hotspot unavailable; restoring normal Wi-Fi for 60 seconds" >&2
        remove_follower_network
        clear_follower_ip
        sleep 60
        return 1
      fi
    fi
    sleep 1
  done
}

remove_follower_network() {
  local network_id=""
  if [ -f "$WPA_NETWORK_ID_FILE" ]; then
    network_id="$(cat "$WPA_NETWORK_ID_FILE")"
  fi
  case "$network_id" in
    *[!0-9]*|"") ;;
    *) wpa_cli -i "$UMI_PAIR_IFACE" remove_network "$network_id" >/dev/null 2>&1 || true ;;
  esac
  wpa_cli -i "$UMI_PAIR_IFACE" set ap_scan 1 >/dev/null 2>&1 || true
  wpa_cli -i "$UMI_PAIR_IFACE" reconfigure >/dev/null 2>&1 || true
}

case "$ACTION" in
  up)
    ensure_interface
    if [ "$UMI_PAIR_ROLE" = leader ]; then
      ip address replace "$UMI_PAIR_LEADER_IP/24" dev "$UMI_PAIR_IFACE"
      write_leader_config
      add_forwarding_rules
    else
      iw dev "$UMI_PAIR_IFACE" set power_save off
      configure_follower_network
    fi
    ;;
  run)
    if [ "$UMI_PAIR_ROLE" = leader ]; then
      exec hostapd "$HOSTAPD_CONFIG"
    else
      run_follower
    fi
    ;;
  down)
    if [ "$UMI_PAIR_ROLE" = leader ]; then
      remove_forwarding_rules
    else
      remove_follower_network
      clear_follower_ip
    fi
    if interface_exists && [ "$UMI_PAIR_IFACE" != "$UMI_PAIR_RADIO_IFACE" ]; then
      iw dev "$UMI_PAIR_IFACE" del
    fi
    rm -f "$HOSTAPD_CONFIG" "$WPA_NETWORK_ID_FILE"
    rmdir "$RUNTIME_DIR" 2>/dev/null || true
    ;;
  *) die "usage: pair_wifi_service.sh up|run|down" ;;
esac
