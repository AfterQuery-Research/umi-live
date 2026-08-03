#!/usr/bin/env bash
# Resilient live feed: tap ONE Pi's lidar UDP and pipe it into that UMI's GLIM
# container's box_unpack (which UDPs it to loopback for rslidar -> GLIM).
#   feed.sh <user@pi-ip> <glim-container> [ports ...]     (default ports 6699 7788 6688)
# Loops so a container restart / SSH drop auto-reconnects. Uses the Mac's SSH key
# by default (add it to the Pi's authorized_keys); set FEED_SSH_PW=<pw> to use a
# password instead (older Pis on password auth).
set -u
PI="${1:?usage: feed.sh <user@pi-ip> <container> [ports...]}"
CTR="${2:?usage: feed.sh <user@pi-ip> <container> [ports...]}"
shift 2
PORTS="${*:-6699 7788 6688}"

SSH=(ssh -o ConnectTimeout=10 -o ServerAliveInterval=15 -o StrictHostKeyChecking=no)
if [ -n "${FEED_SSH_PW:-}" ]; then
  SSH=(sshpass -p "${FEED_SSH_PW}" ssh -o PreferredAuthentications=password \
       -o PubkeyAuthentication=no -o ConnectTimeout=10 -o ServerAliveInterval=15 \
       -o StrictHostKeyChecking=no)
fi

while true; do
  "${SSH[@]}" "$PI" "python3 ~/robot-umi/device/scripts/pi_tap.py $PORTS 2>/dev/null" \
    | docker exec -i "$CTR" python3 /box_unpack.py 2>/dev/null
  echo "[feed $CTR] dropped; reconnecting in 3s..." >&2
  sleep 3
done
