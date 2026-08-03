#!/usr/bin/env bash
# Stand up live GLIM streaming for ONE lidar UMI on the Mac.
#   ./setup_glim_stream.sh <user@pi-ip> [ports ...]
#
# Builds the arm64 GLIM image if missing, runs a per-UMI container named
# glim-<pi-ip> (dashes), and starts the resilient feed (Pi tap -> box_unpack ->
# that container). Run once per UMI with distinct Pi IPs to stream several at
# once -- each gets its own container + feed, no collisions. Then view with
#   python3 mac/live_pose_plot.py --pi-ip <pi-ip>
set -eo pipefail

PI="${1:?usage: setup_glim_stream.sh <user@pi-ip> [ports...]}"
shift || true
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG=umi-glim-live-arm64
IP="${PI##*@}"
CTR="glim-${IP//./-}"

if ! docker image inspect "$IMG" >/dev/null 2>&1; then
  echo "=== building $IMG from mac/Dockerfile (one-time, ~1h) ==="
  docker build -f "$HERE/Dockerfile" -t "$IMG" "$HERE"
fi

echo "=== (re)start container ${CTR} ==="
docker rm -f "$CTR" >/dev/null 2>&1 || true
docker run -d --name "$CTR" "$IMG" >/dev/null
RENV='source /opt/ros/jazzy/setup.bash; source /ws/install/setup.bash; export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST;'
i=0
until docker exec "$CTR" bash -lc "$RENV ros2 topic list 2>/dev/null" | grep -q lidar_odom_corrected; do
  i=$((i+1)); [ $i -ge 30 ] && { echo "WARN: GLIM odom topic not up yet"; break; }
  sleep 1
done
echo "GLIM ready in ${CTR}"

echo "=== start feed (Pi ${IP} tap -> ${CTR}) ==="
pkill -f "feed.sh ${PI} ${CTR}" 2>/dev/null || true
nohup "${HERE}/feed.sh" "$PI" "$CTR" "$@" >"/tmp/feed-${CTR}.log" 2>&1 &
disown
echo "feed running (log: /tmp/feed-${CTR}.log)"
echo
echo "DONE. View this UMI:"
echo "    python3 ${HERE}/live_pose_plot.py --pi-ip ${IP}"
