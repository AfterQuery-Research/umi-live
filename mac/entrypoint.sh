#!/bin/bash
# Two modes:
#
#   (default, live)  Combined decode+GLIM for the live viewer. box_unpack (fed via
#     `docker exec -i <ctr> python3 /box_unpack.py` with the Pi tap piped in) UDPs
#     lidar packets to 127.0.0.1 where rslidar_sdk listens; rslidar publishes
#     /rslidar_points + /rslidar_imu_data; glim_rosnode (CPU) subscribes and
#     publishes /glim_ros/lidar_odom_corrected. pose_sub is run separately via docker exec.
#
#   replay <pcap> <out_dir>   Offline SLAM for the Mac processing pipeline (mac_glim.py):
#     replay ONE episode's pcap through rslidar_sdk into a rosbag, run GLIM's offline
#     glim_rosbag over it (auto_quit), and export the optimized trajectory as
#     <out_dir>/trajectory.tum. Mirrors docker/slam-glim (amd64) but sources from a
#     pcap instead of a pre-decoded bag; CPU, no GPU.
source /opt/ros/jazzy/setup.bash
source /rslidar_ws/install/setup.bash
source /ws/install/setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
export ROS_DOMAIN_ID=0

if [ "${1:-live}" = "replay" ]; then
  set -eo pipefail
  PCAP="${2:?usage: entrypoint.sh replay <pcap> <out_dir>}"
  OUT="${3:?usage: entrypoint.sh replay <pcap> <out_dir>}"
  [ -f "$PCAP" ] || { echo "replay: pcap not found: $PCAP" >&2; exit 66; }
  BAG=/tmp/replay_bag
  DUMP=/tmp/glim_dump
  rm -rf "$BAG" "$DUMP"
  sed "s#@PCAP@#$PCAP#" /rslidar_replay.yaml > /tmp/rslidar_replay.yaml

  # 1. record the replayed lidar topics to a rosbag2. rslidar_sdk replays the pcap to EOF
  #    and prints "PcapExit" but does NOT self-exit the node, so wait for that marker.
  #    THE CRITICAL BIT is `set -m`: a non-interactive shell starts background jobs with
  #    SIGINT set to SIG_IGN, which the python `ros2` CLI inherits + honors — so a bare
  #    `kill -INT` never reaches the recorder and metadata.yaml is never written (glim then
  #    falls back to sqlite3 and aborts). Job control gives the recorder its own process
  #    group + default signal dispositions. -s mcap pins the storage format. (Mirrors the
  #    proven docker/decode stage.)
  echo "replay: pcap -> bag ($PCAP)"
  set -m
  ros2 bag record -s mcap -o "$BAG" /rslidar_points /rslidar_imu_data >/tmp/bag_record.log 2>&1 &
  REC=$!
  set +m
  sleep 3
  ros2 run rslidar_sdk rslidar_sdk_node --ros-args -p config_path:=/tmp/rslidar_replay.yaml \
    >/tmp/rslidar_replay.log 2>&1 &
  RS=$!
  REPLAY_TIMEOUT="${REPLAY_TIMEOUT:-900}"
  for _ in $(seq 1 "$REPLAY_TIMEOUT"); do
    grep -q "PcapExit" /tmp/rslidar_replay.log 2>/dev/null && break
    kill -0 "$RS" 2>/dev/null || break
    sleep 1
  done
  sleep 3   # let the final scans land in the bag
  # stop the driver (SIGINT to the ros2 run wrapper doesn't reach the node; kill by name).
  pkill -INT -f rslidar_sdk_node 2>/dev/null || true
  for _ in $(seq 1 15); do pgrep -f rslidar_sdk_node >/dev/null || break; sleep 1; done
  pkill -KILL -f rslidar_sdk_node 2>/dev/null || true
  # flush the recorder: SIGINT its process group (it has one now, via set -m), then WAIT
  # for metadata.yaml — rosbag2 writes it only on a clean close — escalating to KILL.
  kill -INT -- -"$REC" 2>/dev/null || kill -INT "$REC" 2>/dev/null || true
  for _ in $(seq 1 30); do [ -f "$BAG/metadata.yaml" ] && break; kill -0 "$REC" 2>/dev/null || break; sleep 1; done
  kill -KILL -- -"$REC" 2>/dev/null || kill -KILL "$REC" 2>/dev/null || true
  [ -f "$BAG/metadata.yaml" ] || { echo "replay: ERROR bag metadata.yaml not written (bag did not finalize)" >&2; exit 1; }

  # 2. GLIM offline over the bag (auto_quit dumps + exits), then export TUM.
  #    Per-lidar extrinsic: mac_glim.py resolves this lidar's own T_lidar_imu and passes
  #    it as T_LIDAR_IMU="x y z qx qy qz qw"; splice it into a writable config copy so each
  #    arm of a bimanual rig gets its own calibration (baked value is the fallback).
  GLIM_CONFIG=/glim_config
  if [ -n "${T_LIDAR_IMU:-}" ]; then
    GLIM_CONFIG=/tmp/glim_config
    rm -rf "$GLIM_CONFIG"
    cp -r /glim_config "$GLIM_CONFIG"
    arr="[$(echo "$T_LIDAR_IMU" | tr ' ' ',')]"
    sed -i "s#\"T_lidar_imu\": \[[^]]*\]#\"T_lidar_imu\": $arr#" "$GLIM_CONFIG/config_sensors.json"
    echo "replay: T_lidar_imu = $arr"
  fi
  echo "replay: GLIM over bag"
  ros2 run glim_ros glim_rosbag "$BAG" --ros-args \
    -p config_path:="$GLIM_CONFIG" -p auto_quit:=true -p dump_path:="$DUMP"
  mkdir -p "$OUT"
  python3 /export_tum.py "$DUMP" "$OUT/trajectory.tum"

  # Cross-lidar alignment (opt-in): also export the GLIM map so the server-side align
  # stage can register the two lidars into one frame. Gated so the non-joint pipeline is
  # byte-for-byte unchanged (trajectory.tum only).
  if [ "${EXPORT_MAP:-0}" = "1" ]; then
    echo "replay: EXPORT_MAP -> $OUT/map.ply + submaps/"
    ros2 run umi_map_export export_map "$DUMP" "$OUT"
  fi
  echo "replay: done -> $OUT/trajectory.tum"
  exit 0
fi

# --- default: live streaming ---
# glim_rosnode needs the ROS odom-publishing extension (librviz_viewer.so) so the live
# viewer's pose_sub can subscribe to /glim_ros/lidar_odom_corrected. The shared /glim_config
# is deliberately headless (offline glim_rosbag CRASHES with the viewer extension), so patch
# a LIVE-ONLY copy here that adds it back. (librviz_viewer only publishes ROS topics — no GUI
# — so it's headless-safe under the live node, unlike libstandard_viewer.so.)
LIVE_GLIM_CONFIG=/tmp/glim_config_live
rm -rf "$LIVE_GLIM_CONFIG"; cp -r /glim_config "$LIVE_GLIM_CONFIG"
sed -i 's#\("libmemory_monitor.so"\)#\1,\n        "librviz_viewer.so"#' "$LIVE_GLIM_CONFIG/config_ros.json"
ros2 run rslidar_sdk rslidar_sdk_node --ros-args -p config_path:=/rslidar_live.yaml &
sleep 4
exec ros2 run glim_ros glim_rosnode --ros-args -p config_path:="$LIVE_GLIM_CONFIG"
