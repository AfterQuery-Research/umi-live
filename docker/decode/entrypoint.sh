#!/usr/bin/env bash
# docker/decode/entrypoint.sh — umi-decode stage CLI.
#
#   entrypoint.sh <episode_dir> <out_dir>
#     <episode_dir>  hermetic raw episode, e.g. /data/raw_episodes/<id>
#                    (contains manifest.json, umiN_lidar.pcap, ...)
#     <out_dir>      stage output,        e.g. /data/derived/<id>/decoded_bag_v1
#
# Replays the episode's pcap through rslidar_sdk (msg_source: 3) while
# `ros2 bag record` captures /rslidar_points + /rslidar_imu_data into
# <out_dir>/bag. On success touches <out_dir>/COMPLETE.
#
# Replay-end handling (pcap_repeat: false): with rs_driver the process does
# NOT reliably exit when the pcap is exhausted — the decode thread just goes
# quiet. We therefore treat EITHER of these as "replay finished":
#   a) the driver node exits on its own, or
#   b) the recorded bag stops growing for DECODE_IDLE_SECS (default 20s).
# A hard DECODE_MAX_SECS cap (default 7200s; episodes are minutes long at
# 1x replay) backstops both. Idle/timeout values need on-GPU-box validation.
#
# no -u: ROS/colcon setup.bash reference unset vars (AMENT_TRACE_SETUP_FILES,
# COLCON_TRACE, ...) and are sourced repeatedly below.
set -eo pipefail

usage() { echo "usage: $0 <episode_dir> <out_dir> [scope]" >&2; exit 64; }
[[ $# -eq 2 || $# -eq 3 ]] || usage
EPISODE_DIR=$1
OUT_DIR=$2
SCOPE=${3:-}

IDLE_SECS="${DECODE_IDLE_SECS:-20}"
MAX_SECS="${DECODE_MAX_SECS:-7200}"
START_TIMEOUT="${DECODE_START_TIMEOUT:-120}"

[[ -d "$EPISODE_DIR" ]] || { echo "error: episode dir not found: $EPISODE_DIR" >&2; exit 66; }

# --- locate the pcap ---------------------------------------------------------
shopt -s nullglob
if [[ -n "$SCOPE" ]]; then pcaps=("$EPISODE_DIR/${SCOPE}_lidar.pcap"); else pcaps=("$EPISODE_DIR"/umi*_lidar.pcap); fi
shopt -u nullglob
if [[ ${#pcaps[@]} -eq 0 ]]; then
    echo "error: no umiN_lidar.pcap in $EPISODE_DIR" >&2
    exit 66
fi
PCAP="${pcaps[0]}"
if [[ ${#pcaps[@]} -gt 1 ]]; then
    echo "error: decode scope resolved more than one lidar pcap" >&2
    exit 65
fi
echo "decode: pcap = $PCAP"

# --- generate per-run rslidar config from the template -----------------------
RUN_CFG=/tmp/rslidar_run.yaml
# '|' as sed delimiter: pcap paths contain '/'.
sed "s|@PCAP_PATH@|$PCAP|" /rslidar_pcap_template.yaml > "$RUN_CFG"

# --- ROS environment ----------------------------------------------------------
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
# shellcheck disable=SC1091
source /ws/install/setup.bash

BAG_DIR="$OUT_DIR/bag"
mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR/COMPLETE"
rm -rf "$BAG_DIR"   # rosbag2 refuses an existing -o dir; make reruns idempotent

NODE_PID=""
RECORD_PID=""
OUTPUT_OWNER=$(stat -c '%u:%g' "$OUT_DIR")
stop_group() {
    local pid=$1 grace=$2
    [[ -n "$pid" ]] || return 0
    kill -INT -- -"$pid" 2>/dev/null || true
    for ((i=0; i<grace; i++)); do
        kill -0 -- -"$pid" 2>/dev/null || return 0
        sleep 1
    done
    kill -KILL -- -"$pid" 2>/dev/null || true
}
shutdown() {
    stop_group "$NODE_PID" 20
    NODE_PID=""
    # Give rosbag2 time to flush metadata after the driver has stopped.
    stop_group "$RECORD_PID" 30
    RECORD_PID=""
}
cleanup() {
    local status=$?
    trap - EXIT INT TERM
    shutdown
    # ROS needs root in the image; retries on the host must still own the output.
    chown -R "$OUTPUT_OWNER" "$OUT_DIR" || true
    return "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Recorder first so no early frames are missed (it waits for the topics).
# -s mcap pins the storage format explicitly => "decoded_bag_v1" is stable
# even if a future ROS release changes the default.
# set -m: non-interactive shells start background jobs with SIGINT ignored
# (SIG_IGN inherits into the python ros2 CLI, which honors it — the recorder
# would never see our shutdown signal and metadata.yaml would never be
# written). Job control gives it its own process group + default dispositions.
set -m
ros2 bag record -s mcap -o "$BAG_DIR" /rslidar_points /rslidar_imu_data &
RECORD_PID=$!
set +m
sleep 3

set -m
ros2 run rslidar_sdk rslidar_sdk_node --ros-args -p config_path:="$RUN_CFG" &
NODE_PID=$!
set +m

# --- wait for replay to finish (see header for the strategy) ------------------
start=$(date +%s)
last_size=0
last_change=$start
got_data=0
while :; do
    sleep 2
    now=$(date +%s)
    if ! kill -0 "$NODE_PID" 2>/dev/null; then
        echo "decode: driver node exited — replay finished"
        break
    fi
    size=$(du -sb "$BAG_DIR" 2>/dev/null | cut -f1 || echo 0)
    if (( size > last_size )); then
        last_size=$size
        last_change=$now
        got_data=1
    fi
    if (( got_data && now - last_change >= IDLE_SECS )); then
        echo "decode: bag idle for ${IDLE_SECS}s — assuming pcap replay done"
        break
    fi
    if (( !got_data && now - start >= START_TIMEOUT )); then
        echo "error: no bag data after ${START_TIMEOUT}s — bad pcap or driver misconfig" >&2
        exit 1
    fi
    if (( now - start >= MAX_SECS )); then
        echo "error: hit DECODE_MAX_SECS=${MAX_SECS}; refusing partial decode" >&2
        exit 1
    fi
done

shutdown

# rosbag2 writes metadata.yaml only on clean close — treat it as the success probe.
if [[ ! -s "$BAG_DIR/metadata.yaml" ]]; then
    echo "error: $BAG_DIR/metadata.yaml missing — bag did not close cleanly" >&2
    exit 1
fi

# A marker is not proof of usable decode output: empty/truncated metadata from
# disk-full failures used to pass through to GLIM and retry forever.
python3 - "$BAG_DIR/metadata.yaml" <<'PY'
import sys
from pathlib import Path
import yaml

try:
    info = yaml.safe_load(Path(sys.argv[1]).read_text())["rosbag2_bagfile_information"]
    counts = {
        entry["topic_metadata"]["name"]: entry["message_count"]
        for entry in info["topics_with_message_count"]
    }
    if any(counts.get(topic, 0) <= 0 for topic in ("/rslidar_points", "/rslidar_imu_data")):
        raise ValueError("bag needs both LiDAR points and IMU messages")
except (OSError, TypeError, KeyError, ValueError, yaml.YAMLError) as exc:
    raise SystemExit(f"error: unusable decoded bag metadata: {exc}") from exc
PY

touch "$OUT_DIR/COMPLETE"
echo "decode: done -> $BAG_DIR ($(du -sh "$BAG_DIR" | cut -f1))"
