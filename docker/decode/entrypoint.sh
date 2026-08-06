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
cleanup() {
    # SIGINT (not KILL): rosbag2 finalizes metadata.yaml on SIGINT.
    [[ -n "$NODE_PID" ]] && kill -INT "$NODE_PID" 2>/dev/null || true
    [[ -n "$RECORD_PID" ]] && kill -INT "$RECORD_PID" 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

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

ros2 run rslidar_sdk rslidar_sdk_node --ros-args -p config_path:="$RUN_CFG" &
NODE_PID=$!

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
        echo "warning: hit DECODE_MAX_SECS=${MAX_SECS}; stopping" >&2
        break
    fi
done

# Orderly shutdown: driver first, then flush the recorder. SIGINT to the
# `ros2 run` wrapper does NOT reliably reach the real node — kill by name with
# bounded waits, then escalate (an unkillable node must not hang the stage).
stop_all() {
    local sig=$1
    kill -"$sig" "$NODE_PID" 2>/dev/null || true
    pkill -"$sig" -f "rslidar_sdk_node" 2>/dev/null || true
}
stop_all INT
for _ in $(seq 1 20); do pgrep -f "rslidar_sdk_node" >/dev/null || break; sleep 1; done
pgrep -f "rslidar_sdk_node" >/dev/null && stop_all KILL
NODE_PID=""
kill -INT -- -"$RECORD_PID" 2>/dev/null || kill -INT "$RECORD_PID" 2>/dev/null || true
for _ in $(seq 1 30); do kill -0 "$RECORD_PID" 2>/dev/null || break; sleep 1; done
kill -0 "$RECORD_PID" 2>/dev/null && kill -KILL -- -"$RECORD_PID" 2>/dev/null
wait "$RECORD_PID" 2>/dev/null || true
RECORD_PID=""
trap - EXIT

# rosbag2 writes metadata.yaml only on clean close — treat it as the success probe.
if [[ ! -f "$BAG_DIR/metadata.yaml" ]]; then
    echo "error: $BAG_DIR/metadata.yaml missing — bag did not close cleanly" >&2
    exit 1
fi

touch "$OUT_DIR/COMPLETE"
echo "decode: done -> $BAG_DIR ($(du -sh "$BAG_DIR" | cut -f1))"
