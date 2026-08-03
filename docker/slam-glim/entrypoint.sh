#!/usr/bin/env bash
# docker/slam-glim/entrypoint.sh — umi-slam stage CLI.
#
#   entrypoint.sh <bag_dir> <out_dir>
#     <bag_dir>  umi-decode output — either the stage dir containing bag/
#                (e.g. /data/derived/<id>/decoded_bag_v1) or the rosbag2 dir
#                itself (.../decoded_bag_v1/bag). Both are accepted.
#     <out_dir>  stage output, e.g. /data/derived/<id>/slam_glim_v1
#
# Runs GLIM's offline glim_rosbag over the decoded bag with our /glim_config
# overrides (rslidar topics, factory T_lidar_imu, headless module list —
# bags are SI-units, see docker/decode), dumps the optimized map to
# <out_dir>/glim_dump/, exports <out_dir>/trajectory.tum, touches COMPLETE.
#
# glim_rosbag reads the bag directly (no live topics), runs the full
# odometry + local/global mapping pipeline, then with auto_quit:=true saves
# the dump and exits — verified against glim_ros2 master source 2026-07-15.
# Needs on-GPU validation: CUDA kernels actually running (watch for GLIM
# falling back / erroring without a visible GPU), overall trajectory quality.

# no -u: ROS/colcon setup.bash reference unset vars (AMENT_TRACE_SETUP_FILES,
# COLCON_TRACE, ...) and are sourced repeatedly below.
set -eo pipefail

usage() { echo "usage: $0 <bag_dir> <out_dir> [scope]" >&2; exit 64; }
[[ $# -eq 2 || $# -eq 3 ]] || usage
BAG_ARG=$1
OUT_DIR=$2

# --- ROS + GLIM workspace (path recorded at image build, /etc/glim_ws_path) --
# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST
GLIM_WS=$(cat /etc/glim_ws_path 2>/dev/null || echo none)
if [[ "$GLIM_WS" != "none" && -f "$GLIM_WS/install/setup.bash" ]]; then
    # shellcheck disable=SC1091
    source "$GLIM_WS/install/setup.bash"
fi

# --- resolve the rosbag2 directory -------------------------------------------
if [[ -f "$BAG_ARG/metadata.yaml" ]]; then
    BAG="$BAG_ARG"
elif [[ -f "$BAG_ARG/bag/metadata.yaml" ]]; then
    BAG="$BAG_ARG/bag"
else
    echo "error: no rosbag2 (metadata.yaml) at $BAG_ARG or $BAG_ARG/bag" >&2
    exit 66
fi
if [[ ! -f "$(dirname "$BAG")/COMPLETE" && ! -f "$BAG_ARG/COMPLETE" ]]; then
    echo "warning: upstream decode stage has no COMPLETE marker — bag may be partial" >&2
fi
echo "slam: bag = $BAG"

DUMP="$OUT_DIR/glim_dump"
mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR/COMPLETE"
rm -rf "$DUMP"

# Per-lidar IMU extrinsic. The runner resolves THIS lidar's own T_lidar_imu (config
# override > its DIFOP packet > hardcoded default) and passes it as
# T_LIDAR_IMU="x y z qx qy qz qw". Splice it into a writable copy of the sensor config
# so a bimanual rig's two lidars each get their own factory calibration; the value baked
# into config_sensors.json is only the fallback for manual `docker run`s. Per-container
# /tmp (each `compose run --rm` is fresh) — no cross-run contamination.
GLIM_CONFIG=/glim_config
if [[ -n "${T_LIDAR_IMU:-}" ]]; then
    GLIM_CONFIG=/tmp/glim_config
    rm -rf "$GLIM_CONFIG"
    cp -r /glim_config "$GLIM_CONFIG"
    arr="[$(echo "$T_LIDAR_IMU" | tr ' ' ',')]"
    sed -i "s#\"T_lidar_imu\": \[[^]]*\]#\"T_lidar_imu\": $arr#" "$GLIM_CONFIG/config_sensors.json"
    echo "slam: T_lidar_imu = $arr"
fi

# config_path must be ABSOLUTE — relative paths resolve against GLIM's own
# share directory (glim_ros.cpp).
ros2 run glim_ros glim_rosbag "$BAG" --ros-args \
    -p config_path:="$GLIM_CONFIG" \
    -p auto_quit:=true \
    -p dump_path:="$DUMP"

# GLIM's dump already contains TUM-format trajectories (traj_lidar.txt is the
# globally-optimized one); export_tum.py picks, validates, sorts, and writes
# the canonical stage artifact.
python3 /export_tum.py "$DUMP" "$OUT_DIR/trajectory.tum"

# Cross-lidar alignment (opt-in): also export the GLIM map so the server-side align stage
# can register the two lidars into one frame. Gated so the non-joint pipeline is unchanged.
if [[ "${EXPORT_MAP:-0}" == "1" ]]; then
    echo "slam: EXPORT_MAP -> $OUT_DIR/map.ply + submaps/"
    [[ -f /overlay_ws/install/setup.bash ]] && source /overlay_ws/install/setup.bash
    ros2 run umi_map_export export_map "$DUMP" "$OUT_DIR"
fi

touch "$OUT_DIR/COMPLETE"
echo "slam: done -> $OUT_DIR/trajectory.tum"
