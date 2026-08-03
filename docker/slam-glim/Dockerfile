# docker/slam-glim/Dockerfile — "umi-slam" stage-CLI image (GPU).
#
# Runs GLIM (koide3) offline over the decoded rosbag from umi-decode and
# exports the optimized trajectory as TUM. One-shot CLI:
#   docker compose run --rm umi-slam /data/derived/<id>/decoded_bag_v1 /data/derived/<id>/slam_glim_v1
#
# BASE IMAGE TAGS — verified against Docker Hub 2026-07-15:
#   koide3/glim_ros2 publishes: jazzy_cuda13.1, jazzy_cuda12.5, jazzy (CPU),
#   humble, humble_cuda12.2.
#   The plan's guess `jazzy_cuda12.2` DOES NOT EXIST — we pin the real
#   `jazzy_cuda12.5` (needs NVIDIA driver >= 555 on the host; Lambda images
#   ship newer). `jazzy_cuda13.1` is available for very new drivers.
#   CPU fallback (slam_mode=cpu): rebuild with
#     --build-arg BASE_IMAGE=koide3/glim_ros2:jazzy
#   The stock GLIM config we copy out of the image at build time is the one
#   matching that image's CPU/GPU build, so the fallback needs no config edits.
#
# Verified (from source, 2026-07-15): glim_rosbag declares `auto_quit` and
# `dump_path` params and always calls glim->save(dump_path) at end of bag;
# GLIM's save() writes traj_lidar.txt / traj_imu.txt already in TUM format
# (glim/src/glim/mapping/global_mapping.cpp write_tum_frame).
# Needs on-GPU validation: CUDA runtime inside the container, GLIM parameter
# tuning for the Airy, end-to-end trajectory quality.

ARG BASE_IMAGE=koide3/glim_ros2:jazzy_cuda12.5
FROM ${BASE_IMAGE}

# Bake the stock GLIM config out of the image into /glim_config, then overlay
# our two override files. Copying the image's own config keeps every
# module-selection key (CPU vs CUDA odometry, submapping, ...) consistent
# with the GLIM build inside this image; we replace only config_ros.json and
# config_sensors.json.
# The koide3 images build GLIM in a colcon workspace; discover it rather than
# hardcoding, and remember it for entrypoint.sh in /etc/glim_ws_path.
RUN bash -c 'set -e; \
    source /opt/ros/jazzy/setup.bash; \
    GLIM_WS=""; \
    for ws in /root/ros2_ws /ros2_ws /opt/glim_ws; do \
        [ -f "$ws/install/setup.bash" ] && GLIM_WS="$ws" && break; \
    done; \
    if [ -z "$GLIM_WS" ]; then \
        GLIM_WS=$(find / -maxdepth 4 -path "*/install/setup.bash" 2>/dev/null \
                  | head -n1 | sed "s|/install/setup.bash||"); \
    fi; \
    [ -n "$GLIM_WS" ] && source "$GLIM_WS/install/setup.bash"; \
    echo "${GLIM_WS:-none}" > /etc/glim_ws_path; \
    CFG="$(ros2 pkg prefix glim)/share/glim/config"; \
    [ -d "$CFG" ] || { echo "GLIM stock config not found" >&2; exit 1; }; \
    cp -r "$CFG" /glim_config'

# Our overrides (full-file replacements; see the files for the key diffs:
# rslidar topics, headless extension modules, T_lidar_imu from the Airy's
# factory DIFOP calibration).
COPY config/config_ros.json /glim_config/config_ros.json
COPY config/config_sensors.json /glim_config/config_sensors.json

COPY export_tum.py /export_tum.py

# --- our GLIM map exporter: offline dump -> PLY, for the cross-lidar align stage ---
# The koide3 base ships GLIM prebuilt in the workspace recorded at /etc/glim_ws_path; build
# our small ament overlay against it. Invoked by entrypoint.sh ONLY when EXPORT_MAP=1 (so
# the non-joint pipeline is unchanged). Keep map_export/ in sync with mac/map_export/.
# NOTE: unlike the arm64 mac/ image (GLIM built from source with a clean cmake config),
# this links the prebuilt base's glim/gtsam_points — build verification pending on a GPU host.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential cmake python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*
COPY map_export/ /overlay_ws/src/umi_map_export/
RUN bash -c 'set -e; \
    source /opt/ros/jazzy/setup.bash; \
    GLIM_WS=$(cat /etc/glim_ws_path 2>/dev/null || echo none); \
    [ "$GLIM_WS" != none ] && source "$GLIM_WS/install/setup.bash"; \
    cd /overlay_ws && colcon build --packages-select umi_map_export \
        --cmake-args -DCMAKE_BUILD_TYPE=Release'

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Loopback-only DDS: stage runs must not cross-talk with parallel containers.
ENV ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
    ROS_LOCALHOST_ONLY=1

ENTRYPOINT ["/entrypoint.sh"]
