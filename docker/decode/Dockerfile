# docker/decode/Dockerfile — "umi-decode" stage-CLI image (CPU).
#
# Replays a raw episode's umiN_lidar.pcap through the RoboSense rslidar_sdk
# ROS 2 driver (PCAP mode, msg_source: 3) and records the decoded
# /rslidar_points + /rslidar_imu_data into a rosbag2 — the "decoded_bag_v1"
# stage output consumed by umi-slam. One-shot CLI, invoked by the pipeline
# runner as:
#   docker compose run --rm umi-decode /data/raw_episodes/<id> /data/derived/<id>/decoded_bag_v1
#
# Verified (2026-07-15, without building — no builder on this machine):
#   - rslidar_sdk v1.5.20 CMakeLists.txt contains BOTH sed target lines
#     (`set(CMAKE_CXX_STANDARD 14)` at L139, `set(POINT_TYPE XYZI)` at L8);
#     grep guards below fail the build loudly if upstream moves them.
#   - rslidar_sdk_node declares a `config_path` ROS parameter, so we can feed
#     the per-run generated YAML without touching the package's config dir.
#   - v1.5.20 sample config documents pcap_path/pcap_repeat/pcap_rate under
#     lidar[].driver and lists RSAIRY with imu_port support.
# Needs on-GPU-box validation: full colcon build, replay-end idle detection
# timing in entrypoint.sh, recorded bag completeness (QoS/drops).
#
# Pinned tag verified on Docker Hub 2026-07-15 (jazzy-ros-base on Ubuntu
# 24.04 Noble). ros_base includes rosbag2; the explicit apt packages below
# are belt-and-braces + self-documentation.
FROM ros:jazzy-ros-base-noble

# Upstream pins (latest release tags as of 2026-07-15).
ARG RSLIDAR_SDK_REF=v1.5.20
ARG RSLIDAR_MSG_REF=v1.5.10

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        cmake \
        libpcap-dev \
        libyaml-cpp-dev \
        python3-colcon-common-extensions \
        ros-jazzy-rosbag2 \
        ros-jazzy-rosbag2-storage-mcap \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws/src

# rslidar_sdk carries rs_driver as a git submodule -> clone recursively.
# rslidar_msg is a separate required package (find_package(rslidar_msg REQUIRED)).
RUN git clone --depth 1 --branch ${RSLIDAR_SDK_REF} \
        --recurse-submodules --shallow-submodules \
        https://github.com/RoboSense-LiDAR/rslidar_sdk \
    && git clone --depth 1 --branch ${RSLIDAR_MSG_REF} \
        https://github.com/RoboSense-LiDAR/rslidar_msg

# Our two known build fixes (same ones we apply on the Pis, see
# _vendor/02_build_rslidar_sdk.sh):
#  1. Jazzy's rclcpp needs C++17; upstream pins C++14 (build fails with
#     "is_same_v is not a member of std").
#  2. XYZIRT point type: per-point ring + timestamp fields are required for
#     GLIM/SLAM deskewing; upstream default is bare XYZI.
# grep guards make the build FAIL if a future tag changes these lines instead
# of silently no-op'ing the seds.
RUN grep -q 'set(CMAKE_CXX_STANDARD 14)' rslidar_sdk/CMakeLists.txt \
    && sed -i 's/set(CMAKE_CXX_STANDARD 14)/set(CMAKE_CXX_STANDARD 17)/' \
        rslidar_sdk/CMakeLists.txt \
    && grep -q 'set(POINT_TYPE XYZI)' rslidar_sdk/CMakeLists.txt \
    && sed -i 's/set(POINT_TYPE XYZI)/set(POINT_TYPE XYZIRT)/' \
        rslidar_sdk/CMakeLists.txt

#  NOTE on gyro scale: the driver's 250-dps FSR base is CORRECT for the Airy.
#  Hardware-verified 2026-07-16 with a desk-corner-enforced exact 90 deg
#  rotation: 250-base integrates to 87 deg (right), a patched 500-base to
#  174 deg (wrong). Do not "fix" this to 500 — human angle estimates of
#  hand rotations run ~2x generous and once falsely validated a 500 patch.

#  3. rs_driver decodes accel in g, but the ROS layer copies it verbatim into
#     sensor_msgs/Imu, whose spec (and GLIM's preintegration) is m/s^2.
#     Hardware-confirmed 2026-07-16: /rslidar_imu_data reads |a| ~= 1.0 at
#     rest. GLIM then has to explain away ~8.8 m/s^2 as accel bias, wrecking
#     attitude estimation. Scale to SI at the two publish sites (ROS1 + ROS2
#     paths in source_pointcloud_ros.hpp).
RUN test "$(grep -c 'imu_msg.linear_acceleration.x = data->linear_acceleration_x;' \
        rslidar_sdk/src/source/source_pointcloud_ros.hpp)" = 2 \
    && sed -i \
        -e 's/imu_msg.linear_acceleration.\([xyz]\) = data->linear_acceleration_\([xyz]\);/imu_msg.linear_acceleration.\1 = data->linear_acceleration_\2 * 9.80665;  \/\/ umi fix: g -> m\/s^2 (sensor_msgs\/Imu spec)/' \
        rslidar_sdk/src/source/source_pointcloud_ros.hpp

WORKDIR /ws

# ENABLE_IMU_DATA_PARSE=ON -> driver publishes sensor_msgs/Imu from the Airy
# built-in IMU (required; GLIM is LiDAR-inertial).
RUN bash -c 'source /opt/ros/jazzy/setup.bash \
    && colcon build --event-handlers console_direct+ \
        --cmake-args -DCMAKE_BUILD_TYPE=Release -DENABLE_IMU_DATA_PARSE=ON'

# Confine DDS discovery to this container's loopback so parallel stage runs
# on the same compose network can't cross-talk topics.
# (ROS_LOCALHOST_ONLY is the pre-Jazzy spelling, kept for safety.)
ENV ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST \
    ROS_LOCALHOST_ONLY=1

COPY rslidar_pcap_template.yaml /rslidar_pcap_template.yaml
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
