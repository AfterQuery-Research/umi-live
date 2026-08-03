# mac/Dockerfile — live-streaming GLIM container for the Mac (arm64, CPU).
#
# koide3 publishes no arm64 GLIM image, so we build the whole stack from source
# on ros:jazzy-ros-base (which IS multi-arch): gtsam/iridescence from koide3's
# PPAs (arm64 debs), gtsam_points + GLIM + glim_ros2 from source (CUDA OFF), and
# rslidar_sdk (XYZIRT + IMU parse, accel scaled to SI). rslidar_sdk listens on
# loopback UDP (fed by box_unpack via the Pi tap); glim_rosnode subscribes and
# publishes /glim_ros/lidar_odom_corrected. Distinct from docker/slam-glim/
# (that's the amd64/GPU OFFLINE stage that consumes a decoded rosbag).
#
#   docker build -f mac/Dockerfile -t umi-glim-live-arm64 mac
#
# Reconstructed from the original image's build history (the /tmp Dockerfile was
# lost); verified by rebuilding.
FROM ros:jazzy-ros-base

ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-}
ENV MAKEFLAGS=-j6

# --- GLIM build deps ---------------------------------------------------------
RUN apt-get update && apt-get install --no-install-recommends -y \
      curl ca-certificates gpg-agent software-properties-common \
      git cmake build-essential wget unzip \
      libfmt-dev libspdlog-dev libopencv-dev libboost-all-dev \
      libmetis-dev libomp-dev libpng-dev libjpeg-dev libeigen3-dev libglm-dev libglfw3-dev \
      python3-colcon-common-extensions \
      ros-jazzy-cv-bridge ros-jazzy-image-transport \
      ros-jazzy-rosbag2 ros-jazzy-rosbag2-storage-mcap ros-jazzy-tf2-ros \
    && rm -rf /var/lib/apt/lists/*

# gtsam + iridescence from koide3's PPAs (they ship arm64 debs)
RUN add-apt-repository -y ppa:koide3/iridescence \
    && add-apt-repository -y ppa:koide3/gtsam \
    && apt-get update \
    && apt-get install -y libiridescence-dev libgtsam-no-tbb-dev \
    && rm -rf /var/lib/apt/lists/*

# gtsam_points from source (CPU). Pinned for reproducibility.
ARG GTSAM_POINTS_REF=9d32e7dbecf6015560d84b4901d6b0a6f483ec46
WORKDIR /root
RUN git clone https://github.com/koide3/gtsam_points \
    && git -C gtsam_points checkout ${GTSAM_POINTS_REF} \
    && cmake -S gtsam_points -B gtsam_points/build -DCMAKE_BUILD_TYPE=Release -DBUILD_WITH_CUDA=OFF \
    && cmake --build gtsam_points/build -j6 \
    && cmake --install gtsam_points/build \
    && ldconfig

# GLIM + glim_ros2 from source (CPU, viewer on, no -march=native for portability). Pinned.
ARG GLIM_REF=1007fecd3389b44a4c2f278035bd57ceba459fee
ARG GLIM_ROS2_REF=4a9e7a4cb084967c8525a1be529ad3ba2a118ae7
WORKDIR /ws/src
RUN git clone https://github.com/koide3/glim && git -C glim checkout ${GLIM_REF} \
    && git clone https://github.com/koide3/glim_ros2 && git -C glim_ros2 checkout ${GLIM_ROS2_REF}
WORKDIR /ws
RUN . /opt/ros/jazzy/setup.sh \
    && colcon build --executor sequential --event-handlers console_cohesion+ \
       --cmake-args -DBUILD_WITH_CUDA=OFF -DBUILD_WITH_VIEWER=ON \
                    -DBUILD_WITH_MARCH_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release

# --- our GLIM map exporter: offline dump -> PLY, for the cross-lidar align stage ---
# Small ament package linking the just-built glim + gtsam_points; built as an overlay
# in the same /ws (sourcing /ws/install so find_package(glim) resolves). Invoked by
# entrypoint.sh replay ONLY when EXPORT_MAP=1 (alignment enabled).
COPY map_export/ /ws/src/umi_map_export/
RUN . /opt/ros/jazzy/setup.sh && . /ws/install/setup.sh \
    && colcon build --packages-select umi_map_export \
       --event-handlers console_cohesion+ --cmake-args -DCMAKE_BUILD_TYPE=Release

# --- rslidar_sdk (RoboSense Airy: XYZIRT point type, C++17, IMU parse + SI accel) ---
ARG RSLIDAR_SDK_REF=v1.5.20
ARG RSLIDAR_MSG_REF=v1.5.10
RUN apt-get update && apt-get install -y --no-install-recommends \
      libpcap-dev libyaml-cpp-dev python3 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /rslidar_ws/src
RUN git clone --depth 1 --branch ${RSLIDAR_SDK_REF} --recurse-submodules --shallow-submodules \
      https://github.com/RoboSense-LiDAR/rslidar_sdk \
    && git clone --depth 1 --branch ${RSLIDAR_MSG_REF} https://github.com/RoboSense-LiDAR/rslidar_msg
RUN grep -q 'set(CMAKE_CXX_STANDARD 14)' rslidar_sdk/CMakeLists.txt \
    && sed -i 's/set(CMAKE_CXX_STANDARD 14)/set(CMAKE_CXX_STANDARD 17)/' rslidar_sdk/CMakeLists.txt \
    && grep -q 'set(POINT_TYPE XYZI)' rslidar_sdk/CMakeLists.txt \
    && sed -i 's/set(POINT_TYPE XYZI)/set(POINT_TYPE XYZIRT)/' rslidar_sdk/CMakeLists.txt
# The Airy IMU reports linear acceleration in g; rslidar_sdk publishes it raw.
# Scale to m/s^2 so GLIM's IMU integration (acc_scale=1.0) is correct.
RUN test "$(grep -c 'imu_msg.linear_acceleration.x = data->linear_acceleration_x;' rslidar_sdk/src/source/source_pointcloud_ros.hpp)" = 2 \
    && sed -i -e 's#imu_msg.linear_acceleration.\([xyz]\) = data->linear_acceleration_\([xyz]\);#imu_msg.linear_acceleration.\1 = data->linear_acceleration_\2 * 9.80665;#' \
        rslidar_sdk/src/source/source_pointcloud_ros.hpp
WORKDIR /rslidar_ws
RUN . /opt/ros/jazzy/setup.sh \
    && colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release -DENABLE_IMU_DATA_PARSE=ON

# --- our config + the tap/pose plumbing --------------------------------------
# glim_config/ is the pinned working GLIM config (CPU odometry + our sensor/ros
# overrides for the Airy). rslidar_live.yaml drives the decode; box_unpack demuxes
# the Pi tap to loopback; pose_sub streams odometry as JSON for the viewer.
COPY glim_config/ /glim_config/
COPY rslidar_live.yaml /rslidar_live.yaml
COPY box_unpack.py /box_unpack.py
COPY pose_sub.py /pose_sub.py
# Offline replay (entrypoint.sh replay) reuses the amd64 stage's TUM exporter and a
# pcap-source rslidar config; both are self-contained (stdlib python / yaml).
COPY export_tum.py /export_tum.py
COPY rslidar_replay.yaml /rslidar_replay.yaml
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
