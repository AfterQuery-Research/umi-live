#!/usr/bin/env bash
# Clone + build the RoboSense rslidar_sdk ROS 2 driver (with IMU parsing ON)
# into ~/airy_ws. Run as the normal user AFTER ROS 2 Jazzy is installed.
#   bash 02_build_rslidar_sdk.sh
set -eo pipefail

WS="${WS:-$HOME/airy_ws}"
source /opt/ros/jazzy/setup.bash

mkdir -p "$WS/src"
cd "$WS/src"

# rslidar_sdk carries rs_driver as a submodule -> clone recursively.
if [[ ! -d rslidar_sdk ]]; then
  git clone --recurse-submodules https://github.com/RoboSense-LiDAR/rslidar_sdk
else
  git -C rslidar_sdk submodule update --init --recursive
fi

# rslidar_msg is a separate required package (find_package(rslidar_msg REQUIRED)).
if [[ ! -d rslidar_msg ]]; then
  git clone https://github.com/RoboSense-LiDAR/rslidar_msg
fi

# Jazzy fix: rslidar_sdk pins CMAKE_CXX_STANDARD 14, but Jazzy's rclcpp needs C++17
# (build fails with "is_same_v is not a member of std"). Bump it to 17.
sed -i 's/set(CMAKE_CXX_STANDARD 14)/set(CMAKE_CXX_STANDARD 17)/' \
  "$WS/src/rslidar_sdk/CMakeLists.txt"

cd "$WS"
# ENABLE_IMU_DATA_PARSE=ON -> driver publishes sensor_msgs/Imu from the Airy IMU.
colcon build --symlink-install \
  --cmake-args -DENABLE_IMU_DATA_PARSE=ON -DCMAKE_BUILD_TYPE=Release

echo
echo "=== build done. Source it with: ==="
echo "  source /opt/ros/jazzy/setup.bash && source $WS/install/setup.bash"
