cmake_minimum_required(VERSION 3.16)
project(umi_map_export)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(GTSAM REQUIRED)
find_package(gtsam_points REQUIRED)
find_package(glim REQUIRED)

add_executable(export_map src/export_map.cpp)
target_include_directories(export_map PRIVATE ${EIGEN3_INCLUDE_DIRS})
# GTSAM / Eigen are pulled in transitively via the exported glim/gtsam_points targets.
target_link_libraries(export_map glim::glim gtsam_points::gtsam_points)

install(TARGETS export_map DESTINATION lib/${PROJECT_NAME})

ament_package()
