cmake_minimum_required(VERSION 3.5)
project(sim_local)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(message_filters REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

# ── Build a single "localization_lib" that contains:
#      - nuscenes_local
#      - particle_filter
#      - LinK3D_extractor
#      - nuscenes_map (the class)
add_library(localization_lib
  src/nuscenes_local.cpp
  src/nclt_local.cpp
  src/Particle_filter.cpp
  src/LinK3D_extractor.cpp
  src/nuscenes_map.cpp
  src/nclt_map.cpp
  src/frame_dumper.cpp
)
ament_target_dependencies(localization_lib
  rclcpp
  sensor_msgs
  nav_msgs
  geometry_msgs
  message_filters
  pcl_conversions
  PCL
  tf2
  tf2_ros
  tf2_msgs
  tf2_geometry_msgs
  tf2_sensor_msgs
  cv_bridge
  OpenCV
  Eigen3
)

# ── Localization entry‐point 
add_executable(localization_main src/localizationMain.cpp)
ament_target_dependencies(localization_main
  rclcpp
)
target_link_libraries(localization_main
  localization_lib
  ${PCL_LIBRARIES}
)

# ── Map generator entry‐point
add_executable(map_main src/mapMain.cpp)
ament_target_dependencies(map_main
  rclcpp
)
target_link_libraries(map_main
  localization_lib
  ${PCL_LIBRARIES}
)

# ── NCLT Frame generator
add_executable(frame_dumper src/frame_dumper.cpp)
ament_target_dependencies(frame_dumper
  rclcpp
)
target_link_libraries(frame_dumper
  localization_lib
  ${PCL_LIBRARIES}
)

install(
  TARGETS
    localization_lib
    localization_main
    map_main
    frame_dumper
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/ DESTINATION include/)

install(DIRECTORY launch/ DESTINATION share/${PROJECT_NAME}/launch)

ament_package()
