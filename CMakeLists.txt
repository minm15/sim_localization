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
find_package(tf2_msgs REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)

add_executable(localization_node
  src/localization_node.cpp
  src/Particle_filter.cpp
  src/LinK3D_extractor.cpp
)
ament_target_dependencies(localization_node
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
  Eigen3
  cv_bridge
  OpenCV
)

target_link_libraries(localization_node
  ${PCL_LIBRARIES}
)

install(TARGETS localization_node
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY include/
  DESTINATION include/
)
install(DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

ament_package()