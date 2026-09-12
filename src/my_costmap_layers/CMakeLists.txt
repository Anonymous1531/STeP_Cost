cmake_minimum_required(VERSION 3.8)
project(my_costmap_layers)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(nav2_util REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

add_library(my_costmap_layers_library SHARED
  src/ObjectAvoidanceLayer.cpp
)

target_include_directories(my_costmap_layers_library PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(my_costmap_layers_library
  geometry_msgs
  nav2_costmap_2d
  nav2_util
  pluginlib
  rclcpp
  tf2
  tf2_geometry_msgs
  tf2_ros
)

pluginlib_export_plugin_description_file(
  nav2_costmap_2d
  plugins/costmap_plugins.xml
)

install(
  TARGETS my_costmap_layers_library
  EXPORT my_costmap_layers_library
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

install(
  FILES plugins/costmap_plugins.xml
  DESTINATION share/${PROJECT_NAME}/plugins
)

ament_export_targets(my_costmap_layers_library HAS_LIBRARY_TARGET)
ament_export_dependencies(
  geometry_msgs
  nav2_costmap_2d
  nav2_util
  pluginlib
  rclcpp
  tf2
  tf2_geometry_msgs
  tf2_ros
)

ament_package()
