cmake_minimum_required(VERSION 3.8)
project(my_costmap_layers)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(nav2_costmap_2d REQUIRED)
find_package(rclcpp REQUIRED)
find_package(pluginlib REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(nav2_util REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

include_directories(include)

set(PLUGIN_SRCS 
  src/KeepoutCommandLayer.cpp
  src/ObjectAvoidanceLayer.cpp
  src/ZoneSoftCostLayer.cpp
)

add_library(my_costmap_layers_library SHARED ${PLUGIN_SRCS})

ament_target_dependencies(my_costmap_layers_library 
  rclcpp 
  nav2_costmap_2d 
  pluginlib 
  geometry_msgs
  std_msgs
  tf2
  nav2_util
)

install(FILES plugins/costmap_plugins.xml
  DESTINATION share/${PROJECT_NAME}/plugins
)

install(DIRECTORY include/
  DESTINATION include
)

install(TARGETS my_costmap_layers_library
  EXPORT my_costmap_layers_library
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

pluginlib_export_plugin_description_file(
  nav2_costmap_2d
  plugins/costmap_plugins.xml
)


ament_export_targets(my_costmap_layers_library HAS_LIBRARY_TARGET)
ament_export_dependencies(
  nav2_costmap_2d
  rclcpp
  pluginlib
  geometry_msgs
  std_msgs
  tf2
  nav2_util 
  tf2_ros
  tf2_geometry_msgs
)

ament_package()
