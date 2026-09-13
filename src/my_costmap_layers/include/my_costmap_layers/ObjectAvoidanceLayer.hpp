#pragma once

#include <memory>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_array.hpp>
#include <nav2_costmap_2d/costmap_layer.hpp>
#include <rclcpp/rclcpp.hpp>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace my_costmap_layers
{

class ObjectAvoidanceLayer : public nav2_costmap_2d::CostmapLayer
{
public:
  ObjectAvoidanceLayer();

  void onInitialize() override;
  void updateBounds(
    double robot_x, double robot_y, double robot_yaw,
    double * min_x, double * min_y, double * max_x, double * max_y) override;
  void updateCosts(
    nav2_costmap_2d::Costmap2D & master_grid,
    int min_i, int min_j, int max_i, int max_j) override;
  void reset() override;
  bool isClearable() override {return true;}

private:
  void objectPositionsCallback(const geometry_msgs::msg::PoseArray::SharedPtr msg);

  std::string object_topic_name_{"/object_world_positions"};
  double avoidance_radius_{1.0};
  double bounds_padding_cells_{10.0};
  bool enabled_{true};

  std::vector<tf2::Vector3> detected_objects_;

  double last_min_x_{0.0};
  double last_min_y_{0.0};
  double last_max_x_{0.0};
  double last_max_y_{0.0};

  double prev_min_x_{0.0};
  double prev_min_y_{0.0};
  double prev_max_x_{0.0};
  double prev_max_y_{0.0};
  bool prev_valid_{false};
  bool updated_{false};

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr object_positions_sub_;
};

}  // namespace my_costmap_layers
