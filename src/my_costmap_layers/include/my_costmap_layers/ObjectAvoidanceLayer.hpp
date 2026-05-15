#pragma once
#include <string>
#include <vector>
#include <unordered_map>

#include <rclcpp/rclcpp.hpp>
#include <nav2_costmap_2d/costmap_layer.hpp>
#include <nav2_costmap_2d/costmap_2d.hpp>
#include <nav2_costmap_2d/cost_values.hpp>
#include <geometry_msgs/msg/pose_array.hpp>
#include <tf2/LinearMath/Vector3.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <geometry_msgs/msg/pose_stamped.hpp>

namespace my_costmap_layers {

class ObjectAvoidanceLayer : public nav2_costmap_2d::CostmapLayer {
public:
  ObjectAvoidanceLayer();

  void onInitialize() override;
  void updateBounds(double, double, double, double*, double*, double*, double*) override;
  void updateCosts(nav2_costmap_2d::Costmap2D&, int, int, int, int) override;
  void reset() override;
  bool isClearable() override { return true; }

private:
  void objectPositionsCallback(const geometry_msgs::msg::PoseArray::SharedPtr);

  std::string object_topic_name_;
  double avoidance_radius_{0.5};
  bool   enabled_{true};
  double bounds_padding_cells_{2.0};
  double hold_after_clear_s_{0.6};
  double decay_ttl_s_{0.6};
  double decay_step_s_{0.1};

  double last_min_x_{0.0}, last_min_y_{0.0}, last_max_x_{0.0}, last_max_y_{0.0};

  double prev_min_x_{0.0}, prev_min_y_{0.0}, prev_max_x_{0.0}, prev_max_y_{0.0};
  bool   prev_valid_{false};

  bool updated_{false};

  std::vector<tf2::Vector3> detected_objects_;

  std::vector<tf2::Vector3> last_objects_cache_;
  rclcpp::Time              last_seen_stamp_{0, 0, RCL_ROS_TIME};

  std::vector<float> ttl_grid_;
  rclcpp::Time       last_decay_tick_{0, 0, RCL_ROS_TIME};

  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;

  rclcpp::Subscription<geometry_msgs::msg::PoseArray>::SharedPtr object_positions_sub_;

  inline size_t idx(unsigned int x, unsigned int y) const {
    return static_cast<size_t>(y) * getSizeInCellsX() + static_cast<size_t>(x);
  }
  void ensureTtlGridSized();
  void decayTtl(double dt);
  void stampDiskTtl(unsigned int cx, unsigned int cy, int r);
  void expandRoiToLiveTtl();
};

}

