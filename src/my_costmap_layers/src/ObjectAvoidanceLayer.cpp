#include "my_costmap_layers/ObjectAvoidanceLayer.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav2_costmap_2d/cost_values.hpp>
#include <nav2_costmap_2d/layered_costmap.hpp>
#include <nav2_util/node_utils.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace my_costmap_layers
{

ObjectAvoidanceLayer::ObjectAvoidanceLayer() = default;

void ObjectAvoidanceLayer::onInitialize()
{
  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error("Failed to lock node in ObjectAvoidanceLayer::onInitialize");
  }

  nav2_util::declare_parameter_if_not_declared(
    node, name_ + ".object_positions_topic", rclcpp::ParameterValue("/object_world_positions"));
  nav2_util::declare_parameter_if_not_declared(
    node, name_ + ".avoidance_radius", rclcpp::ParameterValue(2.0));
  nav2_util::declare_parameter_if_not_declared(
    node, name_ + ".bounds_padding_cells", rclcpp::ParameterValue(10.0));
  nav2_util::declare_parameter_if_not_declared(
    node, name_ + ".enabled", rclcpp::ParameterValue(true));

  // Backward-compatible parameters from the earlier layer implementation.
  // TTL lifetime is intentionally managed by policy_bridge, not by this C++ layer.
  nav2_util::declare_parameter_if_not_declared(
    node, name_ + ".hold_after_clear_s", rclcpp::ParameterValue(0.0));
  nav2_util::declare_parameter_if_not_declared(
    node, name_ + ".decay_ttl_s", rclcpp::ParameterValue(0.0));
  nav2_util::declare_parameter_if_not_declared(
    node, name_ + ".decay_step_s", rclcpp::ParameterValue(0.0));

  node->get_parameter(name_ + ".object_positions_topic", object_topic_name_);
  node->get_parameter(name_ + ".avoidance_radius", avoidance_radius_);
  node->get_parameter(name_ + ".bounds_padding_cells", bounds_padding_cells_);
  node->get_parameter(name_ + ".enabled", enabled_);

  avoidance_radius_ = std::max(0.0, avoidance_radius_);
  bounds_padding_cells_ = std::max(0.0, bounds_padding_cells_);

  matchSize();

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

  rclcpp::QoS qos(rclcpp::KeepLast(10));
  qos.reliable();
  object_positions_sub_ = node->create_subscription<geometry_msgs::msg::PoseArray>(
    object_topic_name_, qos,
    std::bind(&ObjectAvoidanceLayer::objectPositionsCallback, this, std::placeholders::_1));

  current_ = true;

  RCLCPP_INFO(
    node->get_logger(),
    "ObjectAvoidanceLayer initialized: topic='%s', residual_radius=%.3f m",
    object_topic_name_.c_str(), avoidance_radius_);
}

void ObjectAvoidanceLayer::objectPositionsCallback(
  const geometry_msgs::msg::PoseArray::SharedPtr msg)
{
  const std::string global_frame = layered_costmap_->getGlobalFrameID();
  const std::string source_frame = msg->header.frame_id;

  std::lock_guard<nav2_costmap_2d::Costmap2D::mutex_t> lock(*getMutex());

  detected_objects_.clear();

  double new_min_x = std::numeric_limits<double>::infinity();
  double new_min_y = std::numeric_limits<double>::infinity();
  double new_max_x = -std::numeric_limits<double>::infinity();
  double new_max_y = -std::numeric_limits<double>::infinity();
  bool new_valid = false;

  for (const auto & pose_in : msg->poses) {
    double x = pose_in.position.x;
    double y = pose_in.position.y;

    if (!source_frame.empty() && source_frame != global_frame) {
      try {
        geometry_msgs::msg::PoseStamped in;
        geometry_msgs::msg::PoseStamped out;
        in.header = msg->header;
        in.pose = pose_in;

        const auto transform = tf_buffer_->lookupTransform(
          global_frame, source_frame, tf2::TimePointZero, tf2::durationFromSec(0.1));
        tf2::doTransform(in, out, transform);
        x = out.pose.position.x;
        y = out.pose.position.y;
      } catch (const std::exception & e) {
        RCLCPP_WARN(
          rclcpp::get_logger("ObjectAvoidanceLayer"),
          "Skipping residual pose because TF transform failed: %s", e.what());
        continue;
      }
    }

    if (!std::isfinite(x) || !std::isfinite(y)) {
      continue;
    }

    detected_objects_.emplace_back(x, y, 0.0);
    new_min_x = std::min(new_min_x, x - avoidance_radius_);
    new_min_y = std::min(new_min_y, y - avoidance_radius_);
    new_max_x = std::max(new_max_x, x + avoidance_radius_);
    new_max_y = std::max(new_max_y, y + avoidance_radius_);
    new_valid = true;
  }

  // Include both previous and current footprints so removed residual entries are cleared.
  if (prev_valid_ && new_valid) {
    last_min_x_ = std::min(prev_min_x_, new_min_x);
    last_min_y_ = std::min(prev_min_y_, new_min_y);
    last_max_x_ = std::max(prev_max_x_, new_max_x);
    last_max_y_ = std::max(prev_max_y_, new_max_y);
  } else if (prev_valid_) {
    last_min_x_ = prev_min_x_;
    last_min_y_ = prev_min_y_;
    last_max_x_ = prev_max_x_;
    last_max_y_ = prev_max_y_;
  } else if (new_valid) {
    last_min_x_ = new_min_x;
    last_min_y_ = new_min_y;
    last_max_x_ = new_max_x;
    last_max_y_ = new_max_y;
  } else {
    last_min_x_ = last_min_y_ = last_max_x_ = last_max_y_ = 0.0;
  }

  updated_ = prev_valid_ || new_valid;

  prev_valid_ = new_valid;
  prev_min_x_ = new_min_x;
  prev_min_y_ = new_min_y;
  prev_max_x_ = new_max_x;
  prev_max_y_ = new_max_y;

  current_ = false;
}

void ObjectAvoidanceLayer::updateBounds(
  double, double, double,
  double * min_x, double * min_y, double * max_x, double * max_y)
{
  if (!enabled_ || !updated_) {
    return;
  }

  const double padding = bounds_padding_cells_ * getResolution();
  *min_x = std::min(*min_x, last_min_x_ - padding);
  *min_y = std::min(*min_y, last_min_y_ - padding);
  *max_x = std::max(*max_x, last_max_x_ + padding);
  *max_y = std::max(*max_y, last_max_y_ + padding);
}

void ObjectAvoidanceLayer::updateCosts(
  nav2_costmap_2d::Costmap2D & master,
  int min_i, int min_j, int max_i, int max_j)
{
  if (!enabled_) {
    return;
  }

  std::lock_guard<nav2_costmap_2d::Costmap2D::mutex_t> lock(*getMutex());

  const int width = static_cast<int>(getSizeInCellsX());
  const int height = static_cast<int>(getSizeInCellsY());
  const int x0 = std::max(0, min_i);
  const int y0 = std::max(0, min_j);
  const int x1 = std::min(width, max_i);
  const int y1 = std::min(height, max_j);

  // Clear this layer's previous marks in the affected region.
  for (int y = y0; y < y1; ++y) {
    for (int x = x0; x < x1; ++x) {
      setCost(
        static_cast<unsigned int>(x), static_cast<unsigned int>(y),
        nav2_costmap_2d::NO_INFORMATION);
    }
  }

  const double radius_sq = avoidance_radius_ * avoidance_radius_;
  const int radius_cells = std::max(
    1, static_cast<int>(std::ceil(avoidance_radius_ / getResolution())));

  for (const auto & object : detected_objects_) {
    unsigned int cx = 0;
    unsigned int cy = 0;
    if (!worldToMap(object.getX(), object.getY(), cx, cy)) {
      continue;
    }

    const int ix0 = std::max(0, static_cast<int>(cx) - radius_cells);
    const int iy0 = std::max(0, static_cast<int>(cy) - radius_cells);
    const int ix1 = std::min(width - 1, static_cast<int>(cx) + radius_cells);
    const int iy1 = std::min(height - 1, static_cast<int>(cy) + radius_cells);

    for (int y = iy0; y <= iy1; ++y) {
      for (int x = ix0; x <= ix1; ++x) {
        double wx = 0.0;
        double wy = 0.0;
        mapToWorld(static_cast<unsigned int>(x), static_cast<unsigned int>(y), wx, wy);
        const double dx = wx - object.getX();
        const double dy = wy - object.getY();
        if (dx * dx + dy * dy <= radius_sq) {
          setCost(
            static_cast<unsigned int>(x), static_cast<unsigned int>(y),
            nav2_costmap_2d::LETHAL_OBSTACLE);
        }
      }
    }
  }

  // Merge only this layer's known cells into the standard Nav2 master costmap.
  for (int y = y0; y < y1; ++y) {
    for (int x = x0; x < x1; ++x) {
      const auto ux = static_cast<unsigned int>(x);
      const auto uy = static_cast<unsigned int>(y);
      const unsigned char layer_cost = getCost(ux, uy);
      if (layer_cost == nav2_costmap_2d::NO_INFORMATION) {
        continue;
      }
      master.setCost(ux, uy, std::max(master.getCost(ux, uy), layer_cost));
    }
  }

  current_ = true;
  updated_ = false;
}

void ObjectAvoidanceLayer::reset()
{
  std::lock_guard<nav2_costmap_2d::Costmap2D::mutex_t> lock(*getMutex());

  for (unsigned int y = 0; y < getSizeInCellsY(); ++y) {
    for (unsigned int x = 0; x < getSizeInCellsX(); ++x) {
      setCost(x, y, nav2_costmap_2d::NO_INFORMATION);
    }
  }

  detected_objects_.clear();
  prev_valid_ = false;

  last_min_x_ = getOriginX();
  last_min_y_ = getOriginY();
  last_max_x_ = getOriginX() + getSizeInCellsX() * getResolution();
  last_max_y_ = getOriginY() + getSizeInCellsY() * getResolution();

  updated_ = true;
  current_ = false;
}

}  // namespace my_costmap_layers

PLUGINLIB_EXPORT_CLASS(
  my_costmap_layers::ObjectAvoidanceLayer,
  nav2_costmap_2d::Layer)
