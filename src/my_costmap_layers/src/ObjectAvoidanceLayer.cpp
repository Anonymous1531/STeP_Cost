#include "my_costmap_layers/ObjectAvoidanceLayer.hpp"
#include "pluginlib/class_list_macros.hpp"
#include <nav2_costmap_2d/layered_costmap.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <atomic>
#include "nav2_util/node_utils.hpp"

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

namespace my_costmap_layers
{
namespace {
  std::atomic<bool> s_printed_geo_warn{false};
}

ObjectAvoidanceLayer::ObjectAvoidanceLayer()
: avoidance_radius_(2.0)
, enabled_(true)
, bounds_padding_cells_(10.0)
, hold_after_clear_s_(0.0)
, decay_ttl_s_(0.0)
, decay_step_s_(0.0)
, last_min_x_(0.0), last_min_y_(0.0), last_max_x_(0.0), last_max_y_(0.0)
, prev_min_x_(0.0), prev_min_y_(0.0), prev_max_x_(0.0), prev_max_y_(0.0)
, prev_valid_(false)
, updated_(false)
{}

void ObjectAvoidanceLayer::onInitialize()
{
  auto node = node_.lock();
  if (!node) {
    throw std::runtime_error("Failed to lock node in ObjectAvoidanceLayer::onInitialize");
  }

  nav2_util::declare_parameter_if_not_declared(
    node, name_ + ".object_positions_topic", rclcpp::ParameterValue("/object_world_positions"));
  nav2_util::declare_parameter_if_not_declared(
    node, name_ + ".avoidance_radius", rclcpp::ParameterValue(0.5));
  nav2_util::declare_parameter_if_not_declared(
    node, name_ + ".enabled", rclcpp::ParameterValue(true));
  nav2_util::declare_parameter_if_not_declared(
    node, name_ + ".bounds_padding_cells", rclcpp::ParameterValue(10.0));
  // Keep these parameters for backward compatibility, but they are not used in pose-only mode.
  nav2_util::declare_parameter_if_not_declared(
    node, name_ + ".hold_after_clear_s", rclcpp::ParameterValue(0.0));
  nav2_util::declare_parameter_if_not_declared(
    node, name_ + ".decay_ttl_s", rclcpp::ParameterValue(0.0));
  nav2_util::declare_parameter_if_not_declared(
    node, name_ + ".decay_step_s", rclcpp::ParameterValue(0.0));

  node->get_parameter(name_ + ".object_positions_topic", object_topic_name_);
  node->get_parameter(name_ + ".avoidance_radius", avoidance_radius_);
  node->get_parameter(name_ + ".enabled", enabled_);
  node->get_parameter(name_ + ".bounds_padding_cells", bounds_padding_cells_);
  node->get_parameter(name_ + ".hold_after_clear_s", hold_after_clear_s_);
  node->get_parameter(name_ + ".decay_ttl_s", decay_ttl_s_);
  node->get_parameter(name_ + ".decay_step_s", decay_step_s_);

  matchSize();
  ensureTtlGridSized();
  last_decay_tick_ = node->now();

  tf_buffer_ = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);

  rclcpp::QoS qos(rclcpp::KeepLast(10));
  qos.reliable();
  object_positions_sub_ = node->create_subscription<geometry_msgs::msg::PoseArray>(
    object_topic_name_, qos,
    std::bind(&ObjectAvoidanceLayer::objectPositionsCallback, this, std::placeholders::_1));

  current_ = true;

  RCLCPP_INFO(
    rclcpp::get_logger("ObjectAvoidanceLayer"),
    "init pose-only: r=%.2f, topic='%s', pad_cells=%.1f (hold/ttl/step params ignored: %.2f/%.2f/%.2f)",
    avoidance_radius_, object_topic_name_.c_str(), bounds_padding_cells_,
    hold_after_clear_s_, decay_ttl_s_, decay_step_s_);
}

// Kept only for header compatibility. Pose-only mode does not use TTL state.
void ObjectAvoidanceLayer::ensureTtlGridSized()
{
  const size_t need = static_cast<size_t>(getSizeInCellsX()) * getSizeInCellsY();
  if (ttl_grid_.size() != need) {
    ttl_grid_.assign(need, 0.0f);
  }
}

void ObjectAvoidanceLayer::decayTtl(double)
{
  // no-op in pose-only mode
}

void ObjectAvoidanceLayer::stampDiskTtl(unsigned int, unsigned int, int)
{
  // no-op in pose-only mode
}

void ObjectAvoidanceLayer::expandRoiToLiveTtl()
{
  // no-op in pose-only mode
}

void ObjectAvoidanceLayer::updateBounds(
  double, double, double,
  double * min_x, double * min_y, double * max_x, double * max_y)
{
  if (!enabled_ || !updated_) {
    return;
  }

  const double pad = std::max(0.0, bounds_padding_cells_) * getResolution();
  if (!std::isfinite(last_min_x_) || !std::isfinite(last_min_y_) ||
      !std::isfinite(last_max_x_) || !std::isfinite(last_max_y_)) {
    return;
  }

  *min_x = std::min(*min_x, last_min_x_ - pad);
  *min_y = std::min(*min_y, last_min_y_ - pad);
  *max_x = std::max(*max_x, last_max_x_ + pad);
  *max_y = std::max(*max_y, last_max_y_ + pad);
}

void ObjectAvoidanceLayer::objectPositionsCallback(
  const geometry_msgs::msg::PoseArray::SharedPtr msg)
{
  const std::string global_frame = layered_costmap_->getGlobalFrameID();
  const std::string & msg_frame = msg->header.frame_id;

  std::lock_guard<nav2_costmap_2d::Costmap2D::mutex_t> lock(*getMutex());

  detected_objects_.clear();
  double new_min_x = std::numeric_limits<double>::infinity();
  double new_min_y = std::numeric_limits<double>::infinity();
  double new_max_x = -std::numeric_limits<double>::infinity();
  double new_max_y = -std::numeric_limits<double>::infinity();
  bool new_valid = false;

  for (const auto & p_in : msg->poses) {
    double x = p_in.position.x;
    double y = p_in.position.y;

    if (!msg_frame.empty() && msg_frame != global_frame) {
      try {
        geometry_msgs::msg::PoseStamped ps_in, ps_out;
        ps_in.header.frame_id = msg_frame;
        ps_in.header.stamp = msg->header.stamp;
        ps_in.pose = p_in;

        auto tf = tf_buffer_->lookupTransform(
          global_frame, msg_frame, tf2::TimePointZero, tf2::durationFromSec(0.1));
        tf2::doTransform(ps_in, ps_out, tf);
        x = ps_out.pose.position.x;
        y = ps_out.pose.position.y;
      } catch (...) {
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

  // Expand ROI to include both previous and current footprint so old costs are cleared immediately.
  if (prev_valid_ && new_valid) {
    last_min_x_ = std::min(prev_min_x_, new_min_x);
    last_min_y_ = std::min(prev_min_y_, new_min_y);
    last_max_x_ = std::max(prev_max_x_, new_max_x);
    last_max_y_ = std::max(prev_max_y_, new_max_y);
  } else if (prev_valid_ && !new_valid) {
    last_min_x_ = prev_min_x_;
    last_min_y_ = prev_min_y_;
    last_max_x_ = prev_max_x_;
    last_max_y_ = prev_max_y_;
  } else if (!prev_valid_ && new_valid) {
    last_min_x_ = new_min_x;
    last_min_y_ = new_min_y;
    last_max_x_ = new_max_x;
    last_max_y_ = new_max_y;
  } else {
    last_min_x_ = last_min_y_ = last_max_x_ = last_max_y_ = 0.0;
  }

  const bool need_update = (prev_valid_ || new_valid);
  prev_valid_ = new_valid;
  prev_min_x_ = new_min_x;
  prev_min_y_ = new_min_y;
  prev_max_x_ = new_max_x;
  prev_max_y_ = new_max_y;

  // Clear any old compatibility state.
  last_objects_cache_.clear();
  ttl_grid_.assign(ttl_grid_.size(), 0.0f);

  updated_ = need_update;
  current_ = false;
}

void ObjectAvoidanceLayer::updateCosts(
  nav2_costmap_2d::Costmap2D & master,
  int min_i, int min_j, int max_i, int max_j)
{
  if (!enabled_) {
    return;
  }

  std::lock_guard<nav2_costmap_2d::Costmap2D::mutex_t> lock(*getMutex());

  if (!s_printed_geo_warn.load(std::memory_order_relaxed)) {
    const bool same_res = std::fabs(getResolution() - master.getResolution()) < 1e-9;
    const bool same_w = getSizeInCellsX() == master.getSizeInCellsX();
    const bool same_h = getSizeInCellsY() == master.getSizeInCellsY();
    if (!(same_res && same_w && same_h)) {
      RCLCPP_WARN(
        rclcpp::get_logger("ObjectAvoidanceLayer"),
        "Geometry mismatch: layer(res=%.6f,%ux%u) vs master(res=%.6f,%ux%u)",
        getResolution(), getSizeInCellsX(), getSizeInCellsY(),
        master.getResolution(), master.getSizeInCellsX(), master.getSizeInCellsY());
    }
    s_printed_geo_warn.store(true, std::memory_order_relaxed);
  }

  const int lx = static_cast<int>(getSizeInCellsX());
  const int ly = static_cast<int>(getSizeInCellsY());
  const int x0 = std::max(0, min_i);
  const int y0 = std::max(0, min_j);
  const int x1 = std::min(lx, max_i);
  const int y1 = std::min(ly, max_j);

  // First clear this layer's previous marks inside the ROI.
  for (int y = y0; y < y1; ++y) {
    for (int x = x0; x < x1; ++x) {
      setCost(static_cast<unsigned int>(x), static_cast<unsigned int>(y), nav2_costmap_2d::NO_INFORMATION);
    }
  }

  // Stamp only the latest PoseArray objects.
  const int r = std::max(1, static_cast<int>(std::ceil(avoidance_radius_ / getResolution())));
  for (const auto & p : detected_objects_) {
    unsigned int cx, cy;
    if (!worldToMap(p.getX(), p.getY(), cx, cy)) {
      continue;
    }

    const int ix_min = std::max(0, static_cast<int>(cx) - r);
    const int iy_min = std::max(0, static_cast<int>(cy) - r);
    const int ix_max = std::min(lx - 1, static_cast<int>(cx) + r);
    const int iy_max = std::min(ly - 1, static_cast<int>(cy) + r);
    const double res = getResolution();
    const double r2 = (r * res) * (r * res);

    for (int y = iy_min; y <= iy_max; ++y) {
      for (int x = ix_min; x <= ix_max; ++x) {
        const double dx = (x - static_cast<int>(cx)) * res;
        const double dy = (y - static_cast<int>(cy)) * res;
        if (dx * dx + dy * dy <= r2) {
          setCost(static_cast<unsigned int>(x), static_cast<unsigned int>(y), nav2_costmap_2d::LETHAL_OBSTACLE);
        }
      }
    }
  }

  // Merge this layer into master.
  const int mx0 = x0;
  const int my0 = y0;
  const int mx1 = std::min(lx - 1, max_i - 1);
  const int my1 = std::min(ly - 1, max_j - 1);
  for (int j = my0; j <= my1; ++j) {
    for (int i = mx0; i <= mx1; ++i) {
      const unsigned char lc = getCost(static_cast<unsigned int>(i), static_cast<unsigned int>(j));
      if (lc == nav2_costmap_2d::NO_INFORMATION) {
        continue;
      }
      const unsigned char mc = master.getCost(static_cast<unsigned int>(i), static_cast<unsigned int>(j));
      master.setCost(static_cast<unsigned int>(i), static_cast<unsigned int>(j), std::max(mc, lc));
    }
  }

  current_ = true;
  updated_ = false;
}

void ObjectAvoidanceLayer::reset()
{
  std::lock_guard<nav2_costmap_2d::Costmap2D::mutex_t> lock(*getMutex());

  for (unsigned int j = 0; j < getSizeInCellsY(); ++j) {
    for (unsigned int i = 0; i < getSizeInCellsX(); ++i) {
      setCost(i, j, nav2_costmap_2d::NO_INFORMATION);
    }
  }

  detected_objects_.clear();
  last_objects_cache_.clear();
  ttl_grid_.assign(ttl_grid_.size(), 0.0f);

  prev_valid_ = false;
  prev_min_x_ = prev_min_y_ = prev_max_x_ = prev_max_y_ = 0.0;

  const double origin_x = getOriginX();
  const double origin_y = getOriginY();
  const double wx_max = origin_x + getSizeInCellsX() * getResolution();
  const double wy_max = origin_y + getSizeInCellsY() * getResolution();
  last_min_x_ = origin_x;
  last_min_y_ = origin_y;
  last_max_x_ = wx_max;
  last_max_y_ = wy_max;

  updated_ = true;
  current_ = false;
}

}  // namespace my_costmap_layers

PLUGINLIB_EXPORT_CLASS(my_costmap_layers::ObjectAvoidanceLayer, nav2_costmap_2d::Layer)