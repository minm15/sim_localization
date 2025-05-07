#pragma once

#include "sim_local/LinK3D_extractor.h"
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <Eigen/Core>

namespace sim_local {

class NCLTMapNode : public rclcpp::Node {
public:
  explicit NCLTMapNode(const rclcpp::NodeOptions& opts = rclcpp::NodeOptions{});

private:
  // callbacks
  void tfStaticCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg);
  void tfDynamicCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg);
  void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

  // subscriptions
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr static_tf_sub_;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr dynamic_tf_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;

  // TF machinery
  tf2_ros::Buffer            tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // LinK3D extractor & state
  std::shared_ptr<LinK3D_SLAM::LinK3D_Extractor> extractor_;
  cv::Mat           prevDescriptors_;
  bool              first_frame_{true};
  size_t            frame_count_{0};

  // static transforms cache
  bool              have_world_odom_{false};
  bool              have_base_velo_{false};
  Eigen::Matrix4f   world_T_odom_, base_T_velo_;

  // scan‐interval timing
  rclcpp::Time      last_scan_time_;
  bool              has_last_scan_{false};
  double            max_scan_interval_sec_{0.0};
};

} // namespace sim_local
