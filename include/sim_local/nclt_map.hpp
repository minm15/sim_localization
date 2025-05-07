// sim_local/nclt_map.hpp
#pragma once
#include "sim_local/LinK3D_extractor.h"
#include <deque>
#include <optional>
#include <unordered_set>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace sim_local {

class NCLTMapNode : public rclcpp::Node {
public:
  explicit NCLTMapNode(const rclcpp::NodeOptions& opts = rclcpp::NodeOptions{});

private:
  void tfStaticCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg);
  void tfDynamicCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg);
  void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
  void processPendingTFs();
  void processOneTF(const geometry_msgs::msg::TransformStamped& ts);

  template<typename T>
  std::optional<T> findLastBefore(const std::deque<T>& buf,
                                  const rclcpp::Time& t);

  // subscriptions
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr static_tf_sub_;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr dynamic_tf_sub_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;

  // TF machinery
  tf2_ros::Buffer     tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  // buffers
  std::deque<std::pair<rclcpp::Time,
            sensor_msgs::msg::PointCloud2::SharedPtr>> lidar_history_;
  std::deque<geometry_msgs::msg::TransformStamped>  tf_queue_;
  // std::unordered_set<uint64_t>                      processed_tf_;

  // LinK3D
  std::shared_ptr<LinK3D_SLAM::LinK3D_Extractor> extractor_;
  cv::Mat           prevDescriptors_;
  bool              first_frame_;
  size_t            frame_count_;

  // cached statics
  bool       have_world_odom_, have_base_velo_;
  Eigen::Matrix4f world_T_odom_, base_T_velo_;
};

} // namespace sim_local
