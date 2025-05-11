#pragma once

#include "sim_local/LinK3D_extractor.h"
#include <filesystem>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace sim_local {

class FrameDumperNode : public rclcpp::Node {
  public:
    explicit FrameDumperNode(const rclcpp::NodeOptions& opts = rclcpp::NodeOptions{});

  private:
    void tfStaticCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg);
    void tfDynamicCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg);
    void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    // TF machinery
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    // LinK3D extractor
    std::shared_ptr<LinK3D_SLAM::LinK3D_Extractor> extractor_;

    // subscriptions
    rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr static_tf_sub_;
    rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr dynamic_tf_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;

    // cached static transforms
    bool have_world_odom_;
    bool have_base_velo_;
    Eigen::Matrix4f world_T_odom_;
    Eigen::Matrix4f base_T_velo_;

    // output paths
    std::filesystem::path root_path_;
    std::filesystem::path frames_path_;

    // frame counter
    size_t frame_count_;
};

} // namespace sim_local
