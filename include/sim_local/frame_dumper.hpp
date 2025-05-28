#pragma once

#include "sim_local/LinK3D_extractor.h"

#include <Eigen/Core>
#include <deque>
#include <filesystem>
#include <geometry_msgs/msg/transform.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

namespace sim_local {

struct GTSample {
    rclcpp::Time stamp;
    geometry_msgs::msg::Pose pose;
};

class FrameDumperNode : public rclcpp::Node {
  public:
    explicit FrameDumperNode(const rclcpp::NodeOptions& opts = rclcpp::NodeOptions{});

  private:
    // callbacks
    void tfStaticCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg);
    void tfDynamicCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg);
    void groundTruthCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    // subscriptions
    rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr static_tf_sub_;
    rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr dynamic_tf_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr gt_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;

    // TF
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    // LinK3D
    std::shared_ptr<LinK3D_SLAM::LinK3D_Extractor> extractor_;

    // cached static transform
    bool have_base_velo_{false};
    Eigen::Matrix4f base_T_velo_{Eigen::Matrix4f::Identity()};

    // GT buffer
    static constexpr size_t kGTMax = 200;
    std::deque<GTSample> gt_queue_;

    // NED→ENU fixed
    Eigen::Matrix4f T_w_o_{Eigen::Matrix4f::Identity()};

    // output
    std::filesystem::path root_path_;
    std::filesystem::path frames_path_;
    size_t frame_count_{0};

    // helper
    Eigen::Matrix4f transformMsgToEigen(const geometry_msgs::msg::Transform& t);
};

} // namespace sim_local