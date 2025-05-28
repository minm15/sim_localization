#pragma once

#include "sim_local/LinK3D_extractor.h"

#include <deque>
#include <filesystem>
#include <string>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <Eigen/Core>

namespace sim_local {

struct GTSample {
    rclcpp::Time stamp;
    geometry_msgs::msg::Pose pose;
};

class NCLTMapNode : public rclcpp::Node {
  public:
    explicit NCLTMapNode(const rclcpp::NodeOptions& opts = rclcpp::NodeOptions{});

  private:
    // Subscriber callbacks
    void tfStaticCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg);
    void tfDynamicCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg);
    void groundTruthCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    // Helpers
    Eigen::Matrix4f transformMsgToEigen(const geometry_msgs::msg::Transform& t);
    Eigen::Matrix4f poseToEigen(const geometry_msgs::msg::Pose& p);

    // Subscribers
    rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr static_tf_sub_;
    rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr dynamic_tf_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr gt_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;

    // TF buffer & listener
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    // LinK3D extractor
    std::shared_ptr<LinK3D_SLAM::LinK3D_Extractor> extractor_;

    // Cached static transform base_link→velodyne
    bool have_base_velo_{false};
    Eigen::Matrix4f base_T_velo_{Eigen::Matrix4f::Identity()};

    // Ground-truth queue (keep recent)
    static constexpr size_t kGTMax = 200;
    std::deque<GTSample> gt_queue_;

    // Descriptor dump
    std::string descriptor_path_;
    bool first_frame_{true};
    cv::Mat prev_descriptors_;
    size_t frame_count_{0};

    // Scan timing stats
    bool has_last_scan_{false};
    double max_scan_interval_sec_{0.0};
    rclcpp::Time last_scan_time_;
};

} // namespace sim_local