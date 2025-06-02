#pragma once

#include "sim_local/LinK3D_extractor.h"

#include <deque>
#include <string>
#include <vector>
#include <unordered_set>

#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <pcl/point_types.h>

namespace sim_local {

struct GTSample {
    rclcpp::Time stamp;
    geometry_msgs::msg::Pose pose;
};

class NCLTMapNode : public rclcpp::Node {
public:
    explicit NCLTMapNode(const rclcpp::NodeOptions& opts = rclcpp::NodeOptions{});

private:
    // callbacks
    void tfStaticCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg);
    void tfDynamicCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg);
    void groundTruthCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    // helpers
    Eigen::Matrix4f transformMsgToEigen(const geometry_msgs::msg::Transform& t);
    Eigen::Matrix4f poseToEigen(const geometry_msgs::msg::Pose& p);

    // subscribers
    rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr static_tf_sub_;
    rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr dynamic_tf_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr gt_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;

    // TF
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    // extractor
    std::shared_ptr<LinK3D_SLAM::LinK3D_Extractor> extractor_;

    // static transform base_link -> velodyne
    bool have_base_velo_{false};
    Eigen::Matrix4f base_T_velo_{Eigen::Matrix4f::Identity()};

    // ground truth buffer
    static constexpr size_t kGTMax = 200;
    std::deque<GTSample> gt_queue_;

    // descriptor dump file
    std::string descriptor_path_;
    bool first_frame_{true};

    // descriptors from last frame
    cv::Mat prev_descriptors_;

    // 全局存储已写入文件的 descriptor 和 3D 点
    cv::Mat global_descriptors_;
    std::vector<pcl::PointXYZ> global_points_;

    // bookkeeping
    size_t frame_count_{0};
    bool has_last_scan_{false};
    rclcpp::Time last_scan_time_;
    double max_scan_interval_sec_{0.0};
};

} // namespace sim_local