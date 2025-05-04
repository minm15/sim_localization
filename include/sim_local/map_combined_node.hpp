#pragma once

#include "sim_local/LinK3D_extractor.h"
#include <deque>
#include <opencv2/core.hpp>
#include <optional>
#include <pcl/point_types.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <unordered_set>

namespace sim_local {

class MapCombinedNode : public rclcpp::Node {
  public:
    // single constructor, opts is optional
    explicit MapCombinedNode(const rclcpp::NodeOptions& opts = rclcpp::NodeOptions{});

  private:
    void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void tfCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg);
    void processPendingTFs();
    void processOneTF(const geometry_msgs::msg::TransformStamped& ts);

    template <typename T>
    std::optional<T> findLastBefore(const std::deque<T>& buf, const rclcpp::Time& t);

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
    rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr tf_sub_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    std::deque<std::pair<rclcpp::Time, sensor_msgs::msg::PointCloud2::SharedPtr>> lidar_history_;
    std::deque<geometry_msgs::msg::TransformStamped> tf_queue_;
    std::unordered_set<uint64_t> processed_tf_;

    std::shared_ptr<LinK3D_SLAM::LinK3D_Extractor> extractor_;
    cv::Mat prevDescriptors_;
    bool first_frame_;
    size_t frame_count_;
};

} // namespace sim_local
