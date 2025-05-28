#pragma once

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/transform.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <opencv2/core.hpp>
#include <pcl/point_types.h>
#include <vector>
#include <deque>
#include <optional>
#include <algorithm>
#include <fstream>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

namespace sim_local {
namespace util {

/// Load our 183-dim descriptors from the binary dump into a cv::Mat
cv::Mat loadBinaryFileToMat(const std::string & path);

/// Find the last element in a time-sorted buffer whose .first <= t
template<typename BufferT>
std::optional<typename BufferT::value_type>
findLastBefore(const BufferT & buf, const rclcpp::Time & t);

/// Convert a ROS Transform into a 4×4 Eigen matrix
Eigen::Matrix4f transformMsgToEigen(const geometry_msgs::msg::Transform & t);

/// Convert a ROS Pose into a 4×4 Eigen matrix
Eigen::Matrix4f poseToEigen(const geometry_msgs::msg::Pose & p);

/// Apply a 4×4 transform to a list of pcl::PointXYZ
std::vector<pcl::PointXYZ>
transformKeyPoints(const std::vector<pcl::PointXYZ> & pts, const Eigen::Matrix4f & T);

}  // namespace util
}  // namespace sim_local

// We leave the template declaration here; instantiations will live in util.cpp  