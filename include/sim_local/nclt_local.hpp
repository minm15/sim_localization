// include/sim_local/nclt_local.hpp
#ifndef SIM_LOCAL_NCLT_LOCAL_HPP
#define SIM_LOCAL_NCLT_LOCAL_HPP

#include <rclcpp/rclcpp.hpp>

// ROS messages
#include <geometry_msgs/msg/pose_array.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

// TF2 listeners
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

// STL / other
#include <Eigen/Dense>
#include <deque>
#include <opencv2/core.hpp>
#include <optional>
#include <pcl/point_cloud.h>
#include <unordered_set>

// Your modules
#include "sim_local/LinK3D_extractor.h"
#include "sim_local/Particle_filter.hpp"
#include "sim_local/kdtree.hpp"

// bring these into the global namespace for brevity
using geometry_msgs::msg::PoseArray;
using geometry_msgs::msg::TransformStamped;
using nav_msgs::msg::Odometry;
using sensor_msgs::msg::PointCloud2;
using tf2_msgs::msg::TFMessage;

namespace sim_local {

class NcltNode : public rclcpp::Node {
public:
    explicit NcltNode(const rclcpp::NodeOptions& opts);

private:
    // Callbacks
    void tfStaticCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg);
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void groundTruthCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    // Utilities
    cv::Mat loadBinaryFileToMat(const std::string& fp);
    template<typename BufferT>
    std::optional<typename BufferT::value_type>
    findLastBefore(const BufferT& buf, const rclcpp::Time& t);
    Eigen::Matrix4f transformMsgToEigen(const geometry_msgs::msg::Transform& t);
    Eigen::Matrix4f poseToEigen(const geometry_msgs::msg::Pose& p);
    std::vector<pcl::PointXYZ>
    transformKeyPoints(const std::vector<pcl::PointXYZ>& pts, const Eigen::Matrix4f& T);
    void logFrameInfo(const rclcpp::Time& scan_t,
                      const rclcpp::Time& odom_t,
                      const rclcpp::Time& gt_t,
                      double error_xy,
                      const geometry_msgs::msg::Pose& pred_pose,
                      const geometry_msgs::msg::Pose& gt_pose);

    // ROS interfaces
    rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr     static_tf_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr     odom_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr     gt_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr   pub_;

    // Static transforms
    Eigen::Matrix4f world_T_odom_;
    Eigen::Matrix4f base_T_velo_;
    bool have_world_odom_{false}, have_base_velo_{false};

    // Particle filter & odometry
    std::shared_ptr<ParticleFilter>              particle_filter_;
    geometry_msgs::msg::Pose                      current_pose_;
    rclcpp::Time                                  last_odom_time_;
    bool                                          has_last_odom_time_{false};

    // Histories for time alignment
    std::deque<std::pair<rclcpp::Time, geometry_msgs::msg::Pose>> odom_history_;
    std::deque<std::pair<rclcpp::Time, geometry_msgs::msg::Pose>> gt_history_;

    // LinK3D
    cv::Mat                                         vectorDatabase_;
    std::shared_ptr<LinK3D_SLAM::LinK3D_Extractor>  extractor_;
    size_t                                          frame_count_{0};

    // Parameters
    std::string desc_file_;
    double init_x_, init_y_, init_z_, init_roll_, init_pitch_, init_yaw_;

    // Kdtree
    std::unique_ptr<Kdtree> kdtree;
};

} // namespace sim_local

#endif // SIM_LOCAL_NCLT_LOCAL_HPP