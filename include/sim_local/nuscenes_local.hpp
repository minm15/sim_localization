#ifndef SIM_LOCAL_LOCALIZATION_NODE_HPP
#define SIM_LOCAL_LOCALIZATION_NODE_HPP

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

// bring these into the global namespace for brevity
using geometry_msgs::msg::PoseArray;
using geometry_msgs::msg::TransformStamped;
using nav_msgs::msg::Odometry;
using sensor_msgs::msg::PointCloud2;
using tf2_msgs::msg::TFMessage;

namespace sim_local {

class NuscenesNode : public rclcpp::Node {
  public:
    explicit NuscenesNode(const rclcpp::NodeOptions& opts);

  private:
    // load descriptor binary into cv::Mat
    cv::Mat loadBinaryFileToMat(const std::string& fp);

    // subscribers
    void odomCallback(const Odometry::SharedPtr odom);
    void lidarCallback(const PointCloud2::SharedPtr cloud);
    void tfCallback(const TFMessage::SharedPtr msg);

    // process any enqueued TFs once buffers have caught up
    void processPendingTFs();
    void processOneTF(const TransformStamped& ts);

    // helper to find last ≤ t
    template <typename BufferT>
    std::optional<typename BufferT::value_type> findLastBefore(const BufferT& buf, const rclcpp::Time& t);

    // transforms & geometry utils
    Eigen::Matrix4f transformMsgToEigen(const geometry_msgs::msg::Transform& t);
    Eigen::Matrix4f poseToEigen(const geometry_msgs::msg::Pose& p);
    std::vector<pcl::PointXYZ> transformKeyPoints(const std::vector<pcl::PointXYZ>& pts, const Eigen::Matrix4f& T);

    // parameters & state
    std::string desc_file_;
    double init_x_, init_y_, init_z_, init_roll_, init_pitch_, init_yaw_;

    cv::Mat vectorDatabase_;
    std::shared_ptr<LinK3D_SLAM::LinK3D_Extractor> extractor_;
    std::shared_ptr<ParticleFilter> particle_filter_;

    geometry_msgs::msg::Pose current_pose_;
    rclcpp::Time last_odom_time_;
    bool has_last_odom_time_;
    size_t frame_count_;

    // data buffers
    std::deque<std::pair<rclcpp::Time, geometry_msgs::msg::Pose>> odom_history_;
    std::deque<std::pair<rclcpp::Time, PointCloud2::SharedPtr>> lidar_history_;
    std::deque<TransformStamped> tf_queue_;
    std::unordered_set<uint64_t> processed_tf_;

    // TF listener (unused directly but kept alive)
    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    // ROS interfaces
    rclcpp::Subscription<Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<PointCloud2>::SharedPtr lidar_sub_;
    rclcpp::Subscription<TFMessage>::SharedPtr tf_sub_;
    rclcpp::Publisher<PoseArray>::SharedPtr pub_;

    // Log
    void logFrameInfo(const rclcpp::Time& tf_t, const rclcpp::Time& odom_t, const rclcpp::Time& lidar_t, double e_o,
                      double ox, double oy, double e_p, double px, double py, double gx, double gy);
};

} // namespace sim_local

#endif // SIM_LOCAL_LOCALIZATION_NODE_HPP