#ifndef SIM_LOCAL_NCLT_LOCAL_HPP
#define SIM_LOCAL_NCLT_LOCAL_HPP

#include <geometry_msgs/msg/pose_array.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include <Eigen/Dense>
#include <deque>
#include <opencv2/core.hpp>
#include <optional>
#include <pcl/point_cloud.h>
#include <unordered_map>
#include <vector>

#include "sim_local/LinK3D_extractor.h"
#include "sim_local/Particle_filter.hpp"
#include "sim_local/util.hpp"

namespace sim_local {

class NcltNode : public rclcpp::Node {
  public:
    explicit NcltNode(const rclcpp::NodeOptions& opts);

  private:
    void tfStaticCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg);
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg);
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void groundTruthCallback(const nav_msgs::msg::Odometry::SharedPtr msg);
    void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    void logFrameInfo(const rclcpp::Time& scan_t, const rclcpp::Time& odom_t, const rclcpp::Time& gt_t, double error_xy,
                      const geometry_msgs::msg::Pose& pred_pose, const geometry_msgs::msg::Pose& gt_pose, int kpsize);

    void buildBuckets(int B);
    int bucketIndex(float left, float d0, float right) const;

    rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr static_tf_sub_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr gt_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr pub_;

    Eigen::Matrix4f base_T_velo_;
    bool have_base_velo_{false};
    bool have_received_first_scan_{false};

    std::deque<std::pair<rclcpp::Time, geometry_msgs::msg::Pose>> odom_history_;
    std::deque<std::pair<rclcpp::Time, geometry_msgs::msg::Pose>> gt_history_;
    rclcpp::Time last_odom_time_;
    bool has_last_odom_time_{false};

    sensor_msgs::msg::Imu::SharedPtr last_imu_msg_;
    bool has_last_imu_{false};

    cv::Mat vectorDatabase_;
    std::unordered_map<int, std::vector<int>> db_buckets_;
    std::vector<float> cuts_left_, cuts_d0_, cuts_right_;
    int buckets_per_dim_{4};

    std::shared_ptr<LinK3D_SLAM::LinK3D_Extractor> extractor_;
    std::shared_ptr<ParticleFilter> particle_filter_;
    size_t frame_count_{0};
    std::string desc_file_;
    double init_x_, init_y_, init_z_, init_roll_, init_pitch_, init_yaw_;
};

} // namespace sim_local

#endif // SIM_LOCAL_NCLT_LOCAL_HPP