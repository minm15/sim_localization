// nclt_local.hpp

#ifndef SIM_LOCAL_NCLT_LOCAL_HPP
#define SIM_LOCAL_NCLT_LOCAL_HPP

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose_array.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include <Eigen/Dense>
#include <deque>
#include <optional>
#include <opencv2/core.hpp>
#include <pcl/point_cloud.h>
#include <unordered_map>

#include "sim_local/LinK3D_extractor.h"
#include "sim_local/Particle_filter.hpp"

using geometry_msgs::msg::PoseArray;
using nav_msgs::msg::Odometry;
using sensor_msgs::msg::PointCloud2;
using sensor_msgs::msg::Imu;
using tf2_msgs::msg::TFMessage;

namespace sim_local {

class NcltNode : public rclcpp::Node {
public:
  explicit NcltNode(const rclcpp::NodeOptions& opts);

private:
  // callbacks
  void tfStaticCallback(const TFMessage::SharedPtr msg);
  void imuCallback(const Imu::SharedPtr msg);
  void odomCallback(const Odometry::SharedPtr msg);
  void groundTruthCallback(const Odometry::SharedPtr msg);
  void lidarCallback(const PointCloud2::SharedPtr msg);

  // utils
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

  // bucket build
  void buildBuckets();
  int  bucketIndex(float left, float d0, float right) const;

  // ROS interfaces
  rclcpp::Subscription<TFMessage>::SharedPtr   static_tf_sub_;
  rclcpp::Subscription<Imu>::SharedPtr         imu_sub_;
  rclcpp::Subscription<Odometry>::SharedPtr    odom_sub_;
  rclcpp::Subscription<Odometry>::SharedPtr    gt_sub_;
  rclcpp::Subscription<PointCloud2>::SharedPtr lidar_sub_;
  rclcpp::Publisher<PoseArray>::SharedPtr      pub_;

  // transforms & state
  Eigen::Matrix4f base_T_velo_;
  bool have_base_velo_{false};

  std::shared_ptr<ParticleFilter> particle_filter_;
  bool have_received_first_scan_{false};

  rclcpp::Time  last_odom_time_;
  bool          has_last_odom_time_{false};

  Imu::SharedPtr last_imu_msg_;
  bool           has_last_imu_{false};

  std::deque<std::pair<rclcpp::Time, geometry_msgs::msg::Pose>> odom_history_;
  std::deque<std::pair<rclcpp::Time, geometry_msgs::msg::Pose>> gt_history_;

  // descriptor database + buckets
  cv::Mat                                         vectorDatabase_;
  std::unordered_map<int, std::vector<int>>       db_buckets_;
  std::array<float,3>                             quartiles_[3];  // [0]=left, [1]=d0, [2]=right

  std::shared_ptr<LinK3D_SLAM::LinK3D_Extractor>  extractor_;
  size_t                                          frame_count_{0};

  std::string desc_file_;
  double      init_x_, init_y_, init_z_, init_roll_, init_pitch_, init_yaw_;
};

} // namespace sim_local

#endif // SIM_LOCAL_NCLT_LOCAL_HPP