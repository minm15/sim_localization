#pragma once
/*  Particle Filter header
 *  ----------------------------------------------------------
 *  • 位姿 6-DOF，但运动模型采用 SE(2)（x-y-yaw）；
 *    z / roll / pitch 仅加入小幅随机扰动以避免数值漂移。
 *  • 线速度取自 Odometry.linear.x，角速度取自 Imu.angular_velocity.z。
 */

#include <Eigen/Dense>
#include <array>
#include <geometry_msgs/msg/pose.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <opencv2/core.hpp>
#include <pcl/point_types.h>
#include <random>
#include <rclcpp/rclcpp.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <vector>

struct Particle
{
  geometry_msgs::msg::Pose pose;   ///< 6-DOF pose
  double distance{0.0};            ///< map-matching score
  double weight{1.0};              ///< importance weight

  void map_matching(std::vector<pcl::PointXYZ>& transformedKeyPoints,
                    cv::Mat& map_descriptors,
                    std::vector<std::pair<int, int>>& vMatched);
};

namespace sim_local
{

class ParticleFilter
{
public:
  ParticleFilter(double init_x,
                 double init_y,
                 double init_z,
                 double init_roll,
                 double init_pitch,
                 double init_yaw,
                 int num_particles);

  /// 用最新的里程计和 IMU 做一次 predict，dt 是二者时间差
  void update(const nav_msgs::msg::Odometry::ConstSharedPtr& odom,
              const sensor_msgs::msg::Imu::ConstSharedPtr& imu,
              double dt);

  void weighting();
  void resampling();

  std::vector<Particle>& getParticles();
  Particle getBestParticle(int top_k = 1);
  void printParticleInfo(int i);

private:
  void initializeParticles(double init_x,
                           double init_y,
                           double init_z,
                           double init_roll,
                           double init_pitch,
                           double init_yaw);

  /// 内部的运动模型，只接受线速度和角速度
  void predict(double linear_x,
               double angular_z,
               double dt);

  void systematicResample();

  std::vector<Particle>          particles_;
  std::array<double, 36>         odom_pose_cov_{};
  std::array<double, 36>         odom_twist_cov_{};
  bool                           have_odom_cov_{false};

  std::default_random_engine     gen_{std::random_device{}()};
};

}  // namespace sim_local