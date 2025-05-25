#pragma once

#include <Eigen/Dense>
#include <array>
#include <geometry_msgs/msg/pose.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <opencv2/core.hpp>
#include <pcl/point_types.h>
#include <random>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <vector>

struct Particle {
    geometry_msgs::msg::Pose pose; ///< 6-DOF pose
    double distance{0.0};          ///< map-matching score
    double weight{1.0};            ///< importance weight

    void map_matching(std::vector<pcl::PointXYZ>& transformedKeyPoints, cv::Mat& map_descriptors,
                      std::vector<std::pair<int, int>>& vMatched);
};

namespace sim_local {

class ParticleFilter {
  public:
    ParticleFilter(double init_x, double init_y, double init_z, double init_roll, double init_pitch, double init_yaw,
                   int num_particles);

    void update(const nav_msgs::msg::Odometry::ConstSharedPtr& odom, const sensor_msgs::msg::Imu::ConstSharedPtr& imu,
                double dt);

    void weighting();
    void resampling();

    std::vector<Particle>& getParticles();
    Particle getBestParticle(int top_k = 1);
    void printParticleInfo(int i);

  private:
    void initializeParticles(double init_x, double init_y, double init_z, double init_roll, double init_pitch,
                             double init_yaw);

    void predict(double linear_x, double angular_z, double dt);

    void systematicResample();

    std::vector<Particle> particles_;
    std::array<double, 36> odom_pose_cov_{};
    std::array<double, 36> odom_twist_cov_{};
    bool have_odom_cov_{false};

    std::default_random_engine gen_{std::random_device{}()};
};

} // namespace sim_local