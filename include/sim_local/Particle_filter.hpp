#pragma once

#include <Eigen/Dense>
#include <geometry_msgs/msg/pose.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <tf2/LinearMath/Quaternion.h>
#include <vector>
#include <opencv2/core/mat.hpp>
#include <opencv2/core.hpp>

struct Particle {
    geometry_msgs::msg::Pose pose;
    double distance;
    double weight;

    void map_matching(std::vector<pcl::PointXYZ>& transformedKeyPoints, cv::Mat& map_descriptors,
                      std::vector<std::pair<int, int>>& vMatched) {
        distance = 0.0;
        for (auto &matched : vMatched) {
            int map_index = matched.second;
            int sensor_index = matched.first;
            float map_x = map_descriptors.at<float>(map_index, 180);
            float map_y = map_descriptors.at<float>(map_index, 181);
            float sensor_x = transformedKeyPoints[sensor_index].x;
            float sensor_y = transformedKeyPoints[sensor_index].y;

            float dx = map_x - sensor_x;
            float dy = map_y - sensor_y;
            float dist = std::sqrt(dx * dx + dy * dy);
            if (dist <= 5.0f) {
                distance += dist;
            }
        }
    }
};

namespace sim_local {

class ParticleFilter {
  public:
    ParticleFilter(double init_x, double init_y, double init_z, double init_roll, double init_pitch,
                   double init_yaw, int num_particles);

    void update(const nav_msgs::msg::Odometry::ConstSharedPtr& odom_msg, double time_diff);

    void weighting();
    void resampling();

    std::vector<Particle>& getParticles();
    Particle getBestParticle(int top_k);

  private:
    void initializeParticles(double init_x, double init_y, double init_z, double init_roll,
                             double init_pitch, double init_yaw);

    std::vector<Particle> particles_;
};

} // namespace sim_local