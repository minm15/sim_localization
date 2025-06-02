#include "sim_local/Particle_filter.hpp"

#include <algorithm>
#include <numeric>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

using namespace sim_local;

// initialize
void Particle::map_matching(std::vector<pcl::PointXYZ>& transformedKeyPoints, cv::Mat& map_descriptors,
                            std::vector<std::pair<int, int>>& vMatched) {
    distance = 0.0;
    int success = 0;
    for (auto& m : vMatched) {
        int mi = m.second, si = m.first;
        float mx = map_descriptors.at<float>(mi, 180);
        float my = map_descriptors.at<float>(mi, 181);
        float sx = transformedKeyPoints[si].x;
        float sy = transformedKeyPoints[si].y;
        float d = std::hypot(mx - sx, my - sy);
        
        if (d <= 2.0f) {
            distance += d;
            ++success;
        }
    }
    auto logger = rclcpp::get_logger("particle_filter");
    RCLCPP_INFO(logger, "matches: %d / %zu", success, vMatched.size());
}

void Particle::printInfo(const char* str) {
    auto logger = rclcpp::get_logger("particle_filter");
    RCLCPP_INFO(logger, "%s dist: %.3f", str, distance);
}

// helper: convert geometry_msgs::Pose to Sophus::SE3f
static Sophus::SE3f poseMsgToSE3(const geometry_msgs::msg::Pose& p) {
    Eigen::Quaternionf q{float(p.orientation.w), float(p.orientation.y), float(p.orientation.x),
                         float(p.orientation.z)};
    Eigen::Vector3f t{float(p.position.y), float(p.position.x), float(p.position.z)};
    return Sophus::SE3f{q, t};
}

// helper: convert Sophus::SE3f back to geometry_msgs::Pose
static geometry_msgs::msg::Pose se3ToPoseMsg(const Sophus::SE3f& g) {
    geometry_msgs::msg::Pose p;
    auto q = g.unit_quaternion();
    p.orientation.w = q.w();
    p.orientation.x = q.y();
    p.orientation.y = q.x();
    p.orientation.z = q.z();
    auto t = g.translation();
    p.position.x = t.y();
    p.position.y = t.x();
    p.position.z = t.z();
    return p;
}

ParticleFilter::ParticleFilter(double ix, double iy, double iz, double ir, double ip, double iyaw, int num_particles) {
    particles_.resize(num_particles);
    std::normal_distribution<double> nz(0.0, 0.01), nr(0.0, 0.005);
    for (auto& p : particles_) {
        p.pose.position.x = ix;
        p.pose.position.y = iy;
        p.pose.position.z = iz + nz(gen_);

        tf2::Quaternion qq;
        qq.setRPY(ir + nr(gen_), ip + nr(gen_), iyaw + nr(gen_));
        p.pose.orientation = tf2::toMsg(qq);

        p.weight = 1.0 / num_particles;
        p.distance = 0.0;
    }
}

void ParticleFilter::update(const nav_msgs::msg::Odometry::ConstSharedPtr& odom,
                            const sensor_msgs::msg::Imu::ConstSharedPtr& imu, double dt) {
    // cache covariance
    for (int i = 0; i < 36; ++i) {
        odom_pose_cov_[i] = odom->pose.covariance[i];
        odom_twist_cov_[i] = odom->twist.covariance[i];
    }
    have_odom_cov_ = true;

    double vx = odom->twist.twist.linear.x;
    double wz = imu->angular_velocity.z;
    predict(vx, wz, dt);
}

void ParticleFilter::predict(double vx, double wz, double dt) {
    // invert to match ROS coordinate frame
    wz = -wz;

    // compute noise standard deviations
    double std_v = 0.05, std_w = 0.01;
    if (have_odom_cov_) {
        std_v = std::sqrt(std::max(odom_twist_cov_[0], 1e-6));
        std_w = std::sqrt(std::max(odom_twist_cov_[35], 1e-6));
    }

    const float truncated_angular_std = std_w * std::clamp(static_cast<float>(std::sqrt(std::abs(vx))), 0.1f, 1.0f);
    const float truncated_linear_std = std::clamp(static_cast<float>(std_v * vx), 0.1f, 2.0f);

    for (auto &particle : particles_) {
        Sophus::SE3f se3_pose = poseMsgToSE3(particle.pose);
        Eigen::Matrix<float, 6, 1> noised_xi;
        noised_xi.setZero();
        noised_xi(0) = vx + nrand(truncated_linear_std);
        noised_xi(5) = wz + nrand(truncated_angular_std);
        se3_pose *= Sophus::SE3f::exp(noised_xi * dt);

        geometry_msgs::msg::Pose pose = se3ToPoseMsg(se3_pose);
        particle.pose = pose;
    }
}

void ParticleFilter::weighting() {
    constexpr double eps = 1e-6;
    double sum = 0;
    for (auto& p : particles_) {
        p.weight = 1.0 / (p.distance + eps);
        sum += p.weight;
    }
    if (sum < eps) {
        double u = 1.0 / particles_.size();
        for (auto& p : particles_)
            p.weight = u;
    } else {
        for (auto& p : particles_)
            p.weight /= sum;
    }
}

void ParticleFilter::resampling() {
    int N = static_cast<int>(particles_.size());
    if (N == 0) return;

    // 1. pick the best particle
    Particle best = getBestParticle(1);
    int nr = N / 3;

    // 2. sort by weight ascending and collect indices of worst particles
    std::vector<std::pair<double,int>> widx;
    widx.reserve(N);
    for (int i = 0; i < N; i++)
        widx.emplace_back(particles_[i].weight, i);
    std::sort(widx.begin(), widx.end()); // weights from small to large

    // 3. for each of the worst nr particles, clone the best and add jitter
    std::normal_distribution<double> nd_xyz(0.0, 0.022);    // position jitter σ≈2cm
    std::normal_distribution<double> nd_rpy(0.0, 0.001);    // orientation jitter σ≈0.01rad (~0.6°)
    for (int k = 0; k < nr; k++) {
        int idx = widx[k].second;
        // 3a. copy best
        particles_[idx] = best;
        // 3b. jitter position
        particles_[idx].pose.position.x += nd_xyz(gen_);
        particles_[idx].pose.position.y += nd_xyz(gen_);
        // 3c. jitter orientation
        tf2::Quaternion q;
        tf2::fromMsg(particles_[idx].pose.orientation, q);
        double r,p,y;
        tf2::Matrix3x3(q).getRPY(r,p,y);
        r += nd_rpy(gen_);
        p += nd_rpy(gen_);
        y += nd_rpy(gen_);
        tf2::Quaternion q2;
        q2.setRPY(r,p,y);
        particles_[idx].pose.orientation = tf2::toMsg(q2);
    }

    // 4. reset all weights
    double uw = 1.0 / N;
    for (auto &p : particles_)
        p.weight = uw;
}

std::vector<Particle>& ParticleFilter::getParticles() { return particles_; }

Particle ParticleFilter::getBestParticle(int) {
    // choose max-weight
    return *std::max_element(particles_.begin(), particles_.end(),
                             [](auto& a, auto& b) { return a.weight < b.weight; });
}