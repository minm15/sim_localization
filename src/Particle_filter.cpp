#include "sim_local/Particle_filter.hpp"

#include <algorithm>
#include <numeric>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

using namespace sim_local;

// init
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

ParticleFilter::ParticleFilter(double ix, double iy, double iz, double ir, double ip, double iyaw, int num_particles) {
    particles_.resize(num_particles);
    initializeParticles(ix, iy, iz, ir, ip, iyaw);
}

void ParticleFilter::initializeParticles(double ix, double iy, double iz, double ir, double ip, double iyaw) {
    std::normal_distribution<double> nz(0.0, 0.01);
    std::normal_distribution<double> nr(0.0, 0.005);
    std::normal_distribution<double> np(0.0, 0.005);

    for (auto& p : particles_) {
        p.pose.position.x = ix;
        p.pose.position.y = iy;
        p.pose.position.z = iz + nz(gen_);

        double roll = ir + nr(gen_);
        double pitch = ip + np(gen_);
        double yaw = iyaw + nr(gen_);

        tf2::Quaternion q;
        q.setRPY(roll, pitch, yaw);
        p.pose.orientation = tf2::toMsg(q);

        p.distance = 0.0;
        p.weight = 1.0 / particles_.size();
    }
}

// update with imu and odometry
void ParticleFilter::update(const nav_msgs::msg::Odometry::ConstSharedPtr& odom,
                            const sensor_msgs::msg::Imu::ConstSharedPtr& imu, double dt) {
    for (int i = 0; i < 36; ++i) {
        odom_pose_cov_[i] = odom->pose.covariance[i];
        odom_twist_cov_[i] = odom->twist.covariance[i];
    }
    have_odom_cov_ = true;

    double vx = odom->twist.twist.linear.x;
    double wz = imu->angular_velocity.z;

    predict(vx, wz, dt);
}

// SE(2) + z/r/p noise
void ParticleFilter::predict(double vx, double wz, double dt) {
    double std_vx = 0.05, std_wz = 0.01;
    if (have_odom_cov_) {
        std_vx = std::sqrt(std::max(odom_twist_cov_[0], 1e-6)) * std::sqrt(dt);
        std_wz = std::sqrt(std::max(odom_twist_cov_[35], 1e-6)) * std::sqrt(dt);
    }
    std_vx = std::clamp(std_vx, 1e-3, 1.0);
    std_wz = std::clamp(std_wz, 1e-3, 0.3);

    {
        double min_d = (wz - std_wz) * dt;
        double max_d = (wz + std_wz) * dt;
        RCLCPP_INFO(rclcpp::get_logger("particle_filter"),
                    "predict(): dt=%.4f raw_wz=%.4f σ_wz=%.4f ⇒ yaw ∈ [%.4f, %.4f]", dt, wz, std_wz, min_d, max_d);
    }

    std::normal_distribution<double> n_vx(0.0, std_vx);
    std::normal_distribution<double> n_wz(0.0, std_wz);
    std::normal_distribution<double> n_z(0.0, 0.005);
    std::normal_distribution<double> n_rp(0.0, 0.002);

    for (size_t i = 0; i < particles_.size(); ++i) {
        auto& p = particles_[i];

        tf2::Quaternion q0;
        tf2::fromMsg(p.pose.orientation, q0);
        double roll, pitch, yaw;
        tf2::Matrix3x3(q0).getRPY(roll, pitch, yaw);

        double v = vx + n_vx(gen_);
        double wzn = wz + n_wz(gen_);

        p.pose.position.x += v * std::cos(yaw) * dt;
        p.pose.position.y += v * std::sin(yaw) * dt;
        yaw += wzn * dt;

        p.pose.position.z += n_z(gen_);
        roll += n_rp(gen_);
        pitch += n_rp(gen_);

        tf2::Quaternion q1;
        q1.setRPY(roll, pitch, yaw);
        p.pose.orientation = tf2::toMsg(q1);

        if (i < 4) {
            RCLCPP_INFO(rclcpp::get_logger("particle_filter"),
                        "[P%zu] dt=%.3f vx=%.3f wz=%.3f → x=%.3f y=%.3f z=%.3f r=%.3f p=%.3f y=%.3f", i, dt, vx, wz,
                        p.pose.position.x, p.pose.position.y, p.pose.position.z, roll, pitch, yaw);
        }
    }
}

// weighting / resampling / helpers
void ParticleFilter::weighting() {
    constexpr double eps = 1e-6;
    double sumw = 0.0;
    for (auto& p : particles_) {
        p.weight = 1.0 / (p.distance + eps);
        sumw += p.weight;
    }
    if (sumw < eps) {
        double u = 1.0 / particles_.size();
        for (auto& p : particles_)
            p.weight = u;
    } else {
        for (auto& p : particles_)
            p.weight /= sumw;
    }
}

void ParticleFilter::resampling() { systematicResample(); }

void ParticleFilter::systematicResample() {
    int N = static_cast<int>(particles_.size());
    std::vector<Particle> old = particles_;
    std::vector<double> cdf(N);
    cdf[0] = old[0].weight;
    for (int i = 1; i < N; ++i)
        cdf[i] = cdf[i - 1] + old[i].weight;
    double tot = cdf.back();
    for (auto& v : cdf)
        v /= tot;

    std::uniform_real_distribution<double> uni(0.0, 1.0 / N);
    double r = uni(gen_);
    int idx = 0;
    for (int m = 0; m < N; ++m) {
        double u = r + double(m) / N;
        while (u > cdf[idx])
            ++idx;
        particles_[m] = old[idx];
        particles_[m].weight = 1.0 / N;
    }
}

std::vector<Particle>& ParticleFilter::getParticles() { return particles_; }

Particle ParticleFilter::getBestParticle(int) {
    return *std::min_element(particles_.begin(), particles_.end(),
                             [](auto& a, auto& b) { return a.distance < b.distance; });
}

void ParticleFilter::printParticleInfo(int i) {
    auto& p = particles_[i];
    tf2::Quaternion q;
    tf2::fromMsg(p.pose.orientation, q);
    double r, pit, y;
    tf2::Matrix3x3(q).getRPY(r, pit, y);
    RCLCPP_INFO(rclcpp::get_logger("particle_filter"), "Particle[%d] → x=%.3f y=%.3f z=%.3f r=%.3f p=%.3f y=%.3f", i,
                p.pose.position.x, p.pose.position.y, p.pose.position.z, r, pit, y);
}