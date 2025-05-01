#include "sim_local/Particle_filter.hpp"
#include <algorithm>
#include <random>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

using namespace sim_local;

ParticleFilter::ParticleFilter(double ix, double iy, double iz, double ir, double ip, double iyaw,
                               int num_particles) {
    particles_.resize(num_particles);
    initializeParticles(ix, iy, iz, ir, ip, iyaw);
}

void ParticleFilter::initializeParticles(double ix, double iy, double iz, double ir, double ip,
                                         double iyaw) {
    std::default_random_engine gen(std::random_device{}());
    std::normal_distribution<double> pos_dist(0.0, 0.1);
    std::normal_distribution<double> ang_dist(0.0, 0.01);

    for (auto& p : particles_) {
        p.pose.position.x = ix + pos_dist(gen);
        p.pose.position.y = iy + pos_dist(gen);
        p.pose.position.z = iz + pos_dist(gen);
        tf2::Quaternion q;
        q.setRPY(ir + ang_dist(gen), ip + ang_dist(gen), iyaw + ang_dist(gen));
        p.pose.orientation = tf2::toMsg(q);
        p.distance = 0.0;
        p.weight = 1.0 / particles_.size();
    }
}

void ParticleFilter::update(const nav_msgs::msg::Odometry::ConstSharedPtr& odom_msg, double dt) {
    double vx = odom_msg->twist.twist.linear.x;
    double vy = odom_msg->twist.twist.linear.y;
    double vz = odom_msg->twist.twist.linear.z;
    double rr = odom_msg->twist.twist.angular.x;
    double pr = odom_msg->twist.twist.angular.y;
    double yr = odom_msg->twist.twist.angular.z;

    for (auto& p : particles_) {
        tf2::Quaternion q;
        tf2::fromMsg(p.pose.orientation, q);
        tf2::Quaternion dq;
        dq.setRPY(rr * dt, pr * dt, yr * dt);
        q *= dq;
        q.normalize();
        p.pose.orientation = tf2::toMsg(q);

        tf2::Matrix3x3 R(q);
        Eigen::Matrix3d rot;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                rot(i, j) = R[i][j];

        Eigen::Vector3d vel(vx, vy, vz);
        auto tv = rot * vel;
        p.pose.position.x += tv.x() * dt;
        p.pose.position.y += tv.y() * dt;
        p.pose.position.z += tv.z() * dt;
    }
}

void ParticleFilter::weighting() {
    double sum = 0.0;
    for (auto& p : particles_)
        sum += p.weight;
    if (sum > 0)
        for (auto& p : particles_)
            p.weight /= sum;
}

void ParticleFilter::resampling() {
    if (particles_.empty())
        return;
    // find best
    auto best = getBestParticle(1);
    std::default_random_engine gen(std::random_device{}());
    std::normal_distribution<double> pos_dist(0.0, 0.1);
    std::normal_distribution<double> ang_dist(0.0, 0.01);

    // sort by weight ascending
    std::sort(particles_.begin(), particles_.end(),
              [](auto& a, auto& b) { return a.weight < b.weight; });
    size_t half = particles_.size() / 2;
    for (size_t i = 0; i < half; ++i) {
        auto& p = particles_[particles_.size() - 1 - i];
        p.pose.position.x = best.pose.position.x + pos_dist(gen);
        p.pose.position.y = best.pose.position.y + pos_dist(gen);
        p.pose.position.z = best.pose.position.z + pos_dist(gen);
        tf2::Quaternion q;
        tf2::fromMsg(best.pose.orientation, q);
        double r, pitch, y;
        tf2::Matrix3x3(q).getRPY(r, pitch, y);
        r += ang_dist(gen);
        pitch += ang_dist(gen);
        y += ang_dist(gen);
        q.setRPY(r, pitch, y);
        p.pose.orientation = tf2::toMsg(q);
        p.distance = 0.0;
        p.weight = best.weight;
    }
}

std::vector<Particle>& ParticleFilter::getParticles() { return particles_; }

Particle ParticleFilter::getBestParticle(int k) {
    if (particles_.empty())
        throw std::runtime_error("No particles");
    // pick min distance
    std::vector<Particle> cp = particles_;
    std::sort(cp.begin(), cp.end(), [](auto& a, auto& b) { return a.distance < b.distance; });
    return cp.front();
}