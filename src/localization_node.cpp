// src/localization_node.cpp

#include "sim_local/LinK3D_extractor.h"
#include "sim_local/Particle_filter.hpp"

#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <deque>
#include <fstream>
#include <geometry_msgs/msg/pose_array.hpp>
#include <iomanip>
#include <nav_msgs/msg/odometry.hpp>
#include <opencv2/core.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sstream>
#include <stdexcept>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <vector>

using geometry_msgs::msg::PoseArray;
using nav_msgs::msg::Odometry;
using sensor_msgs::msg::PointCloud2;

class LocalizationNode : public rclcpp::Node {
public:
    LocalizationNode()
        : Node("map_based_localization"),
          frame_count_(0),
          has_last_odom_time_(false) {
        // 1) Declare and retrieve parameters
        this->declare_parameter<std::string>("key_stamp_file", "key_stamp.txt");
        this->declare_parameter<std::string>("descriptor_file", "descriptors.bin");
        this->declare_parameter<std::string>("ground_truth_file", "ground_truth.txt");
        this->declare_parameter<double>("initial_pose_x", 411.303935);
        this->declare_parameter<double>("initial_pose_y", 1180.890379);
        this->declare_parameter<double>("initial_pose_z", 0.0);
        this->declare_parameter<double>("initial_roll", 0.0);
        this->declare_parameter<double>("initial_pitch", 0.0);
        this->declare_parameter<double>("initial_yaw", -1.923645);

        this->get_parameter("key_stamp_file", key_stamp_file_);
        this->get_parameter("descriptor_file", desc_file_);
        this->get_parameter("ground_truth_file", gt_file_);
        this->get_parameter("initial_pose_x", init_x_);
        this->get_parameter("initial_pose_y", init_y_);
        this->get_parameter("initial_pose_z", init_z_);
        this->get_parameter("initial_roll", init_roll_);
        this->get_parameter("initial_pitch", init_pitch_);
        this->get_parameter("initial_yaw", init_yaw_);

        // 2) Load external data files
        loadKeyframeTimestamps(key_stamp_file_);
        vectorDatabase_ = loadBinaryFileToMat(desc_file_);
        readGroundTruthFile(groundTruthData_, gt_file_);

        // 3) Initialize odometry-only pose with full state
        current_pose_.position.x = init_x_;
        current_pose_.position.y = init_y_;
        current_pose_.position.z = init_z_;
        tf2::Quaternion q;
        q.setRPY(init_roll_, init_pitch_, init_yaw_);
        current_pose_.orientation = tf2::toMsg(q);

        // 4) Create LinK3D extractor and ParticleFilter instances
        extractor_ = std::make_shared<LinK3D_SLAM::LinK3D_Extractor>(
            32, 0.1f, 0.4f, 0.3f, 0.3f, 12, 4, 3);
        particle_filter_ = std::make_shared<sim_local::ParticleFilter>(
            init_x_, init_y_, init_z_, init_roll_, init_pitch_, init_yaw_, 128);

        // 5) Create publisher for particle poses
        pub_ = this->create_publisher<PoseArray>("particle_pose", 10);

        // 6) Subscribe to /odom and /LIDAR_TOP topics
        odom_subscriber_ = this->create_subscription<Odometry>(
            "/odom", rclcpp::SystemDefaultsQoS(),
            std::bind(&LocalizationNode::odomCallback, this, std::placeholders::_1));
        lidar_subscriber_ = this->create_subscription<PointCloud2>(
            "/LIDAR_TOP", rclcpp::SensorDataQoS(),
            std::bind(&LocalizationNode::lidarCallback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Localization node initialized.");
    }

private:
    // Load keyframe timestamps as double seconds
    void loadKeyframeTimestamps(const std::string& fn) {
        std::ifstream f(fn);
        if (!f.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Unable to open keyframe file: %s", fn.c_str());
            return;
        }
        double t;
        while (f >> t) {
            keyframe_ts_.push_back(t);
        }
        std::sort(keyframe_ts_.begin(), keyframe_ts_.end());
        RCLCPP_INFO(this->get_logger(), "Loaded %zu keyframes.", keyframe_ts_.size());
    }

    // Load binary descriptor file into OpenCV matrix
    cv::Mat loadBinaryFileToMat(const std::string& fp) {
        std::ifstream file(fp, std::ios::binary);
        if (!file) throw std::runtime_error("Failed to open " + fp);
        std::vector<std::vector<float>> data;
        while (true) {
            int idx;
            file.read(reinterpret_cast<char*>(&idx), sizeof(int));
            if (file.eof()) break;
            std::vector<float> buf(183);
            file.read(reinterpret_cast<char*>(buf.data()), buf.size() * sizeof(float));
            data.push_back(buf);
        }
        cv::Mat m(static_cast<int>(data.size()), 183, CV_32F);
        for (int i = 0; i < static_cast<int>(data.size()); ++i)
            for (int j = 0; j < 183; ++j)
                m.at<float>(i, j) = data[i][j];
        return m;
    }

    // Read ground truth file (x, y, z per line)
    void readGroundTruthFile(
        std::vector<std::vector<double>>& d, const std::string& fp) {
        d.assign(3, {});
        std::ifstream f(fp);
        double x, y, z;
        while (f >> x >> y >> z) {
            d[0].push_back(x);
            d[1].push_back(y);
            d[2].push_back(z);
        }
    }

    // Odometry callback: integrate motion and store history
    void odomCallback(const Odometry::ConstSharedPtr& odom) {
        rclcpp::Time stamp(odom->header.stamp);
        if (!has_last_odom_time_) {
            last_odom_time_ = stamp;
            has_last_odom_time_ = true;
        } else {
            double dt = (stamp - last_odom_time_).seconds();
            last_odom_time_ = stamp;

            // Update orientation
            double vx = odom->twist.twist.linear.x;
            double vy = odom->twist.twist.linear.y;
            double vz = odom->twist.twist.linear.z;
            double rr = odom->twist.twist.angular.x;
            double pr = odom->twist.twist.angular.y;
            double yr = odom->twist.twist.angular.z;
            tf2::Quaternion oq;
            tf2::fromMsg(current_pose_.orientation, oq);
            tf2::Quaternion dq;
            dq.setRPY(rr * dt, pr * dt, yr * dt);
            oq *= dq;
            oq.normalize();
            current_pose_.orientation = tf2::toMsg(oq);

            // Update position in world frame
            tf2::Matrix3x3 mat(oq);
            Eigen::Matrix3d R;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    R(i, j) = mat[i][j];
            Eigen::Vector3d vel(vx, vy, vz);
            auto tv = R * vel * dt;
            current_pose_.position.x += tv.x();
            current_pose_.position.y += tv.y();
            current_pose_.position.z += tv.z();

            // Propagate particle filter motion update
            particle_filter_->update(odom, dt);
        }

        // Push to history buffer
        odom_history_.emplace_back(stamp, current_pose_);
        if (odom_history_.size() > 1000)
            odom_history_.pop_front();
    }

    // LIDAR callback: match timestamp to odometry history and run measurement update
    void lidarCallback(const PointCloud2::ConstSharedPtr& lidar) {
        double t = rclcpp::Time(lidar->header.stamp).seconds();
        // Find nearest ground-truth timestamp within tolerance
        auto it = std::lower_bound(keyframe_ts_.begin(), keyframe_ts_.end(), t);
        double t_gt;
        if (it == keyframe_ts_.end())
            t_gt = keyframe_ts_.back();
        else if (it == keyframe_ts_.begin())
            t_gt = *it;
        else {
            double up = *it;
            double down = *(it - 1);
            t_gt = (std::abs(up - t) < std::abs(down - t)) ? up : down;
        }
        // Only proceed if within 10 ms of a keyframe
        if (std::abs(t - t_gt) > 0.010)
            return;

        // Retrieve odom pose at or before lidar timestamp
        rclcpp::Time lidar_time(lidar->header.stamp);
        bool found = false;
        geometry_msgs::msg::Pose odom_pose;
        for (auto rit = odom_history_.rbegin(); rit != odom_history_.rend(); ++rit) {
            if (rit->first <= lidar_time) {
                odom_pose = rit->second;
                found = true;
                break;
            }
        }
        if (!found) return;
        current_pose_ = odom_pose;

        // Convert ROS msg to PCL and extract features
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*lidar, *cloud);
        std::vector<pcl::PointXYZ> keypts;
        cv::Mat desc;
        std::vector<int> idx;
        LinK3D_SLAM::MatPt clus;
        (*extractor_)(*cloud, keypts, desc, idx, clus);

        // Match descriptors against map database
        std::vector<std::pair<int, int>> vMatched;
        extractor_->matcher(desc, vectorDatabase_, vMatched);

        // Run particle filter measurement update
        for (auto& p : particle_filter_->getParticles()) {
            Eigen::Matrix4f Tm = poseToTransformationMatrix(p.pose);
            auto tks = transformKeyPoints(keypts, Tm);
            p.map_matching(tks, vectorDatabase_, vMatched);
        }
        particle_filter_->weighting();
        auto best = particle_filter_->getBestParticle(1);
        particle_filter_->resampling();

        // Log errors and publish best pose
        if (frame_count_ < groundTruthData_[0].size() - 1) {
            double gx = groundTruthData_[0][frame_count_+1];
            double gy = groundTruthData_[1][frame_count_+1];
            double ox = current_pose_.position.x;
            double oy = current_pose_.position.y;
            double e_odom = std::hypot(ox - gx, oy - gy);
            double px = best.pose.position.x;
            double py = best.pose.position.y;
            double e_part = std::hypot(px - gx, py - gy);
            RCLCPP_INFO(this->get_logger(),
                "Frame %zu\n"
                "  ODOM  err: %6.3f  pos: (%6.3f, %6.3f)\n"
                "  PART  err: %6.3f  pos: (%6.3f, %6.3f)\n"
                "  GT    pos: (%6.3f, %6.3f)",
                frame_count_, e_odom, ox, oy,
                e_part, px, py,
                gx, gy);
        }
        ++frame_count_;

        PoseArray pa;
        pa.header = lidar->header;
        pa.poses.push_back(best.pose);
        pub_->publish(pa);
    }

    // Convert pose to 4x4 transformation matrix
    Eigen::Matrix4f poseToTransformationMatrix(const geometry_msgs::msg::Pose& p) {
        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        T(0,3) = p.position.x;
        T(1,3) = p.position.y;
        T(2,3) = p.position.z;
        tf2::Quaternion q;
        tf2::fromMsg(p.orientation, q);
        tf2::Matrix3x3 m(q);
        Eigen::Matrix3f R;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                R(i,j) = m[i][j];
        T.block<3,3>(0,0) = R;
        return T;
    }

    // Transform keypoints using given transformation
    std::vector<pcl::PointXYZ> transformKeyPoints(
        const std::vector<pcl::PointXYZ>& pts, const Eigen::Matrix4f& T) {
        std::vector<pcl::PointXYZ> out;
        out.reserve(pts.size());
        for (const auto& pt : pts) {
            Eigen::Vector4f v(pt.x, pt.y, pt.z, 1.0f);
            auto w = T * v;
            out.emplace_back(w.x(), w.y(), w.z());
        }
        return out;
    }

    // Member variables
    std::string key_stamp_file_, desc_file_, gt_file_;
    double init_x_, init_y_, init_z_, init_roll_, init_pitch_, init_yaw_;
    std::shared_ptr<LinK3D_SLAM::LinK3D_Extractor> extractor_;
    std::shared_ptr<sim_local::ParticleFilter> particle_filter_;
    cv::Mat vectorDatabase_;
    std::vector<std::vector<double>> groundTruthData_;
    std::vector<double> keyframe_ts_;

    geometry_msgs::msg::Pose current_pose_;
    rclcpp::Time last_odom_time_;
    bool has_last_odom_time_;
    size_t frame_count_;
    std::deque<std::pair<rclcpp::Time, geometry_msgs::msg::Pose>> odom_history_;

    rclcpp::Subscription<Odometry>::SharedPtr odom_subscriber_;
    rclcpp::Subscription<PointCloud2>::SharedPtr lidar_subscriber_;
    rclcpp::Publisher<PoseArray>::SharedPtr pub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LocalizationNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
