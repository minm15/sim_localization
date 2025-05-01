// src/localization_node.cpp

#include "sim_local/LinK3D_extractor.h"
#include "sim_local/Particle_filter.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <deque>
#include <fstream>
#include <geometry_msgs/msg/pose_array.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <opencv2/core.hpp>
#include <optional>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_msgs/msg/tf_message.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <unordered_set>

using geometry_msgs::msg::PoseArray;
using nav_msgs::msg::Odometry;
using sensor_msgs::msg::PointCloud2;
using tf2_msgs::msg::TFMessage;

class LocalizationNode : public rclcpp::Node {
  public:
    LocalizationNode()
        : Node("map_based_localization"), frame_count_(0), has_last_odom_time_(false) {
        // 1) parameters
        declare_parameter<std::string>("descriptor_file", "descriptors.bin");
        declare_parameter<double>("initial_pose_x", 409.743152);
        declare_parameter<double>("initial_pose_y", 1176.676973);
        declare_parameter<double>("initial_pose_z", 0.0);
        declare_parameter<double>("initial_roll", 0.0);
        declare_parameter<double>("initial_pitch", 0.0);
        declare_parameter<double>("initial_yaw", -1.917831);

        get_parameter("descriptor_file", desc_file_);
        get_parameter("initial_pose_x", init_x_);
        get_parameter("initial_pose_y", init_y_);
        get_parameter("initial_pose_z", init_z_);
        get_parameter("initial_roll", init_roll_);
        get_parameter("initial_pitch", init_pitch_);
        get_parameter("initial_yaw", init_yaw_);

        // 2) load descriptor database
        vectorDatabase_ = loadBinaryFileToMat(desc_file_);

        // 3) initialize odom‐only pose
        current_pose_.position.x = init_x_;
        current_pose_.position.y = init_y_;
        current_pose_.position.z = init_z_;
        {
            tf2::Quaternion q;
            q.setRPY(init_roll_, init_pitch_, init_yaw_);
            current_pose_.orientation.x = q.x();
            current_pose_.orientation.y = q.y();
            current_pose_.orientation.z = q.z();
            current_pose_.orientation.w = q.w();
        }

        // 4) extractor & particle filter
        extractor_ =
            std::make_shared<LinK3D_SLAM::LinK3D_Extractor>(32, 0.1f, 0.4f, 0.3f, 0.3f, 12, 4, 3);
        particle_filter_ = std::make_shared<sim_local::ParticleFilter>(
            init_x_, init_y_, init_z_, init_roll_, init_pitch_, init_yaw_, 128);

        // 5) TF listener (unused)
        tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        // 6) publisher
        pub_ = create_publisher<PoseArray>("particle_pose", 10);

        // 7) subscribers
        odom_sub_ = create_subscription<Odometry>(
            "/odom", rclcpp::SystemDefaultsQoS(),
            std::bind(&LocalizationNode::odomCallback, this, std::placeholders::_1));
        lidar_sub_ = create_subscription<PointCloud2>(
            "/LIDAR_TOP", rclcpp::SensorDataQoS(),
            std::bind(&LocalizationNode::lidarCallback, this, std::placeholders::_1));
        tf_sub_ = create_subscription<TFMessage>(
            "/tf", rclcpp::SystemDefaultsQoS(),
            std::bind(&LocalizationNode::tfCallback, this, std::placeholders::_1));

        RCLCPP_INFO(get_logger(), "Localization node initialized.");
    }

  private:
    // load binary descriptors
    cv::Mat loadBinaryFileToMat(const std::string& fp) {
        std::ifstream file(fp, std::ios::binary);
        if (!file)
            throw std::runtime_error("Failed to open " + fp);
        std::vector<std::vector<float>> data;
        while (true) {
            int idx;
            file.read(reinterpret_cast<char*>(&idx), sizeof(int));
            if (file.eof())
                break;
            std::vector<float> buf(183);
            file.read(reinterpret_cast<char*>(buf.data()), buf.size() * sizeof(float));
            data.push_back(buf);
        }
        cv::Mat m((int)data.size(), 183, CV_32F);
        for (int i = 0; i < (int)data.size(); ++i)
            for (int j = 0; j < 183; ++j)
                m.at<float>(i, j) = data[i][j];
        return m;
    }

    // odom callback: integrate & buffer
    void odomCallback(const Odometry::SharedPtr odom) {
        rclcpp::Time t(odom->header.stamp);
        if (!has_last_odom_time_) {
            last_odom_time_ = t;
            has_last_odom_time_ = true;
        } else {
            double dt = (t - last_odom_time_).seconds();
            last_odom_time_ = t;
            tf2::Quaternion oq(current_pose_.orientation.x, current_pose_.orientation.y,
                               current_pose_.orientation.z, current_pose_.orientation.w);
            const auto& twist = odom->twist.twist;
            tf2::Quaternion dq;
            dq.setRPY(twist.angular.x * dt, twist.angular.y * dt, twist.angular.z * dt);
            oq = oq * dq;
            oq.normalize();
            current_pose_.orientation.x = oq.x();
            current_pose_.orientation.y = oq.y();
            current_pose_.orientation.z = oq.z();
            current_pose_.orientation.w = oq.w();
            tf2::Matrix3x3 Rm(oq);
            Eigen::Matrix3d R;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    R(i, j) = Rm[i][j];
            Eigen::Vector3d v{twist.linear.x, twist.linear.y, twist.linear.z};
            auto dp = R * v * dt;
            current_pose_.position.x += dp.x();
            current_pose_.position.y += dp.y();
            current_pose_.position.z += dp.z();
            particle_filter_->update(odom, dt);
        }
        odom_history_.emplace_back(t, current_pose_);
        if (odom_history_.size() > 1000)
            odom_history_.pop_front();
        processPendingTFs();
    }

    // lidar callback: buffer only
    void lidarCallback(const PointCloud2::SharedPtr cloud) {
        rclcpp::Time t(cloud->header.stamp);
        lidar_history_.emplace_back(t, cloud);
        if (lidar_history_.size() > 1000)
            lidar_history_.pop_front();
        processPendingTFs();
    }

    // tf callback: enqueue
    void tfCallback(const TFMessage::SharedPtr msg) {
        for (auto& ts : msg->transforms) {
            if (ts.header.frame_id != "map" || ts.child_frame_id != "base_link")
                continue;
            uint64_t key =
                (uint64_t(ts.header.stamp.sec) << 32) | uint64_t(ts.header.stamp.nanosec);
            if (!processed_tf_.insert(key).second)
                continue;
            rclcpp::Time t(ts.header.stamp);
            auto it = std::upper_bound(
                tf_queue_.begin(), tf_queue_.end(), t,
                [&](const rclcpp::Time& a, const geometry_msgs::msg::TransformStamped& b) {
                    return a < rclcpp::Time(b.header.stamp);
                });
            tf_queue_.insert(it, ts);
        }
    }

    // drain any TFs once both buffers caught up
    void processPendingTFs() {
        while (!tf_queue_.empty()) {
            auto& ts = tf_queue_.front();
            rclcpp::Time t(ts.header.stamp);
            if (odom_history_.empty() || lidar_history_.empty())
                break;
            if (odom_history_.back().first < t || lidar_history_.back().first < t)
                break;
            processOneTF(ts);
            tf_queue_.pop_front();
        }
    }

    // actual TF processing
    void processOneTF(const geometry_msgs::msg::TransformStamped& ts) {
        rclcpp::Time tf_time(ts.header.stamp);
        Eigen::Matrix4f M = transformMsgToEigen(ts.transform);
        double gx = M(0, 3), gy = M(1, 3);

        auto odom_opt = findLastBefore(odom_history_, tf_time);
        auto lidar_opt = findLastBefore(lidar_history_, tf_time);
        if (!odom_opt || !lidar_opt) {
            RCLCPP_WARN(get_logger(), "Missing buffered odom or lidar ≤ TF, skipping");
            return;
        }
        auto [odom_t, odom_pose] = *odom_opt;
        auto [lidar_t, lidar_msg] = *lidar_opt;

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*lidar_msg, *cloud);
        std::vector<pcl::PointXYZ> keypts;
        cv::Mat desc;
        std::vector<int> idx;
        LinK3D_SLAM::MatPt clus;
        (*extractor_)(*cloud, keypts, desc, idx, clus);

        std::vector<std::pair<int, int>> vM;
        extractor_->matcher(desc, vectorDatabase_, vM);

        for (auto& p : particle_filter_->getParticles()) {
            Eigen::Matrix4f Tp = poseToEigen(p.pose);
            auto tks = transformKeyPoints(keypts, Tp);
            p.map_matching(tks, vectorDatabase_, vM);
        }
        particle_filter_->weighting();
        auto best = particle_filter_->getBestParticle(1);
        particle_filter_->resampling();

        double ox = odom_pose.position.x, oy = odom_pose.position.y;
        double e_o = std::hypot(ox - gx, oy - gy);
        double px = best.pose.position.x, py = best.pose.position.y;
        double e_p = std::hypot(px - gx, py - gy);

        auto to_pair = [&](const rclcpp::Time& tt) {
            int64_t ns = tt.nanoseconds();
            return std::pair<uint32_t, uint32_t>{uint32_t(ns / 1000000000),
                                                 uint32_t(ns % 1000000000)};
        };
        auto [o_s, o_ns] = to_pair(odom_t);
        auto [l_s, l_ns] = to_pair(lidar_t);

        RCLCPP_INFO(get_logger(),
                    "Frame %zu\n"
                    "TF   stamp: %u.%09u\n"
                    "ODOM stamp: %u.%09u\n"
                    "LIDAR stamp: %u.%09u\n"
                    "  ODOM err:%5.3f pos:(%6.3f,%6.3f)\n"
                    "  PART err:%5.3f pos:(%6.3f,%6.3f)\n"
                    "  GT   pos:(%6.3f,%6.3f)",
                    frame_count_++, ts.header.stamp.sec, ts.header.stamp.nanosec, o_s, o_ns, l_s,
                    l_ns, e_o, ox, oy, e_p, px, py, gx, gy);

        PoseArray pa;
        pa.header.stamp = tf_time;
        pa.header.frame_id = "map";
        pa.poses.push_back(best.pose);
        pub_->publish(pa);
    }

    // helper to find last ≤ t in a time-ordered buffer
    template <typename BufferT>
    std::optional<typename BufferT::value_type> findLastBefore(const BufferT& buf,
                                                               const rclcpp::Time& t) {
        using PairT = typename BufferT::value_type;
        if (buf.empty())
            return std::nullopt;
        auto it =
            std::lower_bound(buf.begin(), buf.end(), t,
                             [](const PairT& a, const rclcpp::Time& ts) { return a.first < ts; });
        if (it == buf.begin()) {
            if (it->first == t)
                return *it;
            return std::nullopt;
        }
        if (it == buf.end()) {
            return buf.back();
        }
        if (it->first == t)
            return *it;
        return *std::prev(it);
    }

    // msg→Eigen4x4
    Eigen::Matrix4f transformMsgToEigen(const geometry_msgs::msg::Transform& t) {
        Eigen::Matrix4f M = Eigen::Matrix4f::Identity();
        M(0, 3) = t.translation.x;
        M(1, 3) = t.translation.y;
        M(2, 3) = t.translation.z;
        // **fixed**: build quaternion then use single‐arg constructor
        tf2::Quaternion q(t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w);
        tf2::Matrix3x3 Rm(q);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                M(i, j) = Rm[i][j];
        return M;
    }

    Eigen::Matrix4f poseToEigen(const geometry_msgs::msg::Pose& p) {
        geometry_msgs::msg::Transform T;
        T.translation.x = p.position.x;
        T.translation.y = p.position.y;
        T.translation.z = p.position.z;
        T.rotation = p.orientation;
        return transformMsgToEigen(T);
    }

    std::vector<pcl::PointXYZ> transformKeyPoints(const std::vector<pcl::PointXYZ>& pts,
                                                  const Eigen::Matrix4f& T) {
        std::vector<pcl::PointXYZ> out;
        out.reserve(pts.size());
        for (auto& pt : pts) {
            Eigen::Vector4f v(pt.x, pt.y, pt.z, 1.0f);
            auto w = T * v;
            out.emplace_back(w.x(), w.y(), w.z());
        }
        return out;
    }

    // members
    std::string desc_file_;
    double init_x_, init_y_, init_z_, init_roll_, init_pitch_, init_yaw_;
    cv::Mat vectorDatabase_;
    std::shared_ptr<LinK3D_SLAM::LinK3D_Extractor> extractor_;
    std::shared_ptr<sim_local::ParticleFilter> particle_filter_;

    geometry_msgs::msg::Pose current_pose_;
    rclcpp::Time last_odom_time_;
    bool has_last_odom_time_;
    size_t frame_count_;

    std::deque<std::pair<rclcpp::Time, geometry_msgs::msg::Pose>> odom_history_;
    std::deque<std::pair<rclcpp::Time, PointCloud2::SharedPtr>> lidar_history_;
    std::deque<geometry_msgs::msg::TransformStamped> tf_queue_;
    std::unordered_set<uint64_t> processed_tf_;

    std::unique_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    rclcpp::Subscription<Odometry>::SharedPtr odom_sub_;
    rclcpp::Subscription<PointCloud2>::SharedPtr lidar_sub_;
    rclcpp::Subscription<TFMessage>::SharedPtr tf_sub_;
    rclcpp::Publisher<PoseArray>::SharedPtr pub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<LocalizationNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
