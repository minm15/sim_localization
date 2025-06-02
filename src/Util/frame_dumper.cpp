#include "sim_local/frame_dumper.hpp"
#include "sim_local/LinK3D_extractor.h"

#include <filesystem>
#include <fstream>
#include <geometry_msgs/msg/transform.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

using sim_local::FrameDumperNode;

// Pose → Eigen4×4
static Eigen::Matrix4f poseToEigen(const geometry_msgs::msg::Pose& p) {
    Eigen::Matrix4f M = Eigen::Matrix4f::Identity();
    M(0, 3) = p.position.x;
    M(1, 3) = p.position.y;
    M(2, 3) = p.position.z;
    tf2::Quaternion q{p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w};
    tf2::Matrix3x3 R(q);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            M(i, j) = R[i][j];
    return M;
}

namespace sim_local {

/* constructor */
FrameDumperNode::FrameDumperNode(const rclcpp::NodeOptions& opts)
    : Node("frame_dumper_node", opts), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_) {
    // fixed NED→ENU rotation
    // (example: swap X/Y and flip Z; adjust to your convention)
    T_w_o_.setIdentity();
    T_w_o_(0, 0) = 0;
    T_w_o_(0, 1) = 1;
    T_w_o_(1, 0) = 1;
    T_w_o_(1, 1) = 0;
    T_w_o_(2, 2) = -1;

    declare_parameter<std::string>("root_path", ".");

    std::string rp;
    get_parameter("root_path", rp);
    root_path_ = rp;
    frames_path_ = root_path_ / "frames";
    std::filesystem::create_directories(frames_path_);
    RCLCPP_INFO(get_logger(), "Writing frames to %s", frames_path_.c_str());

    extractor_ = std::make_shared<LinK3D_SLAM::LinK3D_Extractor>(32, 0.1f, 0.4f, 0.3f, 0.3f, 12, 4, 3);

    // static TFs
    static_tf_sub_ = create_subscription<tf2_msgs::msg::TFMessage>(
        "/tf_static", rclcpp::SystemDefaultsQoS(),
        std::bind(&FrameDumperNode::tfStaticCallback, this, std::placeholders::_1));

    // dynamic TFs
    dynamic_tf_sub_ = create_subscription<tf2_msgs::msg::TFMessage>(
        "/tf", rclcpp::SystemDefaultsQoS(),
        std::bind(&FrameDumperNode::tfDynamicCallback, this, std::placeholders::_1));

    // ground truth
    gt_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        "/ground_truth", rclcpp::SensorDataQoS().keep_last(kGTMax).reliable(),
        std::bind(&FrameDumperNode::groundTruthCallback, this, std::placeholders::_1));

    // Velodyne
    lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "/velodyne_points", rclcpp::SensorDataQoS().keep_last(200).reliable(),
        std::bind(&FrameDumperNode::lidarCallback, this, std::placeholders::_1));
}

/* static TF callback */
void FrameDumperNode::tfStaticCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg) {
    for (auto& ts : msg->transforms) {
        if (!have_base_velo_ && ts.header.frame_id == "base_link" && ts.child_frame_id == "velodyne") {
            base_T_velo_ = transformMsgToEigen(ts.transform);
            have_base_velo_ = true;
            RCLCPP_INFO(get_logger(), "Cached base_link→velodyne");
        }
    }
}

/* dynamic TF callback (no-op) */
void FrameDumperNode::tfDynamicCallback(const tf2_msgs::msg::TFMessage::SharedPtr) {}

/* GT odometry callback */
void FrameDumperNode::groundTruthCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    gt_queue_.push_back({msg->header.stamp, msg->pose.pose});
    if (gt_queue_.size() > kGTMax)
        gt_queue_.pop_front();
}

/* LiDAR callback */
void FrameDumperNode::lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    auto stamp = msg->header.stamp;

    // 1) lookup odom→base_link
    geometry_msgs::msg::TransformStamped dyn;
    try {
        dyn = tf_buffer_.lookupTransform("odom_link", "base_link", stamp, rclcpp::Duration::from_seconds(0.1));
    } catch (const tf2::TransformException& e) {
        RCLCPP_WARN(get_logger(), "TF lookup failed: %s", e.what());
        return;
    }

    // 2) pick latest GT ≤ LiDAR stamp
    if (gt_queue_.empty()) {
        RCLCPP_WARN(get_logger(), "GT queue empty");
        return;
    }
    // find first > stamp
    auto it = std::upper_bound(gt_queue_.begin(), gt_queue_.end(), stamp,
                               [](const rclcpp::Time& t, const GTSample& s) { return t < s.stamp; });
    if (it == gt_queue_.begin()) {
        RCLCPP_WARN(get_logger(), "No GT ≤ LiDAR %u.%09u", stamp.sec, stamp.nanosec);
        return;
    }
    --it;
    rclcpp::Time gt_time = it->stamp;
    const auto& gt_pose = it->pose;

    // 3) need static base→velo
    if (!have_base_velo_) {
        RCLCPP_WARN(get_logger(), "Missing static base→velo");
        return;
    }

    // 4) compose full chain: NED→ENU * world←base(gt) * base→velo
    Eigen::Matrix4f T_w_base = poseToEigen(gt_pose);
    Eigen::Matrix4f chain = T_w_base        /* world←base */
                            * base_T_velo_; /* base→velo */

    // 5) extract
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::fromROSMsg(*msg, *cloud);
    std::vector<pcl::PointXYZ> keypts;
    cv::Mat desc;
    std::vector<int> idx;
    LinK3D_SLAM::MatPt clus;
    (*extractor_)(*cloud, keypts, desc, idx, clus);

    // 6) transform
    std::vector<pcl::PointXYZ> wpts;
    wpts.reserve(keypts.size());
    for (auto& p : keypts) {
        Eigen::Vector4f v{p.x, p.y, p.z, 1.0f}, w = chain * v;
        wpts.emplace_back(w.x(), w.y(), w.z());
    }

    // 7) write .bin
    auto fname = frames_path_ / ("frame" + std::to_string(frame_count_) + ".bin");
    std::ofstream f(fname, std::ios::binary);
    if (!f) {
        RCLCPP_ERROR(get_logger(), "open %s failed", fname.c_str());
        return;
    }
    for (int i = 0; i < desc.rows; i++) {
        int id = i;
        f.write(reinterpret_cast<char*>(&id), sizeof(id));
        f.write(reinterpret_cast<char*>(desc.ptr<float>(i)), 180 * sizeof(float));
        auto& P = wpts[i];
        f.write(reinterpret_cast<char*>(&P.x), sizeof(P.x));
        f.write(reinterpret_cast<char*>(&P.y), sizeof(P.y));
        f.write(reinterpret_cast<char*>(&P.z), sizeof(P.z));
    }

    // 8) log stamps + GT 6-DOF
    auto split = [&](const rclcpp::Time& t) {
        int64_t ns = t.nanoseconds();
        return std::make_pair<uint32_t, uint32_t>(uint32_t(ns / 1000000000LL), uint32_t(ns % 1000000000LL));
    };
    auto gt_pair = split(gt_time);

    // compute RPY from GT quaternion
    tf2::Quaternion q{gt_pose.orientation.x, gt_pose.orientation.y, gt_pose.orientation.z, gt_pose.orientation.w};
    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

    RCLCPP_INFO(get_logger(),
                "frame %zu | lidar %u.%09u | tf %u.%09u | "
                "gt %u.%09u pos=(%.3f,%.3f,%.3f) rpy=(%.3f,%.3f,%.3f) | kp %d",
                frame_count_, stamp.sec, stamp.nanosec, dyn.header.stamp.sec, dyn.header.stamp.nanosec, gt_pair.first,
                gt_pair.second, gt_pose.position.x, gt_pose.position.y, gt_pose.position.z, roll, pitch, yaw,
                desc.rows);

    ++frame_count_;
}

/* Transform→Eigen */
Eigen::Matrix4f FrameDumperNode::transformMsgToEigen(const geometry_msgs::msg::Transform& t) {
    Eigen::Matrix4f M = Eigen::Matrix4f::Identity();
    M(0, 3) = t.translation.x;
    M(1, 3) = t.translation.y;
    M(2, 3) = t.translation.z;
    tf2::Quaternion q{t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w};
    tf2::Matrix3x3 R(q);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            M(i, j) = R[i][j];
    return M;
}

} // namespace sim_local

/* main */
int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<sim_local::FrameDumperNode>());
    rclcpp::shutdown();
    return 0;
}