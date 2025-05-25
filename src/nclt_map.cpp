#include "sim_local/nclt_map.hpp"
#include "sim_local/LinK3D_extractor.h"

#include <nav_msgs/msg/odometry.hpp>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include <fstream>
#include <unordered_set>

using sim_local::NCLTMapNode;

// Truncate descriptor file at startup
static void initializeDescriptorFile(const std::string& path) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f)
        throw std::runtime_error("Cannot create " + path);
}

// Append idx + 180‐dim + xyz
static void appendDescriptorsToFile(const std::string& path, const cv::Mat& desc,
                                    const std::vector<pcl::PointXYZ>& pts) {
    if (desc.rows != int(pts.size()))
        throw std::runtime_error("Descriptor/point count mismatch");
    std::ofstream f(path, std::ios::binary | std::ios::app);
    if (!f)
        throw std::runtime_error("Append failed");
    for (int i = 0; i < desc.rows; ++i) {
        int idx = i;
        f.write(reinterpret_cast<const char*>(&idx), sizeof(idx));
        f.write(reinterpret_cast<const char*>(desc.ptr<float>(i)), 180 * sizeof(float));
        auto& p = pts[i];
        f.write(reinterpret_cast<const char*>(&p.x), sizeof(p.x));
        f.write(reinterpret_cast<const char*>(&p.y), sizeof(p.y));
        f.write(reinterpret_cast<const char*>(&p.z), sizeof(p.z));
    }
}

//------------------------------------------------------------------------------
// Constructor
NCLTMapNode::NCLTMapNode(const rclcpp::NodeOptions& opts)
    : Node("nclt_map_node", opts), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_) {
    RCLCPP_INFO(get_logger(), "NCLTMapNode starting…");

    // descriptor file
    descriptor_path_ = "2013_01_10_nclt_descriptors.bin";
    initializeDescriptorFile(descriptor_path_);

    // LinK3D
    extractor_ = std::make_shared<LinK3D_SLAM::LinK3D_Extractor>(32, 0.1f, 0.4f, 0.3f, 0.3f, 12, 4, 5);

    // 1. cache static base_link→velodyne
    static_tf_sub_ = create_subscription<tf2_msgs::msg::TFMessage>(
        "/tf_static", rclcpp::SystemDefaultsQoS(),
        std::bind(&NCLTMapNode::tfStaticCallback, this, std::placeholders::_1));

    // 2. feed dynamic TFs into buffer
    dynamic_tf_sub_ = create_subscription<tf2_msgs::msg::TFMessage>(
        "/tf", rclcpp::SystemDefaultsQoS(), std::bind(&NCLTMapNode::tfDynamicCallback, this, std::placeholders::_1));

    // 3. ground truth
    gt_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        "/ground_truth", rclcpp::SensorDataQoS().keep_last(kGTMax).reliable(),
        std::bind(&NCLTMapNode::groundTruthCallback, this, std::placeholders::_1));

    // 4. LiDAR scans
    lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "/velodyne_points", rclcpp::SensorDataQoS().keep_last(200).reliable(),
        std::bind(&NCLTMapNode::lidarCallback, this, std::placeholders::_1));
}

//------------------------------------------------------------------------------
// Static TF handler: cache base_link→velodyne
void NCLTMapNode::tfStaticCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg) {
    for (auto& ts : msg->transforms) {
        if (!have_base_velo_ && ts.header.frame_id == "base_link" && ts.child_frame_id == "velodyne") {
            base_T_velo_ = transformMsgToEigen(ts.transform);
            have_base_velo_ = true;
            RCLCPP_INFO(get_logger(), "Cached static base_link→velodyne");
        }
    }
}

void NCLTMapNode::tfDynamicCallback(const tf2_msgs::msg::TFMessage::SharedPtr) {
    // no-op
}

//------------------------------------------------------------------------------
// Ground-truth handler: keep recent kGTMax samples
void NCLTMapNode::groundTruthCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    gt_queue_.push_back({msg->header.stamp, msg->pose.pose});
    if (gt_queue_.size() > kGTMax)
        gt_queue_.pop_front();
}

//------------------------------------------------------------------------------
// Main LiDAR callback
void NCLTMapNode::lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // 1) log arrival
    auto& stamp = msg->header.stamp;
    RCLCPP_INFO(get_logger(), "Scan @ %u.%09u", stamp.sec, stamp.nanosec);

    // 2) compute interval
    rclcpp::Time now{stamp};
    if (has_last_scan_) {
        double dt = (now - last_scan_time_).seconds();
        max_scan_interval_sec_ = std::max(max_scan_interval_sec_, dt);
    }
    last_scan_time_ = now;
    has_last_scan_ = true;

    // 3) find latest GT ≤ scan time
    if (gt_queue_.empty()) {
        RCLCPP_WARN(get_logger(), "GT queue empty, skip");
        return;
    }
    auto it = std::upper_bound(gt_queue_.begin(), gt_queue_.end(), stamp,
                               [](const rclcpp::Time& t, const GTSample& s) { return t < s.stamp; });
    if (it == gt_queue_.begin()) {
        RCLCPP_WARN(get_logger(), "No GT ≤ scan@%u.%09u", stamp.sec, stamp.nanosec);
        return;
    }
    --it;
    rclcpp::Time gt_stamp = it->stamp;
    Eigen::Matrix4f T_w_base = poseToEigen(it->pose);

    // 4) require static base→velo
    if (!have_base_velo_) {
        RCLCPP_WARN(get_logger(), "Static base→velo not ready, skip");
        return;
    }

    // 5) chain = world←base (GT) * base→velo
    Eigen::Matrix4f chain = T_w_base * base_T_velo_;

    // 6) extract keypoints & descriptors
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::fromROSMsg(*msg, *cloud);
    std::vector<pcl::PointXYZ> keypts;
    cv::Mat desc;
    std::vector<int> idx;
    LinK3D_SLAM::MatPt clus;
    (*extractor_)(*cloud, keypts, desc, idx, clus);

    // 7) transform to world
    std::vector<pcl::PointXYZ> wpts;
    wpts.reserve(keypts.size());
    for (auto& p : keypts) {
        Eigen::Vector4f v{p.x, p.y, p.z, 1.f}, w = chain * v;
        wpts.emplace_back(w.x(), w.y(), w.z());
    }

    // 8) dump unmatched
    std::vector<pcl::PointXYZ> outp;
    cv::Mat outd;
    if (first_frame_) {
        outd = desc.clone();
        outp = wpts;
        first_frame_ = false;
    } else {
        std::vector<std::pair<int, int>> matches;
        extractor_->matcher(prev_descriptors_, desc, matches);
        std::unordered_set<int> used;
        for (auto& m : matches)
            used.insert(m.second);
        for (int i = 0; i < desc.rows; ++i) {
            if (!used.count(i)) {
                outp.push_back(wpts[i]);
                outd.push_back(desc.row(i));
            }
        }
    }
    prev_descriptors_ = desc.clone();
    if (!outd.empty())
        appendDescriptorsToFile(descriptor_path_, outd, outp);

    // 9) log stamps & kp count
    auto split = [&](const rclcpp::Time& t) {
        int64_t ns = t.nanoseconds();
        return std::make_pair<uint32_t, uint32_t>(uint32_t(ns / 1000000000LL), uint32_t(ns % 1000000000LL));
    };
    auto gt_pair = split(gt_stamp);
    int total_count = desc.rows;
    int unmatched_count = (first_frame_ ? desc.rows : outp.size());
    RCLCPP_INFO(get_logger(), "frame%3zu | lidar %u.%09u | gt %u.%09u | kp %d/%d", frame_count_, stamp.sec,
                stamp.nanosec, gt_pair.first, gt_pair.second, unmatched_count, total_count);

    ++frame_count_;
}

//------------------------------------------------------------------------------
// Helpers

Eigen::Matrix4f NCLTMapNode::transformMsgToEigen(const geometry_msgs::msg::Transform& t) {
    Eigen::Matrix4f M = Eigen::Matrix4f::Identity();
    M(0, 3) = t.translation.x;
    M(1, 3) = t.translation.y;
    M(2, 3) = t.translation.z;
    tf2::Quaternion q{t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w};
    tf2::Matrix3x3 Rm(q);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            M(i, j) = Rm[i][j];
    return M;
}

Eigen::Matrix4f NCLTMapNode::poseToEigen(const geometry_msgs::msg::Pose& p) {
    Eigen::Matrix4f M = Eigen::Matrix4f::Identity();
    M(0, 3) = p.position.x;
    M(1, 3) = p.position.y;
    M(2, 3) = p.position.z;
    tf2::Quaternion q{p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w};
    tf2::Matrix3x3 Rm(q);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            M(i, j) = Rm[i][j];
    return M;
}