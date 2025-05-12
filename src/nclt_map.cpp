/******************************************************
 NCLTMapNode — lead with velodyne, chain static & dynamic TF:
   world→odom_link   (static from /tf_static)
   odom_link→base_link (dynamic via TF2 buffer lookup)
   base_link→velodyne (static from /tf_static)
 Then LinK3D extract & unmatched→descriptors.bin in world frame.
******************************************************/

#include "sim_local/nclt_map.hpp"
#include "sim_local/LinK3D_extractor.h"

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include <fstream>
#include <unordered_set>

// where we dump our 180‐dim + xyz keypoints
static const std::string descriptorFilePath = "nclt_descriptors.bin";

// truncate at startup
static void initializeDescriptorFile() {
    std::ofstream f(descriptorFilePath, std::ios::binary | std::ios::trunc);
    if (!f)
        throw std::runtime_error("Failed to create " + descriptorFilePath);
}

// append idx + 180 floats + xyz
static void appendDescriptorsToFile(const cv::Mat& desc, const std::vector<pcl::PointXYZ>& pts) {
    if (desc.rows != int(pts.size()))
        throw std::runtime_error("Descriptor/point count mismatch");
    std::ofstream f(descriptorFilePath, std::ios::binary | std::ios::app);
    if (!f)
        throw std::runtime_error("Failed to open for append");
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

// geometry_msgs→Eigen 4×4
static Eigen::Matrix4f transformMsgToEigen(const geometry_msgs::msg::Transform& t) {
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

namespace sim_local {

NCLTMapNode::NCLTMapNode(const rclcpp::NodeOptions& opts)
    : Node("nclt_map_node", opts), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_) {
    RCLCPP_INFO(get_logger(), "NCLTMapNode starting up…");
    initializeDescriptorFile();

    // one LinK3D extractor
    extractor_ = std::make_shared<LinK3D_SLAM::LinK3D_Extractor>(32, 0.1f, 0.4f, 0.3f, 0.3f, 12, 4, 3);

    // static TF subscription only to cache our two needed statics
    static_tf_sub_ = create_subscription<tf2_msgs::msg::TFMessage>(
        "/tf_static", rclcpp::SystemDefaultsQoS(),
        std::bind(&NCLTMapNode::tfStaticCallback, this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(), "Subscribed to /tf_static");

    // dynamic TF subscription simply to feed tf_buffer_
    dynamic_tf_sub_ = create_subscription<tf2_msgs::msg::TFMessage>(
        "/tf", rclcpp::SystemDefaultsQoS(), std::bind(&NCLTMapNode::tfDynamicCallback, this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(), "Subscribed to /tf");

    // Velodyne scans
    auto qos = rclcpp::SensorDataQoS()
                   .keep_last(200) // buffer up to 50 scans
                   .reliable();
    lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "/velodyne_points", qos, std::bind(&NCLTMapNode::lidarCallback, this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(), "Subscribed to /velodyne_points");
}

void NCLTMapNode::tfStaticCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg) {
    // cache world→odom_link and base_link→velodyne
    for (auto& ts : msg->transforms) {
        if (!have_world_odom_ && ts.header.frame_id == "world" && ts.child_frame_id == "odom_link") {
            world_T_odom_ = transformMsgToEigen(ts.transform);
            have_world_odom_ = true;
            RCLCPP_INFO(get_logger(), "Cached static world→odom_link");
        }
        if (!have_base_velo_ && ts.header.frame_id == "base_link" && ts.child_frame_id == "velodyne") {
            base_T_velo_ = transformMsgToEigen(ts.transform);
            have_base_velo_ = true;
            RCLCPP_INFO(get_logger(), "Cached static base_link→velodyne");
        }
    }
}

void NCLTMapNode::tfDynamicCallback(const tf2_msgs::msg::TFMessage::SharedPtr /*msg*/) {
    // no-op: tf2_ros::TransformListener already fills tf_buffer_
}

void NCLTMapNode::lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // 1) log scan arrival
    RCLCPP_INFO(get_logger(), "Received scan @ %u.%09u", msg->header.stamp.sec, msg->header.stamp.nanosec);

    // 2) compute and report scan‐interval, update max
    rclcpp::Time current_scan(msg->header.stamp);
    if (has_last_scan_) {
        double dt = (current_scan - last_scan_time_).seconds();
        max_scan_interval_sec_ = std::max(max_scan_interval_sec_, dt);
        RCLCPP_INFO(get_logger(), "  this interval: %.6f sec, max so far: %.6f sec", dt, max_scan_interval_sec_);
    }
    last_scan_time_ = current_scan;
    has_last_scan_ = true;

    // 3) lookup the dynamic odom_link→base_link at exactly this scan time
    if (!have_world_odom_ || !have_base_velo_) {
        RCLCPP_WARN(get_logger(), "Static transforms not yet ready, skipping");
        return;
    }
    geometry_msgs::msg::TransformStamped dyn;
    try {
        // request the transform at the scan timestamp
        dyn = tf_buffer_.lookupTransform("odom_link", "base_link", msg->header.stamp,
                                         rclcpp::Duration::from_seconds(0.1));
    } catch (const tf2::TransformException& e) {
        RCLCPP_WARN(get_logger(), "Could not lookup odom_link→base_link@%u.%09u: %s", msg->header.stamp.sec,
                    msg->header.stamp.nanosec, e.what());
        return;
    }
    // 4) log the matched TF stamp
    RCLCPP_INFO(get_logger(), "Matched TF @ %u.%09u", dyn.header.stamp.sec, dyn.header.stamp.nanosec);

    // 5) form full chain world→odom→base→velodyne
    Eigen::Matrix4f chain = world_T_odom_ * transformMsgToEigen(dyn.transform) * base_T_velo_;

    // 6) convert to PCL and extract
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::fromROSMsg(*msg, *cloud);
    RCLCPP_INFO(get_logger(), "raw cloud: height=%u, width=%u (points=%zu)", cloud->height, cloud->width, cloud->points.size());
    std::vector<pcl::PointXYZ> keypts;
    cv::Mat desc;
    std::vector<int> idx;
    LinK3D_SLAM::MatPt clus;
    (*extractor_)(*cloud, keypts, desc, idx, clus);

    // 7) transform keypoints into world frame
    std::vector<pcl::PointXYZ> wpts;
    wpts.reserve(keypts.size());
    for (auto& p : keypts) {
        Eigen::Vector4f v{p.x, p.y, p.z, 1.0f}, w = chain * v;
        wpts.emplace_back(w.x(), w.y(), w.z());
    }

    // 8) dump new descriptors exactly as before
    if (first_frame_) {
        appendDescriptorsToFile(desc, wpts);
        prevDescriptors_ = desc.clone();
        first_frame_ = false;
        RCLCPP_INFO(get_logger(), "Frame %3zu initial: %d pts", frame_count_, desc.rows);
    } else {
        std::vector<std::pair<int, int>> matches;
        extractor_->matcher(prevDescriptors_, desc, matches);
        std::unordered_set<int> used;
        for (auto& m : matches)
            used.insert(m.second);

        cv::Mat outdesc;
        std::vector<pcl::PointXYZ> outkp;
        for (int i = 0; i < desc.rows; ++i) {
            if (!used.count(i)) {
                outkp.push_back(wpts[i]);
                outdesc.push_back(desc.row(i));
            }
        }
        if (!outdesc.empty()) {
            appendDescriptorsToFile(outdesc, outkp);
        }
        RCLCPP_INFO(get_logger(), "Frame %3zu +%d new", frame_count_, outdesc.rows);
        prevDescriptors_ = desc.clone();
    }

    ++frame_count_;
}

} // namespace sim_local
