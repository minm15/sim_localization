// src/nclt_map_node.cpp
/******************************************************
 NCLTMapNode — lead with velodyne, chain static & dynamic TF:
   world→odom_link   (static from /tf_static)
   odom_link→base_link (dynamic from /tf or inverted)
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
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <deque>
#include <fstream>
#include <optional>
#include <signal.h>
#include <unordered_set>

// Where we write out each keypoint: 180‐dim descriptor + x,y,z
static const std::string descriptorFilePath = "nclt_descriptors.bin";

// Truncate on startup
static void initializeDescriptorFile() {
    std::ofstream f(descriptorFilePath, std::ios::binary | std::ios::trunc);
    if (!f)
        throw std::runtime_error("Failed to create " + descriptorFilePath);
}

// Append (idx + 180‐dim + x,y,z)
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
    : Node("nclt_map_node", opts), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_), have_world_odom_(false),
      have_base_velo_(false), first_frame_(true), frame_count_(0) {
    RCLCPP_INFO(get_logger(), "NCLTMapNode starting up…");
    initializeDescriptorFile();

    // LinK3D extractor
    extractor_ = std::make_shared<LinK3D_SLAM::LinK3D_Extractor>(32, 0.1f, 0.4f, 0.3f, 0.3f, 12, 4, 3);

    // static TF：world→odom_link, base_link→velodyne
    static_tf_sub_ = create_subscription<tf2_msgs::msg::TFMessage>(
        "/tf_static", rclcpp::SystemDefaultsQoS(),
        std::bind(&NCLTMapNode::tfStaticCallback, this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(), "Subscribed to /tf_static");

    // dynamic TF：odom_link↔base_link
    dynamic_tf_sub_ = create_subscription<tf2_msgs::msg::TFMessage>(
        "/tf", rclcpp::SystemDefaultsQoS(), std::bind(&NCLTMapNode::tfDynamicCallback, this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(), "Subscribed to /tf");

    // Velodyne scan
    lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "/velodyne_points", rclcpp::SensorDataQoS(),
        std::bind(&NCLTMapNode::lidarCallback, this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(), "Subscribed to /velodyne_points");
}

void NCLTMapNode::tfStaticCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg) {
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

void NCLTMapNode::tfDynamicCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg) {
    for (auto& ts_in : msg->transforms) {
        bool is_dyn = false;
        auto ts = ts_in; // copy

        // odom_link→base_link
        if (ts.header.frame_id == "odom_link" && ts.child_frame_id == "base_link") {
            is_dyn = true;
        }
        // base_link→odom_link
        else if (ts.header.frame_id == "base_link" && ts.child_frame_id == "odom_link") {
            tf2::Transform T;
            tf2::fromMsg(ts.transform, T);
            ts.transform = tf2::toMsg(T.inverse());
            std::swap(ts.header.frame_id, ts.child_frame_id);
            is_dyn = true;
        }
        if (!is_dyn)
            continue;

        // remove duplicate tf (might be unused in NCLT)
        // uint64_t key = (uint64_t(ts.header.stamp.sec) << 32) | uint64_t(ts.header.stamp.nanosec);
        // if (!processed_tf_.insert(key).second)
        //     continue;

        // insert queue for sorting
        rclcpp::Time t(ts.header.stamp);
        auto it = std::upper_bound(tf_queue_.begin(), tf_queue_.end(), t,
                                   [&](const rclcpp::Time& a, const geometry_msgs::msg::TransformStamped& b) {
                                       return a < rclcpp::Time(b.header.stamp);
                                   });
        tf_queue_.insert(it, ts);
    }
}

void NCLTMapNode::lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    RCLCPP_INFO(get_logger(), "Received scan @ %u.%09u, TFs in queue=%zu", msg->header.stamp.sec,
                msg->header.stamp.nanosec, tf_queue_.size());

    // find the newest tf <= scan velodyne 
    rclcpp::Time scan_t(msg->header.stamp);
    auto it = std::upper_bound(tf_queue_.begin(), tf_queue_.end(), scan_t,
                               [&](const rclcpp::Time& a, const geometry_msgs::msg::TransformStamped& b) {
                                   return a < rclcpp::Time(b.header.stamp);
                               });
    if (it == tf_queue_.begin()) {
        RCLCPP_WARN(get_logger(), "No dynamic TF ≤ scan@%u.%09u", msg->header.stamp.sec, msg->header.stamp.nanosec);
        return;
    }
    --it;
    auto ts = *it;
    // delete the current TF in queue
    tf_queue_.erase(tf_queue_.begin(), it + 1);
	RCLCPP_INFO(get_logger(), "Matched TF msg @ %u.%09u", ts.header.stamp.sec, ts.header.stamp.nanosec);

    // lookup odom_link→base_link @ this TF
    geometry_msgs::msg::TransformStamped dyn;
    try {
        dyn =
            tf_buffer_.lookupTransform("odom_link", "base_link", ts.header.stamp, rclcpp::Duration::from_seconds(0.1));
    } catch (const tf2::TransformException& e) {
        RCLCPP_WARN(get_logger(), "Missing odom_link→base_link@%u.%09u: %s", ts.header.stamp.sec,
                    ts.header.stamp.nanosec, e.what());
        return;
    }

    // full chain world→odom_link→base_link→velodyne
    Eigen::Matrix4f chain = world_T_odom_ * transformMsgToEigen(dyn.transform) * base_T_velo_;

    // to PCL
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::fromROSMsg(*msg, *cloud);

    // extract
    std::vector<pcl::PointXYZ> keypts;
    cv::Mat desc;
    std::vector<int> idx;
    LinK3D_SLAM::MatPt clus;
    (*extractor_)(*cloud, keypts, desc, idx, clus);

    // transform keypts→world
    std::vector<pcl::PointXYZ> wpts;
    wpts.reserve(keypts.size());
    for (auto& p : keypts) {
        Eigen::Vector4f v{p.x, p.y, p.z, 1.0f}, w = chain * v;
        wpts.emplace_back(w.x(), w.y(), w.z());
    }

    // dump unmatched
    if (first_frame_) {
        appendDescriptorsToFile(desc, wpts);
        prevDescriptors_ = desc.clone();
        first_frame_ = false;
        RCLCPP_INFO(get_logger(), "Frame %3zu initial: %4zu pts", frame_count_, size_t(desc.rows));
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
        RCLCPP_INFO(get_logger(), "Frame %3zu +%4zu new", frame_count_, size_t(outdesc.rows));
        prevDescriptors_ = desc.clone();
    }

    ++frame_count_;
}

} // namespace sim_local
