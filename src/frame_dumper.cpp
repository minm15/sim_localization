/******************************************************
 FrameDumperNode — writes one bin per Velodyne scan:

 0. Reads “root_path” parameter and mkdir <root>/frames/
 1. Caches static world→odom_link & base_link→velodyne from /tf_static
 2. For each /velodyne_points:
    • log scan stamp
    • lookup odom_link→base_link @ scan stamp (via tf_buffer_)
    • log matched TF stamp
    • extract LinK3D keypoints+descriptors
    • transform keypoints→world
    • dump idx+180‐dim+xyz into <root>/frames/frame{N}.bin
******************************************************/

#include "sim_local/frame_dumper.hpp"
#include "sim_local/LinK3D_extractor.h"

#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_msgs/msg/tf_message.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>

#include <filesystem>
#include <fstream>

static Eigen::Matrix4f transformMsgToEigen(const geometry_msgs::msg::Transform& t) {
    Eigen::Matrix4f M = Eigen::Matrix4f::Identity();
    M(0, 3) = t.translation.x;
    M(1, 3) = t.translation.y;
    M(2, 3) = t.translation.z;
    tf2::Quaternion q{t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w};
    tf2::Matrix3x3 R(q);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            M(i, j) = R[i][j];
    return M;
}

namespace sim_local {

FrameDumperNode::FrameDumperNode(const rclcpp::NodeOptions& opts)
    : Node("frame_dumper_node", opts), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_), have_world_odom_(false),
      have_base_velo_(false), frame_count_(0) {
    // 0. setup root & frames directory
    declare_parameter<std::string>("root_path", ".");
    std::string rp;
    get_parameter("root_path", rp);
    root_path_ = std::filesystem::path(rp);
    frames_path_ = root_path_ / "frames";
    std::filesystem::create_directories(frames_path_);
    RCLCPP_INFO(get_logger(), "Writing frame files to %s", frames_path_.c_str());

    // LinK3D extractor
    extractor_ = std::make_shared<LinK3D_SLAM::LinK3D_Extractor>(32, 0.1f, 0.4f, 0.3f, 0.3f, 12, 4, 3);

    // 1. static TFs
    static_tf_sub_ = create_subscription<tf2_msgs::msg::TFMessage>(
        "/tf_static", rclcpp::SystemDefaultsQoS(),
        std::bind(&FrameDumperNode::tfStaticCallback, this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(), "Subscribed to /tf_static");

    // 2. dynamic TFs feed into tf_buffer_ automatically
    dynamic_tf_sub_ = create_subscription<tf2_msgs::msg::TFMessage>(
        "/tf", rclcpp::SystemDefaultsQoS(),
        std::bind(&FrameDumperNode::tfDynamicCallback, this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(), "Subscribed to /tf");

    // 3. Velodyne scans
    lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "/velodyne_points", rclcpp::SensorDataQoS().keep_last(200).reliable(),
        std::bind(&FrameDumperNode::lidarCallback, this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(), "Subscribed to /velodyne_points");
}

void FrameDumperNode::tfStaticCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg) {
    // cache world→odom_link & base_link→velodyne
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

void FrameDumperNode::tfDynamicCallback(const tf2_msgs::msg::TFMessage::SharedPtr /*msg*/) {
    // No-op: TransformListener auto-populates tf_buffer_
}

void FrameDumperNode::lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    // log scan arrival
    RCLCPP_INFO(get_logger(), "Received scan @ %u.%09u", msg->header.stamp.sec, msg->header.stamp.nanosec);

    // ensure statics
    if (!have_world_odom_ || !have_base_velo_) {
        RCLCPP_WARN(get_logger(), "Static transforms not ready, skipping frame");
        return;
    }

    // lookup odom_link→base_link at scan time
    geometry_msgs::msg::TransformStamped dyn;
    try {
        dyn = tf_buffer_.lookupTransform("odom_link", "base_link", msg->header.stamp,
                                         rclcpp::Duration::from_seconds(0.1));
    } catch (const tf2::TransformException& e) {
        RCLCPP_WARN(get_logger(), "Could not lookup dynamic TF @ %u.%09u: %s", msg->header.stamp.sec,
                    msg->header.stamp.nanosec, e.what());
        return;
    }
    RCLCPP_INFO(get_logger(), "Matched TF @ %u.%09u", dyn.header.stamp.sec, dyn.header.stamp.nanosec);

    // build full chain
    Eigen::Matrix4f chain = world_T_odom_ * transformMsgToEigen(dyn.transform) * base_T_velo_;

    // convert and extract
    auto cloud = std::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    pcl::fromROSMsg(*msg, *cloud);
    std::vector<pcl::PointXYZ> keypts;
    cv::Mat desc;
    std::vector<int> idx;
    LinK3D_SLAM::MatPt clus;
    (*extractor_)(*cloud, keypts, desc, idx, clus);

    // transform keypoints→world
    std::vector<pcl::PointXYZ> wpts;
    wpts.reserve(keypts.size());
    for (auto& p : keypts) {
        Eigen::Vector4f v{p.x, p.y, p.z, 1.0f}, w = chain * v;
        wpts.emplace_back(w.x(), w.y(), w.z());
    }

    // dump to per-frame bin
    auto filename = frames_path_ / ("frame" + std::to_string(frame_count_) + ".bin");
    std::ofstream f{filename, std::ios::binary};
    if (!f) {
        RCLCPP_ERROR(get_logger(), "Failed to open %s for writing", filename.c_str());
        return;
    }
    for (int i = 0; i < desc.rows; ++i) {
        int ii = i;
        f.write(reinterpret_cast<const char*>(&ii), sizeof(ii));
        f.write(reinterpret_cast<const char*>(desc.ptr<float>(i)), 180 * sizeof(float));
        auto& P = wpts[i];
        f.write(reinterpret_cast<const char*>(&P.x), sizeof(P.x));
        f.write(reinterpret_cast<const char*>(&P.y), sizeof(P.y));
        f.write(reinterpret_cast<const char*>(&P.z), sizeof(P.z));
    }
    RCLCPP_INFO(get_logger(), "Wrote frame %zu → %s (%zu keypoints)", frame_count_, filename.c_str(),
                (size_t)desc.rows);
    ++frame_count_;
}

} // namespace sim_local

// -- standalone main() --------------------------------------------------------
int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<sim_local::FrameDumperNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
