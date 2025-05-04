#include "sim_local/nuscenes_map.hpp"
#include <algorithm>
#include <fstream>
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <signal.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>

using namespace sim_local;

static const std::string descriptorFilePath = "descriptors.bin";

static void initializeDescriptorFile() {
    std::ofstream f(descriptorFilePath, std::ios::binary | std::ios::trunc);
    if (!f)
        throw std::runtime_error("Failed to create " + descriptorFilePath);
}

static void appendDescriptorsToFile(const cv::Mat& desc, const std::vector<pcl::PointXYZ>& pts) {
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

//--- ctor
NuscenesMapNode::NuscenesMapNode(const rclcpp::NodeOptions& opts)
    : Node("map_combined", opts), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_),
      first_frame_(true), frame_count_(0) {
    initializeDescriptorFile();

    extractor_ =
        std::make_shared<LinK3D_SLAM::LinK3D_Extractor>(32, 0.1f, 0.4f, 0.3f, 0.3f, 12, 4, 3);

    lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "/LIDAR_TOP", rclcpp::SensorDataQoS(),
        std::bind(&NuscenesMapNode::lidarCallback, this, std::placeholders::_1));

    tf_sub_ = create_subscription<tf2_msgs::msg::TFMessage>(
        "/tf", rclcpp::SystemDefaultsQoS(),
        std::bind(&NuscenesMapNode::tfCallback, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "map_combined node ready");
}

void NuscenesMapNode::lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    auto t = rclcpp::Time(msg->header.stamp);
    lidar_history_.emplace_back(t, msg);
    if (lidar_history_.size() > 1000u)
        lidar_history_.pop_front();
    processPendingTFs();
}

void NuscenesMapNode::tfCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg) {
    for (auto& ts : msg->transforms) {
        if (ts.header.frame_id != "map" || ts.child_frame_id != "base_link")
            continue;
        uint64_t key = (uint64_t(ts.header.stamp.sec) << 32) | uint64_t(ts.header.stamp.nanosec);
        if (!processed_tf_.insert(key).second)
            continue;
        rclcpp::Time t(ts.header.stamp);
        auto it = std::upper_bound(
            tf_queue_.begin(), tf_queue_.end(), t,
            [&](auto const& a, auto const& b) { return a < rclcpp::Time(b.header.stamp); });
        tf_queue_.insert(it, ts);
    }
    processPendingTFs();
}

void NuscenesMapNode::processPendingTFs() {
    while (!tf_queue_.empty()) {
        auto& ts = tf_queue_.front();
        rclcpp::Time tf_t(ts.header.stamp);
        if (lidar_history_.empty() || lidar_history_.back().first < tf_t)
            break;
        processOneTF(ts);
        tf_queue_.pop_front();
    }
}

template <typename T>
std::optional<T> NuscenesMapNode::findLastBefore(const std::deque<T>& buf, const rclcpp::Time& t) {
    if (buf.empty())
        return {};
    auto it = std::lower_bound(buf.begin(), buf.end(), t,
                               [](auto const& a, auto const& ts) { return a.first < ts; });
    if (it == buf.begin()) {
        if (it->first == t)
            return *it;
        return {};
    }
    if (it == buf.end() || it->first > t)
        --it;
    return *it;
}

void NuscenesMapNode::processOneTF(const geometry_msgs::msg::TransformStamped& ts) {
    auto tf_t = rclcpp::Time(ts.header.stamp);
    auto lidar_opt = findLastBefore(lidar_history_, tf_t);
    if (!lidar_opt) {
        RCLCPP_WARN(get_logger(), "No LIDAR for TF@%u.%09u", ts.header.stamp.sec,
                    ts.header.stamp.nanosec);
        return;
    }
    auto [lidar_t, lidar_msg] = *lidar_opt;

    // PCL
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*lidar_msg, *cloud);

    // extract
    std::vector<pcl::PointXYZ> keypts;
    cv::Mat desc;
    std::vector<int> idx;
    LinK3D_SLAM::MatPt clus;
    (*extractor_)(*cloud, keypts, desc, idx, clus);

    // to map
    Eigen::Matrix4f M = transformMsgToEigen(ts.transform);
    std::vector<pcl::PointXYZ> map_pts;
    map_pts.reserve(keypts.size());
    for (auto& p : keypts) {
        auto v = Eigen::Vector4f{p.x, p.y, p.z, 1.0f};
        auto w = M * v;
        map_pts.emplace_back(w.x(), w.y(), w.z());
    }

    if (first_frame_) {
        appendDescriptorsToFile(desc, map_pts);
        RCLCPP_INFO(get_logger(), "Frame %3zu  TF@%u.%09u  new:%4zu (initial)", frame_count_,
                    ts.header.stamp.sec, ts.header.stamp.nanosec, static_cast<size_t>(desc.rows));
        first_frame_ = false;
        prevDescriptors_ = desc.clone();
    } else {
        std::vector<std::pair<int, int>> matches;
        extractor_->matcher(prevDescriptors_, desc, matches);
        std::unordered_set<int> mset;
        for (auto& m : matches)
            mset.insert(m.second);
        std::vector<pcl::PointXYZ> unkp;
        cv::Mat undesc;
        for (int i = 0; i < desc.rows; ++i) {
            if (!mset.count(i)) {
                unkp.push_back(map_pts[i]);
                undesc.push_back(desc.row(i));
            }
        }
        if (!undesc.empty())
            appendDescriptorsToFile(undesc, unkp);
        RCLCPP_INFO(get_logger(), "Frame %3zu  TF@%u.%09u  new:%4zu", frame_count_,
                    ts.header.stamp.sec, ts.header.stamp.nanosec, static_cast<size_t>(undesc.rows));
        prevDescriptors_ = desc.clone();
    }
    ++frame_count_;
}
