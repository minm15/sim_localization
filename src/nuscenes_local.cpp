#include "sim_local/nuscenes_local.hpp"

#include <cmath>
#include <fstream>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

namespace sim_local {

NuscenesNode::NuscenesNode(const rclcpp::NodeOptions& opts)
    : Node("map_based_localization", opts), frame_count_(0), has_last_odom_time_(false) {
    // parameters
    declare_parameter<std::string>("descriptor_file", "nuscene_descriptors.bin");
    declare_parameter<double>("initial_pose_x", 411.303935);
    declare_parameter<double>("initial_pose_y", 1180.890379);
    declare_parameter<double>("initial_pose_z", 0.0);
    declare_parameter<double>("initial_roll", 0.0);
    declare_parameter<double>("initial_pitch", 0.0);
    declare_parameter<double>("initial_yaw", -1.923645);

    get_parameter("descriptor_file", desc_file_);
    get_parameter("initial_pose_x", init_x_);
    get_parameter("initial_pose_y", init_y_);
    get_parameter("initial_pose_z", init_z_);
    get_parameter("initial_roll", init_roll_);
    get_parameter("initial_pitch", init_pitch_);
    get_parameter("initial_yaw", init_yaw_);

    // load descriptor DB
    vectorDatabase_ = loadBinaryFileToMat(desc_file_);

    // init odom‐only pose
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

    // extractor & PF
    extractor_ = std::make_shared<LinK3D_SLAM::LinK3D_Extractor>(32, 0.1f, 0.4f, 0.3f, 0.3f, 12, 4, 3);
    particle_filter_ =
        std::make_shared<ParticleFilter>(init_x_, init_y_, init_z_, init_roll_, init_pitch_, init_yaw_, 128);

    // TF buffer/listener (just to keep alive)
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // publisher
    pub_ = create_publisher<PoseArray>("particle_pose", 10);

    // subscriptions
    odom_sub_ = create_subscription<Odometry>("/odom", rclcpp::SystemDefaultsQoS(),
                                              std::bind(&NuscenesNode::odomCallback, this, std::placeholders::_1));
    lidar_sub_ = create_subscription<PointCloud2>("/LIDAR_TOP", rclcpp::SensorDataQoS(),
                                                  std::bind(&NuscenesNode::lidarCallback, this, std::placeholders::_1));
    tf_sub_ = create_subscription<TFMessage>("/tf", rclcpp::SystemDefaultsQoS(),
                                             std::bind(&NuscenesNode::tfCallback, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "Localization node initialized.");
}

cv::Mat NuscenesNode::loadBinaryFileToMat(const std::string& fp) {
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

void NuscenesNode::odomCallback(const Odometry::SharedPtr odom) {
    rclcpp::Time t(odom->header.stamp);
    if (!has_last_odom_time_) {
        last_odom_time_ = t;
        has_last_odom_time_ = true;
    } else {
        double dt = (t - last_odom_time_).seconds();
        last_odom_time_ = t;

        tf2::Quaternion oq(current_pose_.orientation.x, current_pose_.orientation.y, current_pose_.orientation.z,
                           current_pose_.orientation.w);
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

void NuscenesNode::lidarCallback(const PointCloud2::SharedPtr cloud) {
    rclcpp::Time t(cloud->header.stamp);
    lidar_history_.emplace_back(t, cloud);
    if (lidar_history_.size() > 1000)
        lidar_history_.pop_front();
    processPendingTFs();
}

void NuscenesNode::tfCallback(const TFMessage::SharedPtr msg) {
    for (auto& ts : msg->transforms) {
        if (ts.header.frame_id != "map" || ts.child_frame_id != "base_link")
            continue;
        uint64_t key = (uint64_t(ts.header.stamp.sec) << 32) | uint64_t(ts.header.stamp.nanosec);
        if (!processed_tf_.insert(key).second)
            continue;
        rclcpp::Time t(ts.header.stamp);
        auto it = std::upper_bound(
            tf_queue_.begin(), tf_queue_.end(), t,
            [&](const rclcpp::Time& a, const TransformStamped& b) { return a < rclcpp::Time(b.header.stamp); });
        tf_queue_.insert(it, ts);
    }
}

void NuscenesNode::processPendingTFs() {
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

void NuscenesNode::processOneTF(const TransformStamped& ts) {
    rclcpp::Time tf_time(ts.header.stamp);
    Eigen::Matrix4f M = transformMsgToEigen(ts.transform);
    double gx = M(0, 3), gy = M(1, 3);

    auto odom_opt = findLastBefore(odom_history_, tf_time); // find the last odom data <= TF's timestamp
    auto lidar_opt = findLastBefore(lidar_history_, tf_time);
    if (!odom_opt || !lidar_opt) {
        RCLCPP_WARN(get_logger(), "Missing buffered odom or lidar ≤ TF, skipping");
        return;
    }
    auto [odom_t, odom_pose] = *odom_opt;
    auto [lidar_t, lidar_msg] = *lidar_opt;

    // feature extraction
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*lidar_msg, *cloud);
    RCLCPP_INFO(get_logger(), "raw cloud: height=%u, width=%u (points=%zu)", cloud->height, cloud->width, cloud->points.size());
    std::vector<pcl::PointXYZ> keypts;
    cv::Mat desc;
    std::vector<int> idx;
    LinK3D_SLAM::MatPt clus;
    (*extractor_)(*cloud, keypts, desc, idx, clus);

    // descriptor matching + PF update
    // desciptor match
    std::vector<std::pair<int, int>> matches;
    extractor_->matcher(desc, vectorDatabase_, matches);
    // for each particle, do geometry match computation
    for (auto& p : particle_filter_->getParticles()) {
        Eigen::Matrix4f Tp = poseToEigen(p.pose);
        auto tks = transformKeyPoints(keypts, Tp);
        p.map_matching(tks, vectorDatabase_, matches);
    }
    particle_filter_->weighting();
    // print log
    // particle_filter_->printParticleInfo();
    auto best = particle_filter_->getBestParticle(1);
    particle_filter_->resampling();


    // compute errors
    double ox = odom_pose.position.x, oy = odom_pose.position.y;
    double e_o = std::hypot(ox - gx, oy - gy);
    double px = best.pose.position.x, py = best.pose.position.y;
    double e_p = std::hypot(px - gx, py - gy);
    
    // output log info
    logFrameInfo(ts.header.stamp, odom_t, lidar_t, e_o, ox, oy, e_p, px, py, gx, gy);

    PoseArray pa;
    pa.header.stamp = tf_time;
    pa.header.frame_id = "map";
    pa.poses.push_back(best.pose);
    pub_->publish(pa);
}

void NuscenesNode::logFrameInfo(const rclcpp::Time& tf_t, const rclcpp::Time& odom_t, const rclcpp::Time& lidar_t,
                                double e_o, double ox, double oy, double e_p, double px, double py, double gx,
                                double gy) {
    // helper to split secs and nanosecs
    auto stamp_pair = [&](const rclcpp::Time& tt) {
        int64_t ns = tt.nanoseconds();
        return std::make_pair<uint32_t, uint32_t>(uint32_t(ns / 1000000000), uint32_t(ns % 1000000000));
    };
    // extract tf, odom, lidar
    auto [tf_s, tf_ns] = stamp_pair(tf_t);
    auto [o_s, o_ns] = stamp_pair(odom_t);
    auto [l_s, l_ns] = stamp_pair(lidar_t);

    RCLCPP_INFO(get_logger(),
                "Frame %zu\n"
                "TF   stamp: %u.%09u\n"
                "ODOM stamp: %u.%09u\n"
                "LIDAR stamp: %u.%09u\n"
                "  ODOM err:%5.3f pos:(%6.3f,%6.3f)\n"
                "  PART err:%5.3f pos:(%6.3f,%6.3f)\n"
                "  GT   pos:(%6.3f,%6.3f)",
                frame_count_++, tf_s, tf_ns, o_s, o_ns, l_s, l_ns, e_o, ox, oy, e_p, px, py, gx, gy);
}

template <typename BufferT>
std::optional<typename BufferT::value_type> NuscenesNode::findLastBefore(const BufferT& buf, const rclcpp::Time& t) {
    using PairT = typename BufferT::value_type;
    if (buf.empty())
        return std::nullopt;

    auto it = std::lower_bound(buf.begin(), buf.end(), t,
                               [](const PairT& a, const rclcpp::Time& ts) { return a.first < ts; });

    if (it == buf.begin()) {
        if (it->first == t)
            return *it;
        return std::nullopt;
    }
    if (it == buf.end()) {
        return buf.back();
    }
    if (it->first == t) {
        return *it;
    }
    return *std::prev(it);
}

Eigen::Matrix4f NuscenesNode::transformMsgToEigen(const geometry_msgs::msg::Transform& t) {
    Eigen::Matrix4f M = Eigen::Matrix4f::Identity();
    M(0, 3) = t.translation.x;
    M(1, 3) = t.translation.y;
    M(2, 3) = t.translation.z;
    tf2::Quaternion q(t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w);
    tf2::Matrix3x3 Rm(q);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            M(i, j) = Rm[i][j];
    return M;
}

Eigen::Matrix4f NuscenesNode::poseToEigen(const geometry_msgs::msg::Pose& p) {
    geometry_msgs::msg::Transform t;
    t.translation.x = p.position.x;
    t.translation.y = p.position.y;
    t.translation.z = p.position.z;
    t.rotation = p.orientation;
    return transformMsgToEigen(t);
}

std::vector<pcl::PointXYZ> NuscenesNode::transformKeyPoints(const std::vector<pcl::PointXYZ>& pts,
                                                            const Eigen::Matrix4f& T) {
    std::vector<pcl::PointXYZ> out;
    out.reserve(pts.size());
    for (auto& pt : pts) {
        Eigen::Vector4f v(pt.x, pt.y, pt.z, 1.0f);
        Eigen::Vector4f w = T * v;
        out.emplace_back(w.x(), w.y(), w.z());
    }
    return out;
}

} // namespace sim_local
