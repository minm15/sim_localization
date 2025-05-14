// src/nclt_local.cpp
#include "sim_local/nclt_local.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

namespace sim_local {

NcltNode::NcltNode(const rclcpp::NodeOptions& opts) : Node("nclt_localization", opts) {
    // -- parameters --
    declare_parameter<std::string>("descriptor_file", "2012_01_15_nclt_descriptors.bin");
    declare_parameter<double>("initial_pose_x", 0.430);
    declare_parameter<double>("initial_pose_y", -0.486);
    declare_parameter<double>("initial_pose_z", 0.0);
    declare_parameter<double>("initial_roll", -0.023);
    declare_parameter<double>("initial_pitch", 0.001);
    declare_parameter<double>("initial_yaw", -0.123);
    get_parameter("descriptor_file", desc_file_);
    get_parameter("initial_pose_x", init_x_);
    get_parameter("initial_pose_y", init_y_);
    get_parameter("initial_pose_z", init_z_);
    get_parameter("initial_roll", init_roll_);
    get_parameter("initial_pitch", init_pitch_);
    get_parameter("initial_yaw", init_yaw_);

    // -- load map descriptors --
    vectorDatabase_ = loadBinaryFileToMat(desc_file_);
	if (vectorDatabase_.empty()) {
		RCLCPP_ERROR(get_logger(), "[ERROR] vectorDatabase_ is empty!");
		return;
	}
    int R = vectorDatabase_.rows;
    int C = vectorDatabase_.cols;
    RCLCPP_INFO(get_logger(), "[DEBUG] Loaded descriptor DB: rows=%d, cols=%d", R, C);

    // build kdtree
    kdtree = std::make_unique<Kdtree>(vectorDatabase_, 10);

    // -- initialize pose & PF --
    {
        tf2::Quaternion q;
        q.setRPY(init_roll_, init_pitch_, init_yaw_);
        current_pose_.position.x = init_x_;
        current_pose_.position.y = init_y_;
        current_pose_.position.z = init_z_;
        current_pose_.orientation.x = q.x();
        current_pose_.orientation.y = q.y();
        current_pose_.orientation.z = q.z();
        current_pose_.orientation.w = q.w();
    }
	extractor_ = std::make_shared<LinK3D_SLAM::LinK3D_Extractor>(32, 0.1f, 0.4f, 0.3f, 0.3f, 12, 4, 3);
    particle_filter_ =
        std::make_shared<ParticleFilter>(init_x_, init_y_, init_z_, init_roll_, init_pitch_, init_yaw_, 128);

    // -- subscriptions --
    static_tf_sub_ = create_subscription<tf2_msgs::msg::TFMessage>(
        "/tf_static", rclcpp::SystemDefaultsQoS(), std::bind(&NcltNode::tfStaticCallback, this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(), "[DEBUG] Subscribed to /tf_static");

    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        "/odom", rclcpp::SystemDefaultsQoS(), std::bind(&NcltNode::odomCallback, this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(), "[DEBUG] Subscribed to /odom");

    gt_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        "/ground_truth", rclcpp::SystemDefaultsQoS(),
        std::bind(&NcltNode::groundTruthCallback, this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(), "[DEBUG] Subscribed to /ground_truth");

    // lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
    //     "/velodyne_points", rclcpp::SensorDataQoS().keep_last(200).reliable(),
    //     std::bind(&NcltNode::lidarCallback, this, std::placeholders::_1));
	auto qos = rclcpp::SensorDataQoS()
                   .keep_last(200) // buffer up to 50 scans
                   .reliable();
    lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "/velodyne_points", qos, std::bind(&NcltNode::lidarCallback, this, std::placeholders::_1));
    RCLCPP_INFO(get_logger(), "[DEBUG] Subscribed to /velodyne_points");

    pub_ = create_publisher<geometry_msgs::msg::PoseArray>("particle_pose", 10);

    RCLCPP_INFO(get_logger(), "NCLT localization node started.");
}

void NcltNode::tfStaticCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg) {
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

void NcltNode::odomCallback(const nav_msgs::msg::Odometry::SharedPtr odom) {
    rclcpp::Time t(odom->header.stamp);
    if (!has_last_odom_time_) {
        last_odom_time_ = t;
        has_last_odom_time_ = true;
    } else {
        double dt = (t - last_odom_time_).seconds();
        last_odom_time_ = t;

        // integrate angular
        tf2::Quaternion oq{current_pose_.orientation.x, current_pose_.orientation.y, current_pose_.orientation.z,
                           current_pose_.orientation.w};
        const auto& tw = odom->twist.twist;
        tf2::Quaternion dq;
        dq.setRPY(tw.angular.x * dt, tw.angular.y * dt, tw.angular.z * dt);
        oq = (oq * dq).normalized();
        current_pose_.orientation.x = oq.x();
        current_pose_.orientation.y = oq.y();
        current_pose_.orientation.z = oq.z();
        current_pose_.orientation.w = oq.w();

        // integrate linear
        tf2::Matrix3x3 Rm(oq);
        Eigen::Matrix3d R;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                R(i, j) = Rm[i][j];
        Eigen::Vector3d v{tw.linear.x, tw.linear.y, tw.linear.z};
        auto dp = R * v * dt;
        current_pose_.position.x += dp.x();
        current_pose_.position.y += dp.y();
        current_pose_.position.z += dp.z();

        particle_filter_->update(odom, dt);
    }

    odom_history_.emplace_back(t, current_pose_);
    if (odom_history_.size() > 1000)
        odom_history_.pop_front();
}

void NcltNode::groundTruthCallback(const nav_msgs::msg::Odometry::SharedPtr gt) {
    // ground_truth is already in "world" frame
    rclcpp::Time t(gt->header.stamp);
    gt_history_.emplace_back(t, gt->pose.pose);
    if (gt_history_.size() > 1000)
        gt_history_.pop_front();
}

void NcltNode::lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (!have_world_odom_ || !have_base_velo_) {
        RCLCPP_WARN(get_logger(), "Static TF not ready, skipping scan");
        return;
    }
    uint32_t s = msg->header.stamp.sec;
    uint32_t ns = msg->header.stamp.nanosec;
    RCLCPP_INFO(get_logger(), "Scan @ %u.%09u", s, ns);
    rclcpp::Time scan_t(msg->header.stamp);

    auto odom_opt = findLastBefore(odom_history_, scan_t);
    auto gt_opt = findLastBefore(gt_history_, scan_t);
    if (!odom_opt || !gt_opt) {
        RCLCPP_WARN(get_logger(), "No odom/gt ≤ scan, skipping");
        return;
    }

    auto [odom_t, odom_pose] = *odom_opt;
    auto [gt_t, gt_pose] = *gt_opt;

    // extract LinK3D features
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud);
	// RCLCPP_INFO(get_logger(), "raw cloud: height=%u, width=%u (points=%zu)", cloud->height, cloud->width, cloud->points.size());
    std::vector<pcl::PointXYZ> keypts;
    cv::Mat desc;
    std::vector<int> idx;
    LinK3D_SLAM::MatPt clus;
    (*extractor_)(*cloud, keypts, desc, idx, clus);

    // PF transform and match
    for (auto& p : particle_filter_->getParticles()) {
        Eigen::Matrix4f Tp = poseToEigen(p.pose);
        auto tks = transformKeyPoints(keypts, Tp);
        auto knn_indices = kdtree->queryKNN(tks);

        // LinK3D matching
        std::vector<std::pair<int, int>> matches;
        extractor_->matcher(desc, vectorDatabase_, matches, knn_indices);

        // particle geometry matching 
        p.map_matching(tks, vectorDatabase_, matches);
    }
    particle_filter_->weighting();
    // print log
    particle_filter_->printParticleInfo();
    auto best = particle_filter_->getBestParticle(1);
    particle_filter_->resampling();


    // compute XY error
    double ex = best.pose.position.x - gt_pose.position.x;
    double ey = best.pose.position.y - gt_pose.position.y;
    double err_xy = std::hypot(ex, ey);

    // log + publish
    logFrameInfo(scan_t, odom_t, gt_t, err_xy, best.pose, gt_pose);
    geometry_msgs::msg::PoseArray pa;
    pa.header.stamp = scan_t;
    pa.header.frame_id = "world";
    pa.poses.push_back(best.pose);
    pub_->publish(pa);

    frame_count_++;
}

cv::Mat NcltNode::loadBinaryFileToMat(const std::string& fp) {
    std::ifstream f(fp, std::ios::binary);
    if (!f)
        throw std::runtime_error("Failed to open " + fp);
    std::vector<std::vector<float>> D;
    while (true) {
        int idx;
        f.read(reinterpret_cast<char*>(&idx), sizeof(idx));
        if (!f || f.eof())
            break;
        std::vector<float> buf(183);
        f.read(reinterpret_cast<char*>(buf.data()), buf.size() * sizeof(float));
        D.push_back(buf);
    }
    cv::Mat M((int)D.size(), 183, CV_32F);
    for (int i = 0; i < (int)D.size(); i++)
        for (int j = 0; j < 183; j++)
            M.at<float>(i, j) = D[i][j];
    return M;
}

template <typename BufferT>
std::optional<typename BufferT::value_type> NcltNode::findLastBefore(const BufferT& buf, const rclcpp::Time& t) {
    if (buf.empty())
        return std::nullopt;
    using PairT = typename BufferT::value_type;
    auto it = std::lower_bound(buf.begin(), buf.end(), t,
                               [](const PairT& a, const rclcpp::Time& ts) { return a.first < ts; });
    if (it == buf.begin()) {
        if (it->first == t)
            return *it;
        return std::nullopt;
    }
    if (it == buf.end() || it->first > t)
        --it;
    return *it;
}

Eigen::Matrix4f NcltNode::transformMsgToEigen(const geometry_msgs::msg::Transform& t) {
    Eigen::Matrix4f M = Eigen::Matrix4f::Identity();
    M(0, 3) = t.translation.x;
    M(1, 3) = t.translation.y;
    M(2, 3) = t.translation.z;
    tf2::Quaternion q{t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w};
    tf2::Matrix3x3 Rm(q);
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            M(i, j) = Rm[i][j];
    return M;
}

Eigen::Matrix4f NcltNode::poseToEigen(const geometry_msgs::msg::Pose& p) {
    geometry_msgs::msg::Transform t;
    t.translation.x = p.position.x;
    t.translation.y = p.position.y;
    t.translation.z = p.position.z;
    t.rotation = p.orientation;
    return transformMsgToEigen(t);
}

std::vector<pcl::PointXYZ> NcltNode::transformKeyPoints(const std::vector<pcl::PointXYZ>& pts,
                                                        const Eigen::Matrix4f& T) {
    std::vector<pcl::PointXYZ> out;
    out.reserve(pts.size());
    for (auto& pt : pts) {
        Eigen::Vector4f v{pt.x, pt.y, pt.z, 1.0f}, w = T * v;
        out.emplace_back(w.x(), w.y(), w.z());
    }
    return out;
}

void NcltNode::logFrameInfo(const rclcpp::Time& scan_t,
                            const rclcpp::Time& odom_t,
                            const rclcpp::Time& gt_t,
                            double error_xy,
                            const geometry_msgs::msg::Pose& pred_pose,
                            const geometry_msgs::msg::Pose& gt_pose)
{
    auto split = [&](const rclcpp::Time& tt) {
        int64_t ns = tt.nanoseconds();
        return std::make_pair<uint32_t,uint32_t>(
            uint32_t(ns / 1'000'000'000),
            uint32_t(ns % 1'000'000'000));
    };
    auto [s_s, s_ns] = split(scan_t);
    auto [o_s, o_ns] = split(odom_t);
    auto [g_s, g_ns] = split(gt_t);

    // extract GT quaternion
    const auto &q = gt_pose.orientation;
    // convert to RPY if you like:
    tf2::Quaternion tf_q(q.x, q.y, q.z, q.w);
    double roll, pitch, yaw;
    tf2::Matrix3x3(tf_q).getRPY(roll, pitch, yaw);

    RCLCPP_INFO(get_logger(),
        "Frame %3zu\n"
        "SCAN stamp: %u.%09u\n"
        "ODOM stamp: %u.%09u\n"
        "GT   stamp: %u.%09u\n"
        "  ERROR XY: %.3f m\n"
        "  PRED pos:(%6.3f,%6.3f)  PRED ori quat:(%.3f,%.3f,%.3f,%.3f)\n"
        "  GT   pos:(%6.3f,%6.3f)  GT   ori quat:(%.3f,%.3f,%.3f,%.3f)\n"
        "  GT   ori RPY:(%.3f,%.3f,%.3f)",
        frame_count_,
        s_s, s_ns,
        o_s, o_ns,
        g_s, g_ns,
        error_xy,
        pred_pose.position.x,
        pred_pose.position.y,
        pred_pose.orientation.x,
        pred_pose.orientation.y,
        pred_pose.orientation.z,
        pred_pose.orientation.w,
        gt_pose.position.x,
        gt_pose.position.y,
        q.x, q.y, q.z, q.w,
        roll, pitch, yaw
    );
}

} // namespace sim_local