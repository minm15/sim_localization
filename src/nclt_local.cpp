#include "sim_local/nclt_local.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

using sim_local::util::loadBinaryFileToMat;
using sim_local::util::findLastBefore;
using sim_local::util::transformMsgToEigen;
using sim_local::util::poseToEigen;
using sim_local::util::transformKeyPoints;

namespace sim_local {

//------------------------------------------------------------------------------
NcltNode::NcltNode(const rclcpp::NodeOptions& opts) : Node("nclt_localization", opts) {
    // 1) param
    declare_parameter<std::string>("descriptor_file", "2013_01_10_nclt_descriptors.bin");
    declare_parameter<int>("buckets_per_dim", 8);
    declare_parameter<double>("initial_pose_x", 0.0);
    declare_parameter<double>("initial_pose_y", 0.0);
    declare_parameter<double>("initial_pose_z", 6.545);
    declare_parameter<double>("initial_roll", -0.029);
    declare_parameter<double>("initial_pitch", -0.010);
    declare_parameter<double>("initial_yaw", -0.148);

    get_parameter("descriptor_file", desc_file_);
    get_parameter("buckets_per_dim", buckets_per_dim_);
    get_parameter("initial_pose_x", init_x_);
    get_parameter("initial_pose_y", init_y_);
    get_parameter("initial_pose_z", init_z_);
    get_parameter("initial_roll", init_roll_);
    get_parameter("initial_pitch", init_pitch_);
    get_parameter("initial_yaw", init_yaw_);

    // 2) load the desc bin file to vectorDatabase_
    vectorDatabase_ = loadBinaryFileToMat(desc_file_);
    if (vectorDatabase_.empty()) {
        RCLCPP_ERROR(get_logger(), "descriptor DB is empty!");
        return;
    }
    RCLCPP_INFO(get_logger(), "Loaded DB: %dx%d", vectorDatabase_.rows, vectorDatabase_.cols);

    // 3) construct the bucket
    buildBuckets(buckets_per_dim_);

    // 4) initialize extractor & PF
    extractor_ = std::make_shared<LinK3D_SLAM::LinK3D_Extractor>(32, 0.1f, 0.4f, 0.3f, 0.3f, 12, 4, 3);
    particle_filter_ =
        std::make_shared<ParticleFilter>(init_x_, init_y_, init_z_, init_roll_, init_pitch_, init_yaw_, 1024);

    // 5) subscribe & publish
    static_tf_sub_ = create_subscription<tf2_msgs::msg::TFMessage>(
        "/tf_static", rclcpp::SystemDefaultsQoS(), std::bind(&NcltNode::tfStaticCallback, this, std::placeholders::_1));
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        "/imu/data", rclcpp::SensorDataQoS(), std::bind(&NcltNode::imuCallback, this, std::placeholders::_1));
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        "/odom", rclcpp::SystemDefaultsQoS(), std::bind(&NcltNode::odomCallback, this, std::placeholders::_1));
    gt_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        "/ground_truth", rclcpp::SystemDefaultsQoS(),
        std::bind(&NcltNode::groundTruthCallback, this, std::placeholders::_1));
    auto qos = rclcpp::SensorDataQoS().keep_last(200).reliable();
    lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "/velodyne_points", qos, std::bind(&NcltNode::lidarCallback, this, std::placeholders::_1));
    pub_ = create_publisher<geometry_msgs::msg::PoseArray>("particle_pose", 10);

    RCLCPP_INFO(get_logger(), "NCLT localization node started.  buckets_per_dim=%d", buckets_per_dim_);
}

// encode left／d0／right subarray for each desc in db
static void computeBucketFeatures(const cv::Mat& db, int l0, int l1, int r0, int r1, std::vector<float>& lefts,
                                  std::vector<float>& d0s, std::vector<float>& rights) {
    const float eps = 1e-6f;
    int N = db.rows;
    lefts.resize(N);
    d0s.resize(N);
    rights.resize(N);
    for (int i = 0; i < N; ++i) {
        d0s[i] = db.at<float>(i, 0);
        // left
        {
            float sum = 0;
            int cnt = 0;
            for (int c = l0; c <= l1; ++c) {
                float v = db.at<float>(i, c);
                if (std::fabs(v) > eps) {
                    sum += v;
                    ++cnt;
                }
            }
            lefts[i] = cnt ? sum / float(cnt) : 0.f;
        }
        // right
        {
            float sum = 0;
            int cnt = 0;
            for (int c = r0; c <= r1; ++c) {
                float v = db.at<float>(i, c);
                if (std::fabs(v) > eps) {
                    sum += v;
                    ++cnt;
                }
            }
            rights[i] = cnt ? sum / float(cnt) : 0.f;
        }
    }
}

//------------------------------------------------------------------------------
// split B sections for each encoded value (left, d0, right)
// obtain the B-1 split point
// we have three encoded value, so num of buckets: B^3
void NcltNode::buildBuckets(int B) {
    // 1) obtain encoded value
    std::vector<float> lefts, d0s, rights;
    computeBucketFeatures(vectorDatabase_, 1, 5, 175, 179, lefts, d0s, rights);

    // 2) find B-1 split points
    auto makeCuts = [&](const std::vector<float>& vals, std::vector<float>& cuts) {
        std::vector<float> tmp(vals.begin(), vals.end());
        std::sort(tmp.begin(), tmp.end());
        cuts.resize(B - 1);
        int N = tmp.size();
        for (int i = 1; i < B; ++i) {
            int idx = std::floor((double)i * N / B);
            cuts[i - 1] = tmp[std::min(idx, N - 1)];
        }
    };
    makeCuts(lefts, cuts_left_);
    makeCuts(d0s, cuts_d0_);
    makeCuts(rights, cuts_right_);

    // 3) allocate to bucket (bL,b0,bR) → key = bL*B*B + b0*B + bR
    db_buckets_.clear();
    int N = vectorDatabase_.rows;
    for (int i = 0; i < N; ++i) {
        int bL = std::upper_bound(cuts_left_.begin(), cuts_left_.end(), lefts[i]) - cuts_left_.begin();
        int b0 = std::upper_bound(cuts_d0_.begin(), cuts_d0_.end(), d0s[i]) - cuts_d0_.begin();
        int bR = std::upper_bound(cuts_right_.begin(), cuts_right_.end(), rights[i]) - cuts_right_.begin();
        int key = bL * B * B + b0 * B + bR;
        db_buckets_[key].push_back(i);
    }
}

void NcltNode::tfStaticCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg) {
    for (auto& t : msg->transforms) {
        if (!have_base_velo_ && t.header.frame_id == "base_link" && t.child_frame_id == "velodyne") {
            base_T_velo_ = transformMsgToEigen(t.transform);
            have_base_velo_ = true;
            RCLCPP_INFO(get_logger(), "Cached base_link→velodyne");
        }
    }
}

void NcltNode::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg) {
    last_imu_msg_ = msg;
    has_last_imu_ = true;
}

void NcltNode::odomCallback(const nav_msgs::msg::Odometry::SharedPtr odom) {
    rclcpp::Time now(odom->header.stamp);
    double dt = 0;
    if (has_last_odom_time_)
        dt = (now - last_odom_time_).seconds();
    last_odom_time_ = now;
    has_last_odom_time_ = true;
    odom_history_.emplace_back(now, odom->pose.pose);
    if (odom_history_.size() > 1000)
        odom_history_.pop_front();
    if (have_received_first_scan_ && has_last_imu_)
        particle_filter_->update(odom, last_imu_msg_, dt);
}

void NcltNode::groundTruthCallback(const nav_msgs::msg::Odometry::SharedPtr gt) {
    rclcpp::Time now(gt->header.stamp);
    gt_history_.emplace_back(now, gt->pose.pose);
    if (gt_history_.size() > 1000)
        gt_history_.pop_front();
}

void NcltNode::lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    auto t_start = std::chrono::high_resolution_clock::now();
    if (!have_base_velo_) {
        RCLCPP_WARN(get_logger(), "Static TF not ready, skipping scan");
        return;
    }
    rclcpp::Time scan_t(msg->header.stamp);
    RCLCPP_INFO(get_logger(), "Scan @ %.3f", scan_t.seconds());
    // find odom/gt
    auto od = findLastBefore(odom_history_, scan_t);
    auto gt = findLastBefore(gt_history_, scan_t);
    if (!od || !gt) {
        RCLCPP_WARN(get_logger(), "No odom/gt ≤ scan, skip");
        return;
    }
    if (!have_received_first_scan_) {
        have_received_first_scan_ = true;
        RCLCPP_INFO(get_logger(), "First scan → enabling updates");
    }
    // 1) extract
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud);
    std::vector<pcl::PointXYZ> keypts;
    cv::Mat desc;
    std::vector<int> idx;
    LinK3D_SLAM::MatPt clus;
    (*extractor_)(*cloud, keypts, desc, idx, clus);

    // 2) per-query 3 encode feature → bucket → candidates
    int M = desc.rows;
    std::vector<float> lq(M), d0q(M), rq(M);
    computeBucketFeatures(desc, 1, 5, 175, 179, lq, d0q, rq);
    std::vector<std::vector<int>> candidates(M);
    for (int i = 0; i < M; ++i) {
        int bL = std::upper_bound(cuts_left_.begin(), cuts_left_.end(), lq[i]) - cuts_left_.begin();
        int b0 = std::upper_bound(cuts_d0_.begin(), cuts_d0_.end(), d0q[i]) - cuts_d0_.begin();
        int bR = std::upper_bound(cuts_right_.begin(), cuts_right_.end(), rq[i]) - cuts_right_.begin();
        int key = bL * buckets_per_dim_ * buckets_per_dim_ + b0 * buckets_per_dim_ + bR;
        auto it = db_buckets_.find(key);
        if (it != db_buckets_.end())
            candidates[i] = it->second;
    }

    // 3) matcher
    std::vector<std::pair<int, int>> vMatched;
    extractor_->matcher(desc, vectorDatabase_, candidates, vMatched);

    // 4) PF
    Eigen::Matrix4f B = base_T_velo_;
    for (auto& p : particle_filter_->getParticles()) {
        Eigen::Matrix4f W = poseToEigen(p.pose);
        auto tks = transformKeyPoints(keypts, W * B);
        p.map_matching(tks, vectorDatabase_, vMatched);
    }
    particle_filter_->weighting();
    auto best = particle_filter_->getBestParticle(1);
    particle_filter_->resampling();

    // 5) log
    auto [gt_t, gt_pose] = *gt;
    double ex = best.pose.position.x - gt_pose.position.x;
    double ey = best.pose.position.y - gt_pose.position.y;
    double err_xy = std::hypot(ex, ey);
    int kpsize = static_cast<int>(keypts.size());
    logFrameInfo(scan_t, od->first, gt_t, err_xy, best.pose, gt_pose, kpsize);
    geometry_msgs::msg::PoseArray pa;
    pa.header.stamp = scan_t;
    pa.header.frame_id = "world";
    pa.poses.push_back(best.pose);
    pub_->publish(pa);

    auto t_end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    RCLCPP_INFO(get_logger(), "Processing time: %ld ms", (long)dur);

    ++frame_count_;
}

void NcltNode::logFrameInfo(const rclcpp::Time& scan_t, const rclcpp::Time& odom_t, const rclcpp::Time& gt_t,
                            double error_xy, const geometry_msgs::msg::Pose& pred_pose,
                            const geometry_msgs::msg::Pose& gt_pose, int kpsize) {
    auto split = [&](const rclcpp::Time& tt) {
        int64_t ns = tt.nanoseconds();
        return std::make_pair(uint32_t(ns / 1'000'000'000), uint32_t(ns % 1'000'000'000));
    };
    auto [s_s, s_ns] = split(scan_t);
    auto [o_s, o_ns] = split(odom_t);
    auto [g_s, g_ns] = split(gt_t);

    tf2::Quaternion q_pred{pred_pose.orientation.x, pred_pose.orientation.y, pred_pose.orientation.z,
                           pred_pose.orientation.w};
    double pr, pp, py;
    tf2::Matrix3x3(q_pred).getRPY(pr, pp, py);

    tf2::Quaternion q_gt{gt_pose.orientation.x, gt_pose.orientation.y, gt_pose.orientation.z, gt_pose.orientation.w};
    double gr, gp, gy;
    tf2::Matrix3x3(q_gt).getRPY(gr, gp, gy);

    RCLCPP_INFO(get_logger(),
                "Frame %3zu\n"
                "SCAN:%u.%09u  ODOM:%u.%09u  GT:%u.%09u\n"
                "ERR XY:%.3f m\n"
                "KP size: %d \n"
                "PRED pos:(%.3f,%.3f,%.3f) RPY:(%.3f,%.3f,%.3f)\n"
                " GT  pos:(%.3f,%.3f,%.3f) RPY:(%.3f,%.3f,%.3f)",
                frame_count_, s_s, s_ns, o_s, o_ns, g_s, g_ns, error_xy, kpsize, pred_pose.position.x, pred_pose.position.y,
                pred_pose.position.z, pr, pp, py, gt_pose.position.x, gt_pose.position.y, gt_pose.position.z, gr, gp,
                gy);
}

} // namespace sim_local