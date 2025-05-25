#include "sim_local/nclt_local.hpp"

#include <chrono>
#include <cmath>
#include <fstream>
#include <pcl_conversions/pcl_conversions.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2/LinearMath/Quaternion.h>

namespace sim_local {

NcltNode::NcltNode(const rclcpp::NodeOptions& opts) : Node("nclt_localization", opts) {
    // parameters
    declare_parameter<std::string>("descriptor_file", "2013_01_10_nclt_descriptors.bin");
    declare_parameter<double>("initial_pose_x", 0.0);
    declare_parameter<double>("initial_pose_y", 0.0);
    declare_parameter<double>("initial_pose_z", 6.545);
    declare_parameter<double>("initial_roll", -0.029);
    declare_parameter<double>("initial_pitch", -0.010);
    declare_parameter<double>("initial_yaw", -0.148);

    get_parameter("descriptor_file", desc_file_);
    get_parameter("initial_pose_x", init_x_);
    get_parameter("initial_pose_y", init_y_);
    get_parameter("initial_pose_z", init_z_);
    get_parameter("initial_roll", init_roll_);
    get_parameter("initial_pitch", init_pitch_);
    get_parameter("initial_yaw", init_yaw_);

    // load whole database
    vectorDatabase_ = loadBinaryFileToMat(desc_file_);
    if (vectorDatabase_.empty()) {
        RCLCPP_ERROR(get_logger(), "descriptor DB is empty!");
        return;
    }
    RCLCPP_INFO(get_logger(), "Loaded DB: %d×%d", vectorDatabase_.rows, vectorDatabase_.cols);

    // build 4×4×4 buckets offline
    buildBuckets();

    // extractor & PF
    extractor_ = std::make_shared<LinK3D_SLAM::LinK3D_Extractor>(32, 0.1f, 0.4f, 0.3f, 0.3f, 12, 4, 3);
    particle_filter_ =
        std::make_shared<ParticleFilter>(init_x_, init_y_, init_z_, init_roll_, init_pitch_, init_yaw_, 128);

    // subs & pubs
    static_tf_sub_ = create_subscription<TFMessage>(
        "/tf_static", rclcpp::SystemDefaultsQoS(), std::bind(&NcltNode::tfStaticCallback, this, std::placeholders::_1));
    imu_sub_ = create_subscription<Imu>("/imu/data", rclcpp::SensorDataQoS(),
                                        std::bind(&NcltNode::imuCallback, this, std::placeholders::_1));
    odom_sub_ = create_subscription<Odometry>("/odom", rclcpp::SystemDefaultsQoS(),
                                              std::bind(&NcltNode::odomCallback, this, std::placeholders::_1));
    gt_sub_ = create_subscription<Odometry>("/ground_truth", rclcpp::SystemDefaultsQoS(),
                                            std::bind(&NcltNode::groundTruthCallback, this, std::placeholders::_1));
    auto qos = rclcpp::SensorDataQoS().keep_last(200).reliable();
    lidar_sub_ = create_subscription<PointCloud2>("/velodyne_points", qos,
                                                  std::bind(&NcltNode::lidarCallback, this, std::placeholders::_1));
    pub_ = create_publisher<PoseArray>("particle_pose", 10);

    RCLCPP_INFO(get_logger(), "NCLT localization node started.");
}

void computeBucketFeatures(const cv::Mat& db, int left_start, int left_end, int right_start, int right_end,
                           std::vector<float>& lefts, std::vector<float>& d0s, std::vector<float>& rights) {
    const float eps = 1e-6f;
    int N = db.rows;
    lefts.assign(N, 0.0f);
    d0s.assign(N, 0.0f);
    rights.assign(N, 0.0f);

    for (int i = 0; i < N; ++i) {
        d0s[i] = db.at<float>(i, 0);

        // compute avg(subarr_left)
        {
            float sum = 0.0f;
            int cnt = 0;
            for (int c = left_start; c <= left_end; ++c) {
                float v = db.at<float>(i, c);
                if (std::fabs(v) > eps) {
                    sum += v;
                    ++cnt;
                }
            }
            lefts[i] = (cnt > 0 ? sum / float(cnt) : 0.0f);
        }

        // compute avg(subarr_right)
        {
            float sum = 0.0f;
            int cnt = 0;
            for (int c = right_start; c <= right_end; ++c) {
                float v = db.at<float>(i, c);
                if (std::fabs(v) > eps) {
                    sum += v;
                    ++cnt;
                }
            }
            rights[i] = (cnt > 0 ? sum / float(cnt) : 0.0f);
        }
    }
}

// for each desc, build their left/d0/right and make bucket
void NcltNode::buildBuckets() {
    const int N = vectorDatabase_.rows;
    std::vector<float> lefts(N), d0s(N), rights(N);
    computeBucketFeatures(vectorDatabase_, 1, 5, 175, 179, lefts, d0s, rights);
    // compute quartiles
    auto quartile = [&](const std::vector<float>& src, std::array<float, 3>& q) {
        std::vector<float> tmp = src;
        std::sort(tmp.begin(), tmp.end());
        int N = (int)tmp.size();
        q[0] = tmp[int(0.25 * N)];
        q[1] = tmp[int(0.50 * N)];
        q[2] = tmp[int(0.75 * N)];
    };

    quartile(lefts, quartiles_[0]);
    quartile(d0s, quartiles_[1]);
    quartile(rights, quartiles_[2]);
    // assign each db row into bucket
    for (int i = 0; i < N; ++i) {
        int k = bucketIndex(lefts[i], d0s[i], rights[i]);
        db_buckets_[k].push_back(i);
    }
}

// find the bucket for query
int NcltNode::bucketIndex(float left, float d0, float right) const {
    auto binOf = [&](float x, const std::array<float, 3>& q) {
        if (x < q[0])
            return 0;
        if (x < q[1])
            return 1;
        if (x < q[2])
            return 2;
        return 3;
    };
    int bL = binOf(left, quartiles_[0]);
    int b0 = binOf(d0, quartiles_[1]);
    int bR = binOf(right, quartiles_[2]);
    return (bL << 4) | (b0 << 2) | bR;
}

void NcltNode::tfStaticCallback(const TFMessage::SharedPtr msg) {
    for (auto& ts : msg->transforms) {
        if (!have_base_velo_ && ts.header.frame_id == "base_link" && ts.child_frame_id == "velodyne") {
            base_T_velo_ = transformMsgToEigen(ts.transform);
            have_base_velo_ = true;
            RCLCPP_INFO(get_logger(), "Cached base_link→velodyne");
        }
    }
}

void NcltNode::imuCallback(const Imu::SharedPtr msg) {
    last_imu_msg_ = msg;
    has_last_imu_ = true;
}

void NcltNode::odomCallback(const Odometry::SharedPtr odom) {
    rclcpp::Time now(odom->header.stamp);
    double dt = 0;
    if (has_last_odom_time_)
        dt = (now - last_odom_time_).seconds();
    last_odom_time_ = now;
    has_last_odom_time_ = true;

    odom_history_.emplace_back(now, odom->pose.pose);
    if (odom_history_.size() > 1000)
        odom_history_.pop_front();

    if (have_received_first_scan_ && has_last_imu_) {
        particle_filter_->update(odom, last_imu_msg_, dt);
    }
}

void NcltNode::groundTruthCallback(const Odometry::SharedPtr gt) {
    rclcpp::Time now(gt->header.stamp);
    gt_history_.emplace_back(now, gt->pose.pose);
    if (gt_history_.size() > 1000)
        gt_history_.pop_front();
}

void NcltNode::lidarCallback(const PointCloud2::SharedPtr msg) {
    auto t_start = std::chrono::high_resolution_clock::now();
    if (!have_base_velo_) {
        RCLCPP_WARN(get_logger(), "Static TF not ready, skipping scan");
        return;
    }
    rclcpp::Time scan_t(msg->header.stamp);
    RCLCPP_INFO(get_logger(), "Scan @ %.3f", scan_t.seconds());

    auto odom_opt = findLastBefore(odom_history_, scan_t);
    auto gt_opt = findLastBefore(gt_history_, scan_t);
    if (!odom_opt || !gt_opt) {
        RCLCPP_WARN(get_logger(), "No odom/gt ≤ scan, skipping");
        return;
    }
    auto [odom_t, odom_pose] = *odom_opt;
    auto [gt_t, gt_pose] = *gt_opt;

    if (!have_received_first_scan_) {
        have_received_first_scan_ = true;
        RCLCPP_INFO(get_logger(), "First LiDAR scan → enabling updates");
    }

    // 1) extract keypts & desc
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *cloud);
    std::vector<pcl::PointXYZ> keypts;
    cv::Mat desc;
    std::vector<int> idx;
    LinK3D_SLAM::MatPt clus;
    (*extractor_)(*cloud, keypts, desc, idx, clus);

    // 2) generate candidates for each query descriptor
    const int M = desc.rows;
    std::vector<float> lefts_q(M), d0s_q(M), rights_q(M);
    computeBucketFeatures(desc, 1, 5, 175, 179, lefts_q, d0s_q, rights_q);

    std::vector<std::vector<int>> candidates(M);
    candidates.assign(M, {});

    for (int i = 0; i < M; ++i) {
        float l = lefts_q[i];
        float d0 = d0s_q[i];
        float r = rights_q[i];

        int key = bucketIndex(l, d0, r);
        auto it = db_buckets_.find(key);
        if (it != db_buckets_.end()) {
            candidates[i] = it->second; // take all of the desc id in this bucket as candidates
        }
    }
    // 3) matching base candidates
    std::vector<std::pair<int, int>> vMatched;
    extractor_->matcher(desc, vectorDatabase_, candidates, vMatched);

    // 4) map matching & PF
    Eigen::Matrix4f B = base_T_velo_;
    for (auto& p : particle_filter_->getParticles()) {
        Eigen::Matrix4f W = poseToEigen(p.pose);
        auto tks = transformKeyPoints(keypts, W * B);
        p.map_matching(tks, vectorDatabase_, vMatched);
    }
    particle_filter_->weighting();
    auto best = particle_filter_->getBestParticle(1);
    particle_filter_->resampling();

    double ex = best.pose.position.x - gt_pose.position.x;
    double ey = best.pose.position.y - gt_pose.position.y;
    double err_xy = std::hypot(ex, ey);
    logFrameInfo(scan_t, odom_t, gt_t, err_xy, best.pose, gt_pose);

    PoseArray pa;
    pa.header.stamp = scan_t;
    pa.header.frame_id = "world";
    pa.poses.push_back(best.pose);
    pub_->publish(pa);

    auto t_end = std::chrono::high_resolution_clock::now();
    auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    RCLCPP_INFO(get_logger(), "Processing time: %ld ms", (long)dur);

    ++frame_count_;
}

// --------------------------------------------------------------------------
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
    auto it = std::lower_bound(buf.begin(), buf.end(), t, [](auto& a, const rclcpp::Time& ts) { return a.first < ts; });
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
        Eigen::Vector4f v{pt.x, pt.y, pt.z, 1.f}, w = T * v;
        out.emplace_back(w.x(), w.y(), w.z());
    }
    return out;
}

void NcltNode::logFrameInfo(const rclcpp::Time& scan_t, const rclcpp::Time& odom_t, const rclcpp::Time& gt_t,
                            double error_xy, const geometry_msgs::msg::Pose& pred_pose,
                            const geometry_msgs::msg::Pose& gt_pose) {
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
                "PRED pos:(%.3f,%.3f,%.3f) RPY:(%.3f,%.3f,%.3f)\n"
                " GT  pos:(%.3f,%.3f,%.3f) RPY:(%.3f,%.3f,%.3f)",
                frame_count_, s_s, s_ns, o_s, o_ns, g_s, g_ns, error_xy, pred_pose.position.x, pred_pose.position.y,
                pred_pose.position.z, pr, pp, py, gt_pose.position.x, gt_pose.position.y, gt_pose.position.z, gr, gp,
                gy);
}

} // namespace sim_local