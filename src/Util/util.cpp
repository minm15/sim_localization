#include "sim_local/util.hpp"

namespace sim_local {
namespace util {

cv::Mat loadBinaryFileToMat(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f)
        throw std::runtime_error("Failed to open " + path);
    std::vector<std::vector<float>> rows;
    while (true) {
        int idx;
        f.read(reinterpret_cast<char*>(&idx), sizeof(idx));
        if (!f || f.eof())
            break;
        std::vector<float> buf(183);
        f.read(reinterpret_cast<char*>(buf.data()), buf.size() * sizeof(float));
        rows.push_back(std::move(buf));
    }
    cv::Mat M((int)rows.size(), 183, CV_32F);
    for (int i = 0; i < (int)rows.size(); ++i)
        for (int j = 0; j < 183; ++j)
            M.at<float>(i, j) = rows[i][j];
    return M;
}

template <typename BufferT>
std::optional<typename BufferT::value_type> findLastBefore(const BufferT& buf, const rclcpp::Time& t) {
    if (buf.empty())
        return std::nullopt;
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

// Explicit instantiations for the buffers we actually use:
template std::optional<std::deque<std::pair<rclcpp::Time, geometry_msgs::msg::Pose>>::value_type>
findLastBefore(const std::deque<std::pair<rclcpp::Time, geometry_msgs::msg::Pose>>&, const rclcpp::Time&);

Eigen::Matrix4f transformMsgToEigen(const geometry_msgs::msg::Transform& t) {
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

Eigen::Matrix4f poseToEigen(const geometry_msgs::msg::Pose& p) {
    geometry_msgs::msg::Transform t;
    t.translation.x = p.position.x;
    t.translation.y = p.position.y;
    t.translation.z = p.position.z;
    t.rotation = p.orientation;
    return transformMsgToEigen(t);
}

std::vector<pcl::PointXYZ> transformKeyPoints(const std::vector<pcl::PointXYZ>& pts, const Eigen::Matrix4f& T) {
    std::vector<pcl::PointXYZ> out;
    out.reserve(pts.size());
    for (auto& pt : pts) {
        Eigen::Vector4f v{pt.x, pt.y, pt.z, 1.f};
        Eigen::Vector4f w = T * v;
        out.emplace_back(w.x(), w.y(), w.z());
    }
    return out;
}

} // namespace util
} // namespace sim_local