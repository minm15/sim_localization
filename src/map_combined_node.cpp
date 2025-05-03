// src/map_combined_node.cpp
/******************************************************
This file generates a map of unmatched LinK3D descriptors
in the „map“ frame by synchronizing incoming tf (map→base_link)
and LIDAR_TOP pointclouds.  For each new /tf (map→base_link)
message we find the nearest earlier LIDAR_TOP, extract keypoints
+ descriptors, match against the previous frame, keep only the
unmatched ones, transform their 3D locations into the map frame,
and append them (180‐dim float + x,y,z) into descriptors.bin.
******************************************************/

#include "sim_local/LinK3D_extractor.h"
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_msgs/msg/tf_message.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_sensor_msgs/tf2_sensor_msgs.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <deque>
#include <vector>
#include <unordered_set>
#include <optional>
#include <algorithm>
#include <fstream>
#include <signal.h>

// where we write out the 180+3 floats per keypoint
static const std::string descriptorFilePath = "descriptors.bin";

// Create or truncate the descriptor file on startup
static void initializeDescriptorFile()
{
  std::ofstream f(descriptorFilePath, std::ios::binary | std::ios::trunc);
  if (!f) throw std::runtime_error("Failed to create " + descriptorFilePath);
}

// Append a batch of descriptors + their map‐frame points
static void appendDescriptorsToFile(const cv::Mat &desc,
                                    const std::vector<pcl::PointXYZ> &pts)
{
  if (desc.rows != int(pts.size())) {
    throw std::runtime_error("Descriptor/point count mismatch");
  }
  std::ofstream f(descriptorFilePath, std::ios::binary | std::ios::app);
  if (!f) throw std::runtime_error("Failed to open " + descriptorFilePath + " for appending");
  for (int i = 0; i < desc.rows; ++i) {
    int idx = i;
    f.write(reinterpret_cast<const char*>(&idx), sizeof(idx));
    f.write(reinterpret_cast<const char*>(desc.ptr<float>(i)), 180 * sizeof(float));
    auto &p = pts[i];
    f.write(reinterpret_cast<const char*>(&p.x), sizeof(p.x));
    f.write(reinterpret_cast<const char*>(&p.y), sizeof(p.y));
    f.write(reinterpret_cast<const char*>(&p.z), sizeof(p.z));
  }
}

// Convert a geometry_msgs::msg::Transform into a 4×4 Eigen matrix
static Eigen::Matrix4f transformMsgToEigen(const geometry_msgs::msg::Transform &t)
{
  Eigen::Matrix4f M = Eigen::Matrix4f::Identity();
  M(0,3) = t.translation.x;
  M(1,3) = t.translation.y;
  M(2,3) = t.translation.z;
  tf2::Quaternion q{t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w};
  tf2::Matrix3x3 Rm(q);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      M(i,j) = Rm[i][j];
  return M;
}

class MapCombinedNode : public rclcpp::Node {
public:
  MapCombinedNode()
  : Node("map_combined"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_),
    frame_count_(0),
    first_frame_(true)
  {
    // Prepare the output file
    initializeDescriptorFile();

    // Build a single extractor
    extractor_ = std::make_shared<LinK3D_SLAM::LinK3D_Extractor>(
      32, 0.1f, 0.4f, 0.3f, 0.3f, 12, 4, 3
    );

    // Subscribe to LIDAR_TOP and /tf
    lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      "/LIDAR_TOP", rclcpp::SensorDataQoS(),
      std::bind(&MapCombinedNode::lidarCallback, this, std::placeholders::_1)
    );
    tf_sub_ = create_subscription<tf2_msgs::msg::TFMessage>(
      "/tf", rclcpp::SystemDefaultsQoS(),
      std::bind(&MapCombinedNode::tfCallback, this, std::placeholders::_1)
    );

    RCLCPP_INFO(get_logger(), "map_combined node ready");
  }

private:
  // Buffer incoming LIDAR scans
  void lidarCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    auto t = rclcpp::Time(msg->header.stamp);
    lidar_history_.emplace_back(t, msg);
    if (lidar_history_.size() > 1000u) lidar_history_.pop_front();
    processPendingTFs();
  }

  // Buffer unique TFs (map→base_link), sorted by timestamp
  void tfCallback(const tf2_msgs::msg::TFMessage::SharedPtr msg) {
    for (auto &ts : msg->transforms) {
      if (ts.header.frame_id != "map" || ts.child_frame_id != "base_link") continue;
      uint64_t key =
        (uint64_t(ts.header.stamp.sec) << 32)
        | uint64_t(ts.header.stamp.nanosec);
      if (!processed_tf_.insert(key).second) continue;
      rclcpp::Time t(ts.header.stamp);
      auto it = std::upper_bound(
        tf_queue_.begin(), tf_queue_.end(), t,
        [&](const rclcpp::Time &a, const geometry_msgs::msg::TransformStamped &b){
          return a < rclcpp::Time(b.header.stamp);
        }
      );
      tf_queue_.insert(it, ts);
    }
    processPendingTFs();
  }

  // Whenever we have at least one LIDAR whose stamp ≥ front TF, process that TF
  void processPendingTFs() {
    while (!tf_queue_.empty()) {
      auto &ts = tf_queue_.front();
      rclcpp::Time tf_t(ts.header.stamp);
      if (lidar_history_.empty() ||
          lidar_history_.back().first < tf_t)
      {
        // need to wait for more LIDAR
        break;
      }
      processOneTF(ts);
      tf_queue_.pop_front();
    }
  }

  // Find the last element in buf whose timestamp ≤ t
  template<typename T>
  std::optional<T> findLastBefore(
    const std::deque<T> &buf,
    const rclcpp::Time &t)
  {
    if (buf.empty()) return {};
    auto it = std::lower_bound(
      buf.begin(), buf.end(), t,
      [](const T &a, const rclcpp::Time &ts){
        return a.first < ts;
      }
    );
    if (it == buf.begin()) {
      if (it->first == t) return *it;
      return {};
    }
    if (it == buf.end() || it->first > t) --it;
    return *it;
  }

  void processOneTF(const geometry_msgs::msg::TransformStamped &ts) {
    auto tf_t = rclcpp::Time(ts.header.stamp);
    // pick the nearest LIDAR ≤ this TF
    auto lidar_opt = findLastBefore(lidar_history_, tf_t);
    if (!lidar_opt) {
      RCLCPP_WARN(get_logger(),
        "No LIDAR for TF@%u.%09u",
        ts.header.stamp.sec,
        ts.header.stamp.nanosec
      );
      return;
    }

    auto [lidar_t, lidar_msg] = *lidar_opt;
    // Convert to PCL
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*lidar_msg, *cloud);

    // Extract keypoints + descriptors
    std::vector<pcl::PointXYZ> keypts;
    cv::Mat desc;
    std::vector<int> idx;
    LinK3D_SLAM::MatPt clus;
    (*extractor_)(*cloud, keypts, desc, idx, clus);

    // Transform keypoints into “map” via ts.transform
    Eigen::Matrix4f M = transformMsgToEigen(ts.transform);
    std::vector<pcl::PointXYZ> map_pts;
    map_pts.reserve(keypts.size());
    for (auto &p : keypts) {
      Eigen::Vector4f v{p.x,p.y,p.z,1.0f}, w = M * v;
      map_pts.emplace_back(w.x(), w.y(), w.z());
    }

    if (first_frame_) {
      // dump everything on the first TF
      appendDescriptorsToFile(desc, map_pts);
      RCLCPP_INFO(get_logger(),
        "Frame %3zu  TF@%u.%09u  LiDAR@%llu.%09llu  new:%4zu (initial)",
        frame_count_,
        ts.header.stamp.sec,
        ts.header.stamp.nanosec,
        static_cast<unsigned long long>(lidar_t.nanoseconds()/1000000000ULL),
        static_cast<unsigned long long>(lidar_t.nanoseconds()%1000000000ULL),
        static_cast<size_t>(desc.rows)
      );
      first_frame_    = false;
      prevDescriptors_ = desc.clone();
    } else {
      // match previous → current
      std::vector<std::pair<int,int>> matches;
      extractor_->matcher(prevDescriptors_, desc, matches);

      // collect unmatched
      std::unordered_set<int> mset;
      for (auto &m : matches) mset.insert(m.second);

      std::vector<pcl::PointXYZ> unkp;
      cv::Mat undesc;
      for (int i = 0; i < desc.rows; ++i) {
        if (!mset.count(i)) {
          unkp.push_back(map_pts[i]);
          undesc.push_back(desc.row(i));
        }
      }
      if (!undesc.empty()) {
        appendDescriptorsToFile(undesc, unkp);
      }
      RCLCPP_INFO(get_logger(),
        "Frame %3zu  TF@%u.%09u  LiDAR@%llu.%09llu  new:%4zu",
        frame_count_,
        ts.header.stamp.sec,
        ts.header.stamp.nanosec,
        static_cast<unsigned long long>(lidar_t.nanoseconds()/1000000000ULL),
        static_cast<unsigned long long>(lidar_t.nanoseconds()%1000000000ULL),
        static_cast<size_t>(undesc.rows)
      );
      prevDescriptors_ = desc.clone();
    }

    ++frame_count_;
  }

  //–– Members ––
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;
  rclcpp::Subscription<tf2_msgs::msg::TFMessage>::SharedPtr        tf_sub_;
  tf2_ros::Buffer     tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  std::deque<std::pair<rclcpp::Time, sensor_msgs::msg::PointCloud2::SharedPtr>> lidar_history_;
  std::deque<geometry_msgs::msg::TransformStamped> tf_queue_;
  std::unordered_set<uint64_t> processed_tf_;

  std::shared_ptr<LinK3D_SLAM::LinK3D_Extractor> extractor_;
  cv::Mat prevDescriptors_;
  bool first_frame_;
  size_t frame_count_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<MapCombinedNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
