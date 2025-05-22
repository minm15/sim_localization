// ground_truth_recorder.cpp
#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <fstream>
#include <iomanip>

class GroundTruthRecorder : public rclcpp::Node
{
public:
  GroundTruthRecorder()
  : Node("ground_truth_recorder")
  {
    // 打開輸出檔案，並寫入表頭
    csv_.open("ground_truth.csv", std::ios::out | std::ios::trunc);
    if (!csv_.is_open()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to open ground_truth.csv for writing");
      rclcpp::shutdown();
      return;
    }
    csv_ << "timestamp_sec,x,y,z,roll_rad,pitch_rad,yaw_rad\n";

    // 訂閱 /ground_truth
    sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/ground_truth",
      10,
      std::bind(&GroundTruthRecorder::odomCallback, this, std::placeholders::_1)
    );
    RCLCPP_INFO(this->get_logger(), "Subscribed to /ground_truth");
  }

  ~GroundTruthRecorder()
  {
    if (csv_.is_open()) {
      csv_.close();
      RCLCPP_INFO(this->get_logger(), "ground_truth.csv saved");
    }
  }

private:
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    // 時間戳 (double, s)
    double t = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;

    // 位置
    double x = msg->pose.pose.position.x;
    double y = msg->pose.pose.position.y;
    double z = msg->pose.pose.position.z;

    // 從四元數轉 roll/pitch/yaw
    tf2::Quaternion q(
      msg->pose.pose.orientation.x,
      msg->pose.pose.orientation.y,
      msg->pose.pose.orientation.z,
      msg->pose.pose.orientation.w
    );
    double roll, pitch, yaw;
    tf2::Matrix3x3(q).getRPY(roll, pitch, yaw);

    // 寫入 CSV
    csv_ << std::fixed << std::setprecision(9)
         << t << ","
         << x << "," << y << "," << z << ","
         << roll << "," << pitch << "," << yaw
         << "\n";
  }

  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_;
  std::ofstream csv_;
};

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<GroundTruthRecorder>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}