#pragma once
#include "rclcpp/rclcpp.hpp"
#include "unitree_go/msg/low_state.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp" // !!! 关键：引入 TwistStamped 头文件 !!!
#include "geometry_msgs/msg/vector3.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include <eigen3/Eigen/Dense>
#include <deque>

class EKFVelocityEstimator : public rclcpp::Node
{
public:
    EKFVelocityEstimator();
    void predict(const Eigen::Vector3d &imu_acc, double dt);
    void update(const Eigen::Vector3d &vel_obs);

private:
    void lowStateCallback(const unitree_go::msg::LowState::SharedPtr msg);
    void obsCallback(const geometry_msgs::msg::Twist::SharedPtr msg);
    void calibrateBias();
    void publishVelocity();
    void contactDetector(const std_msgs::msg::Float32MultiArray::SharedPtr msg);
    void updateForceBuffer(std::deque<float> &force_buffer, float new_value);
    bool isForceStable(std::deque<float> &force_buffer_);

    rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr lowstate_sub_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr vel_obs_sub_;
    rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr contact_sub;
    
    // !!! 关键：将 Twist 修改为 TwistStamped !!!
    rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr velocity_pub_;
    
    rclcpp::TimerBase::SharedPtr timer_;
    Eigen::Vector3d velocity_base_;
    Eigen::Vector3d accel_bias_;
    std::deque<Eigen::Vector3d> accel_samples_;
    bool is_bias_calibrated_ = false;
    rclcpp::Time last_time_;
    int bias_calibration_samples = 1000;
    bool small_acc_ = true;
    const int window_size_ = 20;
    const float stable_threshold_stddev = 10;
    std::deque<float> fl_force_buffer_;
    std::deque<float> rl_force_buffer_;
    std::deque<float> fr_force_buffer_;
    std::deque<float> rr_force_buffer_;
    Eigen::VectorXd x_ = Eigen::VectorXd::Zero(9);
    Eigen::MatrixXd P_ = Eigen::MatrixXd::Identity(9, 9) * 5;
    Eigen::MatrixXd Q_ = Eigen::MatrixXd::Identity(9, 9)*0.01;
    Eigen::MatrixXd R_ = Eigen::MatrixXd::Identity(3, 3) * 10;
    Eigen::MatrixXd H_ = Eigen::MatrixXd::Zero(3, 9);
};