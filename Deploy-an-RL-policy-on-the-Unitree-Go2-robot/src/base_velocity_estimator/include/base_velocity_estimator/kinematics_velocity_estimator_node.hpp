#ifndef KINEMATICS_VELOCITY_ESTIMATOR_NODE_H
#define KINEMATICS_VELOCITY_ESTIMATOR_NODE_H

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/imu.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "unitree_go/msg/low_state.hpp"
#include <eigen3/Eigen/Dense>
#include <deque>
#include <unordered_map>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/jacobian.hpp>

using namespace pinocchio;
class Vector3dMedianFilter
{
public:
    Vector3dMedianFilter(size_t window_size)
        : window_size_(window_size) {}

    Eigen::Vector3d filter(const Eigen::Vector3d& input)
    {
        // 添加当前值到各自的滑动窗口
        for (int i = 0; i < 3; ++i)
        {
            buffers_[i].push_back(input[i]);
            if (buffers_[i].size() > window_size_)
                buffers_[i].erase(buffers_[i].begin());
        }

        // 计算每一维的中位数
        Eigen::Vector3d output;
        for (int i = 0; i < 3; ++i)
        {
            std::vector<double> sorted = buffers_[i];  // 拷贝
            std::sort(sorted.begin(), sorted.end());
            output[i] = sorted[sorted.size() / 2];
        }

        return output;
    }

private:
    size_t window_size_;
    std::vector<double> buffers_[3];  // x, y, z 各一个滑动窗口
    // std::vector<std::vector<double>> buffers_;  // x, y, z 各一个滑动窗口

};
class KinematicsVelocityEstimator : public rclcpp::Node
{
public:
    KinematicsVelocityEstimator();

private:
    // !!! 核心修改：统一的回调函数 !!!
    void lowStateCallback(const unitree_go::msg::LowState::SharedPtr msg);
    
    void publishVelocity();
    
    std::pair<Eigen::Vector3d, Eigen::Matrix3d> compute_kinematics_and_jacobian(
        Eigen::VectorXd &theta,
        const std::string &frame_name, 
        const std::vector<std::string> &joint_names
    );

    // !!! 替换旧的订阅者 !!!
    // rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr joint_vel_sub_;
    // rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr joint_sub_;
    // rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr omega_sub_;
    // rclcpp::Subscription<std_msgs::msg::Float32MultiArray>::SharedPtr contact_sub_;
    
    rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr low_state_sub_;

    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr velocity_pub_;
    rclcpp::TimerBase::SharedPtr timer_;

    // 数据缓存
    std::vector<double> cur_joint_velocities_;
    std::vector<double> cur_joint_angles_;
    std::vector<double> cur_omega_;
    std::vector<float> cur_foot_force_; // 新增：存储足端压力

    std::unordered_map<int, std::string> name_mapping_;
    std::unordered_map<int, std::vector<std::string>> joints_mapping_;

    std::unique_ptr<Vector3dMedianFilter> filters_[4];
    
    // Pinocchio 相关变量
    Model model;
    Data data;
    bool model_loaded_ = false;
};

#endif // KINEMATICS_VELOCITY_ESTIMATOR_NODE_H