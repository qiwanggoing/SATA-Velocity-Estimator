#include "base_velocity_estimator/ekf_velocity_estimator.hpp"
#include <numeric>

EKFVelocityEstimator::EKFVelocityEstimator()
    : Node("base_velocity_estimator")
{   
    this->declare_parameter<bool>("is_simulation", false);
    bool is_simulation = this->get_parameter("is_simulation").as_bool();

    // EKF 参数微调
    Q_.block<3,3>(3,3) *= 1e-5;  
    Q_.block<3,3>(6,6) *= 0.001;  
    bias_calibration_samples = 1000;
    H_.setIdentity(); 

    velocity_base_ = Eigen::Vector3d::Zero();
    accel_bias_ = Eigen::Vector3d::Zero();
    is_bias_calibrated_ = false;

    // 1. 话题选择
    std::string lowstate_topic = is_simulation ? "/mujoco/lowstate" : "lowstate";
    
    lowstate_sub_ = create_subscription<unitree_go::msg::LowState>(
        lowstate_topic, rclcpp::SensorDataQoS(), 
        std::bind(&EKFVelocityEstimator::lowStateCallback, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "Subscribing to LowState topic: %s", lowstate_topic.c_str());

    // 2. 发布 TwistStamped (解决类型冲突)
    velocity_pub_ = create_publisher<geometry_msgs::msg::TwistStamped>("/ekf/velocity", 10);
    
    vel_obs_sub_ = create_subscription<geometry_msgs::msg::Twist>(
        "/obs_velocity", 10,
        std::bind(&EKFVelocityEstimator::obsCallback, this, std::placeholders::_1));

    contact_sub = create_subscription<std_msgs::msg::Float32MultiArray>(
        "/contact", 10, std::bind(&EKFVelocityEstimator::contactDetector, this, std::placeholders::_1));

    double publish_rate = 200; 
    timer_ = create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(1000.0 / publish_rate)),
        std::bind(&EKFVelocityEstimator::publishVelocity, this));
        
    RCLCPP_INFO(get_logger(), "EKF Velocity Estimator Initialized (TwistStamped & Tilt-Robust).");
}

void EKFVelocityEstimator::lowStateCallback(const unitree_go::msg::LowState::SharedPtr msg)
{
    auto current_time = this->now();

    Eigen::Vector3d accel_measured(
        msg->imu_state.accelerometer[0],
        msg->imu_state.accelerometer[1],
        msg->imu_state.accelerometer[2]
    );

    Eigen::Quaterniond q(
        msg->imu_state.quaternion[0], 
        msg->imu_state.quaternion[1], 
        msg->imu_state.quaternion[2], 
        msg->imu_state.quaternion[3]  
    );

    // A. 智能零偏校准 (无需改动 .hpp)
    if (!is_bias_calibrated_) {
        accel_samples_.push_back(accel_measured);
        if (accel_samples_.size() >= bias_calibration_samples) {
            calibrateBias(); // 这一步算出了 accel_measured 的均值
            
            // !!! 核心修正 !!!
            // 不要假设机器人是水平的 (不要直接减 [0,0,9.8])
            // 而是用当前的姿态 q 算出重力在机身下的真实分量
            Eigen::Vector3d gravity_world(0, 0, 9.81);
            Eigen::Vector3d gravity_body_now = q.inverse() * gravity_world;
            
            // 真实的 Sensor Bias = 测量均值 - 当前姿态下的重力分量
            accel_bias_ = accel_bias_ - gravity_body_now;
            
            is_bias_calibrated_ = true;
            RCLCPP_INFO(get_logger(), "Bias calibrated (Tilt Corrected): [%f, %f, %f]",
                        accel_bias_.x(), accel_bias_.y(), accel_bias_.z());
        }
        return;
    }

    // B. 计算 dt
    if (last_time_.nanoseconds() == 0) {
        last_time_ = current_time;
        return;
    }
    double dt = (current_time - last_time_).seconds();
    last_time_ = current_time;
    if(dt > 0.05 || dt < 0.0001) dt = 0.005; 

    // C. 物理计算
    Eigen::Vector3d gravity_world(0, 0, 9.81);
    Eigen::Vector3d gravity_body = q.inverse() * gravity_world; 
    
    // 运动加速度 = 测量值 - Bias - 重力分量
    Eigen::Vector3d accel_motion = accel_measured - accel_bias_ - gravity_body;

    predict(accel_motion, dt);
}

// 注意：此函数保持不变，仅计算均值，具体的重力移除在 lowStateCallback 中完成
void EKFVelocityEstimator::calibrateBias()
{
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    for (const auto &sample : accel_samples_) {
        sum += sample;
    }
    accel_bias_ = sum / static_cast<double>(accel_samples_.size());
    accel_samples_.clear();
}

// 下面的函数直接复制即可，保持逻辑不变
void EKFVelocityEstimator::publishVelocity()
{
    if (!is_bias_calibrated_) return;

    auto msg = geometry_msgs::msg::TwistStamped();
    msg.header.stamp = this->now();
    msg.header.frame_id = "base_link"; 
    msg.twist.linear.x = x_(0);
    msg.twist.linear.y = x_(1);
    msg.twist.linear.z = x_(2);
    velocity_pub_->publish(msg);
}

void EKFVelocityEstimator::predict(const Eigen::Vector3d &imu_acc, double dt)
{
    double b_ax = x_(6); double b_ay = x_(7); double b_az = x_(8);
    double ax = imu_acc(0) - b_ax;
    double ay = imu_acc(1) - b_ay;
    double az = imu_acc(2) - b_az;
    x_(0) += ax * dt; x_(1) += ay * dt; x_(2) += az * dt;
    Eigen::MatrixXd F = Eigen::MatrixXd::Identity(9, 9);
    F(0, 6) = -dt; F(1, 7) = -dt; F(2, 8) = -dt; 
    P_ = F * P_ * F.transpose() + Q_;
    if (imu_acc.norm() < 0.5) small_acc_ = true; else small_acc_ = false;
}

void EKFVelocityEstimator::update(const Eigen::Vector3d &vel_obs)
{
    Eigen::Vector3d z = vel_obs;
    Eigen::Vector3d y = z - H_ * x_;
    Eigen::Matrix3d S = H_ * P_ * H_.transpose() + R_;
    Eigen::MatrixXd K = P_ * H_.transpose() * S.inverse();
    x_ += K * y;
    P_ = (Eigen::MatrixXd::Identity(9, 9) - K * H_) * P_;
}

void EKFVelocityEstimator::obsCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
{
    Eigen::Vector3d obs_vel;
    obs_vel(0) = msg->linear.x; obs_vel(1) = msg->linear.y; obs_vel(2) = msg->linear.z;
    update(obs_vel);
}

void EKFVelocityEstimator::contactDetector(const std_msgs::msg::Float32MultiArray::SharedPtr msg)
{
    if (msg->data.size() < 4) return;
    float FL = msg->data[0]; float FR = msg->data[1]; float RL = msg->data[2]; float RR = msg->data[3];
    updateForceBuffer(fl_force_buffer_, FL); updateForceBuffer(fr_force_buffer_, FR);
    updateForceBuffer(rl_force_buffer_, RL); updateForceBuffer(rr_force_buffer_, RR);
    bool stable = isForceStable(fl_force_buffer_) && isForceStable(fr_force_buffer_) && 
                  isForceStable(rr_force_buffer_) && isForceStable(rl_force_buffer_);
    if (small_acc_ && stable) { x_(0) = 0; x_(1) = 0; x_(2) = 0; }
}

void EKFVelocityEstimator::updateForceBuffer(std::deque<float> &force_buffer, float new_value) {
    force_buffer.push_back(new_value);
    if (force_buffer.size() > window_size_) force_buffer.pop_front();
}
bool EKFVelocityEstimator::isForceStable(std::deque<float> &force_buffer) {
    if (force_buffer.size() < window_size_) return false;
    float sum = std::accumulate(force_buffer.begin(), force_buffer.end(), 0.0f);
    float mean = sum / window_size_;
    float sq_sum = 0.0f;
    for (auto x : force_buffer) sq_sum += x * x;
    float stddev = std::sqrt(sq_sum / window_size_ - mean * mean);
    return stddev < stable_threshold_stddev;
}