#include "base_velocity_estimator/imu_velocity_estimator_node.hpp"


imuVelocityEstimator::imuVelocityEstimator() : Node("base_velocity_estimator"){
    bias_calibration_samples=1000;
    velocity_base_=Eigen::Vector3d::Zero();
    accel_bias_=Eigen::Vector3d::Zero();
    is_bias_calibrated_=false;
    imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
        "/imu/data",10,std::bind(&imuVelocityEstimator::imuCallback,this,std::placeholders::_1)
    );
    contact_sub=create_subscription<std_msgs::msg::Float32MultiArray>(
      "/contact",10,std::bind(&imuVelocityEstimator::contactDetector,this,std::placeholders::_1)  
    );
    // omega_sub_=create_subscription<std_msgs::msg::Float32MultiArray>(
    //     "/contact",10,std::bind(&imuVelocityEstimator::contactDetector,this,std::placeholders::_1)  
    //   );
    velocity_pub_=create_publisher<geometry_msgs::msg::Twist>(
        "/imu_velocity",10
    );

    double publish_rate = 50;

    timer_ = create_wall_timer(
        std::chrono::milliseconds(static_cast<int>(1000.0 / publish_rate)),
        std::bind(&imuVelocityEstimator::publishVelocity, this));
        RCLCPP_INFO(get_logger(), "Node initialized. Waiting for IMU data...");

}

void imuVelocityEstimator::imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg){
    auto current_time = rclcpp::Time(msg->header.stamp);
    
    Eigen::Vector3d accel_measured={
        msg->linear_acceleration.x,
        msg->linear_acceleration.y,
        msg->linear_acceleration.z
    };
    if (!is_bias_calibrated_) {
        accel_samples_.push_back(accel_measured);
        if (accel_samples_.size() >= bias_calibration_samples) {
            calibrateBias();
            is_bias_calibrated_ = true;
            RCLCPP_INFO(get_logger(), "Bias calibration complete. Bias: [%f, %f, %f]",
                accel_bias_.x(), accel_bias_.y(), accel_bias_.z());
        }
        return;
    }
    if (last_time_.nanoseconds() == 0) {
        last_time_ = current_time;
        return;
    }
    double dt = (current_time - last_time_).seconds();
    last_time_ = current_time;

    Eigen::Vector3d accel_corrected = accel_measured - accel_bias_;
    velocity_base_ += accel_corrected * dt;
    accel_bias_.norm()>0.01?small_acc_=false:small_acc_=true;
}
void imuVelocityEstimator::contactDetector(const std_msgs::msg::Float32MultiArray::SharedPtr msg){
    float FL_force=msg->data[0];
    float FR_force=msg->data[1];
    float RL_force=msg->data[2];
    float RR_force=msg->data[3];
    updateForceBuffer(fl_force_buffer_,FL_force);
    updateForceBuffer(fr_force_buffer_,FR_force);
    updateForceBuffer(rl_force_buffer_,RL_force);
    updateForceBuffer(rr_force_buffer_,RR_force);
    bool legs_stable=isForceStable(fl_force_buffer_) && isForceStable(fr_force_buffer_) 
                     && isForceStable(rr_force_buffer_) && isForceStable(rl_force_buffer_);
    // RCLCPP_INFO(get_logger(), "FL: %.2f, FR: %.2f, RL: %.2f, RR: %.2f", 
    // FL_force, FR_force, RL_force, RR_force);    
    if (small_acc_ && legs_stable)
        velocity_base_=Eigen::Vector3d::Zero();
    }

void imuVelocityEstimator::updateForceBuffer(std::deque<float> &force_buffer,float new_value) {
    force_buffer.push_back(new_value);
        if (force_buffer.size() > window_size_)
        force_buffer.pop_front();
    }
bool imuVelocityEstimator::isForceStable(std::deque<float> &force_buffer){

    if (force_buffer.size() < window_size_) return false;
    float sq_sum = 0.0f;
    for (auto x : force_buffer) {
        sq_sum += x * x;
    }
    float mean = std::accumulate(force_buffer.begin(), force_buffer.end(), 0.0f) / window_size_;
    float stddev = std::sqrt(sq_sum / window_size_ - mean * mean);
    return stddev < stable_threshold_stddev;
}
void imuVelocityEstimator::calibrateBias() {
    Eigen::Vector3d sum = Eigen::Vector3d::Zero();
    for (const auto& sample : accel_samples_) {
        sum += sample;
    }
    accel_bias_ = sum / static_cast<double>(accel_samples_.size());
    accel_samples_.clear();
}
void imuVelocityEstimator::publishVelocity() {
    if (!is_bias_calibrated_) return;

    auto msg = geometry_msgs::msg::Twist();
    msg.linear.x = velocity_base_.x();
    msg.linear.y = velocity_base_.y();
    msg.linear.z = velocity_base_.z();
    velocity_pub_->publish(msg);
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<imuVelocityEstimator>();
    rclcpp::spin(node);

    rclcpp::shutdown();
    return 0;
}