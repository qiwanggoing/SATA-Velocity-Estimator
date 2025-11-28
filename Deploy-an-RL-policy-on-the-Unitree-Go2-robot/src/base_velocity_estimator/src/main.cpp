#include "base_velocity_estimator/ekf_velocity_estimator.hpp"
#include "base_velocity_estimator/kinematics_velocity_estimator_node.hpp"
#include "rclcpp/rclcpp.hpp"

int main(int argc, char ** argv){
    rclcpp::init(argc,argv);
    auto obs_node=std::make_shared<KinematicsVelocityEstimator>();
    auto ekf_node=std::make_shared<EKFVelocityEstimator>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(obs_node);
    executor.add_node(ekf_node);
    executor.spin();
    rclcpp::shutdown();
    return 0;
}