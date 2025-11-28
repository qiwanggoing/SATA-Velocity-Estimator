#include "base_velocity_estimator/kinematics_velocity_estimator_node.hpp"
#include "unitree_go/msg/low_state.hpp"

KinematicsVelocityEstimator::KinematicsVelocityEstimator() : Node("kinematics_velocity_estimator")
{
    low_state_sub_ = create_subscription<unitree_go::msg::LowState>(
        "lowstate", rclcpp::SensorDataQoS(),
        std::bind(&KinematicsVelocityEstimator::lowStateCallback, this, std::placeholders::_1));

    velocity_pub_ = create_publisher<geometry_msgs::msg::Twist>("/obs_velocity", 10);

    // 3. 初始化 Pinocchio 模型 (请确认 URDF 路径正确!)
    std::string urdf_file_name = "/home/qiwang/Deploy-an-RL-policy-on-the-Unitree-Go2-robot/resources/go2/go2.urdf";
    try {
        pinocchio::urdf::buildModel(urdf_file_name, model);
        data = Data(model);
        model_loaded_ = true;
        RCLCPP_INFO(get_logger(), "Pinocchio model loaded successfully.");
    } catch (const std::exception &e) {
        RCLCPP_ERROR(get_logger(), "Failed to load URDF: %s", e.what());
        model_loaded_ = false;
    }

    // 4. 初始化数据容器 (12个关节)
    cur_joint_angles_.resize(12, 0.0);
    cur_joint_velocities_.resize(12, 0.0);
    cur_omega_.resize(3, 0.0);
    cur_foot_force_.resize(4, 0.0); // 用于存储足端压力

    // 5. 定义足端映射 (对应 Policy/URDF 的顺序: FL, FR, RL, RR)
    // 这里的 key 0,1,2,3 对应 publishVelocity 循环中的 i
    name_mapping_ = {{0, "FL_foot"}, {1, "FR_foot"}, {2, "RL_foot"}, {3, "RR_foot"}};
    
    // 定义关节名称 (Pinocchio 期望的顺序)
    joints_mapping_[0] = {"FL_hip_joint", "FL_thigh_joint", "FL_calf_joint"};
    joints_mapping_[1] = {"FR_hip_joint", "FR_thigh_joint", "FR_calf_joint"};
    joints_mapping_[2] = {"RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"};
    joints_mapping_[3] = {"RR_hip_joint", "RR_thigh_joint", "RR_calf_joint"};

    // 初始化滤波器
    for (int i = 0; i < 4; ++i) {
        filters_[i] = std::make_unique<Vector3dMedianFilter>(10);
    }
    
    RCLCPP_INFO(get_logger(), "Kinematics Estimator (Sim2Real) Initialized.");
}

// !!! 全新的回调函数：解析 LowState 并处理映射 !!!
void KinematicsVelocityEstimator::lowStateCallback(const unitree_go::msg::LowState::SharedPtr msg)
{
    if (!model_loaded_) return;

    // 1. 解析 IMU 角速度
    cur_omega_[0] = msg->imu_state.gyroscope[0];
    cur_omega_[1] = msg->imu_state.gyroscope[1];
    cur_omega_[2] = msg->imu_state.gyroscope[2];

    // 2. 关节数据重映射 (Joint Remapping)
    // ---------------------------------------------------------
    // Unitree 实机源顺序 (Source): [FR, FL, RR, RL]
    // 你的 URDF/策略目标顺序 (Dest): [FL, FR, RL, RR]
    // ---------------------------------------------------------

    // 辅助 Lambda: 将 Unitree 的某条腿 (src_idx) 复制到 估计器的目标位置 (dst_idx)
    auto map_leg_data = [&](int unitree_leg_start_idx, int estimator_leg_start_idx) {
        for(int i=0; i<3; i++) {
            cur_joint_angles_[estimator_leg_start_idx + i] = msg->motor_state[unitree_leg_start_idx + i].q;
            cur_joint_velocities_[estimator_leg_start_idx + i] = msg->motor_state[unitree_leg_start_idx + i].dq;
        }
    };

    // --- 执行严格映射 ---
    
    // A. 目标: FL (Policy 0-2) <--- 源: FL (Unitree 3-5)
    map_leg_data(3, 0); 
    
    // B. 目标: FR (Policy 3-5) <--- 源: FR (Unitree 0-2)
    map_leg_data(0, 3); 
    
    // C. 目标: RL (Policy 6-8) <--- 源: RL (Unitree 9-11)
    map_leg_data(9, 6); 

    // D. 目标: RR (Policy 9-11) <--- 源: RR (Unitree 6-8)
    map_leg_data(6, 9);

    // ---------------------------------------------------------

    // 3. 解析足端压力 (用于接触检测)
    // Unitree foot_force 顺序: [FR, FL, RR, RL]
    // 目标 filters_ 顺序: 0:FL, 1:FR, 2:RL, 3:RR
    
    cur_foot_force_[0] = msg->foot_force[1]; // FL <- Unitree FL (Index 1)
    cur_foot_force_[1] = msg->foot_force[0]; // FR <- Unitree FR (Index 0)
    cur_foot_force_[2] = msg->foot_force[3]; // RL <- Unitree RL (Index 3)
    cur_foot_force_[3] = msg->foot_force[2]; // RR <- Unitree RR (Index 2)

    // 4. 立即计算并发布
    publishVelocity();
}

void KinematicsVelocityEstimator::publishVelocity()
{
    // 此时 cur_joint_angles_ 已经是 FL, FR, RL, RR 的正确顺序
    Eigen::VectorXd q = Eigen::Map<Eigen::VectorXd>(cur_joint_angles_.data(), 12);
    Eigen::VectorXd dq = Eigen::Map<Eigen::VectorXd>(cur_joint_velocities_.data(), 12);
    Eigen::Vector3d omega = Eigen::Map<Eigen::Vector3d>(cur_omega_.data());

    Eigen::Vector3d estimated_linear_velocity = Eigen::Vector3d::Zero();
    int contact_count = 0;

    // 遍历 4 条腿 (0:FL, 1:FR, 2:RL, 3:RR)
    for (int i = 0; i < 4; i++) 
    {
        // 简单的接触判断：压力 > 15N 认为接地
        if (cur_foot_force_[i] > 15.0) 
        {
            auto result = compute_kinematics_and_jacobian(q, name_mapping_[i], joints_mapping_[i]);
            Eigen::Vector3d p_foot_rel = result.first;
            Eigen::Matrix3d J_linear = result.second;

            // 提取该腿的 dq (3维)
            // 注意：这里的 dq 也是重排后的，直接取 i*3 即可对应
            Eigen::Vector3d dq_leg = dq.segment(i * 3, 3); 
            
            // v_foot_rel = J * dq_leg
            Eigen::Vector3d v_foot_rel = J_linear * dq_leg;

            // v_base = -v_foot_rel - omega x p_foot_rel
            Eigen::Vector3d v_base_i = -v_foot_rel - omega.cross(p_foot_rel);

            // 滤波
            v_base_i = filters_[i]->filter(v_base_i);
            
            estimated_linear_velocity += v_base_i;
            contact_count++;
        }
    }

    if (contact_count > 0) {
        estimated_linear_velocity /= contact_count;
    } else {
        estimated_linear_velocity.setZero(); 
    }

    // 发布观测速度
    geometry_msgs::msg::Twist msg;
    msg.linear.x = estimated_linear_velocity.x();
    msg.linear.y = estimated_linear_velocity.y();
    msg.linear.z = estimated_linear_velocity.z();
    velocity_pub_->publish(msg);
}

std::pair<Eigen::Vector3d, Eigen::Matrix3d> KinematicsVelocityEstimator::compute_kinematics_and_jacobian(
    Eigen::VectorXd &q,
    const std::string &frame_name, 
    const std::vector<std::string> &joint_names)
{
    // 获取 Frame ID
    FrameIndex frame_id = model.getFrameId(frame_name);

    // 前向运动学
    forwardKinematics(model, data, q);
    updateFramePlacements(model, data);

    // 获取相对基座的变换矩阵
    const auto &T_base_link = data.oMf[frame_id]; 
    Eigen::Vector3d position = T_base_link.translation();

    // 计算雅可比矩阵
    Eigen::MatrixXd J(6, model.nv); 
    J.setZero();
    computeFrameJacobian(model, data, q, frame_id, LOCAL_WORLD_ALIGNED, J);
    
    // 提取相关列
    // 这里非常关键：我们需要根据 Pinocchio 模型中的关节 ID 来提取 J 的列
    std::vector<int> indices;
    for (const auto &name : joint_names)
    {
        if(model.existJointName(name)){
            JointIndex jid = model.getJointId(name);
            indices.push_back(model.joints[jid].idx_v());
        } else {
            // 如果找不到关节名，说明 URDF 可能不匹配
            // RCLCPP_ERROR(get_logger(), "Joint %s not found in model!", name.c_str());
        }
    }

    Eigen::Matrix3d J_linear = Eigen::Matrix3d::Zero();
    if(indices.size() == 3) {
        for (int i = 0; i < 3; i++)
            J_linear.col(i) = J.block(0, indices[i], 3, 1);
    }

    return {position, J_linear};
}