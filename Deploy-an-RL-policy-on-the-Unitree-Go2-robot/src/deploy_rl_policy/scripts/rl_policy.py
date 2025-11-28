#!/usr/bin/env python3
import rclpy
import torch
# 绝对导入
from sata_config import Go2Config 
from sata_policy_loader import RLPolicy 

import numpy as np
import os
import sys
from rclpy.node import Node
import argparse
from pathlib import Path
from unitree_go.msg import LowState
from std_msgs.msg import Float32MultiArray
# 绝对导入
from xbox_command import XboxController 
# !!! 新增导入 !!!
from geometry_msgs.msg import TwistStamped 

project_root=Path(__file__).parents[4]

# a new import
try:
    from estimator import VelocityEstimator 
except ImportError:
    print("错误: 无法导入 'estimator.VelocityEstimator'。")
    print("请确保 estimator.py 在 src/deploy_rl_policy/scripts/ 目录下。")
    sys.exit(1)
import collections

class dataReciever(Node):
    def __init__(self,config:Go2Config, policy: RLPolicy, simulation: bool, use_ground_truth: bool = False, use_velocity_estimator: bool = False):
        super().__init__("data_reciever")
        self.config = config
        self.policy = policy
        self.cmd_sub=XboxController(self)
        self.use_velocity_estimator=use_velocity_estimator
        
        self.num_actions = 12
        self.last_torques = np.zeros(self.num_actions, dtype=np.float32)
        self.motor_fatigue = np.zeros(self.num_actions, dtype=np.float32)
        self.activation_sign = np.zeros(self.num_actions, dtype=np.float32)
        self.dof_vel_limits = np.full(self.num_actions, self.config.DOF_VEL_LIMITS, dtype=np.float32)
        self.dof_vel_limits = np.full(self.num_actions, self.config.DOF_VEL_LIMITS, dtype=np.float32)
        self.torque_limits = np.full(self.num_actions, self.config.TORQUE_LIMITS, dtype=np.float32)
        self.GAMMA = self.config.GAMMA 
        self.BETA = self.config.BETA
        self.ACTION_SCALE = self.config.ACTION_SCALE
        self.CMD_SCALES = np.array([
            self.config.COMMANDS_SCALES.lin_vel_x,
            self.config.COMMANDS_SCALES.lin_vel_y,
            self.config.COMMANDS_SCALES.ang_vel_yaw
        ], dtype=np.float32)
        
        self.qj = np.zeros(self.num_actions, dtype=np.float32)
        self.dqj = np.zeros(self.num_actions, dtype=np.float32)
        self.obs = np.zeros(60, dtype=np.float32) 
        self.cmd = np.array([0.0, 0.0, 0.0])
        self.low_state=LowState()
        self.base_lin_vel = np.zeros(3, dtype=np.float32) 

        # !!! 新增: 在线零点校准相关变量 !!!
        self.estimator_bias = np.zeros(3, dtype=np.float32)
        self.is_calibrated = False
        self.calibration_samples = []
        # 校准步数，例如 400步 = 2秒 (200Hz 控制频率)
        self.CALIBRATION_STEPS = 400 

        # !!! 新增: PD站立控制相关变量 (从 low_level_ctrl.cpp 提取) !!!
        self.kp_stand = np.full(12, 20.0, dtype=np.float32)
        self.kd_stand = np.full(12, 2.0, dtype=np.float32)
        # 目标站立角度 (FR, FL, RR, RL 顺序)
        self.target_q_stand = np.array([-0.1, 0.8, -1.5,    # FR
                                         0.1, 0.8, -1.5,    # FL
                                        -0.1, 1.0, -1.5,    # RR
                                         0.1, 1.0, -1.5], dtype=np.float32)

        # !!! 新增: 状态机变量 !!!
        self.fsm_state = "CALIBRATE" # 初始状态为趴地校准
        self.pd_stand_counter = 0 # 用于跟踪各状态的时间
        
        # ... (默认关节角度不变) ...

        
        self.default_dof_pos_policy_order = np.array([
            self.config.DEFAULT_JOINT_ANGLES['FL_hip_joint'], self.config.DEFAULT_JOINT_ANGLES['FL_thigh_joint'], self.config.DEFAULT_JOINT_ANGLES['FL_calf_joint'],
            self.config.DEFAULT_JOINT_ANGLES['FR_hip_joint'], self.config.DEFAULT_JOINT_ANGLES['FR_thigh_joint'], self.config.DEFAULT_JOINT_ANGLES['FR_calf_joint'],
            self.config.DEFAULT_JOINT_ANGLES['RL_hip_joint'], self.config.DEFAULT_JOINT_ANGLES['RL_thigh_joint'], self.config.DEFAULT_JOINT_ANGLES['RL_calf_joint'],
            self.config.DEFAULT_JOINT_ANGLES['RR_hip_joint'], self.config.DEFAULT_JOINT_ANGLES['RR_thigh_joint'], self.config.DEFAULT_JOINT_ANGLES['RR_calf_joint']
        ], dtype=np.float32)

        if simulation:
            self.low_state_sub=self.create_subscription(LowState,"/mujoco/lowstate",self.low_state_callback,10)
            print("reading data from simuation")
        else:
            self.low_state_sub=self.create_subscription(LowState,"/lowstate",self.low_state_callback,10) 
            print("reading data from reality")

        if self.use_velocity_estimator:
            self.get_logger().info("Using LSTM Velocity Estimator")
            estimator_path = os.path.join(project_root, "resources", "policies", "velocity_estimator.pt")
            if not os.path.exists(estimator_path):
                self.get_logger().error(f"Velocity estimator model not found at: {estimator_path}")
                sys.exit(1)
            
            # NEW: input_dim=33, LSTM architecture
            self.estimator_model = VelocityEstimator(input_dim=33, hidden_dim=128, output_dim=3)
            self.estimator_model.load_state_dict(torch.load(estimator_path, map_location='cpu'))
            self.estimator_model.eval()
            
            # NEW: History length 11
            self.estimator_history_buffer = collections.deque(maxlen=11)
            self.get_logger().info(f"Velocity estimator loaded successfully. Input dim: 33, History: 11")
        elif use_ground_truth:
            self.get_logger().info("Using Ground Truth Velocity (Sim)")
            # 订阅 MuJoCo 发布的真值速度，并绑定到 velocity_callback 更新 base_lin_vel
            self.velocity_sub = self.create_subscription(
                TwistStamped,
                "/mujoco/ground_truth_velocity", 
                self.velocity_callback,
                10)
        else:
            self.get_logger().info("Using EKF Estimated Velocity")
            self.velocity_sub = self.create_subscription(
                TwistStamped,
                "/ekf/velocity", 
                self.velocity_callback,
                10)
        
        # DEBUG: Subscribe to GT velocity for comparison
        self.gt_vel = np.zeros(3, dtype=np.float32)
        self.gt_vel_sub = self.create_subscription(
            TwistStamped,
            "/mujoco/ground_truth_velocity",
            self.gt_vel_callback,
            10
        )

        self.get_logger().info("Waiting for data")
        
        # self.timer = self.create_timer(0.005, self.run) 
        
        self.target_torque_puber=self.create_publisher(Float32MultiArray,"/rl/target_torques",10)

        self.target_speed_level = 0.2
        self.last_y_state = 0
        self.last_b_state = 0
        self.debug_counter = 0
        self.global_counter = 0

    def gt_vel_callback(self, msg: TwistStamped):
        self.gt_vel[0] = msg.twist.linear.x
        self.gt_vel[1] = msg.twist.linear.y
        self.gt_vel[2] = msg.twist.linear.z

    def velocity_callback(self, msg: TwistStamped):
        self.base_lin_vel[0] = msg.twist.linear.x
        self.base_lin_vel[1] = msg.twist.linear.y
        self.base_lin_vel[2] = msg.twist.linear.z
        
    def low_state_callback(self,msg:LowState):
        self.low_state=msg
        self.run()
    
    def _compute_sata_torques(self, raw_actions, dof_vel_policy_order):
        actions_scaled = raw_actions * self.ACTION_SCALE
        torques_limits = self.torque_limits
        current_activation_sign = np.tanh(actions_scaled / torques_limits)
        self.activation_sign = (current_activation_sign - self.activation_sign) * self.GAMMA + self.activation_sign
        torques = self.activation_sign * torques_limits * (
            1 - np.sign(self.activation_sign) * dof_vel_policy_order / self.dof_vel_limits
        )
        dt = 0.005 
        self.motor_fatigue += np.abs(torques) * dt
        self.motor_fatigue *= self.BETA
        return torques.astype(np.float32)

    def _compute_pd_torques(self, target_q, current_q, current_dq):
        """
        Calculates PD control torques.
        Args:
            target_q (np.array): Desired joint positions (12,).
            current_q (np.array): Current joint positions (12,).
            current_dq (np.array): Current joint velocities (12,).
        Returns:
            np.array: Calculated PD torques (12,).
        """
        # Ensure target_q is mapped to policy order before use if necessary,
        # but here target_q_stand is already in robot order.
        torques = (target_q - current_q) * self.kp_stand - current_dq * self.kd_stand
        return torques.astype(np.float32)

    def run(self):
        self.global_counter += 1
        # if self.use_velocity_estimator and self.global_counter % 50 == 0:
        #      buf_len = len(self.estimator_history_buffer) if hasattr(self, 'estimator_history_buffer') else 0
        #      self.get_logger().info(f"[DEBUG] State: {self.fsm_state} | Cnt: {self.pd_stand_counter} | BufLen: {buf_len}")

        if (self.cmd_sub.axes and self.cmd_sub.axes[2] == -1 and self.cmd_sub.axes[5] == -1):
            sys.exit()
            
        for i in range(12):
            self.qj[i] = self.low_state.motor_state[i].q
            self.dqj[i] = self.low_state.motor_state[i].dq
        
        policy_to_robot_map = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.int32)
        robot_to_policy_map = np.array([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8], dtype=np.int32)

        qj_policy = self.qj[robot_to_policy_map]
        dqj_policy = self.dqj[robot_to_policy_map]
        
        quat = self.low_state.imu_state.quaternion
        ang_vel = np.array(self.low_state.imu_state.gyroscope, dtype=np.float32)
        
        self.cmd = np.zeros(3)
        self.left_button, self.right_button = self.cmd_sub.is_pressed()
        
        if self.left_button and self.right_button:
            raw_lx, raw_ly = self.cmd_sub.get_left_stick()
            raw_az = self.cmd_sub.get_right_stick()
            
            if hasattr(self.cmd_sub, 'buttons') and len(self.cmd_sub.buttons) > 3:
                current_y_state = self.cmd_sub.buttons[3]
                if current_y_state == 1 and self.last_y_state == 0: 
                    self.target_speed_level += 0.1
                    print(f"Speed UP: {self.target_speed_level:.1f} m/s")
                self.last_y_state = current_y_state
                
                current_b_state = self.cmd_sub.buttons[1]
                if current_b_state == 1 and self.last_b_state == 0: 
                    self.target_speed_level -= 0.1
                    if self.target_speed_level < 0.0: 
                        self.target_speed_level = 0.0
                    print(f"Speed DOWN: {self.target_speed_level:.1f} m/s")
                self.last_b_state = current_b_state

            magnitude = np.sqrt(raw_lx**2 + raw_ly**2)
            target_vx = 0.0
            target_vy = 0.0
            
            if magnitude > 0.1: 
                dir_x = raw_lx / magnitude
                dir_y = raw_ly / magnitude
                
                target_vx = dir_x * self.target_speed_level
                target_vy = dir_y * self.target_speed_level
            
            target_wz = raw_az * 1.0 
            self.cmd = np.array([target_vx, target_vy, target_wz])
        
        gravity_orientation = self.get_gravity_orientation(quat)
        
        final_torques_robot_order = np.zeros(12, dtype=np.float32) # Default if not handled by FSM/estimator
        
        # Prepare Scaled Observations for Input Construction (Common to both states)
        obs_ang_vel = ang_vel * self.config.OBS_SCALES.ang_vel
        obs_dof_pos = (qj_policy - self.default_dof_pos_policy_order) * self.config.OBS_SCALES.dof_pos
        obs_dof_vel = dqj_policy * self.config.OBS_SCALES.dof_vel
        commands_scaled = self.cmd * self.CMD_SCALES

        # --- FSM 逻辑 (仅当使用估计器时才激活) ---
        if self.use_velocity_estimator:
            # Shared Feature Calculation for Estimator
            # 1. Linear Acceleration (Body Frame)
            # IMU Acc = a_body - g_body (reaction force). Gravity vector = g_body.
            # So a_body = IMU Acc + g_body
            lin_acc_raw = np.array(self.low_state.imu_state.accelerometer, dtype=np.float32)
            pure_lin_acc = lin_acc_raw + (gravity_orientation * 9.81)
            
            # 2. Construct Feature Vector: [Pos(12), Vel(12), Acc(3), AngVel(3), Grav(3)] -> 33 dim
            # Note: Using RAW joint data (qj_policy, dqj_policy) as training did not scale them for estimator.
            current_features = np.concatenate([
                qj_policy,          # 12
                dqj_policy,         # 12
                pure_lin_acc,       # 3
                ang_vel,            # 3
                gravity_orientation # 3
            ]).astype(np.float32)

            if self.fsm_state == "CALIBRATE":
                self.get_logger().info(f"FSM State: CALIBRATE ({self.pd_stand_counter} / {self.CALIBRATION_STEPS})")
                # 1. 零力矩 (趴在地上)
                final_torques_robot_order = np.zeros(12, dtype=np.float32)

                # 2. 收集估计器校准样本
                self.estimator_history_buffer.append(current_features)

                if len(self.estimator_history_buffer) == 11:
                    input_sequence = np.stack(list(self.estimator_history_buffer), axis=0)
                    input_tensor = torch.from_numpy(input_sequence).unsqueeze(0).float()
                    with torch.no_grad():
                        raw_estimated_velocity = self.estimator_model(input_tensor).squeeze().numpy()
                    
                    self.calibration_samples.append(raw_estimated_velocity)
                    self.pd_stand_counter += 1
                    
                    if self.pd_stand_counter >= 1000:
                        # 校准完成
                        # DISABLE BIAS CALCULATION: The non-zero estimate might be due to tilt, not sensor drift.
                        # Subtracting it permanently is wrong.
                        self.estimator_bias = np.zeros(3, dtype=np.float32)
                            
                        self.is_calibrated = True
                        self.get_logger().info(f"速度估计器准备就绪 (Bias禁用)，默认偏置: {np.array2string(self.estimator_bias, precision=3, floatmode='fixed')}")
                        
                        self.fsm_state = "RUNNING" 
                        self.estimator_history_buffer.clear() # Reset buffer for RL
                        return # End frame
                
                self.base_lin_vel = np.zeros(3, dtype=np.float32)

            elif self.fsm_state == "RUNNING":
                # DEBUG LOGGING (Only log every 50 steps to reduce spam)
                est = self.base_lin_vel
                gt = self.gt_vel
                grav = gravity_orientation
                if self.global_counter % 50 == 0:
                    self.get_logger().info(f"[RUNNING] CMD: [{self.cmd[0]:.2f}, {self.cmd[1]:.2f}, {self.cmd[2]:.2f}] | GT: [{gt[0]:.2f}, {gt[1]:.2f}, {gt[2]:.2f}] | EST: [{est[0]:.2f}, {est[1]:.2f}, {est[2]:.2f}] | GRAV: [{grav[0]:.2f}, {grav[1]:.2f}, {grav[2]:.2f}]")

                self.estimator_history_buffer.append(current_features)

                if len(self.estimator_history_buffer) == 11:
                    input_sequence = np.stack(list(self.estimator_history_buffer), axis=0)
                    input_tensor = torch.from_numpy(input_sequence).unsqueeze(0).float()
                    with torch.no_grad():
                        raw_estimated_velocity = self.estimator_model(input_tensor).squeeze().numpy()
                    
                    # 减去测得的偏置
                    self.base_lin_vel = raw_estimated_velocity - self.estimator_bias
                    
                else:
                    self.base_lin_vel = np.zeros(3, dtype=np.float32)

                # 2. 运行RL策略
                obs_lin_vel = self.base_lin_vel * self.config.OBS_SCALES.lin_vel
                
                self.obs = np.concatenate([
                    obs_lin_vel,
                    obs_ang_vel,
                    gravity_orientation,
                    obs_dof_pos,
                    obs_dof_vel,
                    commands_scaled,
                    self.last_torques,
                    self.motor_fatigue
                ]).astype(np.float32)
                
                self.obs = np.clip(self.obs, -100, 100)
                
                raw_action = self.policy.get_action(self.obs) 
                final_torques_policy_order = self._compute_sata_torques(raw_action, dqj_policy)
                self.last_torques = final_torques_policy_order.copy()
                
                final_torques_robot_order = final_torques_policy_order[policy_to_robot_map]
                
            else: # 未知状态
                self.get_logger().error(f"FSM in unknown state: {self.fsm_state}. Sending zero torques.")
                final_torques_robot_order = np.zeros(12, dtype=np.float32)
                self.base_lin_vel = np.zeros(3, dtype=np.float32)

        else: # EKF or Ground Truth mode
            # This part is the existing code for EKF/Ground Truth mode
            obs_lin_vel = self.base_lin_vel * self.config.OBS_SCALES.lin_vel
            obs_ang_vel = ang_vel * self.config.OBS_SCALES.ang_vel
            obs_dof_pos = (qj_policy - self.default_dof_pos_policy_order) * self.config.OBS_SCALES.dof_pos
            obs_dof_vel = dqj_policy * self.config.OBS_SCALES.dof_vel
            commands_scaled = self.cmd * self.CMD_SCALES

            self.obs = np.concatenate([
                obs_lin_vel,
                obs_ang_vel,
                gravity_orientation,
                obs_dof_pos,
                obs_dof_vel,
                commands_scaled,
                self.last_torques,
                self.motor_fatigue
            ]).astype(np.float32)
            
            self.obs = np.clip(self.obs, -100, 100)
            
            raw_action = self.policy.get_action(self.obs) 
            final_torques_policy_order = self._compute_sata_torques(raw_action, dqj_policy)
            self.last_torques = final_torques_policy_order.copy()
            
            final_torques_robot_order = final_torques_policy_order[policy_to_robot_map]
            
        msg=Float32MultiArray()
        msg.data.extend(final_torques_robot_order.astype(np.float32).tolist())
        self.target_torque_puber.publish(msg)
    @staticmethod
    def get_gravity_orientation(quaternion):
        qw = quaternion[0]
        qx = quaternion[1]
        qy = quaternion[2]
        qz = quaternion[3]
        
        gravity_vec = np.array([0., 0., -1.], dtype=np.float32)
        
        # Revert to standard inverse rotation (conjugate)
        qv = np.array([-qx, -qy, -qz], dtype=np.float32) 
        
        uv = np.cross(qv, gravity_vec)
        uuv = np.cross(qv, uv)
        projected_gravity = gravity_vec + 2 * (qw * uv + uuv)
        
        return projected_gravity

def main():
    rclpy.init()
    
    try:
        from sata_config import Go2Config
        from sata_policy_loader import RLPolicy
    except ImportError:
        print("错误：无法导入 'sata_config' 或 'sata_policy_loader'。")
        print("请确保 sata_config.py 和 sata_policy_loader.py 在 src/deploy_rl_policy/scripts/ 目录下。")
        sys.exit(1)

    config = Go2Config()
    policy_path = "/home/qiwang/SATA/legged_gym/logs/SATA/exported/policies/policy_1.pt" 
    
    if not os.path.exists(policy_path):
        print(f"错误: 找不到策略文件: {policy_path}")
        sys.exit(1)
        
    policy = RLPolicy(policy_path)
    policy.policy.eval() 

    parser = argparse.ArgumentParser()
    parser.add_argument('--is_simulation', type=str, choices=["True", "False"], default="True")
    parser.add_argument('--use_ground_truth', action='store_true', help="Use ground truth velocity from simulation instead of EKF")
    parser.add_argument('--use_velocity_estimator', action='store_true', help="Use the trained LSTM velocity estimator")
    args = parser.parse_args()
    simulation = args.is_simulation == "True"
    use_ground_truth = args.use_ground_truth
    use_velocity_estimator=args.use_velocity_estimator
    
    reciever_node=dataReciever(config=config, policy=policy, simulation=simulation, use_ground_truth=use_ground_truth, use_velocity_estimator=use_velocity_estimator)
    rclpy.spin(reciever_node)
    reciever_node.destroy_node()
    rclpy.shutdown()

if __name__=="__main__":
    main()
