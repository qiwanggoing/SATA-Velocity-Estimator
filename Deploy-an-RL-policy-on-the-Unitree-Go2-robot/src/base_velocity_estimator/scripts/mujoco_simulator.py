#!/usr/bin/env python3
import time
import threading
import rclpy
from rclpy.node import Node
import os

import mujoco.viewer
import mujoco
import numpy as np
from legged_gym import LEGGED_GYM_ROOT_DIR
import torch
import yaml
from pathlib import Path
from xbox_command import XboxController
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Twist, Vector3
from std_msgs.msg import Float32MultiArray

# Import the VelocityEstimator class
# Ensure estimator.py is in the same directory or PYTHONPATH
from estimator import VelocityEstimator 

class MujocoSimulator(Node):
    def __init__(self):
        super().__init__("mujoco_simulator")
        self.cmd_sub = XboxController(self)  
        self.load_config()
        
        # Publishers
        self.imu_pub=self.create_publisher(Imu,"/imu/data",10)
        self.contact_force_pub=self.create_publisher(Float32MultiArray,"/contact",10)
        self.joint_pub=self.create_publisher(Float32MultiArray,"/joint_angels",10)
        self.joint_vel_pub=self.create_publisher(Float32MultiArray,"/joint_velocities",10)
        self.omega_pub=self.create_publisher(Float32MultiArray,"gyro",10)
        self.z_axis_force_pub=self.create_publisher(Float32MultiArray,"/z_axis_force",10)
        self.true_vel_pub=self.create_publisher(Float32MultiArray,"/true_velocities",10)
        self.est_vel_pub=self.create_publisher(Float32MultiArray,"/estimated_velocities",10) # New publisher

        self.step_counter = 0

        # Initialize Mujoco
        self.init_mujoco()

        # --- Load Policy and Estimator ---
        # 1. Load Policy (JIT)
        print(f"Loading policy from: {self.policy_path}")
        self.policy = torch.jit.load(self.policy_path)
        
        # 2. Load Estimator
        # Assuming estimator weights are in the same folder as policy with name 'sata_estimator.pt'
        # You might need to adjust the filename below matches what you copied
        self.estimator_path = str(Path(self.policy_path).parent / "sata_estimator.pt")
        print(f"Loading estimator from: {self.estimator_path}")
        
        self.estimator = VelocityEstimator(input_dim=45, temporal_steps=6).to('cpu')
        self.estimator.load_state_dict(torch.load(self.estimator_path, map_location='cpu'))
        self.estimator.eval()

        # --- Initialize State Variables ---
        self.action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_dof_pos = self.default_angles.copy()
        self.obs = np.zeros(self.num_obs, dtype=np.float32)
        
        # --- History Buffer for Estimator ---
        # Shape: (Batch=1, Seq=6, Dim=45)
        self.history_len = 6
        self.proprio_dim = 45
        self.obs_history_buffer = torch.zeros(1, self.history_len, self.proprio_dim)

        self.timer = self.create_timer(self.simulation_dt, self.step_simulation)

    def load_config(self):
        current_file = Path(__file__).resolve()
        parent_dir = current_file.parent.parent
        config_file = parent_dir /'config'/'go2.yaml'
        with open(config_file, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            self.policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
            self.xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
            self.simulation_duration = config["simulation_duration"]
            self.simulation_dt = config["simulation_dt"]
            self.control_decimation = config["control_decimation"]
            self.kps = np.array(config["kps"], dtype=np.float32)
            self.kds = np.array(config["kds"], dtype=np.float32)
            self.default_angles = np.array(config["default_angles"], dtype=np.float32)
            self.lin_vel_scale = config["lin_vel_scale"]
            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]

    def init_mujoco(self):
        """Initialize Mujoco model and data"""
        self.m = mujoco.MjModel.from_xml_path(self.xml_path)
        self.d = mujoco.MjData(self.m)
        # Initial pose
        self.d.qpos[3:7]=[0.9932722449302673, 0.008041736669838428, 0.0063380408100783825, -0.11535090208053589]
        laydown=[-0.63224804,  0.8544461,  -2.7341957,   0.73985434,  0.9071291,  -2.6931078,
                 -0.7145372,   0.43530887, -2.7260761,   0.76852196,  0.8470082,  -2.7184937 ]
        for i in range(12):
            self.d.qpos[7+i] = laydown[i] 
        self.m.opt.timestep = self.simulation_dt
        self.viewer = mujoco.viewer.launch_passive(self.m, self.d)

    def step_simulation(self):
        """Main simulation step"""
        self.step_counter += 1

        # PD Control
        tau = self.pd_control(
            self.target_dof_pos,
            self.d.qpos[7:],
            self.kps,
            self.d.qvel[6:],
            self.kds
        )
        sequence = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        tau = [tau[index] for index in sequence]
        self.d.ctrl[:] = tau
        
        # Mujoco step
        mujoco.mj_step(self.m, self.d)
        
        # Publish sensor data
        self.publish_sensor_data()
        
        # Policy inference (every N steps)
        if self.step_counter % self.control_decimation == 0:
            self.run_policy()
        
        # Sync Mujoco viewer
        self.viewer.sync()

    def run_policy(self):
        """Run estimator and policy"""
        
        # 1. Read Joystick Command
        self.cmd = np.zeros(3)
        self.left_button, self.right_button = self.cmd_sub.is_pressed()
        if self.left_button and self.right_button:
            linear_x, linear_y = self.cmd_sub.get_left_stick()
            angular_z = self.cmd_sub.get_right_stick()
            self.cmd = np.array([linear_x, linear_y, angular_z])

        # 2. Build Proprioceptive Observation (45 dims)
        # Indices correspond to:
        # 0-2: Angular Velocity
        # 3-5: Projected Gravity
        # 6-8: Commands
        # 9-20: DOF Pos
        # 21-32: DOF Vel
        # 33-44: Last Actions
        
        proprio_obs = np.zeros(45, dtype=np.float32)
        
        ang_vel = self.d.sensordata[40:43] * self.ang_vel_scale
        gravity_vec = self.get_gravity_orientation(self.d.qpos[3:7])
        self.grav_acc = 9.81 * gravity_vec # For IMU pub
        
        proprio_obs[0:3] = ang_vel
        proprio_obs[3:6] = gravity_vec
        proprio_obs[6:9] = self.cmd * self.cmd_scale
        proprio_obs[9:21] = (self.d.qpos[7:19] - self.default_angles) * self.dof_pos_scale
        proprio_obs[21:33] = self.d.qvel[6:18] * self.dof_vel_scale
        proprio_obs[33:45] = self.action

        # 3. Update History Buffer
        current_proprio_tensor = torch.from_numpy(proprio_obs).unsqueeze(0).unsqueeze(0) # (1, 1, 45)
        self.obs_history_buffer = torch.cat((self.obs_history_buffer[:, 1:], current_proprio_tensor), dim=1)
        
        # 4. Run Estimator
        with torch.no_grad():
            estimated_vel = self.estimator(self.obs_history_buffer) # (1, 3)
            est_vel_np = estimated_vel.cpu().numpy().squeeze()

        # Publish estimated velocity for debugging
        est_vel_msg = Float32MultiArray()
        est_vel_msg.data = list(est_vel_np)
        self.est_vel_pub.publish(est_vel_msg)

        # 5. Build Full Observation for Policy (235 dims)
        # obs[0:3] = Linear Velocity (Using ESTIMATED value now!)
        # obs[3:48] = Proprioception (Same as above)
        # ... remaining dims are height measurements (if any) or zeros if blind/simulated flat ground
        
        self.obs[:3] = est_vel_np * self.lin_vel_scale # IMPORTANT: Apply scale if your training did! 
        # Note: Usually lin_vel_scale is 2.0. Ensure training estimator target was unscaled or scaled. 
        # In standard legged_gym, the 'lin_vel' in obs buffer is scaled. 
        # BUT, our estimator target in on_policy_runner was: target_vel = obs[:, :3].detach() 
        # which IS ALREADY SCALED by 2.0.
        # So the estimator PREDICTS the SCALED velocity. 
        # Therefore, we might NOT need to multiply by scale again if the estimator output is already scaled.
        # HOWEVER, looking at typical implementations, usually we regress real velocity.
        # Let's check on_policy_runner again. 
        # obs[:, :3] IS SCALED in legged_robot.py compute_observations().
        # So the estimator learned to predict SCALED velocity directly.
        # So: self.obs[:3] = est_vel_np is correct (no double scaling).
        # Wait, let's check safety.
        # If I trained `target = obs[:, :3]`, then `pred` is scaled.
        # So `self.obs[:3] = pred` is correct. 
        # The line `self.obs[:3] = est_vel_np * self.lin_vel_scale` implies est_vel_np is physical units.
        # **CORRECTION**: The estimator trained on `obs[:, :3]`. `obs` is already scaled.
        # So `est_vel_np` is already scaled.
        # We should assign it directly: `self.obs[:3] = est_vel_np`
        
        self.obs[:3] = est_vel_np 

        self.obs[3:48] = proprio_obs # Fill the rest
        
        # Handle height measurements if needed (obs[48:]). 
        # For flat ground sim, usually assumed 0 or handled by environment. 
        # If your policy expects 235 dims, ensure the rest are filled appropriately.
        
        # 6. Run Policy
        obs_tensor = torch.from_numpy(self.obs).unsqueeze(0)
        self.action = self.policy(obs_tensor).detach().numpy().squeeze()
        
        self.target_dof_pos = self.action * self.action_scale + self.default_angles
        # print(self.target_dof_pos)

    def publish_sensor_data(self):
        imu_msg = Imu()
        if not hasattr(self,"grav_acc"): return
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.linear_acceleration.x = self.d.sensordata[43]+self.grav_acc[0]
        imu_msg.linear_acceleration.y = self.d.sensordata[44]+self.grav_acc[1]
        imu_msg.linear_acceleration.z = self.d.sensordata[45]+self.grav_acc[2]
        self.imu_pub.publish(imu_msg)
        
        # Publish True Velocity (Ground Truth) for comparison
        true_velocity_array=Float32MultiArray()
        true_velocity_array.data=list(self.d.sensordata[52:55]) # Ensure this index matches your XML sensor
        self.true_vel_pub.publish(true_velocity_array)
        
        # ... (Rest of the publishers same as before) ...
        array=Float32MultiArray()
        fl_force_list=np.array([self.d.sensordata[i] for i in range (55,58)])
        fr_force_list=np.array([self.d.sensordata[i] for i in range (58,61)])
        rl_force_list=np.array([self.d.sensordata[i] for i in range (61,64)])
        rr_force_list=np.array([self.d.sensordata[i] for i in range (64,67)])
        FL_force=np.linalg.norm(fl_force_list)
        FR_force=np.linalg.norm(fr_force_list)
        RL_force=np.linalg.norm(rl_force_list)
        RR_force=np.linalg.norm(rr_force_list)
        array.data=[FL_force,FR_force,RL_force,RR_force]
        self.contact_force_pub.publish(array)
        angles_array=Float32MultiArray()
        angles_array.data=list(self.d.qpos[7:])
        self.joint_pub.publish(angles_array)
        q_dot_array=Float32MultiArray()
        q_dot_array.data=list(self.d.qvel[6:])
        self.joint_vel_pub.publish(q_dot_array)
        omega_array=Float32MultiArray()
        omega_array.data=list(self.d.sensordata[40:43])
        self.omega_pub.publish(omega_array)
        z_axis_force_array=Float32MultiArray()
        z_axis_force_array.data=[self.d.sensordata[i] for i in range (57,67,3)]
        self.z_axis_force_pub.publish(z_axis_force_array)


    @staticmethod
    def get_gravity_orientation(quaternion):
        qw = quaternion[0]
        qx = quaternion[1]
        qy = quaternion[2]
        qz = quaternion[3]

        gravity_orientation = np.zeros(3)

        gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
        gravity_orientation[1] = -2 * (qz * qy + qw * qx)
        gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

        return gravity_orientation

    @staticmethod
    def pd_control(target_q, q, kp, dq, kd):
        torques=(target_q - q) * kp -  dq * kd
        return torques
    
def main(args=None):
    rclpy.init(args=args)
    node = MujocoSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.viewer.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()