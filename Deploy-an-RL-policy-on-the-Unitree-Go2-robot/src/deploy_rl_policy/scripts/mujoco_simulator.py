#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import mujoco.viewer
import mujoco
import numpy as np
from unitree_go.msg import LowState,LowCmd
from pathlib import Path
from xbox_command import XboxController # 假设 xbox_command 在同一目录
from std_msgs.msg import Float32MultiArray
import threading
import time
from geometry_msgs.msg import TwistStamped

project_root=Path(__file__).parents[4]

class MujocoSimulator(Node):
    def __init__(self):
        super().__init__("mujoco_simulator")
        self.cmd_sub = XboxController(self)  
        self.low_state_puber=self.create_publisher(LowState,"/mujoco/lowstate",10)
        self.pos_pub=self.create_publisher(Float32MultiArray,"/mujoco/pos",10)
        self.force_pub=self.create_publisher(Float32MultiArray,"/mujoco/force",10)
        self.torque_pub=self.create_publisher(Float32MultiArray,"/mujoco/torque",10)
        
        # !!! 在这里添加新的发布者 !!!
        self.gt_vel_pub = self.create_publisher(TwistStamped, "/mujoco/ground_truth_velocity", 10)

        # !!!修改!!!: 订阅 /mujoco/lowcmd (回调函数将被修改)
        self.target_torque_suber=self.create_subscription(LowCmd,"/mujoco/lowcmd",self.target_torque_callback,10)

        # !!! [新增] 直接订阅 RL 输出的纯力矩 !!!
        self.rl_torque_sub = self.create_subscription(
            Float32MultiArray, 
            "/rl/target_torques", 
            self.rl_torque_callback, 
            10
        )

        self.step_counter = 0
        self.xml_path=project_root/"resources"/"go2"/"scene_terrain.xml"
        
        # 初始化 Mujoco
        self.init_mujoco()
        self.tau=[0.0]*12 # tau 将由 target_torque_callback 直接计算和设置
        
        # !!! 修复: 删除 500Hz 定时器，避免双重发布 !!!
        # self.timer = self.create_timer(0.002, self.publish_sensor_data) 
        
        self.running=True
        self.recieve_data=True # !!! 修复: 默认允许仿真运行，避免死锁 !!!
        self.sim_thread=threading.Thread(target=self.step_simulation)
        self.sim_thread.start()

    def init_mujoco(self):
        """Initialize Mujoco model and data"""
        
        self.m = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.d = mujoco.MjData(self.m)
        self.m.opt.timestep = 0.005 # 保持 0.005s (200Hz)
        self.viewer = mujoco.viewer.launch_passive(self.m, self.d)
        print("Number of qpos:", self.m.nq)
        print("Joint order:")
        for i in range(self.m.njnt):
            print(f"{i}: {self.m.joint(i).name}")
   
    # !!!修改!!!: target_torque_callback 现在处理 PD 和 纯力矩
    def target_torque_callback(self,msg: LowCmd):
        self.recieve_data=True
        temp_tau = [0.0] * 12 # 临时力矩列表
        
        for i in range(12):
            kp = msg.motor_cmd[i].kp
            kd = msg.motor_cmd[i].kd
            
            if kp > 0:
                # 1. PD 命令 (用于站立/趴下)
                target_q = msg.motor_cmd[i].q
                target_dq = msg.motor_cmd[i].dq 
                
                # 从MuJoCo数据中获取当前状态
                current_q = self.d.qpos[7+i]
                current_dq = self.d.qvel[6+i]
                
                # 计算PD力矩 (静态 PD 控制)
                pd_torque = self.pd_control(target_q, current_q, kp, current_dq, kd)
                
                # 加上前馈力矩 (如果有的话)
                temp_tau[i] = pd_torque + msg.motor_cmd[i].tau
                
            else:
                # 2. 纯力矩命令 (Kp=0, Kd=0) (用于RL策略)
                temp_tau[i] = msg.motor_cmd[i].tau
        
        # 将计算好的力矩赋给 self.tau (供仿真线程使用)
        self.tau = temp_tau
    
    # !!! [新增] 处理纯力矩的回调函数 !!!
    def rl_torque_callback(self, msg: Float32MultiArray):
        self.recieve_data = True # 标记已收到数据，开始仿真步进
        
        # 确保数据长度正确 (12个关节)
        if len(msg.data) == 12:
            self.tau = list(msg.data)
        else:
            self.get_logger().warn(f"Received torque data with wrong length: {len(msg.data)}")
            
    # !!!删除!!!: update_tau 方法
    
    def step_simulation(self):
        while self.viewer.is_running() and self.running :
            if not self.recieve_data:
                continue
            step_start=time.time()
            
            # !!!修改!!!: 直接应用 self.tau (由回调函数计算)
            self.d.ctrl[:]=self.tau 
            
            Torque=Float32MultiArray()
            Torque.data=self.tau
            self.torque_pub.publish(Torque)
            
            # Mujoco step
            mujoco.mj_step(self.m, self.d)  
            self.publish_sensor_data()
            # Sync Mujoco viewer
            self.viewer.sync()
            
            time_until_next_step = self.m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            
    def stop_simulation(self):
        self.running=False
        self.sim_thread.join()


    def publish_sensor_data(self):
        # ... (发布传感器数据的逻辑保持不变) ...
        low_state_msg=LowState()
        for i in range(12):
            low_state_msg.motor_state[i].q=self.d.qpos[7+i]
            low_state_msg.motor_state[i].dq=self.d.qvel[6+i]
        low_state_msg.imu_state.quaternion=self.d.qpos[3:7].astype(np.float32)
        low_state_msg.imu_state.gyroscope=self.d.sensordata[40:43].astype(np.float32)
        # !!! 新增: 填充加速度计数据 !!!
        # 注意: d.sensordata[43:46] 对应 xml 中的 accelerometer 传感器
        low_state_msg.imu_state.accelerometer = self.d.sensordata[43:46].astype(np.float32)
        
        self.low_state_puber.publish(low_state_msg)
        pos=Float32MultiArray()
        pos.data=self.d.qpos[:19].tolist()
        self.pos_pub.publish(pos)
  
        Force=Float32MultiArray()
        f1=self.d.sensordata[55:55+3]+[0,0,0]
        f2=self.d.sensordata[55+3:55+6]+[0,0,0]
        f3=self.d.sensordata[55+6:55+9]+[0,0,0]
        f4=self.d.sensordata[55+9:55+12]+[0,0,0]
        # Force.data=f1+f2+f3+f4
        # self.force_pub.publish(Force)
        # !!! 修改开始: 真值速度发布 !!!
        gt_vel_msg = TwistStamped()
        gt_vel_msg.header.stamp = self.get_clock().now().to_msg()
        gt_vel_msg.header.frame_id = "base_link" # 标记为机身坐标系
        
        # 1. 获取原始数据
        world_vel = self.d.qvel[0:3]     # 世界系线速度
        quat = self.d.qpos[3:7]          # 姿态四元数 [w, x, y, z]
        
        # 2. 执行坐标转换 (World -> Body)
        body_vel = self.transform_world_to_body(world_vel, quat)
        
        # 3. 赋值 (注意: 这里的 x 是车头方向, y 是左侧方向)
        gt_vel_msg.twist.linear.x = float(body_vel[0])
        gt_vel_msg.twist.linear.y = float(body_vel[1])
        gt_vel_msg.twist.linear.z = float(body_vel[2])
        
        # 4. 也可以顺便填充角速度 (虽然策略用的是 IMU 数据，但 GT 也可以发一下)
        # 注意: MuJoCo 的 qvel[3:6] 对于自由关节通常是世界系角速度，但也需要转换
        # 不过你的策略直接用的 IMU (imu_state.gyroscope), 这里可以保持 0 或者只发线速度
        
        self.gt_vel_pub.publish(gt_vel_msg)
    
    @staticmethod
    def transform_world_to_body(v, q):
        """
        将世界坐标系下的向量 v 旋转到机身坐标系。
        q: 四元数 [w, x, y, z] (MuJoCo 格式)
        v: 向量 [vx, vy, vz]
        """
        # 提取四元数分量
        w, x, y, z = q
        
        # 我们需要旋转 q 的逆（共轭），因为 q 表示从机身到世界的旋转
        # 对于单位四元数，逆就是 [w, -x, -y, -z]
        x, y, z = -x, -y, -z
        
        # 使用 Rodrigues 旋转公式的高效实现: v' = v + 2 * r x (r x v + w * v)
        # 其中 r 是四元数的向量部分 (x, y, z)
        q_vec = np.array([x, y, z], dtype=np.float32)
        v_vec = np.array(v, dtype=np.float32)
        
        t = 2.0 * np.cross(q_vec, v_vec)
        v_body = v_vec + w * t + np.cross(q_vec, t)
        
        return v_body

    @staticmethod
    def pd_control(target_q, q, kp, dq, kd):
        """Calculates torques from position commands"""
        # 注意：原始代码的pd是 (target_q - q) * kp - dq * kd
        # 我们这里假设 target_dq = 0
        torques = (target_q - q) * kp - dq * kd
        return torques
    
def main(args=None):
    rclpy.init(args=args)
    node = MujocoSimulator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_simulation()
        node.viewer.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()