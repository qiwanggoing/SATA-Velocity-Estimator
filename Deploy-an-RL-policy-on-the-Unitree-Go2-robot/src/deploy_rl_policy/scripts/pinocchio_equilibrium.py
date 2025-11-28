import pinocchio as pin
from pinocchio import JointModelFreeFlyer # type: ignore
from pinocchio.robot_wrapper import RobotWrapper
import numpy as np
import rclpy
import rclpy.node as Node
from std_msgs.msg import Float32MultiArray

urdf_path = "/home/song/unitree_rl_gym/resources/robots/go2/urdf/go2.urdf"
model_dir = "/home/song/unitree_rl_gym/resources/robots/go2/dae"  # 如果有 STL/DAE 网格
class EquiValidation(Node):
    def __init__(self):
        super().__init__("validate equilibrium")
        self.force_sub=self.create_subscription(Float32MultiArray,"/mujoco/force",self.force_callback)        
        self.torque_sub=self.create_subscription(Float32MultiArray,"/mujoco/torque",self.torque_callback)        
        self.pos_sub=self.create_subscription(Float32MultiArray,"/mujoco/pos",self.pos_callback)    
        self.pos=Float32MultiArray()
        self.force=Float32MultiArray()
        self.torque=Float32MultiArray()    

    def force_callback(self,msg):
        self.force.data=msg.data

    def torque_callback(self,msg):
        self.torque.data=msg.data
        
    def pos_callback(self,msg):
        self.pos.data=msg.data

    def validate(self):
        robot = RobotWrapper.BuildFromURDF(
            filename=urdf_path,
            package_dirs=model_dir,
            root_joint=pin.JointModelFreeFlyer()  # Go2 是浮动底座机器人 # type: ignore
        )
        model = robot.model
        data = model.createData()

        total_mass = pin.computeTotalMass(model)
        print("Total mass:", total_mass)  # Unitree Go2 总质量应 ≈ 20-30 kg

        q = pin.neutral(model)  # 默认中立配置（基座水平，关节角度为0）
        q[0:3]=[0.002,-0.07,0.30]
        quat = np.array([ 0.002, -0.01, 0.71,0.69])
        q[3:7] = quat / np.linalg.norm(quat)
        q[7:]=np.array([0.08,0.8,-1.59,-0.07,0.86,-1.57,0.17,0.95,-1.61,-0.18,0.952,-1.63])
        pin.computeGeneralizedGravity(model, data, q)
        G = data.g  # G(q)
        foot_names=["FL_foot","FR_foot","RL_foot","RR_foot"]
        # f_contacts=np.array([
        #     [-23.66897,9.82887127, -38.30377442,0,0,0],
        #     [-17.85930665, -10.86583988, -32.23682423,0,0,0],
        #     [-24.31775983,  -6.91372673, -13.45616742 ,0,0,0],
        #     [-28.50262735,  7.88130676, -21.53058759 ,0,0,0]
        # ])
        f_contacts=np.array([
            [-26, 4, -29.23682423,0,0,0],
            [-23.66897,-1.8, -33.30377442,0,0,0],
            [-8,  3, -27 ,0,0,0],
            [-2.50262735,  -7.88130676, -26.53058759 ,0,0,0]
        ])
        tau_c=np.zeros(model.nv)
        for foot_name, f_contact in zip(foot_names, f_contacts):
            frame_id = model.getFrameId(foot_name)
            pin.forwardKinematics(model, data, q)
            pin.computeJointJacobians(model, data, q)
            J = pin.getFrameJacobian(model, data, frame_id, pin.LOCAL)  # 6 x nv
            # print(J)
            tau_c += J.T @ f_contact
        tau_total = G + tau_c
        print(tau_total[6:])

        # print("误差:", tau_total)