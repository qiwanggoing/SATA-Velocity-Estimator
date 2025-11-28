import numpy as np

class Go2Config:
    # --- 关节默认角度 ---
    # 来源: marmotlab/sata/SATA-8fc422af3fec463a408779b1685c2453d0040be8/legged_gym/legged_gym/envs/go2/go2_torque/go2_torque_config.py
    #
    DEFAULT_JOINT_ANGLES = { 
        'FL_hip_joint': 0.1,
        'RL_hip_joint': 0.1,
        'FR_hip_joint': -0.1,
        'RR_hip_joint': -0.1,

        'FL_thigh_joint': 1.45,
        'RL_thigh_joint': 1.45,
        'FR_thigh_joint': 1.45,
        'RR_thigh_joint': 1.45,

        'FL_calf_joint': -2.5,
        'RL_calf_joint': -2.5,
        'FR_calf_joint': -2.5,
        'RR_calf_joint': -2.5,
    }

    # --- 观测缩放因子 ---
    # 来源: marmotlab/sata/SATA-8fc422af3fec463a408779b1685c2453d0040be8/legged_gym/legged_gym/envs/base/legged_robot_config.py
    #
    class OBS_SCALES:
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
    
    # --- 动作与力矩参数 ---
    # 来源: marmotlab/sata/SATA-8fc422af3fec463a408779b1685c2453d0040be8/legged_gym/legged_gym/envs/go2/go2_torque/go2_torque_config.py
    #
    ACTION_SCALE = 5.0 
    
    # 来源: marmotlab/sata/SATA-8fc422af3fec463a408779b1685c2453d0040be8/legged_gym/legged_gym/envs/go2/go2_torque/go2_torque.py (_init_buffers)
    #
    TORQUE_LIMITS = 23.5 

    # --- SATA MODIFICATION: 增加生物力学模型参数 ---

    # 来源: sata_jump/.../go2_torque.py (_compute_torques)
    # 对应论文 Eq. 2 中的 γ (gamma)
    #
    GAMMA = 0.6 # 激活模型平滑因子

    # 来源: sata_jump/.../go2_torque.py (_compute_torques)
    # 对应论文 Eq. 4 中的 β (beta)
    #
    BETA = 0.9 # 疲劳恢复因子

    # 来源: sata_jump/.../legged_robot.py (_process_dof_props)
    # (在 go2_torque.py 中使用，但从基类读取)
    # 对应论文 Eq. 3 中的 q_dot_limit
    #
    DOF_VEL_LIMITS = 30.0 # 关节速度限制 (rad/s), 保持原样
    # --- END SATA MODIFICATION ---

    # --- 命令缩放因子 ---
    # 来源: legged_gym/envs/base/legged_robot.py (self.commands_scale的初始化方式)
    #
    class COMMANDS_SCALES:
        lin_vel_x = 2.0
        lin_vel_y = 2.0
        ang_vel_yaw = 0.25

# 注意：原有的 P_GAINS 和 D_GAINS 我们暂时保留，但在后续步骤中会绕过它们
# 因为SATA是纯力矩控制
P_GAINS = np.array([100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100])
D_GAINS = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

class Commands:
    def __init__(self):
        self.lin_vel_x = 0.0
        self.lin_vel_y = 0.0
        self.ang_vel_yaw = 0.0