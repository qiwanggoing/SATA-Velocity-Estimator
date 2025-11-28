# SATA Sim2Real: 在 Unitree Go2 上部署 SATA 力矩控制策略

本仓库是 [**glowing-torch/Deploy-an-RL-policy-on-the-Unitree-Go2-robot**](https://github.com/glowing-torch/Deploy-an-RL-policy-on-the-Unitree-Go2-robot) 项目的修改版。

**核心变更：**
> 将原版的 **PD 位置控制** 替换为 **200Hz 的纯力矩控制 (SATA)** 策略。

---

## 核心逻辑变更

为实现端到端的 **力矩控制 (Torque Control)**，对原框架进行了以下关键修改：

### 1. `rl_policy.py`
- 不再计算目标**关节位置**，而是加载 **SATA 策略 (`.pt` 文件)**，直接输出**关节力矩**。
- 控制频率从 **50Hz 提升至 200Hz**（SATA 训练步长为 `dt=0.005s`）。
- 新增订阅 `/ekf/velocity` 话题，用于获取**基座线速度**（SATA 观测向量的必要输入）。
- 输出的力矩发布到新话题：`/rl/target_torques`。

### 2. `low_level_ctrl.cpp`
- 状态机（站立 / 趴下）的 **PD 控制逻辑保持不变**。
- 当按下 **LB + RB** 进入 `run_policy()` 状态时，该节点变为**力矩转发器**：
  - 发送 `LowCmd`，其中：
    - `Kp = 0`
    - `Kd = 0`
    - 仅填充 `tau`（力矩命令）。

### 3. `mujoco_simulator.py`
- 修改为可接受 **纯力矩命令 (`Kp=Kd=0`)**，并直接将其应用至仿真。

### 4. `base_velocity_estimator`
- 此节点在 **Sim** 和 **Real** 模式下均为**必需**。
- SATA 策略需要线速度作为观测输入。
- 节点订阅 `/lowstate` 或 `/mujoco/lowstate`，并发布 `/ekf/velocity`。

---

## 环境配置 (与原版一致)

环境、编译与安装步骤与原仓库 **完全一致**。  
请参考原版 [README.md](https://github.com/glowing-torch/Deploy-an-RL-policy-on-the-Unitree-Go2-robot) 的  
**"Environment"** 和 **"Access Robot Sensor Data via ROS 2"** 部分。

| 组件 | 推荐版本 |
|------|-----------|
| **Ubuntu** | 20.04 (Foxy) / 22.04 (Humble) |
| **ROS 2** | Foxy / Humble |
| **MuJoCo** | 3.2.3 |
| **Python** | 3.8 (Foxy) / 3.10 (Humble) |
| **Pinocchio** | 3.4.0 |

> **确保你已正确编译 `unitree_ros2` 驱动。**

---

## 运行方式

### Sim2Sim 模式

`Sim2Sim` 模式需 **4 个终端**。  
在每个终端运行前，必须执行：

```bash
source /opt/ros/foxy/setup.bash
source /home/qiwang/unitree_ros2/cyclonedds_ws/install/setup.bash
source install/setup.bash
```

#### 终端 1 — 启动 MuJoCo 仿真器
```bash
ros2 run deploy_rl_policy mujoco_simulator.py
```

#### 终端 2 — 启动 XBox 手柄
```bash
ros2 run joy joy_node
```

#### 终端 3 — 启动状态机 (C++)
```bash
ros2 run deploy_rl_policy low_level_ctrl --ros-args -p is_simulation:=true
```

#### 终端 4 — 启动 SATA 策略节点 (Python)
```bash
ros2 run deploy_rl_policy rl_policy.py --is_simulation True
```

---

### 🚘 Sim2Real 模式运行指令

在每个终端执行以下操作：
```bash
source /opt/ros/foxy/setup.bash
source ~/unitree_ros2/install/setup.bash
source install/setup.bash
```

#### 终端 1 — 启动真实机器人驱动 (替代 MuJoCo)
```bash
ros2 launch unitree_ros2_examples go2_base.launch.py
```

#### 终端 2 — 启动 XBox 手柄
```bash
ros2 run joy joy_node
```

#### 终端 3 — 启动状态机 (C++) (Real 模式)
```bash
ros2 run deploy_rl_policy low_level_ctrl --ros-args -p is_simulation:=false
```

#### 终端 4 — 启动 EKF 速度估计器 (Real 模式)
```bash
ros2 run base_velocity_estimator ekf_velocity_estimator_node --ros-args -p is_simulation:=false
```

#### 终端 5 — 启动 SATA 策略节点 (Python) (Real 模式)
```bash
ros2 run deploy_rl_policy rl_policy.py --is_simulation False
```

---

## 🎮 手柄控制逻辑 (基本不变)

| 按键 | 功能 | 控制模式 |
|------|------|-----------|
| **B** | 站立 | PD 控制 |
| **A** | 趴下 | PD 控制 |
| **LB + RB** | 激活 SATA 力矩控制策略 | 力矩控制 |
| **松开 LB + RB** | 退出 SATA 控制，恢复 PD 站立 | 安全模式 |
| **X** | 立即切断所有电机的力矩 | 安全模式 |
> 在 SATA 模式下：
> - **左摇杆** 控制线速度  
> - **右摇杆** 控制角速度

---

## 🙏 致谢

- 基于 [**glowing-torch/Deploy-an-RL-policy-on-the-Unitree-Go2-robot**](https://github.com/glowing-torch/Deploy-an-RL-policy-on-the-Unitree-Go2-robot) 的出色工作。  
- 力矩控制策略 (**SATA**) 来源于 [**marmotlab/SATA**](https://github.com/marmotlab/SATA) 的研究成果。

---

## 📄 License

本项目仅用于学术研究与机器人控制实验，使用前请遵守 Unitree 官方硬件与软件使用规范。