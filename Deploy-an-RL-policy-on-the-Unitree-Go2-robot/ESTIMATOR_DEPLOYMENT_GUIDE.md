# 独立部署速度估计器 (LSTM) 完全指南

本文档提供将我们训练的 LSTM 速度估计器集成到**任何新项目**（无论是 sim2sim 还是 sim2real）所需的所有信息和代码，旨在实现完全独立、万无一失的部署。

---

## 核心原理：解耦

为了实现独立部署，我们需要解除对 `rsl_rl` 训练库的依赖。这需要两样东西：
1. **模型结构定义**：即 `VelocityEstimator` 类的 Python 代码。
2. **模型权重**：你已经训练好的 `.pt` 文件。

本指南将提供前者，并告诉你如何将两者结合使用。

---

## 步骤一：创建 `estimator.py` 模型定义文件

首先，在你的新部署项目中（例如，在 `.../deploy_rl_policy/scripts/` 目录下），创建一个名为 `estimator.py` 的新文件。将以下**完整代码**复制并粘贴到该文件中。

```python
# 文件名: estimator.py
import torch
import torch.nn as nn

class VelocityEstimator(nn.Module):
    """
    一个使用 LSTM 的速度估计器网络。
    它接收一个包含11个时间步的历史传感器数据序列，并输出当前时刻的基座线速度。
    """
    def __init__(self, input_dim=33, hidden_dim=128, output_dim=3, num_layers=2):
        """
        初始化模型结构。
        Args:
            input_dim (int): 每个时间步的输入特征维度 (固定为33)。
            hidden_dim (int): LSTM 隐藏层的维度。
            output_dim (int): 输出维度 (固定为3，即 vx, vy, vz)。
            num_layers (int): LSTM 的层数。
        """
        super(VelocityEstimator, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM 层，batch_first=True 表示输入的维度顺序为 (batch, sequence, feature)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
        # 用于回归速度的 MLP (多层感知机) 头
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        """
        模型的前向传播。
        Args:
            x (torch.Tensor): 输入张量，形状必须为 (batch_size, 11, 33)。
        Returns:
            torch.Tensor: 估计的线速度，形状为 (batch_size, 3)。
        """
        # 初始化 LSTM 的隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # LSTM 的输出包含所有时间步的输出，我们只关心最后一个
        lstm_out, _ = self.lstm(x, (h0, c0))
        
        # 取序列中最后一个时间步的输出
        last_time_step_out = lstm_out[:, -1, :]
        
        # 将其输入到 MLP 头中以获得最终的速度估计
        velocity_estimation = self.mlp(last_time_step_out)
        
        return velocity_estimation
```

---

## 步骤二：模型接口与数据准备（关键步骤）

这是集成过程中最重要、最需要注意细节的部分。你必须在你的代码中**严格按照顺序**准备一个 `(1, 11, 33)` 形状的输入张量。

#### **33维特征向量的精确定义**

在**每个控制周期**，你需要收集以下数据，并**严格按照表格顺序**拼接成一个33维的向量：

| 顺序 | 特征 | 维度 | 单位 | 描述与来源 | **注意事项** |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1-12** | `dof_pos` | 12 | rad | **关节角度** | 来自电机编码器。**顺序必须与训练时完全一致** (FL, FR, RL, RR) |
| **13-24** | `dof_vel` | 12 | rad/s | **关节速度** | 来自电机编码器。顺序同上。 |
| **25-27**| `lin_acc` | 3 | m/s² | **基座线加速度** | 来自IMU。**必须是机身坐标系 (Body Frame)**。 |
| **28-30**| `ang_vel` | 3 | rad/s | **基座角速度** | 来自IMU。**必须是机身坐标系 (Body Frame)**。 |
| **31-33**|`gravity_vec`| 3 | - | **机身坐标系下的重力向量** | 来自IMU姿态解算（例如将世界系重力向量`[0,0,-g]`通过机身姿态的逆旋转矩阵转换得到）。|

**警告：以上顺序、单位、坐标系缺一不可，任何一项不匹配都会导致估计器输出错误！**

---

## 步骤三：在部署代码中集成（示例）

以下是一个完整的、可操作的 Python 伪代码示例，展示了如何加载模型并在实时循环中使用它。

```python
import torch
import collections
import numpy as np
import time

# 步骤一：从你刚刚创建的文件中导入模型类
from estimator import VelocityEstimator 

# --- 初始化阶段 ---

# 你的模型权重文件路径
ESTIMATOR_WEIGHTS_PATH = "/path/to/your/velocity_estimator_....pt" 

# 1. 创建模型实例
estimator_model = VelocityEstimator(input_dim=33)

# 2. 加载训练好的权重
#    如果在没有GPU的设备上运行，请使用 map_location='cpu'
estimator_model.load_state_dict(torch.load(ESTIMATOR_WEIGHTS_PATH, map_location='cpu'))

# 3. 切换到评估模式（非常重要！）
estimator_model.eval()

# 4. 初始化一个长度为11的历史数据队列
history_buffer = collections.deque(maxlen=11)

print("速度估计器已加载并准备就绪。")


# --- 实时控制循环 ---

while True:
    # a. 从机器人硬件或模拟器获取当前传感器数据
    #    (确保数据格式和单位符合步骤二中的要求)
    current_dof_pos = get_joint_positions() # shape (12,)
    current_dof_vel = get_joint_velocities() # shape (12,)
    current_lin_acc = get_imu_acceleration() # shape (3,)
    current_ang_vel = get_imu_gyro()         # shape (3,)
    current_gravity_vec = get_gravity_vector() # shape (3,)

    # b. 严格按顺序拼接成当前帧的特征向量
    current_features = np.concatenate([
        current_dof_pos, current_dof_vel, current_lin_acc,
        current_ang_vel, current_gravity_vec
    ])

    # c. 更新历史队列
    history_buffer.append(current_features)

    # d. 检查队列是否已满（即是否收集了足够长的历史数据）
    if len(history_buffer) == 11:
        
        # e. 将队列数据整理成模型所需的张量
        #    从deque -> numpy array -> torch tensor
        input_sequence = np.stack(list(history_buffer), axis=0)
        
        #    增加 batch 维度 (1, 11, 33) 并转换为 float32
        input_tensor = torch.from_numpy(input_sequence).unsqueeze(0).float()
        
        # f. 运行模型推理
        with torch.no_grad():
            estimated_velocity = estimator_model(input_tensor).squeeze().numpy() # shape (3,)

        # g. 使用估计出的速度
        #    现在，你可以用 estimated_velocity 来构建你的RL策略所需的观测向量了！
        #    ... 在这里构建你的RL观测向量 ...
        #    ... 调用你的RL策略 ...
        
        # print(f"当前估计速度: {estimated_velocity}")

    # 等待下一个控制周期
    time.sleep(0.02) # 假设控制频率为 50Hz
```

---

## 步骤四：ROS 2 集成注意事项

如果你使用的是像 `deploy_rl_policy` 这样的 `ament_cmake` 包，请确保将新的 `estimator.py` 文件也添加到 `CMakeLists.txt` 的安装列表中，否则 `ros2 run` 会找不到它。

**示例 `CMakeLists.txt` 修改：**
```cmake
install(PROGRAMS 
  scripts/rl_policy.py
  scripts/sata_config.py
  scripts/estimator.py  # <-- 把这个新文件加进来
  # ... 其他脚本 ...
  DESTINATION lib/${PROJECT_NAME}
)
```
修改后，需要**清理并重新编译**你的ROS 2工作区。

这份指南提供了从代码定义到实时集成的所有必要信息，希望能帮助你万无一失地在任何项目中使用这个估计器。
