# 最终版：LSTM速度估计器高可靠性部署指南 V2.0

本文档是在真实部署调试后，对 `ESTIMATOR_DEPLOYMENT_GUIDE.md` 的一次重大更新与最终总结。它包含了在实践中遇到的、未在原始文档中明确提及但至关重要的细节，旨在实现**真正万无一失**的部署。

---

## 核心原则：消除领域鸿沟 (Domain Gap) 与在线校准

部署失败的根源在于，部署环境（Sim）的数据与训练环境的数据存在多个“领域鸿沟”。同时，模型本身也存在固有的“零点漂移”。最终的解决方案是“**精确复现预处理**” + “**在线零点校准**”。

---

## 第一步：项目结构（与V1.0一致）

1.  **`estimator.py`**: 确保 `VelocityEstimator` 类的Python代码完整地保存在 `src/deploy_rl_policy/scripts/estimator.py` 文件中。
2.  **`CMakeLists.txt`**: 确保 `scripts/estimator.py` 已被添加到 `install` 列表中。
3.  **模型权重**: 将训练好的 `.pt` 权重文件存放在固定位置，如 `resources/policies/`。

---

## 第二步：输入数据的精确预处理 (最关键！)

这是整个流程中最重要的部分。在你的 `rl_policy.py` 脚本中，你必须**严格**按照以下方式准备33维输入向量。

#### **33维特征向量的最终正确实现**

| 顺序 | 特征 | 维度 | **最终实现要点** |
| :--- | :--- | :--- | :--- |
| 1-12 | `dof_pos` | 12 | **关节顺序映射**：必须将从机器人/仿真器读出的关节顺序，通过索引映射，转换成模型训练时使用的 `(FL, FR, RL, RR)` 顺序。|
| 13-24| `dof_vel` | 12 | 同上，关节顺序必须正确。 |
| 25-27| `lin_acc` | 3 | **移除重力**：IMU加速度计的原始读数包含重力。必须用以下公式计算纯机体加速度：`pure_lin_acc = raw_imu_acc + gravity_vector_from_quat`。(注意是**加号!**) |
| 28-30| `ang_vel` | 3 | 直接使用IMU陀螺仪读数 (body frame)。 |
| 31-33|`gravity_vec`| 3 | **单位向量**：必须传入由IMU四元数解算出的、在机身坐标系下的**单位重力向量**（长度为1）。**绝对不能**乘以 `g` (9.81)。|

#### **代码示例 (在`rl_policy.py`的`run`方法中)**
```python
# ... 获取 quat, ang_vel, qj, dqj ...

# 关节顺序映射 (robot -> policy)
qj_policy = self.qj[robot_to_policy_map]
dqj_policy = self.dqj[robot_to_policy_map]

# 1. 计算单位重力向量 (用于第31-33维输入)
gravity_orientation = self.get_gravity_orientation(quat) # 这是一个单位向量

# 2. 计算纯线加速度 (用于第25-27维输入)
lin_acc_raw = np.array(self.low_state.imu_state.accelerometer, dtype=np.float32)
pure_lin_acc = lin_acc_raw + (gravity_orientation * 9.81)

# 3. 严格按顺序拼接
current_features = np.concatenate([
    qj_policy,
    dqj_policy,
    pure_lin_acc,
    ang_vel,
    gravity_orientation # <-- 注意：这里是单位向量
]).astype(np.float32)
```

---

## 第三步：在线零点校准 (最终解决方案)

这是解决模型固有偏置（Bias）、确保机器人静止的关键。

**原理**: 利用一个简单、可靠的**PD控制器**让机器人先稳定站立。在这个绝对静止的窗口期，测量估计器的输出，得到其“零点漂移”的精确值，然后在后续运行中持续减去这个偏置。

**实现**: 在 `rl_policy.py` 中实现一个有限状态机（FSM）。

#### 1. 在 `__init__` 中初始化

```python
# FSM 状态
self.fsm_state = "PD_STAND" 

# 校准相关变量
self.is_calibrated = False
self.estimator_bias = np.zeros(3, dtype=np.float32)
self.calibration_samples = []
self.CALIBRATION_STEPS = 400  # 2秒 (200Hz)，确保机器人能完全站稳

# PD站立控制器参数 (从 low_level_ctrl.cpp 提取)
self.kp_stand = np.full(12, 50.0, dtype=np.float32)
self.kd_stand = np.full(12, 1.0, dtype=np.float32)
self.target_q_stand = np.array([
    -0.1, 0.8, -1.5,  # FR
     0.1, 0.8, -1.5,  # FL
    -0.1, 1.0, -1.5,  # RR
     0.1, 1.0, -1.5   # RL
], dtype=np.float32)
```

#### 2. 在 `dataReciever` 类中添加PD力矩计算辅助函数

```python
def _compute_pd_torques(self, target_q, current_q, current_dq):
    # 注意: 输入的 current_q, current_dq 必须是机器人原始顺序 (FR, FL, RR, RL)
    torques = (target_q - current_q) * self.kp_stand - current_dq * self.kd_stand
    return torques.astype(np.float32)
```

#### 3. 在 `run()` 方法中实现状态机逻辑

```python
def run(self):
    # ... (读取 qj, dqj, quat, ang_vel 等) ...

    # --- 状态机核心逻辑 ---
    if self.fsm_state == "PD_STAND":
        self.get_logger().info(f"FSM State: PD_STAND, Calibrating... ({len(self.calibration_samples)}/{self.CALIBRATION_STEPS})")
        
        # 1. 计算并发布PD力矩让机器人站立
        final_torques_robot_order = self._compute_pd_torques(self.target_q_stand, self.qj, self.dqj)
        
        # 2. 运行估计器并收集样本
        # ... (准备 current_features 的逻辑如第二步所示) ...
        self.estimator_history_buffer.append(current_features)
        if len(self.estimator_history_buffer) == 11:
            # ... (运行模型得到 raw_estimated_velocity) ...
            self.calibration_samples.append(raw_estimated_velocity)
        
        # 3. 检查是否完成校准
        if len(self.calibration_samples) >= self.CALIBRATION_STEPS:
            self.estimator_bias = np.mean(self.calibration_samples, axis=0)
            self.is_calibrated = True
            self.fsm_state = "RUNNING"
            self.get_logger().info(f"在线校准完成，测得偏置: {self.estimator_bias}")
            
    elif self.fsm_state == "RUNNING":
        # 1. 运行估计器并校准
        # ... (准备 current_features, 运行模型得到 raw_estimated_velocity) ...
        calibrated_velocity = raw_estimated_velocity - self.estimator_bias
        self.base_lin_vel = calibrated_velocity # 将校准后的速度用于后续步骤

        # 2. 运行RL策略
        # ... (构建观测向量 self.obs, 注意使用 self.base_lin_vel) ...
        # ... (调用 self.policy.get_action(self.obs) 得到 raw_action) ...
        # ... (调用 self._compute_sata_torques(...) 得到 final_torques_policy_order) ...
        final_torques_robot_order = final_torques_policy_order[policy_to_robot_map]

    # ... (发布最终的 final_torques_robot_order) ...
```

---

## 第四步 (可选): 性能调优

如果机器人在校准后虽然能站立，但动态性能不佳或有微小抖动，可以对**校准后的速度**应用一个低通滤波器。

#### 在 `RUNNING` 状态中加入EMA平滑

```python
# ... 得到 calibrated_velocity ...

# 对校准后的速度进行平滑
alpha = 0.5 # 可调平滑系数, 0-1
self.smoothed_velocity = alpha * calibrated_velocity + (1 - alpha) * self.smoothed_velocity
self.base_lin_vel = self.smoothed_velocity # 使用平滑后的速度
```
通过调整 `alpha` 值，可以在响应速度和动作平滑度之间找到最佳平衡。

---
遵循以上经过实践验证的步骤，即可实现对该LSTM速度估计器的高可靠性、高性能部署。
