# 这是你新的 rl_policy.py (一个简单的非ROS2版本)
import torch
import numpy as np
import os

class RLPolicy:
    def __init__(self, policy_path):
        """
        加载 TorchScript 策略模型。
        对应 mujoco_simulator.py
        """
        if not os.path.exists(policy_path):
            raise FileNotFoundError(f"错误：在 {policy_path} 找不到策略文件")
            
        print(f"正在从 {policy_path} 加载策略...")
        try:
            self.policy = torch.jit.load(policy_path)
            self.policy.eval() # 设置为评估模式
            print("策略加载成功。")
        except Exception as e:
            print(f"加载策略失败: {e}")
            raise

    def get_action(self, observation):
        """
        根据观测值计算动作。
        对应 mujoco_simulator.py
        """
        # 1. 将 NumPy 观测值转换为 Torch Tensor
        obs_tensor = torch.from_numpy(observation).unsqueeze(0) # 添加 batch 维度
        
        # 2. 执行策略推理
        with torch.no_grad(): # 关闭梯度计算
            action_tensor = self.policy(obs_tensor)
            
        # 3. 将 Torch Tensor 转换回 NumPy 数组
        raw_action = action_tensor.detach().numpy().squeeze() # 移除 batch 维度
        
        return raw_action