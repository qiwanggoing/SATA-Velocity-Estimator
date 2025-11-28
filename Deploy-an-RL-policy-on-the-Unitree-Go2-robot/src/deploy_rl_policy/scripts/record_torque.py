#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
import numpy as np
from unitree_go.msg import LowCmd
from std_msgs.msg import Float32MultiArray
import time
import csv
from threading import Lock

class DataPlotter(Node):
    def __init__(self):
        super().__init__('data_plotter')
        
        # 数据存储配置
        self.max_points = 500  # 存储500个数据点
        self.tau_history = [[] for _ in range(12)]  # 12个电机的历史数据
        self.timestamps = []  # 时间戳记录
        self.data_lock = Lock()  # 线程安全锁
        
        # CSV记录配置
        self.csv_file = open('motor_data.csv', 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['timestamp'] + [f'motor_{i}' for i in range(12)])
        
        # Matplotlib初始化
        plt.ion()  # 启用交互模式
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.lines = [self.ax.plot([], [], label=f'Motor {i}')[0] for i in range(12)]
        self.ax.set_ylim(-3, 3)
        self.ax.set_xlabel('Time (s)')
        self.ax.set_ylabel('Torque (Nm)')
        self.ax.legend()
        # self.fig.canvas.set_window_title('Real-time Motor Torque Visualization')
        
        # ROS订阅
        # self.subscription = self.create_subscription(
        #     LowCmd,
            # '/mujoco/lowcmd',
        #     self.data_callback,
        #     10)
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/pos',
            self.data_callback,
            10)
        # 定时更新图形
        self.plot_timer = self.create_timer(0.05, self.update_plot)  # 20Hz刷新
        
    def data_callback(self, msg):
        """处理收到的ROS消息"""
        timestamp = time.time()
        
        with self.data_lock:
            # 记录时间戳
            self.timestamps.append(timestamp)
            if len(self.timestamps) > self.max_points:
                self.timestamps.pop(0)
            
            # 记录电机数据
            row_data = [timestamp]
            for i in range(min(12, len(msg.data))):
                tau = msg.data[i]
                self.tau_history[i].append(tau)
                if len(self.tau_history[i]) > self.max_points:
                    self.tau_history[i].pop(0)
                row_data.append(tau)
            
            # 写入CSV
            self.csv_writer.writerow(row_data)
    
    def update_plot(self):
        """更新Matplotlib图形"""
        with self.data_lock:
            if not self.timestamps:
                return
            
            # 计算相对时间
            base_time = self.timestamps[0]
            xdata = [t - base_time for t in self.timestamps]
            
            # 更新每条曲线
            for i, line in enumerate(self.lines):
                if len(self.tau_history[i]) == len(xdata):
                    line.set_data(xdata, self.tau_history[i])
            
            # 调整X轴范围
            self.ax.set_xlim(0, xdata[-1] if xdata else 10)
            
            # 重绘图形
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
    
    def destroy_node(self):
        """清理资源"""
        self.csv_file.close()
        plt.close('all')
        super().destroy_node()

def main():
    rclpy.init()
    node = DataPlotter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()