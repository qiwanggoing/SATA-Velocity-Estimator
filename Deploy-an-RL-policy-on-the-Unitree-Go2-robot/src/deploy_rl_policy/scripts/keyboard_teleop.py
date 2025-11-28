#!/usr/bin/env python3
import sys
import select
import termios
import tty
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy

# --- 按键映射配置 ---
# 模拟 LB(4) + RB(5) 同时按下，用于激活策略
ENABLE_KEY = 'o' 

import time # Added import

# 运动控制参数
MAX_LIN_VEL = 0.4
MAX_ANG_VEL = 0.8
STOP_DELAY = 0.2 # 停车防抖时间 (秒)

# 速度档位控制 (模拟 Y 和 B 键)
SPEED_BINDINGS = {
    'i': 3, # Y键 (Index 3) -> 加速
    'k': 1, # B键 (Index 1) -> 减速
}

STOP_KEY = 'x'
SPACE_KEY = ' '
CTRL_C_KEY = '\x03'
ENABLE_KEY = 'o'

class KeyboardJoystick(Node):
    def __init__(self):
        super().__init__('keyboard_joystick_node')
        self.publisher_ = self.create_publisher(Joy, '/joy', 10)
        self.timer = self.create_timer(0.05, self.publish_joy) # 20Hz 发布
        
        # 状态变量
        self.enabled = False # 是否激活 (LB+RB)
        self.x_val = 0.0
        self.y_val = 0.0
        self.z_val = 0.0
        
        self.last_key_time = time.time() # 上次按键时间
        
        # 按键触发器 (用于模拟按一下松开)
        self.button_triggers = {1: 0, 3: 0} 

        print(self.get_instructions())
        self.settings = termios.tcgetattr(sys.stdin)

    def get_instructions(self):
        return """
        ---------- 键盘控制面板 (Sim2Sim 优化版) ----------
        
        【状态控制】
        o : 切换 激活/待机 (对应手柄 LB + RB)
        x / 空格 : 立即停止
        
        【移动控制 (按住不放)】
        w/s : 前进 / 后退 (松开 Q/E 可恢复直行)
        a/d : 左移 / 右移
        q/e : 转向 (可在按住 W/S 时叠加转向)
        
        【说明】
        - 按住 W + Q : 边走边转
        - 松开 Q (保持 W) : 恢复直行
        - 全部松开 : 停止
        
        CTRL-C : 退出
        ------------------------------------------
        """

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def publish_joy(self):
        # 1. 读取按键
        key = self.get_key()
        
        if key == CTRL_C_KEY:
            self.destroy_node()
            rclpy.shutdown()
            sys.exit()
            
        # 2. 处理按键逻辑
        if key == ENABLE_KEY:
            self.enabled = not self.enabled
            status = "激活 (Active)" if self.enabled else "待机 (Standby)"
            print(f"模式切换: {status}")
            
        elif key == STOP_KEY or key == SPACE_KEY:
            self.x_val = 0.0
            self.y_val = 0.0
            self.z_val = 0.0
            print("停止指令!")

        # --- 移动逻辑优化 ---
        elif key in ['w', 's']:
            self.last_key_time = time.time() # 更新活跃时间
            self.x_val = MAX_LIN_VEL if key == 'w' else -MAX_LIN_VEL
            self.y_val = 0.0
            self.z_val = 0.0
            
        elif key in ['a', 'd']:
            self.last_key_time = time.time() # 更新活跃时间
            self.y_val = MAX_LIN_VEL if key == 'a' else -MAX_LIN_VEL
            self.x_val = 0.0
            self.z_val = 0.0
            
        elif key in ['q', 'e']:
            self.last_key_time = time.time() # 更新活跃时间
            self.z_val = MAX_ANG_VEL if key == 'q' else -MAX_ANG_VEL
            # 保留 X/Y
            
        elif key in SPEED_BINDINGS:
            self.last_key_time = time.time()
            idx = SPEED_BINDINGS[key]
            self.button_triggers[idx] = 1 
            print(f"触发功能键: {key}")
            
        else:
            # 无按键 (松开)：带延迟的归零
            if key == '': 
                if (time.time() - self.last_key_time) > STOP_DELAY:
                    self.x_val = 0.0
                    self.y_val = 0.0
                    self.z_val = 0.0
                # 否则保持最后的速度 (防抖)


        
        # 3. 构建 Joy 消息 (伪装成 Xbox 手柄)
        msg = Joy()
        msg.header.stamp = self.get_clock().now().to_msg()
        
        # Axes (至少 6 个)
        # [0: LeftStick X, 1: LeftStick Y, 2: LT, 3: RightStick X, 4: RightStick Y, 5: RT]
        # xbox_command.py 读取: axes[1] (Y), axes[0] (X), axes[3] (Yaw)
        msg.axes = [0.0] * 6
        msg.axes[1] = self.x_val  # 前后
        msg.axes[0] = self.y_val  # 左右
        msg.axes[3] = self.z_val  # 转向
        
        # Buttons (至少 6 个)
        # [0: A, 1: B, 2: X, 3: Y, 4: LB, 5: RB]
        msg.buttons = [0] * 12
        
        # 激活状态 (LB + RB)
        if self.enabled:
            msg.buttons[4] = 1 # LB
            msg.buttons[5] = 1 # RB
            
        # 瞬时按钮 (速度调节)
        for idx, val in self.button_triggers.items():
            msg.buttons[idx] = val
        
        # 发布消息
        self.publisher_.publish(msg)
        
        # 重置瞬时按钮 (只触发一帧)
        for k in self.button_triggers:
            self.button_triggers[k] = 0

def main():
    rclpy.init()
    node = KeyboardJoystick()
    try:
        while rclpy.ok():
            node.publish_joy() # 主动在循环里调用
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()