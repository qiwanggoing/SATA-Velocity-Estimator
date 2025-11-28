from xbox_command import XboxController
from rclpy.node import Node
import rclpy
class CommandNode(Node):
    def __init__(self):
        super().__init__("COMMAND_NODE")
        self.joy = XboxController(self)  # 正确传入已初始化的 Node
        self.timer = self.create_timer(0.1, self.loop)

    def loop(self):
        lx, ly = self.joy.get_left_stick()
        a = self.joy.is_pressed(4)
        self.get_logger().info(f"LX: {lx:.2f}, LY: {ly:.2f}, A pressed: {a}")

def main():
    rclpy.init()
    node = CommandNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()