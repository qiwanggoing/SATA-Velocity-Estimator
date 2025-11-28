import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy
import time
class XboxController:
    def __init__(self, node: Node):
        self.node = node
        self.axes = []
        self.buttons = []
        self._sub = node.create_subscription(
            Joy,
            '/joy',
            self._joy_callback,
            10
        )
        self._last_msg_time = 0
        self._timeout = 0.1

    def _joy_callback(self, msg: Joy):
        self.axes = msg.axes
        self.buttons = msg.buttons
        self._last_msg_time=time.time()

    def get_left_stick(self):
        return (self.axes[1], self.axes[0]) if len(self.axes) >= 2 else (0.0, 0.0)

    def get_right_stick(self):
        return self.axes[3] if len(self.axes) >= 5 else 0

    def is_pressed(self, index1=4,index2=5):
        # elapsed_time=time.time() - self._last_msg_time
        # if  elapsed_time> self._timeout:
        #     print(elapsed_time)
        #     return 0,0
        if len(self.buttons) >= 6:
            return self.buttons[index1] == 1, self.buttons[index2] == 1
        return 0, 0
