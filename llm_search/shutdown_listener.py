import rclpy
from rclpy.node import Node
from std_msgs.msg import Empty
import os
import signal

class ShutdownListener(Node):
    def __init__(self):
        super().__init__('shutdown_listener')
        self.sub = self.create_subscription(Empty, '/shutdown_signal', self.callback, 10)

    def callback(self, msg):
        self.get_logger().info('Shutdown signal received. Shutting down...')
        # Send SIGINT to the whole process group
        os.kill(os.getpid(), signal.SIGINT)

def main(args=None):
    rclpy.init(args=args)
    node = ShutdownListener()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
