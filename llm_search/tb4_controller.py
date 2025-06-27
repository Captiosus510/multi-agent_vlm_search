import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class TurtleBotController(Node):
    def __init__(self):
        super().__init__('turtlebot_controller')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info('TurtleBot Controller has been started. Publishing to /cmd_vel.')

    def timer_callback(self):
        msg = Twist()
        # To make the robot move forward
        msg.linear.x = 0.1
        # To make the robot turn
        msg.angular.z = 0.7
        
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: linear.x={msg.linear.x}, angular.z={msg.angular.z}')

def main(args=None):
    rclpy.init(args=args)
    turtlebot_controller = TurtleBotController()
    rclpy.spin(turtlebot_controller)
    turtlebot_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()