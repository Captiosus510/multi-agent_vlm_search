import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class TurtleBotController(Node):
    def __init__(self):
        super().__init__('turtlebot_controller')
        self.publisher_my_robot_ = self.create_publisher(Twist, 'my_robot/cmd_vel', 10)
        self.publisher_other_robot_ = self.create_publisher(Twist, 'other_robot/cmd_vel', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info('TurtleBot Controller has been started. Publishing to /my_robot/cmd_vel.')

    def timer_callback(self):
        msg_1 = Twist()
        # To make the robot move forward
        msg_1.linear.x = 0.1
        # To make the robot turn
        msg_1.angular.z = -0.7

        self.publisher_my_robot_.publish(msg_1)
        self.get_logger().info(f'Publishing: linear.x={msg_1.linear.x}, angular.z={msg_1.angular.z}')

        msg_2 = Twist()
        # To make the other robot move forward
        msg_2.linear.x = 0.1
        # To make the other robot turn
        msg_2.angular.z = 0.7

        self.publisher_other_robot_.publish(msg_2)
        self.get_logger().info(f'Publishing: linear.x={msg_2.linear.x}, angular.z={msg_2.angular.z}')

def main(args=None):
    rclpy.init(args=args)
    turtlebot_controller = TurtleBotController()
    rclpy.spin(turtlebot_controller)
    turtlebot_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()