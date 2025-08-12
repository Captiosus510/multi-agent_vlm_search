import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class Navigator(Node):
    def __init__(self):
        super().__init__('navigator')
        self.declare_parameter('robot_name', 'my_robot')
        self.declare_parameter('robot_speed', 0.0)
        self.declare_parameter('robot_turn_speed', 0.0)
        self.declare_parameter('behavior', 'default')

        self.robot_name = self.get_parameter('robot_name').get_parameter_value().string_value
        self.robot_speed = self.get_parameter('robot_speed').get_parameter_value().double_value
        self.robot_turn_speed = self.get_parameter('robot_turn_speed').get_parameter_value().double_value
        self.behavior = self.get_parameter('behavior').get_parameter_value().string_value

        # self.goal_position = self.create_subscription(Twist, f'/{self.robot_name}/goal_position', self.goal_callback, 10)
        self.publisher_ = self.create_publisher(Twist, f'/{self.robot_name}/cmd_vel', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info(f'TurtleBot Controller has been started. Publishing to /{self.robot_name}/cmd_vel.')

    def timer_callback(self):
        if self.behavior == 'monitor':
            # Monitor behavior: move and turn at a constant speed
            msg = Twist()
            msg.linear.x = self.robot_speed
            msg.angular.z = self.robot_turn_speed
        elif self.behavior == 'search':
            # implemeent later
            msg = Twist()
            msg.linear.x = self.robot_speed
            msg.angular.z = self.robot_turn_speed
        else:
            # Default behavior: stop the robot
            msg = Twist()
            msg.linear.x = 0.0
            msg.angular.z = 0.0

        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: linear.x={msg.linear.x}, angular.z={msg.angular.z}')

def main(args=None):
    rclpy.init(args=args)
    navigator = Navigator()
    rclpy.spin(navigator)
    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()