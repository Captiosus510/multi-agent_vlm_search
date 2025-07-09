import rclpy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu
# from tf_transformations import quaternion_from_euler

HALF_DISTANCE_BETWEEN_WHEELS = 0.045
WHEEL_RADIUS = 0.025

class MyRobotDriver:
    def init(self, webots_node, properties):
        self.__robot = webots_node.robot
        self.__robot_name = self.__robot.getName()

        self.__left_motor = self.__robot.getDevice('left_wheel_joint')
        self.__right_motor = self.__robot.getDevice('right_wheel_joint')

        self.__left_motor.setPosition(float('inf'))
        self.__left_motor.setVelocity(0)

        self.__right_motor.setPosition(float('inf'))
        self.__right_motor.setVelocity(0)

        self.__inertial_unit = self.__robot.getDevice('p3d inertial')
        self.__inertial_unit.enable(32)

        self.__target_twist = Twist()
        if not rclpy.ok():
            rclpy.init(args=None)
        self.__node = rclpy.create_node(f'{self.__robot_name}_driver')
        topic = f'{self.__robot_name}/cmd_vel'
        self.__node.create_subscription(Twist, topic, self.__cmd_vel_callback, 1)
        self.__imu_publisher = self.__node.create_publisher(Imu, f'{self.__robot_name}/imu', 32)

    def __cmd_vel_callback(self, twist):
        self.__target_twist = twist

    def step(self):
        rclpy.spin_once(self.__node, timeout_sec=0)
    
        forward_speed = self.__target_twist.linear.x
        angular_speed = self.__target_twist.angular.z

        command_motor_left = (forward_speed - angular_speed * HALF_DISTANCE_BETWEEN_WHEELS) / WHEEL_RADIUS
        command_motor_right = (forward_speed + angular_speed * HALF_DISTANCE_BETWEEN_WHEELS) / WHEEL_RADIUS

        self.__left_motor.setVelocity(command_motor_left)
        self.__right_motor.setVelocity(command_motor_right)

        imu_msg = Imu()
        imu_msg.header.stamp = self.__node.get_clock().now().to_msg()
        imu_msg.header.frame_id = f'{self.__robot_name}/imu'

        rpy = self.__inertial_unit.getRollPitchYaw()
        imu_msg.orientation.x = rpy[0]
        imu_msg.orientation.y = rpy[1]
        imu_msg.orientation.z = rpy[2]
        imu_msg.orientation.w = 1.0  # Assuming no rotation around the w-axis
        # # convert roll, pitch, yaw to quaternion
        # quaternion = quaternion_from_euler(rpy[0], rpy[1], rpy[2])

        # imu_msg.orientation.x = quaternion[0]
        # imu_msg.orientation.y = quaternion[1]
        # imu_msg.orientation.z = quaternion[2]
        # imu_msg.orientation.w = quaternion[3]

        self.__imu_publisher.publish(imu_msg)