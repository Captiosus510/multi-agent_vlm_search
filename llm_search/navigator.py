import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Imu
import math

class Navigator(Node):
    def __init__(self):
        super().__init__('navigator')
        self.declare_parameter('robot_name', 'my_robot')
        self.declare_parameter('robot_speed', 0.2)
        self.declare_parameter('robot_turn_speed', 0.2)
        self.declare_parameter('behavior', 'default')

        self.robot_name = self.get_parameter('robot_name').get_parameter_value().string_value
        self.robot_speed = self.get_parameter('robot_speed').get_parameter_value().double_value
        self.robot_turn_speed = self.get_parameter('robot_turn_speed').get_parameter_value().double_value
        self.behavior = self.get_parameter('behavior').get_parameter_value().string_value

        self.vel_publisher = self.create_publisher(Twist, f'/{self.robot_name}/cmd_vel', 10)

        self.path_subscription = self.create_subscription(
            Path, f'/{self.robot_name}/path', self.path_callback, 10
        )
        
        self.gps_subscription = self.create_subscription(
            PointStamped, f'/{self.robot_name}/p3d_gps', self.gps_callback, 10
        )
        
        self.imu_subscription = self.create_subscription(
            Imu, f'/{self.robot_name}/imu', self.imu_callback, 10
        )

        # Path following state
        self.path = []
        self.current_waypoint_index = 0
        self.waypoint_tolerance = 0.3  # Distance tolerance for reaching waypoint
        self.angle_tolerance = 0.1  # Angle tolerance in radians (~5.7 degrees)
        self.following_path = False
        
        # Robot state
        self.current_position = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.current_yaw = 0.0  # Current robot heading in radians
        
        # Control parameters
        self.max_linear_speed = self.robot_speed
        self.max_angular_speed = self.robot_turn_speed
        self.linear_kp = 1.0  # Proportional gain for linear velocity
        self.angular_kp = 2.0  # Proportional gain for angular velocity
        
        # Smoothing parameters
        self.prev_linear_vel = 0.0
        self.prev_angular_vel = 0.0
        self.vel_smoothing_factor = 0.8  # Higher = smoother but slower response

        timer_period = 0.1  # 10 Hz for smooth control
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info(f'Navigator started for {self.robot_name}')

    def gps_callback(self, msg):
        """Update robot's current position from GPS"""
        self.current_position = {
            'x': msg.point.x,
            'y': msg.point.y,
            'z': msg.point.z
        }

    def imu_callback(self, msg):
        """Update robot's current orientation from IMU"""
        self.current_yaw = self.quaternion_to_yaw(msg.orientation)

    def quaternion_to_yaw(self, q):
        """Convert quaternion to yaw angle in radians"""
        return math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

    def normalize_angle(self, angle):
        """Normalize angle to [-pi, pi]"""
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def timer_callback(self):
        """Main control loop"""
        msg = Twist()

        if self.following_path and len(self.path) > 0:
            # Path following mode
            msg = self.follow_path_smooth()
        else:
            # Default behavior modes
            if self.behavior == 'monitor':
                # msg.linear.x = self.robot_speed * 0.5  # Slower for monitoring
                msg.linear.x = 0.0
                msg.angular.z = self.robot_turn_speed * 0.3
            elif self.behavior == 'search':
                msg.linear.x = self.robot_speed * 0.7
                msg.angular.z = self.robot_turn_speed * 0.5
            else:
                msg.linear.x = 0.0
                msg.angular.z = 0.0

        self.vel_publisher.publish(msg)

    def path_callback(self, msg):
        """Handle received path message"""
        self.get_logger().info(f'Received new path with {len(msg.poses)} waypoints.')
        
        if len(msg.poses) > 0:
            self.path = msg.poses
            self.current_waypoint_index = 0
            self.following_path = True
            
            # Reset velocity for smooth start
            self.prev_linear_vel = 0.0
            self.prev_angular_vel = 0.0
            
            self.get_logger().info(f"Starting path navigation to {len(self.path)} waypoints")
        else:
            self.get_logger().warn("Received empty path")

    def follow_path_smooth(self):
        """Follow the path with smooth control using IMU orientation"""
        msg = Twist()
        
        if self.current_waypoint_index >= len(self.path):
            # Path completed - smooth stop
            target_linear = 0.0
            target_angular = 0.0
            self.following_path = False
            self.get_logger().info("Path completed!")
        else:
            # Get current target waypoint
            current_waypoint = self.path[self.current_waypoint_index]
            target_x = current_waypoint.pose.position.x
            target_y = current_waypoint.pose.position.y
            
            # Calculate distance and angle to target
            dx = target_x - self.current_position['x']
            dy = target_y - self.current_position['y']
            distance_to_target = math.sqrt(dx*dx + dy*dy)
            
            # Calculate desired heading to target
            desired_yaw = math.atan2(dy, dx)
            
            # Calculate angle error (shortest path)
            angle_error = self.normalize_angle(desired_yaw - self.current_yaw)
            
            self.get_logger().debug(f"Waypoint {self.current_waypoint_index}: "
                                  f"distance={distance_to_target:.2f}m, "
                                  f"angle_error={math.degrees(angle_error):.1f}°")
            
            # Check if we've reached the current waypoint
            if distance_to_target < self.waypoint_tolerance:
                self.get_logger().info(f"Reached waypoint {self.current_waypoint_index}")
                self.current_waypoint_index += 1
                
                if self.current_waypoint_index >= len(self.path):
                    target_linear = 0.0
                    target_angular = 0.0
                    self.following_path = False
                    self.get_logger().info("Final destination reached!")
                else:
                    # Continue to next waypoint - recalculate for smooth transition
                    return self.follow_path_smooth()
            else:
                # Calculate control commands
                
                # Angular velocity - proportional to angle error
                target_angular = self.angular_kp * angle_error
                
                # Limit angular velocity
                target_angular = max(-self.max_angular_speed, 
                                   min(self.max_angular_speed, target_angular))
                
                # Linear velocity - reduce when turning
                angle_factor = max(0.1, 1.0 - abs(angle_error) / math.pi)  # Slow down when turning
                distance_factor = min(1.0, distance_to_target / 1.0)  # Slow down when close
                
                target_linear = self.linear_kp * self.max_linear_speed * angle_factor * distance_factor
                
                # Only move forward if roughly facing the target
                if abs(angle_error) > math.pi/2:  # If facing away, just turn
                    target_linear = 0.0  # stop forward motion completely
        
        # Apply velocity smoothing to avoid jerky motion
        smooth_linear = (self.vel_smoothing_factor * self.prev_linear_vel + 
                        (1 - self.vel_smoothing_factor) * target_linear)
        smooth_angular = (self.vel_smoothing_factor * self.prev_angular_vel + 
                         (1 - self.vel_smoothing_factor) * target_angular)
        
        # Store for next iteration
        self.prev_linear_vel = smooth_linear
        self.prev_angular_vel = smooth_angular
        
        # Set message
        msg.linear.x = smooth_linear
        msg.angular.z = smooth_angular
        
        return msg

def main(args=None):
    rclpy.init(args=args)
    navigator = Navigator()
    rclpy.spin(navigator)
    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()