#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: navigator.py
Author: Mahd Afzal
Date: 2025-08-19
Version: 1.0
Description:
    Robot motion controller. Subscribes to various sensor topics for navigation (GPS, IMU, Lidar).
    Implements basic obstacle avoidance and smooth path following behaviors based on waypoints. 


    TODO: Implement path smoothing algorithm to prevent unnecessary turning. (Ex. Funnel algorithm)
    Note: quaternion_to_yaw is not used because the robot_driver currently publishes RPY data NOT quaternions.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import Imu, LaserScan
import math
import numpy as np

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

        self.lidar_subscription = self.create_subscription(
            LaserScan, f'/{self.robot_name}/rplidar', self.lidar_callback, 10
        )

        # Store obstacle info
        self.min_obstacle_distance = float('inf')
        self.safe_distance = 0.5  # meters
        self.lidar_ranges = None
        self.lidar_angle_increment = None
        self.lidar_angle_min = None

        # Path following state
        self.path = []
        self.current_waypoint_index = 0
        self.waypoint_tolerance = 0.35  # Distance tolerance for reaching waypoint
        self.angle_tolerance = 0.1  # Angle tolerance in radians (~5.7 degrees)
        self.following_path = False
        
        # Robot state
        self.current_position = {'x': 0.0, 'y': 0.0, 'z': 0.0}
        self.current_yaw = 0.0  # Current robot heading in radians
        
        # Control parameters
        self.max_linear_speed = self.robot_speed
        self.max_angular_speed = self.robot_turn_speed
        self.linear_kp = 1.5  # Proportional gain for linear velocity
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
        self.current_yaw = msg.orientation.z  # Assuming z is the yaw angle in radians

    def lidar_callback(self, msg: LaserScan):
        # Convert LaserScan ranges into numpy array for easier processing
        ranges = np.array(msg.ranges)

        # Filter invalid values (inf, NaN)
        self.lidar_ranges = ranges[np.isfinite(ranges)]
        self.lidar_angle_increment = msg.angle_increment
        self.lidar_angle_min = msg.angle_min

        if len(self.lidar_ranges) > 0:
            # Focus on forward-facing 60° sector
            angle_range = 30  # degrees
            center_index = len(ranges) // 2
            window = int(angle_range / (180 / len(ranges)))  # convert to indices
            front_ranges = ranges[center_index - window : center_index + window]

            front_valid = front_ranges[np.isfinite(front_ranges)]
            if len(front_valid) > 0:
                self.min_obstacle_distance = np.min(front_valid)
            else:
                self.min_obstacle_distance = float('inf')
        else:
            self.min_obstacle_distance = float('inf')

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
                msg.angular.z = self.robot_turn_speed
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
        """Follow the path with smooth control using IMU orientation - anti-overshoot version"""
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
            
            # Desired heading from path
            desired_yaw = math.atan2(dy, dx)

            # Avoidance heading
            avoid_heading = self.compute_avoidance_heading(
                self.lidar_ranges, self.lidar_angle_increment, self.lidar_angle_min
            )

            if avoid_heading is not None:
                # Blend: weighted sum of desired path and obstacle-free direction
                alpha = 0.7  # favor path, 0.3 favor obstacle avoidance
                blended_heading = alpha * desired_yaw + (1 - alpha) * avoid_heading
                angle_error = self.normalize_angle(blended_heading - self.current_yaw)
            else:
                # If blocked completely, stop/turn
                angle_error = self.normalize_angle(desired_yaw - self.current_yaw)
            
            # self.get_logger().info(f"Waypoint {self.current_waypoint_index}: "
            #                       f"distance={distance_to_target:.3f}m, "
            #                       f"angle_error={math.degrees(angle_error):.1f}°, "
            #                       f"pos=({self.current_position['x']:.2f}, {self.current_position['y']:.2f})")
            
            # PREDICTIVE WAYPOINT DETECTION - Check if we'll overshoot
            # Calculate where robot will be in next time step
            dt = 0.1  # Timer period
            predicted_x = self.current_position['x'] + self.prev_linear_vel * math.cos(self.current_yaw) * dt
            predicted_y = self.current_position['y'] + self.prev_linear_vel * math.sin(self.current_yaw) * dt
            
            # Distance to waypoint from predicted position
            predicted_dx = target_x - predicted_x
            predicted_dy = target_y - predicted_y
            predicted_distance = math.sqrt(predicted_dx*predicted_dx + predicted_dy*predicted_dy)
            
            # Check if we're about to overshoot or already very close
            will_overshoot = (predicted_distance > distance_to_target and distance_to_target < 0.4)
            
            if distance_to_target < self.waypoint_tolerance or will_overshoot:
                self.get_logger().info(f"Waypoint {self.current_waypoint_index} reached! "
                                     f"(distance={distance_to_target:.3f}, predicted={predicted_distance:.3f})")
                self.current_waypoint_index += 1
                
                # Immediate velocity reduction when switching waypoints
                self.prev_linear_vel *= 0.5
                self.prev_angular_vel *= 0.5
                
                if self.current_waypoint_index >= len(self.path):
                    target_linear = 0.0
                    target_angular = 0.0
                    self.following_path = False
                    self.get_logger().info("Final destination reached!")
                else:
                    # Continue to next waypoint
                    return self.follow_path_smooth()
            else:
                # CONSERVATIVE MOTION CONTROL
                
                # Angular control - proportional to angle error
                target_angular = self.angular_kp * angle_error
                
                # Reduce angular velocity when very close to prevent spinning
                if distance_to_target < 0.5:
                    target_angular *= min(1.0, distance_to_target / 0.5)
                
                # Linear velocity - VERY conservative approach
                if abs(angle_error) < math.pi/6:  # Within 30 degrees - more restrictive
                    # Distance-based speed control with early braking
                    if distance_to_target > 1.0:
                        # Far away - use moderate speed
                        speed_factor = 0.8
                    elif distance_to_target > 0.5:
                        # Medium distance - start slowing down
                        speed_factor = 0.4 + 0.4 * (distance_to_target - 0.5) / 0.5
                    else:
                        # Very close - crawl speed
                        speed_factor = 0.2 + 0.2 * (distance_to_target / 0.5)
                    
                    # Angle-based reduction
                    angle_factor = max(0.3, 1.0 - abs(angle_error) / (math.pi/6))
                    
                    target_linear = self.linear_kp * self.max_linear_speed * speed_factor * angle_factor
                    
                elif abs(angle_error) < math.pi/3:  # 30-60 degrees - turn while moving slowly
                    target_linear = 0.05  # Very slow forward motion
                else:
                    # Large angle error - just turn in place
                    target_linear = 0.0
    
        # AGGRESSIVE VELOCITY SMOOTHING - prevent sudden changes
        # Much more conservative smoothing to prevent overshooting
        smoothing_factor = 0.95  # Very high smoothing
        
        # Apply smoothing
        smooth_linear = (smoothing_factor * self.prev_linear_vel + 
                        (1 - smoothing_factor) * target_linear)
        smooth_angular = (smoothing_factor * self.prev_angular_vel + 
                         (1 - smoothing_factor) * target_angular)
        
        # Additional acceleration limiting
        max_linear_accel = 0.2  # Very conservative acceleration
        max_angular_accel = 0.5
        dt = 0.1
        
        # Limit linear acceleration
        linear_change = smooth_linear - self.prev_linear_vel
        if abs(linear_change) > max_linear_accel * dt:
            smooth_linear = self.prev_linear_vel + (max_linear_accel * dt * (1 if linear_change > 0 else -1))
        
        # Limit angular acceleration  
        angular_change = smooth_angular - self.prev_angular_vel
        if abs(angular_change) > max_angular_accel * dt:
            smooth_angular = self.prev_angular_vel + (max_angular_accel * dt * (1 if angular_change > 0 else -1))
        
        # Store for next iteration
        self.prev_linear_vel = smooth_linear
        self.prev_angular_vel = smooth_angular
        
        # Final clamping
        smooth_linear = max(0.0, min(self.max_linear_speed, smooth_linear))
        smooth_angular = max(-self.max_angular_speed, min(self.max_angular_speed, smooth_angular))
        
        # Set message
        msg.linear.x = smooth_linear
        msg.angular.z = smooth_angular
        
        return msg


    def compute_avoidance_heading(self, ranges, angle_increment, angle_min):
        """Find avoidance direction based on free space"""
        # Convert LIDAR to polar sectors
        free_angles = []
        for i, r in enumerate(ranges):
            if np.isfinite(r) and r > self.safe_distance:
                angle = angle_min + i * angle_increment
                free_angles.append(angle)

        if not free_angles:
            return None  # blocked everywhere
        
        # Pick the free angle closest to forward
        return min(free_angles, key=lambda a: abs(a))
    

def main(args=None):
    rclpy.init(args=args)
    navigator = Navigator()
    rclpy.spin(navigator)
    navigator.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()