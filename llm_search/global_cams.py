#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: global_cams.py
Author: Mahd Afzal
Date: 2025-08-19
Version: 1.0
Description: 
    Subscribes to camera feed topics of all robots. Displays the camera feeds in a grid.
    This collects everything to visualize into a single window, making it easier to monitor all robot cameras at once.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class GlobalCamsNode(Node):
    def __init__(self):
        super().__init__('global_cams')
        self.get_logger().info('GlobalCamsNode has been started.')

        self.bridge = CvBridge()

        # Declare parameters
        self.declare_parameter('topic_name', '/rgb_camera/image_color')
        self.topic_name = self.get_parameter('topic_name').get_parameter_value().string_value

        # subscribe to robot names topic
        self.robot_names_subscription = self.create_subscription(
            String,
            'robot_names',
            self.robot_names_callback,
            10
        )

        self.robot_names = []
        self.cam_subscriptions = {}
        self.cam_stream = {}
    
    def robot_names_callback(self, msg):
        robot_names = [name.strip() for name in msg.data.split(',') if name.strip()]
        if robot_names != self.robot_names:
            self.robot_names = robot_names
            subs = {}
            for robot_name in self.robot_names:
                topic = f'/{robot_name}{self.topic_name}'
                sub = self.create_subscription(
                    Image,
                    topic,
                    self.camera_callback,
                    10
                )
                subs[robot_name] = sub
            self.cam_subscriptions = subs

    def camera_callback(self, msg):
        robot_name = msg.header.frame_id  # Assuming the robot name is in the frame_id
        
        cv2_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.cam_stream[robot_name] = cv2_image
    
    def display_cameras(self):
        if not self.cam_stream:
            return

        images = [img for img in self.cam_stream.values() if img is not None]
        if not images:
            return
        
        self.get_logger().info(f"Displaying {len(images)} cameras.")

        # Determine grid size
        n = len(images)
        cols = int(n**0.5)
        rows = (n + cols - 1) // cols

        # Resize images to same size
        h, w = images[0].shape[:2]
        size = (320, 240)
        resized = [cv2.resize(img, size) for img in images]

        # Create blank canvas
        canvas = np.zeros((size[1]*rows, size[0]*cols, 3), dtype=np.uint8)

        # Paste images into canvas
        for idx, img in enumerate(resized):
            r = idx // cols
            c = idx % cols
            y, x = r*size[1], c*size[0]
            canvas[y:y+size[1], x:x+size[0]] = img

        cv2.imshow('All Cameras', canvas)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = GlobalCamsNode()
    try:
        while rclpy.ok():
            rclpy.spin_once(node)
            node.display_cameras()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()