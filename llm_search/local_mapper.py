import cv2
import numpy as np
from llm_search.utils.siglip import SigLipInterface
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, Imu
from rclpy.node import Node
from message_filters import Subscriber, ApproximateTimeSynchronizer
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String

class LocalMapper(Node):
    map_size = (250, 250)  # Size of the map in pixels
    resolution = 0.05  # Resolution of the map in meters per pixel
    
    def __init__(self):
        super().__init__('mapper')
        self.declare_parameter('robot_name', 'my_robot')
        self.robot_name = self.get_parameter('robot_name').get_parameter_value().string_value

        self.declare_parameter('show_maps', True)
        self.show_maps = self.get_parameter('show_maps').get_parameter_value().bool_value

        self.bridge = CvBridge()
        self.siglip_interface = SigLipInterface()
        self.goal = None
        
        # Initialize value map with semantic values and confidence scores
        self.semantic_value_map = np.zeros(self.map_size, dtype=np.float32)
        self.confidence_map = np.zeros(self.map_size, dtype=np.float32)
        
        # Camera FOV parameters (will be updated from camera info)
        self.horizontal_fov = np.pi / 3  # Default 60 degrees, will be calculated from camera params
        self.max_depth = 5.0  # Maximum depth for cone projection
        # subscribe to depth sensor info topic
        self.depth_info_subscription = Subscriber(self, CameraInfo, f'/{self.robot_name}/depth_sensor/camera_info')
        self.get_logger().info(f'Subscribed to depth sensor info topic: /{self.robot_name}/depth_sensor/camera_info')
        # subscribe to depth sensor topic
        self.depth_sensor_subscription = Subscriber(self, Image, f'/{self.robot_name}/depth_sensor/image')
        self.get_logger().info(f'Subscribed to depth sensor topic: /{self.robot_name}/depth_sensor/image')
        # subscribe to IMU topic
        self.imu_subscription = Subscriber(self, Imu, f'/{self.robot_name}/imu')
        self.get_logger().info(f'Subscribed to IMU topic: /{self.robot_name}/imu')
        # subscribe to gps topic
        self.gps_subscription = Subscriber(self, PointStamped, f'/{self.robot_name}/p3d_gps')
        self.get_logger().info(f'Subscribed to GPS topic: /{self.robot_name}/p3d_gps')
        # subscribe to image topic
        self.image_subscription = Subscriber(self, Image, f'/{self.robot_name}/rgb_camera/image_color')
        self.get_logger().info(f'Subscribed to image topic: /{self.robot_name}/rgb_camera/image_color')
        # subscribe to goal topic
        self.goal_subscription = self.create_subscription(String, f'robot_goal', self.goal_callback, 10)
        self.get_logger().info(f'Subscribed to goal topic: /robot_goal')
        # local map publishers
        self.local_semantic_map_pub = self.create_publisher(
        Image, f'/{self.robot_name}/local_semantic_map', 10)

        self.local_confidence_map_pub = self.create_publisher(
        Image, f'/{self.robot_name}/local_confidence_map', 10)

        self.ts = ApproximateTimeSynchronizer(
            [self.depth_info_subscription, self.depth_sensor_subscription, 
             self.imu_subscription, self.gps_subscription, self.image_subscription],
            queue_size=500,
            slop=1.0  # Adjust the slop as needed
        )
        self.ts.registerCallback(self.update_map)
    
    def goal_callback(self, msg: String):
        """Callback to update the goal from the robot's goal topic."""
        if self.goal != msg.data:
            self.get_logger().info(f"Received new goal: {msg.data}")
            self.goal = msg.data.split(',')
        

    def create_cone_mask(self, robot_x, robot_y, robot_yaw, depth_array, fx, fy, cx, cy):
        """Create a cone-shaped mask representing the camera FOV with confidence scores"""
        mask = np.zeros(self.map_size, dtype=np.float32)
        confidence_mask = np.zeros(self.map_size, dtype=np.float32)
        
        height, width = depth_array.shape
        
        # Create pixel coordinates
        u, v = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert to normalized camera coordinates
        x_cam = (u - cx) / fx
        y_cam = (v - cy) / fy
        
        # Calculate angle from optical axis for each pixel
        angles = np.arctan2(np.abs(x_cam), 1.0)  # Angle from center
        
        # Calculate confidence based on distance from optical axis
        # cos²(θ/(θ_fov/2) * π/2) as described in the paper
        normalized_angles = angles / (self.horizontal_fov / 2)
        confidence_scores = np.cos(normalized_angles * np.pi / 2) ** 2
        
        # Only consider pixels within FOV and valid depth
        valid_fov = angles <= (self.horizontal_fov / 2)
        valid_depth = (depth_array > 0.1) & (depth_array <= self.max_depth)
        valid_pixels = valid_fov & valid_depth
        
        # Get valid pixel indices
        valid_v, valid_u = np.where(valid_pixels)
        
        if len(valid_v) == 0:
            return mask, confidence_mask
        
        # Vectorized computation for valid pixels
        depths = depth_array[valid_v, valid_u]
        x_cam_valid = x_cam[valid_v, valid_u]
        y_cam_valid = y_cam[valid_v, valid_u]
        confidence_valid = confidence_scores[valid_v, valid_u]
        
        # Convert to 3D points in camera frame
        x_3d = x_cam_valid * depths
        y_3d = y_cam_valid * depths
        z_3d = depths
        
        # Rotate by robot yaw and translate to world coordinates
        cos_yaw, sin_yaw = np.cos(robot_yaw), np.sin(robot_yaw)
        world_x = robot_x + (cos_yaw * z_3d - sin_yaw * x_3d)
        world_y = robot_y + (sin_yaw * z_3d + cos_yaw * x_3d)
        
        # Convert to map coordinates
        map_x = ((world_x / self.resolution) + self.map_size[0] // 2).astype(int)
        map_y = ((world_y / self.resolution) + self.map_size[1] // 2).astype(int)
        
        # Filter valid map coordinates
        valid_map = (map_x >= 0) & (map_x < self.map_size[0]) & (map_y >= 0) & (map_y < self.map_size[1])
        
        if np.any(valid_map):
            map_x_valid = map_x[valid_map]
            map_y_valid = map_y[valid_map]
            confidence_map_valid = confidence_valid[valid_map]
            
            # Update mask and confidence (take maximum for overlapping pixels)
            for i in range(len(map_x_valid)):
                mx, my = map_x_valid[i], map_y_valid[i]
                mask[mx, my] = 1.0
                confidence_mask[mx, my] = max(confidence_mask[mx, my], confidence_map_valid[i])
        
        return mask, confidence_mask

    def update_value_map(self, semantic_score, cone_mask, confidence_mask):
        """Update the value map using weighted averaging as described in the paper"""
        # Areas where we have new observations
        new_observation_mask = cone_mask > 0
        
        if not np.any(new_observation_mask):
            return
        
        # Get indices of pixels with new observations
        update_indices = np.where(new_observation_mask)
        
        for idx in zip(*update_indices):
            i, j = idx
            curr_confidence = confidence_mask[i, j]
            curr_semantic = semantic_score  # Same score for all pixels in current view
            
            prev_confidence = self.confidence_map[i, j]
            prev_semantic = self.semantic_value_map[i, j]
            
            if prev_confidence > 0:  # Previously seen pixel
                # Weighted average for semantic value
                total_confidence = curr_confidence + prev_confidence
                new_semantic = (curr_confidence * curr_semantic + prev_confidence * prev_semantic) / total_confidence
                
                # Weighted average biased towards higher confidence
                new_confidence = ((curr_confidence ** 2) + (prev_confidence ** 2)) / total_confidence
            else:  # First time seeing this pixel
                new_semantic = curr_semantic
                new_confidence = curr_confidence
            
            self.semantic_value_map[i, j] = new_semantic
            self.confidence_map[i, j] = new_confidence

    def visualize_value_map(self):
        """Visualize the semantic value map with confidence blending"""
        # Normalize semantic values to 0-1 range for visualization
        if np.max(self.semantic_value_map) > 0:
            normalized_values = self.semantic_value_map / np.max(self.semantic_value_map)
        else:
            normalized_values = self.semantic_value_map.copy()
        
        # Create RGB visualization
        # Use confidence as alpha channel for blending
        confidence_normalized = np.clip(self.confidence_map, 0, 1)
        
        # Create heatmap using jet colormap for semantic values
        semantic_colored = cv2.applyColorMap((normalized_values * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Blend with confidence (higher confidence = more opaque)
        blended_map = semantic_colored.copy()
        for i in range(3):  # RGB channels
            blended_map[:, :, i] = (semantic_colored[:, :, i] * confidence_normalized).astype(np.uint8)
        
        # Display both maps if enabled
        if self.show_maps:
            cv2.imshow(f'{self.robot_name} Semantic Value Map', blended_map)
            cv2.imshow(f'{self.robot_name} Confidence Map', (confidence_normalized * 255).astype(np.uint8))
            cv2.waitKey(1)

        # Publish the local colorized map
        semantic_msg = self.bridge.cv2_to_imgmsg(blended_map, encoding='bgr8')

        confidence_vis = (confidence_normalized * 255).astype(np.uint8)
        confidence_msg = self.bridge.cv2_to_imgmsg(confidence_vis, encoding='mono8')

        semantic_msg.header.stamp = self.get_clock().now().to_msg()
        semantic_msg.header.frame_id = "map"
        confidence_msg.header.stamp = semantic_msg.header.stamp
        confidence_msg.header.frame_id = "map"

        self.local_semantic_map_pub.publish(semantic_msg)
        self.local_confidence_map_pub.publish(confidence_msg)


    def get_value_map_data(self):
        """Return the current semantic value map and confidence map for external use"""
        return self.semantic_value_map.copy(), self.confidence_map.copy()
    
    def save_value_map(self, filename_prefix="value_map"):
        """Save the current value maps as images"""
        # Save semantic value map
        if np.max(self.semantic_value_map) > 0:
            normalized_values = (self.semantic_value_map / np.max(self.semantic_value_map) * 255).astype(np.uint8)
            cv2.imwrite(f"{filename_prefix}_semantic.png", normalized_values)
        
        # Save confidence map
        confidence_img = (np.clip(self.confidence_map, 0, 1) * 255).astype(np.uint8)
        cv2.imwrite(f"{filename_prefix}_confidence.png", confidence_img)
        
        # self.get_logger().info(f"Value maps saved with prefix: {filename_prefix}")

    def update_map(self, depth_info_msg: CameraInfo, depth_msg: Image, imu_msg: Imu, gps_msg: PointStamped, image_msg: Image):
        
        # self.get_logger().info(f"Updating value map with semantic confidence scores")
        
        # Extract camera parameters from the depth info message
        fx = depth_info_msg.k[0]  # Focal length in x
        fy = depth_info_msg.k[4]  # Focal length in y
        cx = depth_info_msg.k[2]  # Optical center x
        cy = depth_info_msg.k[5]  # Optical center y
        
        # Calculate horizontal FOV from camera parameters
        width = depth_info_msg.width
        self.horizontal_fov = 2 * np.arctan(width / (2 * fx))

        # Convert ROS Image message to OpenCV image
        depth_array = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')

        # Get orientation from IMU message (assuming z is yaw)
        yaw = imu_msg.orientation.z
        
        # Get GPS position
        robot_x = gps_msg.point.x
        robot_y = gps_msg.point.y

        # Process RGB image and get semantic similarity score
        image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        semantic_score = 0.0
        if self.goal is not None:
            semantic_score = self.siglip_interface.compute_confidence(rgb_image, self.goal)
            self.get_logger().info(f"Semantic score for goal: {semantic_score:.3f}")

        # Create cone-shaped mask with confidence scores
        cone_mask, confidence_mask = self.create_cone_mask(
            robot_x, robot_y, yaw, depth_array, fx, fy, cx, cy
        )
        
        # Update the value map using weighted averaging
        self.update_value_map(semantic_score, cone_mask, confidence_mask)
        
        # Visualize the value map
        self.visualize_value_map()

        
def main(args=None):
    import rclpy
    rclpy.init(args=args)
    mapper = LocalMapper()
    rclpy.spin(mapper)
    mapper.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()






