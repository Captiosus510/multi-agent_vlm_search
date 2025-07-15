from message_filters import Subscriber, ApproximateTimeSynchronizer
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
import numpy as np
import cv2
from cv_bridge import CvBridge

class GlobalMapMerger(Node):
    """
    This node merges the global map from multiple robots into a single map.
    It subscribes to the global maps of all robots and publishes the merged map.
    """
    def __init__(self):
        super().__init__('global_map_merger')
        self.get_logger().info("Initializing Global Map Merger Node...")
        self.declare_parameter('robot_names', ['my_robot', 'other_robot'])
        self.robot_names = self.get_parameter('robot_names').get_parameter_value().string_array_value

        self.declare_parameter('show_maps', True)
        self.show_maps = self.get_parameter('show_maps').get_parameter_value().bool_value

        self.bridge = CvBridge()

        self.value_maps = []
        self.confidence_maps = []
        self.robot_positions = []

        for robot_name in self.robot_names:
            self.get_logger().info(f"Subscribing to local maps for {robot_name}")
            value_map_sub = Subscriber(self, Image, f'/{robot_name}/local_semantic_map')
            confidence_map_sub = Subscriber(self, Image, f'/{robot_name}/local_confidence_map')
            position_sub = Subscriber(self, PointStamped, f'/{robot_name}/p3d_gps')

            self.value_maps.append(value_map_sub)
            self.confidence_maps.append(confidence_map_sub)
            self.robot_positions.append(position_sub)

        self.ts = ApproximateTimeSynchronizer(
            self.value_maps + self.confidence_maps + self.robot_positions,
            queue_size=10,
            slop=0.5
        )
        self.ts.registerCallback(self.merge_callback)


         # Example: global map size 1000x1000
        self.global_map_size = (1000, 1000)
        self.resolution = 0.05  # meters per pixel
        self.center_x = self.global_map_size[0] // 2
        self.center_y = self.global_map_size[1] // 2

        # Initialize empty global maps (3-channel for color; adjust if grayscale)
        self.global_value_map = np.zeros((*self.global_map_size, 3), dtype=np.float32)
        self.global_confidence_map = np.zeros(self.global_map_size, dtype=np.float32)


    def merge_callback(self, *args):
        """
        Callback function to merge the global maps from multiple robots.
        It averages the value and confidence maps from all robots.
        """
        self.get_logger().info("Merging global maps from all robots...")
        value_maps = args[:len(self.robot_names)]
        confidence_maps = args[len(self.robot_names):2*len(self.robot_names)]
        positions = args[2*len(self.robot_names):]

        self.global_value_map.fill(0)
        self.global_confidence_map.fill(0)

        for i, robot_name in enumerate(self.robot_names):
            value_map = self.bridge.imgmsg_to_cv2(value_maps[i], desired_encoding='bgr8').astype(np.float32)
            confidence_map = self.bridge.imgmsg_to_cv2(confidence_maps[i], desired_encoding='mono8').astype(np.float32) / 255.0  # Normalize to [0, 1]
            position = positions[i].point

            # Convert position to pixel coordinates
            pixel_x = int(position.x / self.resolution) + self.center_x
            pixel_y = int(position.z / self.resolution) + self.center_y

            # half width and height of the value map
            half_w = value_map.shape[1] // 2
            half_h = value_map.shape[0] // 2

            slice_x = slice(max(0, pixel_x - half_w), min(self.global_map_size[0], pixel_x + half_w))
            slice_y = slice(max(0, pixel_y - half_h), min(self.global_map_size[1], pixel_y + half_h))

            # Compute local map crop to match clipped global slice
            start_x_local = max(0, -(pixel_x - half_w))
            start_y_local = max(0, -(pixel_y - half_h))
            end_x_local = start_x_local + (slice_x.stop - slice_x.start)
            end_y_local = start_y_local + (slice_y.stop - slice_y.start)

            # Crop the value map and confidence map to the local slice
            value_map = value_map[start_y_local:end_y_local, start_x_local:end_x_local]
            confidence_map = confidence_map[start_y_local:end_y_local, start_x_local:end_x_local]


            self.global_confidence_map[slice_y, slice_x] += confidence_map * (confidence_map > 0)  # Avoid division by zero

            for c in range(3):  # Assuming 3 channels (BGR)
                self.global_value_map[slice_y, slice_x, c] += value_map[:, :, c] * confidence_map
        
        # Normalize the global value map by the confidence map
        with np.errstate(divide='ignore', invalid='ignore'):
            for c in range(3):
                self.global_value_map[:, :, c] = np.divide(
                    self.global_value_map[:, :, c],
                    self.global_confidence_map,
                    out=np.zeros_like(self.global_value_map[:, :, c]),
                    where=self.global_confidence_map != 0
                )

        # Display the merged global map
        if self.show_maps:
            cv2.imshow('Global Value Map', self.global_value_map.astype(np.uint8))
            cv2.imshow('Global Confidence Map', (self.global_confidence_map * 255).astype(np.uint8))
            cv2.waitKey(1)  # Needed to update the OpenCV window


        
def main(args=None):
    import rclpy
    rclpy.init(args=args)
    node = GlobalMapMerger()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()