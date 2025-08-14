import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
from std_msgs.msg import String, Empty
import numpy as np

class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer')
        self.bridge = CvBridge()
        self.declare_parameter('robot_name', 'my_robot')
        self.declare_parameter('show_depth', False)
        self.declare_parameter('show_rgb', False)
        self.declare_parameter('camera_topic', '/rgb_camera/image_color')
        self.robot_name = self.get_parameter('robot_name').get_parameter_value().string_value
        self.show_depth = self.get_parameter('show_depth').get_parameter_value().bool_value
        self.show_rgb = self.get_parameter('show_rgb').get_parameter_value().bool_value
        self.camera_topic = self.get_parameter('camera_topic').get_parameter_value().string_value
        self.window_name = f"{self.robot_name} Camera View"

        # Subscribe to the camera topic
        if self.show_rgb:
            self.cam_subscription = self.create_subscription(
                Image,
                f'/{self.robot_name}{self.camera_topic}',
                self.image_callback,
                10
            )
            self.get_logger().info('Subscribed to camera topic')
        # subscribe to depth camera topic if it exists
        if self.show_depth:
            self.depth_subscription = self.create_subscription(
                Image,
                f'/{self.robot_name}/depth_sensor/image',
                self.depth_callback,
                10
            )
        
            
    def goal_callback(self, msg):
        if self.goal is None or self.goal != msg.data:
            self.get_logger().info(f"Received goal: {msg.data}")
            self.goal = msg.data
    

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image (BGR by default)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return

        # Show the image
        cv2.imshow(self.window_name, cv_image)
        cv2.waitKey(1)  # Needed to update the OpenCV window
    
    def depth_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image (BGR by default)
            depth_array = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            # np.save('test_depth.npy', depth_array)  # Save depth array for debugging
            # self.get_logger().info(depth_array.flatten()[:100])
            depth_image = self.visualize_depth(depth_array)
            # Process the depth image if needed
            cv2.imshow(f"{self.robot_name} Depth Camera View", depth_image)
            cv2.waitKey(1)  # Needed to update the OpenCV window
        except Exception as e:
            self.get_logger().error(f"Depth CV Bridge error: {e}")

    def visualize_depth(self, depth_array: np.ndarray) -> np.ndarray:
        """
        Visualizes the depth array as a grayscale image.
        
        Args:
            depth_array (np.ndarray): The depth data as a NumPy array.
        
        Returns:
            np.ndarray: The visualized depth image.
        """

        depth_image = np.clip(depth_array, 0, 10)  # Clip values to [0, 10] for visualization
        depth_image = (depth_image * 255 / 10).astype(np.uint8)  # Scale to [0, 255] for visualization
        return depth_image


def main(args=None):
    rclpy.init(args=args)
    node = CameraViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
