import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer')
        self.bridge = CvBridge()
        self.declare_parameter('robot_name', 'my_robot')
        self.robot_name = self.get_parameter('robot_name').get_parameter_value().string_value
        self.window_name = f"{self.robot_name} Camera View"

        self.subscription = self.create_subscription(
            Image,
            f'/{self.robot_name}/rgb_camera/image_color',
            self.image_callback,
            10
        )
        self.get_logger().info('Subscribed to camera topic')

    def image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image (BGR by default)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            scaled_image = cv2.resize(cv_image, (640, 480))  # Resize for better visibility
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return

        # Show the image
        cv2.imshow(self.window_name, scaled_image)
        cv2.waitKey(1)  # Needed to update the OpenCV window


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
