import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO


class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer')
        self.bridge = CvBridge()
        self.declare_parameter('robot_name', 'my_robot')
        self.robot_name = self.get_parameter('robot_name').get_parameter_value().string_value
        self.window_name = f"{self.robot_name} Camera View"

        self.det_model = YOLO('yolo11s.pt')

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
            img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            # Perform inference using YOLO model
            results = self.det_model(img_rgb)
            self.get_logger().info(f"Detected {len(results[0].boxes)} objects")
            image = cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR)  # Get the image with detections
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return

        # Show the image
        cv2.imshow(self.window_name, image)
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
