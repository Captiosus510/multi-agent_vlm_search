import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from llm_search.utils import YOLOWrapper

class ObjectDetector(Node):
    def __init__(self):
        super().__init__('object_detector')
        self.bridge = CvBridge()
        self.declare_parameter('robot_name', 'my_robot')
        self.robot_name = self.get_parameter('robot_name').get_parameter_value().string_value
        self.yolo_model = YOLOWrapper()

        # Subscribe to the camera topic
        self.cam_subscription = self.create_subscription(
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
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            detections = self.yolo_model.detect(cv_image)

            # Process detections (e.g., draw bounding boxes)
            for detection in detections:
                x1, y1, x2, y2, conf, cls_id = detection
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(cv_image, f'ID: {cls_id} Conf: {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the image with detections
            cv2.imshow(self.robot_name + ' Camera View', cv_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")


def main(args=None):
    rclpy.init(args=args)
    detector = ObjectDetector()
    rclpy.spin(detector)
    detector.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()