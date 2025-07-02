import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import torch
from std_msgs.msg import String, Empty
import tempfile
from llm_search_interfaces.srv import Analysis

class CameraViewer(Node):
    def __init__(self):
        super().__init__('camera_viewer')
        self.bridge = CvBridge()
        self.declare_parameter('robot_name', 'my_robot')
        self.robot_name = self.get_parameter('robot_name').get_parameter_value().string_value
        self.window_name = f"{self.robot_name} Camera View"

        # self.det_model = YOLO('yoloe-11s-seg.pt')
        self.det_model = YOLO('yolo11m.pt')
        self.class_names = self.det_model.names
        # Subscribe to the camera top   ic
        self.cam_subscription = self.create_subscription(
            Image,
            f'/{self.robot_name}/rgb_camera/image_color',
            self.image_callback,
            10
        )
        self.get_logger().info('Subscribed to camera topic')
        # Subscribe to the goal topic
        self.goal_subscription = self.create_subscription(
            String,
            '/robot_goal',
            self.goal_callback,
            10
        )
        self.get_logger().info('Subscribed to goal topic')
        self.goal = None

        # Register as client for vlm service
        self.vlm_client = self.create_client(Analysis, 'analysis')

        self.req = Analysis.Request()
        self.service_call_in_progress = False
    
    def goal_callback(self, msg):
        """
        Callback function to handle incoming goal messages.
        This function can be used to update the robot's goal or perform actions based on the goal.
        """
        self.goal = msg.data        


    def image_callback(self, msg):
        self.get_logger().info(f"Goal: {self.goal}")
        if self.goal is None:
            self.get_logger().info("No goal set, skipping image processing.")
            return
        try:
            # Convert ROS Image message to OpenCV image (BGR by default)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            img_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # Perform inference using YOLO model
            # self.det_model.set_classes([self.goal], self.det_model.get_text_pe([self.goal]))
            results = self.det_model(img_rgb)

            # Parse the results
            result = results[0]
            boxes = result.boxes
            if boxes is None or boxes.conf.numel() == 0:
                self.get_logger().info("No detections")
            else:     
                # get the highest confidence detection
                max_conf_idx = boxes.conf.argmax()
                top_class = int(boxes.cls[max_conf_idx].item())
                top_conf = boxes.conf[max_conf_idx].item()
                top_box = boxes.xyxy[max_conf_idx].cpu().numpy()


                if top_conf > 0.6 and self.class_names[top_class] == self.goal and not self.service_call_in_progress:
                    self.get_logger().info(f"Goal '{self.goal}' found with confidence {top_conf:.2f}")
                    self.service_call_in_progress = True
                    self.req.image = msg
                    self.vlm_client.call_async(self.req).add_done_callback(self.vlm_response_callback)

                    # with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                    #     cv2.imwrite(temp_file.name, cv_image)
                    #     self.get_logger().info(f"Image saved to {temp_file.name}")
                    #     self.req.filepath = temp_file.name
                    #     self.vlm_client.call_async(self.req).add_done_callback(self.vlm_response_callback)

            # Draw bounding boxes and labels on the image
            self.get_logger().info(f"Detected {len(boxes)} objects")
            image = cv2.cvtColor(results[0].plot(), cv2.COLOR_RGB2BGR)  # Get the image with detections
        except Exception as e:
            self.get_logger().error(f"CV Bridge error: {e}")
            return

        # Show the image
        cv2.imshow(self.window_name, image)
        cv2.waitKey(1)  # Needed to update the OpenCV window
    
    def vlm_response_callback(self, future):
        try:
            response = future.result()
            if response.found:
                self.get_logger().info("Object found and confirmed by VLM service.")
                self.shutdown_all()
            else:
                self.get_logger().info("Object not found or not confirmed by VLM service.")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

        self.service_call_in_progress = False

    def shutdown_all(self):
        self.get_logger().info('Shutting down system')
        cv2.destroyAllWindows()
        self.destroy_node()


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
