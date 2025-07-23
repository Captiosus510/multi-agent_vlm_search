import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from llm_search.utils import YOLOWrapper
import numpy as np
from std_msgs.msg import String

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

        # subscribe to goal topic
        self.goal_subscription = self.create_subscription(
            String,
            '/robot_goal',
            self.goal_callback,
            10
        )
        self.get_logger().info('Subscribed to goal topic')
        # Initialize goal variable
        self.goal = None
        
        # set up publisher for detection results
        self.detector_publisher = self.create_publisher(Image, f'/{self.robot_name}/detector/image', 10)
        self.found_objects_publisher = self.create_publisher(Image, f'/{self.robot_name}/detector/found', 10)
    
    def goal_callback(self, msg):
        if self.goal is None or self.goal != msg.data:
            self.get_logger().info(f"Received goal: {msg.data}")
            self.goal = msg.data
        else:
            self.get_logger().info("Goal has not changed, skipping processing.")

    def image_callback(self, msg):
        if self.goal is None:
            self.get_logger().warn("No goal set for object detection. Skipping image processing.")
            return
        class_list = self.goal.split(',') if isinstance(self.goal, str) else self.goal
        try:
            # Convert ROS Image message to OpenCV image (BGR by default)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            # Set classes for YOLO model
            self.yolo_model.set_classes(class_list)
            boxes, scores, class_ids, masks = self.yolo_model.detect(cv_image)

            # create a flag if any of the confidences are above threshold
            found_objects = any(score > 0.5 for score in scores)

            # create a copy for mask overlay
            overlay = cv_image.copy()

            # Process detections (e.g., draw bounding boxes)
            for i in range(len(boxes)):
                x1, y1, x2, y2 = map(int, boxes[i])
                conf = scores[i]
                cls_id = class_ids[i]
                class_name = class_list[int(cls_id)]
                # overlay bounding box and class name
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(cv_image, f'ID: {cls_id} Conf: {conf:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(cv_image, class_name, (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # If masks are available, overlay them
                if masks is not None and len(masks) > i:
                    mask = masks[i]
                    # Resize mask to match image size if needed
                    if mask.shape[:2] != cv_image.shape[:2]: 
                        mask = cv2.resize(mask, (cv_image.shape[1], cv_image.shape[0]), interpolation=cv2.INTER_NEAREST)
                    # Create a colored mask
                    colored_mask = np.zeros_like(cv_image, dtype=np.uint8)
                    colored_mask[mask > 0.5] = (0, 0, 255) # Example: Red color for mask
                    
                    # Apply this single colored mask to the overlay
                    cv2.addWeighted(overlay, 1, colored_mask, 0.5, 0, dst=overlay)
            
            # Blend the overlay with the original image once
            final_image = cv2.addWeighted(cv_image, 0.7, overlay, 0.3, 0)
            
            # Publish the processed image
            output_msg = self.bridge.cv2_to_imgmsg(final_image, encoding='bgr8')
            output_msg.header.stamp = self.get_clock().now().to_msg()
            output_msg.header.frame_id = f'{self.robot_name}'
            self.detector_publisher.publish(output_msg)
            self.get_logger().info('Published detection image')

            # Only publish found objects image for a few frames after initial detection, then space out uploads
            if not hasattr(self, 'found_counter'):
                self.found_counter = 0
                self.last_found_frame = 0
                self.found_interval = 30  # frames to wait between uploads after initial burst
                self.initial_burst = 5    # number of frames to upload immediately after found

            if found_objects:
                if self.found_counter < self.initial_burst or \
                   (self.found_counter >= self.initial_burst and (self.found_counter - self.initial_burst) % self.found_interval == 0):
                    found_objects_msg = self.bridge.cv2_to_imgmsg(final_image, encoding='bgr8')
                    found_objects_msg.header.stamp = self.get_clock().now().to_msg()
                    found_objects_msg.header.frame_id = f'{self.robot_name}_camera_frame'
                    self.found_objects_publisher.publish(found_objects_msg)
                    self.get_logger().info('Published found objects image')
                self.found_counter += 1
            else:
                self.found_counter = 0  # Reset counter when no objects found

            # Display the image with detections
            # cv2.imshow(self.robot_name + ' Camera View', cv_image)
            # cv2.waitKey(1)
            # Convert back to ROS Image message

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