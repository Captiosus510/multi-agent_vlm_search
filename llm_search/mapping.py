import cv2
import numpy as np
from llm_search.utils.siglip import SigLipInterface
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo, Imu
from rclpy.node import Node
from message_filters import Subscriber, ApproximateTimeSynchronizer
from geometry_msgs.msg import PointStamped

class Mapper(Node):
    map_size = (500, 500)  # Size of the map in pixels
    resolution = 0.05  # Resolution of the map in meters per pixel
    map_array = np.zeros(map_size, dtype=np.float32)  # Initialize the map as a 2D array
    def __init__(self, robot_name: str):
        super().__init__('mapper')
        self.robot_name = robot_name
        self.bridge = CvBridge()
        self.siglip_interface = SigLipInterface()
        self.goal = None
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
        self.image_subscription = Subscriber(self, Image, f'/{self.robot_name}/camera/image_color')
        self.get_logger().info(f'Subscribed to image topic: /{self.robot_name}/camera/image_color')

        self.ts = ApproximateTimeSynchronizer(
            [self.depth_info_subscription, self.depth_sensor_subscription, 
             self.imu_subscription, self.gps_subscription, self.image_subscription],
            queue_size=500,
            slop=1.0  # Adjust the slop as needed
        )
        self.ts.registerCallback(self.update_map)


    def update_map(self, depth_info_msg: CameraInfo, depth_msg: Image, imu_msg: Imu, gps_msg: PointStamped, image_msg: Image):
        
        self.get_logger().info(f"Updating map with depth and IMU data")
        # Extract camera parameters from the depth info message
        fx = depth_info_msg.k[0]  # Focal length in x
        fy = depth_info_msg.k[4]  # Focal length in y
        cx = depth_info_msg.k[2]  # Optical center x
        cy = depth_info_msg.k[5]  # Optical center y

        # Convert ROS Image message to OpenCV image (BGR by default)
        depth_array = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')

        # get orientation from IMU message
        # imu_msg.orientation is a quaternion (x, y, z, w)
        roll = imu_msg.orientation.x
        pitch = imu_msg.orientation.y
        yaw = imu_msg.orientation.z  # Assuming z is the yaw angle
           

        # gps position
        gps_position = gps_msg.point

        # process image
        image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert depth to 3D points in camera frame
        height, width = depth_array.shape
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))

        X = (xx - cx) * depth_array / fx
        Y = (yy - cy) * depth_array / fy
        Z = depth_array

        points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
        points = points[Z.flatten() > 0.1]  # remove invalid

        # Rotate points by robot yaw
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
        rotated_points = points @ rotation.T

        # Translate by GPS position (x, y)
        translated_points = rotated_points.copy()
        translated_points[:, 0] += gps_position.x
        translated_points[:, 1] += gps_position.y

        # Clear current map or keep updating
        for p in translated_points:
            mx = int(p[0] / self.resolution + self.map_size[0] // 2)
            my = int(p[1] / self.resolution + self.map_size[1] // 2)
            if 0 <= mx < self.map_size[0] and 0 <= my < self.map_size[1]:
                self.map_array[mx, my] = 1.0  # Mark occupied
        
        # Publish the map as an image
        map_image = (self.map_array * 255).astype(np.uint8)
        map_image = cv2.applyColorMap(map_image, cv2.COLORMAP_JET)
        cv2.imshow('Map', map_image)
        cv2.waitKey(1)  # Display the map in a window

        
def main(args=None):
    import rclpy
    rclpy.init(args=args)
    robot_name = 'my_robot'  # Replace with your robot name
    mapper = Mapper(robot_name)
    rclpy.spin(mapper)
    mapper.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
        

    


    
