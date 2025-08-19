#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Filename: vlm_services.py
Author: Mahd Afzal
Date: 2025-08-19
Version: 1.0

Description: 
    Main interface node for Vision-Language Model (VLM) services in a multi-robot system. 
    This node provides the bridge between user natural language queries and robot actions, 
    enabling semantic understanding of the environment, task allocation, and multi-robot coordination.  

    The system leverages a VLM (via OpenAI API) for high-level reasoning and planning, 
    while interfacing with ROS 2 topics for sensor data, robot states, and path planning.

Key Features:
    - Natural language interaction with the user through GPT-based structured responses.
    - Centralized multi-robot scene planning (monitoring and search behaviors).
    - Semantic prompt refinement and goal setting for robots.
    - Integration of global RGB and depth cameras with segmentation and mesh generation.
    - Automatic conversion of user intent into robot paths via Conflict-Based Search (CBS).
    - Interrupt-driven reporting of search results when goal objects are detected.
    - Real-time visualization of triangulated environment and robot positions.

Core Components:
    - **Callbacks**  
      All incoming sensor data (images, depth, GPS, IMU) are stored within the class instance 
      and updated via subscription callbacks. These provide the latest perception state for planning.

    - **Conversation Loop**  
      A continuous interactive loop with the VLM (GPT) that guides task execution. 
      It ensures structured reasoning, prompt refinement, function calls (take_picture, set_goal, show_grid, direct_robot), 
      and final task termination (`stop`).  
      Conversation is run in a separate thread so ROS spinning remains responsive.

    - **user_input_with_interrupt**  
      Reads console input from the user while allowing asynchronous interruptions.  
      For example, if a robot finds an object during a search, the loop is interrupted 
      and the detection is reported immediately.

    - **Image Processing**  
      Preprocesses RGB and depth images into a triangulated mesh of the environment.  
      Overlays a grid to allow VLM-guided semantic reasoning about search/monitor locations.  
      Maintains adjacency graphs for path planning.

    - **Path Planning**  
      Uses a CBS (Conflict-Based Search) planner to assign robots to triangles in the mesh 
      and generate conflict-free, synchronized paths.  
      Paths are published as ROS `nav_msgs/Path` messages.

    - **Visualization**  
      Provides both OpenCV and Matplotlib visualizations of grid overlays, 
      triangle IDs, robot positions, orientations, and allocated tasks.

ROS Interfaces:
    Subscribed Topics:
        - /global_cam/rgb_camera/image_color (sensor_msgs/Image) : Global RGB image.
        - /global_cam/depth_sensor/image (sensor_msgs/Image) : Global depth map.
        - /<robot_name>/p3d_gps (geometry_msgs/PointStamped) : Robot GPS position.
        - /<robot_name>/imu (sensor_msgs/Imu) : Robot orientation.

    Published Topics:
        - /robot_goal (std_msgs/String) : Parsed/refined goal prompts for robots.
        - /robot_names (std_msgs/String) : List of active robot names in the environment.
        - /<robot_name>/path (nav_msgs/Path) : Conflict-free path assignments for each robot.

TODO:
    - Simplify and optimize input/context prompts to improve reasoning efficiency.
    - Add support for dynamic robot availability updates.
    - Improve handling of quaternion orientation data (full conversion for yaw).
    - Extend monitoring/search behaviors with richer semantic categories.
    - Implement proper search behavior.

Usage:
    Run this node within a ROS 2 launch standalone.  
    The node automatically starts a background conversation loop with the VLM and 
    processes user queries, robot state updates, and environment observations.

"""


import rclpy
from rclpy.node import Node
import time
import threading, queue, sys, select
import cv2
from llm_search.utils.openai_interface import OpenAIInterface
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import String
import tempfile
import numpy as np
from sensor_msgs.msg import Image
from ultralytics import SAM
from llm_search.utils.vector import Vector7D
from llm_search.utils.mesher import generate_mesh, inverse_mesh
from llm_search.utils.graphing import *
import matplotlib.pyplot as plt
from geometry_msgs.msg import PointStamped
from functools import partial
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Imu
import math
from llm_search.utils.mapf import CBSPlanner

class VLMServices(Node):
    
    system_prompt = """
You are given a view of the scene from the perspective of a human. The environment has multiple TurtleBots already deployed and available. You are a centralised scene planner.\
Semantically analyze the scene for environment space, objects, and their relationships to provide structured responses.

You can perform two tasks with TurtleBots in the scene: multi-robot search and multi-robot monitoring in the given scene. \
First ask/Analyse what behavior the user wants (monitor or search).For each task, \
you interactively refine the user's input prompt and provide structured output to meaningfully direct robots to locations in the scene.

The robots are named sequentially: tb1, tb2, tb3, etc. You will have access to a specific number of robots that are already present in the environment.

Each interaction should go roughly like this:

1. You make sure the user specifies what they are looking for and if they want to monitor or search for an object.

2. You refine the user input prompt and return the final list of 3 prompts. 
For example, "I want to monitor near door for a package" → "Box, Package, cardboard Box"

3. Use the take_picture function to capture the current scene. You will also see the grid overlay that is numbered. 
Try to semantically analyze the scene and identify points of interest where turtlebots can be directed to search or monitor.
ALLOCATION: only propose the minimum number of robots needed to achieve the task and the minimum number of grid cells. If you have multiple robots
spread them out meaningfully.

4. Confirm with the user about the grid cells you want to direct robots to. Use show_grid to show the grid image. DO NOT SHOW THE GRID CELL NUMBERS, JUST THE IMAGE.

For monitoring:

5. If one of the robots find the goal object, initiate a conversation with the user. You will be given an image of the monitoring object.
   Analyze the image and report the semantic location and the description of the object found (no coordinates).

6. Ask the user if they are satisfied with the result. \
    If they are, you will end the conversation. \
    Use stop when the conversation is complete. \
    ONLY USE STOP WHEN THE USER SAYS THEY ARE DONE.

Function usage:
- take_picture: No parameters needed ONLY USE THIS ONCE
- set_goal: Requires prompts (string) - comma-separated list (after using this function you will be prompted again, no user input in between)
- show_grid: Requires triangle_ids (list of integers) - list of triangle IDs that you selected to direct robots to
- direct_robot: Requires triangle_id (int), robot_name (string), behavior ("MONITOR" or "SEARCH")
- stop: No parameters needed

Remember:
- When you call functions, you will be prompted again for the next step without user input in between.
- You may redirect the same robot as many times as needed.
- DO NOT DIRECT ROBOTS TO UNFEASIBLE TRIANGLES (e.g., walls, tables).
- CHECK WITH THE USER ABOUT THE TRIANGLE ID REASONING BEFORE DIRECTING ROBOTS
- YOU MUST SET A GOAL FOR THE ROBOT TO LOOK FOR EVEN FOR MONITORING BEHAVIOR
- Robot names follow the pattern: tb1, tb2, tb3, etc.
- REASON IN MULTIPLE STEPS: Provide a chain_of_thought array with multiple reasoning steps, breaking down your thinking process into distinct steps.

Always provide both text responses for conversation and appropriate function calls when needed.
"""
    def __init__(self):
        super().__init__('vlm_services')
        self.interface = OpenAIInterface(self.system_prompt, model="gpt-4.1", max_messages=100)
        self.get_logger().info('VLM Services Node has been started')

        self.conversation_state = False
        self.parsed_prompt = None
        self.declare_parameter('num_robots', 1)
        self.num_robots = self.get_parameter('num_robots').get_parameter_value().integer_value

        self.goal_publisher = self.create_publisher(String, 'robot_goal', 10)
        self.goal_timer = self.create_timer(2, self.timer_callback)

        self.is_image_processed = False

        self.global_cam_subscription = self.create_subscription(
            Image,
            '/global_cam/rgb_camera/image_color',
            self.image_callback,
            10
        )

        self.global_depth_subscription = self.create_subscription(
            Image,
            '/global_cam/depth_sensor/image',
            self.depth_callback,
            10
        )

        self.latest_depth = np.zeros((1, 1), dtype=np.float32)  # empty depth array as placeholder

        self.latest_image = np.zeros((1, 1, 3), dtype=np.uint8)  # empty black image as placeholder
        self.bridge = CvBridge()

        self.image_preprocess_callback = self.create_timer(0.1, self.image_preprocess)
        self.latest_preprocessed = np.zeros((1, 1, 3), dtype=np.uint8)  # empty black image as placeholder

        self.search_result_queue = queue.Queue()
        self.interface_lock = threading.Lock()

        self.robot_names_publisher = self.create_publisher(String, 'robot_names', 10)
        self.robot_names = [f'tb{i}' for i in range(1, self.num_robots + 1)]
        self.robot_gps_subs = {name: self.create_subscription(PointStamped, f'/{name}/p3d_gps', partial(self.gps_callback, name), 10) for name in self.robot_names}
        self.robot_orientation_subs = {name: self.create_subscription(Imu, f'/{name}/imu', partial(self.orientation_callback, name), 10) for name in self.robot_names}
        self.robot_positions = {name: {'x': 0.0, 'y': 0.0, 'z': 0.0, 'heading': 0.0} for name in self.robot_names}
        self.robot_path_publishers = {name: self.create_publisher(Path, f'/{name}/path', 10) for name in self.robot_names}

        # Start conversation in a separate thread so ROS can keep spinning
        self._conversation_thread = threading.Thread(target=self.conversation, daemon=True)
        self._conversation_thread.start()

        self.segmentation_model = SAM("sam2.1_b.pt")

        self.show_grid = False  # Flag to control grid display

        self.pib_pos = Vector7D(3.3786, 3.29366, 1.89995, 0.13050301753046564, 0.01717530230715774, -0.9912991331611769, 2.88204)
        self.fx = 686.9927350716014
        self.fy = 686.9927350716014
        self.cx = 640.0
        self.cy = 360.0

        self.interface.add_message("system", f"There are currently only {self.num_robots} robots available in the environment. \
        You can use the following robot names: {', '.join(self.robot_names)}.")

    def conversation(self):
        """
        Interactive conversation with GPT using structured output.
        """
        self.conversation_state = True

        while True:
            reminders = f"""
            - ONLY USE AS MANY ROBOTS AS NEEDED FOR THE TASK. TRY TO BE MINIMAL.
            - When you call functions, you will be prompted again for the next step without user input in between.
            - DO NOT DIRECT ROBOTS TO UNFEASIBLE TRIANGLES (e.g., walls, tables).
            - YOU MAY REDIRECT THE SAME ROBOT AS MANY TIMES AS NEEDED
            - ONLY ASK FOR CONFIRMATION ONCE FOR THE TRIANGLE IDS
            - DO NOT SHOW TRIANGLE ID NUMBERS, JUST THE IMAGE.
            - There are currently only {self.num_robots} robots available in the environment.
            - The robot names you may use are: {', '.join(self.robot_names)}.   
            - REASON IN MULTIPLE STEPS: Provide a chain_of_thought array with multiple reasoning steps, breaking down your thinking process into distinct steps.
            """
            with self.interface_lock:
                self.interface.add_message("system", reminders)
                
                # Get structured response
                response, model = self.interface.get_response()
            self.get_logger().info(f"GPT model used: {model}")
            # Display thinking
            if response and hasattr(response, 'chain_of_thought') and response.chain_of_thought:
                print(f"\n🤔 GPT thoughts:")
                for step in response.chain_of_thought:
                    print(f"  - {step.reasoning}")
            # Display the text response
            if response and hasattr(response, 'text') and response.text:
                print(f"\n🤖 GPT says: {response.text}")
            
            # Handle function calls
            if response and hasattr(response, 'take_picture') and response.take_picture:
                self.handle_take_picture()
                self.interface.add_message("system", "Picture taken successfully. Please continue.")
                continue
            if response and hasattr(response, 'set_goal') and response.set_goal:
                self.parse_prompt(response.set_goal.prompts)
                self.interface.add_message("system", f"Goal set successfully: {self.parsed_prompt}")
                continue
            if response and hasattr(response, 'show_grid') and response.show_grid:
                self.get_logger().info("Showing grid image to the user.")
                self.show_grid_image(response.show_grid.triangle_ids)
                self.interface.add_message("system", "Grid image displayed successfully.")
            if response and hasattr(response, 'direct_robot') and response.direct_robot:
                goal_assignments = {assignment.robot_name: assignment.triangle_id for assignment in response.direct_robot.assignments}
                self.plan_multi_agent(goal_assignments, weight=1.0, goal_wait=0, max_time=300)
                self.interface.add_message("system", "Directed robots to the specified triangles successfully.")
                continue
            if response and hasattr(response, 'stop') and response.stop:
                self.get_logger().info("Conversation ended by GPT.")
                break
                
            # If no function call, get user input
            user_reply = self.user_input_with_interrupt()
            with self.interface_lock:
                self.interface.add_message("user", user_reply)

        self.conversation_state = False

    def user_input_with_interrupt(self):
        """
        Get user input with a timeout to allow for periodic checks.
        """
        print(f"\n✏️ Your answer (type 'exit' to stop): ", end='', flush=True)
        enable_interrupt = True
        user_input = ""
        while True:
            # handle interrupt (queue being filled with search results)
            if not self.search_result_queue.empty() and enable_interrupt: 
                self.process_search_results()
                response, model = self.interface.get_response()  # Get the latest response to update the conversation
                # Display the text response
                if response and hasattr(response, 'text') and response.text:
                    print(f"\n🤖 GPT says: {response.text}")
                print(f"\n✏️ Your answer (type 'exit' to stop): ", end='', flush=True)
                enable_interrupt = False
            
            # Check if stdin has data available (non-blocking)
            if select.select([sys.stdin], [], [], 0.1)[0]:
                line = sys.stdin.readline()
                if line:
                    user_input = line.strip()
                    break
            
            # Small delay to prevent busy waiting
            time.sleep(0.1)
        
        return user_input

    def process_search_results(self):
        """Process any pending search results from the queue"""
        while not self.search_result_queue.empty():
            try:
                cv_image = self.search_result_queue.get_nowait()
                print(f"\n🔍 SEARCH UPDATE: Object detected by robot!")
                
                # Upload image and add to conversation
                image_file = self.upload_image_to_openai(cv_image)
                with self.interface_lock:
                    self.interface.add_message("user", [{"type": "input_image", "file_id": image_file.id}])
                    self.interface.add_message("user", "Image of potential goal object found by robot. Analyze this image and provide confirmation and semantic location of the object.")
            except queue.Empty:
                break
            except Exception as e:
                self.get_logger().error(f"Error processing search result: {e}")

    def handle_take_picture(self):
        """Handle take_picture function call"""
        self.get_logger().info("GPT requested to take a picture.")
        with self.interface_lock:
            if self.latest_image is not None:
                image_file = self.upload_image_to_openai(self.latest_image.copy())
                self.interface.add_message("user", [{"type": "input_image", "file_id": image_file.id}])

                image_file = self.upload_image_to_openai(self.latest_preprocessed.copy())
                self.interface.add_message("user", [{"type": "input_image", "file_id": image_file.id}])

                self.interface.add_message("user", "Use these images to answer the previous question. You have a grid image as well as the original image. The grid image has a grid overlay for your use.")
            else:
                self.interface.add_message("user", "No image available at the moment.")


    def parse_prompt(self, final_prompts: str) -> None:
        """
        This function will refine the user input prompt and return the final list of 10 prompts.
        """
        self.get_logger().info("Starting interactive prompt refinement...")

        self.parsed_prompt = final_prompts
        self.analyzed_prompt = True
        self.get_logger().info(f"Final parsed prompt: {self.parsed_prompt}")
        with self.interface_lock:
            self.interface.add_message("user", f"Goal set successfully: {self.parsed_prompt}")

    
    def timer_callback(self):
        if self.parsed_prompt is not None:
            self.goal_publisher.publish(String(data=self.parsed_prompt))
        if self.robot_names_publisher.get_subscription_count() > 0:
            self.robot_names_publisher.publish(String(data=','.join(list(self.robot_names))))

    def image_callback(self, msg: Image):
        """
        Callback to handle images from the global camera.
        """
        # self.get_logger().info("Received image from global camera.")
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    
    def found_objects_callback(self, msg: Image):
        """
        Callback to handle found objects from the detector.
        """
        try:
            # Convert ROS Image message to OpenCV image (BGR by default)
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.search_result_queue.put(cv_image)
        except Exception as e:
            self.get_logger().error(f"CV Bridge error in found objects callback: {e}")
    
    def depth_callback(self, msg: Image):
        """
        Callback to handle depth images from the global camera.
        """
        try:
            # Convert ROS Image message to OpenCV image (BGR by default)
            cv_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
            self.latest_depth = cv_depth.astype(np.float32)
        except Exception as e:
            self.get_logger().error(f"CV Bridge error in depth callback: {e}")

    def gps_callback(self, robot_name: str, msg: PointStamped):
        """
        Callback to handle GPS data for each robot.
        """
        if robot_name in self.robot_positions:
            self.robot_positions[robot_name]['x'] = msg.point.x
            self.robot_positions[robot_name]['y'] = msg.point.y
            self.robot_positions[robot_name]['z'] = msg.point.z
            # self.get_logger().info(f"Updated GPS for {robot_name}: {self.robot_positions[robot_name]}")
        else:
            self.get_logger().warn(f"Received GPS data for unknown robot: {robot_name}")
    
    def orientation_callback(self, robot_name: str, msg: Imu):
        """
        Callback to handle orientation data for each robot.
        """
        if robot_name in self.robot_positions:
            # Convert quaternion to yaw angle
            yaw = msg.orientation.z  # Assuming z is the yaw angle in radians
            self.robot_positions[robot_name]['heading'] = yaw

    def quaternion_to_yaw(self, q):
        """Convert quaternion to yaw angle in radians"""
        # First, normalize the quaternion
        norm = math.sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z)
        
        if norm < 1e-6:  # Handle near-zero quaternions
            self.get_logger().warn("Quaternion norm is nearly zero, using default orientation")
            return 0.0
        
        # Normalize quaternion components
        qw = q.w / norm
        qx = q.x / norm
        qy = q.y / norm  
        qz = q.z / norm
        
        # Convert normalized quaternion to yaw
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz)
        )
    
        return yaw

    def image_preprocess(self):
        """
        Preprocess the latest image for display.
        """
        if hasattr(self, 'adjacency_list'):
            return
        mask = np.load('src/llm_search/llm_search/temp/best_mask.npy')
        if (self.latest_image is not None and 
            mask is not None and 
            not self.is_image_processed and
            self.latest_depth.shape[0] > 1 and
            self.latest_image.shape[:2] == self.latest_depth.shape[:2]):
            # Resize latest_image to match mask dimensions
            resized_image = cv2.resize(self.latest_image, (mask.shape[1], mask.shape[0]))

            depth_array = self.latest_depth.copy()
            fx, fy, cx, cy = self.fx, self.fy, self.cx, self.cy
            pib_pos = self.pib_pos

            # Generate the mesh from the mask and depth information
            self.tri, self.tri_ids, self.verts, self.centroids_world_xy, self.polygon, self.mask_world_points = generate_mesh(mask, depth_array, fx, fy, cx, cy, pib_pos)
            self.get_logger().info(f"Generated mesh with {len(self.tri)} triangles and {len(self.verts)} vertices.")
            # Get the inverse mesh to get the triangle centers
            self.ids_proj, self.tri_proj, self.centroids_img, u, v = inverse_mesh(self.mask_world_points, self.verts, self.tri, self.tri_ids, fx, fy, cx, cy, pib_pos)

            # Create overlay image
            overlay = resized_image.copy()
            h, w = overlay.shape[:2]

            # Draw triangles
            for i, tri_indices in enumerate(self.tri_proj):
                # Get pixel coordinates for triangle vertices
                pts = np.stack([u[tri_indices], v[tri_indices]], axis=1)
                
                # Skip triangles with invalid projections
                if not np.all(np.isfinite(pts)):
                    continue
                
                pts_int = np.round(pts).astype(np.int32)
                
                # Skip triangles completely outside image bounds
                if (np.all(pts_int[:, 0] < 0) or np.all(pts_int[:, 0] >= w) or 
                    np.all(pts_int[:, 1] < 0) or np.all(pts_int[:, 1] >= h)):
                    continue
                
                # Draw triangle edges
                cv2.polylines(overlay, [pts_int.reshape(-1, 1, 2)], 
                            isClosed=True, color=(0, 255, 0), thickness=1)

            # Draw triangle IDs at centroids
            for i, triangle_id in enumerate(self.ids_proj):
                if i < len(self.centroids_img):
                    centroid = self.centroids_img[i]
                    if np.all(np.isfinite(centroid)):
                        x, y = int(round(centroid[0])), int(round(centroid[1]))
                        if 0 <= x < w and 0 <= y < h:
                            cv2.putText(overlay, str(int(triangle_id)), (x, y), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            
            # Save the overlay as preprocessed image
            self.latest_preprocessed = overlay
            self.is_image_processed = True

            self.adjacency_list = build_adjacency_list(
                self.polygon, self.tri, self.tri_ids, 
                self.centroids_img, self.centroids_world_xy
            )

            self.get_logger().info(f"Built adjacency list with {len(self.adjacency_list)} nodes.")
            # self.get_logger().info(f"Adjacency list: {self.adjacency_list}")
    
    def find_nearest_triangle(self, robot_name: str):
        """
        Find the nearest triangle to the robot's current position.
        """
        if robot_name not in self.robot_positions:
            self.get_logger().error(f"Robot {robot_name} not found in positions.")
            return None
        
        pos = self.robot_positions[robot_name]
        if not pos or 'x' not in pos or 'y' not in pos:
            self.get_logger().error(f"Invalid position for robot {robot_name}: {pos}")
            return None
        
        
        # Calculate distances to all triangle centroids
        distances = np.linalg.norm(self.centroids_world_xy - np.array([pos['x'], pos['y']]), axis=1)
        
        # Find the index of the nearest triangle
        tri_id = np.argmin(distances)

        self.get_logger().info(f"Nearest triangle for {robot_name} is triangle ID {tri_id} at distance {distances[tri_id]:.2f}.")

        return tri_id
    
    def plan_multi_agent(self, goal_assignments: dict, weight: float = 1.0, goal_wait: int = 0, max_time: int = 300):
        """
        goal_assignments: {robot_name: goal_triangle_id}
        weight: 1.0 => optimal CBS, >1.0 => ECBS bounded suboptimal (approx)
        goal_wait: extra wait steps allowed at goal (default = number of robots)
        max_time: hard cap on per-agent planning horizon
        """
        if not hasattr(self, 'adjacency_list'):
            self.get_logger().error("Adjacency list not built yet.")
            return
        robots = list(goal_assignments.keys())
        # Start triangles
        starts = {}
        for r in robots:
            tri = self.find_nearest_triangle(r)
            if tri is None:
                self.get_logger().error(f"Cannot find start triangle for {r}")
                return
            starts[r] = tri
        goals = goal_assignments

        if goal_wait is None:
            goal_wait = max(3, len(robots))  # small buffer

        planner = CBSPlanner(self.adjacency_list,
                             goal_wait=goal_wait,
                             weight=weight,
                             max_time=max_time,
                             logger=self.get_logger())

        success, paths = planner.solve(robots, starts, goals)
        if not success:
            self.get_logger().error("CBS planning failed to find conflict-free solution.")
            return

        # Publish each path as Path message (convert triangles to world coords)
        for r, tri_path in paths.items():
            path_msg = self.create_path_message(tri_path, r, behavior="monitor")
            self.robot_path_publishers[r].publish(path_msg)
        self.get_logger().info(f"CBS planning success. Published synchronized paths for {len(paths)} robots.")

        
    def create_path_message(self, triangle_path: list, robot_name: str, behavior: str = "default") -> Path:
        """
        Convert a list of triangle IDs to a ROS Path message.
        """
        path_msg = Path()
        path_msg.poses = []  # Ensure poses is a mutable list
        
        # Set header
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "map"  # or whatever your world frame is
        
        # Convert triangle IDs to world coordinates
        for triangle_id in triangle_path:
            # Find the world coordinates of this triangle's centroid
            if triangle_id is not None and triangle_id < len(self.centroids_world_xy):
                world_pos = self.centroids_world_xy[triangle_id]

                # Create PoseStamped for this waypoint
                pose_stamped = PoseStamped()
                pose_stamped.header.stamp = path_msg.header.stamp
                pose_stamped.header.frame_id = path_msg.header.frame_id
                
                # Set position
                pose_stamped.pose.position.x = float(world_pos[0])
                pose_stamped.pose.position.y = float(world_pos[1])
                pose_stamped.pose.position.z = 0.0  # Assuming ground level
                
                # Set orientation (facing toward next waypoint or default)
                pose_stamped.pose.orientation.x = 0.0
                pose_stamped.pose.orientation.y = 0.0
                pose_stamped.pose.orientation.z = 0.0
                pose_stamped.pose.orientation.w = 1.0  # Default orientation
                
                path_msg.poses.append(pose_stamped)
        
        # Add robot name and behavior as custom fields (if your navigator expects them)
        # You might need custom message types for this, or pass via parameter server
        
        self.get_logger().info(f"Created path message with {len(path_msg.poses)} waypoints for {robot_name}")
        return path_msg
    
    def show_grid_image(self, triangle_ids: list[int]):
        """
        Display the grid image in a separate window, highlighting selected cells.
        """
        if (self.latest_image is not None and 
            hasattr(self, 'tri_proj') and hasattr(self, 'centroids_img') and
            hasattr(self, 'ids_proj')):
            
            # Create a clean resized image to draw highlights on
            mask = np.load('src/llm_search/llm_search/temp/best_mask.npy')
            highlight_img = cv2.resize(self.latest_image, (mask.shape[1], mask.shape[0]))
            overlay = highlight_img.copy()

            # Use existing projected data - no need to recalculate
            for triangle_id in triangle_ids:
                # Find the triangle in tri_proj that corresponds to this triangle_id
                triangle_idx = None
                for i, proj_triangle_id in enumerate(self.ids_proj):
                    if proj_triangle_id == triangle_id:
                        triangle_idx = i
                        break
                
                if triangle_idx is not None and triangle_idx < len(self.centroids_img):
                    # Draw a red dot at the centroid
                    center = self.centroids_img[triangle_idx]
                    if np.all(np.isfinite(center)):
                        center_int = tuple(np.round(center).astype(int))
                        cv2.circle(highlight_img, center_int, 8, (0, 0, 255), -1)  # Red dot
                        # Optional: add a colored circle around it for better visibility
                        cv2.circle(highlight_img, center_int, 15, (0, 255, 0), 2)  # Green circle

            cv2.imshow("Grid Image", highlight_img)
            cv2.waitKey(0)
            cv2.destroyWindow("Grid Image")
        else:
            self.get_logger().error("Grid data not available for display.")
    

    def show_grid_persistent(self):
        """ 
        Display the grid image in a persistent window, updating it with new triangles, robot positions, and paths.
        """
        if self.latest_image is not None:
            # Create simple matplotlib plot of triangles in world frame
            if hasattr(self, 'tri') and hasattr(self, 'verts') and hasattr(self, 'tri_ids'):
                plt.figure("World Frame Triangles", figsize=(12, 8))
                plt.clf()  # Clear previous plot
                
                # Plot all triangles (lighter color to not overshadow robots)
                for i, tri_indices in enumerate(self.tri):
                    triangle_verts = self.verts[tri_indices]
                    triangle_closed = np.vstack([triangle_verts, triangle_verts[0]])
                    plt.plot(triangle_closed[:, 0], triangle_closed[:, 1], 'g-', linewidth=0.5, alpha=0.7)
                    
                    # Add triangle ID at centroid (smaller and lighter)
                    if hasattr(self, 'centroids_world_xy') and i < len(self.centroids_world_xy):
                        centroid = self.centroids_world_xy[i]
                        plt.text(centroid[0], centroid[1], str(self.tri_ids[i]), 
                                ha='center', va='center', fontsize=6, color='blue', alpha=0.6)
                
                # Plot robot positions with enhanced visualization
                robot_colors = ['red', 'orange', 'purple', 'brown', 'pink', 'cyan', 'magenta', 'yellow']
                robot_info = []
                
                for idx, (robot_name, pos) in enumerate(self.robot_positions.items()):
                    if pos and 'x' in pos and 'y' in pos:
                        color = robot_colors[idx % len(robot_colors)]
                        
                        # Draw robot as a larger circle with border
                        plt.scatter(pos['x'], pos['y'], s=150, c=color, marker='o', 
                                  edgecolors='black', linewidth=3, label=robot_name, zorder=10)
                        
                        # Add robot name as text with background
                        plt.text(pos['x'] + 0.15, pos['y'] + 0.15, robot_name, 
                                fontsize=12, fontweight='bold', color='white',
                                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))
                        # Add orientation arrow if heading data is available
                        if 'heading' in pos and pos['heading'] is not None:
                            heading = pos['heading']
                            arrow_length = 0.4
                            dx = arrow_length * np.cos(heading)
                            dy = arrow_length * np.sin(heading)
                            plt.arrow(pos['x'], pos['y'], dx, dy,
                                    head_width=0.08, head_length=0.08, 
                                    fc=color, ec='black', alpha=0.9, linewidth=2)
                        # Find and highlight the closest triangle
                        if hasattr(self, 'centroids_world_xy'):
                            distances = np.linalg.norm(self.centroids_world_xy - np.array([pos['x'], pos['y']]), axis=1)
                            closest_triangle_idx = np.argmin(distances)
                            closest_centroid = self.centroids_world_xy[closest_triangle_idx]
                            
                            # Draw line to closest triangle
                            plt.plot([pos['x'], closest_centroid[0]], [pos['y'], closest_centroid[1]], 
                                    '--', color=color, alpha=0.5, linewidth=2)
                            
                            # Highlight the closest triangle
                            closest_tri_indices = self.tri[closest_triangle_idx]
                            triangle_verts = self.verts[closest_tri_indices]
                            triangle_closed = np.vstack([triangle_verts, triangle_verts[0]])
                            plt.fill(triangle_closed[:, 0], triangle_closed[:, 1], 
                                    color=color, alpha=0.2, edgecolor=color, linewidth=2)
                        
                        # Store robot info for status display
                        robot_info.append(f"{robot_name}: ({pos['x']:.2f}, {pos['y']:.2f})")
                
                # Add status text
                status_text = "Robot Positions:\n" + "\n".join(robot_info)
                plt.text(0.02, 0.98, status_text, transform=plt.gca().transAxes, 
                        verticalalignment='top', fontsize=10, 
                        bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
                
                plt.axis('equal')
                plt.xlabel('X World (meters)')
                plt.ylabel('Y World (meters)')
                plt.title('Floor Triangulation Grid with Robot Positions (World Frame)')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.draw()
                plt.pause(0.001)  # Non-blocking update
                    

    def upload_image_to_openai(self, image_cv2: np.ndarray):
        """
        Upload an image to OpenAI and return the file ID.
        """
        # Convert the latest image to a format suitable for OpenAI API
        success, encoded_image = cv2.imencode('.jpg', image_cv2)
        if not success:
            self.get_logger().error("Failed to encode image.")
            raise ValueError("Failed to encode image.")

        # Create a temporary file with proper extension
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            temp_file.write(encoded_image.tobytes())
            temp_file_path = temp_file.name
        
        # Upload the file to OpenAI
        with open(temp_file_path, 'rb') as f:
            image_file = self.interface.client.files.create(file=f, purpose='vision')
        
        return image_file
    
    

    

def main():
    rclpy.init()
    node = VLMServices()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            node.show_grid_persistent() 
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()