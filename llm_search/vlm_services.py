from urllib import response
import rclpy
from rclpy.node import Node
import time
import threading, queue, sys, select
from openai import OpenAI
import json, cv2
from llm_search.utils.openai_interface import OpenAIInterface
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import String
import tempfile
import subprocess
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from webots_ros2_msgs.srv import SpawnNodeFromString
import re
import scipy
from scipy.spatial import Delaunay
from scipy.spatial.transform import Rotation as R
from ultralytics import SAM
from llm_search.utils.vector import Vector7D

class VLMServices(Node):
    
    system_prompt = """
You are given a view of the scene from the perspective of a human. The environment can be deployed with multiple TurtleBots. You are a centralised scene planner.\
Semantically analyze the scene for environment space, objects, and their relationships to provide structured responses.

You can perform two tasks with TurtleBots in the scene: multi-robot search and multi-robot monitoring in the given scene. \
First ask/Analyse what behavior the user wants (monitor or search).For each task, \
you interactively refine the user's input prompt and provide structured output to meaningfully allocate robots in the scene.


Each task should follow this general step:
1. Ask the user for their specific goal or task they want to achieve with the robots.

2. Use the take_picture function to see the environment with a grid overlay. The grids are numbered. \
   Based on the user input, analyze this image to understand the scene layout and objects. \
   ROBOT ALLOCATION: Identify feasible and meaningful locations(grids) to spawn robots in the scene. \
   Reason how many grids are allocated and why those chosen grid cells are suitable for the task. \
   Try to spread the robots out into different relevant areas to maximize coverage. \
   CHOOSE ONLY GRID CELLS THAT ARE FLOORS. \
   THE MOBILE ROBOTS CAN ONLY BE SPAWNED ON THE FLOOR. \
   MORE THAN ONE ROBOT CANNOT BE SPAWNED IN THE SAME GRID. 

3. Confirm with the user about the grid cells you have chosen. Use show_grid to show the grid image. DO NOT SHOW THE GRID CELL NUMBERS, JUST THE IMAGE. \
   **After the user confirms the location, you MUST use the spawn_robot function. Do NOT use show_grid again after getting confirmation.**

For multi-robot monitoring:
1. Monitoring Goal: Ask the user if they are monitoring for anything particular, say an object or a person if not mentioned. \
    Generate the monitoring goal into a list of 3 simple prompts.\
    Do not output prompts that are too general or indicate a general space. 
    GOOD: "I want to monitor near door for a package" → "Box, Package, carboard Box"
    Finally, Use set_goal to set the search prompts for all robots.

2. If one of the robots find the goal object, initiate a conversation with the user. You will be given an image of the monitoring object.
   Analyze the image and report the semantic location and the description of the object found (no coordinates).

3. Ask the user if they are satisfied with the result. \
    If they are, you will end the conversation. \
    Use stop when the conversation is complete. \
    ONLY USE STOP WHEN THE USER SAYS THEY ARE DONE.

Function usage:
- take_picture: No parameters needed
- set_goal: Requires prompts (string) - comma-separated list
- show_grid: Requires grid_cells (list of integers) - list of grid cells that you selected to spawn robots in
- spawn_robot: Requires grid_cell (int), robot_name (string), behavior ("MONITOR" or "SEARCH")
- stop: No parameters needed

Remember:
- DO NOT SPAWN ROBOTS IN UNFEASIBLE GRIDS (e.g., walls, tables).
- CHECK WITH THE USER ABOUT THE GRID CELL REASONING BEFORE SPAWNING
- DO NOT SPAWN A ROBOT AGAIN AFTER IT HAS BEEN SPAWNED
- YOU MUST SET A GOAL FOR THE ROBOT TO LOOK FOR EVEN FOR MONITORING BEHAVIOR
- REASON IN MULTIPLE STEPS: Provide a chain_of_thought array with multiple reasoning steps, breaking down your thinking process into distinct steps.

Always provide both text responses for conversation and appropriate function calls when needed.
"""
    def __init__(self):
        super().__init__('vlm_services')
        self.interface = OpenAIInterface(self.system_prompt, model="gpt-4.1-nano", max_messages=100)
        self.get_logger().info('VLM Services Node has been started')

        self.conversation_state = False
        self.parsed_prompt = None
        self.declare_parameter('input_prompt', 'User forgot to specify input prompt, begin the conversation with the user.')
        self.input_prompt = self.get_parameter('input_prompt').get_parameter_value().string_value

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

        # Start conversation in a separate thread so ROS can keep spinning
        self._conversation_thread = threading.Thread(target=self.conversation, daemon=True)
        self._conversation_thread.start()

        self.spawn_service = self.create_client(SpawnNodeFromString, '/Ros2Supervisor/spawn_node_from_string')
        self.req = SpawnNodeFromString.Request()

        self.num_cols = 15
        self.num_rows = 10

        self.boundaries = {
            "bottom_right": (6.65, -2.88),
            "top_left": (-6.15, 4.77)
        }

        self.spawned_robots = set()  # Keep track of spawned robots to avoid duplicates
        self.detection_subscriptions = list()  # List to keep track of detection subscriptions

        self.robot_names_publisher = self.create_publisher(String, 'robot_names', 10)

        self.segmentation_model = SAM("sam2.1_b.pt")

        self.show_grid = False  # Flag to control grid display

        self.pib_pos = Vector7D(3.3786, 3.29366, 1.89995, 0.13050301753046564, 0.01717530230715774, -0.9912991331611769, 2.88204)
        self.fx = 686.9927350716014
        self.fy = 686.9927350716014
        self.cx = 640.0
        self.cy = 360.0

    def conversation(self):
        """
        Interactive conversation with GPT using structured output.
        """
        self.conversation_state = True
        with self.interface_lock:
            self.interface.add_message("user", self.input_prompt)

        while True:
            reminders = """
            - DO NOT SPAWN ROBOTS IN UNFEASIBLE GRIDS (e.g., walls, tables).
            - CHECK WITH THE USER ABOUT THE GRID CELL REASONING BEFORE SPAWNING
            - DO NOT SPAWN A ROBOT AGAIN AFTER IT HAS BEEN SPAWNED
            - YOU MUST SET A GOAL FOR THE ROBOT TO LOOK FOR EVEN FOR MONITORING BEHAVIOR
            - ONLY ASK FOR CONFIRMATION ONCE FOR THE GRID CELLS
            - DO NOT SHOW GRID CELL NUMBERS, JUST THE IMAGE.
            - REASON IN MULTIPLE STEPS: Provide a chain_of_thought array with multiple reasoning steps, breaking down your thinking process into distinct steps.
            """
            with self.interface_lock:
                self.interface.add_message("system", reminders)
                
                # Get structured response
                response = self.interface.get_response()
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
                continue
            elif response and hasattr(response, 'set_goal') and response.set_goal:
                self.parse_prompt(response.set_goal.prompts)
                continue
            elif response and hasattr(response, 'show_grid') and response.show_grid:
                self.get_logger().info("Showing grid image to the user.")
                self.show_grid_image(response.show_grid.grid_cells)
            elif response and hasattr(response, 'spawn_robot') and response.spawn_robot:
                self.spawn_robot(
                    response.spawn_robot.robot_name,
                    response.spawn_robot.grid_cell,
                    response.spawn_robot.behavior.value
                )
                continue
            elif response and hasattr(response, 'stop') and response.stop:
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
                response = self.interface.get_response()  # Get the latest response to update the conversation
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
            self.robot_names_publisher.publish(String(data=','.join(list(self.spawned_robots))))

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

    def image_preprocess(self):
        """
        Preprocess the latest image for display.
        """
        mask = np.load('src/llm_search/llm_search/best_mask.npy')
        if (self.latest_image is not None and 
            mask is not None and 
            not self.is_image_processed and
            self.latest_depth.shape[0] > 1 and
            self.latest_image.shape[:2] == self.latest_depth.shape[:2]):
            # Resize latest_image to match mask dimensions
            resized_image = cv2.resize(self.latest_image, (mask.shape[1], mask.shape[0]))

            # best_mask: binary mask from SAM+CLIP
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Usually there is only 1 large contour, but you can pick the biggest
            floor_contour = max(contours, key=cv2.contourArea)

            epsilon = 0.01 * cv2.arcLength(floor_contour, True)
            approx = cv2.approxPolyDP(floor_contour, epsilon, True)
            # Convert polygon points to Nx2 numpy array
            polygon = approx.reshape(-1, 2)

            # bounding box
            min_x, min_y = polygon.min(axis=0)
            max_x, max_y = polygon.max(axis=0)

            step = 115  # smaller step → denser grid → more triangles

            # Create a grid of points within the bounding box
            # This will create a grid of points spaced by 'step' pixels
            # and only include points that are inside the polygon
            grid_points = []
            for y in range(min_y, max_y, step):
                for x in range(min_x, max_x, step):
                    if cv2.pointPolygonTest(polygon, (x,y), False) >= 0:
                        grid_points.append([x,y])

            grid_points = np.array(grid_points)

            # Convert polygon points to Nx2 numpy array
            all_points = np.vstack([polygon, grid_points])

            # Perform Delaunay triangulation
            self.tri = Delaunay(all_points)

            # compute the centroids of each triangle
            self.triangle_centers = np.array([np.mean(all_points[simplex], axis=0) for simplex in self.tri.simplices])

            # get the coordinates of each centroid in world coordinates
            world_points = []
            valid_simplices = []
            valid_centers = []
            for i, simplex in enumerate(self.tri.simplices):
                centroid = np.mean(all_points[simplex], axis=0)
                u, v = centroid
                depth = self.latest_depth[int(v), int(u)]
                if np.isnan(depth) or depth <= 0 or depth > 10:  # Adjust threshold as needed
                    continue  # Skip invalid depth values

                # The original deprojection assumed a standard CV camera frame (Z-fwd, X-right, Y-down).
                # However, many robotics systems and simulators use a different frame (X-fwd, Y-left, Z-up).
                # We'll convert the pixel coordinates into this robotics-standard frame.
                X_cam = depth
                Y_cam = -(u - self.cx) * depth / self.fx
                Z_cam = -(v - self.cy) * depth / self.fy
                P_camera = np.array([X_cam, Y_cam, Z_cam])

                # Convert axis-angle to rotation matrix
                axis = np.array([self.pib_pos.x_axis, self.pib_pos.y_axis, self.pib_pos.z_axis])
                angle = self.pib_pos.rotation
                r = R.from_rotvec(axis * angle)
                R_matrix = r.as_matrix()

                # Camera position in world coordinates
                T = np.array([self.pib_pos.x, self.pib_pos.y, self.pib_pos.z])

                # Transform point from camera to world coordinates
                P_world = R_matrix @ P_camera + T

                if P_world[2] < 0.2:
                    world_points.append(P_world)
                    valid_simplices.append(simplex)
                    valid_centers.append(centroid)

            # Draw filtered triangles
            for simplex in valid_simplices:
                pts = all_points[simplex].astype(int)
                cv2.polylines(resized_image, [pts], isClosed=True, color=(255, 0, 255), thickness=2)

            # Draw numbers for filtered triangles
            for i, center in enumerate(valid_centers):
                c = tuple(np.round(center).astype(int))
                cv2.putText(
                    resized_image,
                    str(i),
                    c,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # smaller font size
                    (0, 255, 255),
                    1,  # thinner line for less bold
                    cv2.LINE_AA
                )

            self.world_points = np.array(world_points)
            self.triangle_centers = np.array(valid_centers)
            self.tri.simplices = np.array(valid_simplices)

            # Store the latest preprocessed image
            self.latest_preprocessed = resized_image.copy()

            self.is_image_processed = True
            

    def show_grid_image(self, grid_cells: list[int]):
        """
        Display the grid image in a separate window, highlighting selected cells.
        """
        if self.latest_image is not None and self.tri is not None:
            # Create a clean resized image to draw highlights on
            mask = np.load('src/llm_search/llm_search/best_mask.npy')
            highlight_img = cv2.resize(self.latest_image, (mask.shape[1], mask.shape[0]))
            overlay = highlight_img.copy()

            for grid_cell in grid_cells:
                if 0 <= grid_cell < len(self.tri.simplices):
                    # Get the simplex (triangle) for the given grid cell index
                    simplex = self.tri.simplices[grid_cell]
                    # Get the points of the triangle
                    pts = self.tri.points[simplex].astype(int)
                    # Draw a filled polygon on the overlay
                    cv2.fillPoly(overlay, [pts], (0, 255, 0))  # Green highlight

                    # Draw a dot in the center of the polygon
                    center = self.triangle_centers[grid_cell]
                    cv2.circle(highlight_img, tuple(np.round(center).astype(int)), 5, (0, 0, 255), -1) # Red dot

            # Blend the overlay with the original image
            alpha = 0.4  # Transparency factor
            cv2.addWeighted(overlay, alpha, highlight_img, 1 - alpha, 0, highlight_img)

            cv2.imshow("Grid Image", highlight_img)
            # self.get_logger().info("Press any key to close the grid image window and move to user prompt.")
            cv2.waitKey(0)
            cv2.destroyWindow("Grid Image")
            

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
    
    def get_coords_from_grid(self, grid_num: int) -> tuple:
        img_h, img_w = self.latest_image.shape[:2]
        num_cols = self.num_cols
        num_rows = self.num_rows

        col = grid_num % num_cols
        row = grid_num // num_cols

        world_min_x = self.boundaries["top_left"][0]
        world_max_x = self.boundaries["bottom_right"][0]
        world_max_y = self.boundaries["top_left"][1]
        world_min_y = self.boundaries["bottom_right"][1]

        world_interval_x = (world_max_x - world_min_x) / num_cols
        world_interval_y = (world_max_y - world_min_y) / num_rows

        spawn_x = world_min_x + (col + 0.5) * world_interval_x
        spawn_y = world_max_y - (row + 0.5) * world_interval_y

        return spawn_x, spawn_y

    def spawn_robot(self, robot_name: str, grid_cell: int, behavior: str):
        """
        Spawn a robot in the simulation using the SpawnNodeFromString service.
        """
        if robot_name in self.spawned_robots:
            with self.interface_lock:
                self.interface.add_message("user", f"ERROR: Robot {robot_name} has already been spawned!")
            return
        
        if grid_cell < 0 or grid_cell >= len(self.world_points):
            self.get_logger().error(f"Invalid grid_cell {grid_cell} provided for spawning.")
            with self.interface_lock:
                self.interface.add_message("user", f"ERROR: Invalid grid cell {grid_cell}. Please choose a valid grid cell from the image.")
            return
        
        spawn_x, spawn_y, spawn_z = self.world_points[grid_cell]
        position = f"{spawn_x} {spawn_y} {spawn_z}"
        
        try:
            data_string = "Turtlebot4 {name \"" + robot_name + "\" translation " + position + " controller \"<extern>\"}"
            self.req.data = data_string
            self.future = self.spawn_service.call_async(self.req)
            self.launch_ros2_file('llm_search', 
                                'spawn_robot.py', 
                                {'robot_name': robot_name, 
                                    'robot_speed': 0.0, 
                                    'robot_turn_speed': 0.7, 
                                    'behavior': behavior})
            # Add the robot to the list of spawned robots and also subscribe to its detection topic
            self.spawned_robots.add(robot_name)
            self.detection_subscriptions.append(self.create_subscription(
                Image,
                f'/{robot_name}/detector/found',
                self.found_objects_callback,
                10
            ))
            with self.interface_lock:
                self.interface.add_message("user", f"Robot {robot_name} has been spawned at position {position}.")
            self.get_logger().info(f"Robot {robot_name} spawned at position {position} with behavior {behavior}.")
        except Exception as e:
            self.get_logger().error(f"Failed to spawn robot {robot_name} at position {position}: {e}")
            with self.interface_lock:
                self.interface.add_message("user", f"Error in spawning robot {robot_name}. Please check the format and try again.")
            return


    def handle_spawn_response(self, future, robot_name, position):
        result = future.result
        if result.success:
            self.get_logger().info(f"Spawn service response: {result}")
            self.interface.add_message("user", f"Robot {robot_name} has been spawned at position {position}.")
        else:
            self.get_logger().error(f"Failed to spawn robot {robot_name}. Response: {result}")
            self.interface.add_message("user", f"Error in spawning robot {robot_name}. Please check the format and try again.")
        
            
    def launch_ros2_file(self, package, launch_file, args=None):
        cmd = ['ros2', 'launch', package, launch_file]
        if args:
            for k, v in args.items():
                cmd.append(f'{k}:={v}')
        subprocess.Popen(['gnome-terminal', '--', *cmd])

def main():
    rclpy.init()
    node = VLMServices()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            #node.show_grid_image(response.show_grid.grid_cells)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()