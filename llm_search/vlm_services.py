import rclpy
from rclpy.node import Node
import time
import threading
from openai import OpenAI
import json, cv2
from llm_search.utils.openai_interface import OpenAIInterface
# from llm_search_interfaces.srv import Analysis
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
# from ament_index_python.packages import get_package_share_directory


class VLMServices(Node):
    
    system_prompt = """
You are an AI assistant that helps users use a multi-robot search and monitoring system in a simulation environment.
Your task is to interactively refine the user's input prompt and provide structured output for spawning robots in the simulation.

You communicate through structured responses that include both text and function calls. Your responses should always include:
- text: Your conversational response to the user
- Optional function calls: take_picture, set_goal, spawn_robot, or stop

Each interaction should follow these general steps:

1. Ask the user for the object they want the robot to search or monitor for. Simplify these into a list of 3 simple prompts.
   Also make sure the user specifies the behavior of the robot (monitor or search).
   Example: "a telephone on the floor next to a cup" → "telephone, telephone next to a cup, cup"

2. Use the take_picture function to see the environment with a grid overlay. The grid will be numbered 0-149.
   You can use this image to help you understand the environment and choose spawn locations.

3. Choose suitable grid cells to spawn robots using the spawn_robot function. Confer with the user
   to determine if the chosen cells are good for spawning the robot. DO NOT CHOOSE OCCLUDED AREAS. 
   DO NOT SPAWN MORE THAN ONE ROBOT IN THE SAME LOCATION. 

4. Use set_goal to set the search prompts for all robots.

5. Use stop when the conversation is complete. ONLY USE STOP WHEN THE USER SAYS THEY ARE DONE.

Function usage:
- take_picture: No parameters needed
- set_goal: Requires prompts (string) - comma-separated list
- spawn_robot: Requires grid_cell (int 0-149), robot_name (string), behavior ("MONITOR" or "SEARCH")
- stop: No parameters needed

Remember:
- DO NOT SPAWN ROBOTS IN OCCLUDED AREAS
- DO NOT SPAWN MORE THAN ONE ROBOT IN THE SAME LOCATION  
- YOU MAY ONLY SPAWN THREE ROBOTS
- CHECK WITH THE USER ABOUT THE GRID CELL NUMBER BEFORE SPAWNING
- DO NOT SPAWN A ROBOT AGAIN AFTER IT HAS BEEN SPAWNED
- YOU MUST SET A GOAL FOR THE ROBOT TO LOOK FOR EVEN FOR MONITORING BEHAVIOR

Always provide both text responses for conversation and appropriate function calls when needed.
"""
    def __init__(self):
        super().__init__('vlm_services')
        self.interface = OpenAIInterface(self.system_prompt, model="gpt-4o", max_messages=100)
        self.get_logger().info('VLM Services Node has been started')

        self.conversation_state = False
        self.parsed_prompt = None
        self.declare_parameter('input_prompt', 'User forgot to specify input prompt, begin the conversation with the user.')
        self.input_prompt = self.get_parameter('input_prompt').get_parameter_value().string_value

        self.goal_publisher = self.create_publisher(String, 'robot_goal', 10)
        self.goal_timer = self.create_timer(2, self.timer_callback)

        self.global_cam_subscription = self.create_subscription(
            Image,
            '/global_cam/rgb_camera/image_color',
            self.image_callback,
            10
        )

        self.latest_image = np.zeros((1, 1, 3), dtype=np.uint8)  # empty black image as placeholder
        self.bridge = CvBridge()

        self.image_preprocess_callback = self.create_timer(0.1, self.image_preprocess)
        self.latest_preprocessed = np.zeros((1, 1, 3), dtype=np.uint8)  # empty black image as placeholder

        # Start conversation in a separate thread so ROS can keep spinning
        self._conversation_thread = threading.Thread(target=self.conversation, daemon=True)
        self._conversation_thread.start()

        self.spawn_service = self.create_client(SpawnNodeFromString, '/Ros2Supervisor/spawn_node_from_string')
        self.req = SpawnNodeFromString.Request()

        # self.grid_size = 75  # size of each grid box in pixels

        self.num_cols = 15
        self.num_rows = 10

        self.boundaries = {
            "bottom_right": (6.65, -2.88),
            "top_left": (-6.15, 4.77)
        }

        self.spawned_robots = set()  # Keep track of spawned robots to avoid duplicates

    def conversation(self):
        """
        Interactive conversation with GPT using structured output.
        """
        self.conversation_state = True
        self.interface.add_message("user", self.input_prompt)

        while True:
            reminders = """
            DO NOT SPAWN ROBOTS IN OCCLUDED AREAS. DO NOT SPAWN MORE THAN ONE ROBOT IN THE SAME LOCATION. YOU MAY ONLY SPAWN THREE ROBOTS.
            CHECK WITH THE USER ABOUT THE GRID CELL NUMBER BEFORE SPAWNING THE ROBOT.
            DO NOT SPAWN A ROBOT AGAIN AFTER IT HAS BEEN SPAWNED. DO NOT SPAWN IN AN OCCLUDED AREA.
            YOU MUST SET A GOAL FOR THE ROBOT TO LOOK FOR EVEN FOR MONITORING BEHAVIOR.
            """
            self.interface.add_message("system", reminders)
            
            # Get structured response
            response = self.interface.get_response()
            
            # Display the text response
            if response and hasattr(response, 'text') and response.text:
                print(f"\n🤖 GPT says: {response.text}")
            
            # Handle function calls
            if response and hasattr(response, 'take_picture') and response.take_picture:
                self.handle_take_picture()
                continue
            elif response and hasattr(response, 'set_goal') and response.set_goal:
                self.handle_set_goal(response.set_goal.prompts)
                continue
            elif response and hasattr(response, 'spawn_robot') and response.spawn_robot:
                self.handle_spawn_robot(
                    response.spawn_robot.grid_cell,
                    response.spawn_robot.robot_name,
                    response.spawn_robot.behavior.value
                )
                continue
            elif response and hasattr(response, 'stop') and response.stop:
                self.get_logger().info("Conversation ended by GPT.")
                break
                
            # If no function call, get user input
            user_reply = input("\n✏️ Your answer: ")
            self.interface.add_message("user", user_reply)
        
        self.conversation_state = False

    def handle_take_picture(self):
        """Handle take_picture function call"""
        self.get_logger().info("GPT requested to take a picture.")
        if self.latest_image is not None:
            image_file = self.upload_image_to_openai(self.latest_image.copy())
            self.interface.add_message("user", [{"type": "input_image", "file_id": image_file.id}])

            image_file = self.upload_image_to_openai(self.latest_preprocessed.copy())
            self.interface.add_message("user", [{"type": "input_image", "file_id": image_file.id}])

            self.interface.add_message("user", "Use these images to answer the previous question. You have a grid image as well as the original image. The grid image has a grid overlay for your use.")
        else:
            self.interface.add_message("user", "No image available at the moment.")

    def handle_set_goal(self, prompts):
        """Handle set_goal function call"""
        self.parse_prompt(prompts)
        self.interface.add_message("user", f"Goal set successfully: {prompts}")

    def handle_spawn_robot(self, grid_cell, robot_name, behavior):
        """Handle spawn_robot function call and check if robot has already been spawned"""
        if robot_name in self.spawned_robots:
            self.interface.add_message("user", f"ERROR: Robot {robot_name} has already been spawned!")
            return
        spawn_x, spawn_y = self.get_coords_from_grid(grid_cell)
        position = f"{spawn_x} {spawn_y} -0.0065"
        self.spawn_robot(robot_name, position, behavior)
        self.spawned_robots.add(robot_name)

    def parse_prompt(self, final_prompts: str) -> None:
        """
        This function will refine the user input prompt and return the final list of 10 prompts.
        """
        self.get_logger().info("Starting interactive prompt refinement...")

        self.parsed_prompt = final_prompts
        self.analyzed_prompt = True
        self.get_logger().info(f"Final parsed prompt: {self.parsed_prompt}")

    
    def timer_callback(self):
        if self.parsed_prompt is not None:
            self.goal_publisher.publish(String(data=self.parsed_prompt))
    
    def image_callback(self, msg: Image):
        """
        Callback to handle images from the global camera.
        """
        # self.get_logger().info("Received image from global camera.")
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def image_preprocess(self):
        """
        Preprocess the latest image for display.
        """
        if self.latest_image is not None and self.conversation_state:
            grid_image = self.latest_image.copy()
            height, width, _ = grid_image.shape
            cell_width = width // self.num_cols
            cell_height = height // self.num_rows

            # draw grid and number the cells
            cell_number = 0
            num_cols = self.num_cols
            num_rows = self.num_rows
            for row in range(num_rows):
                for col in range(num_cols):
                    x = int(col * cell_width)
                    y = int(row * cell_height)
                    top_left = (x, y)
                    bottom_right = (min(x + cell_width, width - 1), min(y + cell_height, height - 1))
                    cv2.rectangle(grid_image, top_left, bottom_right, (255, 0, 0), 1)
                    # Calculate the center of the grid cell for numbering
                    center_x = x + (min(cell_width, width - x) // 2)
                    center_y = y + (min(cell_height, height - y) // 2)
                    cv2.putText(
                        grid_image,
                        str(cell_number),
                        (center_x - 10, center_y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA
                    )
                    cell_number += 1

            self.latest_preprocessed = grid_image.copy()

    def show_grid_image(self):
        """
        Display the grid image in a separate window.
        """
        if self.latest_preprocessed is not None:
            cv2.imshow("Grid Image", self.latest_preprocessed)
            cv2.waitKey(1)

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

    def spawn_robot(self, robot_name: str, position: str, behavior: str):
        """
        Spawn a robot in the simulation using the SpawnNodeFromString service.
        """
        try:
            self.get_logger().info(f"Spawning robot {robot_name} at position {position}...")
            data_string = "Turtlebot4 {name \"" + robot_name + "\" translation " + position + " controller \"<extern>\"}"
            self.req.data = data_string
            self.get_logger().info(f"Requesting spawn with data: {self.req}")
            self.future = self.spawn_service.call_async(self.req)
            self.launch_ros2_file('llm_search', 
                                'spawn_robot.py', 
                                {'robot_name': robot_name, 
                                    'robot_speed': 0.0, 
                                    'robot_turn_speed': 0.7, 
                                    'behavior': behavior})
            self.interface.add_message("user", f"Robot {robot_name} has been spawned at position {position}.")
        except Exception as e:
            self.get_logger().error(f"Failed to spawn robot {robot_name} at position {position}: {e}")
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
            node.show_grid_image()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()