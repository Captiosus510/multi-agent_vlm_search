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

from sensor_msgs.msg import Image
from webots_ros2_msgs.srv import SpawnNodeFromString
import re
# from ament_index_python.packages import get_package_share_directory


class VLMServices(Node):
    """
    This will publish the goal for the robot to a topic and provide a double check service to confirm if the robot has found the object or not.
    It will use the OpenAI API to analyze the image and confirm if the object is indeed.
    """
    system_prompt = """

        You are part of a ROS2 robot framework to carry out multi-robot tasks. \
        The multi-robot tasks are "monitoring environment" and "searching for an object" in an environment.

        The idea is to have multiple robots either to surveil or explore an environment looking for objects asked by the user,
        and report back to a central node. \
        You are the one who will provide them with information about their tasks.
        You will be interfacing with the user and potentially issue updates to the multi robot system as the search progresses.

        You have these functionalities for now:

        1. You will have a global bird's eye view camera that will provide you with images of the environment.
            If you want to take a picture, send a message back while
            that says "TAKE PICTURE" and the system will send a picture to you. 
            IT IS VERY IMPORTANT THAT YOU ONLY ASK FOR A PICTURE WHEN YOU ARE READY TO ANALYZE IT.
            IT IS ALSO VERY IMPORTANT THAT THE REQUEST MESSAGE IS "TAKE PICTURE" EXACTLY LIKE THIS, 
            OTHERWISE THE SYSTEM WILL NOT WORK.
        
        2.  You will also be able to spawn robots in the environment. 
            You can spawn robots at specific positions in the environment. 
            When you take a picture, you will also be given a map with grid cells. These grid cells should be numbered in increasing order along the rows. 
            More specifically, you should number the grid cells from left to right, starting from the top left corner of the image. 
            The grid cells should be numbered starting from 0, and the numbering should increase by 1 for each cell. 

            The robots will spawn in the middle of the grid cell.
            
            When you choose a number for a grid cell, you should use the following format:
            SPAWN ROBOT AT: <cell_number> <robot_name>
            where <cell_number> is the number of the grid cell you want to spawn the robot in, and <robot_name> is the name of the robot you want to spawn. 
            The <cell_number> should be an integer, and the <robot_name> should be a string without spaces. 
            For example, if you want to spawn a robot named "my_robot" in the grid cell number 5, you should send the message:
            SPAWN ROBOT AT: 5 my_robot

            IT IS VERY IMPORTANT THAT YOU FOLLOW THIS FORMAT EXACTLY, OTHERWISE THE SYSTEM WILL NOT WORK. 
            
            If you do not follow this format, the robot will NOT be spawned. 

            DO NOT SPAWN A ROBOT IN AN OCCLUDED AREA.
        
        3. The user may want to set a goal for the robot to search for an object.
            You will be given a prompt from the user that describes an object to look for.

            Simply take this prompt and simplify to a few words that describe the object.
            For example, if the user inputs "a silver cat at home", you could return a list of
            silver cat, cat at home, cat on wooden floor

            MAKE SURE IT's A COMMA SEPARATED LIST OF PROMPTS, NOT A SINGLE STRING. YOU MAY RETURN UP TO 3 PROMPTS.
            The robot will use these prompts to search for the object in the environment.

        """

        # For now, you won't have to coordinate any robots directly. You do have the ability to converse with the user.
        
        # Here are your two primary objectives as a ROS2 Node:
#         1. You will be given a prompt from the user that describes an object to look for. We want to generate a semantic map of the environment using this goal prompt.
#         Each robot has SigLip, a vision-language model, that will be used to compute the confidence of the robot in finding the object in the frame it is looking at. 
#         SigLip benefits from having multiple descriptive prompts, so you should take what the user inputs and return a list of 2 related prompts that describe the object.
# ```
#         For example, if the user inputs "a silver cat at home", you could return a list of
#         an image of a silver cat lying on wooden floor, a photo of a cat, a close-up photo of a cat, a photo in the direction of a cat, a photo of a cat sitting on a couch, etc.
        
#         2. You will also be able to analyze images from a global bird's eye view camera. If you want to take a picture, send a message back while 
#         that says "TAKE PICTURE" and the system will send a picture to you. IT IS VERY IMPORTANT THAT YOU ONLY ASK FOR A PICTURE WHEN YOU ARE READY TO ANALYZE IT.
#         IT IS ALSO VERY IMPORTANT THAT THE REQUEST MESSAGE IS "TAKE PICTURE" EXACTLY LIKE THIS, OTHERWISE THE SYSTEM WILL NOT WORK.

        # """
    
    def __init__(self):
        super().__init__('vlm_services')
        self.interface = OpenAIInterface(self.system_prompt, model="gpt-4o", max_messages=100)
        self.goal_publisher = self.create_publisher(String, 'robot_goal', 10)
        self.timer = self.create_timer(2, self.timer_callback)
        self.get_logger().info('VLM Services Node has been started. Waiting for requests...')

        self.conversation_state = False
        self.parsed_prompt = None
        self.declare_parameter('input_prompt', 'User forgot to specify input prompt, begin the conversation with the user.')
        self.input_prompt = self.get_parameter('input_prompt').get_parameter_value().string_value

        self.global_cam_subscription = self.create_subscription(
            Image,
            '/global_cam/rgb_camera/image_color',
            self.image_callback,
            10
        )

        self.latest_image = np.zeros((1, 1, 3), dtype=np.uint8)  # empty black image as placeholder
        self.bridge = CvBridge()

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

    def conversation(self):
        """
        Interactive conversation with GPT to refine the prompt.
        """
        self.conversation_state = True
        # Start conversation with system message explaining GPT should ask questions if it needs more info
        self.interface.add_message("system", """
                                   
            You can ask questions to clarify the user's intent and refine the the location to spawn the robot.
            If you need more information, ask a specific question. Do not be complacent and assume the user knows everything. 
            ENGAGE PROPERLY WITH THE USER.
                                   
            If you want to take a picture, send a message that says "TAKE PICTURE" and the system will send a picture to you.
            IT IS VERY IMPORTANT THAT YOU ONLY ASK FOR A PICTURE WHEN YOU ARE READY TO ANALYZE IT.
            IT IS ALSO VERY IMPORTANT THAT THE REQUEST MESSAGE IS "TAKE PICTURE" EXACTLY LIKE THIS, 
            OTHERWISE THE SYSTEM WILL NOT WORK.
            
            IF you want to spawn a robot, reply with:
            SPAWN ROBOT AT: <cell_number> <robot_name>
            where <cell_number> is the number of the grid cell you want to spawn the robot in, and <robot_name> is the name of the robot you want to spawn.
            The <cell_number> should be an integer, and the <robot_name> should be a string without spaces.
            For example, if you want to spawn a robot named "my_robot" in the grid cell number 5, you should send the message:
            SPAWN ROBOT AT: 5 my_robot

            Do NOT add any explanation, punctuation, or extra text before or after the command.
            If you do not follow this format exactly, the robot will NOT be spawned.

            BAD: Here is the command: SPAWN ROBOT AT: 5 my_robot
            BAD: SPAWN ROBOT AT: 5 my_robot. (with a period)
            GOOD: SPAWN ROBOT AT: 5 my_robot
                                   
            CHECK WITH THE USER ABOOUT THE GRID CELL NUMBER BEFORE SPAWNING THE ROBOT.

            If you want to set a goal for the robot to search for an object, you can do so by replying with:
            FINAL_PROMPTS: <comma-separated list of prompts>
            where <comma-separated list of prompts> is a list of up to 3 prompts that
            describe the object to look for. The prompts should be related to the object and should help
            the robot to find the object in the environment.
            For example, if the user inputs "a silver cat at home", you could return a list of
            silver cat, cat at home, cat on wooden floor
                                   
            DO NOT ADD ANYTHING ELSE BEFORE 'FINAL_PROMPTS' AT THE START OF YOUR FINAL MESSAGE OR THE PROGRAM WILL NOT WORK.
                                   
            You can also converse with the user and take pictures from the global camera and analyze them. If at any point you
            or the user wants to stop the conversation, just say "STOP" and the conversation will end. 
        """)
        self.interface.add_message("user", self.input_prompt)

        while True:
            reminders = """
            DO NOT SPAWN ROBOTS IN OCCLUDED AREAS. DO NOT SPAWN MORE THAN ONE ROBOT IN THE SAME LOCATION. YOU MAY ONLY SPAWN THREE ROBOTS.
            DO NOT ADD ANYTHING ELSE BEFORE 'FINAL_PROMPTS' AT THE START OF YOUR FINAL MESSAGE OR THE PROGRAM WILL NOT WORK.
            CHECK WITH THE USER ABOUT THE GRID CELL NUMBER BEFORE SPAWNING THE ROBOT.
            DO NOT SPAWN A ROBOT AGAIN AFTER IT HAS BEEN SPAWNED.

            """
            assistant_msg = self.interface.get_response()
            print(f"\n🤖 GPT asks: {assistant_msg}")

            # Check if GPT finished
            if "FINAL_PROMPTS:" in assistant_msg.strip():
                prompts_line = assistant_msg.strip()[len("FINAL_PROMPTS:"):].strip()
                self.parse_prompt(prompts_line)
            if "STOP" in assistant_msg.strip():
                self.get_logger().info("Conversation ended by GPT.")
                break
            if "TAKE PICTURE" in assistant_msg.strip():
                self.get_logger().info("GPT requested to take a picture.")
                if self.latest_image is not None:

                    image_file = self.upload_image_to_openai(self.latest_image)
                    self.interface.add_message("user", [{"type": "input_image", "file_id": image_file.id}])

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

                    cv2.imshow("Global Camera View with Grid", grid_image)
                    cv2.waitKey(0)  # Needed to update the OpenCV window

                    image_file = self.upload_image_to_openai(grid_image)
                    self.interface.add_message("user", [{"type": "input_image", "file_id": image_file.id}])

                    # send the image to GPT

                    self.interface.add_message("user", "Use these images to answer the previous question. You have a grid image as well as the original image." \
                                               "The grid image has a grid overlay for your use.")
                    continue  # Wait for GPT to process the images
                
                else:
                    self.interface.add_message("user", "No image available at the moment.")
                    
            if "SPAWN ROBOT AT:" in assistant_msg:
                try:
                    # Regex to find: SPAWN ROBOT AT: <cell_number> <robot_name>
                    match = re.search(r"SPAWN ROBOT AT:\s*(\d+)\s+(\S+)", assistant_msg)
                    if match:
                        grid_num = int(match.group(1))
                        robot_name = match.group(2)
                        spawn_x, spawn_y = self.get_coords_from_grid(grid_num)
                        position = f"{spawn_x} {spawn_y} -0.0065"
                        result = self.spawn_robot(robot_name, position)
                        if result is not None:
                            self.get_logger().info(f"Robot {robot_name} spawned at position {position}.")
                            self.interface.add_message("user", f"Robot {robot_name} has been spawned at position {position}.")
                        else:
                            self.get_logger().error("Failed to spawn robot.")
                            self.interface.add_message("user", "Error in spawning robot. Please check the format and try again.")
                    else:
                        raise ValueError("Could not parse SPAWN ROBOT AT command.")
                except Exception as e:
                    self.get_logger().error(f"Error in spawning robot: {e}")
                    self.interface.add_message("user", "Error in spawning robot. Please check the format and try again.")
            
                
            user_reply = input("\n✏️ Your answer: ")
            self.interface.add_message("user", user_reply)
        
        self.conversation_state = False


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

    def spawn_robot(self, robot_name: str, position: str):
        """
        Spawn a robot in the simulation using the SpawnNodeFromString service.
        """
        self.get_logger().info(f"Spawning robot {robot_name} at position {position}...")
        data_string = "Turtlebot4 {name \"" + robot_name + "\" translation " + position + " controller \"<extern>\"}"
        self.req.data = data_string
        self.get_logger().info(f"Requesting spawn with data: {self.req}")
        self.future = self.spawn_service.call_async(self.req)
        self.launch_ros2_file('llm_search', 'spawn_robot.py', {'robot_name': robot_name})
        self.future.add_done_callback(lambda fut: self.handle_spawn_response(fut, robot_name, position))


    def handle_spawn_response(self, future, robot_name, position):
        try:
            result = future.result()
            self.get_logger().info(f"Spawn service response: {result}")
        except Exception as e:
            self.get_logger().error(f"Error in spawning robot: {e}")
            
    def launch_ros2_file(self, package, launch_file, args=None):
        cmd = ['ros2', 'launch', package, launch_file]
        if args:
            for k, v in args.items():
                cmd.append(f'{k}:={v}')
        subprocess.Popen(['gnome-terminal', '--', *cmd])

def main():
    rclpy.init()
    node = VLMServices()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()