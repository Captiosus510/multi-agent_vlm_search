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

from sensor_msgs.msg import Image

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
        self.interface = OpenAIInterface(self.system_prompt, model="gpt-4o", max_messages=20)
        # self.analysis_service = self.create_service(Analysis, 'analysis', self.analysis_callback)
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

        self.latest_image = None  # contains the latest image from the global camera as a cv2 image
        self.bridge = CvBridge()

        # Start conversation in a separate thread so ROS can keep spinning
        self._conversation_thread = threading.Thread(target=self.conversation, daemon=True)
        self._conversation_thread.start()

    def conversation(self):
        """
        Interactive conversation with GPT to refine the prompt.
        """
        self.conversation_state = True
        # Start conversation with system message explaining GPT should ask questions if it needs more info
        self.interface.add_message("system", """
                                   
            This is an interactive conversation with the user to refine the prompt for the robot search.
            You will be given an initial prompt from the user.
            You can ask questions to clarify the user's intent and refine the prompt.
            If you need more information, ask a specific question. 
            Once you have enough, reply with:
            FINAL_PROMPTS: prompt1, prompt2,
            Make sure it's exactly like this: starts with 'FINAL_PROMPTS:' then comma separated.
            DO NOT ADD ANYTHING ELSE BEFORE 'FINAL_PROMPTS' AT THE START OF YOUR FINAL MESSAGE OR THE PROGRAM WILL NOT WORK.
                                   

            You can also converse with the user and take pictures from the global camera and analyze them. If at any point you
            or the user wants to stop the conversation, just say "STOP" and the conversation will end. 
        """)
        self.interface.add_message("user", self.input_prompt)

        while True:
            assistant_msg = self.interface.get_response()
            print(f"\n🤖 GPT asks: {assistant_msg}")

            # Check if GPT finished
            if "FINAL_PROMPTS:" in assistant_msg.strip():
                prompts_line = assistant_msg.strip()[len("FINAL_PROMPTS:"):].strip()
                self.parse_prompt(prompts_line)
            elif "STOP" in assistant_msg.strip():
                self.get_logger().info("Conversation ended by GPT.")
                break
            elif "TAKE PICTURE" in assistant_msg.strip():
                self.get_logger().info("GPT requested to take a picture.")
                if self.latest_image is not None:
                    # Convert the latest image to a format suitable for OpenAI API
                    success, encoded_image = cv2.imencode('.jpg', self.latest_image)
                    if not success:
                        self.get_logger().error("Failed to encode image.")
                        continue
                    
                    # Create a temporary file with proper extension
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
                        temp_file.write(encoded_image.tobytes())
                        temp_file_path = temp_file.name
                    
                    # Upload the file to OpenAI
                    with open(temp_file_path, 'rb') as f:
                        image_file = self.interface.client.files.create(file=f, purpose='vision')
                    
                    self.interface.add_message("user", [{"type": "input_image", "file_id": image_file.id}])
                    # self.interface.add_message("user", "Use this image to answer the previous question.")
                    continue  # Continue to get response after sending image
                else:
                    self.interface.add_message("user", "No image available at the moment.")
                    continue

            else:
                # Otherwise, user needs to reply
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

    # def analysis_callback(self, request, response):
        # self.get_logger().info(f"Received analysis request for image.")
        # cv_image = self.bridge.imgmsg_to_cv2(request.image, desired_encoding='bgr8')

        # with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
        #     cv2.imwrite(temp_file.name, cv_image)
        #     self.get_logger().info(f"Image saved to {temp_file.name}")
        #     filepath = temp_file.name
        # image_file = self.interface.client.files.create(file=open(filepath, 'rb'), purpose='vision')

        # # success, encoded_image = cv2.imencode('.jpg', cv_image)
        # # if not success:
        # #     self.get_logger().error("Failed to encode image.")
        # #     response.found = False
        # #     return response
        # # image_bytes = encoded_image.tobytes()
        # # image_file = self.interface.client.files.create(file=image_bytes, purpose='vision')

        # self.interface.add_message("user", [{"type": "input_image", "file_id": image_file.id}])
        # self.interface.add_message("user", "Is this the object I am looking for? Please answer with 'yes' or 'no'.")
        # result = self.interface.get_response()
        # self.get_logger().info(f"Analysis response: {result}")

        # if result.lower() == "yes":
        #     response.found = True
        # elif result.lower() == "no":
        #     response.found = False
        # return response
    



def main():
    rclpy.init()
    node = VLMServices()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()