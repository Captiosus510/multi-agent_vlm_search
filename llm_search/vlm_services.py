import rclpy
from rclpy.node import Node
import time
from openai import OpenAI
import json, cv2
from llm_search.utils.openai import OpenAIInterface
from llm_search_interfaces.srv import Analysis
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import String
import tempfile

class VLMServices(Node):
    """
    This will publish the goal for the robot to a topic and provide a double check service to confirm if the robot has found the object or not.
    It will use the OpenAI API to analyze the image and confirm if the object is indeed.
    """
    system_prompt = """

        You are part of a ROS2 robot framework to carry out multi-agent robot search. The idea is to have multiple robots explore an environment looking for something
        and report back to a central node. You are the one who will provide them with information about their tasks.
        You will set their goal and then confirm they have found it at the very end, since querying you takes a lot of time and can't be done in realtime.

        For now, you won't have to coordinate any robots. Here are your two primary objectives as a ROS2 Service Provider Node:
        1. You will be given a prompt from the user that describes an object to look for. Each agent is equipped with YOLO11 trained on the COCO dataset, so you can use that to search for objects.
        You have to take this input prompt and output comma seperated class names based on this list:

        person, bicycle, car, motorbike, aeroplane, bus, train, truck, boat, traffic light, fire hydrant,
        stop sign, parking meter, bench, bird, cat, dog, horse, sheep,
        cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie
        suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat,
        baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass,
        cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange,
        broccoli, carrot, hot dog, pizza, donut, cake, chair, sofa,
        pottedplant, bed, diningtable, toilet, tvmonitor, laptop, mouse,
        remote, keyboard, cell phone, microwave, oven, toaster, sink,
        refrigerator, book, clock, vase, scissors, teddy bear, hair drier,
        toothbrush

        Make sure the most closely related class names are at the top of the list and only output a max of 3.

        2. You will also be requested to confirm if the robot has found the object or not. You will get an image to analyze
        when the robot has a high confidence that it has found the object. Since you are better able to describe images, you will be the 
        one to confirm if the object is indeed the one the user is looking for. Output a simple "yes" or "no" response based on the image.
        COLOR MATTERS AND IS VERY IMPORTANT, so make sure to analyze the image carefully.

        """
    
    def __init__(self):
        super().__init__('vlm_services')
        self.interface = OpenAIInterface(self.system_prompt, model="gpt-4o")
        self.analysis_service = self.create_service(Analysis, 'analysis', self.analysis_callback)
        self.goal_publisher = self.create_publisher(String, 'robot_goal', 10)
        self.timer = self.create_timer(2, self.timer_callback)
        self.get_logger().info('VLM Services Node has been started. Waiting for requests...')

        self.analyzed_prompt = False
        self.parsed_prompt = None
        self.declare_parameter('input_prompt', 'What object are you looking for? Please describe it in detail.')
        self.input_prompt = self.get_parameter('input_prompt').get_parameter_value().string_value

        self.bridge = CvBridge()

    def parse_prompt(self):
        """
        This function will parse the user input prompt and return a comma separated list of class names.
        """
        self.get_logger().info(f"Parsing user input prompt: {self.input_prompt}")

        # Get the response from the OpenAI API
        self.interface.add_message("system", "Based on the user's input, provide a comma separated list of class names from the COCO dataset that best describes the object.")
        goal = self.interface.get_response(self.input_prompt)
        self.interface.add_message("assistant", goal)

        self.get_logger().info(f"Goal: {goal}")
        
        # Parse the response
        goal = goal.strip().split(',')
        self.parsed_prompt = goal[0]

        self.analyzed_prompt = True
        self.get_logger().info(f"Parsed prompt: {self.parsed_prompt}")
    
    def timer_callback(self):
        if not self.analyzed_prompt:
            self.parse_prompt()
        self.goal_publisher.publish(String(data=self.parsed_prompt))

    def analysis_callback(self, request, response):
        self.get_logger().info(f"Received analysis request for image.")
        cv_image = self.bridge.imgmsg_to_cv2(request.image, desired_encoding='bgr8')

        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp_file:
            cv2.imwrite(temp_file.name, cv_image)
            self.get_logger().info(f"Image saved to {temp_file.name}")
            filepath = temp_file.name
        image_file = self.interface.client.files.create(file=open(filepath, 'rb'), purpose='vision')

        # success, encoded_image = cv2.imencode('.jpg', cv_image)
        # if not success:
        #     self.get_logger().error("Failed to encode image.")
        #     response.found = False
        #     return response
        # image_bytes = encoded_image.tobytes()
        # image_file = self.interface.client.files.create(file=image_bytes, purpose='vision')

        self.interface.add_message("user", [{"type": "input_image", "file_id": image_file.id}])
        result = self.interface.get_response("Analyze this image and confirm if the object is indeed the one I am looking for. Respond with 'yes' or 'no'.")
        self.get_logger().info(f"Analysis response: {result}")

        if result.lower() == "yes":
            response.found = True
        elif result.lower() == "no":
            response.found = False
        return response
    



def main():
    rclpy.init()
    node = VLMServices()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()