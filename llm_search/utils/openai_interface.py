from openai import OpenAI
from pydantic import BaseModel
from enum import Enum
from typing import List, Optional

class BehaviorType(Enum):
    MONITOR = "monitor"
    SEARCH = "search"

class TakePictureCall(BaseModel):
    name: str = "take_picture"

class SetGoalCall(BaseModel):
    name: str = "set_goal"
    prompts: str

class SpawnRobotCall(BaseModel):
    name: str = "spawn_robot"
    grid_cell: int
    robot_name: str
    behavior: BehaviorType

class StopCall(BaseModel):
    name: str = "stop"

class ChainOfThoughtStep(BaseModel):
    reasoning: str

class Conversation(BaseModel):
    text: str  # The text of the conversation step
    take_picture: Optional[TakePictureCall] = None
    set_goal: Optional[SetGoalCall] = None  
    spawn_robot: Optional[SpawnRobotCall] = None
    stop: Optional[StopCall] = None

class OpenAIInterface:
    def __init__(self, system_prompt: str, model: str = "gpt-4o", max_messages: int = 20):
        self.system_prompt = system_prompt
        self.client = OpenAI()
        self.model = model
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.max_messages = max_messages
        self.structured_mode = False  # Toggle for structured/free-form output

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
        self.prune_messages(self.messages, self.max_messages)

    def prune_messages(self, messages, max_length):
        self.messages = [{"role": "system", "content": self.system_prompt}] + messages[-max_length:] if len(messages) > max_length else messages
    
    def get_messages(self):
        return self.messages
    
    def toggle_structured_mode(self, enable: bool):
        """
        Toggle between structured output mode and free-form chat completion.
        """
        self.structured_mode = enable

    def get_response(self, response_format=None):
        """
        Get a response from the model using structured output with the Conversation format.
        """
        if not response_format:
            response_format = Conversation
            
        response = self.client.responses.parse(
            model=self.model,
            input=self.messages, # type: ignore
            text_format=response_format,
        )
        
        parsed_output = response.output_parsed

        if parsed_output is None:
            raise ValueError("No valid output received from the model.")
        self.add_message("assistant", parsed_output.text) # type: ignore
        return parsed_output
    

