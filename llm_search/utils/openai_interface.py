from openai import OpenAI


class OpenAIInterface:
    def __init__(self, system_prompt, model="gpt-4o", max_messages=20):
        self.system_prompt = system_prompt
        self.client = OpenAI()
        self.model = model
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.max_messages = max_messages

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
        self.prune_messages(self.messages, self.max_messages)

    def prune_messages(self, messages, max_length):
        self.messages = [{"role": "system", "content": self.system_prompt}] + messages[-max_length:] if len(messages) > max_length else messages
    
    def get_messages(self):
        return self.messages
    
    def get_response(self):
        response = self.client.responses.create(
            model=self.model,
            input=self.messages,  # type: ignore
            tools=[],
            tool_choice="auto",
            stream=False
        ) # type: ignore

        output_text = response.output_text
        self.add_message("assistant", output_text)
        return output_text