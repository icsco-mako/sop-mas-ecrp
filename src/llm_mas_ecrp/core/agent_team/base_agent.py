from typing import List, Dict, Optional
from openai import OpenAI
from llm_mas_ecrp.utils.llm_call import llm_call


class BaseAgent:
    """专家类"""

    NAME: str = ""
    ROLE_DESCRIPTION: str = "You're a helpful assistant."

    def __init__(self, client: OpenAI, model: str, agent_id: int) -> None:
        self.model = model
        self.name = self.NAME
        self.client = client
        self.agent_id = agent_id

    def llm_call(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List] = None,
        seed: int = 10,
    ) -> str:
        return llm_call(
            self.client,
            self.model,
            self.ROLE_DESCRIPTION,
            prompt=prompt,
            messages=messages,
            seed=seed,
        )

    def forward_step(self, problem: Dict, message_pool) -> str:
        pass

    def backward_step(self, problem: Dict, message_pool) -> str:
        pass

    def strip_str(self, str_type, response):
        # delete until the first '```json'
        # delete until the last '```'
        if str_type in response:
            response = response[response.find(str_type) + len(str_type) :]
            response = response[: response.rfind("```")]
        return response

    def save_response_file(self, filepath, response):
        if "```json" in response:
            # delete until the first '```json'
            s = "```json"
            response = response[response.find(s) + len(s) :]
            # delete until the last '```'
            response = response[: response.rfind("```")]
        elif "```python" in response:
            # delete until the first '```python'
            s = "```python"
            response = response[response.find(s) + len(s) :]
            # delete until the last '```'
            response = response[: response.rfind("```")]
        with open(filepath, "w+", encoding="utf-8") as f:
            f.write(response)
        print("file saved suceesfully: ", filepath)

    def __str__(self):
        return f"{self.name}"

    def __repr__(self):
        return self.__str__()
