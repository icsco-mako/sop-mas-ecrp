import json
from typing import Dict
from .base_agent import BaseAgent
from llm_mas_ecrp.utils.user_proxy_config import get_llm_client
from ..message_pool import MessagePool


class PythonDeveloper(BaseAgent):
    NAME = "Python Developer"
    ROLE_DESCRIPTION = """You are a senior Python developer specializing in operations research and mathematical optimization. Your proficiency in utilizing third-party libraries such as Gurobi is essential. In addition to your expertise in Gurobi, it would be great if you could also provide some background in related libraries or tools, like NumPy, SciPy, or PuLP.
"""

    # :
    # 1. Mastery of Gurobi Optimizer ecosystem for building MIP/LP models
    # 2. Proficient in scientific computing stack (NumPy/SciPy) and optimization tools (PuLP/CPLEX)
    # 3. Expert in translating mathematical programming problems into efficient code implementations
    # 4. Adhere to PEP8 standards with industrial-grade code readability and maintainability
    # 5. Capable of adding clear documentation for critical algorithm logic
    # 6. Knowledgeable about modern optimization algorithms and best practices
    FORWARD_TASK = """
# Problem-Solving Framework
## Input 
- Now the origin problem is as follow: {problem_description}
- And the comments from other experts are as follow: {comments_text}
- For the naming of parameters, please refer to the Parameter Naming Convention Document {parameter_naming_convention_document}.

# Code Development Requirements
## Core Constraints
1. Implement solution strictly using function encapsulation (essential imports allowed)
2. The function input parameter is: data, which is a dictionary
3. Strictly follow Gurobi API specifications for solvable models
4. Ensure mathematical rigor (explicit objective functions, variable types, constraints)
5. Implement essential exception handling (infeasible/unbounded status detection)
6. Output format must comply with downstream system integration requirements

## Optimization Guidelines
- Prioritize vectorized operations for large-scale data processing
- Design key parameters as configurable options
- Utilize context managers for resource handling
- Enhance maintainability with type annotations

## Output Specifications
1. Directly output complete executable Python functions, the function should be named as `optimize`
2. Function returns must use standard data structures (dict), including decision variables, solution status, objective function value (named as `obj_value`), and so on. 
3. Include essential model status checking logic
4. Exclude any test code or sample invocations
"""
    BACKWARD_TASK = """When you are solving a problem, you get a feedback from the external environment. You need to judge whether this is a problem caused by you or by other experts (other experts have given some results before you). If it is your problem, you need to give Come up with solutions and refined code.

The original problem is as follow:
{problem_description}

The code you give previously is as follow:
{previous_code}
    
The feedback is as follow:
{feedback}

The output format is a JSON structure like this:
{{
    'is_caused_by_you': false,
    'reason': 'leave empty string if the problem is not caused by you',
    'refined_result': 'if the error is caused by you, please provide your refined code...'
}}
"""

    def __init__(self, client, model, agent_id: int) -> None:
        self.client = get_llm_client(client)
        self.model = model
        self.name = self.NAME
        super().__init__(self.client, model, agent_id)

    # @override
    def forward_step(self, problem: Dict, message_pool: MessagePool):
        pormpt = self.FORWARD_TASK.format(
            problem_description=problem["description"],
            parameter_naming_convention_document=problem["sample"][0]["input"],
            comments_text=message_pool.get_content(),
        )
        response = self.llm_call(prompt=pormpt)
        response = self.strip_str("```python", response)
        return response

    # @override
    def backward_step(
        self, problem: Dict, hist_msg: str, feedback: str
    ) -> Dict[str, str]:
        prompt = self.BACKWARD_TASK.format(
            problem_description=problem["description"],
            previous_code=hist_msg,
            feedback=feedback,
        )
        response: str = self.llm_call(prompt=prompt)
        response: str = self.strip_str("```json", response)
        return json.loads(response)  # str->dict
