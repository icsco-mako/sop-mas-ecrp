import json
from typing import Dict
from llm_mas_ecrp.utils.user_proxy_config import get_llm_client
from .base_agent import BaseAgent
from ..message_pool import MessagePool


class DataEngineer(BaseAgent):
    NAME = "DE-Agent"
    ROLE_DESCRIPTION = "You are a product manager who provides additional domain-specific knowledge to enhance problem understanding and formulation."
    FORWARD_TASK = """
# ROLE
Algorithm Product Manager（Maritime shipping operations domain）
## SKILL
- Multi-dimensional analysis and prioritization of business requirements
- Cross-domain team communication and coordination
- Balancing technical feasibility with business objectives
## KNOWLEDGE BASE
- Maritime shipping operation's knowledge {knowledge}
- Operations research methodologies (Linear Programming, Integer Programming, Mixed-Integer Programming)
- Typical business scenario modeling patterns (Empty container repositioning, Resource balancing, Transportation problems)
## TASKs
The comments from other experts are as follow: {message_pool}, you should extract key parameters, variables, constraints, and objective from {problem_description} for modeling. You can proceed with the following steps: 
    1 Identify the problem type.
        - The type of optimization problem may be linear programming(LP), integer programming(IP), or mixed-integer programming(MIP). You need to identify the problem type from the problem description.
    2 Identify the parameters.
        - Note that parameters are known values upon which the model is built, and they do not change during the optimization process. However, variables are the unknowns that the optimization process seeks to solve. DO NOT include variables in the parameters list!
        - Note that indices are not separate parameters.
        - For the naming of parameters, please refer to the Parameter Naming Convention Document {parameter_naming_convention_document}.
        - Dimensional parameters should also be listed as parameters in the output JSON.
        - Use single `Capital Letters` for symbols that represent dimensions for indices of other parameters (e.g. N, M, etc.).
        - Make sure you include all the parameters in the output json. 
        - If any ambiguities exist in the description, note them specifically in output json.
    3 Identify the variables.
        - Identify and list all variables, including any implicit ones like non-negativity.
        - Note the dimensions of the variables.
        - Use CamelCase and Full Words for symbols, and don't include the indices (e.g. MaxColor instead of maxColor or max_color or maxcolor or MaxCol or Max_C or MaxColor_i or MaxColor_{{i}} or MaxColor_II or MaxCorlorII).
        - If any ambiguities exist in the description, note them specifically in output json.
    4 Identify the constraints.
        - Identify and list all constraints, including any implicit ones like non-negativity.
        - Preferences are not constraints. Do not include them in the list of constraints.
        - Statements that simply define exact values of parameters are not constraints. Do not include them in the list of constraints (e.g., "The cost of producing an X is Y", or "Each X has a size of Y").
        - Statements that define bounds are constraints. Include them in the list of constraints (e.g., "The cost of producing an X is at most Y", or "Each X has a size of at least Y").
        - Please describe the constraints using natural language without using mathematical formulas.
        - If any ambiguities exist in the description, note them specifically in output json.
    5 Identify the objective function.
        - Determine the primary objective of the problem.
        - Please describe the objective using natural language without using mathematical formulas.
## OUTPUT FORMAT
Your output format should be a JSON like this (choose at most 3 hardest terminology):
{{
  "problem_type": "MIP",
  "parameter": [
    {{
      "symbol": str
      "definition": str,
      "value": number,
      "shape": [str]
    }}
  ],
  "variables": [
    {{
      "symbol": str,
      "definition": str,
      "value": number,
      "shape": [str]
    }}
  ],
  "constraints": ["List", "Of", "All", "Constraints"],
  "ambiguities": ["List", "Of", "Identified", "Ambiguities"],
  "objective": "The primary objective to be achieved"
}}

Where 
- "symbol" is the mathematical symbols marked in the original text.
- "definition" is the the definition of the parameter or variable.
- "value" a float or int, representing the numerical value of the parameter/variable (use 0.33 instead of 1/3.0)
- "shape" is a possibly empty list of string representing the dimensions of the parameter in terms of other parameters.

## INSTRUCTIONS
- Please answer in Chinese.
"""
    BACKWARD_TASK = """When you are solving a problem, you get a feedback from the external environment. You need to judge whether this is a problem caused by you or by other agents (other agents have given some results before you). If it is your problem, you need to give Come up with solutions and refined answer.

The original problem is as follow:
{problem_description}

The answer you give previously is as follow:
{previous_answer}
    
The feedback is as follow:
{feedback}

The output format is a JSON structure like this:
{{
    'is_caused_by_you': false,
    'reason': 'leave empty string if the problem is not caused by you',
    'refined_result': 'if the error is caused by you, please provide your refined answer...'
}}
"""

    def __init__(self, client, model, agent_id: int) -> None:
        self.client = get_llm_client(client)
        self.model = model
        self.name = self.NAME
        super().__init__(self.client, model, agent_id)

    # @override
    def forward_step(self, problem: Dict, message_pool: MessagePool) -> str:
        pormpt = self.FORWARD_TASK.format(
            problem_description=problem["description"],
            parameter_naming_convention_document=problem["sample"][0]["input"],
            message_pool=message_pool.get_content(),
            knowledge="",
        )
        response = self.llm_call(prompt=pormpt)
        response = self.strip_str("```json", response)
        return response

    # @override
    def backward_step(
        self, problem: Dict, hist_msg: str, feedback: str
    ) -> Dict[str, str]:
        prompt = self.BACKWARD_TASK.format(
            problem_description=problem["description"],
            previous_answer=hist_msg,
            feedback=feedback,
        )
        response: str = self.llm_call(prompt=prompt)
        response: str = self.strip_str("```json", response)
        return json.loads(response)  # str->dict


if __name__ == "__main__":
    with open("./src/workflow/problem_description.md", "r", encoding="utf8") as f:
        problem = f.read()

    product_manager = ProductManager("DeepSeek", "ep-20250210181347-9n2pl")
    product_manager.forward(problem)
    print("completed")
