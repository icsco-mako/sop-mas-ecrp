import json
from typing import Dict
from .base_agent import BaseAgent
from llm_mas_ecrp.utils.user_proxy_config import get_llm_client
from ..message_pool import MessagePool


class ModelEngineer(BaseAgent):
    NAME = "OR Specialist"
    ROLE_DESCRIPTION = "You are a modeling expert specialized in the field of Operations Research and Optimization. Your expertise lies in Mixed-Integer Programming (MIP), or Integer Programming (IP) models, and you possess an in-depth understanding of various modeling techniques within the realm of operations research. At present, you are given an Operations Research problem, alongside additional insights provided by other experts. The goal is to holistically incorporate these inputs and devise a comprehensive model that addresses the given production challenge."
    FORWARD_TASK = """
# ROLE
modeling expert

## INPUT
Please review the problem description we need you to model: {problem_description}. The comments from other experts are as follow: {message_pool}, and for the naming of parameters, please refer to the Parameter Naming Convention Document {parameter_naming_convention_document}.
## SKILL
- Capable of accurately identifying decision variables in optimization problems.
- Skilled in constructing constraint systems that meet the requirements of MIP solvers.
- Proficient in unit conversion and dimension consistency verification.
- Able to design auxiliary variables to implement complex logical constraints.

## TASKs
Please complete the modeling according to the following stages:
### Phase 1: Problem Analysis
- Read the {problem_description} and the reference information {message_pool} provided by your colleague

### Phase 2: Model Construction
1. Decision Variable Definition:
   - Use CamelCase naming convention
   - Annotate variable types (Continuous/Integer/0-1)
   - Clearly define the decision variables.
2. Objective Function Construction:
   - Explicitly label dimensional units
   - Verify dimensional consistency with constraints
   - Formulate the objective function precisely.During the process, please pay attention to the quantitative units of parameters and decision variables in the problem information.
   - Ensure the formulation is coherent, logical, and solvable.
   - Provide any necessary explanations or clarifications for your formulation.
3. Constraints Construction:
   - Categorize constraints by function (Resource/Logic/Business)
   - List non-negativity constraints separately
   - List all the constraints, ensuring they are complete and non-redundant.During the process, please pay attention to the quantitative units of parameters and decision variables in the problem information.If a type of decision variable is non-negative, it should be listed as a separate type of non-negative constraint. Do not mix non-negative constraints of different decision variables together.
   - Ensure the formulation is coherent, logical, and solvable.
   - Provide any necessary explanations or clarifications for your formulation.
   - No need for model linearization.

### Phase 3: MODEL OUTPUT FORMAT
Your output format should be a Markdown like this: 
```markdown
# parameter
list all parameters

# variable
list all variables

#objective
list the objective function

# constraint
list all constraints

# ambiguity
list all ambiguities
```

## INSTRUCTIONS
- Use Latex syntax to render mathematical symbols and formulas. For example, use $symbol$ to render symbols and $$equation$$ to render formulas. Do not use \(symbol\)、\[equation\] or (symbol)、[equation]!!!
- You can NOT define new parameters!!! You can only define new variables.
- Use empty list ([]) if no new variables are defined.
- Introducing auxiliary or state variables to improve model readability and modularity.
- No need to write code.
- Please answer in Chinese.
"""

    # ## INSTRUCTIONS
    # - Use Latex syntax to render mathematical symbols and formulas. For example, use $symbol$ to render symbols and $$equation$$ to render formulas. Do not use \(symbol\)、\[equation\] or (symbol)、[equation]!!!
    # - You can NOT define new parameters!!! You can only define new variables. Use CamelCase and full words for new variable symbols, and do not include indices in the symbol (e.g. ItemsSold instead of itemsSold or items_sold or ItemsSold_i)
    # - Use \\text{{}} when writing variable and parameter names. For example (\\sum_{{i=1}}^{{N}} \\text{{ItemsSold}}_{{i}} instead of \\sum_{{i=1}}^{{N}} ItemsSold_{{i}})
    # - Use \\quad for spaces.
    # - Use empty list ([]) if no new variables are defined.
    # - Always use non-strict inequalities (e.g. \\leq instead of <), even if the constraint is strict.
    # - Define auxiliary constraints when necessary. Set it to an empty list ([]) if no auxiliary constraints are needed. If new auxiliary constraints need new variables, add them to the "new_variables" list too.
    # - Please note that the information you extract is for the purpose of modeling, which means your variables, constraints, and objectives need to meet the requirements of a solvable LP or MIP model. Within the constraints, the comparison operators must be equal to, greater than or equal to, or less than or equal to (> or < are not allowed to appear and should be replaced to be \geq or \leq).
    # - No need to write code.
    # - Please answer in Chinese.
    BACKWARD_TASK = """When you are solving a problem, you get a feedback from the external environment. You need to judge whether this is a problem caused by you or by other experts (other experts have given some results before you). If it is your problem, you need to give Come up with solutions and refined model.

The original problem is as follow:
{problem_description}

The feedback is as follow:
{feedback}

The modeling you give previously is as follow:
{previous_model}

The output format is a JSON structure like this:
{{
    "is_caused_by_you": false,
    "reason": "leave empty string if the problem is not caused by you",
    "refined_result": "if the error is caused by you, please provide your refined model"
}}
        
    """

    def __init__(self, client, model, agent_id: int) -> None:
        self.client = get_llm_client(client)
        self.model = model
        self.name = self.NAME
        super().__init__(self.client, model, agent_id)

    # @override
    def forward_step(self, problem: Dict, message_pool: MessagePool):
        prompt = self.FORWARD_TASK.format(
            problem_description=problem["description"],
            parameter_naming_convention_document=problem["sample"][0]["input"],
            message_pool=message_pool.get_content(),
        )
        response: str = self.llm_call(prompt=prompt)
        response: str = self.strip_str("```markdown", response)
        return response

    # @override
    def backward_step(
        self, problem: Dict, hist_msg: str, feedback: str
    ) -> Dict[str, str]:
        prompt = self.BACKWARD_TASK.format(
            problem_description=problem["description"],
            previous_model=hist_msg,
            feedback=feedback,
        )
        response: str = self.llm_call(prompt=prompt)
        response: str = self.strip_str("```json", response)
        print("model expert backtrack response", response)
        return json.loads(response)  # str->dict

