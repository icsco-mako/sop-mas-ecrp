import json
import subprocess
import logging
import sys
import io
from json import JSONDecodeError
from typing import Dict
from .base_agent import BaseAgent
from llm_mas_ecrp.utils.user_proxy_config import get_llm_client
from ..message_pool import MessagePool

logger = logging.getLogger(__name__)  # 使用模块名作为日志记录器名称


def run_py_file(file_path, func_name=None, args=None):
    cmd = ["python", file_path]
    if func_name:
        cmd.extend(["--func", func_name])
    if args is not None:
        cmd.extend(["--args", json.dumps(args)])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="gbk",  # 修改编码为系统默认
    )
    return {
        "exit_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


class TestingEngineer(BaseAgent):
    NAME = "Testing Engineer"
    ROLE_DESCRIPTION = """
    需要为解决空箱调运问题的Python-Gurobi求解代码设计测试用例，检查代码是否运行成功。
    """
    FORWARD_TASK = """
# Role: Python Test Engineer
## Background
Need to design test cases for Python-Gurobi solver code addressing empty container repositioning problems, ensuring algorithm robustness and accuracy across different scenarios.

## Input
1. Original problem description: {problem_description}
2. Cross-department feedback: {message_pool} (Includes parameters, decision variables, constraints, optimization objectives, and other key elements)
3. For the naming of parameters, please refer to the Parameter Naming Convention Document {parameter_naming_convention_document}.

## Core Tasks
**Parameter Generation**
   - Generate complete input parameter sets based on problem description
   - The parameters you generate should be same as the Parameter Naming Convention Document. Note the parameter dimensions, and do not generate new parameters.
   - Parameters should comply with business rules (e.g.: N>=1, M>=1)

## Output Requirements
- Only generate one test case.
- Output MUST be a valid JSON without any comments or explanations.
- Example format:
{{
    "param1": [1, 2, 3],
    "param2": [[1, 2], [3, 4]],
    "param3": 5
}}.
    """
    BACKWARD_TASK = """When you are solving a problem, you get a feedback from the external environment. You need to judge whether this is a problem caused by you or by other agents (other agents have given some results before you). If it is your problem, you need to give Come up with solutions and refined testing case.

The original problem is as follow:
{problem_description}

The testing case you give previously is as follow:
{previous_testing_case}
    
The feedback is as follow:
{feedback}

The output format is a JSON structure like this:
{{
    'is_caused_by_you': false,
    'reason': 'leave empty string if the problem is not caused by you',
    'refined_result': 'if the error is caused by you, please provide your refined testing case...'
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
            message_pool=message_pool,
        )
        response: str = self.llm_call(prompt=prompt)
        response = self.strip_str("```json", response)

        try:
            data: dict = json.loads(response)
        except JSONDecodeError as e:
            logging.error(f"JSON解析失败: {e}")
            logging.error(f"原始响应内容: {response}")
            return json.dumps(
                {
                    "error_type": "json_decode_error",
                    "error_msg": f"JSON解析失败: {str(e)}",
                }
            )

        code = message_pool.get("Python Developer")
        
        # 捕获求解器日志输出
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        solver_log = io.StringIO()
        
        try:
            # 重定向标准输出和错误输出
            sys.stdout = solver_log
            sys.stderr = solver_log
            
            # 使用局部作用域执行代码，以便捕获 optimize 函数
            import gurobipy as gp
            from gurobipy import GRB
            local_vars = {}
            exec(code, {"gp": gp, "GRB": GRB, "__builtins__": __builtins__}, local_vars)
            
            # 检查 optimize 函数是否存在
            if "optimize" not in local_vars:
                raise NameError("optimize function not found in the generated code")
            
            # 调用 optimize 函数
            result = local_vars["optimize"](data)
            
        finally:
            # 恢复标准输出
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        # 获取求解器日志
        solver_log_content = solver_log.getvalue()
        
        # 打印测试用例和结果（恢复输出后）
        print(f"测试用例: {data}")
        print(f"求解结果: {result}")
        if solver_log_content:
            logger.info(f"求解器日志:\n{solver_log_content}")
        
        return json.dumps(
            {
                "testing_case_generated": data,
                "testing_result": str(result),
                "solver_log": solver_log_content,  # 添加求解器日志
            }
        )

    # @override
    def backward_step(
        self, problem: Dict, hist_msg: str, feedback: str
    ) -> Dict[str, str]:
        prompt = self.BACKWARD_TASK.format(
            problem_description=problem["description"],
            previous_testing_case=hist_msg,
            feedback=feedback,
        )
        response: str = self.llm_call(prompt=prompt)
        response: str = self.strip_str("```json", response)
        try:
            return json.loads(response)  # str->dict
        except json.JSONDecodeError as e:
            logger.error(
                f"Error: {e}.\n testing engineer Response content is:\n{response}"
            )
            raise e


if __name__ == "__main__":
    with open("./src/workflow/problem_description.md", "r", encoding="utf-8") as f:
        problem = f.read()
    with open("./src/workflow/nlp_term_tags.json", "r", encoding="utf-8") as file:
        nlp_term_tags = json.load(file)
    with open(
        "./src/workflow/structure_model_info.json", "r", encoding="utf-8"
    ) as file:
        semantic_parsing_dict = json.load(file)
    with open("./src/workflow/model.md", "r", encoding="utf-8") as file:
        model = file.read()
    with open("./src/workflow/generate_code.py", "r", encoding="utf-8") as file:
        code = file.read()
    message_pool = {
        "business_expert": nlp_term_tags,
        "product_manager": semantic_parsing_dict,
        "or_specialist": model,
        "python_developer": code,
    }
    testing_engineer = TestingEngineer("DeepSeek", "ep-20250210181347-9n2pl")
    testing_engineer.step(problem, message_pool)
