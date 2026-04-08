import json
import logging
import sys
import io
from json import JSONDecodeError
from typing import Dict
from .base_agent import BaseAgent
from llm_mas_ecrp.utils.user_proxy_config import get_llm_client
from ..message_pool import MessagePool

logger = logging.getLogger(__name__)


class TestingEngineer(BaseAgent):
    NAME = "Testing Engineer"
    ROLE_DESCRIPTION = """
    Performs supplementary testing and fine-grained error diagnosis on Python-Gurobi solver code. Generates structured diagnostic artifacts (execution summaries, solver logs) based on controller-level execution outcomes to support failure localization during exception backtracking.
    """
    FORWARD_TASK = """
# Role: Testing Engineer — Supplementary Testing & Fine-Grained Error Diagnosis

## Background
You are the Testing Engineer in a multi-agent system for Operations Research optimization. Your role is to analyze code execution results and generate a structured diagnostic report that supports failure localization during exception backtracking.

## Input
1. Original problem description: {problem_description}
2. Execution outcome: {execution_outcome}
3. Solver log: {solver_log}
4. Upstream agent outputs: {message_pool}

## Tasks
Based on the controller-level execution outcomes above:
1. **Execution Analysis**: Evaluate whether the solver code executed correctly. If successful, assess the quality of the solution (objective value reasonableness, constraint satisfaction). If failed, identify the likely root cause.
2. **Error Diagnosis**: If execution failed, classify the error:
   - Code-level error (syntax, import, function definition)
   - Model-level error (incorrect formulation, infeasible model)
   - Data-level error (parameter mismatch, invalid input format)
3. **Recommendations**: Provide actionable suggestions for the responsible agent.

## Output Requirements
Output MUST be a valid JSON without any comments or explanations:
{{
    "execution_status": "success or failure",
    "diagnostic_summary": "Brief analysis of the execution result",
    "error_analysis": "Detailed error classification and root cause (empty string if success)",
    "solver_log_summary": "Key observations from the solver log",
    "recommendations": "Suggestions for improvement"
}}
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
        # Step 1: Execute code with actual data (controller-level execution)
        code = message_pool.get("Python Developer")
        data = problem["sample"][0]["input"]

        execution_outcome = {}
        solver_log_content = ""

        old_stdout, old_stderr = sys.stdout, sys.stderr
        solver_log = io.StringIO()

        try:
            sys.stdout = solver_log
            sys.stderr = solver_log

            import gurobipy as gp
            from gurobipy import GRB

            local_vars = {}
            exec(code, {"gp": gp, "GRB": GRB, "__builtins__": __builtins__}, local_vars)

            if "optimize" not in local_vars:
                raise NameError("optimize function not found in the generated code")

            result = local_vars["optimize"](data)

            execution_outcome = {
                "status": "success",
                "obj_value": result.get("obj_value") if isinstance(result, dict) else None,
                "result": str(result),
            }
        except Exception as e:
            execution_outcome = {
                "status": "failure",
                "error_type": type(e).__name__,
                "error_msg": str(e),
            }
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            solver_log_content = solver_log.getvalue()

        logger.info(f"TE execution: {execution_outcome.get('status')}")
        if execution_outcome.get("status") == "success":
            logger.info(f"TE obj_value: {execution_outcome.get('obj_value')}")

        # Step 2: LLM diagnostic analysis
        prompt = self.FORWARD_TASK.format(
            problem_description=problem["description"],
            execution_outcome=json.dumps(execution_outcome, ensure_ascii=False),
            solver_log=solver_log_content[:3000],
            message_pool=message_pool.get_content(),
        )
        response: str = self.llm_call(prompt=prompt)
        response = self.strip_str("```json", response)

        try:
            diagnosis = json.loads(response)
        except JSONDecodeError as e:
            logger.error(f"TE diagnostic JSON parse failed: {e}")
            logger.error(f"Raw response: {response}")
            diagnosis = {"diagnostic_summary": response}

        # Step 3: Return structured diagnostic report
        return json.dumps(
            {
                "execution_status": execution_outcome.get("status", "unknown"),
                "obj_value": execution_outcome.get("obj_value"),
                "diagnostic_summary": diagnosis.get("diagnostic_summary", ""),
                "error_analysis": diagnosis.get("error_analysis", ""),
                "solver_log": solver_log_content,
                "recommendations": diagnosis.get("recommendations", ""),
            },
            ensure_ascii=False,
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
