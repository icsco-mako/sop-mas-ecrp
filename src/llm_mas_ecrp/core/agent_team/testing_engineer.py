import json
import logging
from collections.abc import Mapping, Sequence
from json import JSONDecodeError
from typing import Dict
from .base_agent import BaseAgent
from llm_mas_ecrp.utils.user_proxy_config import get_llm_client
from ..message_pool import MessagePool

logger = logging.getLogger(__name__)


def _truncate_text(text: str, limit: int = 1200) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _summarize_value(value, depth: int = 0, max_depth: int = 2):
    if depth >= max_depth:
        if isinstance(value, Mapping):
            return {"type": "dict", "size": len(value)}
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            return {"type": "list", "size": len(value)}
        return value

    if isinstance(value, Mapping):
        items = list(value.items())
        summary = {"type": "dict", "size": len(items)}

        scalar_items = []
        nested_items = []
        for key, val in items:
            if isinstance(val, (int, float, str, bool)) or val is None:
                scalar_items.append((key, val))
            else:
                nested_items.append((key, val))

        if scalar_items:
            summary["sample_scalars"] = {str(k): v for k, v in scalar_items[:8]}
        if nested_items:
            summary["sample_nested"] = {
                str(key): _summarize_value(val, depth + 1, max_depth)
                for key, val in nested_items[:4]
            }
        return summary

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        sample = list(value[:5])
        return {
            "type": "list",
            "size": len(value),
            "sample": [_summarize_value(v, depth + 1, max_depth) for v in sample],
        }

    return value


def _summarize_execution_result(result) -> Dict:
    if not isinstance(result, dict):
        return {"result_type": str(type(result)), "preview": _truncate_text(str(result), 800)}

    summary = {
        "keys": list(result.keys()),
        "obj_value": result.get("obj_value"),
    }
    compact = {}
    for key, value in result.items():
        if key == "obj_value":
            continue
        compact[key] = _summarize_value(value)
    summary["details"] = compact
    return summary


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
    "error_analysis": "Error classification (code-level / model-level / data-level) and root cause. Empty string if success.",
    "recommendations": "Actionable suggestions for the responsible agent"
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

    def __init__(
        self,
        client,
        model,
        agent_id: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> None:
        self.client = get_llm_client(client)
        self.model = model
        self.name = self.NAME
        super().__init__(self.client, model, agent_id, temperature, top_p)

    # @override
    def forward_step(self, problem: Dict, message_pool: MessagePool):
        # Read framework execution result
        framework_exec = problem.get("_framework_execution", {})
        execution_outcome = framework_exec.get("execution_outcome", {})
        raw_result = framework_exec.get("raw_result")
        solver_log_content = framework_exec.get("solver_log", "")

        # Enhance execution_outcome with summary for LLM diagnosis
        if isinstance(raw_result, dict):
            execution_outcome["result_summary"] = _summarize_execution_result(raw_result)

        logger.info(f"TE execution: {execution_outcome.get('status')}")
        if execution_outcome.get("status") == "success":
            logger.info(f"TE obj_value: {execution_outcome.get('obj_value')}")

        # LLM diagnostic analysis
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
            diagnosis = {"error_analysis": response, "recommendations": ""}

        # Return structured diagnostic report
        return json.dumps(
            {
                "execution_status": execution_outcome.get("status", "unknown"),
                "obj_value": execution_outcome.get("obj_value"),
                "result_summary": execution_outcome.get("result_summary", {}),
                "error_analysis": diagnosis.get("error_analysis", ""),
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
