import json
import logging
from typing import Dict
from json import JSONDecodeError
from .base_agent import BaseAgent
from llm_mas_ecrp.utils.user_proxy_config import get_llm_client
from ..message_pool import MessagePool

logger = logging.getLogger(__name__)


class BusinessExpert(BaseAgent):
    """业务专家代理

    专门负责将数学优化结果（如Gurobi求解结果）转化为业务人员能够理解的自然语言解释。
    主要作用是减少建模专家和业务人员之间的沟通成本，将复杂的数学符号转化为清晰的业务含义。
    """

    NAME = "BusinessExpert"
    ROLE_DESCRIPTION = "You are a professional business expert specializing in translating mathematical optimization results into clear, actionable business insights. You convert complex mathematical notation (like Gurobi solver outputs) into natural language explanations that business stakeholders can easily understand and act upon."

    FORWARD_TASK = """
# ROLE: 优化结果业务解释专家 (Optimization Results Interpreter)

## 核心任务

你是 **BusinessExpert**，专门负责将复杂的数学优化结果转化为业务人员能理解的简洁解释。

**主要职责**：解释清楚三个核心内容
1. **决策变量的业务含义** - 将x[1,2]=23转化为"从A仓库向B客户运送23单位"
2. **目标函数的业务价值** - 说明优化达到的业务目标和量化收益
3. **Gurobi日志关键信息** - 解释求解状态和重要统计信息

---

## 输入数据

你将接收：
- **model_blueprint**: ModelExpert定义的变量含义和目标函数
- **optimization_results**: PyDeveloper的Gurobi求解结果
- **business_context**: 基本的业务背景信息

---

## 输出格式

返回简洁的JSON结构：

```json
{
  "decision_variables_explanation": [
    {
      "variable": "x[1,2]=23",
      "business_meaning": "从A仓库向B客户运送23单位货物",
      "why_this_value": "为了满足客户需求并最小化运输成本"
    }
  ],
  "objective_function_result": {
    "achieved_value": "目标函数的具体数值",
    "business_interpretation": "这个数值在业务上代表什么（如总成本、总利润等）",
    "performance_assessment": "这个结果的好坏评价"
  },
  "solver_status_summary": {
    "solution_quality": "OPTIMAL/FEASIBLE/INFEASIBLE等状态的业务含义",
    "key_statistics": "求解时间、迭代次数等关键信息的简要说明",
    "reliability": "解的可靠性和实用性评估"
  },
  "executive_summary": "一句话总结整个优化结果的业务价值"
}
```

---

## 关键原则

1. **简洁明了** - 避免复杂的业务分析，专注核心解释
2. **精确映射** - 严格按照ModelExpert的变量定义转译
3. **业务语言** - 完全避免数学术语，使用业务人员熟悉的表达
4. **重点突出** - 突出最重要的决策和结果

---

## 特殊情况处理

- **INFEASIBLE**: "当前约束条件下无可行方案，需调整业务规则"
- **UNBOUNDED**: "目标可无限改善，需检查约束设置"
- **TIME_LIMIT**: "在时间限制内找到的最佳方案"
- **变量值为0**: "该选项未被采用的业务原因"
- **分数值**: "建议的实际执行方案（如23.7→运送24单位）"

**输入数据**: 包含完整的模型定义、优化结果和业务上下文的数据包
"""

    BACKWARD_TASK = """当你在解决问题时，收到了外部环境的反馈。你需要判断这是你导致的问题还是其他专家导致的问题（其他专家在你之前已经给出了一些结果）。如果是你的问题，你需要给出解决方案和改进后的业务解读。

原始问题如下：
{problem_description}

你之前给出的业务解读如下：
{previous_business_interpretation}

收到的反馈如下：
{feedback}

输出格式为如下的 JSON 结构：
{{
    "is_caused_by_you": false,
    "reason": "如果问题不是由你引起的，请留空字符串",
    "refined_result": "如果错误是由你引起的，请提供改进后的业务解读..."
}}
"""

    def __init__(self, client: str, model: str, agent_id: int) -> None:
        self.client = get_llm_client(client)
        self.model = model
        self.name = self.NAME
        super().__init__(self.client, model, agent_id)

    # @override
    def forward_step(self, problem: Dict, message_pool: MessagePool) -> str:
        """将优化结果转化为业务解读
        
        从 message_pool 中提取：
        - OR Specialist 的模型定义（变量含义、目标函数）
        - Testing Engineer 的执行结果（优化输出）
        - 原始问题描述作为业务上下文
        """
        # 获取模型蓝图（OR Specialist 的输出）
        model_blueprint = message_pool.get("OR Specialist") or ""
        
        # 获取优化结果（Testing Engineer 的输出）
        testing_output_raw = message_pool.get("Testing Engineer") or ""
        optimization_results = testing_output_raw
        
        # 尝试解析 TestingEngineer 的 JSON 输出
        try:
            parsed = json.loads(testing_output_raw)
            # TestingEngineer 返回格式：{"testing_case_generated": {...}, "testing_result": "..."}
            optimization_results = parsed.get("testing_result", testing_output_raw)
        except JSONDecodeError:
            # 如果解析失败，直接使用原始字符串
            optimization_results = testing_output_raw
        
        # 业务上下文来自原始问题描述
        business_context = problem.get("description", "")
        
        # 构建提示词
        prompt = (
            f"{self.FORWARD_TASK}\n\n"
            f"# 输入数据包\n"
            f"## model_blueprint (OR Specialist 的模型定义):\n{model_blueprint}\n\n"
            f"## optimization_results (Testing Engineer 的执行结果):\n{optimization_results}\n\n"
            f"## business_context (原始问题描述):\n{business_context}\n\n"
            "请严格按照上述'输出格式'以 JSON 输出，并使用```json```包裹；字段名必须与示例保持一致。"
        )
        
        response_str: str = self.llm_call(prompt=prompt)
        response = self.strip_str("```json", response_str)
        
        # 验证输出是否为有效 JSON
        try:
            parsed_response = json.loads(response)
            logger.info(f"BusinessExpert 成功生成业务解读，包含 {len(parsed_response)} 个字段")
        except JSONDecodeError as e:
            logger.warning(f"BusinessExpert 输出的 JSON 格式有误: {e}")
        
        return response

    # @override
    def backward_step(  # type: ignore[override]
        self, problem: Dict, hist_msg: str, feedback: str
    ) -> Dict:
        """回溯步骤：判断反馈是否由自己引起，并给出改进方案"""
        prompt = self.BACKWARD_TASK.format(
            problem_description=problem.get("description", ""),
            previous_business_interpretation=hist_msg,
            feedback=feedback,
        )
        backtrack_response: str = self.llm_call(prompt=prompt)
        backtrack_response = self.strip_str("```json", backtrack_response)
        
        try:
            return json.loads(backtrack_response)  # str -> dict
        except JSONDecodeError as e:
            logger.error(
                f"Error: {e}.\n BusinessExpert backward response content is:\n{backtrack_response}"
            )
            # 返回默认回溯结果
            return {
                "is_caused_by_you": False,
                "reason": "",
                "refined_result": ""
            }