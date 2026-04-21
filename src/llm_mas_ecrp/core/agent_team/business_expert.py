import json
import logging
from typing import Dict
from json import JSONDecodeError
from .base_agent import BaseAgent
from llm_mas_ecrp.utils.user_proxy_config import get_llm_client
from ..message_pool import MessagePool

logger = logging.getLogger(__name__)


def _truncate_text(text: str, limit: int = 5000) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _compact_testing_output(testing_output_raw: str) -> str:
    """Keep only BA-relevant TE fields and drop full solver logs."""
    if not testing_output_raw:
        return ""

    try:
        parsed = json.loads(testing_output_raw)
    except JSONDecodeError:
        return _truncate_text(testing_output_raw, 3000)

    compact = {
        "execution_status": parsed.get("execution_status"),
        "obj_value": parsed.get("obj_value"),
        "result_summary": parsed.get("result_summary", {}),
        "business_summary": parsed.get("business_summary", {}),
        "diagnostic_summary": parsed.get("diagnostic_summary", ""),
        "solver_log_summary": parsed.get("solver_log_summary", ""),
        "recommendations": parsed.get("recommendations", ""),
        "error_analysis": parsed.get("error_analysis", ""),
    }
    return json.dumps(compact, ensure_ascii=False, indent=2)


class BusinessExpert(BaseAgent):
    """业务专家代理

    专门负责将数学优化结果（如Gurobi求解结果）转化为业务人员能够理解的自然语言解释。
    主要作用是减少建模专家和业务人员之间的沟通成本，将复杂的数学符号转化为清晰的业务含义。
    """

    NAME = "BusinessExpert"
    ROLE_DESCRIPTION = "You are a professional business expert specializing in translating mathematical optimization results into clear, actionable business insights. You convert complex mathematical notation (like Gurobi solver outputs) into natural language explanations that business stakeholders can easily understand and act upon."

    FORWARD_TASK = """
# ROLE: 优化结果业务分析专家 (Optimization Results Analyst)

## 核心任务

你是 **BusinessExpert**，专门负责将复杂的数学优化结果转化为业务人员能理解的深度分析报告。你不仅是翻译器，更是业务分析师，需要从优化结果中提炼出对管理层决策有直接价值的洞察。

**核心职责**：
1. **决策变量解释** - 将优化解转化为业务含义
2. **成本归因分析** - 解释各类成本的构成、驱动因素及占比
3. **关键发现** - 从结果中提炼出不直观但重要的业务洞察
4. **操作建议** - 给出面向管理者的具体可执行建议

---

## 输入数据

你将接收：
- **model_blueprint**: ModelExpert定义的变量含义和目标函数
- **optimization_results**: TestingEngineer压缩后的Gurobi求解结果摘要，其中可能包含 business_summary；若 business_summary 中已有精确 cost_breakdown，必须优先使用其中的数值，不要写“估算”
- **business_context**: 原始问题的业务背景信息

---

## 输出格式

返回JSON结构，**必须包含所有以下字段**：

```json
{
  "executive_summary": "一段话总结：优化达到了什么目标、关键结论是什么、对业务的意义",

  "cost_breakdown": {
    "total_cost": "总成本数值（如适用）",
    "cost_items": [
      {
        "category": "成本类别名称（如调运成本、存储成本、租箱成本）",
        "amount": "金额",
        "proportion": "占比百分比",
        "key_driver": "该成本的核心驱动因素是什么？是哪些决策变量或参数导致的？",
        "insight": "这个成本类别揭示了什么业务规律？"
      }
    ],
    "cost_structure_insight": "对整体成本结构的评价：哪个类别是关键杠杆？优化空间在哪里？"
  },

  "key_findings": [
    {
      "finding": "具体的业务发现（如：某节点调入量远大于调出量，说明它是净需求中心）",
      "evidence": "支撑这个发现的数据或变量值",
      "implication": "这个发现对运营决策的意义"
    }
  ],

  "actionable_recommendations": [
    {
      "recommendation": "具体的可操作建议（如：增加从A到B的调运频次）",
      "expected_impact": "预期影响（量化或定性描述）",
      "priority": "high/medium/low"
    }
  ],

  "what_if_insights": "基于当前解的假设分析：如果关键参数变化（如某成本系数上涨），可能的业务影响是什么？"
}
```

---

## 关键原则

1. **分析深度** - 不仅翻译数字，要挖掘数字背后的业务逻辑
2. **成本归因** - 每类成本都要解释"为什么是这个值"，指向具体驱动因素
3. **可操作性** - 建议必须是管理者可以立即执行的，不要空泛的建议
4. **业务语言** - 完全避免数学术语，使用业务人员熟悉的表达
5. **重点突出** - 聚焦最关键的3-5个发现和建议，不要罗列所有变量

---

## 特殊情况处理

- **求解失败/INFEASIBLE**: 在 executive_summary 说明失败原因，给出调整建议（哪些约束最可能需要放松）
- **TIME_LIMIT**: 说明当前可行解与最优解可能的差距，给出是否可接受的判断

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

    def __init__(
        self,
        client: str,
        model: str,
        agent_id: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> None:
        self.client = get_llm_client(client)
        self.model = model
        self.name = self.NAME
        super().__init__(self.client, model, agent_id, temperature, top_p)

    # @override
    def forward_step(self, problem: Dict, message_pool: MessagePool) -> str:
        """将优化结果转化为业务解读
        
        从 message_pool 中提取：
        - OR Specialist 的模型定义（变量含义、目标函数）
        - Testing Engineer 的执行结果（优化输出）
        - 原始问题描述作为业务上下文
        """
        # 获取模型蓝图（OR Specialist 的输出），避免过长模型文本挤占结果解释上下文
        model_blueprint = _truncate_text(message_pool.get("OR Specialist") or "", 5000)
        
        # 获取优化结果（Testing Engineer 的输出），仅保留压缩摘要，显式排除完整 solver_log
        testing_output_raw = message_pool.get("Testing Engineer") or ""
        optimization_results = _compact_testing_output(testing_output_raw)
        
        # 业务上下文来自原始问题描述
        business_context = _truncate_text(problem.get("description", ""), 2500)
        
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
