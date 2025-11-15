import re
import logging
from typing import List, Dict

from .base_agent import BaseAgent
from llm_mas_ecrp.utils.user_proxy_config import get_llm_client
from ..message_pool import MessagePool

logger = logging.getLogger(__name__)
class ProjectManager(BaseAgent):
    """项目管理Agent"""

    NAME = "project manager"
    ROLE_DESCRIPTION = (
        """you will take on the role of the conductor for a multi-agent system."""
    )
    FORWARD_TASK = """Now, you are presented with an operational optimization-related problem: {problem_description}. In this multi-agent system, there are many agents, each of whom is responsible for solving part of the problem. Your task is to CHOOSE THE NEXT AGENT TO CONSULT to solve the optimization problem. The names of the agents and their capabilities are listed below: {agent_list}. Experts that have already been selected include: {selected_agent_list}. Please select an agent to consult from the remaining agent names {allowed_agent_list}. Please note that the maximum number of asked agents is {max_collaborate_nums}, and you can ask {remaining_collaborate_nums} more times. You should output the name of agent directly. The next agent is ?
    
    # Decision Logic
    Select agents based on these priorities:
    1. Capability Alignment Score (relevance to current problem module)
    2. Remaining Invocation Quota: {remaining_collaborate_nums}/{max_collaborate_nums}
    3. Select agents sequentially from the {agent_list} sequence, without requiring continuity, but the relative order of elements in the {agent_list} must be maintained.

    Please provide your decision conclusion, The next agent is: <?>

    """
    FORWARD_TASK_INSTRUCTIONS = """
In this multi-agent system, there are many agents, each of whom is responsible for solving part of the problem. Your task is to CHOOSE THE NEXT AGENT TO CONSULT to solve the optimization problem. The names of the agents and their capabilities are listed below: {agent_list}. Please select the next agent to consult based on the following rules:

# PROBLEM DESCRIPTION
Current optimization problem to solve: {problem_description}

# SELECTION CRITERIA
1. **STRICT ORDERING REQUIREMENT** (HIGHEST PRIORITY):
   - You MUST select agents in the order they appear in {agent_list}
   - The agent_id sequence MUST be monotonically increasing: 0 → 1 → 2 → 3 → 4 → ...
   - If the first agent (agent_id=0) has NOT been selected yet, you MUST select it first
   - After selecting agent_id=N, you MUST select from agents with agent_id > N
   - NEVER skip agents or select out of order unless an agent has already been selected

2. Available Options:
   - Already selected: {selected_agent_list}
   - Available pool: {allowed_agent_list}
   - Remaining quota: {remaining_collaborate_nums} / {max_collaborate_nums}

3. Selection Logic:
   - From {allowed_agent_list}, select the agent with the SMALLEST agent_id
   - This ensures sequential execution following the designed workflow

# WORKFLOW RATIONALE
The agent order is designed to follow a natural problem-solving workflow:
- DataEngineer (agent_id=0): Extract parameters and variables from problem description
- OR Specialist (agent_id=1): Build mathematical model based on extracted information
- Python Developer (agent_id=2): Implement the model in code
- Testing Engineer (agent_id=3): Generate test cases and validate the code
- BusinessExpert (agent_id=4): Translate results into business insights

# ACTION REQUIREMENTS
Output ONLY the selected agent's name (no explanation needed). 
You MUST choose from {allowed_agent_list}.
Next agent to consult: <?>
    """

    def __init__(
        self,
        client: str,
        model: str,
        agent_list: List[BaseAgent],
        max_collaborate_nums: int,
    ):
        self.client = get_llm_client(client)
        self.model = model
        self.name = self.NAME
        self.agent_list = agent_list
        self.max_collaborate_nums = max_collaborate_nums
        super().__init__(self.client, model, 0)

    def forward_step(
        self,
        problem_description: str,
        message_pool: MessagePool,
    ):
        selected_agent_list = message_pool.get_spoken_agents()  # 返回 List[BaseAgent]
        # 获取已选智能体的 agent_id 集合
        selected_agent_ids = {agent.agent_id for agent in selected_agent_list}
        
        # 如果有已选智能体，下一个候选起始位置应该是最后一个已选智能体的下一个位置
        # 但同时要排除所有已选过的智能体（防止重复选择）
        i = selected_agent_list[-1].agent_id + 1 if selected_agent_list else 0

        current_step = len(selected_agent_list)
        all_agents = [
            {
                "agent_name": agent.NAME,
                "agent_capability": agent.ROLE_DESCRIPTION,
                "agent_id": idx,
            }
            for idx, agent in enumerate(self.agent_list)
        ]
        
        # 构建允许选择的智能体列表：从位置 i 开始，但排除已选过的
        # 注意：这里比较的是 agent_id（即列表索引 j），而不是其他属性
        allowed_agent_list = [
            all_agents[j] 
            for j in range(i, len(all_agents)) 
            if all_agents[j]["agent_id"] not in selected_agent_ids  # 修复：使用 agent_id 而不是索引 j
        ]
        
        # 如果从 i 开始没有可选的了，从头开始寻找未选过的
        if not allowed_agent_list:
            allowed_agent_list = [
                all_agents[j] 
                for j in range(len(all_agents)) 
                if all_agents[j]["agent_id"] not in selected_agent_ids  # 修复：使用 agent_id 而不是索引 j
            ]
        
        # 如果没有可选的智能体了，抛出异常
        if not allowed_agent_list:
            raise ValueError("All agents have been selected, cannot select more agents")
        
        prompt = self.FORWARD_TASK_INSTRUCTIONS.format(
            problem_description=problem_description,
            agent_list=all_agents,
            selected_agent_list=[
                {
                    "agent_name": agent.NAME,
                    "agent_capability": agent.ROLE_DESCRIPTION,
                    "agent_id": agent.agent_id,
                }
                for agent in selected_agent_list
            ],
            allowed_agent_list=allowed_agent_list,
            max_collaborate_nums=self.max_collaborate_nums,
            remaining_collaborate_nums=self.max_collaborate_nums - current_step,
        )
        output_agent: str = self.llm_call(prompt)
        # 使用正则表达式去除两端的非字母数字字符
        output_agent = re.sub(r"^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$", "", output_agent)
        
        logger.info(f"selected_agent_list: {[agent.NAME for agent in selected_agent_list]}")
        logger.info(f"selected_agent_ids: {selected_agent_ids}")
        logger.info(f"allowed_agent_list: {[(agent['agent_name'], agent['agent_id']) for agent in allowed_agent_list]}")
        logger.info(f"LLM output: '{output_agent}'")
        logger.info(f"remain: {self.max_collaborate_nums - current_step}")
        
        # 关键修复：只在 allowed_agent_list 中匹配，防止选择已选过的智能体
        for allowed_agent in allowed_agent_list:
            if allowed_agent["agent_name"] == output_agent:
                # 从 agent_list 中找到对应的 agent 对象
                for agent in self.agent_list:
                    if agent.agent_id == allowed_agent["agent_id"]:
                        logger.info(f"Successfully matched agent: {agent.NAME} (agent_id={agent.agent_id})")
                        return agent
        
        # 如果 LLM 输出不在 allowed_agent_list 中，使用回退策略：选择第一个可用的智能体
        logger.warning(f"LLM output '{output_agent}' not in allowed list. Using fallback: selecting first available agent.")
        first_allowed = allowed_agent_list[0]
        for agent in self.agent_list:
            if agent.agent_id == first_allowed["agent_id"]:
                logger.info(f"Fallback selected: {agent.NAME} (agent_id={agent.agent_id})")
                return agent

        raise ValueError(f"No matching agent found for name: {output_agent}")

    def final_inspect(self, problem, message_pool):
        """inspect the final result"""
        prompt = self.BACKWORD_TASK.format(
            problem_description=problem,
            selected_agent_list=message_pool.get_speakers(),
            message_pool=message_pool.get_content(),
        )
        response = self.llm_call(prompt)
        return response
