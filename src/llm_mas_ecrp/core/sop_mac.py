import json
import math
import logging
import sys
import time
import gurobipy
from typing import List, Dict, Tuple, Any
from llm_mas_ecrp.core.agent_team.base_agent import BaseAgent
from llm_mas_ecrp.core.message_pool import MessagePool
from llm_mas_ecrp.core.agent_team.project_manager import ProjectManager
from llm_mas_ecrp.utils.logger import log_separator


logger = logging.getLogger(__name__)


def _is_failed_result(result: Dict) -> bool:
    status = str(result.get("status", "")).lower()
    if status in {"error", "failure", "failed", "infeasible", "unbounded", "inf_or_unbd", "infeasible_or_unbounded"}:
        return True
    return result.get("obj_value") is None


def __create_agent(agent_config: List):
    try:
        agent_class = agent_config["class"]
        args = agent_config["args"]
        kwargs = agent_config.get("kwargs", {})
        return agent_class(*args, **kwargs)
    except (TypeError, ValueError) as e:
        logging.error(f"Failed to create instance of {agent_class.__name__}: {e}")
        return None


def __iterative_modeling(
    problem: Dict,
    project_manager: ProjectManager,
    stack: List[BaseAgent],
    message_pool: MessagePool,
    max_collaborate_nums: int,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """迭代建模过程，返回结果和每个智能体的运行时间"""
    result = {}
    agent_times = {}  # 记录每个智能体的运行时间
    
    for i in range(max_collaborate_nums):
        # 选择智能体
        agent: BaseAgent = project_manager.forward_step(
            problem["description"], message_pool
        )
        
        log_separator(logger, f"Step {i+1}/{max_collaborate_nums}: {agent.name}", "-", 80)
        logger.info(f"选中智能体: {agent.name} | 使用模型: {agent.model}")
        
        # 记录智能体开始时间
        agent_start_time = time.time()
        
        # 执行智能体任务
        msg: str = agent.forward_step(problem, message_pool)
        
        # 计算智能体运行时间
        agent_end_time = time.time()
        agent_execution_time = agent_end_time - agent_start_time
        
        # 保存智能体运行时间（如果同一智能体多次调用，累加时间）
        if agent.NAME in agent_times:
            agent_times[agent.NAME] += agent_execution_time
        else:
            agent_times[agent.NAME] = agent_execution_time
        
        # 格式化时间显示
        minutes = int(agent_execution_time // 60)
        seconds = agent_execution_time % 60
        time_str = f"{minutes}分{seconds:.2f}秒" if minutes > 0 else f"{seconds:.2f}秒"
        
        logger.info(f"{agent.name} 执行完成 | 耗时: {time_str}")
        logger.info(f"{agent.name} 响应内容预览: {msg[:200]}..." if len(msg) > 200 else f"{agent.name} 响应: {msg}")
        
        message_pool.add_message(agent, msg, i)
        stack.append(agent)
        result[agent.NAME] = msg
    
    return result, agent_times


def __exception_traceback(
    problem: Dict,
    project_manager: ProjectManager,
    error_info: str,
    stack: List[BaseAgent],
    message_pool: MessagePool,
) -> Tuple[bool, List[Tuple[BaseAgent, Dict[str, str]]]]:
    logger.info("Starting backtracking process")
    is_error_found = False
    backtrack_hist_log = []
    while stack:
        agent: BaseAgent = stack.pop()
        hist_msg: Dict = message_pool.pop()
        response: Dict[str, str] = agent.backward_step(
            problem, hist_msg["content"], str(error_info)
        )
        backtrack_hist_log.append((agent, response))
        logger.info(f"回溯日志：\n{agent}\n{response}")
        if response["is_caused_by_you"]:
            logger.info(f"Error source found: {agent.name}")
            logger.info(f"error reason: {response['reason']}")
            stack.append(agent)
            message_pool.add_message(
                agent,
                response["refined_result"],
                -1,
            )
            return True, backtrack_hist_log

    return False, backtrack_hist_log


def __run_str_code(
    problem: Dict,
    message_pool: MessagePool,
) -> Dict:
    try:
        # Prefer TE-Agent's execution result if available
        te_output = message_pool.get("Testing Engineer")
        if te_output:
            try:
                te_data = json.loads(te_output)
                if (te_data.get("execution_status") == "success"
                        and te_data.get("obj_value") is not None):
                    logger.info("Using TE-Agent execution result")
                    te_result = {"obj_value": te_data["obj_value"]}
                    signature = _extract_formulation_signature(problem, message_pool)
                    if signature is not None:
                        te_result["formulation_signature"] = signature
                    return te_result
            except (json.JSONDecodeError, KeyError):
                pass

        return _run_python_developer_code(problem, message_pool)
    except Exception as e:
        logger.error(f"Unexpected error in run_str_code: {str(e)}")
        return {"obj_value": None, "error_type": "execution_error", "error_msg": str(e)}


def _extract_formulation_signature(
    problem: Dict,
    message_pool: MessagePool,
) -> Dict[str, int] | None:
    """Best-effort signature extraction used when TE already solved the code."""
    try:
        result = _run_python_developer_code(problem, message_pool)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Unable to extract formulation signature: %s", exc)
        return None
    return result.get("formulation_signature") if isinstance(result, dict) else None


def _run_python_developer_code(
    problem: Dict,
    message_pool: MessagePool,
) -> Dict:
    try:
        globals_dict = dict(globals())
        globals_dict["gp"] = gurobipy
        globals_dict["GRB"] = gurobipy.GRB
        globals_dict["__builtins__"] = __builtins__

        code = message_pool.get("Python Developer")
        exec(code, globals_dict)
        if "optimize" not in globals_dict:
            logger.error("optimize function not found in generated code")
            return {
                "obj_value": None,
                "error_type": "missing_function",
                "error_msg": "optimize function not found",
            }

        data = problem["sample"][0].get("data", problem["sample"][0]["input"])
        result, signature = _call_optimize_with_signature(
            globals_dict["optimize"], data
        )

        if not isinstance(result, dict):
            logger.error("Invalid result type: expected dict")
            error_result = {
                "obj_value": None,
                "error_type": "result dict, invalid__type",
                "error_msg": f"Expected dict, got {type(result)}",
            }
            if signature is not None:
                error_result["formulation_signature"] = signature
            return error_result

        if signature is not None:
            result["formulation_signature"] = signature

        if _is_failed_result(result):
            error_msg = result.get("message") or result.get("error_msg") or f"invalid solve result: {result}"
            logger.error(error_msg)
            return {
                "obj_value": None,
                "error_type": "solver_or_model_failure",
                "error_msg": error_msg,
                "formulation_signature": result.get("formulation_signature"),
            }

        if "obj_value" not in result:
            error_msg = "key `obj_value` not found in result"
            logger.error(error_msg)
            return {
                "obj_value": None,
                "error_type": "missing_obj_value",
                "error_msg": error_msg,
                "formulation_signature": result.get("formulation_signature"),
            }

        if result["obj_value"] is None:
            error_msg = "obj_value is None, model infeasible!!!"
            logger.error(error_msg)
            return {
                "obj_value": None,
                "error_msg": error_msg,
                "formulation_signature": result.get("formulation_signature"),
            }

        return result
    except Exception as e:
        logger.error(f"Unexpected error while executing Python Developer code: {str(e)}")
        return {"obj_value": None, "error_type": "execution_error", "error_msg": str(e)}


def _call_optimize_with_signature(optimize_fn, data: Dict) -> Tuple[Any, Dict[str, int] | None]:
    """Run optimize(data) and capture the last Gurobi model visible in locals."""
    models = []
    signature_holder: Dict[str, Dict[str, int] | None] = {"signature": None}
    generated_filename = getattr(getattr(optimize_fn, "__code__", None), "co_filename", None)
    previous_trace = sys.gettrace()

    def _trace(frame, event, arg):
        if event in {"line", "return"} and frame.f_code.co_filename == generated_filename:
            for value in frame.f_locals.values():
                if isinstance(value, gurobipy.Model):
                    if all(value is not m for m in models):
                        models.append(value)
                    try:
                        # Capture as the generated code runs; some models are
                        # disposed before the function's return event completes.
                        signature_holder["signature"] = _formulation_signature(value)
                    except gurobipy.GurobiError:
                        pass
        return _trace

    sys.settrace(_trace)
    try:
        result = optimize_fn(data)
    finally:
        sys.settrace(previous_trace)

    if signature_holder["signature"] is not None:
        return result, signature_holder["signature"]

    try:
        signature = _formulation_signature(models[-1]) if models else None
    except gurobipy.GurobiError:
        signature = None
    return result, signature


def _formulation_signature(model: gurobipy.Model) -> Dict[str, int]:
    """Return a coarse structural signature for formulation-level diagnostics."""
    try:
        model.update()
    except gurobipy.GurobiError:
        pass
    return {
        "n_vars": int(model.NumVars),
        "n_constrs": int(model.NumConstrs),
        "n_bin": int(model.NumBinVars),
        "n_int": int(model.NumIntVars),
        "nnz": int(model.NumNZs),
    }


def __verify_result(
    result: Dict, sample_result: float, rel_tol: float = 1e-6
) -> Tuple[bool, str]:
    if result.get("obj_value") is None:
        return False, result["error_msg"]

    is_success = math.isclose(result["obj_value"], sample_result, rel_tol=rel_tol)
    if not is_success:
        return (
            False,
            f"result not match. Expected: {sample_result}, Actual: {result['obj_value']}",
        )

    return True, "None"


def sop_mac(
    problem: Dict,
    max_collaborate_nums: int = 2,
    is_backtrack: bool = True,
    AGENTs_CONFIG: list = [],
):
    stack = []
    agent_list = [
        __create_agent(agent_config)
        for agent_config in AGENTs_CONFIG
        if __create_agent(agent_config) is not None
    ]
    pm = ProjectManager(
        client="DeepSeek",
        model="deepseek-chat",
        agent_list=agent_list,
        max_collaborate_nums=max_collaborate_nums,
    )

    message_pool = MessagePool()
    
    log_separator(logger, "Forward Modeling Process", "=", 80)
    logger.info("====================迭代建模开始====================\n")
    
    # 执行前向建模，获取结果和智能体运行时间
    fw_res, agent_times = __iterative_modeling(
        problem=problem,
        project_manager=pm,
        stack=stack,
        message_pool=message_pool,
        max_collaborate_nums=max_collaborate_nums,
    )
    
    # 输出智能体运行时间统计
    log_separator(logger, "Agent Execution Time Summary", "=", 80)
    total_agent_time = 0
    for agent_name, exec_time in agent_times.items():
        minutes = int(exec_time // 60)
        seconds = exec_time % 60
        time_str = f"{minutes}分{seconds:.2f}秒" if minutes > 0 else f"{seconds:.2f}秒"
        logger.info(f"{agent_name:<30} | 耗时: {time_str:>15} ({exec_time:.2f}秒)")
        total_agent_time += exec_time
    
    total_minutes = int(total_agent_time // 60)
    total_seconds = total_agent_time % 60
    total_time_str = f"{total_minutes}分{total_seconds:.2f}秒" if total_minutes > 0 else f"{total_seconds:.2f}秒"
    logger.info(f"{'='*80}")
    logger.info(f"所有智能体总耗时: {total_time_str} ({total_agent_time:.2f}秒)\n")
    
    log_separator(logger, "Code Execution", "=", 80)
    result: Dict = __run_str_code(problem, message_pool)
    logger.info(f"代码执行结果: {result}\n")
    
    if problem["dataset"] == "NLP4ECR":
        if result.get("obj_value") is not None:
            is_success = True
            error_msg = "None"
            logger.info(f"NLP4ECR: obj_value={result['obj_value']}, auto-accepted")
        else:
            is_success = False
            error_msg = result.get("error_msg", "obj_value is None")
    else:
        [sample_result] = problem["sample"][0]["output"]
        is_success, error_msg = __verify_result(result, sample_result)

    if is_success:
        log_separator(logger, "SUCCESS", "=", 80)
        logger.info("✅ 求解成功!\n")
        return {
            "forward_result": fw_res,
            "backward_result": [],
            "agent_times": agent_times,
            "status": True,
            "error_msg": "None",
            "obj_value": result.get("obj_value"),
            "formulation_signature": result.get("formulation_signature"),
        }

    bw_res = []
    if is_backtrack:
        log_separator(logger, "Backward Tracking Process", "=", 80)
        logger.info("====================异常回溯开始====================\n")

        error_info: str = json.dumps(result)
        try:
            is_error_found, bw_res = __exception_traceback(
                problem, pm, error_info, stack, message_pool
            )
        except Exception as e:
            logger.error(f"Backward tracking failed: {e}")
            is_error_found = False
            bw_res = []
        if is_error_found:
            logger.info("回溯后重新执行...\n")
            # After backtracking, only run remaining unselected agents
            remaining = max_collaborate_nums - len(message_pool.get_spoken_agents())
            fw_res_retry = {}
            agent_times_retry = {}
            if remaining > 0:
                fw_res_retry, agent_times_retry = __iterative_modeling(
                    problem=problem,
                    project_manager=pm,
                    stack=stack,
                    message_pool=message_pool,
                    max_collaborate_nums=remaining,
                )
            # 合并智能体时间
            for agent_name, exec_time in agent_times_retry.items():
                if agent_name in agent_times:
                    agent_times[agent_name] += exec_time
                else:
                    agent_times[agent_name] = exec_time

            result: Dict = __run_str_code(problem, message_pool)
            if problem["dataset"] == "NLP4ECR":
                is_success = result.get("obj_value") is not None
                new_error_msg = "None" if is_success else result.get("error_msg", "obj_value is None")
            else:
                is_success, new_error_msg = __verify_result(result, sample_result)
            if is_success:
                log_separator(logger, "SUCCESS AFTER BACKTRACKING", "=", 80)
                logger.info(
                    f"✅ 回溯后求解成功! sop-mac: {result['obj_value']} = sample_result: {sample_result}\n"
                )
                return {
                    "forward_result": fw_res_retry,
                    "backward_result": bw_res,
                    "agent_times": agent_times,
                    "status": True,
                    "error_msg": "None",
                    "obj_value": result.get("obj_value"),
                    "formulation_signature": result.get("formulation_signature"),
                }
            error_msg = new_error_msg

    log_separator(logger, "FAILED", "=", 80)
    logger.error(f"❌ 求解失败: {error_msg}\n")
    return {
        "forward_result": fw_res,
        "backward_result": bw_res,
        "agent_times": agent_times,
        "status": False,
        "error_msg": error_msg,
        "obj_value": result.get("obj_value") if isinstance(result, dict) else None,
        "formulation_signature": result.get("formulation_signature") if isinstance(result, dict) else None,
    }
