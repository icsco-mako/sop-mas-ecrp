import os
import re
import json
import pandas as pd
from typing import Dict, List, Any, Tuple


def load_config_file(config_file: str) -> Dict[str, Any]:
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
    return config


def read_file(dirpath: str, filename: str) -> str | Dict[str, Any]:
    filepath = os.path.join(dirpath, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def save_workflow_result(result: Dict[str, str], filepath: str):
    """保存工作流结果，包括所有智能体的输出和求解器日志
    
    Args:
        result: 字典，key为智能体名称，value为智能体输出内容
        filepath: 保存路径
    """
    for agent, msg in result.items():
        # 保存每个智能体的输出
        filename = os.path.join(filepath, f"{agent}.txt")
        with open(filename, "w", encoding="utf-8") as f:
            # Handle None values by writing a default message
            if msg is None:
                f.write(f"No response from {agent}")
            else:
                f.write(msg)
        
        # 特殊处理 Testing Engineer，提取并单独保存求解器日志
        if agent == "Testing Engineer" and msg:
            try:
                # 尝试解析 Testing Engineer 的 JSON 输出
                testing_output = json.loads(msg)
                if "solver_log" in testing_output and testing_output["solver_log"]:
                    solver_log_filename = os.path.join(filepath, "solver_log.txt")
                    with open(solver_log_filename, "w", encoding="utf-8") as f:
                        f.write(testing_output["solver_log"])
                    print(f"求解器日志已保存到: {solver_log_filename}")
            except json.JSONDecodeError:
                # 如果解析失败，不影响主流程
                pass
    
    print(f"工作流结果已成功保存到: {filepath}")
    print(f"共保存 {len(result)} 个智能体的输出")


def generate_data_schema(sample_data: list) -> dict:
    """从完整 sample 数据中生成精简版 schema，供 DE/ME/PY Agent 理解数据结构。

    策略：
    - 集合/列表：保留完整（如节点名）
    - 标量 dict（如 holding_cost: {H1: v}）：保留所有 key 和值
    - 矩阵 dict（如 reposition_cost: {i: {j: v}}）：只保留 2x2 子矩阵 + 维度说明
    """
    if isinstance(sample_data, dict):
        sample_data = [sample_data]

    schema_list = []
    for item in sample_data:
        if not isinstance(item, dict):
            schema_list.append(item)
            continue
        schema = {}
        for key, value in item.items():
            if isinstance(value, list):
                schema[key] = value
            elif isinstance(value, dict):
                schema[key] = _compress_matrix(value, key)
            else:
                schema[key] = value
        schema_list.append(schema)
    return schema_list


def _compress_matrix(d: dict, field_name: str = "") -> dict:
    """压缩矩阵 dict。

    对于二维矩阵（如 reposition_cost: {H1: {H2: v, ...}, ...}）：
      只保留前 2 个外层 key × 前 2 个内层 key 的子矩阵，其余用维度说明替代。
    对于一维 dict（如 holding_cost: {H1: v, H2: v, ...}）：
      保留所有 key，值用 0 替代。
    """
    # 检查是否为二维矩阵
    first_val = next(iter(d.values()), None)
    if isinstance(first_val, dict):
        # 二维矩阵：只保留 2×2 子矩阵
        outer_keys = list(d.keys())
        result = {}
        for ok in outer_keys[:2]:
            inner = d[ok]
            if isinstance(inner, dict):
                inner_keys = list(inner.keys())
                result[ok] = {ik: inner[ik] for ik in inner_keys[:2]}
            else:
                result[ok] = inner
        result[f"... ({len(outer_keys)} keys total: {', '.join(outer_keys)})"] = {}
        return result
    else:
        # 一维 dict：保留所有 key，值不变（值本身就是标量）
        return {k: v for k, v in d.items()}


def save_backtrack_result(result: List[Tuple[str, str, str]], filepath: str):
    for agent, status, msg in result:
        filename = os.path.join(filepath, f"{agent}_{status}.txt")
        with open(
            filename,
            "w",
            encoding="utf-8",
        ) as f:
            f.write(msg)
    print("backtrack result saved to database")

def dataset_loader(dataset, prob_name):
    match dataset:
        case "NLP4LP":
            ds = pd.read_csv("./data/NLP4LP.csv")
            prob_name = int(prob_name)
            assert (
                isinstance(prob_name, int) and 0 <= prob_name <= 269
            ), "prob_name must be an integer in [0,269]"
            description = ds.loc[prob_name, "description"]
            intput = json.loads(ds.loc[prob_name, "parameters"])
            output = json.loads(ds.loc[prob_name, "solution"])
            try:
                sample = [
                    {
                        "input": intput,
                        "output": [output["objective"]],
                        "solution": output["variables"],
                    }
                ]
            except KeyError:
                sample = None
        case "NLP4CSCO":
            # ds = pd.read_csv("./data/NLP4CSCO.csv")
            ds = pd.read_json("./data/NLP4CSCO.json")
            print(ds)
            prob_name = int(prob_name)
            assert (
                isinstance(prob_name, int) and 0 <= prob_name <= 26
            ), "prob_name must be an integer in [0,26]"
            print(ds.loc[prob_name, "problem_name"])
            description = ds.loc[prob_name, "description"]
            # sample_data = json.loads(ds.loc[prob_name, "sample"])
            sample_data = ds.loc[prob_name, "sample"]
            intput = sample_data[0]["input"]
            output = sample_data[0]["output"]
            try:
                sample = [
                    {
                        "input": intput,
                        "output": [output["objective_value"]],
                        # "solution": output["decision_variables"],
                    }
                ]
            except KeyError:
                sample = None
        case "NLP4ECR":
            dirpath = rf"./data/{dataset}/{prob_name}"  # ./data/NLP4ECR/prob_0
            description: str = read_file(dirpath, "description.txt")
            sample_raw = json.loads(read_file(dirpath, "sample.json"))
            # sample.json 可能是 dict 或 list，统一为 list
            if isinstance(sample_raw, dict):
                sample_raw = [sample_raw]
            sample_schema: list = generate_data_schema(sample_raw)
            full_data = sample_raw[0] if len(sample_raw) == 1 else sample_raw
            sample = [
                {
                    "input": sample_schema,   # 精简 schema，供 DE/ME/PY Agent
                    "data": full_data,         # 完整数据，供 TE Agent 执行
                    "output": "None",
                }
            ]

        case _:  # 默认情况，处理其他数据集
            dirpath = rf"./data/{dataset}/{prob_name}"  # ./data/ComplexOR/aircraft_assignment
            description: str = read_file(dirpath, "description.txt")
            sample: list = json.loads(read_file(dirpath, "sample.json"))
    problem = {"description": description, "sample": sample, "dataset": dataset}
    return problem


if __name__ == "__main__":
    output_agent = re.sub(r"^\W+|\W+$", "", "**OR Specialist**")
    print(output_agent)
