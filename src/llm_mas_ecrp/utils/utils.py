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
            sample: list = json.loads(read_file(dirpath, "sample.json"))
            sample = [
                {
                    "input": sample,
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
