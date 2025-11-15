import json
import os
import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Callable, Union, Optional
from dataclasses import dataclass
import pandas as pd
import multiprocessing
from tabulate import tabulate
from llm_mas_ecrp.core.sop_mac import sop_mac
from llm_mas_ecrp.utils.utils import save_workflow_result, dataset_loader
from llm_mas_ecrp.utils.logger import config_logger, log_separator
from llm_mas_ecrp.core.agent_team.data_engineer import DataEngineer
from llm_mas_ecrp.core.agent_team.or_specialist import ModelEngineer
from llm_mas_ecrp.core.agent_team.python_developer import PythonDeveloper
from llm_mas_ecrp.core.agent_team.testing_engineer import TestingEngineer
from llm_mas_ecrp.core.agent_team.business_expert import BusinessExpert


@dataclass
class Config:
    algorithm: str
    dataset: str
    prob_name: Union[str, int]
    max_collaborate_nums: int
    is_backtrack: bool

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """工厂方法"""
        return cls(
            algorithm=args.algorithm,
            dataset=args.dataset,
            prob_name=args.prob_name,
            max_collaborate_nums=args.max_collaborate_nums,
            is_backtrack=args.is_backtrack,
        )


class ResultLogger:
    LOG_TEMPLATES = {
        "sop_mac": {
            "values": lambda result, config: [
                config.algorithm,
                f"{config.dataset}/{config.prob_name}",
                config.max_collaborate_nums,
                config.is_backtrack,
                result["status"],
                result["error_msg"],
            ],
            "index": [
                "algorithm",
                "dataset",
                "max_collaborate_nums",
                "is_backward",
                "solve_status",
                "err_msg",
            ],
        },
        "spm": {
            "values": lambda result, config: [
                config.algorithm,
                f"{config.dataset}/{config.prob_name}",
                result["status"],
                result["error_msg"],
            ],
            "index": ["algorithm", "dataset", "solve_status", "err_msg"],
        },
        "cot": {
            "values": lambda result, config: [
                config.algorithm,
                f"{config.dataset}/{config.prob_name}",
                result["status"],
                result["error_msg"],
            ],
            "index": ["algorithm", "dataset", "solve_status", "err_msg"],
        },
    }

    @staticmethod
    def log_result(result: Dict, config: Config, logger: logging.Logger) -> None:
        """记录并格式化输出结果"""
        template = ResultLogger.LOG_TEMPLATES[config.algorithm]
        log_info = pd.Series(
            template["values"](result, config), index=template["index"]
        )
        
        log_separator(logger, "Execution Summary", "=", 80)
        logger.info(
            "\n"
            + tabulate(
                log_info.to_frame(), headers=["Attribute", "Value"], tablefmt="grid"
            )
        )
        
        # 如果结果中包含智能体运行时间，也一并输出
        if "agent_times" in result:
            logger.info("\n" + "="*80)
            logger.info("Agent Execution Time Details:")
            logger.info("="*80)
            for agent_name, exec_time in result["agent_times"].items():
                minutes = int(exec_time // 60)
                seconds = exec_time % 60
                time_str = f"{minutes}分{seconds:.2f}秒" if minutes > 0 else f"{seconds:.2f}秒"
                logger.info(f"  {agent_name:<30} | 耗时: {time_str:>15}")
        
        # 输出总执行时间（如果有）
        if "execution_time" in result:
            logger.info("\n" + "="*80)
            logger.info(f"Total Execution Time: {result.get('execution_time_str', 'N/A')}")
            logger.info("="*80 + "\n")


class ProblemSolver:
    def __init__(self, config: Config):
        self.cfg = config
        self.filepath = self._setup_filepath()  # _模块内部函数
        self.logger = config_logger(self.filepath)

    def _setup_filepath(self) -> str:
        filepath = Path(
            "results",
            self.cfg.algorithm,
            f"{self.cfg.dataset}_k{self.cfg.max_collaborate_nums}",
            str(self.cfg.prob_name),
        )
        filepath.mkdir(parents=True, exist_ok=True)
        return str(filepath)

    def solve(self) -> None:
        try:
            # 输出算例信息头
            log_separator(self.logger, "Problem Information", "=", 80)
            self.logger.info(f"算法: {self.cfg.algorithm}")
            self.logger.info(f"数据集: {self.cfg.dataset}")
            self.logger.info(f"算例名称: {self.cfg.prob_name}")
            self.logger.info(f"最大协作智能体数: {self.cfg.max_collaborate_nums}")
            self.logger.info(f"是否回溯: {self.cfg.is_backtrack}")
            self.logger.info(f"输出路径: {self.filepath}\n")

            problem = dataset_loader(self.cfg.dataset, self.cfg.prob_name)
            result = self._run_algorithm(problem)

            if self.cfg.algorithm == "sop_mac" and "forward_result" in result:
                save_workflow_result(result["forward_result"], self.filepath)

            ResultLogger.log_result(result, self.cfg, self.logger)

        except Exception as e:
            log_separator(self.logger, "ERROR", "=", 80)
            self.logger.error(f"处理算例时发生错误: {str(e)}")
            self.logger.error(f"错误类型: {type(e).__name__}")
            import traceback
            self.logger.error(f"堆栈跟踪:\n{traceback.format_exc()}")
            raise
        finally:
            logging.shutdown()

    def _run_algorithm(self, problem: Dict) -> Dict:
        algorithms = {
            "sop_mac": lambda: sop_mac(
                problem,
                self.cfg.max_collaborate_nums,
                self.cfg.is_backtrack,
                [
                    {
                        "class": DataEngineer,
                        "args": ["DeepSeek", "deepseek-v3-250324", 0],  # agent_id=0
                    },
                    {"class": ModelEngineer, "args": ["DeepSeek", "deepseek-v3-250324", 1]},  # agent_id=1
                    {"class": PythonDeveloper, "args": ["DeepSeek", "deepseek-v3-250324", 2]},  # agent_id=2
                    {"class": TestingEngineer, "args": ["DeepSeek", "deepseek-v3-250324", 3]},  # agent_id=3
                    {"class": BusinessExpert, "args": ["DeepSeek", "deepseek-v3-250324", 4]},  # agent_id=4
                ],
            ),
            # "spm": lambda: SPM(problem),
            # "cot": lambda: CoT(problem),
        }

        algorithm = algorithms.get(self.cfg.algorithm)
        if not algorithm:
            raise ValueError(f"Unsupported algorithm: {self.cfg.algorithm}")

        # 记录算法开始执行时间
        start_time = time.time()
        
        log_separator(self.logger, f"Algorithm Execution: {self.cfg.algorithm}", "=", 80)
        self.logger.info(f"开始执行算法: {self.cfg.algorithm}")
        self.logger.info(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # 执行算法
        result = algorithm()
        
        # 记录算法结束时间并计算执行时长
        end_time = time.time()
        execution_time = end_time - start_time
        
        # 格式化执行时间
        hours = int(execution_time // 3600)
        minutes = int((execution_time % 3600) // 60)
        seconds = execution_time % 60
        
        time_str = f"{hours}小时{minutes}分{seconds:.2f}秒" if hours > 0 else \
                   f"{minutes}分{seconds:.2f}秒" if minutes > 0 else \
                   f"{seconds:.2f}秒"
        
        # 输出到日志
        log_separator(self.logger, "Algorithm Completed", "=", 80)
        self.logger.info(f"算法执行完成: {self.cfg.algorithm}")
        self.logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"总执行时间: {time_str} ({execution_time:.2f}秒)\n")
        
        # 将执行时间添加到结果中
        if isinstance(result, dict):
            result['execution_time'] = execution_time
            result['execution_time_str'] = time_str
        
        return result


def get_problem_names(config: Config) -> List[Union[str, int]]:
    if config.dataset == "ComplexOR":
        dataset_path = Path("data", config.dataset)
        return [d for d in os.listdir(dataset_path)]
    elif config.dataset == "NLP4LP":
        if len(str(config.prob_name)) > 1:
            return [int(p) for p in str(config.prob_name).split(",")]
        return list(range(int(config.prob_name), 269))
    elif config.dataset == "NLP4CSCO":
        dataset_path = Path("data", config.dataset)
        if dataset_path.exists():
            return [d for d in os.listdir(dataset_path) if not d.startswith(".")]
        else:
            # 如果目录不存在，返回默认的问题名称
            return [f"prob_{i}" for i in range(1, 11)]
    raise ValueError(f"Unsupported dataset:")


def main(config: Config) -> None:
    solver = ProblemSolver(config)
    solver.solve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimization Algorithm Runner")

    parser.add_argument(
        "--algorithm",
        type=str,
        choices=["sop_mac", "spm", "cot"],
        required=True,
        help="Algorithm to run (sop_mac, spm or cot)",
    )
    parser.add_argument(
        "--batch", action="store_true", help="Run all problems in the dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="NLP4LP",
        help="options: ComplexOR, NLP4LP, NLP4CSCO",
    )
    parser.add_argument(
        "--prob_name",
        type=str,
        default="0",
        help="problem name or index, options: aircraft_assignment, 0",
    )
    parser.add_argument(
        "--max_collaborate_nums",
        type=int,
        default=2,
        help="max num of agents to collaborate",
    )
    parser.add_argument(
        "--is_backtrack", action="store_true", help="Whether to use backtrack"
    )

    return parser.parse_args()


if __name__ == "__main__":
    """
    python main.py --algorithm sop_mac --dataset ComplexOR --batch --max_collaborate_nums 3 --is_backtrack
    python main.py --algorithm sop_mac --dataset ComplexOR --prob_name aircraft_landing --max_collaborate_nums 3 --is_backtrack
    python main.py --algorithm sop_mac --dataset NLP4LP --batch --prob_name "74,105,187"

    poetry run python src/sop_mac_0513/main.py  --algorithm sop_mac --dataset NLP4ECR --prob_name prob_2 --max_collaborate_nums 2

    poetry run dotenv run python src/sop_mac_0513/main.py  --algorithm sop_mac --dataset NLP4ECR --prob_name prob_2 --max_collaborate_nums 2
    """

    args = parse_args()
    config = Config.from_args(args)

    if args.batch:
        prob_names = get_problem_names(config)
        params = [
            Config(
                algorithm=config.algorithm,
                dataset=config.dataset,
                prob_name=str(prob_name),
                max_collaborate_nums=config.max_collaborate_nums,
                is_backtrack=config.is_backtrack,
            )
            for prob_name in prob_names
        ]

        with multiprocessing.Pool() as pool:
            pool.map(main, params)
    else:
        if config.prob_name is None:
            raise ValueError("--prob_name is required when not using --batch")
        main(config)
