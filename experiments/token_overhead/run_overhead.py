#!/usr/bin/env python3
"""Token-overhead experiment orchestrator (Reviewer Comment 4.26).

Runs five paradigms — SOP-MAS in-process and SPM/CoT/CoE/OptiMUS via a mako
subprocess — on the sampled NLP4LP instances, recording per run: number of LLM
calls, prompt/completion/total tokens, objective correctness, and duration.

Usage::

    uv run python experiments/token_overhead/run_overhead.py --smoke
    uv run python experiments/token_overhead/run_overhead.py --skip-existing
    uv run python experiments/token_overhead/run_overhead.py --methods spm,cot
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import yaml

from llm_mas_ecrp.core.sop_mac import sop_mac
from llm_mas_ecrp.core.agent_team.data_engineer import DataEngineer
from llm_mas_ecrp.core.agent_team.or_specialist import ModelEngineer
from llm_mas_ecrp.core.agent_team.python_developer import PythonDeveloper
from llm_mas_ecrp.core.agent_team.testing_engineer import TestingEngineer
from llm_mas_ecrp.core.agent_team.business_expert import BusinessExpert
from llm_mas_ecrp.utils.llm_call import reset_usage, get_usage

# This script is run directly (script dir on sys.path[0]); make the local import explicit.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_REPO_ROOT = _HERE.parents[1]

from adapter import load_problem, to_mako_problem, is_obj_correct  # noqa: E402


def load_config() -> dict:
    cfg = yaml.safe_load((_HERE / "config.yaml").read_text())
    inst = yaml.safe_load((_HERE / "instances.yaml").read_text())
    cfg["_instances"] = inst["instances"]
    return cfg


def build_agents_config(model: str) -> list:
    classes = [DataEngineer, ModelEngineer, PythonDeveloper, TestingEngineer, BusinessExpert]
    return [{"class": c, "args": ["DeepSeek", model, i]} for i, c in enumerate(classes)]


def run_sop_mac(problem: dict, cfg: dict) -> dict:
    model = cfg["llm"]["model"]
    k = cfg["pipeline"]["max_collaborate_nums"]
    is_bt = cfg["pipeline"]["is_backtrack"]

    reset_usage()
    t0 = time.time()
    error_msg = None
    try:
        result = sop_mac(
            problem, k, is_bt, build_agents_config(model),
            client="DeepSeek", model=model,
        )
        obj_value = result.get("obj_value")
        status = result.get("status", False)
    except Exception as exc:  # noqa: BLE001
        error_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        obj_value, status = None, False
    duration = time.time() - t0
    usage = get_usage()
    return {
        "obj_value": obj_value,
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "total_tokens": usage["total_tokens"],
        "llm_calls": usage["num_calls"],
        "status": status,
        "duration_s": round(duration, 3),
        "usage_estimated": usage["estimated"],
        "error_msg": error_msg,
    }


def run_mako_baseline(method: str, problem: dict, cfg: dict) -> dict:
    mako_root = (_REPO_ROOT / cfg["mako_root"]).resolve()
    mako_invoke = _HERE / "mako_invoke.py"
    runs_dir = _HERE / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(dir=runs_dir) as td:
        prob_json = Path(td) / "problem.json"
        out_json = Path(td) / "result.json"
        prob_json.write_text(json.dumps(to_mako_problem(problem), ensure_ascii=False))

        cmd = [
            "uv", "run", "--project", str(mako_root), "python", str(mako_invoke),
            "--method", method,
            "--problem-json", str(prob_json),
            "--out", str(out_json),
            "--provider", cfg["llm"]["provider"],
            "--model", cfg["llm"]["model"],
            "--max-collaborate-nums", str(cfg["pipeline"]["coe_max_collaborate_nums"]),
            "--optimus-max-selections", str(cfg["pipeline"]["optimus_max_selections"]),
        ]
        t0 = time.time()
        proc = subprocess.run(
            cmd, cwd=str(_REPO_ROOT), capture_output=True, text=True,
            timeout=cfg.get("timeout_sec", 1800),
        )
        duration = time.time() - t0

        if proc.returncode != 0 or not out_json.exists():
            return {
                "obj_value": None, "total_tokens": 0, "prompt_tokens": 0,
                "completion_tokens": 0, "llm_calls": 0, "status": False,
                "duration_s": round(duration, 3), "usage_estimated": False,
                "error_msg": (f"mako subprocess rc={proc.returncode}\n"
                              f"--stdout--\n{proc.stdout[-2000:]}\n"
                              f"--stderr--\n{proc.stderr[-2000:]}"),
            }
        res = json.loads(out_json.read_text())
        res["duration_s"] = round(duration, 3)
        return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="3-instance smoke test")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--methods", default=None, help="comma-separated subset of methods")
    ap.add_argument("--instances", type=int, default=None,
                    help="use only the first N instances")
    ap.add_argument("--log-level", default="WARNING")
    args = ap.parse_args()

    logging.basicConfig(level=args.log_level.upper(),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = load_config()
    instances = cfg["_instances"]
    if args.instances is not None:
        instances = instances[:args.instances]
    if args.smoke:
        instances = instances[:3]
    methods = args.methods.split(",") if args.methods else list(cfg["methods"])

    runs_dir = _HERE / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    for idx in instances:
        try:
            problem = load_problem(idx)
        except Exception as exc:  # noqa: BLE001
            print(f"[skip] instance {idx}: load failed ({exc})", flush=True)
            continue
        tag = f"{problem['dataset']}__{problem['prob_name']}"
        for method in methods:
            run_path = runs_dir / method / tag / "run.json"
            if args.skip_existing and run_path.exists():
                print(f"[skip] {method}/{tag}")
                continue
            run_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"[run ] {method}/{tag} ...", flush=True)

            if method == "sop_mac":
                res = run_sop_mac(problem, cfg)
            elif method in {"spm", "cot", "coe", "optimus"}:
                res = run_mako_baseline(method, problem, cfg)
            else:
                print(f"       [warn] unknown method {method}, skipping")
                continue

            res["algorithm"] = method
            res["dataset"] = problem["dataset"]
            res["prob_name"] = problem["prob_name"]
            res["ground_truth"] = problem["ground_truth"]
            res["obj_correct"] = is_obj_correct(res.get("obj_value"), problem["ground_truth"])
            res["model"] = cfg["llm"]["model"]
            res["mako_commit"] = cfg.get("mako_commit")
            run_path.write_text(json.dumps(res, ensure_ascii=False, indent=2))

            err = res.get("error_msg")
            print(f"       -> tokens={res.get('total_tokens')} calls={res.get('llm_calls')} "
                  f"correct={res['obj_correct']} {'ERR' if err else 'ok'}", flush=True)

    print("Done.")


if __name__ == "__main__":
    main()
