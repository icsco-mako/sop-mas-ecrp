"""Sensitivity analysis runner for SOP-MAS.

Sweeps K ∈ {2,3,4,5} × t_ME × t_PY ∈ {0.2, 0.4, 0.8, 1.0}² and writes
per-run JSON results.  Aggregation (AR / OAR / SAR) is handled by
``aggregate.py``.

Model fallback is handled at the LLM-call level: primary OpenRouter models
use fast retries (max_retries=2), and each agent carries a DeepSeek fallback
client so a failed request is immediately retried without restarting the
entire pipeline.

Usage::

    poetry run python experiments/sensitivity/run_sensitivity.py

Quick debug::

    poetry run python experiments/sensitivity/run_sensitivity.py \
        --repeats 1 --concurrency 1 --only-K 2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from llm_mas_ecrp.core.agent_team.business_expert import BusinessExpert
from llm_mas_ecrp.core.agent_team.data_engineer import DataEngineer
from llm_mas_ecrp.core.agent_team.or_specialist import ModelEngineer
from llm_mas_ecrp.core.agent_team.python_developer import PythonDeveloper
from llm_mas_ecrp.core.agent_team.testing_engineer import TestingEngineer
from llm_mas_ecrp.core.sop_mac import sop_mac
from llm_mas_ecrp.utils.user_proxy_config import get_llm_client
from llm_mas_ecrp.utils.utils import dataset_loader

AGENT_REGISTRY = {
    "DataEngineer": DataEngineer,
    "ModelEngineer": ModelEngineer,
    "PythonDeveloper": PythonDeveloper,
    "TestingEngineer": TestingEngineer,
    "BusinessExpert": BusinessExpert,
}

_AGENT_ORDER = [
    ("DataEngineer", 0),
    ("ModelEngineer", 1),
    ("PythonDeveloper", 2),
    ("TestingEngineer", 3),
    ("BusinessExpert", 4),
]


def _build_agents_config(
    client_name: str,
    model: str,
    creative_agents: List[str],
    t_me: float,
    t_py: float,
    top_p: float = 1.0,
    fallback_client_name: Optional[str] = None,
    fallback_model: Optional[str] = None,
    primary_max_retries: int = 2,
) -> List[Dict[str, Any]]:
    """Build AGENTs_CONFIG with per-agent fallback to a secondary LLM provider."""
    fb_client = get_llm_client(fallback_client_name) if fallback_client_name else None

    creative = set(creative_agents)
    agents_config = []
    for name, agent_id in _AGENT_ORDER:
        kwargs: Dict[str, Any] = {
            "max_retries": primary_max_retries,
            "fallback_client": fb_client,
            "fallback_model": fallback_model,
        }
        if name in creative:
            if name == "ModelEngineer":
                kwargs["temperature"] = t_me
                kwargs["top_p"] = top_p
            elif name == "PythonDeveloper":
                kwargs["temperature"] = t_py
                kwargs["top_p"] = top_p
        agents_config.append({
            "class": AGENT_REGISTRY[name],
            "args": [client_name, model, agent_id],
            "kwargs": kwargs,
        })
    return agents_config


def _expand_configs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand K_values × temperature_values into flat config list."""
    k_values = cfg["K_values"]
    t_values = cfg["temperature_values"]
    configs = []
    for K in k_values:
        for t_me in t_values:
            for t_py in t_values:
                configs.append({
                    "id": f"K{K}_tME{t_me}_tPY{t_py}",
                    "K": K,
                    "t_me": t_me,
                    "t_py": t_py,
                })
    return configs


def _pick_model(models_cfg: Dict[str, Any], run_index: int) -> Dict[str, str]:
    """Round-robin through primary models."""
    primaries = models_cfg.get("primary", [])
    if primaries:
        return primaries[run_index % len(primaries)]
    fallback = models_cfg.get("fallback", {})
    return fallback if fallback else {"client": "DeepSeek", "model": "deepseek-chat"}


def _execute_one(
    run_spec: Dict[str, Any],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

    out_path: Path = run_spec["out_path"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    client_name = run_spec["client"]
    model_name = run_spec["model"]
    fallback_cfg = cfg["models"].get("fallback", {})

    record: Dict[str, Any] = {
        "config_id": run_spec["config_id"],
        "K": run_spec["K"],
        "t_me": run_spec["t_me"],
        "t_py": run_spec["t_py"],
        "dataset": run_spec["dataset"],
        "prob_name": run_spec["prob_name"],
        "repeat": run_spec["repeat"],
        "client": client_name,
        "model": model_name,
    }

    timeout_sec = float(cfg.get("timeout_sec", 1800))

    def _run() -> Dict[str, Any]:
        problem = dataset_loader(run_spec["dataset"], run_spec["prob_name"])
        agents_config = _build_agents_config(
            client_name=client_name,
            model=model_name,
            creative_agents=cfg["creative_agents"],
            t_me=run_spec["t_me"],
            t_py=run_spec["t_py"],
            fallback_client_name=fallback_cfg.get("client"),
            fallback_model=fallback_cfg.get("model"),
        )
        return sop_mac(
            problem=problem,
            max_collaborate_nums=run_spec["K"],
            is_backtrack=bool(cfg["pipeline"]["is_backtrack"]),
            AGENTs_CONFIG=agents_config,
        )

    started = time.time()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(_run)
            result = future.result(timeout=timeout_sec)

        record.update(
            status=bool(result.get("status", False)),
            error_msg=result.get("error_msg", "None"),
            obj_value=_json_safe(result.get("obj_value")),
            formulation_signature=_json_safe(result.get("formulation_signature")),
            agent_times=_json_safe(result.get("agent_times")),
        )
    except FuturesTimeout:
        record.update(status=False, error_msg=f"timeout after {timeout_sec:.0f}s", obj_value=None)
        logging.warning("TIMEOUT %s/%s repeat=%d", run_spec["config_id"], run_spec["prob_name"], run_spec["repeat"])
    except Exception as exc:
        record.update(status=False, error_msg=f"runner_exception: {exc}", obj_value=None)
        record["traceback"] = traceback.format_exc()
    finally:
        record["elapsed_sec"] = round(time.time() - started, 2)

    out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    return record


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _json_safe(item())
        except (TypeError, ValueError):
            pass
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist())
        except (TypeError, ValueError):
            pass
    return str(value)


def _setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Sensitivity analysis runner")
    parser.add_argument("--config", default=str(_PROJECT_ROOT / "experiments/sensitivity/config.yaml"))
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument("--only-K", type=int, default=None, action="append", help="Restrict to specific K values")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--rerun-missing-signature", action="store_true")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.repeats is not None:
        cfg["repeats"] = args.repeats
    if args.concurrency is not None:
        cfg["concurrency"] = args.concurrency

    configs = _expand_configs(cfg)
    if args.only_K:
        k_set = set(args.only_K)
        configs = [c for c in configs if c["K"] in k_set]

    output_root = _PROJECT_ROOT / cfg["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_root / "sensitivity.log")

    # Build run specs
    specs = []
    global_run_idx = 0
    for c in configs:
        for inst in cfg["instances"]:
            model_pick = _pick_model(cfg["models"], global_run_idx)
            for r in range(1, int(cfg["repeats"]) + 1):
                out_path = (
                    output_root
                    / c["id"]
                    / f"{inst['dataset']}__{inst['prob_name']}"
                    / f"run_{r}.json"
                )
                specs.append({
                    "config_id": c["id"],
                    "K": c["K"],
                    "t_me": c["t_me"],
                    "t_py": c["t_py"],
                    "dataset": inst["dataset"],
                    "prob_name": str(inst["prob_name"]),
                    "repeat": r,
                    "client": model_pick["client"],
                    "model": model_pick["model"],
                    "out_path": out_path,
                })
                global_run_idx += 1

    if args.rerun_missing_signature:
        specs = [s for s in specs if not _has_formulation_signature(s["out_path"])]
    elif args.skip_existing:
        specs = [s for s in specs if not s["out_path"].exists()]

    logging.info(
        "Starting sensitivity analysis: %d runs (%d configs, %d instances, %d repeats)",
        len(specs), len(configs), len(cfg["instances"]), int(cfg["repeats"]),
    )

    concurrency = max(1, int(cfg.get("concurrency", 1)))
    completed = 0
    failures = 0
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_to_spec = {pool.submit(_execute_one, s, cfg): s for s in specs}
        for fut in as_completed(future_to_spec):
            spec = future_to_spec[fut]
            try:
                rec = fut.result()
                completed += 1
                status_str = "OK" if rec["status"] else "FAIL"
                logging.info(
                    "[%d/%d] %s %s/%s repeat=%d %s %.1fs",
                    completed, len(specs),
                    status_str, spec["config_id"], spec["prob_name"],
                    spec["repeat"], rec["status"], rec["elapsed_sec"],
                )
                if not rec["status"]:
                    failures += 1
            except Exception as exc:
                completed += 1
                failures += 1
                logging.exception("Run crashed: %s -> %s", spec["config_id"], exc)

    logging.info("Finished. completed=%d failures=%d output=%s", completed, failures, output_root)
    return 0


def _has_formulation_signature(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        record = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    sig = record.get("formulation_signature")
    return isinstance(sig, dict) and all(k in sig for k in ("n_vars", "n_constrs", "n_bin", "n_int", "nnz"))


if __name__ == "__main__":
    raise SystemExit(main())
