"""Top-p sensitivity ablation runner.

Loads experiments/topp_ablation/config.yaml and runs sop_mac() for every
(config, instance, repeat) triple. Results are written to
``<output_root>/<config_id>/<dataset>__<prob_name>/run_<r>.json`` so that the
aggregator script can compute AR (accuracy rate) and OAR (objective-agreement
rate) without re-running anything.

Usage::

    poetry run python experiments/topp_ablation/run_ablation.py \
        --config experiments/topp_ablation/config.yaml

Override fields on the command line for quick scoping::

    poetry run python experiments/topp_ablation/run_ablation.py --repeats 1 \
        --concurrency 1 --only-config c4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Make ``src`` importable when running as a script.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from llm_mas_ecrp.core.agent_team.business_expert import BusinessExpert  # noqa: E402
from llm_mas_ecrp.core.agent_team.data_engineer import DataEngineer  # noqa: E402
from llm_mas_ecrp.core.agent_team.or_specialist import ModelEngineer  # noqa: E402
from llm_mas_ecrp.core.agent_team.python_developer import PythonDeveloper  # noqa: E402
from llm_mas_ecrp.core.agent_team.testing_engineer import TestingEngineer  # noqa: E402
from llm_mas_ecrp.core.sop_mac import sop_mac  # noqa: E402
from llm_mas_ecrp.utils.utils import dataset_loader  # noqa: E402

AGENT_REGISTRY = {
    "DataEngineer": DataEngineer,
    "ModelEngineer": ModelEngineer,
    "PythonDeveloper": PythonDeveloper,
    "TestingEngineer": TestingEngineer,
    "BusinessExpert": BusinessExpert,
}

# Fixed agent ordering for SOP-MAS.
_AGENT_ORDER = [
    ("DataEngineer", 0),
    ("ModelEngineer", 1),
    ("PythonDeveloper", 2),
    ("TestingEngineer", 3),
    ("BusinessExpert", 4),
]


@dataclass(frozen=True)
class RunSpec:
    config_id: str
    top_p: float
    temperature: float
    dataset: str
    prob_name: str
    repeat: int
    out_path: Path


def _build_agents_config(
    client: str,
    model: str,
    creative_agents: List[str],
    temperature: float,
    top_p: float,
) -> List[Dict[str, Any]]:
    """Return AGENTs_CONFIG list. Creative agents get the swept (T, top_p);
    everyone else stays deterministic (defaults T=0.0, top_p=1.0)."""
    creative = set(creative_agents)
    agents_config = []
    for name, agent_id in _AGENT_ORDER:
        entry: Dict[str, Any] = {
            "class": AGENT_REGISTRY[name],
            "args": [client, model, agent_id],
        }
        if name in creative:
            entry["kwargs"] = {"temperature": temperature, "top_p": top_p}
        agents_config.append(entry)
    return agents_config


def _expand_run_specs(cfg: Dict[str, Any], output_root: Path) -> List[RunSpec]:
    specs: List[RunSpec] = []
    for c in cfg["configs"]:
        for inst in cfg["instances"]:
            instance_dir = (
                output_root / c["id"] / f"{inst['dataset']}__{inst['prob_name']}"
            )
            for r in range(1, int(cfg["repeats"]) + 1):
                specs.append(
                    RunSpec(
                        config_id=c["id"],
                        top_p=float(c["top_p"]),
                        temperature=float(c["temperature"]),
                        dataset=inst["dataset"],
                        prob_name=str(inst["prob_name"]),
                        repeat=r,
                        out_path=instance_dir / f"run_{r}.json",
                    )
                )
    return specs


def _execute_one(spec: RunSpec, cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run a single (config, instance, repeat). Returns a result dict that is
    also persisted to ``spec.out_path``."""
    spec.out_path.parent.mkdir(parents=True, exist_ok=True)

    record: Dict[str, Any] = {
        "config_id": spec.config_id,
        "top_p": spec.top_p,
        "temperature": spec.temperature,
        "dataset": spec.dataset,
        "prob_name": spec.prob_name,
        "repeat": spec.repeat,
    }

    started = time.time()
    try:
        problem = dataset_loader(spec.dataset, spec.prob_name)
        agents_config = _build_agents_config(
            client=cfg["llm"]["client"],
            model=cfg["llm"]["model"],
            creative_agents=cfg["creative_agents"],
            temperature=spec.temperature,
            top_p=spec.top_p,
        )
        result = sop_mac(
            problem=problem,
            max_collaborate_nums=int(cfg["pipeline"]["max_collaborate_nums"]),
            is_backtrack=bool(cfg["pipeline"]["is_backtrack"]),
            AGENTs_CONFIG=agents_config,
        )

        record.update(
            status=bool(result.get("status", False)),
            error_msg=result.get("error_msg", "None"),
            obj_value=result.get("obj_value"),
        )
    except Exception as exc:  # noqa: BLE001
        record.update(
            status=False,
            error_msg=f"runner_exception: {exc}",
            obj_value=None,
            traceback=traceback.format_exc(),
        )
    finally:
        record["elapsed_sec"] = time.time() - started

    spec.out_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    return record


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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(_PROJECT_ROOT / "experiments/topp_ablation/config.yaml"),
        help="Path to the ablation config YAML.",
    )
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=None)
    parser.add_argument(
        "--only-config",
        default=None,
        help="Optional config_id to restrict the run to (for debugging).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs whose result file already exists.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.repeats is not None:
        cfg["repeats"] = args.repeats
    if args.concurrency is not None:
        cfg["concurrency"] = args.concurrency
    if args.only_config:
        cfg["configs"] = [c for c in cfg["configs"] if c["id"] == args.only_config]
        if not cfg["configs"]:
            raise SystemExit(f"--only-config {args.only_config!r} matched no configs")

    output_root = _PROJECT_ROOT / cfg["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    _setup_logging(output_root / "ablation.log")

    specs = _expand_run_specs(cfg, output_root)
    if args.skip_existing:
        specs = [s for s in specs if not s.out_path.exists()]

    logging.info(
        "Starting top-p ablation: %d runs across %d configs, %d instances, %d repeats",
        len(specs),
        len(cfg["configs"]),
        len(cfg["instances"]),
        int(cfg["repeats"]),
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
                status = "OK" if rec["status"] else "FAIL"
                logging.info(
                    "[%d/%d] %s %s/%s/%s repeat=%d status=%s elapsed=%.1fs",
                    completed,
                    len(specs),
                    status,
                    spec.config_id,
                    spec.dataset,
                    spec.prob_name,
                    spec.repeat,
                    rec["status"],
                    rec["elapsed_sec"],
                )
                if not rec["status"]:
                    failures += 1
            except Exception as exc:  # noqa: BLE001
                failures += 1
                logging.exception("Run crashed: %s -> %s", spec, exc)

    logging.info(
        "Ablation finished. completed=%d failures=%d output=%s",
        completed,
        failures,
        output_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
