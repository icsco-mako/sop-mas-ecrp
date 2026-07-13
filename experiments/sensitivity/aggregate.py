"""Aggregate sensitivity analysis results into AR / OAR / SAR metrics.

Reads per-run JSON files produced by ``run_sensitivity.py`` and computes:
- AR: Accuracy Rate (fraction of runs matching ground truth)
- OAR: Objective-Agreement Rate (pairwise obj_value consistency across repeats)
- SAR: Structural-Agreement Rate (pairwise formulation-signature consistency)

Usage::

    poetry run python experiments/sensitivity/aggregate.py
    poetry run python experiments/sensitivity/aggregate.py --output results/sensitivity/summary.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_runs(output_root: Path) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Load all run JSONs, keyed by config_id -> instance_key -> [runs]."""
    data: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for config_dir in sorted(output_root.iterdir()):
        if not config_dir.is_dir():
            continue
        config_id = config_dir.name
        for instance_dir in sorted(config_dir.iterdir()):
            if not instance_dir.is_dir():
                continue
            instance_key = instance_dir.name
            for run_file in sorted(instance_dir.glob("run_*.json")):
                try:
                    record = json.loads(run_file.read_text())
                    data[config_id][instance_key].append(record)
                except (json.JSONDecodeError, OSError):
                    logging.warning("Skipping invalid file: %s", run_file)
    return data


def _compute_ar(runs: List[Dict[str, Any]]) -> float:
    """Accuracy Rate: fraction of runs where status is True."""
    if not runs:
        return 0.0
    return sum(1 for r in runs if r.get("status")) / len(runs)


def _compute_oar(runs: List[Dict[str, Any]], rel_tol: float = 1e-4) -> Tuple[float, int]:
    """Objective-Agreement Rate: pairwise consistency of obj_value across repeats.

    Returns (OAR, n_pairs).
    """
    obj_values = []
    for r in runs:
        obj = r.get("obj_value")
        if obj is not None:
            obj_values.append(float(obj))

    if len(obj_values) < 2:
        return 1.0, 0

    pairs = list(combinations(obj_values, 2))
    if not pairs:
        return 1.0, 0

    agree = 0
    for a, b in pairs:
        if a == b or (a != 0 and abs((a - b) / a) < rel_tol):
            agree += 1
    return agree / len(pairs), len(pairs)


def _signatures_match(s1: Dict[str, int], s2: Dict[str, int]) -> bool:
    keys = ("n_vars", "n_constrs", "n_bin", "n_int", "nnz")
    return all(s1.get(k) == s2.get(k) for k in keys)


def _compute_sar(runs: List[Dict[str, Any]]) -> Tuple[float, int]:
    """Structural-Agreement Rate: pairwise consistency of formulation signatures.

    Returns (SAR, n_pairs).
    """
    sigs = [r["formulation_signature"] for r in runs
            if isinstance(r.get("formulation_signature"), dict)]

    if len(sigs) < 2:
        return 1.0, 0

    pairs = list(combinations(sigs, 2))
    agree = sum(1 for s1, s2 in pairs if _signatures_match(s1, s2))
    return agree / len(pairs), len(pairs)


def aggregate(output_root: Path) -> List[Dict[str, Any]]:
    data = _load_runs(output_root)
    rows = []

    for config_id in sorted(data.keys()):
        instances = data[config_id]
        all_ar = []
        all_oar = []
        all_sar = []
        total_runs = 0
        total_oar_pairs = 0
        total_sar_pairs = 0

        # Extract K, t_me, t_py from first run of first instance
        first_runs = next(iter(instances.values()), [])
        first_run = first_runs[0] if first_runs else {}
        K = first_run.get("K", "?")
        t_me = first_run.get("t_me", "?")
        t_py = first_run.get("t_py", "?")

        for inst_key, runs in instances.items():
            total_runs += len(runs)
            all_ar.append(_compute_ar(runs))

            oar, oar_pairs = _compute_oar(runs)
            all_oar.append(oar)
            total_oar_pairs += oar_pairs

            sar, sar_pairs = _compute_sar(runs)
            all_sar.append(sar)
            total_sar_pairs += sar_pairs

        n_instances = len(instances)
        rows.append({
            "config_id": config_id,
            "K": K,
            "t_me": t_me,
            "t_py": t_py,
            "n_runs": total_runs,
            "n_instances": n_instances,
            "AR": round(sum(all_ar) / n_instances, 4) if n_instances else 0,
            "OAR": round(sum(all_oar) / n_instances, 4) if n_instances else 0,
            "SAR": round(sum(all_sar) / n_instances, 4) if n_instances else 0,
            "oar_pairs": total_oar_pairs,
            "sar_pairs": total_sar_pairs,
        })

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate sensitivity results")
    parser.add_argument("--output", default=str(_PROJECT_ROOT / "results/sensitivity/summary.csv"))
    parser.add_argument("--config", default=str(_PROJECT_ROOT / "experiments/sensitivity/config.yaml"))
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    output_root = _PROJECT_ROOT / cfg["output_root"]

    if not output_root.exists():
        print(f"No results found at {output_root}")
        return 1

    rows = aggregate(output_root)
    if not rows:
        print("No data to aggregate.")
        return 1

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["config_id", "K", "t_me", "t_py", "n_runs", "n_instances",
                  "AR", "OAR", "SAR", "oar_pairs", "sar_pairs"]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Aggregated {len(rows)} configs -> {out_path}")
    print(f"{'config_id':<30} {'K':>3} {'t_me':>5} {'t_py':>5} {'AR':>6} {'OAR':>6} {'SAR':>6}")
    for r in rows:
        print(f"{r['config_id']:<30} {r['K']:>3} {r['t_me']:>5} {r['t_py']:>5} "
              f"{r['AR']:>6.4f} {r['OAR']:>6.4f} {r['SAR']:>6.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
