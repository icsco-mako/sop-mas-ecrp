"""Aggregate top-p ablation results into AR / OAR / SAR summary tables.

Reads every ``run_*.json`` under ``<output_root>/<config_id>/<dataset>__<prob>/``
and reports three metrics per config:

* **AR (Accuracy Rate)**: fraction of runs (over all instances and repeats)
  whose ``status`` is True. Range [0, 1].
* **OAR (Objective-Agreement Rate)**: per instance, the fraction of pairs of
  repeats whose obj_value agree within ``rel_tol``; then averaged across
  instances. Runs with status=False contribute their actual numeric obj_value
  if present (so structurally different but numerically identical answers
  still agree); runs with obj_value=None count as disagreeing with everyone.
* **SAR (Structural-Agreement Rate)**: per instance, the fraction of repeat
  pairs whose coarse formulation signatures agree. The signature is
  ``(n_vars, n_constrs, n_bin, n_int, nnz)`` as captured from the generated
  Gurobi model. Pairs missing a signature are excluded; if no valid signature
  pairs exist for a config, SAR is left blank instead of being imputed.

Outputs ``summary.csv`` and ``summary.md`` next to the raw results.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _values_agree(a: Optional[float], b: Optional[float], rel_tol: float) -> bool:
    if a is None or b is None:
        return False
    try:
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=1e-9)
    except (TypeError, ValueError):
        return False


def _signature_key(signature: Any) -> Optional[tuple[int, int, int, int, int]]:
    if not isinstance(signature, dict):
        return None
    keys = ("n_vars", "n_constrs", "n_bin", "n_int", "nnz")
    try:
        return tuple(int(signature[k]) for k in keys)
    except (KeyError, TypeError, ValueError):
        return None


def _load_runs(output_root: Path) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Return {config_id: {instance_key: [run_record, ...]}}."""
    runs: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for run_file in output_root.glob("*/*/run_*.json"):
        try:
            rec = json.loads(run_file.read_text())
        except json.JSONDecodeError:
            continue
        config_id = rec.get("config_id") or run_file.parents[1].name
        instance_key = f"{rec.get('dataset')}__{rec.get('prob_name')}"
        runs[config_id][instance_key].append(rec)
    return runs


def _compute_metrics(
    runs: Dict[str, Dict[str, List[Dict[str, Any]]]],
    rel_tol: float,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for config_id in sorted(runs):
        instance_map = runs[config_id]
        all_records = [r for recs in instance_map.values() for r in recs]
        n_runs = len(all_records)
        n_success = sum(1 for r in all_records if r.get("status"))
        ar = n_success / n_runs if n_runs else float("nan")

        oar_per_instance: List[float] = []
        for recs in instance_map.values():
            if len(recs) < 2:
                continue
            values = [r.get("obj_value") for r in recs]
            pairs = list(combinations(values, 2))
            if not pairs:
                continue
            agree = sum(
                1 for a, b in pairs if _values_agree(a, b, rel_tol)
            )
            oar_per_instance.append(agree / len(pairs))
        oar = (
            sum(oar_per_instance) / len(oar_per_instance)
            if oar_per_instance
            else float("nan")
        )

        sar_per_instance: List[float] = []
        n_signature_pairs = 0
        for recs in instance_map.values():
            if len(recs) < 2:
                continue
            signatures = [_signature_key(r.get("formulation_signature")) for r in recs]
            valid_pairs = [
                (a, b)
                for a, b in combinations(signatures, 2)
                if a is not None and b is not None
            ]
            if not valid_pairs:
                continue
            n_signature_pairs += len(valid_pairs)
            agree = sum(1 for a, b in valid_pairs if a == b)
            sar_per_instance.append(agree / len(valid_pairs))
        sar = (
            sum(sar_per_instance) / len(sar_per_instance)
            if sar_per_instance
            else float("nan")
        )

        # Pull (top_p, temperature) from any record (they're identical in a config).
        sample = all_records[0] if all_records else {}
        rows.append(
            {
                "config_id": config_id,
                "top_p": sample.get("top_p"),
                "temperature": sample.get("temperature"),
                "n_runs": n_runs,
                "n_instances": len(instance_map),
                "AR": ar,
                "OAR": oar,
                "SAR": sar,
                "n_signature_pairs": n_signature_pairs,
            }
        )
    return rows


def _compute_agent_timing(
    runs: Dict[str, Dict[str, List[Dict[str, Any]]]],
) -> Dict[str, Dict[str, Any]]:
    """Compute per-agent mean ± std across all runs (all configs combined)."""
    from statistics import mean, stdev

    agent_times_coll: Dict[str, List[float]] = {}
    elapsed_all: List[float] = []

    for instance_map in runs.values():
        for recs in instance_map.values():
            for r in recs:
                at = r.get("agent_times")
                if isinstance(at, dict):
                    for name, t in at.items():
                        try:
                            agent_times_coll.setdefault(name, []).append(float(t))
                        except (TypeError, ValueError):
                            pass
                es = r.get("elapsed_sec")
                if es is not None:
                    try:
                        elapsed_all.append(float(es))
                    except (TypeError, ValueError):
                        pass

    result: Dict[str, Dict[str, Any]] = {}
    for name in sorted(agent_times_coll):
        vals = agent_times_coll[name]
        entry: Dict[str, Any] = {"mean": mean(vals), "n": len(vals)}
        if len(vals) >= 2:
            entry["std"] = stdev(vals)
        else:
            entry["std"] = 0.0
        result[name] = entry

    if elapsed_all:
        result["_total_elapsed"] = {
            "mean": mean(elapsed_all),
            "std": stdev(elapsed_all) if len(elapsed_all) >= 2 else 0.0,
            "n": len(elapsed_all),
        }
    return result


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    import csv

    fieldnames = [
        "config_id",
        "top_p",
        "temperature",
        "n_runs",
        "n_instances",
        "AR",
        "OAR",
        "SAR",
        "n_signature_pairs",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    k: (
                        ""
                        if isinstance(v, float) and math.isnan(v)
                        else f"{v:.4f}" if isinstance(v, float) else v
                    )
                    for k, v in r.items()
                }
            )


def _write_markdown(rows: List[Dict[str, Any]], path: Path) -> None:
    header = "| config | top_p | T | n_runs | n_inst | AR | OAR | SAR | sig_pairs |"
    sep = "|---|---|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        ar = "—" if math.isnan(r["AR"]) else f"{r['AR']:.3f}"
        oar = "—" if math.isnan(r["OAR"]) else f"{r['OAR']:.3f}"
        sar = "—" if math.isnan(r["SAR"]) else f"{r['SAR']:.3f}"
        lines.append(
            f"| {r['config_id']} | {r['top_p']} | {r['temperature']} | "
            f"{r['n_runs']} | {r['n_instances']} | {ar} | {oar} | {sar} | "
            f"{r['n_signature_pairs']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(_PROJECT_ROOT / "experiments/topp_ablation/config.yaml"),
    )
    parser.add_argument("--rel-tol", type=float, default=1e-4)
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    output_root = _PROJECT_ROOT / cfg["output_root"]

    runs = _load_runs(output_root)
    if not runs:
        raise SystemExit(f"No run_*.json files found under {output_root}")

    rows = _compute_metrics(runs, rel_tol=args.rel_tol)
    _write_csv(rows, output_root / "summary.csv")
    _write_markdown(rows, output_root / "summary.md")
    print(f"Wrote {output_root / 'summary.csv'} and summary.md")

    timing = _compute_agent_timing(runs)
    if timing:
        timing_path = output_root / "agent_timing.json"
        timing_path.write_text(
            json.dumps(timing, ensure_ascii=False, indent=2) + "\n"
        )
        print(f"\nAgent timing (mean ± std) across all runs:")
        for name, stats in timing.items():
            print(f"  {name}: {stats['mean']:.2f} ± {stats['std']:.2f}s (n={stats['n']})")
    for r in rows:
        ar = "nan" if math.isnan(r["AR"]) else f"{r['AR']:.3f}"
        oar = "nan" if math.isnan(r["OAR"]) else f"{r['OAR']:.3f}"
        sar = "nan" if math.isnan(r["SAR"]) else f"{r['SAR']:.3f}"
        print(
            f"  {r['config_id']}: top_p={r['top_p']} T={r['temperature']} "
            f"AR={ar} OAR={oar} SAR={sar} "
            f"(n={r['n_runs']}, sig_pairs={r['n_signature_pairs']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
