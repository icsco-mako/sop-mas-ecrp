"""Aggregate top-p ablation results into AR / OAR summary tables.

Reads every ``run_*.json`` under ``<output_root>/<config_id>/<dataset>__<prob>/``
and reports two metrics per config:

* **AR (Accuracy Rate)**: fraction of runs (over all instances and repeats)
  whose ``status`` is True. Range [0, 1].
* **OAR (Objective-Agreement Rate)**: per instance, the fraction of pairs of
  repeats whose obj_value agree within ``rel_tol``; then averaged across
  instances. Runs with status=False contribute their actual numeric obj_value
  if present (so structurally different but numerically identical answers
  still agree); runs with obj_value=None count as disagreeing with everyone.

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
            }
        )
    return rows


def _write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    import csv

    fieldnames = ["config_id", "top_p", "temperature", "n_runs", "n_instances", "AR", "OAR"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {
                    k: (f"{v:.4f}" if isinstance(v, float) and not math.isnan(v) else v)
                    for k, v in r.items()
                }
            )


def _write_markdown(rows: List[Dict[str, Any]], path: Path) -> None:
    header = "| config | top_p | T | n_runs | n_inst | AR | OAR |"
    sep = "|---|---|---|---|---|---|---|"
    lines = [header, sep]
    for r in rows:
        ar = "—" if math.isnan(r["AR"]) else f"{r['AR']:.3f}"
        oar = "—" if math.isnan(r["OAR"]) else f"{r['OAR']:.3f}"
        lines.append(
            f"| {r['config_id']} | {r['top_p']} | {r['temperature']} | "
            f"{r['n_runs']} | {r['n_instances']} | {ar} | {oar} |"
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
    for r in rows:
        ar = "nan" if math.isnan(r["AR"]) else f"{r['AR']:.3f}"
        oar = "nan" if math.isnan(r["OAR"]) else f"{r['OAR']:.3f}"
        print(
            f"  {r['config_id']}: top_p={r['top_p']} T={r['temperature']} "
            f"AR={ar} OAR={oar} (n={r['n_runs']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
