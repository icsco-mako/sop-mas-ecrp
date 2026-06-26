#!/usr/bin/env python3
"""Aggregate token-overhead runs into a summary table (csv / md / tex).

Reads every ``runs/<method>/<dataset>__<prob>/run.json`` and computes, per
method: mean LLM calls, mean prompt / completion / total tokens, and the
25-instance accuracy ``AR(25)`` measured on the same V4-Flash runs. The Table-1
reference ``AR(269)`` (DeepSeek-V3-0324) is pulled from ``config.yaml`` as a
cited reference column.

Usage::
    uv run python experiments/token_overhead/aggregate.py
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

import yaml

_HERE = Path(__file__).resolve().parent
METHOD_ORDER = ["spm", "cot", "coe", "optimus", "sop_mac"]
METHOD_LABEL = {
    "spm": "SPM",
    "cot": "CoT",
    "coe": "CoE",
    "optimus": "OptiMUS",
    "sop_mac": "SOP-MAS",
}


def _allowed_instances() -> set[str]:
    """Prob-name allowlist from instances.yaml (ignore orphan runs)."""
    inst = yaml.safe_load((_HERE / "instances.yaml").read_text())
    return {str(i) for i in inst["instances"]}


def load_runs(runs_dir: Path) -> dict[str, list[dict]]:
    allowed = _allowed_instances()
    by_method: dict[str, list[dict]] = {}
    for run_json in runs_dir.glob("*/*/run.json"):
        method = run_json.parent.parent.name
        rec = json.loads(run_json.read_text())
        if str(rec.get("prob_name")) not in allowed:
            continue  # orphan run from an earlier sample — ignore
        by_method.setdefault(method, []).append(rec)
    return by_method


def aggregate(cfg: dict, runs_dir: Path) -> list[dict]:
    by_method = load_runs(runs_dir)
    ar_ref = cfg["ar_full_benchmark"]
    rows = []
    for method in METHOD_ORDER:
        runs = by_method.get(method, [])
        if not runs:
            continue
        n = len(runs)
        rows.append({
            "method": method,
            "label": METHOD_LABEL[method],
            "n_runs": n,
            "mean_calls": mean(r["llm_calls"] for r in runs),
            "std_calls": stdev([r["llm_calls"] for r in runs]) if n > 1 else 0.0,
            "mean_prompt_tokens": mean(r.get("prompt_tokens") or 0 for r in runs),
            "mean_completion_tokens": mean(r.get("completion_tokens") or 0 for r in runs),
            "mean_total_tokens": mean(r["total_tokens"] for r in runs),
            "std_total_tokens": stdev([r["total_tokens"] for r in runs]) if n > 1 else 0.0,
            "ar_25": mean(1.0 if r["obj_correct"] else 0.0 for r in runs),
            "ar_269_ref": ar_ref.get(method),
        })
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    cols = ["label", "n_runs", "mean_calls", "mean_prompt_tokens",
            "mean_completion_tokens", "mean_total_tokens", "ar_25", "ar_269_ref"]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join([
            r["label"], str(r["n_runs"]),
            f"{r['mean_calls']:.1f}",
            f"{r['mean_prompt_tokens']:.0f}",
            f"{r['mean_completion_tokens']:.0f}",
            f"{r['mean_total_tokens']:.0f}",
            f"{r['ar_25']:.4f}",
            f"{r['ar_269_ref']:.4f}",
        ]))
    path.write_text("\n".join(lines) + "\n")


def write_md(rows: list[dict], path: Path, model: str) -> None:
    lines = [
        f"# Token Overhead Summary (backbone: `{model}`, 25 NLP4LP instances, seed=42)",
        "",
        "| Method | #Calls (mean) | Total tokens (mean) | AR (25) | AR (269, ref) |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['mean_calls']:.1f} | {r['mean_total_tokens']:.0f} | "
            f"{r['ar_25']*100:.2f}\\% | {r['ar_269_ref']*100:.2f}\\% |"
        )
    path.write_text("\n".join(lines) + "\n")


def write_tex(rows: list[dict], path: Path) -> None:
    """Booktabs table matching the style of Table~1 (tab:benchmark)."""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Prompting token overhead across paradigms on a 25-instance "
        r"NLP4LP subset (DeepSeek-V4-Flash backbone, $K{=}5$, seed${=}42$). "
        r"AR(25) is measured on the same runs; AR(269) is cited from "
        r"Table~\ref{tab:benchmark} (DeepSeek-V3-0324 era).}\label{tab:token_overhead}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Method & \#Calls (mean) & Total tokens (mean) & AR (25) & AR (269, ref) \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r['label']} & {r['mean_calls']:.1f} & {r['mean_total_tokens']:.0f} & "
            f"{r['ar_25']*100:.2f}\\% & {r['ar_269_ref']*100:.2f}\\% \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    cfg = yaml.safe_load((_HERE / "config.yaml").read_text())
    runs_dir = _HERE / "runs"
    rows = aggregate(cfg, runs_dir)
    if not rows:
        raise SystemExit(f"No run.json found under {runs_dir}")

    write_csv(rows, _HERE / "summary.csv")
    write_md(rows, _HERE / "summary.md", cfg["llm"]["model"])
    write_tex(rows, _HERE / "summary.tex")
    print(f"Aggregated {sum(r['n_runs'] for r in rows)} runs across {len(rows)} methods.")
    for r in rows:
        print(f"  {r['label']:9} calls={r['mean_calls']:5.1f}  total_tok={r['mean_total_tokens']:8.0f}  "
              f"AR(25)={r['ar_25']*100:5.2f}%  AR(269,ref)={r['ar_269_ref']*100:5.2f}%")


if __name__ == "__main__":
    main()
