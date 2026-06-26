#!/usr/bin/env python3
"""Solver-execution analysis across distinct formulation structures (Comment 4.27).

The formulation-stability experiment (Section 4.6) observed that repeated runs on
the empirical ECR instance produce several structurally distinct yet
mathematically equivalent MIP/LP encodings. The reviewer (Comment 4.27) notes
that the manuscript dismisses this diversity as "equivalent encoding" without
addressing its impact on solver execution. This script quantifies that impact.

Approach (no LLM, zero API cost):
  1. From the historical formulation-stability runs, take one representative
     ``generated_code`` per distinct ``formulation_signature``.
  2. Re-run ONLY the generated ``optimize(data)`` on the empirical instance,
     monkey-patching ``gurobipy.Model.optimize`` to capture solver metrics.
  3. Report per-structure: objective (correctness), Gurobi runtime, simplex/MIP
     iterations, node count, wall time, status.

Usage::
    uv run python experiments/solver_execution/analyze.py
"""
from __future__ import annotations

import glob
import json
import math
import statistics
import time
from collections import OrderedDict
from pathlib import Path

import gurobipy

from llm_mas_ecrp.utils.utils import dataset_loader

_REPO_ROOT = Path(__file__).resolve().parents[2]
FS_GLOB = str(_REPO_ROOT / "results/formulation_stability/*/NLP4ECR__prob_empirical/run_*.json")
DATASET, PROB = "NLP4ECR", "prob_empirical"
GROUND_TRUTH = 1355388.0
REPEATS = 5
OUT_DIR = Path(__file__).resolve().parent


def _sig_tuple(sig):
    return tuple(sig.values()) if isinstance(sig, dict) else None


def collect_representative_codes() -> "OrderedDict[tuple, str]":
    """One generated_code per distinct formulation_signature."""
    reps: "OrderedDict[tuple, str]" = OrderedDict()
    for f in sorted(glob.glob(FS_GLOB)):
        d = json.loads(Path(f).read_text())
        sig = _sig_tuple(d.get("formulation_signature"))
        code = d.get("generated_code")
        if sig is None or not code or "def optimize" not in code:
            continue
        reps.setdefault(sig, code)  # first occurrence per signature
    return reps


def run_one(code: str, data, captured: dict) -> dict:
    """Exec a generated optimize() and capture Gurobi metrics."""
    env = {"gp": gurobipy, "GRB": gurobipy.GRB, "__builtins__": __builtins__}
    exec(code, env)  # noqa: S102 (exec required to load generated code)
    t0 = time.perf_counter()
    res = env["optimize"](data)
    wall = time.perf_counter() - t0
    m = captured.get("model")
    out = {"wall": wall, "obj": None, "runtime": None, "iters": None,
           "nodes": None, "status": None}
    if isinstance(res, dict):
        out["obj"] = res.get("obj_value")
    if m is not None:
        for attr, key in [("Runtime", "runtime"), ("IterCount", "iters"),
                          ("NodeCount", "nodes"), ("Status", "status")]:
            try:
                out[key] = m.getAttr(attr)
            except (gurobipy.GurobiError, AttributeError):
                pass
    return out


def main() -> None:
    import sys
    sys.path.insert(0, str(_REPO_ROOT))
    reps = collect_representative_codes()
    print(f"Distinct formulation structures with code: {len(reps)}")

    prob = dataset_loader(DATASET, PROB)
    data = prob["sample"][0].get("data", prob["sample"][0]["input"])

    # monkey-patch Model.optimize to record the solved model
    captured: dict = {}
    _orig = gurobipy.Model.optimize

    def _patched(self):
        try:
            self.Params.OutputFlag = 0  # silence per-model Gurobi console log
        except (gurobipy.GurobiError, AttributeError):
            pass
        r = _orig(self)
        captured["model"] = self
        return r

    gurobipy.Model.optimize = _patched

    rows = []
    for sig, code in reps.items():
        n_vars, n_constrs, n_bin, n_int, nnz = sig
        struct_type = "expanded" if n_vars >= 800 else "compact"
        trials = []
        err = None
        try:
            for _ in range(REPEATS):
                captured.clear()
                trials.append(run_one(code, data, captured))
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"

        if trials:
            obj = trials[0]["obj"]
            correct = isinstance(obj, (int, float)) and math.isclose(obj, GROUND_TRUTH, rel_tol=1e-6)
            rows.append({
                "structure": f"({n_vars},{n_constrs},nnz={nnz})",
                "type": struct_type,
                "n_vars": n_vars, "n_constrs": n_constrs, "nnz": nnz,
                "obj": obj, "correct": correct,
                "wall_ms_mean": statistics.mean(t["wall"] for t in trials) * 1000,
                "wall_ms_std": statistics.pstdev(t["wall"] for t in trials) * 1000 if len(trials) > 1 else 0.0,
                "runtime_ms_mean": statistics.mean([t["runtime"] for t in trials if t["runtime"] is not None]) * 1000 if any(t["runtime"] is not None for t in trials) else None,
                "iters_mean": statistics.mean([t["iters"] for t in trials if t["iters"] is not None]) if any(t["iters"] is not None for t in trials) else None,
                "nodes": trials[0]["nodes"],
                "status": trials[0]["status"],
                "error": err,
            })
        else:
            rows.append({"structure": f"({n_vars},{n_constrs},nnz={nnz})", "type": struct_type,
                         "n_vars": n_vars, "n_constrs": n_constrs, "nnz": nnz,
                         "obj": None, "correct": False, "wall_ms_mean": None,
                         "wall_ms_std": None, "runtime_ms_mean": None,
                         "iters_mean": None, "nodes": None, "status": None, "error": err})
        print(f"  {rows[-1]['structure']:22} type={struct_type:8} obj={rows[-1]['obj']} "
              f"correct={rows[-1]['correct']} wall={rows[-1]['wall_ms_mean']}")

    gurobipy.Model.optimize = _orig  # restore
    write_outputs(rows)
    print(f"\nWrote {OUT_DIR / 'solver_execution.csv'} and summary.md")


def write_outputs(rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # CSV
    cols = ["structure", "type", "n_vars", "n_constrs", "nnz", "obj", "correct",
            "wall_ms_mean", "wall_ms_std", "runtime_ms_mean", "iters_mean", "nodes", "status"]
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join([
            r["structure"], r["type"], str(r["n_vars"]), str(r["n_constrs"]), str(r["nnz"]),
            str(r["obj"]), str(r["correct"]),
            f"{r['wall_ms_mean']:.1f}" if r["wall_ms_mean"] is not None else "",
            f"{r['wall_ms_std']:.1f}" if r["wall_ms_std"] is not None else "",
            f"{r['runtime_ms_mean']:.1f}" if r["runtime_ms_mean"] is not None else "",
            f"{r['iters_mean']:.0f}" if r["iters_mean"] is not None else "",
            str(r["nodes"]), str(r["status"]),
        ]))
    (OUT_DIR / "solver_execution.csv").write_text("\n".join(lines) + "\n")

    # Markdown summary
    n = len(rows)
    n_ok = sum(1 for r in rows if r["correct"])
    walls = [r["wall_ms_mean"] for r in rows if r["wall_ms_mean"] is not None]
    iters = [r["iters_mean"] for r in rows if r["iters_mean"] is not None]
    md = [
        f"# Solver-execution across formulation structures (Comment 4.27)",
        f"",
        f"Instance: `{DATASET}/{PROB}` (ground-truth optimum {GROUND_TRUTH:,.0f}); "
        f"{REPEATS} repeats per structure; metrics captured by monkey-patching "
        f"`gurobipy.Model.optimize`. No LLM calls.",
        f"",
        f"**{n}** distinct structures, **{n_ok}/{n}** reach the optimum. "
        f"Wall time {min(walls):.1f}–{max(walls):.1f} ms; "
        f"simplex iterations {min(iters):.0f}–{max(iters):.0f}.",
        f"",
        f"| Structure (n_vars,n_constrs,nnz) | Type | Obj | Correct | Wall (ms) | Runtime (ms) | Iters |",
        f"|---|---|---:|:---:|---:|---:|---:|",
    ]
    for r in rows:
        obj = f"{r['obj']:.0f}" if r["obj"] is not None else "—"
        wall = f"{r['wall_ms_mean']:.1f}" if r["wall_ms_mean"] is not None else "—"
        runtime = f"{r['runtime_ms_mean']:.1f}" if r["runtime_ms_mean"] is not None else "—"
        iters = f"{r['iters_mean']:.0f}" if r["iters_mean"] is not None else "—"
        md.append(
            f"| {r['structure']} | {r['type']} | {obj} | "
            f"{'YES' if r['correct'] else 'no'} | {wall} | {runtime} | {iters} |"
        )
    (OUT_DIR / "summary.md").write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
