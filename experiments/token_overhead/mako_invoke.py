#!/usr/bin/env python3
"""Subprocess entry point that runs ONE mako baseline inside the mako uv env.

Invoked by ``run_overhead.py`` as::

    uv run --project <mako_root> python <this_file> \
        --method spm --problem-json problem.json --out result.json \
        --provider DeepSeek --model deepseek-v4-flash \
        --max-collaborate-nums 5 --optimus-max-selections 5

Token accounting:
  * ``total_tokens`` — from the baseline's own ``get_openai_callback`` accumulation
    (mako records this correctly per baseline).
  * ``llm_calls`` — counted by monkey-patching ``openai``'s
    ``Completions.create`` at the SDK level, so every actual API call is counted
    uniformly regardless of how each baseline structures its internal "steps"
    (CoE/OptiMUS collapse a multi-round pipeline into a single recorded step, so
    step-counting would understate their call count). This patch does not modify
    any mako source.

Writes a normalized ``result.json``; ground-truth comparison is done by the caller.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import openai
from dotenv import load_dotenv

load_dotenv()  # mako's llm_config also loads .env; this is belt-and-suspenders

# --- SDK-level call counter (active for the lifetime of this subprocess) -----
_orig_create = openai.resources.chat.completions.Completions.create
_LLM_CALLS = {"n": 0}


def _counting_create(self, *args, **kwargs):  # type: ignore[no-untyped-def]
    _LLM_CALLS["n"] += 1
    return _orig_create(self, *args, **kwargs)


openai.resources.chat.completions.Completions.create = _counting_create


def _run(method: str, problem: dict, provider: str, model: str,
         coe_k: int, optimus_sel: int, out_dir: Path) -> dict:
    if method == "spm":
        from mako_langchain.experiments.spm import run_spm
        return run_spm(problem, provider=provider, model=model)
    if method == "cot":
        from mako_langchain.experiments.cot import run_cot
        return run_cot(problem, provider=provider, model=model)
    if method == "coe":
        from mako_langchain.baselines.chain_of_experts.run_baseline import run_single
        return run_single(problem, provider, model, coe_k)
    if method == "optimus":
        from mako_langchain.baselines.optimus.run_baseline import run_single
        # filepath lets OptiMUS save its generated code; without it the run still
        # produces obj_value/tokens but raises a benign None/"str" TypeError when
        # trying to write the file.
        return run_single(problem, provider, model, optimus_sel, filepath=out_dir)
    raise ValueError(f"unknown mako method: {method}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--problem-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--provider", default="DeepSeek")
    ap.add_argument("--model", default="deepseek-v4-flash")
    ap.add_argument("--max-collaborate-nums", type=int, default=5)
    ap.add_argument("--optimus-max-selections", type=int, default=5)
    args = ap.parse_args()

    problem = json.loads(Path(args.problem_json).read_text())

    t0 = time.time()
    error_msg = None
    res: dict = {}
    try:
        res = _run(args.method, problem, args.provider, args.model,
                   args.max_collaborate_nums, args.optimus_max_selections,
                   Path(args.out).parent) or {}
    except Exception as exc:  # noqa: BLE001
        error_msg = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    duration = time.time() - t0

    out = {
        "algorithm": args.method,
        "obj_value": res.get("obj_value"),
        "total_tokens": res.get("total_tokens") or 0,
        "llm_calls": _LLM_CALLS["n"],
        "status": res.get("status", False),
        "duration_s": round(duration, 3),
        "error_msg": error_msg or res.get("error_msg"),
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"[mako_invoke] {args.method}: tokens={out['total_tokens']} "
          f"calls={out['llm_calls']} status={out['status']}", flush=True)


if __name__ == "__main__":
    main()
