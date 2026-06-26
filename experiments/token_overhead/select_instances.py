#!/usr/bin/env python3
"""Reproducible instance sampler for the token-overhead experiment.

Draws ``size`` instances from the NLP4LP pool (269 rows) with a fixed ``seed``.
The 27 rows whose ``solution`` JSON lacks an ``objective`` (no ground truth) are
rejected and replaced by re-drawing from the remaining pool using the same RNG
stream, so the sample is deterministic and every drawn instance is solvable.
ComplexOR is excluded: its loader hard-codes the ground truth and the mako
baselines' pre-processor may reject its structured ``sample.json``.

The drawn sample is written to ``instances.yaml`` and checked into the repo.

Usage::
    uv run python experiments/token_overhead/select_instances.py
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = _REPO_ROOT / "data" / "NLP4LP.csv"
DEFAULT_OUT = Path(__file__).resolve().parent / "instances.yaml"


def valid_indices(csv_path: Path) -> list[int]:
    """Indices whose ``solution`` JSON contains an ``objective`` (ground truth)."""
    df = pd.read_csv(csv_path)
    out = []
    for i in range(len(df)):
        try:
            if "objective" in json.loads(df.loc[i, "solution"]):
                out.append(i)
        except (TypeError, ValueError, KeyError):
            continue
    return out


def sample_indices(pool_total: int, valid_set: set[int], size: int, seed: int) -> list[int]:
    """Draw ``size`` valid indices: seed=42 sample of the full pool, then reject
    ground-truth-missing rows and top up from the remainder on the same stream."""
    rng = random.Random(seed)
    drawn = rng.sample(range(pool_total), size)
    chosen = [i for i in drawn if i in valid_set]
    need = size - len(chosen)
    if need > 0:
        remaining = [i for i in range(pool_total) if i not in set(drawn)]
        for i in rng.sample(remaining, len(remaining)):
            if i in valid_set:
                chosen.append(i)
                need -= 1
                if need == 0:
                    break
    return sorted(chosen)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--size", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    df_len = len(pd.read_csv(args.csv))
    valid = set(valid_indices(Path(args.csv)))
    if args.size > len(valid):
        raise SystemExit(f"size {args.size} exceeds valid pool size {len(valid)}")

    indices = sample_indices(df_len, valid, args.size, args.seed)

    lines = [
        f"# Token-overhead instance sample (pool=NLP4LP n={df_len}, "
        f"valid={len(valid)}, size={args.size}, seed={args.seed}).",
        "# Ground-truth-missing rows rejected and topped up on the same RNG stream.",
        "# Regenerate:  uv run python experiments/token_overhead/select_instances.py",
        "dataset: NLP4LP",
        f"pool_size: {df_len}",
        f"pool_size_valid: {len(valid)}",
        f"seed: {args.seed}",
        "instances:",
    ]
    lines += [f"  - {i}" for i in indices]
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}: {len(indices)} valid instances "
          f"(from valid pool {len(valid)}/{df_len}, seed={args.seed})")


if __name__ == "__main__":
    main()
