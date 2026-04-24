# Top-p Sensitivity Ablation

Empirical study supporting the response to Reviewer 1, Comment 1.7.

## What it does

For the two creative agents in the SOP-MAS pipeline (`ModelEngineer` and
`PythonDeveloper`), we sweep:

- `top_p ∈ {0.5, 0.7, 0.9, 1.0}`
- `temperature ∈ {0.2, 0.8}`

giving 8 configurations. The remaining four agents (DataEngineer,
TestingEngineer, BusinessExpert, ProjectManager) are held at the deterministic
default (`temperature=0.0, top_p=1.0`) so the variation we observe is
attributable to the swept agents only. Backward tracking is disabled so we
measure raw forward-pass behaviour.

## Metrics

- **AR (Accuracy Rate)** — fraction of runs that match the ground-truth
  objective value, across all instances and repeats.
- **OAR (Objective-Agreement Rate)** — per instance, the fraction of pairs
  of repeats whose objective values agree (within `rel_tol`); averaged
  across instances. Captures stability independent of correctness.
- **SAR (Structural-Agreement Rate)** — per instance, the fraction of pairs
  of repeats whose coarse formulation signatures agree; averaged across
  instances. The signature is `(n_vars, n_constrs, n_bin, n_int, nnz)` as
  captured from the generated Gurobi model. SAR is a structural diagnostic,
  not a proof of symbolic formulation equivalence.

## Files

- `config.yaml` — sweep design, instance list, concurrency.
- `run_ablation.py` — runs every (config, instance, repeat) and writes
  `results/topp_ablation/<config>/<instance>/run_<r>.json`.
- `aggregate.py` — produces `summary.csv` / `summary.md` from the run
  records. Existing records generated before signature capture will have
  blank SAR values until the corresponding runs are regenerated.

## Usage

```bash
# Full sweep (uses concurrency from config.yaml)
poetry run python experiments/topp_ablation/run_ablation.py

# Resume after interruption
poetry run python experiments/topp_ablation/run_ablation.py --skip-existing

# Repair only missing/legacy SAR records
poetry run python experiments/topp_ablation/run_ablation.py \
    --rerun-missing-signature --concurrency 1

# Single-config debug
poetry run python experiments/topp_ablation/run_ablation.py \
    --only-config c4 --repeats 1 --concurrency 1

# Aggregate
poetry run python experiments/topp_ablation/aggregate.py
```

## Caveats

- The runner shells through `sop_mac()`, which uses thread-local logging.
  Using high concurrency may interleave log lines but does not affect
  per-run JSON output.
- API throttling on DeepSeek is the practical concurrency ceiling; start
  at 4 and adjust.
