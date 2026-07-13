# Formulation Stability Experiment

Repeated-run stability analysis for SOP-MAS on the empirical ECR instance.
All batches use the same configuration: **DeepSeek-V4-Flash + thinking mode**, `is_backtrack=False`, on `NLP4ECR/prob_empirical`.

## Batches

| Batch | Runs | AR (exec) | Correct | SAR | Note |
|-------|------|-----------|---------|------|------|
| b1_v4flash | 15 | 86.7% | 33.3% (5/15) | 0.090 | Pilot. Produces diverse wrong obj values (0, 654833, 2295474). |
| b2_v4flash | 15 | 73.3% | 73.3% (11/15) | 0.061 | |
| b3_v4flash | 10 | 70.0% | 70.0% (7/10) | 0.214 | Latest. |

- **AR (exec)**: fraction where `status=True` (code executed without error).
- **Correct**: fraction where `obj_value == 1,355,388` (ground-truth optimum).

**b2 + b3 can be pooled** (consistent behavior, exec AR ~72%, correct rate ~72%). **b1 should be analyzed separately** — it shows distinct wrong objective values absent from later batches, suggesting API-side model drift.

## Metrics

- **AR** — fraction of runs with `status=True` (execution success).
- **OAR (Objective-Agreement Rate)** — pairwise obj agreement across repeats.
- **SAR (Structural-Agreement Rate)** — pairwise formulation-signature agreement `(n_vars, n_constrs, n_bin, n_int, nnz)`.

## Usage

```bash
# Run a single batch
poetry run python experiments/topp_ablation/run_ablation.py \
    --config experiments/formulation_stability/config_b1.yaml

# Aggregate all batches
poetry run python experiments/topp_ablation/aggregate.py \
    --config experiments/formulation_stability/config.yaml

# Aggregate a single batch
poetry run python experiments/topp_ablation/aggregate.py \
    --config experiments/formulation_stability/config_b2.yaml
```

## File Layout

```
experiments/formulation_stability/
  config.yaml              # unified config (all 3 batches for aggregation)
  config_b1.yaml           # batch 1 only (15 repeats)
  config_b2.yaml           # batch 2 only (15 repeats)
  config_b3.yaml           # batch 3 only (10 repeats)
  _archived_config_*.yaml  # original configs from topp_ablation (archived)
  README.md

results/formulation_stability/
  b1_v4flash/NLP4ECR__prob_empirical/run_*.json
  b2_v4flash/NLP4ECR__prob_empirical/run_*.json
  b3_v4flash/NLP4ECR__prob_empirical/run_*.json
  summary.csv / summary.md
```
