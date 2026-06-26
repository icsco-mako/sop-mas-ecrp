# Token Overhead Experiment (Reviewer Comment 4.26)

Supplementary experiment responding to the reviewer's concern that *"the baseline
comparison controls the language model backbone but fails to account for
variations in prompting token overhead across different paradigms."*

Reports, per paradigm, the **number of LLM calls** and **prompt / completion /
total token consumption** on a controlled backbone, plus a self-consistent
25-instance accuracy (`AR(25)`) measured on the same runs.

## Backbone — DeepSeek-V3 has been sunset

The paper's Table 1 uses **DeepSeek-V3-0324** as the backbone for all methods.
The provider has since sunset the **entire DeepSeek-V3 family** on
`api.deepseek.com`, which now serves only **DeepSeek-V4-Pro / V4-Flash**. All
five paradigms are therefore re-measured on the same successor backbone,
**DeepSeek-V4-Flash**, preserving the controlled comparison the reviewer asked
about. (The `deepseek-chat` alias currently routes to V4-Flash; we use the
explicit `deepseek-v4-flash` string for stability.)

> Table 1's AR(269) results stand as historical measurements on V3-0324 and are
> **not** re-run here; only token overhead is re-measured.

## Instance set

25 instances drawn from **NLP4LP** (the 269-row LP/MIP benchmark that is the bulk
of Table 1's 269 instances), `seed=42`, checked into [`instances.yaml`](instances.yaml).

ComplexOR is excluded: (i) its loader hard-codes the ground-truth objective
(`utils.py` ComplexOR branch) and (ii) the mako baselines' data pre-processor is
not guaranteed to accept ComplexOR's structured `sample.json`. NLP4LP-only keeps
the measurement clean; the token overhead is paradigm-driven and a single-dataset
sample is sufficient.

## Methods

| Method   | Runs in        | Token source                       |
|----------|----------------|------------------------------------|
| SOP-MAS  | llm_mas_ecrp   | `llm_call.py` usage accumulator    |
| SPM      | mako (subproc) | langchain `get_openai_callback`    |
| CoT      | mako (subproc) | langchain `get_openai_callback`    |
| CoE      | mako (subproc) | langchain `get_openai_callback`    |
| OptiMUS  | mako (subproc) | langchain `get_openai_callback`    |

The four baselines are invoked via a **subprocess into the mako `uv` environment**
(`mako_invoke.py`); mako source is **not modified**. Both repos read the same
`.env`, so all five methods hit the identical `api.deepseek.com` endpoint.

## Pipeline configuration

`K = 5`, `is_backtrack = True` for SOP-MAS (matches Table 1). CoE and OptiMUS
use a collaboration / selection budget of 5 to align with `K`. `temperature = 0`,
`repeats = 1` (tokens are stable at T=0).

## Files

```
experiments/token_overhead/
  config.yaml              # backbone, sample, methods, pipeline, output paths
  instances.yaml           # the 25 NLP4LP indices (checked in)
  select_instances.py      # regenerates instances.yaml (seed=42)
  adapter.py               # load_problem / to_mako_problem / normalize_result
  mako_invoke.py           # subprocess entry, runs inside mako uv env
  run_overhead.py          # orchestrator: 5 methods × 25 instances -> runs/
  aggregate.py             # runs/ -> summary.csv / summary.md / summary.tex
  plot_tradeoff.py         # accuracy–token scatter -> fig_token_tradeoff.pdf
  runs/                    # raw per-run JSON (method/dataset__prob/run.json)
  README.md
```

## Usage

```bash
# 3-instance × 5-method smoke test (verify token fields + adapter)
uv run python experiments/token_overhead/run_overhead.py --smoke

# Full 5 × 25 (resumable)
uv run python experiments/token_overhead/run_overhead.py --skip-existing

# Aggregate + plot
uv run python experiments/token_overhead/aggregate.py
uv run python experiments/token_overhead/plot_tradeoff.py
```

## Draft disclosure (for the response letter, pending user approval)

> The DeepSeek-V3 endpoint has been sunset by the provider; token overhead was
> re-measured on its successor DeepSeek-V4-Flash, with all five paradigms
> sharing the identical backbone. Table X reports the mean number of LLM calls
> and mean total tokens per instance, alongside the 25-instance accuracy
> (`AR(25)`) measured on the same runs. Table 1's `AR(269)` (DeepSeek-V3-0324)
> is retained as the primary accuracy result.
