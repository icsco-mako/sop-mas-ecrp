# Solver-execution across formulation structures (Comment 4.27)

Instance: `NLP4ECR/prob_empirical` (ground-truth optimum 1,355,388); 5 repeats per structure; metrics captured by monkey-patching `gurobipy.Model.optimize`. No LLM calls.

**10** distinct structures, **8/10** reach the optimum. Wall time 0.8–10.2 ms; simplex iterations 85–107.

| Structure (n_vars,n_constrs,nnz) | Type | Obj | Correct | Wall (ms) | Runtime (ms) | Iters |
|---|---|---:|:---:|---:|---:|---:|
| (975,984,nnz=2652) | expanded | 1355388 | YES | 10.2 | 2.0 | 85 |
| (370,0,nnz=0) | compact | — | no | 0.8 | — | — |
| (458,221,nnz=884) | compact | 1355388 | YES | 5.2 | 1.0 | 102 |
| (370,130,nnz=718) | compact | 1355388 | YES | 2.3 | 0.9 | 107 |
| (361,65,nnz=644) | compact | 1355388 | YES | 2.1 | 0.8 | 107 |
| (370,65,nnz=653) | compact | 1355388 | YES | 2.4 | 0.9 | 107 |
| (361,130,nnz=709) | compact | 1355388 | YES | 4.8 | 0.8 | 107 |
| (975,130,nnz=1798) | expanded | 1355388 | YES | 5.8 | 0.9 | 107 |
| (370,74,nnz=662) | compact | 1355388 | YES | 2.2 | — | — |
| (361,0,nnz=0) | compact | — | no | 0.8 | — | — |
