"""Visualize formulation space — MDS projection + parallel coordinates."""
import json, os, glob
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS
from matplotlib.lines import Line2D

# ── Load data ──────────────────────────────────────────────────────────────
base = "/Users/pengkangzhen/Documents/code/llm_mas_ecrp/results/formulation_stability"
batches = ["b1_v4flash", "b2_v4flash", "b3_v4flash"]

records = []
for batch in batches:
    pattern = os.path.join(base, batch, "NLP4ECR__prob_empirical", "run_*.json")
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        sig = d.get("formulation_signature", {})
        if isinstance(sig, dict) and sig:
            records.append({
                "batch": batch,
                "status": d["status"],
                "obj_correct": bool(d.get("obj_correct", False)),
                "signature": (sig["n_vars"], sig["n_constrs"], sig["n_bin"],
                              sig["n_int"], sig["nnz"]),
            })

# Build unique signature table with metadata
sig_counter = Counter(r["signature"] for r in records)
sig_correct = Counter(r["signature"] for r in records if r["obj_correct"])
sig_batches = {}
for r in records:
    sig_batches.setdefault(r["signature"], set()).add(r["batch"])

unique_sigs = list(sig_counter.keys())
X = np.array(unique_sigs)  # (n_sig, 5)

# Note: n_bin is 0 for all — drop it to avoid degenerate dimension
active_dims = [0, 1, 3, 4]  # n_vars, n_constrs, n_int, nnz
dim_labels = [r"$n_{\mathrm{vars}}$", r"$n_{\mathrm{constrs}}$",
              r"$n_{\mathrm{int}}$", "nnz"]
X_active = X[:, active_dims].astype(float)

# Standardize
X_std = (X_active - X_active.mean(axis=0)) / X_active.std(axis=0)

# MDS projection
mds = MDS(n_components=2, random_state=42, metric=True,
          normalized_stress="auto", n_init=1)
X_mds = mds.fit_transform(X_std)

# ── Figure 1: MDS scatter ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))

# Determine marker properties
sizes = np.array([sig_counter[s] * 40 for s in unique_sigs])
correct_frac = np.array([sig_correct[s] / sig_counter[s] for s in unique_sigs])

# Color: correct rate (grey if 0, gradient blue if >0)
colors = []
for cf in correct_frac:
    if cf == 0:
        colors.append("#e74c3c")  # red = never correct
    elif cf == 1.0:
        colors.append("#27ae60")  # green = always correct
    else:
        colors.append("#f39c12")  # orange = mixed

for i, sig in enumerate(unique_sigs):
    ax.scatter(X_mds[i, 0], X_mds[i, 1], s=sizes[i], c=colors[i],
               edgecolors="black", linewidth=0.5, alpha=0.85, zorder=3)

# Annotate top-3 most frequent
for i in np.argsort(sizes)[-3:]:
    sig = unique_sigs[i]
    label = f"({sig[0]},{sig[1]},{sig[3]},{sig[4]})"
    ax.annotate(label, (X_mds[i, 0], X_mds[i, 1]),
                textcoords="offset points", xytext=(0, 10), fontsize=7,
                ha="center", color="#2c3e50")

# Legend
legend_elements = [
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#27ae60",
           markersize=10, label="Always correct"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#e74c3c",
           markersize=10, label="Never correct"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor="#f39c12",
           markersize=10, label="Mixed"),
]
ax.legend(handles=legend_elements, loc="upper right", fontsize=9)

# Draw faint ellipses for b1 vs b2+b3
from matplotlib.patches import Ellipse
b1_idx = [i for i, s in enumerate(unique_sigs) if "b1_v4flash" in sig_batches[s]]
b23_idx = [i for i, s in enumerate(unique_sigs) if i not in b1_idx]
for idxs, color, lbl in [(b1_idx, "#e74c3c", "b1"), (b23_idx, "#3498db", "b2+b3")]:
    if len(idxs) < 2:
        continue
    pts = X_mds[idxs]
    center = pts.mean(axis=0)
    cov = np.cov(pts.T) if len(idxs) > 2 else np.eye(2) * 0.5
    from numpy.linalg import eigh
    vals, vecs = eigh(cov)
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2.5 * np.sqrt(np.maximum(vals, 1e-6))
    ell = Ellipse(center, w, h, angle=angle, fill=False, color=color,
                  linestyle="--", linewidth=1.2, alpha=0.6)
    ax.add_patch(ell)
    ax.annotate(lbl, center, fontsize=8, color=color, ha="center", va="bottom",
                alpha=0.8)

ax.set_xlabel("MDS dimension 1")
ax.set_ylabel("MDS dimension 2")
ax.set_title("Formulation Space — MDS Projection of 4D Signatures", fontsize=12,
             fontweight="bold")
ax.text(0.02, 0.02,
        f"Stress={mds.stress_:.3f}  |  {len(unique_sigs)} distinct signatures  |  "
        f"{sum(r['obj_correct'] for r in records)}/{len(records)} correct",
        transform=ax.transAxes, fontsize=8, color="grey")

fig.tight_layout()
out1 = os.path.join(os.path.dirname(__file__), "fig_formulation_space_mds.pdf")
fig.savefig(out1, dpi=200, bbox_inches="tight")
fig.savefig(out1.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print(f"Saved: {out1}")
plt.close(fig)

# ── Figure 2: Parallel coordinates ────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

x_pos = np.arange(len(dim_labels))
for r in records:
    vals = np.array([r["signature"][d] for d in active_dims], dtype=float)
    # Standardize per-dimension relative to all data
    vals_std = (vals - X_active.mean(axis=0)) / X_active.std(axis=0)
    if r["obj_correct"]:
        color, alpha, lw, z = "#27ae60", 0.3, 0.8, 2
    elif r["status"]:
        color, alpha, lw, z = "#e74c3c", 0.3, 0.8, 1
    else:
        color, alpha, lw, z = "#95a5a6", 0.2, 0.5, 0
    ax.plot(x_pos, vals_std, c=color, alpha=alpha, linewidth=lw, zorder=z)

# Overlay mean of correct runs
correct_runs = [r for r in records if r["obj_correct"]]
if correct_runs:
    avg_correct = np.mean([[r["signature"][d] for d in active_dims]
                            for r in correct_runs], axis=0)
    avg_correct_std = (avg_correct - X_active.mean(axis=0)) / X_active.std(axis=0)
    ax.plot(x_pos, avg_correct_std, c="black", linewidth=2.5, zorder=10,
            linestyle="--", label="Correct-run mean")

ax.set_xticks(x_pos)
ax.set_xticklabels(dim_labels, fontsize=11)
ax.set_ylabel("Standardized value")
ax.set_title("Parallel Coordinates — Individual Runs Across Formulation Dimensions",
             fontsize=12, fontweight="bold")

legend_elements = [
    Line2D([0], [0], color="#27ae60", lw=2, label="Correct"),
    Line2D([0], [0], color="#e74c3c", lw=2, label="Incorrect (wrong obj)"),
    Line2D([0], [0], color="#95a5a6", lw=2, label="Execution failed"),
    Line2D([0], [0], color="black", lw=2, linestyle="--", label="Correct-run mean"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=8, ncol=2)

fig.tight_layout()
out2 = os.path.join(os.path.dirname(__file__), "fig_formulation_space_parcoords.pdf")
fig.savefig(out2, dpi=200, bbox_inches="tight")
fig.savefig(out2.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close(fig)

# ── Figure 3: Combined (MDS + ParCoords side by side) ─────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# -- Left: MDS --
for i, sig in enumerate(unique_sigs):
    ax1.scatter(X_mds[i, 0], X_mds[i, 1], s=sizes[i], c=colors[i],
                edgecolors="black", linewidth=0.5, alpha=0.85, zorder=3)
for i in np.argsort(sizes)[-3:]:
    sig = unique_sigs[i]
    label = f"({sig[0]},{sig[1]},{sig[3]},{sig[4]})"
    ax1.annotate(label, (X_mds[i, 0], X_mds[i, 1]),
                 textcoords="offset points", xytext=(0, 10), fontsize=6.5,
                 ha="center", color="#2c3e50")
ax1.legend(handles=legend_elements, loc="upper right", fontsize=8)
ax1.set_xlabel("MDS dim 1"); ax1.set_ylabel("MDS dim 2")
ax1.set_title("(a) MDS Projection", fontsize=11, fontweight="bold")

# -- Right: Parallel coordinates --
for r in records:
    vals = np.array([r["signature"][d] for d in active_dims], dtype=float)
    vals_std = (vals - X_active.mean(axis=0)) / X_active.std(axis=0)
    if r["obj_correct"]:
        color, alpha, lw = "#27ae60", 0.25, 0.7
    elif r["status"]:
        color, alpha, lw = "#e74c3c", 0.25, 0.7
    else:
        color, alpha, lw = "#95a5a6", 0.15, 0.5
    ax2.plot(x_pos, vals_std, c=color, alpha=alpha, linewidth=lw)

if correct_runs:
    avg_correct = np.mean([[r["signature"][d] for d in active_dims]
                            for r in correct_runs], axis=0)
    avg_correct_std = (avg_correct - X_active.mean(axis=0)) / X_active.std(axis=0)
    ax2.plot(x_pos, avg_correct_std, c="black", linewidth=2.5, zorder=10,
             linestyle="--")

ax2.set_xticks(x_pos)
ax2.set_xticklabels(dim_labels, fontsize=11)
ax2.set_ylabel("Standardized value")
ax2.set_title("(b) Parallel Coordinates", fontsize=11, fontweight="bold")
ax2.legend(handles=legend_elements, loc="upper left", fontsize=7, ncol=2)

fig.suptitle("Formulation Stability: Structural Diversity Analysis",
             fontsize=14, fontweight="bold", y=1.01)
fig.tight_layout()
out3 = os.path.join(os.path.dirname(__file__), "fig_formulation_space_combined.pdf")
fig.savefig(out3, dpi=200, bbox_inches="tight")
fig.savefig(out3.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
print(f"Saved: {out3}")
plt.close(fig)

# ── Summary stats ─────────────────────────────────────────────────────────
print(f"\nStats: {len(records)} runs, {len(unique_sigs)} distinct signatures")
print(f"  Correct: {sum(1 for r in records if r['obj_correct'])}")
print(f"  Incorrect (exec OK): {sum(1 for r in records if r['status'] and not r['obj_correct'])}")
print(f"  Exec failed: {sum(1 for r in records if not r['status'])}")
print(f"  MDS stress: {mds.stress_:.4f}")
