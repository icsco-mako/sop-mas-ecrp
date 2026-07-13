"""Generate 3D surface (mountain) plots from sensitivity analysis summary.

Reads ``summary.csv`` produced by ``aggregate.py`` and creates a 2×2 panel
of 3D surface plots, one per K value, with t_ME on x-axis, t_PY on y-axis,
and AR on z-axis.

Usage::

    poetry run python experiments/sensitivity/plot_3d_surface.py
    poetry run python experiments/sensitivity/plot_3d_surface.py \
        --input results/sensitivity/summary.csv \
        --output paper/manuscript/figure/figure_sensitivity_analysis.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import rcParams
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Publication-quality defaults
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman"]
rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 300
rcParams["savefig.format"] = "pdf"


def plot_3d_surface(
    df: pd.DataFrame,
    output_path: Path,
    metric: str = "AR",
    cmap: str = "RdYlGn",
) -> None:
    K_values = sorted(df["K"].unique())
    t_values = sorted(df["t_me"].unique())

    fig = plt.figure(figsize=(14, 12))
    axes = []

    for idx, K in enumerate(K_values, 1):
        ax = fig.add_subplot(2, 2, idx, projection="3d")
        axes.append(ax)

        sub = df[df["K"] == K]
        if sub.empty:
            ax.set_title(f"K = {K} (no data)")
            continue

        # Build meshgrid
        T_ME, T_PY = np.meshgrid(t_values, t_values)
        Z = np.full_like(T_ME, np.nan, dtype=float)

        for _, row in sub.iterrows():
            i = t_values.index(row["t_me"])
            j = t_values.index(row["t_py"])
            Z[j, i] = row[metric]

        # Interpolate missing values with nearest neighbour if any nan
        if np.isnan(Z).any():
            from scipy.interpolate import griddata
            points = np.column_stack([T_ME.ravel(), T_PY.ravel()])
            valid = ~np.isnan(Z.ravel())
            if valid.sum() >= 4:
                Z_flat = griddata(
                    points[valid], Z.ravel()[valid], points, method="nearest"
                )
                Z = Z_flat.reshape(Z.shape)

        surf = ax.plot_surface(
            T_ME, T_PY, Z,
            cmap=cmap,
            edgecolor="grey",
            linewidth=0.3,
            alpha=0.9,
            antialiased=True,
        )

        ax.set_xlabel(r"$t_{\mathrm{ME}}$", fontsize=11, labelpad=8)
        ax.set_ylabel(r"$t_{\mathrm{PY}}$", fontsize=11, labelpad=8)
        ax.set_zlabel(metric, fontsize=11, labelpad=8)
        ax.set_title(f"$K = {K}$", fontsize=13, pad=12)

        ax.set_xticks(t_values)
        ax.set_yticks(t_values)
        ax.view_init(elev=30, azim=-50)

    fig.colorbar(
        surf, ax=axes, shrink=0.5, aspect=20, pad=0.08, label=metric
    )

    fig.suptitle(
        f"Sensitivity Analysis: {metric} as a function of "
        r"$K$, $t_{\mathrm{ME}}$, $t_{\mathrm{PY}}$",
        fontsize=14, y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved 3D surface plot to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate 3D surface plots")
    parser.add_argument(
        "--input",
        default=str(_PROJECT_ROOT / "results/sensitivity/summary.csv"),
    )
    parser.add_argument(
        "--output",
        default=str(_PROJECT_ROOT / "paper/manuscript/figure/figure_sensitivity_analysis_3d.pdf"),
    )
    parser.add_argument("--metric", default="AR", choices=["AR", "OAR", "SAR"])
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        print("Run aggregate.py first to generate summary.csv")
        return 1

    df = pd.read_csv(input_path)
    plot_3d_surface(df, Path(args.output), metric=args.metric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
