#!/usr/bin/env python3
"""Figures for the token-overhead experiment.

Plotting style mirrors ``scripts/computational_efficiency_analysis.py``:
Nature/Science ``SCI_COLORS`` palette, Times New Roman (incl. mathtext), bold
sized titles/labels, dashed faint grid, 600 dpi white-background PDFs.

Produces:
* ``fig_token_overhead_bars.pdf`` — single-panel dual-axis: total tokens
  (bars with error bars, left axis) overlaid with #LLM calls (line+markers,
  right axis).
* ``fig_token_tradeoff.pdf`` — accuracy–token scatter; X = mean total tokens
  (log), Y = AR(269) cited from Table 1.

Usage::
    uv run python experiments/token_overhead/plot_tradeoff.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import aggregate as agg  # noqa: E402

# --- style (consistent with scripts/computational_efficiency_analysis.py) ----
SCI_COLORS = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F',
              '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['axes.unicode_minus'] = False

METHOD_ORDER = ["spm", "cot", "coe", "optimus", "sop_mac"]
MARKER = {"spm": "o", "cot": "s", "coe": "^", "optimus": "D", "sop_mac": "*"}
COLOR_CALLS = '#3C5488'  # dark slate for the overlaid calls line


def _ordered_rows(cfg: dict) -> list[dict]:
    rows = agg.aggregate(cfg, _HERE / "runs")
    by = {r["method"]: r for r in rows}
    return [by[m] for m in METHOD_ORDER if m in by]


def plot_bars(rows: list[dict], out: Path) -> None:
    labels = [r["label"] for r in rows]
    toks = [r["mean_total_tokens"] for r in rows]
    calls = [r["mean_calls"] for r in rows]
    colors = SCI_COLORS[:len(rows)]
    x = range(len(rows))

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(list(x), toks, color=colors,
                  edgecolor='black', linewidth=0.8, alpha=0.9, zorder=2,
                  label='Total tokens')
    ax.set_ylabel('Mean total tokens / instance', fontsize=13, labelpad=10)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.tick_params(axis='both', labelsize=11)
    ax.set_ylim(0, max(toks) * 1.2)
    ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.5)
    for b, t in zip(bars, toks):
        ax.text(b.get_x() + b.get_width() / 2, t + max(toks) * 0.015,
                f'{t:,.0f}', ha='center', va='bottom', fontsize=10)

    ax2 = ax.twinx()
    ax2.plot(list(x), calls, 'o-', color=COLOR_CALLS, markersize=8,
             linewidth=1.8, zorder=3, label='#LLM calls')
    ax2.set_ylabel('Mean #LLM calls / instance', fontsize=13, labelpad=10)
    ax2.tick_params(axis='y', labelsize=11)
    ax2.set_ylim(0, max(calls) * 1.25)

    handles = [bars, ax2.lines[0]]
    ax.legend(handles, ['Total tokens', '#LLM calls'],
              loc='upper left', framealpha=0.9, fontsize=11)
    ax.set_title('Prompting token overhead across paradigms\n'
                 '(DeepSeek-V4-Flash, $K{=}5$)',
                 fontsize=14, pad=15, fontweight='bold')

    fig.tight_layout()
    fig.savefig(out, dpi=600, facecolor='white')
    plt.close(fig)
    print(f"Wrote {out}")


def plot_scatter(rows: list[dict], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for idx, r in enumerate(rows):
        ax.scatter(
            r["mean_total_tokens"], r["ar_269_ref"] * 100,
            marker=MARKER.get(r["method"], "o"), s=220,
            color=SCI_COLORS[idx], edgecolors='black', linewidths=0.8,
            alpha=0.9, zorder=3,
        )
        ax.annotate(
            r["label"], (r["mean_total_tokens"], r["ar_269_ref"] * 100),
            textcoords="offset points", xytext=(0, -14),
            ha='center', va='top', fontsize=11,
        )

    ax.set_xscale("log")
    ticks = [1000, 2000, 5000, 10000, 30000]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
    ax.xaxis.set_minor_locator(FixedLocator([]))
    # Headroom both sides so SPM (leftmost) and SOP-MAS (right/top) fit fully.
    ax.set_xlim(900, 55000)
    ax.set_ylim(40, 93)
    ax.set_xlabel('Mean total tokens per instance (DeepSeek-V4-Flash)',
                  fontsize=13, labelpad=10)
    ax.set_ylabel(r'AR (269 instances, Table 1) [\%]', fontsize=13, labelpad=10)
    ax.tick_params(axis='both', labelsize=11)
    ax.set_title('Accuracy–token trade-off ($K{=}5$)',
                 fontsize=14, pad=15, fontweight='bold')
    ax.grid(True, which='major', linestyle='--', alpha=0.4, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(out, dpi=600, facecolor='white')
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> None:
    cfg = yaml.safe_load((_HERE / "config.yaml").read_text())
    rows = _ordered_rows(cfg)
    if not rows:
        raise SystemExit("No runs to plot.")
    plot_bars(rows, _HERE / "fig_token_overhead_bars.pdf")
    plot_scatter(rows, _HERE / "fig_token_tradeoff.pdf")


if __name__ == "__main__":
    main()
