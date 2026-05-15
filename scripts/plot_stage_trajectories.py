#!/usr/bin/env python3
"""Per-trait Elo trajectories across the loving-merged training stages.

X-axis (sequential): base -> DPO-step125 -> DPO-step200 -> DPO-final
                       -> Introspection-step225 -> Introspection-step300 -> Introspection-final
One line per evaluating constitution. OCT-loving(maius) is drawn as a
dashed reference for each trait.

Reads the same ValueArena summary.json files as build_cross_matrix.py and
applies the same API-avg=1500 anchoring.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.build_cross_matrix import DEFAULT_ROWS, REF_NICKS, REF_ANCHOR, fetch_summary  # noqa: E402

STAGES = [
    "Qwen2.5-7B-Instruct (base)",
    "DPO-step125", "DPO-step200", "DPO-final",
    "Introspection-step225", "Introspection-step300", "Introspection-final",
]
STAGE_LABELS = ["base", "DPO-125", "DPO-200", "DPO-final",
                "Intro-225", "Intro-300", "Intro-final"]
REF_MODEL = "OCT-loving(maius)"


def main():
    out_dir = _REPO_ROOT / "runs" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    means = {}     # trait -> [stage Elo]
    stds = {}      # trait -> [stage std]
    ref_vals = {}  # trait -> OCT-loving Elo
    for trait, run_id in DEFAULT_ROWS:
        s = fetch_summary(run_id)
        ref_elos = [s[r]["elo_mean"] for r in REF_NICKS if r in s]
        offset = REF_ANCHOR - sum(ref_elos) / len(ref_elos)
        means[trait] = [s[m]["elo_mean"] + offset if m in s else np.nan for m in STAGES]
        stds[trait] = [s[m]["elo_std"] if m in s else np.nan for m in STAGES]
        ref_vals[trait] = s[REF_MODEL]["elo_mean"] + offset if REF_MODEL in s else np.nan

    x = np.arange(len(STAGES))
    cmap = plt.get_cmap("tab10")
    traits = [t for t, _ in DEFAULT_ROWS]

    # Single combined plot
    fig, ax = plt.subplots(figsize=(11, 7))
    for i, t in enumerate(traits):
        c = cmap(i % 10)
        y = np.array(means[t])
        e = np.array(stds[t])
        ax.errorbar(x, y, yerr=e, fmt="-o", color=c, capsize=2, label=t, linewidth=1.6, markersize=4)
        if not np.isnan(ref_vals[t]):
            ax.axhline(ref_vals[t], color=c, linestyle=":", linewidth=0.9, alpha=0.7)

    ax.axvline(3.5, color="grey", linestyle="--", alpha=0.4)
    ax.text(1.5, ax.get_ylim()[1], "DPO", ha="center", va="bottom", fontsize=10, color="grey")
    ax.text(5, ax.get_ylim()[1], "Introspection", ha="center", va="bottom", fontsize=10, color="grey")

    ax.set_xticks(x)
    ax.set_xticklabels(STAGE_LABELS, rotation=30, ha="right")
    ax.set_ylabel(f"Elo (anchored: API avg = {REF_ANCHOR})")
    ax.set_title("Loving training trajectory — Elo per evaluating constitution\n"
                 "(dotted line = OCT-loving(maius) reference for that trait)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, frameon=False)
    plt.tight_layout()
    out = out_dir / "loving_stage_trajectories.png"
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

    # Faceted small-multiples (one panel per trait)
    n = len(traits)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 2.6), sharex=True, sharey=True)
    axes = axes.flatten()
    for i, t in enumerate(traits):
        a = axes[i]
        c = cmap(i % 10)
        y = np.array(means[t])
        e = np.array(stds[t])
        a.errorbar(x, y, yerr=e, fmt="-o", color=c, capsize=2, linewidth=1.4, markersize=3)
        if not np.isnan(ref_vals[t]):
            a.axhline(ref_vals[t], color="black", linestyle=":", linewidth=0.9,
                      label=f"OCT-loving={ref_vals[t]:.0f}")
        a.axhline(REF_ANCHOR, color="grey", linestyle="-", linewidth=0.5, alpha=0.5)
        a.axvline(3.5, color="grey", linestyle="--", alpha=0.3)
        a.set_title(t, fontsize=10)
        a.grid(True, alpha=0.3)
        a.legend(fontsize=7, loc="best", frameon=False)
    for j in range(n, len(axes)):
        axes[j].axis("off")
    for a in axes[-cols:]:
        a.set_xticks(x)
        a.set_xticklabels(STAGE_LABELS, rotation=45, ha="right", fontsize=8)
    fig.suptitle("Per-trait Elo trajectory across loving training stages", fontsize=13)
    fig.text(0.04, 0.5, f"Elo (API avg = {REF_ANCHOR})", va="center", rotation="vertical", fontsize=11)
    plt.tight_layout(rect=[0.05, 0, 1, 0.96])
    out2 = out_dir / "loving_stage_trajectories_facets.png"
    fig.savefig(out2, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out2}")


if __name__ == "__main__":
    main()
