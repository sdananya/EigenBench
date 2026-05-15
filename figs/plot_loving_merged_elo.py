"""Plot Elo rankings for the loving-merged-quick run with renamed labels.

Source: https://valuearena.github.io/run.html?run=loving-merged-quick
Introspection-final is excluded per request.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


# (display_name, elo_mean, ci_lower, ci_upper)
ROWS = [
    ("OCT (loving)",                  1604.8, 1595.0, 1613.0),
    ("DPO+Introspection (loving)",    1574.7, 1567.0, 1582.0),
    ("Claude 4 Sonnet",               1476.8, 1468.0, 1488.0),
    ("DPO (loving)",                  1473.2, 1463.0, 1483.0),
    ("GPT 4.1",                       1453.9, 1446.0, 1463.0),
    ("Qwen2.5-7B-Instruct (base)",    1444.2, 1439.0, 1450.0),
    ("Gemini 2.5 Pro",                1424.9, 1418.0, 1434.0),
]


def main() -> None:
    # Sort descending so the highest Elo is on the left.
    rows = sorted(ROWS, key=lambda r: r[1], reverse=True)

    labels = [r[0] for r in rows]
    means = np.array([r[1] for r in rows])
    lower = np.array([r[2] for r in rows])
    upper = np.array([r[3] for r in rows])

    x = np.arange(len(rows))
    yerr = np.vstack([means - lower, upper - means])

    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 0.7), 6))
    ax.errorbar(x, means, yerr=yerr, fmt="o", capsize=4, color="#1f77b4")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("EigenBench Elo")
    ax.set_title("Loving — Bootstrap Elo Means with 95% Confidence Intervals")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    out = Path(__file__).parent / "loving_merged_elo.png"
    fig.savefig(str(out), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
