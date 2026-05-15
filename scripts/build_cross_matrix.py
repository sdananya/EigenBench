#!/usr/bin/env python3
"""Build a cross-constitution Elo matrix for a shared model pool.

Adapted from scripts/build_matrix.py with three changes:
  1. Columns are the union of model nicks across runs (no per-constitution
     name lookup, no --nick-prefix).
  2. REF_NICKS updated to the API nick spellings used in these runs
     ("Claude 4 Sonnet", "GPT 4.1", "Gemini 2.5 Pro").
  3. summary.json is fetched from the published ValueArena HF dataset
     instead of being read from a local btd_d{dim}/bootstrap/ directory.

Default: builds the 3-row matrix for the loving-merged pool against
oct_loving / oct_goodness / oct_mathematical.

Usage:
    python scripts/build_cross_matrix.py
    python scripts/build_cross_matrix.py --row oct_loving:loving-merged-vs-oct \\
        --row oct_goodness:loving-merged-vs-oct-goodness \\
        --row oct_mathematical:loving-merged-vs-oct-mathematical \\
        --output runs/analysis/loving_cross_matrix
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.build_matrix import plot_matrix, plot_ci_matrix, save_csv  # noqa: E402

HF_BASE = "https://huggingface.co/datasets/invi-bhagyesh/ValueArena/resolve/main/runs"

# (row label, ValueArena run id)
DEFAULT_ROWS = [
    ("loving",        "loving-merged-vs-oct"),
    ("goodness",      "loving-merged-vs-oct-goodness"),
    ("humor",         "loving-merged-vs-oct-humor"),
    ("impulsiveness", "loving-merged-vs-oct-impulsiveness"),
    ("mathematical",  "loving-merged-vs-oct-mathematical"),
    ("nonchalance",   "loving-merged-vs-oct-nonchalance"),
    ("poeticism",     "loving-merged-vs-oct-poeticism"),
    ("remorse",       "loving-merged-vs-oct-remorse"),
    ("sarcasm",       "loving-merged-vs-oct-sarcasm"),
    ("sycophancy",    "loving-merged-vs-oct-sycophancy"),
]

# API reference models for Elo anchoring (avg -> REF_ANCHOR)
REF_NICKS = ["Claude 4 Sonnet", "GPT 4.1", "Gemini 2.5 Pro"]
REF_ANCHOR = 1500
BASE_NICK = "Qwen2.5-7B-Instruct (base)"

# Preferred column ordering; any extras are appended in sorted order.
PREFERRED_COL_ORDER = [
    BASE_NICK,
    "DPO-step125", "DPO-step200", "DPO-final",
    "Introspection-step225", "Introspection-step300", "Introspection-final",
    "OCT-loving(maius)",
    "Claude 4 Sonnet", "GPT 4.1", "Gemini 2.5 Pro",
]


def fetch_summary(run_id: str) -> dict:
    url = f"{HF_BASE}/{run_id}/summary.json"
    print(f"  fetching {url}")
    with urllib.request.urlopen(url) as r:
        data = json.load(r)
    return {entry["model_name"]: entry for entry in data}


def order_columns(seen: set[str]) -> list[str]:
    cols = [c for c in PREFERRED_COL_ORDER if c in seen]
    extras = sorted(seen - set(cols))
    return cols + extras


def build_matrix(rows: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    summaries = {label: fetch_summary(rid) for label, rid in rows}

    seen: set[str] = set()
    for s in summaries.values():
        seen.update(s.keys())
    col_labels = order_columns(seen)

    nrows, ncols = len(rows), len(col_labels)
    A_mean = np.full((nrows, ncols), np.nan)
    A_std = np.full((nrows, ncols), np.nan)

    for i, (row_label, _) in enumerate(rows):
        s = summaries[row_label]
        ref_elos = [s[r]["elo_mean"] for r in REF_NICKS if r in s]
        if not ref_elos:
            if BASE_NICK in s:
                ref_elos = [s[BASE_NICK]["elo_mean"]]
            else:
                print(f"  {row_label}: no reference models — skipping row")
                continue
        offset = REF_ANCHOR - sum(ref_elos) / len(ref_elos)

        for j, col in enumerate(col_labels):
            if col in s:
                A_mean[i, j] = s[col]["elo_mean"] + offset
                A_std[i, j] = s[col]["elo_std"]

    row_labels = [label for label, _ in rows]
    return A_mean, A_std, row_labels, col_labels


def parse_row(spec: str) -> tuple[str, str]:
    if ":" not in spec:
        raise argparse.ArgumentTypeError(f"--row expects 'label:run_id', got {spec!r}")
    label, run_id = spec.split(":", 1)
    return label.strip(), run_id.strip()


def main():
    parser = argparse.ArgumentParser(description="Build cross-constitution Elo matrix")
    parser.add_argument("--row", action="append", type=parse_row,
                        help="label:run_id (repeatable). Defaults to the loving-merged trio.")
    parser.add_argument("--output", default=None,
                        help="Output prefix (default: runs/analysis/loving_cross_matrix)")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    rows = args.row or DEFAULT_ROWS
    A_mean, A_std, row_labels, col_labels = build_matrix(rows)

    out_prefix = args.output or str(_REPO_ROOT / "runs" / "analysis" / "loving_cross_matrix")
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)

    title = args.title or f"Cross-constitution Elo (API avg = {REF_ANCHOR})"
    plot_matrix(A_mean, A_std, row_labels, Path(f"{out_prefix}.png"),
                col_labels=col_labels, title=title)
    plot_ci_matrix(A_std, row_labels, Path(f"{out_prefix}_ci.png"),
                   col_labels=col_labels,
                   title="Cross-constitution — CI Width (±std)")
    save_csv(A_mean, row_labels, Path(f"{out_prefix}.csv"), col_labels=col_labels)


if __name__ == "__main__":
    main()
