#!/usr/bin/env python3
"""
Surface scenarios from a single evaluations.jsonl where one model beats another.

If --model-a / --model-b are not given, the script lists the model names found
in the file and asks you to pick interactively. Output is written next to the
eval file as `<stem>__<A>_beats_<B>__top<K>.json`, unless --out is supplied.

Usage:
  # Interactive
  python scripts/surface_pairwise_from_eval.py path/to/evaluations.jsonl

  # Non-interactive
  python scripts/surface_pairwise_from_eval.py path/to/evaluations.jsonl \
      --model-a DPO-step200 --model-b Introspection-final --top-k 10
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict


CHOICE_RE = re.compile(r"<criterion_\d+_choice>(\d+)</criterion_\d+_choice>")


def _parse_choices(judge_response: str | None) -> list[int]:
    if not judge_response:
        return []
    return [int(c) for c in CHOICE_RE.findall(judge_response)]


def surface_pairwise_examples(
    eval_file: str,
    winner: str,
    loser: str,
    top_k: int = 10,
    min_judges: int = 1,
) -> list[dict]:
    """Return up to `top_k` scenarios where judges most agreed `winner` > `loser`.

    Each scenario is scored by:
      - comparison-level: how many (judge × ordering) comparisons sided with
        winner vs loser (`winner_chosen`, `loser_chosen`, `tied`).
      - criterion-level: total criterion votes summed across all comparisons
        (used as an internal tiebreaker, not in the output).

    Results are sorted by (winner_chosen − loser_chosen) desc, with criterion
    margin breaking ties.
    """
    if not os.path.exists(eval_file):
        raise FileNotFoundError(eval_file)

    by_scen: dict[int, dict] = defaultdict(lambda: {
        "scenario": None,
        "winner_response": None,
        "loser_response": None,
        "judge_breakdown": [],
        "winner_criterion_wins": 0,
        "loser_criterion_wins": 0,
        "ties": 0,
    })

    with open(eval_file) as fh:
        for line in fh:
            row = json.loads(line)
            e1, e2 = row["eval1_name"], row["eval2_name"]
            if {e1, e2} != {winner, loser}:
                continue
            choices = _parse_choices(row["judge response"])
            if not choices:
                continue
            winner_is_e1 = e1 == winner
            sd = by_scen[row["scenario_index"]]
            if sd["scenario"] is None:
                sd["scenario"] = row["scenario"]
                sd["winner_response"] = (
                    row["eval1 response"] if winner_is_e1 else row["eval2 response"]
                )
                sd["loser_response"] = (
                    row["eval2 response"] if winner_is_e1 else row["eval1 response"]
                )

            w_wins = l_wins = c_ties = 0
            for c in choices:
                if c == 0:
                    c_ties += 1
                elif (c == 1 and winner_is_e1) or (c == 2 and not winner_is_e1):
                    w_wins += 1
                elif (c == 2 and winner_is_e1) or (c == 1 and not winner_is_e1):
                    l_wins += 1

            sd["winner_criterion_wins"] += w_wins
            sd["loser_criterion_wins"] += l_wins
            sd["ties"] += c_ties
            winner_reflection = (
                row.get("eval1 reflection") if winner_is_e1 else row.get("eval2 reflection")
            )
            loser_reflection = (
                row.get("eval2 reflection") if winner_is_e1 else row.get("eval1 reflection")
            )
            sd["judge_breakdown"].append({
                "judge": row["judge_name"],
                "ordering": "winner_first" if winner_is_e1 else "loser_first",
                "winner_wins": w_wins,
                "loser_wins": l_wins,
                "ties": c_ties,
                "verdict": (
                    "winner" if w_wins > l_wins
                    else "loser" if l_wins > w_wins
                    else "tie"
                ),
                "winner_reflection": winner_reflection,
                "loser_reflection": loser_reflection,
            })

    results = []
    for idx, sd in by_scen.items():
        if len(sd["judge_breakdown"]) < min_judges:
            continue
        winner_chosen = sum(1 for j in sd["judge_breakdown"] if j["verdict"] == "winner")
        loser_chosen = sum(1 for j in sd["judge_breakdown"] if j["verdict"] == "loser")
        tied = sum(1 for j in sd["judge_breakdown"] if j["verdict"] == "tie")
        _criterion_net = sd["winner_criterion_wins"] - sd["loser_criterion_wins"]
        results.append({
            "scenario_index": idx,
            "scenario": sd["scenario"],
            "n_judges": len(sd["judge_breakdown"]),
            "winner_chosen": winner_chosen,
            "loser_chosen": loser_chosen,
            "tied": tied,
            "winner_response": sd["winner_response"],
            "loser_response": sd["loser_response"],
            "judge_breakdown": sd["judge_breakdown"],
            "_criterion_net": _criterion_net,
        })

    results.sort(
        key=lambda r: (r["winner_chosen"] - r["loser_chosen"], r["_criterion_net"]),
        reverse=True,
    )
    for r in results:
        r.pop("_criterion_net", None)
    return results[:top_k]


def _list_models(eval_file: str) -> Counter:
    """Return Counter of model names (eval1_name + eval2_name) in the file."""
    counts: Counter = Counter()
    with open(eval_file) as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in ("eval1_name", "eval2_name"):
                if key in row:
                    counts[row[key]] += 1
    return counts


def _prompt_choice(models: list[str], label: str) -> str:
    print(f"\nSelect {label}:")
    for i, m in enumerate(models):
        print(f"  [{i}] {m}")
    while True:
        raw = input(f"{label} (number or exact name): ").strip()
        if not raw:
            continue
        if raw.isdigit() and 0 <= int(raw) < len(models):
            return models[int(raw)]
        if raw in models:
            return raw
        print(f"  '{raw}' not in list, try again.")


def _slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("eval_file", help="Path to evaluations JSONL file.")
    ap.add_argument("--model-a", help="Winner model name (judges prefer this).")
    ap.add_argument("--model-b", help="Loser model name.")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--min-judges", type=int, default=1)
    ap.add_argument("--out", help="Override output path (default: alongside eval file).")
    args = ap.parse_args()

    if not os.path.exists(args.eval_file):
        sys.exit(f"error: file not found: {args.eval_file}")

    counts = _list_models(args.eval_file)
    if not counts:
        sys.exit("error: no rows with eval1_name/eval2_name found in file.")
    models_sorted = [m for m, _ in counts.most_common()]

    # Resolve model-a
    model_a = args.model_a
    if model_a is None:
        print(f"Models found in {args.eval_file} ({len(models_sorted)} unique):")
        for m in models_sorted:
            print(f"  - {m}  ({counts[m]} rows)")
        model_a = _prompt_choice(models_sorted, "model A (winner)")
    elif model_a not in counts:
        sys.exit(f"error: --model-a '{model_a}' not present in file. "
                 f"Found: {', '.join(models_sorted)}")

    # Resolve model-b
    model_b = args.model_b
    if model_b is None:
        remaining = [m for m in models_sorted if m != model_a]
        model_b = _prompt_choice(remaining, "model B (loser)")
    elif model_b not in counts:
        sys.exit(f"error: --model-b '{model_b}' not present in file.")
    if model_a == model_b:
        sys.exit("error: model A and model B must differ.")

    examples = surface_pairwise_examples(
        args.eval_file, winner=model_a, loser=model_b,
        top_k=args.top_k, min_judges=args.min_judges,
    )

    out = args.out
    if out is None:
        eval_dir = os.path.dirname(os.path.abspath(args.eval_file))
        stem = os.path.splitext(os.path.basename(args.eval_file))[0]
        out = os.path.join(
            eval_dir,
            f"{stem}__{_slugify(model_a)}_beats_{_slugify(model_b)}__top{args.top_k}.json",
        )

    payload = {
        "eval_file": os.path.abspath(args.eval_file),
        "winner_name": model_a,
        "loser_name": model_b,
        "top_k": args.top_k,
        "n_examples": len(examples),
        "examples": examples,
    }

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(examples)} examples ({model_a} beats {model_b}) -> {out}")


if __name__ == "__main__":
    main()
