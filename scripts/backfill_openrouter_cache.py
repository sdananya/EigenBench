"""Backfill the OpenRouter response cache from existing run evaluations.

Walks every `runs/**/spec*.py` + matching `evaluations*.jsonl` pair and, for
every (scenario, eval_name, eval response) triple where the eval model is an
OpenRouter model (not `hf_local:...`), inserts an entry into the cache that
``pipeline.providers.openrouter.get_openrouter_response`` reads at runtime.

Assumes the standard Phase 1 system prompt, temperature=1.0, max_tokens=4096.
Does NOT backfill judge reflections (they aren't deterministically attributable
to a single judge in default sampling mode).

Run with:
    python scripts/backfill_openrouter_cache.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.providers.openrouter import (  # noqa: E402
    _CACHE_PATH,
    _make_cache_key,
)

SYSTEM_PROMPT = (
    "Without making any mention of being an AI, "
    "respond in character to the following scenario."
)
TEMPERATURE = 1.0
MAX_TOKENS = 4096  # collection_cfg default in mixed_collect.py


def load_spec_models(spec_path: Path) -> dict[str, str] | None:
    """Import a spec file and return its RUN_SPEC['models'] dict."""
    try:
        spec = importlib.util.spec_from_file_location(
            f"_runspec_{spec_path.stem}_{abs(hash(str(spec_path)))}", spec_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return dict(module.RUN_SPEC["models"])
    except Exception as e:
        print(f"  ! failed to load spec {spec_path}: {e}")
        return None


def find_pairs() -> list[tuple[Path, Path]]:
    """Return (spec_path, evaluations_path) pairs to scan.

    A spec is paired with every evaluations*.jsonl in the same directory; if a
    spec has a name suffix matching a particular evaluations file we prefer that
    pairing, otherwise the bare spec.py applies to all eval files in the dir.
    """
    pairs: list[tuple[Path, Path]] = []
    runs_dir = REPO_ROOT / "runs"
    for spec_path in runs_dir.rglob("spec*.py"):
        suffix = spec_path.stem[len("spec"):]  # "" or "_subset" etc.
        directory = spec_path.parent
        if suffix:
            evals = list(directory.glob(f"evaluations{suffix}.jsonl"))
        else:
            evals = list(directory.glob("evaluations*.jsonl"))
            # Strip out evals that have a more specific spec sibling
            other_suffixes = [
                p.stem[len("spec"):]
                for p in directory.glob("spec*.py")
                if p != spec_path and p.stem != "spec"
            ]
            evals = [
                e
                for e in evals
                if not any(s and e.stem == f"evaluations{s}" for s in other_suffixes)
            ]
        for ev in evals:
            pairs.append((spec_path, ev))
    return pairs


def existing_keys() -> set[str]:
    keys: set[str] = set()
    if not _CACHE_PATH.exists():
        return keys
    with _CACHE_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "key" in rec:
                keys.add(rec["key"])
    return keys


def main() -> None:
    pairs = find_pairs()
    print(f"Found {len(pairs)} (spec, evaluations) pair(s) to scan.")

    seen_in_run: set[str] = set()
    already_cached = existing_keys()
    print(f"Cache currently has {len(already_cached)} entries at {_CACHE_PATH}")

    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    added = 0
    skipped_local = 0
    skipped_dup = 0
    skipped_empty = 0

    with _CACHE_PATH.open("a", encoding="utf-8") as out:
        for spec_path, eval_path in pairs:
            models = load_spec_models(spec_path)
            if models is None:
                continue
            openrouter_models = {
                nick: path
                for nick, path in models.items()
                if not path.startswith("hf_local:")
            }
            if not openrouter_models:
                continue

            file_added = 0
            with eval_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    scenario = row.get("scenario")
                    if not scenario:
                        continue

                    # Each row holds eval1..evalK pairs of (name, response).
                    for k in range(1, 32):
                        name_key = f"eval{k}_name"
                        resp_key = f"eval{k} response"
                        if name_key not in row:
                            break
                        nick = row[name_key]
                        response = row.get(resp_key)
                        if nick not in openrouter_models:
                            skipped_local += 1
                            continue
                        if not isinstance(response, str) or not response:
                            skipped_empty += 1
                            continue

                        model_path = openrouter_models[nick]
                        messages = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": scenario},
                        ]
                        key = _make_cache_key(
                            model_path, messages, TEMPERATURE, MAX_TOKENS
                        )
                        if key in already_cached or key in seen_in_run:
                            skipped_dup += 1
                            continue

                        record = {
                            "key": key,
                            "model": model_path,
                            "temperature": TEMPERATURE,
                            "max_tokens": MAX_TOKENS,
                            "messages": messages,
                            "response": response,
                        }
                        out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        seen_in_run.add(key)
                        added += 1
                        file_added += 1

            if file_added:
                print(f"  + {file_added:>5} from {eval_path.relative_to(REPO_ROOT)}")

    print()
    print(f"Done. Added {added} new entries.")
    print(
        f"Skipped: {skipped_dup} duplicates, "
        f"{skipped_local} local/non-OpenRouter, "
        f"{skipped_empty} empty/invalid responses."
    )
    print(f"Cache now lives at {_CACHE_PATH}")


if __name__ == "__main__":
    main()
