"""OpenRouter client helper with on-disk response caching."""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import openai
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------
#
# Calls to OpenRouter cost money and time, so we cache responses keyed by
# (model, messages, temperature, max_tokens) in a single JSONL file. On the
# next run, identical (scenario, model) requests are served from disk instead
# of being re-issued.
#
# Override the cache location with $OPENROUTER_CACHE_PATH or disable caching
# entirely with $OPENROUTER_CACHE_DISABLE=1.

_DEFAULT_CACHE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "responses" / "openrouter_cache.jsonl"
)
_CACHE_PATH = Path(os.getenv("OPENROUTER_CACHE_PATH", str(_DEFAULT_CACHE_PATH)))
_CACHE_DISABLED = os.getenv("OPENROUTER_CACHE_DISABLE", "").lower() in {"1", "true", "yes"}

_cache_lock = threading.Lock()
_cache: dict[str, str] | None = None


def _make_cache_key(model: str, messages, temperature: float, max_tokens: int) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_cache() -> dict[str, str]:
    global _cache
    if _cache is not None:
        return _cache
    cache: dict[str, str] = {}
    if _CACHE_PATH.exists():
        try:
            with _CACHE_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = record.get("key")
                    response = record.get("response")
                    if key and response is not None:
                        cache[key] = response
        except OSError:
            pass
    _cache = cache
    return _cache


def _append_cache(
    key: str,
    model: str,
    messages,
    temperature: float,
    max_tokens: int,
    response: str,
) -> None:
    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "key": key,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages,
            "response": response,
        }
        with _CACHE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"Warning: failed to write OpenRouter cache: {e}")


def get_openrouter_response(
    messages,
    model: str,
    temperature: float = 1.0,
    max_tokens: int = 1024,
    return_full_response: bool = False,
    use_cache: bool = True,
):
    """Send one chat completion request via OpenRouter, with on-disk caching.

    Cached responses are keyed by ``(model, messages, temperature, max_tokens)``.
    Caching is bypassed when ``return_full_response=True`` (callers in that
    mode need the raw API object) or when ``use_cache=False``.
    """

    cache_enabled = use_cache and not _CACHE_DISABLED and not return_full_response
    cache_key: str | None = None

    if cache_enabled:
        cache_key = _make_cache_key(model, messages, temperature, max_tokens)
        with _cache_lock:
            cache = _load_cache()
            if cache_key in cache:
                return cache[cache_key]

    try:
        load_dotenv()
        api_key = os.getenv("OPENROUTER_API_KEY")
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=messages,
        )
        if return_full_response:
            return response
        content = response.choices[0].message.content
    except Exception as e:
        print(f"Error in OpenRouter API call: {str(e)}")
        return f"Error in OpenRouter API call: {str(e)}"

    if (
        cache_enabled
        and cache_key is not None
        and isinstance(content, str)
        and content
    ):
        with _cache_lock:
            cache = _load_cache()
            if cache_key not in cache:
                cache[cache_key] = content
                _append_cache(cache_key, model, messages, temperature, max_tokens, content)

    return content
