"""API provider helper.

Routes by model prefix: anthropic/* -> Anthropic API directly, openai/* ->
OpenAI API directly (both when their keys are present; higher rate limits than
the OpenRouter middleman), everything else -> OpenRouter. Falls back to
OpenRouter when a direct call keeps failing. Retries with backoff everywhere;
returns an "Error in OpenRouter API call: ..." string only after all retries
(kept verbatim so downstream error accounting is unchanged).
"""

from __future__ import annotations

import os
import time

import openai
from dotenv import load_dotenv

_RETRIES = 4
_BACKOFF = 4  # seconds, doubled per attempt

_ANTHROPIC_IDS = {"anthropic/claude-sonnet-4.5": "claude-sonnet-4-5-20250929",
                  "anthropic/claude-3-haiku": "claude-3-haiku-20240307"}
_OPENAI_IDS = {"openai/gpt-5": "gpt-5", "openai/gpt-4.1": "gpt-4.1",
               "openai/gpt-4o": "gpt-4o", "openai/gpt-4o-mini": "gpt-4o-mini",
               "openai/gpt-3.5-turbo": "gpt-3.5-turbo"}


def _direct_anthropic(model_id, messages, temperature, max_tokens):
    import anthropic
    client = anthropic.Anthropic()  # ANTHROPIC_API_KEY
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    chat = [m for m in messages if m["role"] != "system"]
    kwargs = dict(model=model_id, messages=chat, max_tokens=max_tokens,
                  temperature=temperature)
    if system:
        kwargs["system"] = system
    try:
        r = client.messages.create(**kwargs)
    except (anthropic.BadRequestError, TypeError) as e:
        # SDK 1.x removed the temperature kwarg (TypeError); some models 400 on it
        if "temperature" in str(e):
            kwargs.pop("temperature", None)
            r = client.messages.create(**kwargs)
        else:
            raise
    return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")


def _direct_openai(model_id, messages, temperature, max_tokens):
    key = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")
    client = openai.OpenAI(api_key=key)
    kwargs = dict(model=model_id, messages=messages, temperature=temperature,
                  max_tokens=max_tokens)
    for _ in range(3):  # progressively drop params reasoning models reject
        try:
            r = client.chat.completions.create(**kwargs)
            return r.choices[0].message.content
        except openai.BadRequestError as e:
            s = str(e)
            if "max_tokens" in s and "max_tokens" in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            elif "temperature" in s and "temperature" in kwargs:
                kwargs.pop("temperature")
            else:
                raise
    raise RuntimeError("openai param negotiation failed")


def _via_openrouter(model, messages, temperature, max_tokens, return_full_response=False):
    client = openai.OpenAI(base_url="https://openrouter.ai/api/v1",
                           api_key=os.getenv("OPENROUTER_API_KEY"))
    response = client.chat.completions.create(
        model=model, temperature=temperature, max_tokens=max_tokens, messages=messages)
    if return_full_response:
        return response
    return response.choices[0].message.content


def get_openrouter_response(
    messages,
    model: str,
    temperature: float = 1.0,
    max_tokens: int = 1024,
    return_full_response: bool = False,
):
    """Send one chat completion request (direct API when possible, else OpenRouter)."""

    load_dotenv()
    last_err = None
    for attempt in range(_RETRIES):
        # First two attempts use the direct API when available; later attempts
        # (or full-response requests) go through OpenRouter.
        try:
            if not return_full_response and attempt < 2:
                if model in _ANTHROPIC_IDS and os.getenv("ANTHROPIC_API_KEY"):
                    content = _direct_anthropic(_ANTHROPIC_IDS[model], messages,
                                                temperature, max_tokens)
                elif model in _OPENAI_IDS and (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")):
                    content = _direct_openai(_OPENAI_IDS[model], messages,
                                             temperature, max_tokens)
                else:
                    content = _via_openrouter(model, messages, temperature, max_tokens)
            else:
                content = _via_openrouter(model, messages, temperature, max_tokens,
                                          return_full_response)
                if return_full_response:
                    return content
            if content is None:
                raise ValueError("empty completion content")
            return content
        except Exception as e:  # noqa: BLE001 - provider errors are heterogeneous
            last_err = e
            wait = _BACKOFF * (2 ** attempt)
            print(f"API error (attempt {attempt+1}/{_RETRIES}, model={model}): {e}; retrying in {wait}s")
            time.sleep(wait)
    print(f"Error in OpenRouter API call: {last_err}")
    return f"Error in OpenRouter API call: {last_err}"
