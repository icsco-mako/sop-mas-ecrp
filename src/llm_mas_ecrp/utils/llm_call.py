import time
import logging
import threading
from typing import Optional, List

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds

# ---------------------------------------------------------------------------
# Usage accumulator
# ---------------------------------------------------------------------------
# Captures prompt/completion/total tokens across every successful LLM call made
# through ``llm_call`` in this process. The token-overhead experiment runs
# SOP-MAS sequentially (concurrency=1), so a single process-wide accumulator is
# safe; callers ``reset_usage()`` before a run and ``get_usage()`` after.
_usage_lock = threading.Lock()
_usage: dict = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "num_calls": 0,
    "estimated": False,  # True if any call did not return a usage object
}


def reset_usage() -> None:
    """Zero the accumulator. Call immediately before a measured run."""
    with _usage_lock:
        for key in ("prompt_tokens", "completion_tokens", "total_tokens", "num_calls"):
            _usage[key] = 0
        _usage["estimated"] = False


def get_usage() -> dict:
    """Return a snapshot of the accumulated usage (copy)."""
    with _usage_lock:
        return dict(_usage)


def _accumulate(usage) -> None:
    """Add one completion's usage to the accumulator. ``usage`` may be None."""
    with _usage_lock:
        if usage is None:
            _usage["estimated"] = True
            return
        _usage["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
        _usage["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        _usage["total_tokens"] += getattr(usage, "total_tokens", 0) or 0
        _usage["num_calls"] += 1


def _stream_call(
    client,
    model: str,
    messages: list,
    temperature: float,
    top_p: float,
    max_retries: int,
    thinking: str | None = None,
) -> str:
    """Single-provider streaming call with retries."""
    for attempt in range(max_retries):
        try:
            kwargs: dict = {
                "model": model,
                "messages": messages,
                "stream": True,
                # Final chunk carries the usage object (prompt/completion/total).
                "stream_options": {"include_usage": True},
            }
            if thinking == "enabled":
                kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            else:
                kwargs["temperature"] = temperature
                kwargs["top_p"] = top_p
            completion = client.chat.completions.create(**kwargs)
            chunks = []
            usage = None
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)
                if getattr(chunk, "usage", None) is not None:
                    usage = chunk.usage
            content = "".join(chunks)
            if content.strip():
                _accumulate(usage)
                return content
            logger.warning("Empty response, attempt %d/%d", attempt + 1, max_retries)
        except Exception as e:
            logger.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
        time.sleep(RETRY_DELAY * (attempt + 1))
    raise RuntimeError(f"LLM call failed after {max_retries} retries")


def llm_call(
    client,
    model: str,
    ROLE_DESCRIPTION: str,
    prompt: Optional[str] = None,
    messages: Optional[List] = None,
    seed: int = 10,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_retries: int = MAX_RETRIES,
    fallback_client=None,
    fallback_model: Optional[str] = None,
    thinking: Optional[str] = None,
) -> str:
    """调用openai接口，带重试机制和可选的 fallback provider。"""
    assert (prompt is None) != (messages is None)

    if prompt is not None:
        messages = [
            {"role": "system", "content": ROLE_DESCRIPTION},
            {"role": "user", "content": prompt},
        ]
    if model == "qwen3-235b-a22b":
        completion = client.chat.completions.create(
            model="qwen3-235b-a22b",
            messages=messages,
            stream=True,
            stream_options={"include_usage": True},
            extra_body={"enable_thinking": True},
        )
        response_text = ""
        usage = None
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
        _accumulate(usage)
        return response_text

    try:
        return _stream_call(client, model, messages, temperature, top_p, max_retries, thinking=thinking)
    except RuntimeError:
        if fallback_client is not None and fallback_model is not None:
            logger.info("Primary (%s/%s) failed, retrying with fallback (%s)",
                        getattr(client, 'base_url', ''), model, fallback_model)
            return _stream_call(
                fallback_client, fallback_model, messages,
                temperature, top_p, MAX_RETRIES, thinking=thinking,
            )
        raise
