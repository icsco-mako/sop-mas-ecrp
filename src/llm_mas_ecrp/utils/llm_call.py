import time
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

MAX_RETRIES = 5
RETRY_DELAY = 5  # seconds


def llm_call(
    client,
    model: str,
    ROLE_DESCRIPTION: str,
    prompt: Optional[str] = None,
    messages: Optional[List] = None,
    seed: int = 10,
) -> str:
    """调用openai接口，带重试机制"""
    assert (prompt is None) != (
        messages is None
    )

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
            extra_body={"enable_thinking": True},
        )
        response_text = ""
        for chunk in completion:
            if chunk.choices and chunk.choices[0].delta.content:
                response_text += chunk.choices[0].delta.content
        return response_text

    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                stream=True,
            )
            chunks = []
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)
            content = "".join(chunks)
            if content.strip():
                return content
            logger.warning(f"Empty response, attempt {attempt + 1}/{MAX_RETRIES}")
        except Exception as e:
            logger.warning(f"LLM call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")

        time.sleep(RETRY_DELAY * (attempt + 1))

    raise RuntimeError(f"LLM call failed after {MAX_RETRIES} retries")
