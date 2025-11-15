from typing import Optional, List


def llm_call(
    client,
    model: str,
    ROLE_DESCRIPTION: str,
    prompt: Optional[str] = None,
    messages: Optional[List] = None,
    seed: int = 10,
) -> str:
    """调用openai接口"""
    assert (prompt is None) != (
        messages is None
    )  # make sure exactly one of prompt or messages is provided

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
                content = chunk.choices[0].delta.content
                response_text += content
        content = response_text
    else:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.0,
            stream=False,
        )
        content = completion.choices[0].message.content
    return content
