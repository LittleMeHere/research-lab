"""Fireworks AI chat completions API utils."""

import asyncio
import os

from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_random_exponential
from tqdm.asyncio import tqdm_asyncio

load_dotenv()


def get_fireworks_client() -> AsyncOpenAI:
    """Get an AsyncOpenAI client configured for Fireworks AI."""
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        raise ValueError("FIREWORKS_API_KEY not found in .env file!")

    return AsyncOpenAI(
        base_url="https://api.fireworks.ai/inference/v1",
        api_key=api_key,
    )


@retry(
    retry=retry_if_exception_type((RateLimitError, APIConnectionError, APITimeoutError)),
    stop=stop_after_attempt(8),
    wait=wait_random_exponential(multiplier=2, max=60),
)
async def call_api(
    client: AsyncOpenAI,
    model: str,
    messages: list,
    temperature: float = 1.0,
    max_tokens: int = 5000,
    top_p: float = 1.0,
    top_k: int | None = None,
    logprobs: bool | int = False,
    top_logprobs: int | None = None,
    echo: bool = False,
    echo_last: int | None = None,
    return_token_ids: bool = False,
    raw_output: bool = False,
    reasoning_effort: str | None = None,
    reasoning_history: str | None = None,
    extra_body: dict | None = None,
    stream: bool = False,
):
    """
    Make a chat completion API call to Fireworks.

    Args:
        client: AsyncOpenAI client configured for Fireworks.
        model: Model identifier (e.g., "accounts/fireworks/models/deepseek-v3").
        messages: Chat messages.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling (0-100).
        logprobs: Enable logprobs. Can be bool or int (0-5).
        top_logprobs: Number of top logprobs per position (0-5).
        echo: Echo back prompt in addition to completion.
        echo_last: Echo back last N tokens of prompt.
        return_token_ids: Return token IDs alongside text.
        raw_output: Return raw output with prompt_token_ids, completion_token_ids, completion_logprobs.
        reasoning_effort: Control reasoning behavior ("low", "medium", "high", "none").
        reasoning_history: Control historical reasoning content ("disabled", "interleaved", "preserved").
        extra_body: Additional parameters to pass in request body.

    Returns:
        Full response object from Fireworks API.
    """
    # Build extra_body with Fireworks-specific parameters
    body = extra_body.copy() if extra_body else {}

    if top_k is not None:
        body["top_k"] = top_k
    if echo:
        body["echo"] = echo
    if echo_last is not None:
        body["echo_last"] = echo_last
    if return_token_ids:
        body["return_token_ids"] = return_token_ids
    if raw_output:
        body["raw_output"] = raw_output
    if reasoning_effort is not None:
        body["reasoning_effort"] = reasoning_effort
    if reasoning_history is not None:
        body["reasoning_history"] = reasoning_history

    # Handle logprobs parameter (can be bool or int 0-5)
    # Note: bool is subclass of int in Python, so check bool first
    if isinstance(logprobs, bool):
        lp_enabled = logprobs
        lp_top = top_logprobs
    else:
        # logprobs is an int specifying number of top logprobs
        lp_enabled = True
        lp_top = top_logprobs if top_logprobs is not None else logprobs

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        logprobs=lp_enabled,
        top_logprobs=lp_top,
        extra_body=body if body else None,
        stream=stream
    )
    return response


async def process_one(
    client: AsyncOpenAI,
    model: str,
    messages: list,
    semaphore: asyncio.Semaphore,
    temperature: float = 1.0,
    max_tokens: int = 5000,
    top_p: float = 1.0,
    top_k: int | None = None,
    logprobs: bool | int = False,
    top_logprobs: int | None = None,
    echo: bool = False,
    echo_last: int | None = None,
    return_token_ids: bool = False,
    raw_output: bool = False,
    reasoning_effort: str | None = None,
    reasoning_history: str | None = None,
    extra_body: dict | None = None,
    stream: bool = False,
):
    """Process a single request with semaphore-based concurrency control."""
    async with semaphore:
        return await call_api(
            client=client,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            echo=echo,
            echo_last=echo_last,
            return_token_ids=return_token_ids,
            raw_output=raw_output,
            reasoning_effort=reasoning_effort,
            reasoning_history=reasoning_history,
            extra_body=extra_body,
            stream=stream,
        )


async def process_batch(
    client: AsyncOpenAI,
    model: str,
    messages_list: list,
    temperature: float = 1.0,
    max_tokens: int = 5000,
    max_concurrent: int = 10,
    top_p: float = 1.0,
    top_k: int | None = None,
    logprobs: bool | int = False,
    top_logprobs: int | None = None,
    echo: bool = False,
    echo_last: int | None = None,
    return_token_ids: bool = False,
    raw_output: bool = False,
    reasoning_effort: str | None = None,
    reasoning_history: str | None = None,
    extra_body: dict | None = None,
    return_exceptions: bool = False,
    stream: bool = False,
) -> list:
    """
    Process all requests concurrently with a semaphore.

    Args:
        client: AsyncOpenAI client configured for Fireworks.
        model: Model identifier.
        messages_list: List of chat message lists.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens in response.
        max_concurrent: Maximum concurrent requests.
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling (0-100).
        logprobs: Enable logprobs. Can be bool or int (0-5).
        top_logprobs: Number of top logprobs per position (0-5).
        echo: Echo back prompt in addition to completion.
        echo_last: Echo back last N tokens of prompt.
        return_token_ids: Return token IDs alongside text.
        raw_output: Return raw output with prompt_token_ids, completion_token_ids, completion_logprobs.
        reasoning_effort: Control reasoning behavior ("low", "medium", "high", "none").
        reasoning_history: Control historical reasoning content ("disabled", "interleaved", "preserved").
        extra_body: Additional parameters to pass in request body.
        return_exceptions: If True, exceptions are returned in results instead of raised.

    Returns:
        List of response objects from Fireworks API.
    """
    semaphore = asyncio.Semaphore(max_concurrent)
    coroutines = [
        process_one(
            client=client,
            model=model,
            messages=m,
            semaphore=semaphore,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            logprobs=logprobs,
            top_logprobs=top_logprobs,
            echo=echo,
            echo_last=echo_last,
            return_token_ids=return_token_ids,
            raw_output=raw_output,
            reasoning_effort=reasoning_effort,
            reasoning_history=reasoning_history,
            extra_body=extra_body,
            stream=stream,
        )
        for m in messages_list
    ]

    if return_exceptions:
        async def wrap_with_progress(coro, pbar):
            try:
                result = await coro
                pbar.update(1)
                return result
            except Exception as e:
                pbar.update(1)
                return e

        from tqdm import tqdm
        pbar = tqdm(total=len(coroutines))
        wrapped = [wrap_with_progress(c, pbar) for c in coroutines]
        results = await asyncio.gather(*wrapped)
        pbar.close()
        return results
    else:
        return await tqdm_asyncio.gather(*coroutines)
