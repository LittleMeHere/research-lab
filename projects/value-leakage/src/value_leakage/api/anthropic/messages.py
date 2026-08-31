"""Anthropic Messages API utils.

Never add temperature/top_p/top_k — non-default values 400 on Sonnet 5+.
Optional args are omitted from the request when None.
"""

import asyncio
import os

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from tqdm.asyncio import tqdm_asyncio

load_dotenv()


def get_anthropic_client() -> AsyncAnthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY not found in .env file!")

    return AsyncAnthropic(api_key=api_key)


async def call_api(
    client: AsyncAnthropic,
    model: str,
    messages: list,
    max_tokens: int = 16000,
    system: str | list | None = None,
    thinking: dict | None = None,
    output_config: dict | None = None,
    stop_sequences: list[str] | None = None,
    extra_body: dict | None = None,
):
    """max_tokens caps thinking plus visible text."""
    kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens}

    if system is not None:
        kwargs["system"] = system
    if thinking is not None:
        kwargs["thinking"] = thinking
    if output_config is not None:
        kwargs["output_config"] = output_config
    if stop_sequences is not None:
        kwargs["stop_sequences"] = stop_sequences
    if extra_body is not None:
        kwargs["extra_body"] = extra_body

    return await client.messages.create(**kwargs)


async def process_one(
    client: AsyncAnthropic,
    model: str,
    messages: list,
    semaphore: asyncio.Semaphore,
    max_tokens: int = 16000,
    system: str | list | None = None,
    thinking: dict | None = None,
    output_config: dict | None = None,
    stop_sequences: list[str] | None = None,
    extra_body: dict | None = None,
):
    async with semaphore:
        return await call_api(
            client=client,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            system=system,
            thinking=thinking,
            output_config=output_config,
            stop_sequences=stop_sequences,
            extra_body=extra_body,
        )


async def process_batch(
    client: AsyncAnthropic,
    model: str,
    messages_list: list,
    max_tokens: int = 16000,
    max_concurrent: int = 10,
    system: str | list | None = None,
    thinking: dict | None = None,
    output_config: dict | None = None,
    stop_sequences: list[str] | None = None,
    extra_body: dict | None = None,
    return_exceptions: bool = False,
) -> list:
    semaphore = asyncio.Semaphore(max_concurrent)
    coroutines = [
        process_one(
            client=client,
            model=model,
            messages=m,
            semaphore=semaphore,
            max_tokens=max_tokens,
            system=system,
            thinking=thinking,
            output_config=output_config,
            stop_sequences=stop_sequences,
            extra_body=extra_body,
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


def extract_text(response) -> str:
    """Empty on refusal — refusals carry no content blocks."""
    if response.stop_reason == "refusal":
        return ""
    return "".join(b.text for b in response.content if b.type == "text")
