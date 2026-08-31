"""Small interleaved GCP replication used only for the model go/no-go."""

import os
import asyncio
import json
import math
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, RateLimitError

from value_leakage.judge import NUMBER_JUDGE_PROMPT, parse_tagged_estimate
from value_leakage.sample import build_prompt


PROJECT = os.environ.get("GCP_PROJECT", "<gcp-project>")
LOCATION = "global"
MODEL = "zai-org/glm-5.2-maas"
BASE_URL = (
    f"https://aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/"
    f"{LOCATION}/endpoints/openapi"
)


def client() -> AsyncOpenAI:
    token = subprocess.run(
        ["gcloud", "auth", "print-access-token", "--account=wayman.al@gmail.com"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return AsyncOpenAI(api_key=token, base_url=BASE_URL, timeout=180, max_retries=0)


async def one_call(api: AsyncOpenAI, semaphore: asyncio.Semaphore, body: dict):
    async with semaphore:
        for attempt in range(3):
            try:
                return await api.chat.completions.create(**body)
            except (RateLimitError, APIConnectionError, APITimeoutError):
                if attempt == 2:
                    raise
                await asyncio.sleep(2**attempt)


def flatten(index: int, launch_order: int, request: dict, response) -> dict:
    message = response.choices[0].message
    return {
        "i": index,
        "launch_order": launch_order,
        "sent_at": datetime.now(timezone.utc).isoformat(),
        "request": request,
        "response": response.model_dump(mode="json"),
        "reasoning": getattr(message, "reasoning_content", None) or "",
        "content": message.content or "",
        "finish_reason": response.choices[0].finish_reason,
        "usage": response.usage.model_dump(mode="json") if response.usage else None,
    }


async def sample_condition(
    api: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    condition: str,
    prompt: str,
    count: int,
    max_tokens: int,
) -> list[dict]:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
    }
    responses = await asyncio.gather(
        *[one_call(api, semaphore, body) for _ in range(count)]
    )
    return [flatten(i, i, body, response) for i, response in enumerate(responses)]


async def sample_incentives(
    api: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    threshold: int,
    count: int,
    max_tokens: int,
) -> dict[str, list[dict]]:
    jobs = []
    for index in range(count):
        for condition in ("below_good", "above_good"):
            prompt = build_prompt(condition, threshold)
            body = {
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": max_tokens,
            }
            jobs.append((condition, index, len(jobs), body))

    responses = await asyncio.gather(
        *[one_call(api, semaphore, body) for _, _, _, body in jobs]
    )
    rows = {"below_good": [], "above_good": []}
    for (condition, index, launch_order, body), response in zip(jobs, responses):
        rows[condition].append(flatten(index, launch_order, body, response))
    return rows


async def judge_rows(
    api: AsyncOpenAI,
    semaphore: asyncio.Semaphore,
    rows: list[dict],
) -> tuple[list[float | None], list[dict]]:
    bodies = [
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": NUMBER_JUDGE_PROMPT.format(llm_text=row["content"]),
                }
            ],
            "temperature": 0,
            "max_tokens": 64,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        for row in rows
    ]
    responses = await asyncio.gather(
        *[one_call(api, semaphore, body) for body in bodies]
    )
    raw = [response.model_dump(mode="json") for response in responses]
    estimates = [
        parse_tagged_estimate(response.choices[0].message.content)
        for response in responses
    ]
    return estimates, raw


def write_sample(path: Path, condition: str, threshold, prompt: str, rows: list) -> None:
    path.write_text(
        json.dumps(
            {
                "model": MODEL,
                "backend": "gcp",
                "project": PROJECT,
                "location": LOCATION,
                "condition": condition,
                "threshold": threshold,
                "prompt": prompt,
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


async def run(
    count: int,
    max_concurrent: int,
    target_max_tokens: int,
    baseline_only: bool,
) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("runs") / f"gcp_glm52_go_no_go_{stamp}"
    out.mkdir(parents=True)
    config = {
        "purpose": "go/no-go only",
        "model": MODEL,
        "project": PROJECT,
        "location": LOCATION,
        "count_per_condition": count,
        "max_concurrent": max_concurrent,
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": target_max_tokens,
        "judge_model": MODEL,
        "judge_thinking": False,
        "baseline_only": baseline_only,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    api = client()
    semaphore = asyncio.Semaphore(max_concurrent)

    baseline_prompt = build_prompt("baseline", None)
    baseline = await sample_condition(
        api, semaphore, "baseline", baseline_prompt, count, target_max_tokens
    )
    write_sample(out / "baseline.json", "baseline", None, baseline_prompt, baseline)
    baseline_estimates, baseline_judges = await judge_rows(api, semaphore, baseline)
    valid_baseline = [x for x in baseline_estimates if x is not None]
    if not valid_baseline:
        raise RuntimeError("No baseline estimates parsed")
    threshold = int(round(statistics.median(valid_baseline)))
    (out / "threshold.json").write_text(
        json.dumps(
            {
                "threshold": threshold,
                "n_baseline": len(baseline_estimates),
                "n_valid": len(valid_baseline),
            },
            indent=2,
        )
    )
    (out / "estimates.json").write_text(
        json.dumps({"baseline": baseline_estimates}, indent=2)
    )
    (out / "judge_responses.json").write_text(
        json.dumps({"baseline": baseline_judges}, indent=2, ensure_ascii=False)
    )
    if baseline_only:
        await api.close()
        print(json.dumps({"out": str(out), "threshold": threshold}, indent=2))
        return out

    incentive_rows = await sample_incentives(
        api,
        semaphore,
        threshold=threshold,
        count=count,
        max_tokens=target_max_tokens,
    )
    estimates = {"baseline": baseline_estimates}
    judge_responses = {"baseline": baseline_judges}
    for condition in ("below_good", "above_good"):
        prompt = build_prompt(condition, threshold)
        rows = incentive_rows[condition]
        write_sample(out / f"{condition}.json", condition, threshold, prompt, rows)
        estimates[condition], judge_responses[condition] = await judge_rows(
            api, semaphore, rows
        )

    (out / "estimates.json").write_text(json.dumps(estimates, indent=2))
    (out / "judge_responses.json").write_text(
        json.dumps(judge_responses, indent=2, ensure_ascii=False)
    )
    below = [x for x in estimates["below_good"] if x is not None and x > 0]
    above = [x for x in estimates["above_good"] if x is not None and x > 0]
    summary = {
        "count_per_condition": count,
        "threshold": threshold,
        "median": {
            "baseline": statistics.median(valid_baseline),
            "below_good": statistics.median(below),
            "above_good": statistics.median(above),
        },
        "fraction_above_threshold": {
            "below_good": sum(x > threshold for x in below) / len(below),
            "above_good": sum(x > threshold for x in above) / len(above),
        },
        "median_log_shift_above_minus_below": (
            statistics.median(map(math.log, above))
            - statistics.median(map(math.log, below))
        ),
        "valid": {
            "baseline": len(valid_baseline),
            "below_good": len(below),
            "above_good": len(above),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    await api.close()
    print(json.dumps({"out": str(out), **summary}, indent=2))
    return out


def main(
    count: int = 12,
    max_concurrent: int = 4,
    target_max_tokens: int = 16000,
    baseline_only: bool = False,
) -> None:
    asyncio.run(
        run(
            count=count,
            max_concurrent=max_concurrent,
            target_max_tokens=target_max_tokens,
            baseline_only=baseline_only,
        )
    )


if __name__ == "__main__":
    import fire

    fire.Fire(main)
