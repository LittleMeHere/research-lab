"""Run one model end to end: baseline -> threshold -> incentive conditions ->
judges -> plot.

  uv run python -m value_leakage.run --count 100
"""

import asyncio
import json
from datetime import datetime
from pathlib import Path

import fire
import numpy as np

from value_leakage import plot as plot_mod
from value_leakage.judge import _judge
from value_leakage.sample import sample

RUNS_ROOT = Path("runs")


def compute_threshold(estimates: list, run_path: Path) -> int:
    """Median of parsed baseline estimates (paper: int(round(percentile 50)))."""
    valid = [e for e in estimates if e is not None]
    if not valid:
        raise ValueError("no baseline estimate parsed — cannot set a threshold")
    record = {
        "threshold": int(round(float(np.percentile(valid, 50)))),
        "n_baseline": len(estimates),
        "n_valid": len(valid),
        "unknown_rate": 1 - len(valid) / len(estimates),
    }
    (run_path / "threshold.json").write_text(json.dumps(record, indent=2))
    print(f"threshold = {record['threshold']:,} "
          f"(n_valid={record['n_valid']}/{record['n_baseline']})")
    return record["threshold"]


async def pipeline(run_path: Path, target_model: str, target_backend: str,
                   target_provider: str | None, count: int,
                   target_concurrent: int, target_max_tokens: int,
                   target_reasoning_effort: str | None,
                   judge_model: str, judge_concurrent: int) -> None:
    sample_kw = dict(count=count, max_concurrent=target_concurrent,
                     model=target_model, backend=target_backend,
                     provider=target_provider,
                     max_tokens=target_max_tokens,
                     reasoning_effort=target_reasoning_effort)

    await sample(condition="baseline", threshold=None,
                 out=str(run_path / "baseline.json"), **sample_kw)

    estimates = await _judge("estimates", run_path, judge_model, judge_concurrent)
    (run_path / "estimates.json").write_text(json.dumps(estimates, indent=2))
    threshold = compute_threshold(estimates["baseline"], run_path)

    await asyncio.gather(*[
        sample(condition=c, threshold=threshold,
               out=str(run_path / f"{c}.json"), **sample_kw)
        for c in ("below_good", "above_good")
    ])

    trajectories = await _judge("trajectories", run_path, judge_model, judge_concurrent)
    (run_path / "trajectories.json").write_text(json.dumps(trajectories, indent=2))


def main(
    target_model: str = "accounts/fireworks/models/qwen3p5-122b-a10b",
    target_backend: str = "fireworks",
    target_provider: str | None = None,
    count: int = 100,
    target_concurrent: int = 20,
    target_max_tokens: int = 64000,
    target_reasoning_effort: str | None = "high",
    judge_model: str = "claude-sonnet-5",
    judge_concurrent: int = 50,
):
    """Run one model through the whole giraffe-prompt pipeline. target_* is the
    model under test; judge_* reads its output. target_provider pins one
    OpenRouter upstream."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_path = RUNS_ROOT / f"{target_model.split('/')[-1]}_{stamp}"
    run_path.mkdir(parents=True, exist_ok=True)

    (run_path / "config.json").write_text(json.dumps({
        "model": target_model.split("/")[-1], "model_id": target_model,
        "backend": target_backend, "provider": target_provider,
        "task": "giraffes", "count": count,
        "target_max_tokens": target_max_tokens,
        "target_reasoning_effort": target_reasoning_effort,
        "judge_model": judge_model,
    }, indent=2))

    asyncio.run(pipeline(run_path, target_model, target_backend, target_provider,
                         count, target_concurrent, target_max_tokens,
                         target_reasoning_effort, judge_model, judge_concurrent))
    plot_mod.plot(str(run_path))


if __name__ == "__main__":
    fire.Fire(main)
