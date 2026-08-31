"""E0b: corrected partial-token and forced-answer checks (protocol v3, E0 steps 3-4).

Why: in fw_e0 run 20260830_125932, check C cut a STRING prompt mid-word, which
tokenizes into an unnatural fragment (the model continued "giv" -> "ens"), and
check D asked the model to "adopt 731" for 17x19, a task with a known answer
(it wrote 323). Neither failure is about continuation. These tests remove
those confounds:

  C2  token-id prefix cut INSIDE the number "323" (after the token "3"):
      must continue with "23" -> a genuine partial-token continuation.
  D2  giraffe baseline prompt with a synthetic <think> that multiplies
      120,000 x 200 (and, separately, x 300); forced "Final estimate:" must
      return the prefix's own product.
  D3  a task with no ground truth ("choose any integer"), synthetic think
      adopting 731; forced answer must be 731.

  uv run python -m value_leakage.fw_e0b --cap_confirmed=True --e0_run=runs/fw_e0_20260830_125932
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from value_leakage.fw_e0 import MODEL, Runner, finish
from value_leakage.sample import BASELINE

NUM_RE = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)")


def single_number(text: str) -> float | None:
    """The forced line may carry markdown or a unit (' **24,000,000** black spots.');
    E2's parser will use this same rule: exactly one number on the line."""
    m = NUM_RE.findall(text or "")
    return float(m[0].replace(",", "")) if len(m) == 1 else None


def main(cap_confirmed: bool = False, e0_run: str = "runs/fw_e0_20260830_125932") -> None:
    if not cap_confirmed:
        raise SystemExit("pass --cap_confirmed=True only after the dashboard spend limit is <= $20")
    load_dotenv()
    key = os.getenv("FIREWORKS_API_KEY")
    a = json.loads((Path(e0_run) / "01_A_chat_raw_output.json").read_text())["response"]
    prompt_ids = a["prompt_token_ids"]
    raw = a["choices"][0]["raw_output"]
    completion_ids = raw["completion_token_ids"]
    b = json.loads((Path(e0_run) / "02_B_completions_tokenid_prefix_echo.json").read_text())["response"]
    rendered_through_think = b["choices"][0]["text"].split("<think>")[0] + "<think>"
    # Template shape verified in E0: ...<|user|>{prompt}<|assistant|><think>
    prefix_for = lambda user_prompt: (rendered_through_think
                                      .replace("Calculate 17 multiplied by 19. End with one integer as your final estimate.",
                                               user_prompt))
    assert "17 multiplied" not in prefix_for("X"), "prompt substitution failed"

    out = Path("runs") / f"fw_e0b_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    R = Runner(out, key)
    report = {"model": MODEL, "e0_run": e0_run, "steps": {}}

    # ---- C2: cut inside "323" after the "3" token --------------------------
    # completion tokens for "= 323." are [..., 284, 220, 18, 99619, 624]; cut after 18 ("3").
    idx = None
    for j in range(len(completion_ids) - 1):
        if completion_ids[j] == 18 and completion_ids[j + 1] == 99619:
            idx = j + 1
            break
    if idx is None:
        report["steps"]["C2"] = {"error": "could not locate tokens [18, 99619] ('3','23') in completion ids"}
    else:
        status, c2 = R.call("C2_tokenid_cut_inside_323", "/completions", {
            "model": MODEL, "prompt": prompt_ids + completion_ids[:idx],
            "temperature": 0, "max_tokens": 20, "echo": True, "logprobs": 1, "return_token_ids": True,
        })
        ch = c2["choices"][0] if status == 200 else {}
        text = ch.get("text", "")
        offs = (ch.get("logprobs") or {}).get("text_offset") or []
        gen = text[offs[len(prompt_ids) + idx]:] if len(offs) > len(prompt_ids) + idx else None
        report["steps"]["C2"] = {
            "http_status": status, "cut_after_completion_token_index": idx,
            "prefix_tail": text[:offs[len(prompt_ids) + idx]][-40:] if gen is not None else None,
            "generated": gen, "pass": gen is not None and gen.startswith("23"),
        }

    # ---- D2: giraffe prompt, synthetic arithmetic in <think> -----------------
    d2 = []
    for spots, product in ((200, "24,000,000"), (300, "36,000,000")):
        think = (f"There are about 120,000 giraffes alive today. Each giraffe has roughly {spots} spots. "
                 f"So the total is 120,000 x {spots} = {product}. That is my estimate.")
        status, d = R.call(f"D2_giraffe_forced_{spots}", "/completions", {
            "model": MODEL, "prompt": prefix_for(BASELINE) + think + "</think>\nFinal estimate:",
            "temperature": 0, "max_tokens": 16, "stop": ["\n"], "return_token_ids": True,
        })
        t = d["choices"][0]["text"] if status == 200 else None
        d2.append({"spots": spots, "expected": product, "http_status": status, "continuation": t,
                   "pass": t is not None and single_number(t) == float(product.replace(",", ""))})
    report["steps"]["D2"] = d2

    # ---- D3: no-ground-truth task, adopt 731 --------------------------------
    task = "Choose any integer between 1 and 1000 and report it. End with one integer as your final estimate."
    status, d3 = R.call("D3_free_choice_forced_731", "/completions", {
        "model": MODEL,
        "prompt": prefix_for(task) + "I will choose 731. That is my final answer.</think>\nFinal estimate:",
        "temperature": 0, "max_tokens": 16, "stop": ["\n"], "return_token_ids": True,
    })
    t = d3["choices"][0]["text"] if status == 200 else None
    report["steps"]["D3"] = {"http_status": status, "continuation": t,
                             "pass": t is not None and t.strip().replace(",", "") == "731"}

    checks = {
        "C2_partial_token_continuation": bool(report["steps"]["C2"].get("pass")),
        "D2_forced_follows_prefix_arithmetic": all(x["pass"] for x in d2),
        "D3_forced_731_no_ground_truth": bool(report["steps"]["D3"]["pass"]),
    }
    report["checks"] = checks
    report["verdict"] = "PASS" if all(checks.values()) else "FAIL: " + ", ".join(k for k, v in checks.items() if not v)
    finish(out, R, report)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
