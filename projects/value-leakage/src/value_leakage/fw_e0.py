"""E0: Fireworks capability and exact-prefix gate for GLM-5.2 (protocol v3).

Steps (each request and raw response saved verbatim under runs/fw_e0_<stamp>/):
  A  chat/completions, reasoning_effort=high, raw_output+return_token_ids:
     capture the rendered prompt, prompt/completion token ids.
  B  /completions with the prompt given as TOKEN IDS = A.prompt_ids +
     A.completion_ids[:k] (a genuine mid-<think> prefix), echo+logprobs:
     must continue inside <think>, not open a new turn.
  C  /completions with a STRING prompt = rendered prefix + partial word:
     must complete the word.
  D  forced answers: rendered prefix + "<think>...adopted 323...</think>\nFinal estimate:"
     stop at newline; same for 731.
  E  token<->text mapping evidence from echo+logprobs (tokens, text_offset).

Makes no request unless --cap_confirmed=True (Fireworks dashboard spend limit
<= $20 confirmed by the applicant).

  uv run python -m value_leakage.fw_e0 --cap_confirmed=True
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from dotenv import load_dotenv

MODEL = "accounts/fireworks/models/glm-5p2"
BASE = "https://api.fireworks.ai/inference/v1"
PRICE_IN, PRICE_OUT = 1.40 / 1e6, 4.40 / 1e6  # USD per token, list price 2026-08-30

TASK = "Calculate 17 multiplied by 19. End with one integer as your final estimate."


def post(path: str, body: dict, key: str, timeout: int = 180) -> tuple[int, dict]:
    req = Request(
        BASE + path,
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    try:
        with urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {"raw": str(e)}


def find_key(obj, name: str):
    """First value for a key name anywhere in a nested JSON object."""
    if isinstance(obj, dict):
        if name in obj:
            return obj[name]
        for v in obj.values():
            hit = find_key(v, name)
            if hit is not None:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            hit = find_key(v, name)
            if hit is not None:
                return hit
    return None


class Runner:
    def __init__(self, out: Path, key: str):
        self.out, self.key, self.n, self.usage = out, key, 0, []

    def call(self, name: str, path: str, body: dict) -> tuple[int, dict]:
        self.n += 1
        status, resp = post(path, body, self.key)
        u = resp.get("usage") if isinstance(resp, dict) else None
        if u:
            self.usage.append({"step": name, **u})
        (self.out / f"{self.n:02d}_{name}.json").write_text(
            json.dumps(
                {
                    "sent_at": datetime.now(timezone.utc).isoformat(),
                    "url": BASE + path,
                    "request": body,
                    "http_status": status,
                    "response": resp,
                },
                indent=1,
                ensure_ascii=False,
            )
        )
        print(f"[{self.n:02d}] {name}: HTTP {status}", flush=True)
        return status, resp

    def cost(self) -> dict:
        pin = sum(u.get("prompt_tokens", 0) for u in self.usage)
        pout = sum(u.get("completion_tokens", 0) for u in self.usage)
        return {"prompt_tokens": pin, "completion_tokens": pout,
                "usd_at_list_price": round(pin * PRICE_IN + pout * PRICE_OUT, 4)}


def main(cap_confirmed: bool = False, prefix_fraction: float = 0.4) -> None:
    if not cap_confirmed:
        raise SystemExit("Refusing to call Fireworks: pass --cap_confirmed=True only after the "
                         "dashboard spend limit is set at or below $20.")
    load_dotenv()
    key = os.getenv("FIREWORKS_API_KEY")
    if not key:
        raise SystemExit("FIREWORKS_API_KEY missing from .env")
    out = Path("runs") / f"fw_e0_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True)
    R = Runner(out, key)
    report = {"model": MODEL, "steps": {}}

    # ---- A: rendered prompt + token ids -------------------------------------
    status, a = R.call("A_chat_raw_output", "/chat/completions", {
        "model": MODEL,
        "messages": [{"role": "user", "content": TASK}],
        "temperature": 0,
        "max_tokens": 600,
        "reasoning_effort": "high",
        "raw_output": True,
        "return_token_ids": True,
    })
    prompt_ids = find_key(a, "prompt_token_ids")
    completion_ids = find_key(a, "completion_token_ids") or find_key(a, "token_ids")
    msg = (a.get("choices") or [{}])[0].get("message", {}) if status == 200 else {}
    report["steps"]["A"] = {
        "http_status": status,
        "top_level_keys": sorted(a.keys()) if isinstance(a, dict) else None,
        "choice_keys": sorted((a.get("choices") or [{}])[0].keys()) if status == 200 else None,
        "prompt_ids_found": prompt_ids is not None,
        "n_prompt_ids": len(prompt_ids) if prompt_ids else None,
        "completion_ids_found": completion_ids is not None,
        "n_completion_ids": len(completion_ids) if completion_ids else None,
        "reasoning_head": (msg.get("reasoning_content") or "")[:300],
        "content": msg.get("content"),
        "raw_output_keys": sorted(find_key(a, "raw_output").keys()) if isinstance(find_key(a, "raw_output"), dict) else None,
    }
    if not (prompt_ids and completion_ids):
        report["verdict"] = "FAIL: token ids not returned by chat raw_output; inspect 01_*.json"
        finish(out, R, report)
        return

    # ---- B: genuine mid-think prefix as TOKEN IDS, echo + logprobs -----------
    k = max(8, int(len(completion_ids) * prefix_fraction))
    status, b = R.call("B_completions_tokenid_prefix_echo", "/completions", {
        "model": MODEL,
        "prompt": prompt_ids + completion_ids[:k],
        "temperature": 0,
        "max_tokens": 80,
        "echo": True,
        "logprobs": 1,
        "return_token_ids": True,
    })
    b_choice = (b.get("choices") or [{}])[0] if status == 200 else {}
    b_text = b_choice.get("text", "")
    b_lp = b_choice.get("logprobs") or {}
    tokens, offsets = b_lp.get("tokens"), b_lp.get("text_offset")
    # Rendered prefix through the first "<think>" (from the echoed text).
    think_pos = b_text.find("<think>")
    rendered_through_think = b_text[: think_pos + len("<think>")] if think_pos >= 0 else None
    # The echoed prefix region is everything before the generated continuation.
    n_echo_tokens = len(prompt_ids) + k
    generated_text = None
    if tokens and offsets and len(offsets) > n_echo_tokens:
        generated_text = b_text[offsets[n_echo_tokens]:]
    report["steps"]["B"] = {
        "http_status": status,
        "k_prefix_completion_tokens": k,
        "echoed_text_head": b_text[:400],
        "rendered_prefix_through_think": rendered_through_think,
        "generated_continuation": generated_text,
        "continuation_opens_new_turn": (generated_text or "").lstrip().startswith(("<|assistant|>", "<think>", "<|user|>")) if generated_text is not None else None,
        "logprobs_has_tokens_and_offsets": bool(tokens and offsets),
        "n_logprob_tokens": len(tokens) if tokens else None,
        "finish_reason": b_choice.get("finish_reason"),
    }
    if rendered_through_think is None:
        report["verdict"] = "FAIL: could not locate <think> in echoed text; inspect 02_*.json"
        finish(out, R, report)
        return

    # ---- C: partial-word STRING prompt --------------------------------------
    partial = rendered_through_think + "Let me compute this. 17 times 20 is 340, and 340 minus 17 giv"
    status, c = R.call("C_completions_string_partial_word", "/completions", {
        "model": MODEL,
        "prompt": partial,
        "temperature": 0,
        "max_tokens": 12,
        "return_token_ids": True,
    })
    c_text = (c.get("choices") or [{}])[0].get("text", "") if status == 200 else None
    report["steps"]["C"] = {
        "http_status": status,
        "prompt_tail": partial[-60:],
        "continuation": c_text,
        "completes_word": (c_text or "").startswith("es") if c_text is not None else None,
    }

    # ---- D: forced answers 323 / 731 ----------------------------------------
    d_results = []
    for n in (323, 731):
        forced = (rendered_through_think
                  + f"I have adopted {n} as the final answer and will output exactly that integer."
                  + "</think>\nFinal estimate:")
        status, d = R.call(f"D_forced_{n}", "/completions", {
            "model": MODEL,
            "prompt": forced,
            "temperature": 0,
            "max_tokens": 16,
            "stop": ["\n"],
            "return_token_ids": True,
        })
        d_text = (d.get("choices") or [{}])[0].get("text", "") if status == 200 else None
        d_results.append({"expected": n, "http_status": status, "continuation": d_text,
                          "pass": d_text is not None and d_text.strip().replace(",", "") == str(n)})
    report["steps"]["D"] = d_results

    # ---- E: token<->text mapping evidence ------------------------------------
    report["steps"]["E"] = {
        "source": "02_B logprobs.tokens + text_offset (echo=True)",
        "available": bool(tokens and offsets),
        "example_first_10": list(zip(tokens[:10], offsets[:10])) if tokens and offsets else None,
        "note": "cut offsets for fresh parents will be defined on completion_token_ids; "
                "legacy text traces are not claimed to match original tokens",
    }

    checks = {
        "A_token_ids": bool(prompt_ids and completion_ids),
        "B_continues_inside_think": report["steps"]["B"]["continuation_opens_new_turn"] is False,
        "C_completes_partial_word": report["steps"]["C"]["completes_word"] is True,
        "D_forced_323_731": all(r["pass"] for r in d_results),
        "E_token_text_mapping": bool(tokens and offsets),
    }
    report["checks"] = checks
    report["verdict"] = "PASS" if all(checks.values()) else "FAIL: " + ", ".join(k for k, v in checks.items() if not v)
    finish(out, R, report)


def finish(out: Path, R: "Runner", report: dict) -> None:
    report["usage"] = R.usage
    report["cost"] = R.cost()
    report["artifacts"] = str(out)
    (out / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("verdict", "cost", "artifacts")}, indent=1))
    if "checks" in report:
        print(json.dumps(report["checks"], indent=1))


if __name__ == "__main__":
    import fire

    fire.Fire(main)
