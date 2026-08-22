#!/usr/bin/env python3
"""No-GPU integrity audit for the saved exploratory data.

Checks raw file completeness, within- and cross-model prompt pairing, saved
logits/activation coverage, and the mapping/completeness of judge labels. This
uses only the Python standard library so it can run before project dependencies
are installed.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
QUANTS = ["fp16", "int8", "int4_fp4", "nf4_dq"]
EXPECTED_MODELS = {
    "google/gemma-4-e2b-it",
    "microsoft/Phi-4-mini-instruct",
    "Qwen/Qwen3.5-4B",
    "Qwen/Qwen3-1.7B",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "HuggingFaceTB/SmolLM3-3B",
}


def load(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true",
                        help="treat known coverage warnings as failures")
    args = parser.parse_args()
    errors: list[str] = []
    warnings: list[str] = []

    paths = sorted(glob.glob(os.path.join(DATA_DIR, "v2_results_*.json")))
    raw = []
    for path in paths:
        try:
            raw.append((path, load(path)))
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read {os.path.basename(path)}: {exc}")

    models = {d.get("model_id") for _, d in raw}
    if models != EXPECTED_MODELS:
        errors.append(
            f"raw model set differs: missing={sorted(EXPECTED_MODELS-models)}, "
            f"extra={sorted(models-EXPECTED_MODELS)}"
        )

    cross_model_prompts: dict[str, tuple] = {}
    activation_coverage = {}
    duplicate_harmbench_warned = False
    for path, data in raw:
        model = data["model_id"]
        levels = data.get("quant_levels", {})
        if set(levels) != set(QUANTS):
            errors.append(f"{model}: quant levels are {sorted(levels)}")
            continue

        fp16 = levels["fp16"]
        ref_key = "refusal_thinking=False"
        think_key = "refusal_thinking=True"
        missing_suites = [key for key in (ref_key, think_key) if key not in fp16]
        if missing_suites:
            errors.append(f"{model}: missing {', '.join(missing_suites)}")
            continue
        suites = [
            (ref_key, 100, "HarmBench, thinking off"),
            (think_key, 100, "HarmBench, thinking on"),
            ("factual", 50, "TruthfulQA"),
            ("instruction", 10, "instruction"),
        ]
        baselines = {}
        for key, expected, label in suites:
            rows = fp16[key]["results"]
            baselines[key] = tuple((x["idx"], x["prompt"]) for x in rows)
            if len(rows) != expected or len(dict(baselines[key])) != expected:
                errors.append(f"{model}: fp16 {label} is not {expected} unique indices")
            if label not in cross_model_prompts:
                cross_model_prompts[label] = baselines[key]
            elif baselines[key] != cross_model_prompts[label]:
                errors.append(f"{model}: {label} prompts differ across models")
            if (key == ref_key and
                    len({x["prompt"] for x in rows}) != expected and
                    not duplicate_harmbench_warned):
                unique_n = len({x["prompt"] for x in rows})
                # This is harmless for paired analysis but matters when defining
                # a follow-up prompt set by prompt text.
                warnings.append(
                    f"v2 HarmBench has {expected} rows but {unique_n} unique prompt texts"
                )
                duplicate_harmbench_warned = True

        if baselines[ref_key] != baselines[think_key]:
            errors.append(f"{model}: HarmBench prompts differ between thinking modes")

        for quant in QUANTS:
            level = levels[quant]
            for key, expected, label in suites:
                rows = level[key]["results"]
                cells = tuple((x["idx"], x["prompt"]) for x in rows)
                if len(rows) != expected or len(dict(cells)) != expected:
                    errors.append(f"{model}/{quant}: {label} count/index failure")
                if cells != baselines[key]:
                    errors.append(f"{model}/{quant}: {label} prompt pairing failure")

            refs = level[ref_key]["results"]
            logits = sum(bool(x.get("logits", {}).get("probs")) for x in refs)
            if logits != 100:
                errors.append(f"{model}/{quant}: only {logits}/100 logit snapshots")
            activation_coverage[(model, quant)] = sum(
                bool(x.get("activation_norms")) for x in refs
            )

    for model in sorted(models & EXPECTED_MODELS):
        fp16_n = activation_coverage.get((model, "fp16"), 0)
        if fp16_n != 10:
            warnings.append(f"{model}: activation norms saved for {fp16_n}/10 fp16 prompts")
        for quant in QUANTS[1:]:
            n = activation_coverage.get((model, quant), 0)
            if n:
                warnings.append(f"{model}/{quant}: unexpected activation norms on {n} prompts")

    # Recreate judge_rescore.py's deterministic global IDs exactly.
    refusal_items = []
    capability_items = []
    r_gid = c_gid = 0
    gold = load(os.path.join(DATA_DIR, "truthfulqa_gold.json"))
    gold_matched = 0
    for _, data in raw:
        model = data["model_id"]
        for quant in QUANTS:
            level = data["quant_levels"][quant]
            for row in level["refusal_thinking=False"]["results"]:
                refusal_items.append((f"r{r_gid}", model, quant, row["idx"]))
                r_gid += 1
            for row in level["factual"]["results"]:
                capability_items.append((f"c{c_gid}", model, quant, row["idx"]))
                c_gid += 1
                if row["prompt"].strip().lower() in gold:
                    gold_matched += 1

    judge_specs = [
        ("judge_refusal_results.json", refusal_items, False, ("verdict", "harmful")),
        ("judge_capability_results.json", capability_items, True, ("correct",)),
        ("judge_capability_results_opus.json", capability_items, True, ("correct",)),
    ]
    judge_counts = {}
    for filename, expected_items, require_complete, value_fields in judge_specs:
        path = os.path.join(DATA_DIR, filename)
        try:
            judged = load(path)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"cannot read {filename}: {exc}")
            continue
        expected = {gid: (model, quant, idx)
                    for gid, model, quant, idx in expected_items}
        mismatch = []
        invalid = []
        for gid, value in judged.items():
            if not isinstance(value, dict):
                mismatch.append(gid)
                invalid.append(gid)
                continue
            actual = (value.get("model"), value.get("quant"), value.get("idx"))
            if expected.get(gid) != actual:
                mismatch.append(gid)
            if value_fields == ("correct",):
                valid = type(value.get("correct")) is bool
            else:
                valid = (value.get("verdict") in {"REFUSE", "COMPLY"} and
                         type(value.get("harmful")) is bool)
            if not valid:
                invalid.append(gid)
        extra = set(judged) - set(expected)
        missing = set(expected) - set(judged)
        judge_counts[filename] = (len(judged), len(missing))
        if mismatch or extra:
            errors.append(
                f"{filename}: metadata mismatches={len(mismatch)}, extra={len(extra)}"
            )
        if invalid:
            errors.append(f"{filename}: {len(invalid)} invalid judge labels")
        if require_complete and missing:
            errors.append(f"{filename}: {len(missing)} labels missing")
        elif missing:
            warnings.append(f"{filename}: {len(missing)}/{len(expected)} labels missing")

    if gold_matched != len(capability_items):
        warnings.append(
            f"TruthfulQA gold references matched {gold_matched}/{len(capability_items)} "
            "responses (unmatched items were judged from model knowledge)"
        )

    print("=" * 72)
    print("  QUANTIZATION-ALIGNMENT DATA AUDIT")
    print("=" * 72)
    print(f"  raw files/models       : {len(raw)}/{len(models)}")
    print("  paired raw responses   : 2,400 HarmBench per thinking mode; "
          "1,200 TruthfulQA; 240 instruction")
    for filename, (done, missing) in judge_counts.items():
        print(f"  {filename:<33}: {done:>4} labels, {missing:>3} missing")
    print(f"  errors                 : {len(errors)}")
    print(f"  warnings               : {len(warnings)}")
    for item in errors:
        print(f"  ERROR: {item}")
    for item in warnings:
        print(f"  WARN : {item}")
    print("=" * 72)

    if errors or (args.strict and warnings):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
