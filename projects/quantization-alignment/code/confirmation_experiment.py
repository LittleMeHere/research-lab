#!/usr/bin/env python3
"""Held-out confirmation runner for the quantization-alignment study.

This runner deliberately tests only the configurations specified in the reviewed
version of notes/04_confirmation_plan.md. It excludes every HarmBench and TruthfulQA
prompt used by v2, uses greedy decoding, and records hashes of the held-out
prompt universes in each result file.

Run from the project root on the GPU host, for example:

    MODELS="google/gemma-4-e2b-it,Qwen/Qwen3.5-4B" \
      python3 code/confirmation_experiment.py
"""

from __future__ import annotations

import argparse
import gc
import glob
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

import v2_experiment as v2
from confirmation_spec import (
    EXPECTED_HARMBENCH_N,
    EXPECTED_TRUTHFULQA_N,
    HARMBENCH_HELDOUT_SHA256,
    HARMBENCH_REVISION,
    MODEL_REVISIONS,
    PROTOCOL_VERSION,
    STUDY_PHASE,
    TARGET_CONFIGS,
    TARGET_SUITES,
    TRANSFORMERS_VERSION,
    TRUTHFULQA_HELDOUT_SHA256,
    TRUTHFULQA_REVISION,
    canonical_hash,
    protocol_manifest,
)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
DEFAULT_OUTPUT_DIR = os.path.join(DATA_DIR, "confirmation")
DEFAULT_SMOKE_DIR = os.path.join(DATA_DIR, "confirmation-smoke")
DEFAULT_ENVIRONMENT_LOCK = os.path.join(DATA_DIR, "confirmation_environment.json")
DEFAULT_JUDGE_LOCK = os.path.join(DATA_DIR, "confirmation_judge_lock.json")


def object_hash(value: dict) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

def evaluation_hash(items: list[dict], axis: str) -> str:
    """Hash the ordered prompts and every field used to score them."""
    payload = []
    for item in items:
        row = {"prompt": item["prompt"], "category": item.get("category", "")}
        if axis == "capability":
            row.update({
                "gold_correct": item["gold_correct"],
                "gold_incorrect": item["gold_incorrect"],
            })
        payload.append(row)
    return canonical_hash(payload)


def normalize_prompt(text: str) -> str:
    return " ".join(text.split()).casefold()


def original_prompt_sets() -> tuple[set[str], set[str]]:
    """Read the v2 prompt sets from any complete original result file."""
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "v2_results_*.json"))):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            fp16 = data["quant_levels"]["fp16"]
            refusal = fp16.get("refusal_thinking=False")
            if not refusal:
                continue
            harm = {normalize_prompt(x["prompt"]) for x in refusal["results"]}
            truth = {normalize_prompt(x["prompt"]) for x in fp16["factual"]["results"]}
            # The v2 HarmBench sample has 100 rows but 99 unique prompt texts.
            if len(harm) in {99, 100} and len(truth) == 50:
                return harm, truth
        except (OSError, KeyError, json.JSONDecodeError):
            continue
    raise RuntimeError("Could not find a complete v2 result file for prompt exclusion")


def held_out(items: list[dict], used: set[str], name: str,
             expected_count: int) -> list[dict]:
    universe = {normalize_prompt(x["prompt"]) for x in items}
    absent = used - universe
    if absent:
        raise RuntimeError(
            f"{name} source changed: {len(absent)} original prompts are absent; "
            "refusing to create a contaminated or non-comparable split"
        )
    result = []
    seen = set()
    for item in sorted(
            items, key=lambda row: (normalize_prompt(row["prompt"]),
                                    row.get("category", ""))):
        normalized = normalize_prompt(item["prompt"])
        if normalized in used or normalized in seen:
            continue
        seen.add(normalized)
        result.append(item)
    if len(result) != expected_count:
        raise RuntimeError(
            f"Found {len(result)} held-out {name} items; specified design requires "
            f"exactly {expected_count}. Refusing a changed or fallback source."
        )
    return result


def first_previously_used(items: list[dict], used: set[str], name: str) -> dict:
    """Choose a v2-used prompt so a smoke test never consumes held-out data."""
    for item in items:
        if normalize_prompt(item["prompt"]) in used:
            return item
    raise RuntimeError(f"Could not reconstruct a previously used {name} smoke item")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke", action="store_true",
        help="exercise each target config on v2-used prompts in a separate directory",
    )
    args = parser.parse_args()
    requested = [
        x.strip() for x in os.environ.get("MODELS", "").split(",") if x.strip()
    ]
    if not requested:
        print("ERROR: Set MODELS to one or more specified confirmation model IDs:")
        for model in TARGET_CONFIGS:
            print(f"  {model}")
        sys.exit(2)

    unknown = [m for m in requested if m not in TARGET_CONFIGS]
    if unknown:
        print(f"ERROR: Models not in the specified confirmation plan: {unknown}")
        sys.exit(2)
    if len(set(requested)) != len(requested):
        print("ERROR: MODELS contains a duplicate model ID")
        sys.exit(2)
    if not v2.torch.cuda.is_available():
        print("ERROR: Held-out inference requires a CUDA GPU; none was detected")
        sys.exit(2)

    current_environment = v2.runtime_environment()
    current_pip_freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"], check=True,
        capture_output=True, text=True,
    ).stdout.splitlines()
    installed_transformers = current_environment["software_versions"].get("transformers")
    if installed_transformers != TRANSFORMERS_VERSION:
        print(
            f"ERROR: Transformers {installed_transformers!r} is installed; the reviewed "
            f"confirmation environment requires {TRANSFORMERS_VERSION}"
        )
        sys.exit(2)
    lock_path = os.path.abspath(os.environ.get(
        "ENVIRONMENT_LOCK", DEFAULT_ENVIRONMENT_LOCK
    ))
    judge_lock_path = os.path.abspath(os.environ.get("JUDGE_LOCK", DEFAULT_JUDGE_LOCK))
    manifest = protocol_manifest(PROJECT_DIR)
    if not args.smoke and manifest["git_dirty"] is not False:
        print("ERROR: The confirmation package must be committed and unchanged before inference")
        sys.exit(2)
    environment_lock_hash = None
    if not args.smoke:
        if not os.path.exists(lock_path):
            print(
                f"ERROR: Environment lock missing: {lock_path}\n"
                "Run the complete --smoke command successfully before confirmation."
            )
            sys.exit(2)
        with open(lock_path, encoding="utf-8") as f:
            lock = json.load(f)
        if (lock.get("purpose") != "quantization-alignment held-out confirmation" or
                lock.get("validated_models") != TARGET_CONFIGS or
                lock.get("protocol_manifest_sha256") != manifest["files_sha256"]):
            print(f"ERROR: Invalid or incomplete confirmation environment lock: {lock_path}")
            sys.exit(2)
        if lock.get("runtime_environment") != current_environment:
            print("ERROR: Current software/GPU environment differs from the smoke-tested lock")
            print(f"  lock: {lock_path}")
            sys.exit(2)
        if lock.get("pip_freeze") != current_pip_freeze:
            print("ERROR: Installed packages differ from the smoke-tested environment")
            sys.exit(2)
        environment_lock_hash = object_hash(lock)
        if not os.path.exists(judge_lock_path):
            print(
                f"ERROR: Judge dry-run lock missing: {judge_lock_path}\n"
                "Complete both judge passes and adjudication on the v2 data first."
            )
            sys.exit(2)
        with open(judge_lock_path, encoding="utf-8") as handle:
            judge_lock = json.load(handle)
        if (judge_lock.get("protocol_manifest_sha256") != manifest["files_sha256"] or
                judge_lock.get("environment_lock_sha256") != environment_lock_hash or
                judge_lock.get("status") != "passed"):
            print(f"ERROR: Judge dry-run lock is stale or incomplete: {judge_lock_path}")
            sys.exit(2)

    default_output = DEFAULT_SMOKE_DIR if args.smoke else DEFAULT_OUTPUT_DIR
    output_dir = os.path.abspath(os.environ.get("OUTPUT_DIR", default_output))
    if args.smoke and os.path.normcase(output_dir) == os.path.normcase(
            os.path.abspath(DEFAULT_OUTPUT_DIR)):
        print("ERROR: Smoke results may not be written into the confirmation directory")
        sys.exit(2)
    used_harm, used_truth = original_prompt_sets()

    v2.HARMBENCH_REVISION = HARMBENCH_REVISION
    v2.TRUTHFULQA_REVISION = TRUTHFULQA_REVISION

    # Asking for an oversized sample loads the full source. held_out() establishes
    # a stable order independent of Python's sampling implementation.
    all_harm = v2.load_harmbench(1_000_000)
    all_truth = v2.load_truthfulqa(1_000_000)
    harm = held_out(all_harm, used_harm, "HarmBench", EXPECTED_HARMBENCH_N)
    truth = held_out(all_truth, used_truth, "TruthfulQA", EXPECTED_TRUTHFULQA_N)
    harm_heldout_hash = evaluation_hash(harm, "harmful")
    truth_heldout_hash = evaluation_hash(truth, "capability")
    if harm_heldout_hash != HARMBENCH_HELDOUT_SHA256:
        raise RuntimeError("HarmBench held-out prompt hash differs from the specified split")
    if truth_heldout_hash != TRUTHFULQA_HELDOUT_SHA256:
        raise RuntimeError("TruthfulQA held-out prompt hash differs from the specified split")

    missing_gold = [item["prompt"] for item in truth
                    if not item.get("gold_correct") or not item.get("gold_incorrect")]
    if missing_gold:
        raise RuntimeError(
            f"TruthfulQA source omitted official references for {len(missing_gold)} "
            "held-out questions; refusing knowledge-only judge fallback"
        )

    evaluation_harm = harm
    evaluation_truth = truth
    study_phase = STUDY_PHASE
    prompt_split = "unique normalized source prompts excluding the v2 prompt texts"
    if args.smoke:
        evaluation_harm = [first_previously_used(all_harm, used_harm, "HarmBench")]
        evaluation_truth = [first_previously_used(all_truth, used_truth, "TruthfulQA")]
        study_phase = "confirmation-compatibility-smoke"
        prompt_split = "one previously used v2 prompt per benchmark; never infer from this run"

    metadata = {
        "study_phase": study_phase,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_manifest": manifest,
        "plan": "notes/04_confirmation_plan.md",
        "prompt_split": prompt_split,
        "v2_harmbench_excluded": len(used_harm),
        "v2_truthfulqa_excluded": len(used_truth),
        "harmbench_heldout_n": len(harm),
        "truthfulqa_heldout_n": len(truth),
        "harmbench_heldout_sha256": harm_heldout_hash,
        "truthfulqa_heldout_sha256": truth_heldout_hash,
        "harmbench_evaluation_n": len(evaluation_harm),
        "truthfulqa_evaluation_n": len(evaluation_truth),
        "harmbench_evaluation_sha256": evaluation_hash(evaluation_harm, "harmful"),
        "truthfulqa_evaluation_sha256": evaluation_hash(evaluation_truth, "capability"),
        "primary_hypothesis_count": 4,
        "harmbench_revision": HARMBENCH_REVISION,
        "truthfulqa_revision": TRUTHFULQA_REVISION,
        "smoke_test": args.smoke,
        "environment_lock_sha256": environment_lock_hash,
    }

    print("=" * 72)
    title = "COMPATIBILITY SMOKE (V2-USED PROMPTS)" if args.smoke else "HELD-OUT QUANTIZATION CONFIRMATION"
    print(f"  {title}")
    print(f"  HarmBench evaluation items: {len(evaluation_harm)}")
    print(f"  TruthfulQA evaluation items: {len(evaluation_truth)}")
    print(f"  Output: {output_dir}")
    print("=" * 72)

    failures = []
    for model in requested:
        try:
            suites = [
                "refusal" if axis == "harmful" else "factual"
                for axis in TARGET_SUITES[model]
            ]
            model_harm = evaluation_harm if "harmful" in TARGET_SUITES[model] else []
            model_truth = evaluation_truth if "capability" in TARGET_SUITES[model] else []
            outfile = v2.run_single_model(
                model,
                model_harm,
                model_truth,
                quant_names=TARGET_CONFIGS[model],
                thinking_policy="off",
                capture_logits=False,
                capture_norms=False,
                output_dir=output_dir,
                run_metadata={
                    **metadata,
                    "model_revisions": {model: MODEL_REVISIONS[model]},
                    "target_configs": TARGET_CONFIGS[model],
                    "target_suites": TARGET_SUITES[model],
                    "harmbench_evaluation_n": len(model_harm),
                    "truthfulqa_evaluation_n": len(model_truth),
                },
                model_revision=MODEL_REVISIONS[model],
                suites=suites,
                run_name="confirmation-smoke" if args.smoke else "confirmation",
                resume=True,
            )
            if not args.smoke:
                with open(outfile, encoding="utf-8") as handle:
                    completed = json.load(handle)
                expected_fingerprints = lock["quantization_fingerprints"][model]
                actual = {
                    name: level["quantization_fingerprint"]
                    for name, level in completed["quant_levels"].items()
                }
                if actual != expected_fingerprints:
                    raise RuntimeError(f"{model}: quantization module fingerprint differs from smoke")
        except Exception as exc:  # noqa: BLE001 - preserve failure details in the run log
            print(f"\nFAILED on {model}: {exc}")
            import traceback
            traceback.print_exc()
            gc.collect()
            if v2.torch.cuda.is_available():
                v2.torch.cuda.empty_cache()
            failures.append(model)
            continue

    if failures:
        print(f"ERROR: confirmation failed for {len(failures)} model(s): {failures}")
        sys.exit(1)

    if args.smoke:
        if set(requested) != set(TARGET_CONFIGS):
            print(
                "Smoke succeeded for the requested subset, but no environment lock was "
                "written; exercise every target model in one smoke command."
            )
        else:
            fingerprints = {}
            for model in TARGET_CONFIGS:
                slug = model.replace("/", "_").replace("-", "_")
                path = os.path.join(output_dir, f"v2_results_{slug}_confirmation-smoke.json")
                with open(path, encoding="utf-8") as handle:
                    result = json.load(handle)
                fingerprints[model] = {
                    name: level["quantization_fingerprint"]
                    for name, level in result["quant_levels"].items()
                }
            lock = {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "purpose": "quantization-alignment held-out confirmation",
                "validated_models": TARGET_CONFIGS,
                "runtime_environment": current_environment,
                "protocol_manifest_sha256": manifest["files_sha256"],
                "quantization_fingerprints": fingerprints,
                "pip_freeze": current_pip_freeze,
            }
            v2.atomic_json_dump(lock, lock_path)
            print(f"Environment lock written: {lock_path}")


if __name__ == "__main__":
    main()
