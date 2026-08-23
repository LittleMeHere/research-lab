#!/usr/bin/env python3
"""Prepare and verify the judge pipeline on the exploratory responses."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys

import judge_rescore as judge
import v2_experiment as v2
from confirmation_spec import (
    MODEL_REVISIONS,
    PRIMARY_JUDGES,
    PROTOCOL_VERSION,
    TARGET_CONFIGS,
    TARGET_SUITES,
    TRUTHFULQA_REVISION,
    canonical_hash,
    protocol_manifest,
)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
DEFAULT_PREFLIGHT_DIR = os.path.join(DATA_DIR, "confirmation-judge-preflight")
DEFAULT_LOCK = os.path.join(DATA_DIR, "confirmation_judge_lock.json")
DEFAULT_ENVIRONMENT_LOCK = os.path.join(DATA_DIR, "confirmation_environment.json")


def load(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def file_hash(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def prepare(source_dir: str, output_dir: str) -> None:
    if glob.glob(os.path.join(output_dir, "*.json")):
        raise RuntimeError(f"Preflight directory is not empty: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    sources = {}
    for path in glob.glob(os.path.join(source_dir, "v2_results_*.json")):
        data = load(path)
        model = data.get("model_id")
        if model in TARGET_CONFIGS:
            if model in sources:
                raise RuntimeError(f"Duplicate exploratory result for {model}")
            sources[model] = data
    if set(sources) != set(TARGET_CONFIGS):
        raise RuntimeError(
            f"Missing exploratory models: {sorted(set(TARGET_CONFIGS) - set(sources))}"
        )

    v2.TRUTHFULQA_REVISION = TRUTHFULQA_REVISION
    truth = v2.load_truthfulqa(1_000_000)
    gold = {
        " ".join(row["prompt"].split()).casefold(): row for row in truth
    }
    for model, source in sources.items():
        levels = {}
        for quant in TARGET_CONFIGS[model]:
            source_level = source["quant_levels"][quant]
            level = {"quant": quant}
            if "harmful" in TARGET_SUITES[model]:
                refusal_key = (
                    "refusal_thinking=False" if "refusal_thinking=False" in source_level
                    else "refusal_default"
                )
                level[refusal_key] = source_level[refusal_key]
            if "capability" in TARGET_SUITES[model]:
                factual = json.loads(json.dumps(source_level["factual"]))
                for row in factual["results"]:
                    key = " ".join(row["prompt"].split()).casefold()
                    reference = gold.get(key)
                    if not reference:
                        raise RuntimeError(f"Pinned TruthfulQA key missing: {row['prompt']}")
                    row["gold_correct"] = reference["gold_correct"]
                    row["gold_incorrect"] = reference["gold_incorrect"]
                level["factual"] = factual
            levels[quant] = level
        prepared = {
            "model_id": model,
            "model_revision_requested": MODEL_REVISIONS[model],
            "preflight_source": "exploratory-v2 responses; no held-out prompts",
            "quant_levels": levels,
        }
        slug = model.replace("/", "_").replace("-", "_")
        judge.atomic_json_dump(
            prepared,
            os.path.join(output_dir, f"v2_results_{slug}_judge-preflight.json"),
            indent=2,
        )
    print(f"Prepared judge preflight responses in {output_dir}")


def verify(results_dir: str, environment_lock_path: str, output_path: str) -> None:
    manifest = protocol_manifest(PROJECT_DIR)
    if manifest["git_dirty"] is not False:
        raise RuntimeError("Commit the reviewed confirmation package before verification")
    environment = load(environment_lock_path)
    if environment.get("protocol_manifest_sha256") != manifest["files_sha256"]:
        raise RuntimeError("Environment lock does not match the confirmation package")
    report = {}
    artifacts = {}
    for task, value_key, item_loader in (
        ("capability", "correct", judge.capability_items),
        ("refusal", "harmful", judge.refusal_items),
    ):
        items = item_loader(results_dir, stable_ids=True)
        expected = {row["gid"] for row in items}
        a_path = os.path.join(results_dir, f"judge_{task}_results.json")
        b_path = os.path.join(results_dir, f"judge_{task}_results_codex.json")
        resolved_path = os.path.join(results_dir, f"judge_{task}_adjudicated.json")
        a, b, resolved = load(a_path), load(b_path), load(resolved_path)
        if set(a) != expected or set(b) != expected or set(resolved) != expected:
            raise RuntimeError(f"{task}: judge or adjudicated labels are incomplete")
        pairs = {
            (next(iter(labels.values()))["judge_backend"],
             next(iter(labels.values()))["judge_model"])
            for labels in (a, b)
        }
        if pairs != set(PRIMARY_JUDGES.items()):
            raise RuntimeError(f"{task}: primary judge provenance differs from the specification")
        by_gid = {row["gid"]: row for row in items}
        for gid, row in resolved.items():
            provenance_pairs = {
                (entry.get("judge_backend"), entry.get("judge_model"))
                for entry in row.get("judge_provenance", [])
            }
            if (row.get("input_sha256") != by_gid[gid]["input_sha256"] or
                    type(row.get(value_key)) is not bool or
                    provenance_pairs != set(PRIMARY_JUDGES.items()) or
                    row.get("resolution_method") not in {
                        "two-judge-agreement", "blind-resolution",
                    } or
                    (row.get("resolution_method") == "blind-resolution" and
                     not row.get("resolution_provenance"))):
                raise RuntimeError(f"{task}: invalid adjudicated provenance for {gid}")
        agreement = sum(a[gid][value_key] == b[gid][value_key] for gid in expected)
        report[task] = {
            "responses": len(expected),
            "judge_a_coverage": len(a) / len(expected),
            "judge_b_coverage": len(b) / len(expected),
            "agreement_fraction": agreement / len(expected),
            "resolved_coverage": len(resolved) / len(expected),
        }
        for path in (a_path, b_path, resolved_path):
            artifacts[os.path.relpath(path, results_dir)] = file_hash(path)
    lock = {
        "status": "passed",
        "protocol_version": PROTOCOL_VERSION,
        "protocol_manifest_sha256": manifest["files_sha256"],
        "environment_lock_sha256": canonical_hash(environment),
        "primary_judges": PRIMARY_JUDGES,
        "report": report,
        "artifact_sha256": artifacts,
    }
    judge.atomic_json_dump(lock, output_path, indent=2)
    print(f"Judge preflight passed; lock written to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "verify"))
    parser.add_argument("--source-dir", default=DATA_DIR)
    parser.add_argument("--results-dir", default=DEFAULT_PREFLIGHT_DIR)
    parser.add_argument("--environment-lock", default=DEFAULT_ENVIRONMENT_LOCK)
    parser.add_argument("--output", default=DEFAULT_LOCK)
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare(os.path.abspath(args.source_dir), os.path.abspath(args.results_dir))
    else:
        verify(
            os.path.abspath(args.results_dir),
            os.path.abspath(args.environment_lock),
            os.path.abspath(args.output),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
