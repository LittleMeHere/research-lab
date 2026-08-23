#!/usr/bin/env python3
"""Validate and analyze the four held-out confirmation tests."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import random
import sys
import tempfile

from confirmation_spec import (
    ALPHA,
    BOOTSTRAP_SEED,
    EXPECTED_HARMBENCH_N,
    EXPECTED_TRUTHFULQA_N,
    GENERATION_SETTINGS,
    HARMBENCH_HELDOUT_SHA256,
    HARMBENCH_REVISION,
    HYPOTHESES,
    JUDGE_PROMPT_VERSION,
    MINIMUM_ADVANCE_DELTA_PP,
    MODEL_REVISIONS,
    N_BOOTSTRAP,
    PRIMARY_JUDGES,
    PROTOCOL_VERSION,
    REQUIRED_LABEL_COVERAGE,
    RESOLVER_MODEL,
    SCHEME_CONTROLS,
    STUDY_PHASE,
    TARGET_CONFIGS,
    TARGET_SUITES,
    TRANSFORMERS_VERSION,
    TRUTHFULQA_HELDOUT_SHA256,
    TRUTHFULQA_REVISION,
    canonical_hash,
    protocol_manifest,
    protocol_manifests_match,
)

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS_DIR = os.path.join(PROJECT_DIR, "data", "confirmation")


def load(path: str):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def file_hash(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def atomic_json_dump(value: dict, path: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    fd, temporary = tempfile.mkstemp(prefix=".analysis-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def exact_mcnemar(base: list[int], other: list[int]) -> tuple[int, int, float]:
    """Two-sided exact binomial McNemar test."""
    if len(base) != len(other) or not base:
        raise ValueError("McNemar inputs must be non-empty paired vectors")
    lost = sum(a == 1 and b == 0 for a, b in zip(base, other))
    gained = sum(a == 0 and b == 1 for a, b in zip(base, other))
    discordant = lost + gained
    if not discordant:
        return lost, gained, 1.0
    tail = min(lost, gained)
    p_value = 2 * sum(
        math.comb(discordant, k) for k in range(tail + 1)
    ) / (2 ** discordant)
    return lost, gained, min(1.0, p_value)


def bootstrap_ci(base: list[int], other: list[int]) -> tuple[float, float]:
    """Paired percentile-bootstrap 95% interval for other minus fp16."""
    if len(base) != len(other) or not base:
        raise ValueError("Bootstrap inputs must be non-empty paired vectors")
    rng = random.Random(BOOTSTRAP_SEED)
    deltas = [b - a for a, b in zip(base, other)]
    n = len(deltas)
    samples = sorted(
        100 * sum(deltas[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(N_BOOTSTRAP)
    )
    return samples[int(0.025 * N_BOOTSTRAP)], samples[int(0.975 * N_BOOTSTRAP)]


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [1.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def evaluation_hash(rows: list[dict], axis: str) -> str:
    payload = []
    for row in rows:
        item = {"prompt": row["prompt"], "category": row.get("category", "")}
        if axis == "capability":
            item.update({
                "gold_correct": row["gold_correct"],
                "gold_incorrect": row["gold_incorrect"],
            })
        payload.append(item)
    return canonical_hash(payload)


def label_input_hash(axis: str, row: dict) -> str:
    if axis == "capability":
        value = {
            "question": row["prompt"], "response": row["response"],
            "gold_correct": row["gold_correct"],
            "gold_incorrect": row["gold_incorrect"],
        }
    else:
        value = {"prompt": row["prompt"], "response": row["response"]}
    return canonical_hash(value)


def refusal_suite(level: dict) -> str:
    if "refusal_thinking=False" in level:
        return "refusal_thinking=False"
    if "refusal_default" in level:
        return "refusal_default"
    raise RuntimeError("Missing thinking-off/default HarmBench suite")


def validate_fingerprint(model: str, quant: str, level: dict) -> None:
    fingerprint = level.get("quantization_fingerprint", {})
    claimed = fingerprint.get("manifest_sha256")
    payload = {key: value for key, value in fingerprint.items() if key != "manifest_sha256"}
    if claimed != canonical_hash(payload):
        raise RuntimeError(f"{model}/{quant}: invalid module fingerprint hash")
    if fingerprint.get("floating_parameter_dtypes") != ["torch.float16"]:
        raise RuntimeError(f"{model}/{quant}: floating parameters are not all fp16")
    if quant == "fp16":
        if fingerprint.get("linear_4bit_count") or fingerprint.get("linear_8bit_count"):
            raise RuntimeError(f"{model}/{quant}: unexpected quantized modules")
        return
    modules = fingerprint.get("linear_4bit_modules", [])
    expected_type = "fp4" if quant == "int4_fp4" else "nf4"
    if (not fingerprint.get("is_loaded_in_4bit") or not modules or
            any(row.get("quant_type") != expected_type for row in modules) or
            any(row.get("compute_dtype") != "torch.float16" for row in modules)):
        raise RuntimeError(f"{model}/{quant}: module-level {expected_type} state is invalid")
    expected_double = quant == "nf4_dq"
    if any(bool(row.get("double_quant")) != expected_double for row in modules):
        raise RuntimeError(f"{model}/{quant}: module-level double-quant state is invalid")


def expected_cells(results_dir: str):
    paths = sorted(glob.glob(os.path.join(results_dir, "v2_results_*_confirmation.json")))
    if not paths:
        raise RuntimeError(f"No confirmation result files found in {results_dir}")
    current_manifest = protocol_manifest(PROJECT_DIR)
    if current_manifest["git_dirty"] is not False:
        raise RuntimeError("The analyzed confirmation package is not a clean Git revision")

    cells, input_hashes, models = {}, {}, set()
    signatures, common_environment, common_lock_hash = {}, None, None
    recorded_manifest_hash = None
    for path in paths:
        data = load(path)
        model = data.get("model_id")
        if model not in TARGET_CONFIGS or model in models:
            raise RuntimeError(f"Unexpected or duplicate confirmation model: {model}")
        models.add(model)
        metadata = data.get("run_metadata", {})
        recorded_manifest = metadata.get("protocol_manifest", {})
        if (metadata.get("protocol_version") != PROTOCOL_VERSION or
                not protocol_manifests_match(recorded_manifest, current_manifest)):
            raise RuntimeError(f"{model}: protocol manifest is stale or uncommitted")
        if recorded_manifest_hash is None:
            recorded_manifest_hash = recorded_manifest["files_sha256"]
        elif recorded_manifest["files_sha256"] != recorded_manifest_hash:
            raise RuntimeError(f"{model}: protocol manifest differs across result files")
        checks = {
            "study_phase": STUDY_PHASE,
            "harmbench_revision": HARMBENCH_REVISION,
            "truthfulqa_revision": TRUTHFULQA_REVISION,
            "target_configs": TARGET_CONFIGS[model],
            "target_suites": TARGET_SUITES[model],
            "harmbench_heldout_n": EXPECTED_HARMBENCH_N,
            "truthfulqa_heldout_n": EXPECTED_TRUTHFULQA_N,
            "harmbench_heldout_sha256": HARMBENCH_HELDOUT_SHA256,
            "truthfulqa_heldout_sha256": TRUTHFULQA_HELDOUT_SHA256,
        }
        for key, expected_value in checks.items():
            if metadata.get(key) != expected_value:
                raise RuntimeError(f"{model}: {key} differs from the specification")
        if metadata.get("model_revisions", {}).get(model) != MODEL_REVISIONS[model]:
            raise RuntimeError(f"{model}: model revision metadata mismatch")
        expected_harm_n = EXPECTED_HARMBENCH_N if "harmful" in TARGET_SUITES[model] else 0
        expected_truth_n = EXPECTED_TRUTHFULQA_N if "capability" in TARGET_SUITES[model] else 0
        if (metadata.get("harmbench_evaluation_n") != expected_harm_n or
                metadata.get("truthfulqa_evaluation_n") != expected_truth_n):
            raise RuntimeError(f"{model}: focused-suite item counts differ")
        lock_hash = metadata.get("environment_lock_sha256")
        if not isinstance(lock_hash, str) or len(lock_hash) != 64:
            raise RuntimeError(f"{model}: missing environment-lock hash")
        if common_lock_hash is None:
            common_lock_hash = lock_hash
        elif lock_hash != common_lock_hash:
            raise RuntimeError(f"{model}: environment lock differs across runs")
        if data.get("model_revision_requested") != MODEL_REVISIONS[model]:
            raise RuntimeError(f"{model}: requested model revision mismatch")
        if (data.get("seed") != GENERATION_SETTINGS["seed"] or
                data.get("thinking_policy") != "off" or
                data.get("diagnostics") != {
                    "capture_logits": False, "capture_activation_norms": False,
                } or data.get("generation_kwargs") != {
                    "max_new_tokens": GENERATION_SETTINGS["max_new_tokens"],
                    "do_sample": GENERATION_SETTINGS["do_sample"],
                }):
            raise RuntimeError(f"{model}: generation settings differ")
        expected_runner_suites = [
            "refusal" if axis == "harmful" else "factual"
            for axis in TARGET_SUITES[model]
        ]
        if data.get("suites_requested") != expected_runner_suites:
            raise RuntimeError(f"{model}: runner suite list differs")
        environment = data.get("runtime_environment")
        if (not isinstance(environment, dict) or
                environment.get("software_versions") != data.get("software_versions") or
                environment.get("software_versions", {}).get("transformers") !=
                TRANSFORMERS_VERSION):
            raise RuntimeError(f"{model}: runtime metadata is invalid")
        if common_environment is None:
            common_environment = environment
        elif environment != common_environment:
            raise RuntimeError(f"{model}: runtime environment differs across model files")

        levels = data.get("quant_levels", {})
        if list(levels) != TARGET_CONFIGS[model]:
            raise RuntimeError(f"{model}: configuration list or order differs")
        four_bit_names = []
        for quant, level in levels.items():
            if level.get("model_revision_loaded") != MODEL_REVISIONS[model]:
                raise RuntimeError(f"{model}/{quant}: loaded revision mismatch")
            validate_fingerprint(model, quant, level)
            if quant != "fp16":
                four_bit_names.append([
                    row["module"] for row in
                    level["quantization_fingerprint"]["linear_4bit_modules"]
                ])
            for axis in TARGET_SUITES[model]:
                suite = "factual" if axis == "capability" else refusal_suite(level)
                rows = level.get(suite, {}).get("results", [])
                n = EXPECTED_TRUTHFULQA_N if axis == "capability" else EXPECTED_HARMBENCH_N
                if ([row.get("idx") for row in rows] != list(range(n)) or
                        level[suite].get("completed") != n):
                    raise RuntimeError(f"{model}/{quant}/{axis}: incomplete ordered results")
                if axis == "capability" and any(
                        not row.get("gold_correct") or not row.get("gold_incorrect")
                        for row in rows):
                    raise RuntimeError(f"{model}/{quant}: TruthfulQA references are incomplete")
                signature = tuple(
                    (row["idx"], row["prompt"], row.get("category", ""),
                     tuple(row.get("gold_correct", [])), tuple(row.get("gold_incorrect", [])))
                    for row in rows
                )
                if axis not in signatures:
                    signatures[axis] = signature
                elif signature != signatures[axis]:
                    raise RuntimeError(f"{model}/{quant}/{axis}: scoring inputs differ")
                expected_hash = (
                    TRUTHFULQA_HELDOUT_SHA256 if axis == "capability"
                    else HARMBENCH_HELDOUT_SHA256
                )
                if evaluation_hash(rows, axis) != expected_hash:
                    raise RuntimeError(f"{model}/{quant}/{axis}: evaluation hash mismatch")
                indices = set(range(n))
                cells[(axis, model, quant)] = indices
                for row in rows:
                    input_hashes[(axis, model, quant, row["idx"])] = label_input_hash(axis, row)
        if len(four_bit_names) > 1 and any(
                names != four_bit_names[0] for names in four_bit_names[1:]):
            raise RuntimeError(f"{model}: 4-bit conditions quantized different modules")
    if models != set(TARGET_CONFIGS):
        raise RuntimeError(f"Missing confirmation models: {sorted(set(TARGET_CONFIGS)-models)}")
    return cells, input_hashes, common_lock_hash, current_manifest


def label_cells(path: str, value_key: str, expected: dict, input_hashes: dict,
                axis: str, adjudicated: bool):
    labels = load(path)
    out = {}
    prefix = "c" if axis == "capability" else "r"
    expected_gids = set()
    for (cell_axis, model, quant), indices in expected.items():
        if cell_axis == axis:
            expected_gids.update(f"{prefix}:{model}:{quant}:{index}" for index in indices)
    if set(labels) != expected_gids:
        raise RuntimeError(
            f"{os.path.basename(path)} is incomplete or stale: "
            f"missing={len(expected_gids-set(labels))}, extra={len(set(labels)-expected_gids)}"
        )
    provenance_set = set()
    for gid, value in labels.items():
        if type(value.get(value_key)) is not bool:
            raise RuntimeError(f"Non-boolean {value_key} label for {gid}")
        model, quant, index = value["model"], value["quant"], value["idx"]
        expected_gid = f"{prefix}:{model}:{quant}:{index}"
        if gid != expected_gid or value.get("input_sha256") != input_hashes.get(
                (axis, model, quant, index)):
            raise RuntimeError(f"Stale label identity or input hash for {gid}")
        if adjudicated:
            provenance = value.get("judge_provenance", [])
            pairs = {(row.get("judge_backend"), row.get("judge_model")) for row in provenance}
            if (pairs != set(PRIMARY_JUDGES.items()) or
                    {row.get("judge_prompt_version") for row in provenance} !=
                    {JUDGE_PROMPT_VERSION} or
                    any(not row.get("judge_instruction_sha256") for row in provenance)):
                raise RuntimeError(f"Adjudicated judge provenance differs for {gid}")
            method = value.get("resolution_method")
            if method not in {"two-judge-agreement", "blind-resolution"}:
                raise RuntimeError(f"Missing resolution method for {gid}")
            if method == "blind-resolution":
                resolver = value.get("resolution_provenance")
                if (not resolver or resolver.get("kind") not in {"model", "human"} or
                        (resolver.get("kind") == "model" and
                         resolver.get("name_or_model") != RESOLVER_MODEL)):
                    raise RuntimeError(f"Missing or invalid resolver provenance for {gid}")
        else:
            provenance_set.add((
                value.get("judge_backend"), value.get("judge_model"),
                value.get("judge_interface_version"), value.get("judge_prompt_version"),
                value.get("judge_instruction_sha256"),
            ))
        cell = out.setdefault((model, quant), {})
        if index in cell:
            raise RuntimeError(f"Duplicate label for {model}/{quant}/{index}")
        cell[index] = int(value[value_key])
    if not adjudicated:
        if len(provenance_set) != 1 or None in next(iter(provenance_set)):
            raise RuntimeError(f"Raw judge provenance is inconsistent in {path}")
        backend, model, _, prompt_version, _ = next(iter(provenance_set))
        if (PRIMARY_JUDGES.get(backend) != model or
                prompt_version != JUDGE_PROMPT_VERSION):
            raise RuntimeError(f"Raw judge identity differs from the specification in {path}")
    return out


def comparison(spec: dict, source: dict, expected: dict, primary: bool) -> dict:
    axis, model = spec["axis"], spec["model"]
    base_q, other_q = spec["base"], spec["other"]
    indices = expected[(axis, model, base_q)]
    if indices != expected[(axis, model, other_q)]:
        raise RuntimeError(f"Paired index sets differ for {spec['id']}")
    base, other = source[(model, base_q)], source[(model, other_q)]
    if set(base) != indices or set(other) != indices:
        raise RuntimeError(f"Labels are incomplete for {spec['id']}")
    ordered = sorted(indices)
    base_vec, other_vec = [base[i] for i in ordered], [other[i] for i in ordered]
    delta = 100 * (sum(other_vec) - sum(base_vec)) / len(ordered)
    lost, gained, p_value = exact_mcnemar(base_vec, other_vec)
    lo, hi = bootstrap_ci(base_vec, other_vec)
    return {
        **spec, "n": len(ordered), "base_rate": sum(base_vec) / len(ordered),
        "other_rate": sum(other_vec) / len(ordered), "delta": delta,
        "ci_95_percentile": [lo, hi], "lost": lost, "gained": gained,
        "discordant": lost + gained, "p_two_sided_exact": p_value,
        "primary": primary,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", default="confirmation_primary_analysis.json")
    args = parser.parse_args()
    results_dir = os.path.abspath(args.results_dir)
    required = {
        "capability": (
            "correct", "judge_capability_adjudicated.json",
            "judge_capability_results.json", "judge_capability_results_codex.json",
        ),
        "harmful": (
            "harmful", "judge_refusal_adjudicated.json",
            "judge_refusal_results.json", "judge_refusal_results_codex.json",
        ),
    }
    missing = [
        os.path.join(results_dir, name)
        for names in required.values() for name in names[1:]
        if not os.path.exists(os.path.join(results_dir, name))
    ]
    if missing:
        print("Analysis requires complete raw and adjudicated label files:")
        for path in missing:
            print(f"  missing: {path}")
        return 2

    expected, input_hashes, lock_hash, manifest = expected_cells(results_dir)
    sources, per_judge = {}, {}
    label_hashes = {}
    for axis, (value_key, adjudicated_name, a_name, b_name) in required.items():
        adjudicated_path = os.path.join(results_dir, adjudicated_name)
        sources[axis] = label_cells(
            adjudicated_path, value_key, expected, input_hashes, axis, True
        )
        label_hashes[adjudicated_name] = file_hash(adjudicated_path)
        for name in (a_name, b_name):
            path = os.path.join(results_dir, name)
            cells = label_cells(path, value_key, expected, input_hashes, axis, False)
            first = next(iter(load(path).values()))
            judge_name = first["judge_model"]
            per_judge[(axis, judge_name)] = cells
            label_hashes[name] = file_hash(path)

    results = [comparison(spec, sources[spec["axis"]], expected, True)
               for spec in HYPOTHESES]
    adjusted = holm_adjust([row["p_two_sided_exact"] for row in results])
    for row, adjusted_p in zip(results, adjusted):
        row["p_holm"] = adjusted_p
        lo, hi = row["ci_95_percentile"]
        statistical = (
            row["delta"] * row["direction"] > 0 and
            (lo > 0 or hi < 0) and adjusted_p < ALPHA
        )
        row["advance"] = statistical and abs(row["delta"]) >= MINIMUM_ADVANCE_DELTA_PP
        row["decision"] = "ADVANCE" if row["advance"] else "STOP"
        row["threshold_sensitivity"] = [
            {"minimum_delta_pp": threshold,
             "would_advance": statistical and abs(row["delta"]) >= threshold}
            for threshold in (4.0, 6.0, 8.0)
        ]
        row["judge_robustness"] = {}
        for judge_name in PRIMARY_JUDGES.values():
            judge_row = comparison(
                row, per_judge[(row["axis"], judge_name)], expected, False
            )
            row["judge_robustness"][judge_name] = {
                key: judge_row[key] for key in (
                    "base_rate", "other_rate", "delta", "lost", "gained",
                    "p_two_sided_exact", "ci_95_percentile",
                )
            }
    scheme_controls = [
        comparison(spec, sources[spec["axis"]], expected, False)
        for spec in SCHEME_CONTROLS
    ]
    for row in scheme_controls:
        row["judge_robustness"] = {}
        for judge_name in PRIMARY_JUDGES.values():
            judge_row = comparison(
                row, per_judge[(row["axis"], judge_name)], expected, False
            )
            row["judge_robustness"][judge_name] = {
                key: judge_row[key] for key in (
                    "base_rate", "other_rate", "delta", "lost", "gained",
                    "p_two_sided_exact", "ci_95_percentile",
                )
            }

    print("=" * 112)
    print("  HELD-OUT CONFIRMATION — FOUR PRIMARY TESTS")
    print("=" * 112)
    print(f"  {'ID':<3} {'description':<47} {'n':>4} {'delta':>7} {'95% CI':>17} "
          f"{'raw p':>8} {'Holm p':>8} {'decision':>9}")
    print("-" * 112)
    for row in results:
        lo, hi = row["ci_95_percentile"]
        print(f"  {row['id']:<3} {row['description']:<47} {row['n']:>4} "
              f"{row['delta']:>+6.1f} [{lo:+.1f},{hi:+.1f}] "
              f"{row['p_two_sided_exact']:>8.4f} {row['p_holm']:>8.4f} "
              f"{row['decision']:>9}")
    print("=" * 112)

    output_path = args.output if os.path.isabs(args.output) else os.path.join(
        results_dir, args.output
    )
    atomic_json_dump({
        "study_phase": STUDY_PHASE,
        "protocol_version": PROTOCOL_VERSION,
        "protocol_manifest": manifest,
        "primary_hypothesis_count": len(HYPOTHESES),
        "holm_family_size": len(HYPOTHESES),
        "test": "two-sided exact McNemar",
        "confidence_interval": "paired percentile bootstrap, 95%",
        "bootstrap_resamples": N_BOOTSTRAP,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "required_label_coverage": REQUIRED_LABEL_COVERAGE,
        "environment_lock_sha256": lock_hash,
        "label_file_sha256": label_hashes,
        "decision_rule": {
            "prespecified_direction": True,
            "holm_p_below": ALPHA,
            "paired_ci_excludes_zero": True,
            "minimum_absolute_delta_pp": MINIMUM_ADVANCE_DELTA_PP,
        },
        "h3_design_limit": (
            "All 294 available held-out HarmBench prompts are used. The exploratory "
            "effect was near the 8-point follow-up threshold, so a null result does "
            "not exclude smaller safety effects; the 4/6/8-point sensitivity table "
            "must be reported."
        ),
        "tests": results,
        "scheme_controls": scheme_controls,
    }, output_path)
    print(f"  Machine-readable results: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
