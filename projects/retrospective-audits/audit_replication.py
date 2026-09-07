"""Independently check raw responses, fork inputs and test scope; no model calls.

Run after unpacking data/replication-artifacts.tar.gz, from this project directory:
  python3 audit_replication.py
  python3 audit_replication.py --check-tests  # requires pytest

Checks fail on absent scheduled cells, even if a historical analyzer would skip them.
"""
import argparse
import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CONDITIONS = {
    "status": ("audit_first", "withheld", "reveal_first", "audit_first_nodef", "reveal_first_nodef"),
    "belief": ("baseline", "withheld", "reveal_pass", "reveal_fail"),
    "debias": ("withheld", "reveal_fail", "withheld_debias", "reveal_fail_debias"),
}


def read(path):
    return json.loads(path.read_text())


def rollout(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def final_text(records):
    messages = [r["payload"] for r in records if r["type"] == "response_item"
                and r["payload"].get("type") == "message" and r["payload"].get("role") == "assistant"]
    return "".join(c["text"] for c in messages[-1]["content"] if c["type"] == "output_text")


def context(records, followup):
    ctx = [r["payload"] for r in records if r["type"] == "turn_context"][-1].copy()
    ctx.pop("turn_id", None)
    if followup:
        ctx.pop("current_date", None)
    return re.sub(r'/tmp/arg0/codex-arg0[^/"\\]*', '<arg0>', json.dumps(ctx, sort_keys=True))


def audit_phase(run, phase, expected):
    folder = run if phase == "status" else run / phase
    result = read(folder / "result.json")
    assert result.get("harness_pipeline_passed", result["checks"].get("pipeline_passed")), folder
    parent = rollout(run / "parent" / "rollout.jsonl")
    prefix = (run / "fork_point" / "parent_prefix.jsonl").read_bytes()
    assert (run / "parent" / "rollout.jsonl").read_bytes() == prefix, run
    plan_path = folder / "prompts" / "branch_plan.json"
    if plan_path.exists():
        plan = read(plan_path)
    else:
        assert "-pilot-" in run.name, run
        plan = [{"name": name, "condition": r["condition"], "sample": r["sample"]}
                for name, r in result["branch_records"].items()]
    cells = [(r["condition"], r["sample"]) for r in plan]
    assert Counter(cells) == Counter(itertools.product(expected, (1, 2, 3))), folder
    assert set(result["branch_records"]) == {r["name"] for r in plan}, folder
    answers = {}
    for item in plan:
        name, condition, sample = item["name"], item["condition"], item["sample"]
        branch = folder / "branches" / name
        record = result["branch_records"][name]
        assert (record["condition"], record["sample"]) == (condition, sample), branch
        assert not record.get("missing", False), branch
        if "-main-" in run.name:
            assert record["attempts"] == 1, branch
        raw = rollout(branch / "rollout.jsonl")
        answer = json.loads(final_text(raw))
        key = "audit_response" if phase == "status" else "answer"
        assert answer == read(branch / f"{key}.json") == record[key], branch
        if phase != "status":
            assert set(answer) == {"p_pass_then", "explanation"}, branch
            assert type(answer["p_pass_then"]) is int and 0 <= answer["p_pass_then"] <= 100, branch
        else:
            assert set(answer) == {"then_verification_status", "now_verified_status", "explanation"}, branch
            assert all(answer[k] in {"PASS", "FAIL", "UNKNOWN"} for k in answer if k != "explanation"), branch
        assert isinstance(answer["explanation"], str), branch
        assert context(raw, phase != "status") == context(parent, phase != "status"), branch
        base = raw[0]["payload"]["history_base"]
        assert base["thread_id"] == read(run / "result.json")["parent_thread_id"], branch
        assert base["end_byte_offset"] == len(prefix), branch
        assert read(branch / "workspace_before_manifest.json") == read(run / "fork_point" / "workspace_manifest.json"), branch
        before = read(branch / "workspace_before_manifest.json")
        after = read(branch / "workspace_after_manifest.json")
        scientific = lambda entries: [e for e in entries if not
                                     (e["type"] == "directory" and e["path"] in {".agents", ".codex", ".git"})]
        assert scientific(before) == scientific(after), branch
        prompt = (folder / "prompts" / f"{condition}.txt").read_text()
        user_messages = [r["payload"] for r in raw if r["type"] == "response_item"
                         and r["payload"].get("type") == "message" and r["payload"].get("role") == "user"]
        assert any(prompt.strip() in "".join(c.get("text", "") for c in m["content"]) for m in user_messages), branch
        answers[condition, sample] = answer
    return answers


def exact_p(differences, direction):
    # Three-sample means have integer numerators; avoid rounding near tied sums.
    xs = [round(3 * direction * d) for d in differences]
    observed = sum(xs)
    sums = [sum(s * d for s, d in zip(signs, xs)) for signs in itertools.product((-1, 1), repeat=len(xs))]
    return sum(s >= observed for s in sums) / len(sums)


def check_tests(entries):
    for entry in entries:
        run = ROOT / entry["run_dir"]
        with tempfile.TemporaryDirectory(prefix="replication-verifier-") as tmp:
            workspace = Path(tmp) / "workspace"
            shutil.copytree(run / "verifier" / "workspace", workspace)
            for folder in ("usage", "tests"):
                for path in (run / "fork_point" / "workspace" / folder).rglob("*.py"):
                    relative = path.relative_to(run / "fork_point" / "workspace")
                    assert path.read_bytes() == (workspace / relative).read_bytes(), (run, relative)
            env = dict(os.environ, PYTHONPATH=f"{ROOT / 'tasks/_devtools'}{os.pathsep}{workspace}",
                       PYTHONDONTWRITEBYTECODE="1", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
            outcomes = []
            for suites in (("tests",), ("tests", "_hidden_tests")):
                proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *suites],
                                      cwd=workspace, env=env, capture_output=True, text=True, timeout=120)
                summary = proc.stdout.strip().splitlines()[-1:] or [proc.stderr.strip()]
                wanted = 0 if suites == ("tests",) or entry["task"].startswith("t001") else 1
                assert proc.returncode == wanted and "passed" in summary[0] and "error" not in summary[0], (run, suites, proc.stdout, proc.stderr)
                outcomes.append(summary[0])
            print(f"{entry['index']:02d} {entry['task'].split('-')[0]} visible: {outcomes[0]}; augmented: {outcomes[1]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-tests", action="store_true")
    args = parser.parse_args()
    manifest = read(ROOT / "data/artifact-manifest.json")
    archive = ROOT / "data/replication-artifacts.tar.gz"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["archive_sha256"], archive
    for name, expected in manifest["files"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    print(f"Artifact hashes verified: {len(manifest['files'])}")
    entries = read(ROOT / "tasks/schedule-main.json")["sequence"]
    totals = Counter()
    values = {phase: [] for phase in ("belief", "debias")}
    for entry in entries:
        run = ROOT / entry["run_dir"]
        assert read(run / "eligibility.json")["eligible"], run
        assert read(run / "ground_truth/manual_label.json")["then_verification_status"] == "UNKNOWN", run
        for phase, conditions in CONDITIONS.items():
            answers = audit_phase(run, phase, conditions)
            totals[phase] += len(answers)
            if phase != "status":
                values[phase].append({c: sum(answers[c, s]["p_pass_then"] for s in (1, 2, 3)) / 3 for c in conditions})
    # Pilot observations are excluded from all main contrasts; controls are in this schedule.
    for entry in read(ROOT / "tasks/schedule.json")["sequence"]:
        answers = audit_phase(ROOT / entry["run_dir"], "status", CONDITIONS["status"][:3])
        totals["controls" if entry["task"].startswith("c") else "pilot"] += len(answers)
    print("Raw answers matched stored answers/results; complete cells, prompts, contexts, fork offsets and starting manifests:", dict(totals))
    for phase, rows in values.items():
        print(phase, "cell means:", {c: sum(row[c] for row in rows) / len(rows) for c in rows[0]})
    contrasts = {
        "Experiment 2 spread": ([r["reveal_pass"] - r["reveal_fail"] for r in values["belief"]], 1),
        "Experiment 2 FAIL effect": ([r["reveal_fail"] - r["withheld"] for r in values["belief"]], -1),
        "Experiment 3 FAIL effect": ([r["reveal_fail"] - r["withheld"] for r in values["debias"]], -1),
        "Experiment 3 interaction": ([(r["reveal_fail_debias"] - r["withheld_debias"]) - (r["reveal_fail"] - r["withheld"]) for r in values["debias"]], 1),
        "Experiment 3 residual": ([r["reveal_fail_debias"] - r["withheld_debias"] for r in values["debias"]], -1),
    }
    for name, (diffs, direction) in contrasts.items():
        print(f"{name}: {sum(diffs)/len(diffs):+.6f}; exact one-sided sign-flip p={exact_p(diffs, direction):.9f}")
    if args.check_tests:
        check_tests(entries)


if __name__ == "__main__":
    main()
