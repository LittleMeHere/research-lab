#!/usr/bin/env python3
"""Unit checks for confirmation constants, hashes, and statistics."""

import json
import os
import tempfile
import unittest

import adjudicate_labels
import confirmation_analysis as analysis
import confirmation_spec as spec
import judge_rescore as judge


class ConfirmationTests(unittest.TestCase):
    def test_design_has_exactly_four_primary_hypotheses(self):
        self.assertEqual(len(spec.HYPOTHESES), 4)
        self.assertEqual({row["id"] for row in spec.HYPOTHESES}, {"H1", "H2", "H3", "H4"})
        for row in spec.HYPOTHESES:
            self.assertIn(row["axis"], spec.TARGET_SUITES[row["model"]])
            self.assertIn(row["base"], spec.TARGET_CONFIGS[row["model"]])
            self.assertIn(row["other"], spec.TARGET_CONFIGS[row["model"]])

    def test_focused_generation_count(self):
        total = 0
        for model, configs in spec.TARGET_CONFIGS.items():
            n = (spec.EXPECTED_HARMBENCH_N if "harmful" in spec.TARGET_SUITES[model]
                 else spec.EXPECTED_TRUTHFULQA_N)
            total += len(configs) * n
        self.assertEqual(total, 5_957)
        all_suites = sum(
            len(configs) * (spec.EXPECTED_HARMBENCH_N + spec.EXPECTED_TRUTHFULQA_N)
            for configs in spec.TARGET_CONFIGS.values()
        )
        self.assertEqual(all_suites - total, 3_592)

    def test_truthfulqa_hash_includes_references_and_order(self):
        rows = [{
            "prompt": "Question?", "category": "test",
            "gold_correct": ["yes"], "gold_incorrect": ["no"],
        }]
        original = analysis.evaluation_hash(rows, "capability")
        changed = [dict(rows[0], gold_correct=["different"])]
        self.assertNotEqual(original, analysis.evaluation_hash(changed, "capability"))
        two_rows = rows + [dict(rows[0], prompt="Second?")]
        self.assertNotEqual(
            analysis.evaluation_hash(two_rows, "capability"),
            analysis.evaluation_hash(list(reversed(two_rows)), "capability"),
        )

    def test_exact_mcnemar_is_two_sided(self):
        lost, gained, p_value = analysis.exact_mcnemar(
            [1, 1, 1, 0], [0, 0, 0, 0]
        )
        self.assertEqual((lost, gained), (3, 0))
        self.assertEqual(p_value, 0.25)

    def test_holm_adjustment_is_monotone(self):
        self.assertEqual(analysis.holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])

    def test_blind_ids_require_private_salt(self):
        gid = "c:model:fp16:0"
        first = adjudicate_labels.blind_id(gid, b"a" * 32)
        second = adjudicate_labels.blind_id(gid, b"b" * 32)
        self.assertNotEqual(first, second)
        self.assertNotIn("model", first)

    def test_adjudication_order_is_salt_keyed_and_reproducible(self):
        gids = [f"c:model:fp16:{index}" for index in range(20)]
        first = adjudicate_labels.blind_order(gids, b"a" * 32, "capability")
        repeated = adjudicate_labels.blind_order(gids, b"a" * 32, "capability")
        second = adjudicate_labels.blind_order(gids, b"b" * 32, "capability")
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, sorted(gids))
        self.assertNotEqual(first, second)
        self.assertEqual(set(first), set(gids))

    def test_protocol_identity_ignores_only_commit_hash(self):
        base = {
            "protocol_version": spec.PROTOCOL_VERSION,
            "files": {"plan": "abc"},
            "files_sha256": "def",
            "git_commit": "first",
            "git_dirty": False,
        }
        later_commit = dict(base, git_commit="second")
        changed_files = dict(base, files_sha256="changed")
        dirty = dict(base, git_dirty=True)
        self.assertTrue(spec.protocol_manifests_match(base, later_commit))
        self.assertFalse(spec.protocol_manifests_match(base, changed_files))
        self.assertFalse(spec.protocol_manifests_match(base, dirty))

    def test_primary_judges_are_distinct(self):
        self.assertEqual(set(spec.PRIMARY_JUDGES), {"claude", "codex"})
        self.assertEqual(len(set(spec.PRIMARY_JUDGES.values())), 2)

    def test_protocol_version_is_v2(self):
        self.assertEqual(spec.PROTOCOL_VERSION, "confirmation-v2")
        self.assertEqual(spec.RESOLUTION_POLICY, "model-then-human")
        self.assertIn("code/resolve_blind_packet.py", spec.PROTOCOL_FILES)

    def test_provider_refusal_signatures(self):
        self.assertTrue(spec.is_provider_refusal(
            "API Error: Sonnet 5's safeguards flagged this message. Details: `[cyber]`"
        ))
        self.assertTrue(spec.is_provider_refusal(
            "API Error: Opus 5 can't help with this. Start a new session to continue."
        ))
        self.assertFalse(spec.is_provider_refusal("Error: Reached max turns (1)"))
        self.assertFalse(spec.is_provider_refusal('{"labels":[{"id":0,"correct":true}]}'))
        self.assertFalse(spec.is_provider_refusal(""))

    def test_refusal_entries_are_valid_saved_entries_not_labels(self):
        refused = {"refused": True, "refusal_message": "safeguards flagged this message"}
        self.assertTrue(judge.is_refusal_entry(refused))
        self.assertTrue(judge.valid_entry(refused, ["harmful", "verdict"]))
        self.assertFalse(judge.valid_value(refused, "harmful"))
        self.assertFalse(judge.is_refusal_entry({"refused": True}))
        self.assertFalse(judge.is_refusal_entry({"harmful": False}))
        self.assertTrue(judge.valid_entry({"harmful": False, "verdict": "REFUSE"},
                                          ["harmful", "verdict"]))

    def _resolution_file(self, body: dict) -> str:
        handle = tempfile.NamedTemporaryFile(
            "w", suffix=".json", delete=False, encoding="utf-8"
        )
        json.dump(body, handle)
        handle.close()
        self.addCleanup(os.unlink, handle.name)
        return handle.name

    def test_resolution_policy_model_then_human(self):
        model = {"kind": "model", "name_or_model": spec.RESOLVER_MODEL,
                 "backend": "claude-cli", "interface_version": "x", "completed_utc": "t"}
        human = {"kind": "human", "name_or_model": "A. Resolver",
                 "backend": "manual", "interface_version": "n/a", "completed_utc": "t"}
        base = {"packet_sha256": "p", "resolution_policy": spec.RESOLUTION_POLICY}
        expected = {"item-a", "item-b"}
        good = dict(base, resolvers={"model": model, "human": human}, items=[
            {"id": "item-a", "resolution": True, "resolver": "model"},
            {"id": "item-b", "resolution": False, "resolver": "human",
             "model_refusal": "Opus 5's safeguards flagged this message"},
        ])
        resolutions, provenance, _ = adjudicate_labels.load_resolutions(
            self._resolution_file(good), expected, "p"
        )
        self.assertEqual(resolutions, {"item-a": True, "item-b": False})
        self.assertEqual(provenance["item-b"]["kind"], "human")
        self.assertIn("model_refusal", provenance["item-b"])
        self.assertNotIn("model_refusal", provenance["item-a"])

        human_only = dict(base, resolvers={"human": human}, items=[
            {"id": "item-a", "resolution": True, "resolver": "human"},
            {"id": "item-b", "resolution": False, "resolver": "human"},
        ])
        adjudicate_labels.load_resolutions(self._resolution_file(human_only), expected, "p")

        unjustified_human = dict(good)
        unjustified_human["items"] = [
            {"id": "item-a", "resolution": True, "resolver": "model"},
            {"id": "item-b", "resolution": False, "resolver": "human"},
        ]
        with self.assertRaises(RuntimeError):
            adjudicate_labels.load_resolutions(
                self._resolution_file(unjustified_human), expected, "p"
            )
        wrong_model = dict(good, resolvers={"model": dict(model, name_or_model="other"),
                                            "human": human})
        with self.assertRaises(RuntimeError):
            adjudicate_labels.load_resolutions(self._resolution_file(wrong_model), expected, "p")
        partial = dict(good, items=good["items"][:1])
        with self.assertRaises(RuntimeError):
            adjudicate_labels.load_resolutions(self._resolution_file(partial), expected, "p")
        unresolved = dict(good, items=[
            good["items"][0],
            {"id": "item-b", "resolution": None, "resolver": None, "model_refusal": "x"},
        ])
        with self.assertRaises(RuntimeError):
            adjudicate_labels.load_resolutions(self._resolution_file(unresolved), expected, "p")

    def test_comparison_can_exclude_refused_pairs(self):
        expected = {("harmful", "m", "fp16"): {0, 1, 2, 3}, ("harmful", "m", "nf4_dq"): {0, 1, 2, 3}}
        source = {("m", "fp16"): {0: 0, 1: 0, 2: 0, 3: 1}, ("m", "nf4_dq"): {0: 1, 1: 1, 2: 0, 3: 1}}
        test = {"id": "H3", "model": "m", "axis": "harmful", "base": "fp16",
                "other": "nf4_dq", "direction": 1}
        full = analysis.comparison(test, source, expected, True)
        self.assertEqual((full["n"], full["excluded_pairs"], full["gained"]), (4, 0, 2))
        reduced = analysis.comparison(
            test, source, expected, False, exclude={("m", "nf4_dq"): {0}}
        )
        self.assertEqual((reduced["n"], reduced["excluded_pairs"], reduced["gained"]), (3, 1, 1))
        incomplete = {("m", "fp16"): {0: 0, 1: 0, 2: 0}, ("m", "nf4_dq"): source[("m", "nf4_dq")]}
        with self.assertRaises(RuntimeError):
            analysis.comparison(test, incomplete, expected, True)
        analysis.comparison(test, incomplete, expected, False, exclude={("m", "fp16"): {3}})


if __name__ == "__main__":
    unittest.main()
