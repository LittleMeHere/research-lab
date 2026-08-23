#!/usr/bin/env python3
"""Unit checks for confirmation constants, hashes, and statistics."""

import unittest

import adjudicate_labels
import confirmation_analysis as analysis
import confirmation_spec as spec


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

    def test_primary_judges_use_subscription_clis(self):
        self.assertEqual(set(spec.PRIMARY_JUDGES), {"claude", "codex"})
        self.assertEqual(len(set(spec.PRIMARY_JUDGES.values())), 2)


if __name__ == "__main__":
    unittest.main()
