from __future__ import annotations

import unittest

from scripts.run_sherpa_zipformer import _merge_diagnostic_tsv_fields
from scripts.validate_asr_evaluation import _merge_accounting_diagnostics
from scripts.sanitize_asr_report import sanitize_execution_diagnostics


class MergeAccountingTests(unittest.TestCase):
    def row(self, **overrides):
        value = {
            "overlap_merge_method": "timestamp",
            "merge_input_tokens": "5", "merge_keep_tokens": "3",
            "merge_reject_tokens": "2", "merge_fallback_tokens": "0",
            "merge_unaccounted_tokens": "0", "merge_input_token_chars": "5",
            "merge_keep_token_chars": "3", "merge_reject_token_chars": "2",
            "merge_boundary_rejections": '[{"chunk_position":0,"rejected_tokens":1,"rejected_token_chars":1}]',
        }
        value.update(overrides)
        return value

    def test_runner_writes_all_token_character_accounting_columns(self):
        fields = _merge_diagnostic_tsv_fields({
            "input_tokens": 7, "keep_tokens": 4, "reject_tokens": 3,
            "fallback_tokens": 0, "unaccounted_tokens": 0,
            "input_token_chars": 12, "keep_token_chars": 7,
            "reject_token_chars": 5, "boundary_rejections": [],
        })
        self.assertEqual(fields["merge_input_token_chars"], 12)
        self.assertEqual(fields["merge_keep_token_chars"], 7)
        self.assertEqual(fields["merge_reject_token_chars"], 5)
        self.assertEqual(fields["merge_boundary_rejections"], "[]")

    def test_timestamp_accounting_and_boundary_aggregation(self):
        result = _merge_accounting_diagnostics([self.row(), self.row()])
        accounting = result["merge_accounting"]
        self.assertEqual(accounting["input_tokens"], 10)
        self.assertEqual(accounting["keep_tokens"], 6)
        self.assertEqual(accounting["reject_tokens"], 4)
        self.assertEqual(accounting["unaccounted_tokens"], 0)
        self.assertTrue(accounting["accounting_consistent"])
        self.assertEqual(accounting["boundary_rejections_by_chunk"]["chunk_0"]["rejected_tokens"], 2)

    def test_rejects_unaccounted_tokens(self):
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            _merge_accounting_diagnostics([self.row(merge_unaccounted_tokens="1")])

    def test_fallback_is_separate_from_keep_and_reject(self):
        row = self.row(overlap_merge_method="text_fallback", merge_keep_tokens="0",
                       merge_reject_tokens="0", merge_fallback_tokens="5",
                       merge_unaccounted_tokens="0", merge_input_token_chars="5",
                       merge_keep_token_chars="0", merge_reject_token_chars="0",
                       merge_boundary_rejections="[]")
        result = _merge_accounting_diagnostics([row])["merge_accounting"]
        self.assertEqual(result["fallback_tokens"], 5)
        self.assertEqual(result["keep_tokens"] + result["reject_tokens"], 0)

    def test_sanitizer_keeps_only_aggregate_accounting(self):
        source = {"modes": {"overlap_2s": {
            "audio_duration_seconds": {"min": 1.0, "median": 2.0, "max": 3.0},
            "chunking": {"single_chunk_cases": 0, "multi_chunk_cases": 1,
                         "multi_chunk_case_rate": 1.0, "total_chunks": 2,
                         "max_chunks_per_case": 2, "short_chunks_lt_25ms": 0,
                         "silent_chunks": 0, "empty_output_chunks": 0,
                         "empty_output_chunk_rate": 0.0},
            "merge_accounting": {"input_tokens": 5, "keep_tokens": 3,
                "reject_tokens": 2, "fallback_tokens": 0, "input_token_chars": 5,
                "keep_token_chars": 3, "reject_token_chars": 2, "timestamp_cases": 1,
                "fallback_cases": 0, "unaccounted_tokens": 0,
                "accounting_consistent": True,
                "boundary_rejections_by_chunk": {"chunk_0": {"rejected_tokens": 2, "rejected_token_chars": 2}},
                "private_case_id": "MUST_NOT_LEAK"}}}}
        clean = sanitize_execution_diagnostics(source)
        rendered = str(clean)
        self.assertIn("merge_accounting", rendered)
        self.assertNotIn("MUST_NOT_LEAK", rendered)


if __name__ == "__main__":
    unittest.main()
