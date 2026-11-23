from __future__ import annotations

import unittest

from scripts.run_sherpa_zipformer import _merge_diagnostic_tsv_fields
from scripts.validate_asr_evaluation import _merge_accounting_diagnostics
from scripts.sanitize_asr_report import sanitize_execution_diagnostics
from scripts.audio_chunking import diagnose_timestamped_chunk_merge


class MergeAccountingTests(unittest.TestCase):
    def row(self, **overrides):
        value = {
            "overlap_merge_method": "timestamp",
            "merge_input_tokens": "5", "merge_keep_tokens": "3",
            "merge_reject_tokens": "2", "merge_fallback_tokens": "0",
            "merge_unaccounted_tokens": "0", "merge_input_token_chars": "5",
            "merge_keep_token_chars": "3", "merge_reject_token_chars": "2",
            "merge_boundary_rejections": '[{"chunk_position":0,"rejected_tokens":1,"rejected_token_chars":1}]',
            "merge_boundary_rejection_detail": '[{"side":"after_seam_from_left_chunk","distance_bucket":"lt_100ms","rejected_tokens":2,"rejected_token_chars":2}]',
            "merge_boundary_rejection_correspondence": '[{"side":"after_seam_from_left_chunk","distance_bucket":"lt_100ms","status":"unmatched_no_candidate","rejected_tokens":2,"rejected_token_chars":2}]',
            "merge_boundary_rejection_candidate_multiplicity": '[{"threshold_ms":100,"side":"after_seam_from_left_chunk","distance_bucket":"lt_100ms","candidate_multiplicity":"0","rejected_tokens":2,"rejected_token_chars":2},{"threshold_ms":250,"side":"after_seam_from_left_chunk","distance_bucket":"lt_100ms","candidate_multiplicity":"0","rejected_tokens":2,"rejected_token_chars":2},{"threshold_ms":500,"side":"after_seam_from_left_chunk","distance_bucket":"lt_100ms","candidate_multiplicity":"0","rejected_tokens":2,"rejected_token_chars":2}]',
            "merge_boundary_rejection_sensitivity": '[{"threshold_ms":100,"side":"after_seam_from_left_chunk","distance_bucket":"lt_100ms","status":"unmatched_no_candidate","rejected_tokens":2,"rejected_token_chars":2},{"threshold_ms":250,"side":"after_seam_from_left_chunk","distance_bucket":"lt_100ms","status":"unmatched_no_candidate","rejected_tokens":2,"rejected_token_chars":2},{"threshold_ms":500,"side":"after_seam_from_left_chunk","distance_bucket":"lt_100ms","status":"unmatched_no_candidate","rejected_tokens":2,"rejected_token_chars":2}]',
        }
        value.update(overrides)
        return value

    def test_runner_writes_all_token_character_accounting_columns(self):
        fields = _merge_diagnostic_tsv_fields({
            "input_tokens": 7, "keep_tokens": 4, "reject_tokens": 3,
            "fallback_tokens": 0, "unaccounted_tokens": 0,
            "input_token_chars": 12, "keep_token_chars": 7,
            "reject_token_chars": 5, "boundary_rejections": [], "boundary_rejection_detail": [], "boundary_rejection_correspondence": [], "boundary_rejection_sensitivity": [], "boundary_rejection_candidate_multiplicity": [], "boundary_rejection_candidate_multiplicity": [],
        })
        self.assertEqual(fields["merge_input_token_chars"], 12)
        self.assertEqual(fields["merge_keep_token_chars"], 7)
        self.assertEqual(fields["merge_reject_token_chars"], 5)
        self.assertEqual(fields["merge_boundary_rejections"], "[]")
        self.assertEqual(fields["merge_boundary_rejection_detail"], "[]")
        self.assertEqual(fields["merge_boundary_rejection_correspondence"], "[]")
        self.assertEqual(fields["merge_boundary_rejection_candidate_multiplicity"], "[]")

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
                       merge_boundary_rejections="[]", merge_boundary_rejection_detail="[]", merge_boundary_rejection_correspondence="[]", merge_boundary_rejection_sensitivity="[]", merge_boundary_rejection_candidate_multiplicity="[]")
        result = _merge_accounting_diagnostics([row])["merge_accounting"]
        self.assertEqual(result["fallback_tokens"], 5)
        self.assertEqual(result["keep_tokens"] + result["reject_tokens"], 0)

    def test_boundary_rejection_side_and_distance_are_aggregated(self):
        chunks = [
            {"text": "ab", "tokens": ["a", "b"], "timestamps": [0.8, 1.7]},
            {"text": "cd", "tokens": ["c", "d"], "timestamps": [0.1, 0.9]},
        ]
        _, method, diag = diagnose_timestamped_chunk_merge(chunks, [0.0, 1.0], [2.0, 2.0])
        self.assertEqual(method, "timestamp")
        detail = {(item["side"], item["distance_bucket"]): item for item in diag["boundary_rejection_detail"]}
        self.assertEqual(detail[("after_seam_from_left_chunk", "100_250ms")]["rejected_tokens"], 1)
        self.assertEqual(detail[("before_seam_from_right_chunk", "250_500ms")]["rejected_tokens"], 1)


    def test_reject_correspondence_matches_adjacent_kept_token_one_to_one(self):
        chunks = [
            {"text": "aX", "tokens": ["a", "X"], "timestamps": [0.5, 1.7]},
            {"text": "Xb", "tokens": ["X", "b"], "timestamps": [0.7, 1.5]},
        ]
        _, method, diag = diagnose_timestamped_chunk_merge(chunks, [0.0, 1.0], [2.0, 2.0])
        self.assertEqual(method, "timestamp")
        rows = {(x["side"], x["distance_bucket"], x["status"]): x
                for x in diag["boundary_rejection_correspondence"]}
        self.assertEqual(rows[("after_seam_from_left_chunk", "100_250ms", "matched_unique")]["rejected_tokens"], 1)
        self.assertEqual(sum(x["rejected_tokens"] for x in diag["boundary_rejection_correspondence"]), diag["reject_tokens"])


    def test_candidate_multiplicity_is_counted_before_one_to_one_assignment(self):
        chunks = [
            {"text": "XX", "tokens": ["X", "X"], "timestamps": [1.60, 1.70]},
            {"text": "XXXb", "tokens": ["X", "X", "X", "b"], "timestamps": [0.62, 0.94, 1.50, 1.80]},
        ]
        _, method, diag = diagnose_timestamped_chunk_merge(chunks, [0.0, 1.0], [2.0, 2.0])
        self.assertEqual(method, "timestamp")
        rows = {(x["threshold_ms"], x["candidate_multiplicity"]): x
                for x in diag["boundary_rejection_candidate_multiplicity"]
                if x["side"] == "after_seam_from_left_chunk"
                and x["distance_bucket"] == "100_250ms"}
        self.assertEqual(rows[(100, "1")]["rejected_tokens"], 2)
        self.assertEqual(rows[(250, "1")]["rejected_tokens"], 1)
        self.assertEqual(rows[(250, "2_plus")]["rejected_tokens"], 1)
        self.assertEqual(rows[(500, "2_plus")]["rejected_tokens"], 2)
        for threshold in (100, 250, 500):
            self.assertEqual(
                sum(x["rejected_tokens"] for x in diag["boundary_rejection_candidate_multiplicity"]
                    if x["threshold_ms"] == threshold),
                diag["reject_tokens"],
            )

    def test_candidate_multiplicity_includes_zero_candidate_bucket(self):
        chunks = [
            {"text": "Y", "tokens": ["Y"], "timestamps": [1.8]},
            {"text": "XXb", "tokens": ["X", "X", "b"], "timestamps": [0.6, 0.8, 1.5]},
        ]
        _, method, diag = diagnose_timestamped_chunk_merge(chunks, [0.0, 1.0], [2.0, 2.0])
        self.assertEqual(method, "timestamp")
        zero = [x for x in diag["boundary_rejection_candidate_multiplicity"]
                if x["threshold_ms"] == 500 and x["candidate_multiplicity"] == "0"]
        self.assertEqual(sum(x["rejected_tokens"] for x in zero), 1)

    def test_correspondence_separates_assignment_conflict_from_no_candidate(self):
        chunks = [
            {"text": "XX", "tokens": ["X", "X"], "timestamps": [1.60, 1.70]},
            {"text": "Xb", "tokens": ["X", "b"], "timestamps": [0.65, 1.50]},
        ]
        _, method, diag = diagnose_timestamped_chunk_merge(chunks, [0.0, 1.0], [2.0, 2.0])
        self.assertEqual(method, "timestamp")
        counts = {x["status"]: x["rejected_tokens"] for x in diag["boundary_rejection_correspondence"]}
        self.assertEqual(counts["matched_unique"], 1)
        self.assertEqual(counts["unmatched_conflict"], 1)
        self.assertNotIn("unmatched_no_candidate", counts)

    def test_correspondence_rejects_same_token_outside_250ms_window(self):
        chunks = [
            {"text": "aX", "tokens": ["a", "X"], "timestamps": [0.5, 1.70]},
            {"text": "Xb", "tokens": ["X", "b"], "timestamps": [1.00, 1.50]},
        ]
        _, method, diag = diagnose_timestamped_chunk_merge(chunks, [0.0, 1.0], [2.0, 2.0])
        self.assertEqual(method, "timestamp")
        statuses = {x["status"]: x["rejected_tokens"] for x in diag["boundary_rejection_correspondence"]}
        self.assertEqual(statuses["unmatched_no_candidate"], 1)
        self.assertNotIn("matched_unique", statuses)


    def test_correspondence_marks_multiple_candidates_as_ambiguous(self):
        chunks = [
            {"text": "aX", "tokens": ["a", "X"], "timestamps": [0.5, 1.70]},
            {"text": "XXb", "tokens": ["X", "X", "b"], "timestamps": [0.65, 0.75, 1.50]},
        ]
        _, method, diag = diagnose_timestamped_chunk_merge(chunks, [0.0, 1.0], [2.0, 2.0])
        self.assertEqual(method, "timestamp")
        counts = {x["status"]: x["rejected_tokens"] for x in diag["boundary_rejection_correspondence"]}
        self.assertEqual(counts["matched_ambiguous"], 1)

    def test_correspondence_augmenting_path_maximizes_one_to_one_matches(self):
        chunks = [
            {"text": "XX", "tokens": ["X", "X"], "timestamps": [1.60, 1.70]},
            {"text": "XXXb", "tokens": ["X", "X", "X", "b"], "timestamps": [0.62, 0.94, 1.50, 1.80]},
        ]
        _, method, diag = diagnose_timestamped_chunk_merge(chunks, [0.0, 1.0], [2.0, 2.0])
        self.assertEqual(method, "timestamp")
        counts = {x["status"]: x["rejected_tokens"] for x in diag["boundary_rejection_correspondence"]}
        self.assertEqual(counts["matched_unique"], 1)
        self.assertEqual(counts["matched_ambiguous"], 1)
        self.assertNotIn("unmatched_conflict", counts)

    def test_correspondence_aggregation_reconciles_and_rejects_mismatch(self):
        row = self.row()
        result = _merge_accounting_diagnostics([row])["merge_accounting"]
        key = "after_seam_from_left_chunk|lt_100ms|unmatched_no_candidate"
        self.assertEqual(result["boundary_rejection_correspondence"][key]["rejected_tokens"], 2)
        with self.assertRaisesRegex(ValueError, "does not reconcile"):
            _merge_accounting_diagnostics([self.row(merge_boundary_rejection_correspondence="[]")])


    def test_candidate_multiplicity_aggregates_and_reconciles_each_threshold(self):
        row = self.row()
        result = _merge_accounting_diagnostics([row])["merge_accounting"]
        key = "100|after_seam_from_left_chunk|lt_100ms|0"
        self.assertEqual(result["boundary_rejection_candidate_multiplicity"][key]["rejected_tokens"], 2)
        with self.assertRaisesRegex(ValueError, "candidate multiplicity does not reconcile"):
            _merge_accounting_diagnostics([self.row(merge_boundary_rejection_candidate_multiplicity="[]")])

    def test_candidate_multiplicity_rejects_unknown_bucket(self):
        raw = '[{"threshold_ms":100,"side":"after_seam_from_left_chunk","distance_bucket":"lt_100ms","candidate_multiplicity":"many","rejected_tokens":2,"rejected_token_chars":2}]'
        with self.assertRaisesRegex(ValueError, "bucket is invalid"):
            _merge_accounting_diagnostics([self.row(merge_boundary_rejection_candidate_multiplicity=raw)])

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
                "boundary_rejections_by_side_and_distance": {
                    "after_seam_from_left_chunk|lt_100ms": {"rejected_tokens": 1, "rejected_token_chars": 1, "token_text": "PRIVATE_TOKEN_SENTINEL", "case_id": "PRIVATE_CASE_SENTINEL"}},
                "boundary_rejection_correspondence": {
                    "after_seam_from_left_chunk|lt_100ms|matched_unique": {"rejected_tokens": 1, "rejected_token_chars": 1}},
                "boundary_rejection_candidate_multiplicity": {
                    "100|after_seam_from_left_chunk|lt_100ms|0": {"rejected_tokens": 1, "rejected_token_chars": 1, "token_text": "PRIVATE_TOKEN_SENTINEL", "case_id": "PRIVATE_CASE_SENTINEL"}},
                "private_case_id": "MUST_NOT_LEAK"}}}}
        clean = sanitize_execution_diagnostics(source)
        rendered = str(clean)
        self.assertIn("merge_accounting", rendered)
        self.assertNotIn("MUST_NOT_LEAK", rendered)
        self.assertNotIn("PRIVATE_TOKEN_SENTINEL", rendered)
        self.assertNotIn("PRIVATE_CASE_SENTINEL", rendered)
        self.assertIn("boundary_rejections_by_side_and_distance", rendered)
        self.assertIn("after_seam_from_left_chunk|lt_100ms", rendered)
        self.assertIn("boundary_rejection_correspondence", rendered)
        self.assertIn("matched_unique", rendered)
        self.assertIn("boundary_rejection_candidate_multiplicity", rendered)
        self.assertNotIn("token_text", rendered)


if __name__ == "__main__":
    unittest.main()
