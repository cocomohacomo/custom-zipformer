from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.validate_asr_evaluation import MODES, _stage_diagnostics, main, validate_evaluation


class ValidateAsrEvaluationTests(unittest.TestCase):
    def _write_fixture(self, root: Path):
        ids = ["PRIVATE_CASE_A", "PRIVATE_CASE_B"]
        (root / "manifest.json").write_text(json.dumps({
            "manifest": [{"case_id": value} for value in ids]
        }), encoding="utf-8")
        for mode, (transcript_name, metrics_name) in MODES.items():
            with (root / transcript_name).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "case_id", "reference", "hypothesis", "audio_duration_seconds", "inference_chunks",
                    "inference_empty_chunks", "inference_short_chunks_lt_25ms", "inference_silent_chunks",
                    "chunk_boundaries_seconds", "chunk_start_offsets_seconds", "overlap_merge_method",
                    "raw_token_count", "timestamp_count", "raw_decoder_text_chars", "chunk_text_chars",
                    "merged_text_chars", "final_text_chars", "chunk_merge_char_delta",
                    "final_normalization_char_delta", "token_density_per_second", "decode_seconds",
                    "merge_input_tokens", "merge_keep_tokens", "merge_reject_tokens",
                    "merge_fallback_tokens", "merge_unaccounted_tokens",
                    "merge_input_token_chars", "merge_keep_token_chars", "merge_reject_token_chars",
                    "merge_boundary_rejections", "merge_boundary_rejection_detail", "merge_boundary_rejection_correspondence",
                ], delimiter="\t")
                writer.writeheader()
                for index, case_id in enumerate(ids):
                    writer.writerow({
                        "case_id": case_id,
                        "reference": "これは同一の参照文です",
                        "hypothesis": "" if index == 0 else "認識結果",
                        "audio_duration_seconds": "30.0",
                        "inference_chunks": "1" if mode == "whole" else "2",
                        "inference_empty_chunks": "1" if index == 0 else "0",
                        "inference_short_chunks_lt_25ms": "0",
                        "inference_silent_chunks": "0",
                        "chunk_boundaries_seconds": "15.0" if mode == "silence_aware_15s" else "",
                        "chunk_start_offsets_seconds": "0.0,13.0" if mode == "overlap_2s" else "0.0",
                        "overlap_merge_method": "timestamp",
                        "raw_token_count": "10", "timestamp_count": "10",
                        "raw_decoder_text_chars": "14", "chunk_text_chars": "12",
                        "merged_text_chars": "11", "final_text_chars": "10",
                        "chunk_merge_char_delta": "-1", "final_normalization_char_delta": "-1",
                        "token_density_per_second": "2", "decode_seconds": "0.1",
                        "merge_input_tokens": "10", "merge_keep_tokens": "8",
                        "merge_reject_tokens": "2", "merge_fallback_tokens": "0",
                        "merge_unaccounted_tokens": "0", "merge_input_token_chars": "10",
                        "merge_keep_token_chars": "8", "merge_reject_token_chars": "2",
                        "merge_boundary_rejections": '[{"chunk_position":0,"rejected_tokens":2,"rejected_token_chars":2}]',
                        "merge_boundary_rejection_detail": '[{"side":"after_seam_from_left_chunk","distance_bucket":"lt_100ms","rejected_tokens":2,"rejected_token_chars":2}]',
                        "merge_boundary_rejection_correspondence": '[{"side":"after_seam_from_left_chunk","distance_bucket":"lt_100ms","status":"unmatched","rejected_tokens":2,"rejected_token_chars":2}]',
                    })
            report = {
                "case_count": 2,
                "summary": {"char": {"cer": 0.25, "deletion_rate": 0.1}},
                "complete_omission": {"empty_hypothesis_count": 1},
                "near_complete_omission": {"count": 0},
                "severe_deletion": {"count": 1},
                "cases": [{"case_id": case_id} for case_id in ids],
            }
            (root / metrics_name).write_text(json.dumps(report), encoding="utf-8")

    def test_validates_all_modes_and_counts_empty_hypotheses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root)
            result = validate_evaluation(root)
            self.assertEqual(result["case_count"], 2)
            self.assertEqual(set(result["modes"]), set(MODES))
            for mode in MODES:
                self.assertEqual(result["modes"][mode]["empty_hypotheses"], 1)
                self.assertEqual(result["modes"][mode]["complete_omissions"], 1)
            self.assertEqual(result["modes"]["overlap_2s"]["overlap_merge_methods"], {"timestamp": 2})
            self.assertEqual(result["modes"]["whole"]["chunking"]["multi_chunk_cases"], 0)
            self.assertEqual(result["modes"]["fixed_15s"]["chunking"]["multi_chunk_cases"], 2)
            self.assertEqual(result["modes"]["silence_aware_15s"]["chunking"]["total_chunks"], 4)
            self.assertEqual(result["modes"]["whole"]["chunking"]["empty_output_chunks"], 1)
            self.assertEqual(result["modes"]["whole"]["chunking"]["empty_output_chunk_rate"], 0.5)



    def test_stage_diagnostics_aggregates_separated_stage_loss(self):
        rows = [
            {"raw_token_count": "8", "timestamp_count": "8",
             "raw_decoder_text_chars": "14", "chunk_text_chars": "12",
             "merged_text_chars": "11", "final_text_chars": "10",
             "chunk_merge_char_delta": "-1", "final_normalization_char_delta": "-1",
             "token_density_per_second": "4.0", "decode_seconds": "0.2"},
            {"raw_token_count": "4", "timestamp_count": "3",
             "raw_decoder_text_chars": "6", "chunk_text_chars": "5",
             "merged_text_chars": "5", "final_text_chars": "5",
             "chunk_merge_char_delta": "0", "final_normalization_char_delta": "0",
             "token_density_per_second": "2.0", "decode_seconds": "0.1"},
        ]
        result = _stage_diagnostics(rows)["decoder_stages"]
        self.assertEqual(result["total_raw_tokens"], 12)
        self.assertEqual(result["total_timestamps"], 11)
        self.assertEqual(result["total_raw_decoder_text_chars"], 20)
        self.assertEqual(result["total_chunk_text_chars"], 17)
        self.assertEqual(result["total_merged_text_chars"], 16)
        self.assertEqual(result["total_final_text_chars"], 15)
        self.assertEqual(result["chunk_merge_char_delta"], -1)
        self.assertEqual(result["final_normalization_char_delta"], -1)
        self.assertEqual(result["cases_shorter_after_merge"], 1)
        self.assertEqual(result["cases_shorter_after_normalization"], 1)
        self.assertEqual(result["cases_token_timestamp_count_mismatch"], 1)
        self.assertEqual(result["mean_token_density_per_second"], 3.0)
        self.assertAlmostEqual(result["total_decode_seconds"], 0.3)

    def test_stage_diagnostics_reject_invalid_numeric_values(self):
        rows = [{"raw_token_count": "NaN", "timestamp_count": "1",
                 "raw_decoder_text_chars": "1", "chunk_text_chars": "1",
                 "merged_text_chars": "1", "final_text_chars": "1",
                 "chunk_merge_char_delta": "0", "final_normalization_char_delta": "0",
                 "token_density_per_second": "1", "decode_seconds": "1"}]
        with self.assertRaisesRegex(ValueError, "non-finite"):
            _stage_diagnostics(rows)

    def test_reference_transcripts_must_match_exactly_across_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root)
            path = root / MODES["fixed_15s"][0]
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream, delimiter=chr(9)))
            rows[0]["reference"] = "これは別の参照文です"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=chr(9))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "reference transcripts differ"):
                validate_evaluation(root)

    def test_short_audio_chunking_diagnostics_detects_missing_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root)
            path = root / MODES["fixed_15s"][0]
            with path.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream, delimiter=chr(9)))
            for row in rows:
                row["inference_chunks"] = "1"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter=chr(9))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "fixed chunk count"):
                validate_evaluation(root)

    def test_cli_emits_aggregate_only_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root)
            output = io.StringIO()
            with patch("sys.argv", ["validate_asr_evaluation.py", "--root", str(root)]), contextlib.redirect_stdout(output):
                self.assertEqual(main(), 0)
            rendered = output.getvalue()
            self.assertNotIn("PRIVATE_CASE_A", rendered)
            self.assertNotIn("認識結果", rendered)
            self.assertIn('"empty_hypotheses": 1', rendered)

    def test_mismatched_case_order_fails_without_echoing_identifiers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root)
            path = root / MODES["fixed_15s"][0]
            rows = path.read_text(encoding="utf-8").splitlines()
            rows[1], rows[2] = rows[2], rows[1]
            path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "row count/order"):
                validate_evaluation(root)

    def test_invalid_overlap_offsets_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root)
            path = root / MODES["overlap_2s"][0]
            text = path.read_text(encoding="utf-8").replace("0.0,13.0", "0.0,nan")
            path.write_text(text, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-finite"):
                validate_evaluation(root)

    def test_metrics_case_count_must_match_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root)
            path = root / MODES["whole"][1]
            report = json.loads(path.read_text(encoding="utf-8"))
            report["case_count"] = 999
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "case count"):
                validate_evaluation(root)

    def test_boolean_and_out_of_range_omission_counts_are_rejected(self):
        for invalid in (True, -1, 3):
            with self.subTest(invalid=invalid), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self._write_fixture(root)
                path = root / MODES["whole"][1]
                report = json.loads(path.read_text(encoding="utf-8"))
                report["near_complete_omission"]["count"] = invalid
                path.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "omission metrics"):
                    validate_evaluation(root)

    def test_complete_omission_count_must_match_empty_hypotheses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root)
            path = root / MODES["whole"][1]
            report = json.loads(path.read_text(encoding="utf-8"))
            report["complete_omission"]["empty_hypothesis_count"] = 0
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "counts disagree"):
                validate_evaluation(root)

    def test_non_finite_or_invalid_rates_are_rejected(self):
        for field, invalid in (("cer", float("nan")), ("deletion_rate", 1.1), ("deletion_rate", True)):
            with self.subTest(field=field, invalid=invalid), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self._write_fixture(root)
                path = root / MODES["whole"][1]
                report = json.loads(path.read_text(encoding="utf-8"))
                report["summary"]["char"][field] = invalid
                path.write_text(json.dumps(report), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "rate is invalid"):
                    validate_evaluation(root)

    def test_invalid_overlap_chunk_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root)
            path = root / MODES["overlap_2s"][0]
            content = path.read_text(encoding="utf-8").replace("\t2\t", "\t2.5\t")
            path.write_text(content, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inference chunk count"):
                validate_evaluation(root)


if __name__ == "__main__":
    unittest.main()
