from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.validate_asr_evaluation import MODES, main, validate_evaluation


class ValidateAsrEvaluationTests(unittest.TestCase):
    def _write_fixture(self, root: Path):
        ids = ["PRIVATE_CASE_A", "PRIVATE_CASE_B"]
        (root / "manifest.json").write_text(json.dumps({
            "manifest": [{"case_id": value} for value in ids]
        }), encoding="utf-8")
        for mode, (transcript_name, metrics_name) in MODES.items():
            with (root / transcript_name).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=[
                    "case_id", "hypothesis", "audio_duration_seconds", "inference_chunks",
                    "inference_short_chunks_lt_25ms", "inference_silent_chunks",
                    "chunk_boundaries_seconds", "chunk_start_offsets_seconds", "overlap_merge_method",
                ], delimiter="\t")
                writer.writeheader()
                for index, case_id in enumerate(ids):
                    writer.writerow({
                        "case_id": case_id,
                        "hypothesis": "" if index == 0 else "認識結果",
                        "audio_duration_seconds": "30.0",
                        "inference_chunks": "1" if mode == "whole" else "2",
                        "inference_short_chunks_lt_25ms": "0",
                        "inference_silent_chunks": "0",
                        "chunk_boundaries_seconds": "15.0" if mode == "silence_aware_15s" else "",
                        "chunk_start_offsets_seconds": "0.0,13.0" if mode == "overlap_2s" else "0.0",
                        "overlap_merge_method": "timestamp",
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


    def test_short_audio_chunking_diagnostics_show_no_activation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_fixture(root)
            path = root / MODES["fixed_15s"][0]
            content = path.read_text(encoding="utf-8").replace("\\t30.0\\t", "\\t8.0\\t").replace("\\t2\\t", "\\t1\\t")
            path.write_text(content, encoding="utf-8")
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
            with self.assertRaisesRegex(ValueError, "chunk diagnostics"):
                validate_evaluation(root)


if __name__ == "__main__":
    unittest.main()
