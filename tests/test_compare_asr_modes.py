from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from scripts.compare_asr_modes import compare_reports, main


def write_report(path: Path, rows: list[dict], cer: float) -> None:
    path.write_text(json.dumps({
        "cases": rows,
        "summary": {"char": {"cer": cer, "deletion_rate": cer / 2}},
    }), encoding="utf-8")


def case(case_id: str, deletions: int) -> dict:
    return {
        "case_id": case_id,
        "category": "short",
        "char": {"deletions": deletions, "reference_units": 20},
        "complete_omission": False,
        "near_complete_omission": False,
        "severe_deletion": deletions >= 5,
    }


class CompareAsrModesTests(unittest.TestCase):
    def test_paired_case_deltas_and_omission_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, candidate = root / "base.json", root / "candidate.json"
            write_report(base, [case("a", 5), case("b", 1), case("c", 2)], 0.3)
            write_report(candidate, [case("a", 2), case("b", 3), case("c", 2)], 0.25)
            result = compare_reports([("whole", base), ("overlap", candidate)])
        comparison = result["paired_comparisons"]["overlap"]
        self.assertEqual(comparison["deletion_delta"], -1)
        self.assertEqual(comparison["cases_with_fewer_deletions"], 1)
        self.assertEqual(comparison["cases_with_more_deletions"], 1)
        self.assertEqual(comparison["cases_unchanged"], 1)
        self.assertEqual(result["modes"]["whole"]["severe_deletions"], 1)
        rendered = json.dumps(result)
        self.assertNotIn('"case_id"', rendered)
        self.assertNotIn("improved_case_ids", rendered)
        self.assertNotIn("regressed_case_ids", rendered)

    def test_rejects_unpaired_case_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, candidate = root / "base.json", root / "candidate.json"
            write_report(base, [case("PRIVATE_BASE_ID", 1)], 0.1)
            write_report(candidate, [case("PRIVATE_CANDIDATE_ID", 1)], 0.1)
            with self.assertRaisesRegex(ValueError, "case set mismatch") as caught:
                compare_reports([("whole", base), ("chunked", candidate)])
            self.assertNotIn("PRIVATE_BASE_ID", str(caught.exception))
            self.assertNotIn("PRIVATE_CANDIDATE_ID", str(caught.exception))

    def test_rejects_reference_length_mismatch_for_same_case_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, candidate = root / "base.json", root / "candidate.json"
            baseline_case = case("same-id", 1)
            candidate_case = case("same-id", 2)
            candidate_case["char"]["reference_units"] = 21
            write_report(base, [baseline_case], 0.1)
            write_report(candidate, [candidate_case], 0.2)
            with self.assertRaisesRegex(ValueError, "reference length mismatch"):
                compare_reports([("whole", base), ("chunked", candidate)])

    def test_rejects_duplicate_mode_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "report.json"
            write_report(report, [case("a", 1)], 0.1)
            with self.assertRaisesRegex(ValueError, "unique"):
                compare_reports([("same", report), ("same", report)])

    def test_rejects_duplicate_case_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "duplicate.json"
            write_report(report, [case("PRIVATE_DUPLICATE_ID", 1), case("PRIVATE_DUPLICATE_ID", 2)], 0.1)
            with self.assertRaisesRegex(ValueError, "duplicate case identifier") as caught:
                compare_reports([("whole", report), ("other", report)])
            self.assertNotIn("PRIVATE_DUPLICATE_ID", str(caught.exception))

    def test_cli_stdout_and_comparison_file_contain_aggregate_metrics_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, candidate, output = root / "base.json", root / "candidate.json", root / "comparison.json"
            write_report(base, [case("PRIVATE_UTTERANCE_A", 5), case("PRIVATE_UTTERANCE_B", 1)], 0.3)
            write_report(candidate, [case("PRIVATE_UTTERANCE_A", 2), case("PRIVATE_UTTERANCE_B", 3)], 0.25)
            stdout = io.StringIO()
            argv = [
                "compare_asr_modes.py", "--report", f"whole={base}",
                "--report", f"overlap_2s={candidate}", "--output-json", str(output),
            ]
            with patch.object(sys, "argv", argv), contextlib.redirect_stdout(stdout):
                self.assertEqual(main(), 0)
            self.assertIn('"deletion_delta": -1', stdout.getvalue())
            for private_value in ("PRIVATE_UTTERANCE_A", "PRIVATE_UTTERANCE_B"):
                self.assertNotIn(private_value, stdout.getvalue())
                self.assertNotIn(private_value, output.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
