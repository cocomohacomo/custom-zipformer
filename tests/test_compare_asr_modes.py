from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_asr_modes import compare_reports


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

    def test_rejects_unpaired_case_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base, candidate = root / "base.json", root / "candidate.json"
            write_report(base, [case("a", 1)], 0.1)
            write_report(candidate, [case("b", 1)], 0.1)
            with self.assertRaisesRegex(ValueError, "case set mismatch"):
                compare_reports([("whole", base), ("chunked", candidate)])

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
            write_report(report, [case("a", 1), case("a", 2)], 0.1)
            with self.assertRaisesRegex(ValueError, "duplicate case_id"):
                compare_reports([("whole", report), ("other", report)])


if __name__ == "__main__":
    unittest.main()
