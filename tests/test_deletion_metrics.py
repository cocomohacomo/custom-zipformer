import unittest
import csv
import json
import sys
import tempfile
from pathlib import Path

from scripts.deletion_metrics import main as metrics_main

from scripts.deletion_metrics import edit_counts, ratio


class EditCountsTests(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(edit_counts(list("今日は晴れ"), list("今日は晴れ")), (0, 0, 0))

    def test_japanese_particle_deletion(self):
        self.assertEqual(edit_counts(list("今日は行く"), list("今日行く")), (0, 1, 0))

    def test_insertion(self):
        self.assertEqual(edit_counts(list("猫いる"), list("猫がいる")), (0, 0, 1))

    def test_substitution(self):
        self.assertEqual(edit_counts(list("雨です"), list("雪です")), (1, 0, 0))

    def test_empty_hypothesis_is_all_deletions(self):
        self.assertEqual(edit_counts(list("abc"), []), (0, 3, 0))

    def test_empty_reference_is_all_insertions(self):
        self.assertEqual(edit_counts([], list("abc")), (0, 0, 3))

    def test_both_empty(self):
        self.assertEqual(edit_counts([], []), (0, 0, 0))

    def test_undefined_ratio(self):
        self.assertIsNone(ratio(0, 0))
        self.assertEqual(ratio(1, 4), 0.25)


class OmissionMetricsTests(unittest.TestCase):
    def test_complete_omission_is_reported_separately(self):
        with tempfile.TemporaryDirectory() as td:
            tsv = Path(td) / "transcripts.tsv"
            with tsv.open("w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter="\\t")
                writer.writerow(["case_id", "category", "reference", "hypothesis"])
                writer.writerow(["a", "short", "猫がいる", ""])
                writer.writerow(["b", "short", "犬がいる", "犬いる"])
            out = Path(td) / "metrics.json"
            previous = sys.argv
            try:
                sys.argv = ["metrics", str(tsv), "--output-json", str(out)]
                self.assertEqual(metrics_main(), 0)
            finally:
                sys.argv = previous
            report = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(report["complete_omission"]["empty_hypothesis_count"], 1)
            self.assertEqual(report["complete_omission"]["empty_hypothesis_rate"], 0.5)
            self.assertTrue(report["cases"][0]["complete_omission"])
            self.assertFalse(report["cases"][1]["complete_omission"])



if __name__ == "__main__":
    unittest.main()
