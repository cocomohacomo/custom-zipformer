import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sanitize_asr_report import sanitize_comparisons, sanitize_metrics


class SanitizeAsrReportTests(unittest.TestCase):
    def test_only_aggregate_fields_are_retained(self):
        report = {
            "case_count": 3,
            "summary": {"char": {
                "cer": 0.2, "deletion_rate": 0.1, "insertion_rate": 0.05,
                "substitution_rate": 0.05, "substitutions": 1, "deletions": 2,
                "insertions": 1,
            }},
            "complete_omission": {"empty_hypothesis_count": 0},
            "near_complete_omission": {"count": 1},
            "severe_deletion": {"count": 1},
            "category_summary": {"short": {"case_count": 1}},
            "cases": [{"case_id": "private-utterance-id", "reference": "sensitive text",
                       "hypothesis": "recognized text", "char": {"deletions": 2}}],
            "transcripts": ["private text"],
            "audio_path": "/tmp/private.wav",
        }
        report["category_summary"]["private-utterance-id"] = {
            "case_count": 1, "cer": 0.9, "reference": "private text"
        }
        safe = sanitize_metrics(report)
        self.assertEqual(safe["case_count"], 3)
        self.assertEqual(safe["character_error_rate"], 0.2)
        self.assertNotIn("cases", safe)
        self.assertNotIn("transcripts", safe)
        self.assertNotIn("audio_path", safe)
        self.assertNotIn("reference", safe)
        self.assertNotIn("hypothesis", safe)
        self.assertNotIn("private-utterance-id", safe["category_summary"])
        self.assertEqual(safe["category_summary"]["short"]["case_count"], 1)
        self.assertNotIn("reference", safe["category_summary"]["short"])


    def test_metric_values_must_be_finite_numbers_not_untrusted_strings(self):
        report = {
            "case_count": "private-utterance-id",
            "summary": {"char": {
                "cer": "sensitive transcript", "deletion_rate": float("nan"),
                "insertion_rate": 0.25, "substitution_rate": float("inf"),
                "substitutions": True, "deletions": 4, "insertions": "audio.wav",
            }},
            "category_summary": {
                "short": {"case_count": 2, "cer": "private text", "deletions": 1},
                "private-utterance-id": {"case_count": 99, "cer": 1.0},
            },
            "complete_omission": {"empty_hypothesis_count": "private-id"},
        }
        safe = sanitize_metrics(report)
        self.assertIsNone(safe["case_count"])
        self.assertIsNone(safe["character_error_rate"])
        self.assertIsNone(safe["character_deletion_rate"])
        self.assertEqual(safe["character_insertion_rate"], 0.25)
        self.assertIsNone(safe["character_substitution_rate"])
        self.assertIsNone(safe["substitutions"])
        self.assertEqual(safe["deletions"], 4)
        self.assertIsNone(safe["insertions"])
        self.assertIsNone(safe["complete_omission_count"])
        self.assertEqual(safe["category_summary"]["short"]["deletions"], 1)
        self.assertIsNone(safe["category_summary"]["short"]["cer"])
        self.assertNotIn("private-utterance-id", safe["category_summary"])

    def test_paired_comparison_sanitizer_allowlists_modes_and_numeric_fields(self):
        raw = {
            "paired_comparisons": {
                "fixed_15s": {
                    "baseline": "whole", "deletion_delta": -3, "cer_delta": -0.02,
                    "cases_with_fewer_deletions": 2, "cases_with_more_deletions": 1,
                    "cases_unchanged": 4, "improved_case_ids": ["private-id"],
                },
                "private-utterance-id": {
                    "baseline": "private text", "deletion_delta": "sensitive",
                },
                "overlap_2s": {
                    "baseline": "private-id", "deletion_delta": float("nan"),
                    "cer_delta": "private transcript", "cases_unchanged": True,
                },
            }
        }
        safe = sanitize_comparisons(raw)
        self.assertEqual(set(safe), {"fixed_15s", "overlap_2s"})
        self.assertEqual(safe["fixed_15s"]["deletion_delta"], -3)
        self.assertNotIn("improved_case_ids", safe["fixed_15s"])
        self.assertIsNone(safe["overlap_2s"]["baseline"])
        self.assertIsNone(safe["overlap_2s"]["deletion_delta"])
        self.assertIsNone(safe["overlap_2s"]["cer_delta"])
        self.assertIsNone(safe["overlap_2s"]["cases_unchanged"])

    def test_comparison_sanitizer_handles_unhashable_baselines_and_extreme_numbers(self):
        raw = {
            "paired_comparisons": {
                "fixed_15s": {"baseline": ["private"], "deletion_delta": 10 ** 10000},
                "overlap_2s": {"baseline": {"private": "value"}, "cer_delta": -0.1},
            }
        }
        safe = sanitize_comparisons(raw)
        self.assertIsNone(safe["fixed_15s"]["baseline"])
        self.assertIsNone(safe["fixed_15s"]["deletion_delta"])
        self.assertIsNone(safe["overlap_2s"]["baseline"])
        self.assertEqual(safe["overlap_2s"]["cer_delta"], -0.1)

    def test_rates_reject_negative_and_out_of_range_values(self):
        report = {
            "summary": {"char": {
                "cer": -0.1, "deletion_rate": 1.1,
                "insertion_rate": 0.2, "substitution_rate": -0.01,
            }}
        }
        safe = sanitize_metrics(report)
        self.assertIsNone(safe["character_error_rate"])
        self.assertIsNone(safe["character_deletion_rate"])
        self.assertEqual(safe["character_insertion_rate"], 0.2)
        self.assertIsNone(safe["character_substitution_rate"])

    def test_dataset_workflow_logs_do_not_emit_utterance_identifiers(self):
        root = Path(__file__).resolve().parents[1]
        for dataset in ("jsut", "fleurs"):
            with self.subTest(dataset=dataset):
                workflow = (root / ".github" / "workflows" / f"validate-{dataset}.yml").read_text(encoding="utf-8")
                self.assertNotIn("join(flagged)", workflow)
                self.assertNotIn("Invalid overlap chunk offsets: {", workflow)
                self.assertNotIn("Non-increasing overlap chunk offsets: {", workflow)
                self.assertNotIn("Missing overlap merge method diagnostic: {", workflow)

    def test_dataset_downloaders_suppress_exception_details(self):
        root = Path(__file__).resolve().parents[1]
        for dataset in ("jsut", "fleurs"):
            with self.subTest(dataset=dataset):
                source = (root / "scripts" / f"download_{dataset}_subset.py").read_text(encoding="utf-8")
                self.assertIn("detailed source information is suppressed", source)
                self.assertIn("except Exception:", source)


    def test_dataset_workflows_upload_only_sanitized_aggregate_reports(self):
        root = Path(__file__).resolve().parents[1]
        for dataset in ("jsut", "fleurs"):
            with self.subTest(dataset=dataset):
                workflow = (root / ".github" / "workflows" / f"validate-{dataset}.yml").read_text(encoding="utf-8")
                self.assertIn("sanitize_asr_report.py", workflow)
                self.assertIn("public_metrics.json", workflow)
                self.assertIn("workflow_dispatch:", workflow)
                self.assertIn("acknowledge_license", workflow)
                self.assertIn('default: "no"', workflow)
                self.assertIn(f"rm -rf tests/data/{dataset}_ja/clips", workflow)
                self.assertIn(f"rm -f tests/data/{dataset}_ja/manifest.json", workflow)
                self.assertIn(f"transcripts*.tsv", workflow)
                self.assertIn("metrics*.json", workflow)
                self.assertIn("mode_comparison.json", workflow)
                artifact = workflow.split("uses: actions/upload-artifact@v4", 1)[1]
                artifact = artifact.split("retention-days:", 1)[0]
                self.assertIn(f"path: tests/data/{dataset}_ja/public_metrics.json", artifact)
                self.assertNotIn("mode_comparison.json", artifact)
                self.assertNotIn("transcripts", artifact)
                self.assertNotIn("manifest.json", artifact)


if __name__ == "__main__":
    unittest.main()
