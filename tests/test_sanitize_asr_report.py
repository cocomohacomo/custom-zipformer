import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from sanitize_asr_report import sanitize_metrics


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


    def test_dataset_workflows_upload_only_sanitized_aggregate_reports(self):
        root = Path(__file__).resolve().parents[1]
        for dataset in ("jsut", "fleurs"):
            with self.subTest(dataset=dataset):
                workflow = (root / ".github" / "workflows" / f"validate-{dataset}.yml").read_text(encoding="utf-8")
                self.assertIn("sanitize_asr_report.py", workflow)
                self.assertIn("public_metrics.json", workflow)
                self.assertIn("workflow_dispatch:", workflow)
                self.assertIn("acknowledge_license", workflow)
                self.assertIn("default: \\"no\\"", workflow)
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
