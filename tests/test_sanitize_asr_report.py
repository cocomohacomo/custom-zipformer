import unittest

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
        safe = sanitize_metrics(report)
        self.assertEqual(safe["case_count"], 3)
        self.assertEqual(safe["character_error_rate"], 0.2)
        self.assertNotIn("cases", safe)
        self.assertNotIn("transcripts", safe)
        self.assertNotIn("audio_path", safe)
        self.assertNotIn("reference", safe)
        self.assertNotIn("hypothesis", safe)


if __name__ == "__main__":
    unittest.main()
