import unittest

from scripts.audio_chunking import diagnose_timestamped_chunk_merge


class RejectRescueShadowIntegrationTests(unittest.TestCase):
    def test_shadow_summary_is_emitted_without_changing_merge_output(self):
        results = [
            {"text": "abcdef", "tokens": list("abcdef"), "timestamps": [0, 1, 2, 3, 4, 5]},
            {"text": "defghi", "tokens": list("defghi"), "timestamps": [0, 0.2, 1, 2, 3, 4]},
        ]
        merged, method, diag = diagnose_timestamped_chunk_merge(
            results, [0.0, 4.0], [6.0, 6.0]
        )
        self.assertEqual((merged, method), ("abcdefghi", "timestamp"))
        shadow = diag["reject_rescue_shadow"]
        self.assertEqual(set(shadow), {100, 250, 500})
        for summary in shadow.values():
            self.assertEqual(
                summary["rescued_tokens"] + summary["not_rescued_tokens"],
                summary["rejected_tokens"],
            )
            self.assertTrue(summary["accounting_ok"])

    def test_fallback_does_not_claim_rescue(self):
        results = [{"text": "abcdef", "tokens": ["abc"], "timestamps": [0]}]
        _, method, diag = diagnose_timestamped_chunk_merge(results, [0.0], [3.0])
        self.assertEqual(method, "text_fallback")
        self.assertEqual(diag["reject_rescue_shadow"], {})


if __name__ == "__main__":
    unittest.main()
