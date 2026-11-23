import unittest

from scripts.reject_rescue_simulation import simulate_reject_rescue


class RejectRescueSimulationTests(unittest.TestCase):
    def test_only_strict_one_to_one_candidate_is_rescued(self):
        records = [
            {"rejected_id": "r1", "token_chars": 2, "candidates": {250: ["k1"]}},
            {"rejected_id": "r2", "token_chars": 3, "candidates": {250: []}},
            {"rejected_id": "r3", "token_chars": 1, "candidates": {250: ["k2", "k3"]}},
            {"rejected_id": "r4", "token_chars": 4, "candidates": {250: ["k4"]}},
            {"rejected_id": "r5", "token_chars": 5, "candidates": {250: ["k4"]}},
        ]
        result = simulate_reject_rescue(records, (250,))[250]
        self.assertEqual(result["rescued_tokens"], 1)
        self.assertEqual(result["rescued_token_chars"], 2)
        self.assertEqual(result["not_rescued_tokens"], 4)
        self.assertEqual(result["classification"], {
            "rescued_unique": 1,
            "no_candidate": 1,
            "ambiguous": 1,
            "candidate_collision": 2,
        })
        self.assertTrue(result["accounting_ok"])

    def test_thresholds_are_evaluated_independently(self):
        records = [{
            "rejected_id": "r1", "token_chars": 3,
            "candidates": {100: [], 250: ["k1"], 500: ["k1", "k2"]},
        }]
        result = simulate_reject_rescue(records, (100, 250, 500))
        self.assertEqual(result[100]["rescued_tokens"], 0)
        self.assertEqual(result[250]["rescued_tokens"], 1)
        self.assertEqual(result[500]["classification"]["ambiguous"], 1)

    def test_empty_input_reconciles_to_zero(self):
        result = simulate_reject_rescue([], (100, 250))
        for summary in result.values():
            self.assertEqual(summary["rejected_tokens"], 0)
            self.assertEqual(summary["rescued_tokens"], 0)
            self.assertTrue(summary["accounting_ok"])

    def test_rejects_duplicate_rejected_ids(self):
        records = [
            {"rejected_id": "r", "token_chars": 1},
            {"rejected_id": "r", "token_chars": 1},
        ]
        with self.assertRaisesRegex(ValueError, "unique"):
            simulate_reject_rescue(records)

    def test_rejects_duplicate_candidate_ids_in_one_list(self):
        records = [{
            "rejected_id": "r", "token_chars": 1,
            "candidates": {250: ["k", "k"]},
        }]
        with self.assertRaisesRegex(ValueError, "unique"):
            simulate_reject_rescue(records, (250,))

    def test_rejects_negative_character_count(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            simulate_reject_rescue([{"rejected_id": "r", "token_chars": -1}])


if __name__ == "__main__":
    unittest.main()
