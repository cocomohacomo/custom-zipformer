import unittest

from scripts.reject_rescue_sensitivity import counterfactual_rescue_cer


class CounterfactualRescueCerTests(unittest.TestCase):
    def test_unique_rescue_removes_deletion(self):
        result = counterfactual_rescue_cer(
            "日本語",
            [{"token_id": "k1", "time": 0.0, "text": "日"},
             {"token_id": "k2", "time": 2.0, "text": "語"}],
            [{"token_id": "r1", "time": 1.0, "text": "本",
              "candidates": {100: ["k1"], 250: ["k1"], 500: ["k1"]}}],
        )
        self.assertEqual(result["baseline"]["deletions"], 1)
        self.assertEqual(result["thresholds"]["250"]["cer"], 0.0)
        self.assertEqual(result["thresholds"]["250"]["rescued_tokens"], 1)

    def test_ambiguous_and_collision_are_not_rescued(self):
        result = counterfactual_rescue_cer(
            "abcd",
            [{"token_id": "k1", "time": 1.0, "text": "a"},
             {"token_id": "k2", "time": 2.0, "text": "b"}],
            [{"token_id": "r1", "time": 0.5, "text": "c",
              "candidates": {250: ["k1", "k2"]}},
             {"token_id": "r2", "time": 1.5, "text": "d",
              "candidates": {250: ["k1"]}},
             {"token_id": "r3", "time": 2.5, "text": "x",
              "candidates": {250: ["k1"]}}],
            thresholds_ms=(250,),
        )
        item = result["thresholds"]["250"]
        self.assertEqual(item["rescued_tokens"], 0)
        self.assertEqual(item["classification"]["ambiguous"], 1)
        self.assertEqual(item["classification"]["candidate_collision"], 2)

    def test_result_is_aggregate_only(self):
        result = counterfactual_rescue_cer(
            "a", [{"token_id": "secret-token-id", "time": 0, "text": "a"}], [],
            thresholds_ms=(100,),
        )
        rendered = repr(result)
        self.assertNotIn("secret-token-id", rendered)
        self.assertNotIn('"text"', rendered)

    def test_unknown_candidate_is_rejected(self):
        with self.assertRaises(ValueError):
            counterfactual_rescue_cer(
                "a", [{"token_id": "k", "time": 0, "text": "a"}],
                [{"token_id": "r", "time": 1, "text": "b",
                  "candidates": {250: ["unknown"]}}],
                thresholds_ms=(250,),
            )


if __name__ == "__main__":
    unittest.main()


class BaselineGateTests(unittest.TestCase):
    def test_exact_match_allows_cer(self):
        from scripts.reject_rescue_sensitivity import gated_counterfactual_rescue_cer
        kept = [{"token_id":"k1","time":0.0,"text":"日本"}]
        rejected = [{"token_id":"r1","time":1.0,"text":"語","candidates":{100:[],250:[],500:[]}}]
        result = gated_counterfactual_rescue_cer("日本語","日本",kept,rejected)
        self.assertEqual(result["cer_comparison_status"],"baseline_verified")
        self.assertIn("baseline",result["cer"])

    def test_mismatch_stops_cer_and_classifies(self):
        from scripts.reject_rescue_sensitivity import gated_counterfactual_rescue_cer
        kept = [{"token_id":"k1","time":0.0,"text":"abc"}]
        rejected = []
        result = gated_counterfactual_rescue_cer("abc","abd",kept,rejected)
        self.assertEqual(result["cer_comparison_status"],"stopped_baseline_mismatch")
        self.assertNotIn("cer",result)
        self.assertEqual(result["gate"]["mismatch_reason"],"content_or_order")

    def test_edge_whitespace_mismatch_classified(self):
        from scripts.reject_rescue_sensitivity import compare_baseline_and_gate
        kept = [{"token_id":"k1","time":0.0,"text":" x "}]
        result = compare_baseline_and_gate("x",kept,[])
        self.assertTrue(result["baseline_match"])

