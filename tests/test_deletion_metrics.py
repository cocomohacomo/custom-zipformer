import unittest

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


if __name__ == "__main__":
    unittest.main()
