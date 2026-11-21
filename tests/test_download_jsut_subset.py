from __future__ import annotations

import unittest

from scripts.download_jsut_subset import classify, parse_transcripts


class JsutDatasetPreparationTests(unittest.TestCase):
    def test_parses_colon_inside_transcript(self):
        parsed = parse_transcripts(
            "BASIC5000_4501:だが、エーアイセンター稼動を快く思わない。\n"
            "BASIC5000_4502:時刻は12:30です。\n"
        )
        self.assertEqual(parsed["BASIC5000_4502"], "時刻は12:30です。")

    def test_skips_blank_lines_and_rejects_duplicate_ids(self):
        self.assertEqual(parse_transcripts("\nID_1:文章\n"), {"ID_1": "文章"})
        with self.assertRaises(ValueError):
            parse_transcripts("ID_1:文章\nID_1:別の文章")

    def test_rejects_malformed_transcript(self):
        with self.assertRaises(ValueError):
            parse_transcripts("not-a-transcript-line")

    def test_length_categories(self):
        self.assertEqual(classify("短文です。"), "short")
        self.assertEqual(classify("これは中くらいの長さの評価用文章です。"), "medium")
        self.assertEqual(classify("これは長文の欠落を検出するために用意した十分に長い評価用の日本語文章です。"), "long")


if __name__ == "__main__":
    unittest.main()
